"""
The six CrewAI Agent definitions. Each pulls its LLM from config/models.json
via llm_factory so a roster swap (like the 2026-09-20 mistral->glm-5.3-flash,
gemini-3.1-flash-lite->minimax-m3 change) updates these agents automatically --
nothing here hard-codes a model slug.

COST NOTE (2026-09-20 revision): none of these agents carry CrewAI `tools=`
lists anymore. Every tool call this pipeline needs (pymorphy2 pronoun
detection, idiom lookup, reading calibration_guidelines.md, StackExchange
search, dataset stats) is now pre-computed in Python by the calling CLI
script (agents_annotate.py etc.) and handed to the agent as plain text in its
prompt. A CrewAI agent that owns a tool decides, mid-reasoning, whether to
call it -- that decision is itself an extra LLM round-trip, then another
round-trip to read the tool's result, before the final answer. Every one of
these tool calls was something the orchestrating script already knew it
needed (the sentence being judged, the guidelines file, the search query, the
file paths to audit) -- there was nothing for the agent to decide, so paying
for that decision was pure waste. This was measured, not assumed: see
AGENTS.md's cost section for the before/after call-count estimate per agent.
propose_guideline_update is the one exception -- it's a write action gated on
the agent's own judgment (only write when a real pattern was found), so it
stays a script-level call made *after* reading the agent's structured output,
not a tool the agent can invoke mid-reasoning (which would have risked a
double-write: once from the agent calling it, once from the script calling it
again with the same result -- caught during this revision, never shipped).
"""

from crewai import Agent

from .llm_factory import build_llm, build_llm_by_role, load_models_config


def build_sourcing_agent() -> Agent:
    return Agent(
        role="Sourcing Agent",
        goal="Given real StackExchange search results already fetched for you, extract "
             "genuine RU<->EN sentence pairs that carry a bilingual human's own nuance "
             "commentary -- never invent or paraphrase the commentary yourself.",
        backstory="You mine public bilingual forums (Russian Language StackExchange) for "
                  "threads where someone asks about a register, idiom, or tone distinction "
                  "and a fluent bilingual explains it. The search has already been run for "
                  "you -- your job is picking the best real result and extracting it "
                  "faithfully, keeping the source URL so the claim can be checked, never "
                  "generating a plausible-sounding example of your own.",
        llm=build_llm_by_role("sourcing_agent_model", temperature=0.2),
        verbose=True,
    )


def build_detection_agent() -> Agent:
    return Agent(
        role="Detection Agent",
        goal="Given rule-based tool evidence already gathered for a candidate sentence, "
             "cheaply decide whether it's worth sending to the Annotator Agents at all.",
        backstory="You are the free pre-filter before expensive Annotator attention -- the "
                  "same role deterministic_checks.py plays for Stage A output, but applied "
                  "to raw candidate sentences before generation. The address-pronoun and "
                  "idiom-lexicon checks have already been run for this sentence and are in "
                  "your prompt -- you decide worth_annotating from that evidence plus your "
                  "own read of the sentence, but you never claim a category is a candidate "
                  "without evidence for it in front of you.",
        llm=build_llm_by_role("detection_agent_model", temperature=0.1),
        verbose=True,
    )


def build_annotator_agents() -> list[Agent]:
    """One Agent per active stage_a_generators entry in config/models.json --
    currently deepseek-v4-flash, z-ai/glm-5.3-flash, minimax/minimax-m3,
    google/gemini-3.8-flash (see the 2026-09-20 roster swap). Returns a list so
    callers can fan a Task out across all of them for independent voting; do
    NOT collapse this to one agent -- the whole point of the role is
    disagreement signal for the Adjudicator Agent to work with."""

    cfg = load_models_config()
    agents = []
    for generator in cfg["stage_a_generators"]:
        agents.append(Agent(
            role=f"Annotator Agent ({generator['slug']})",
            goal="Independently judge whether a RU<->EN sentence pair carries hidden "
                 "nuance (formality_shift, sarcasm, idiom, or emotional_subtext) that a "
                 "literal translation would lose, and produce the best translation plus a "
                 "short nuance note when it does.",
            backstory=f"You are one of several independent bilingual judges in this "
                      f"pipeline ({generator['role']}: {generator['notes']}). The current "
                      f"calibration_guidelines.md text and any idiom-lexicon match for this "
                      f"sentence are already in your prompt -- apply them rather than "
                      f"inventing your own idiom claim, and never flag has_subtext=true for "
                      f"a pattern the guidelines say survives direct translation. You answer "
                      f"only for the sentence you're given -- you don't see or try to match "
                      f"the other Annotator Agents' answers.",
            llm=build_llm(generator, temperature=0.4),
            verbose=True,
        ))
    return agents


def build_adjudicator_agent() -> Agent:
    return Agent(
        role="Adjudicator Agent",
        goal="Resolve the Annotator Agents' independent judgments on one sentence into a "
             "single verdict -- auto-resolve when they agree, and produce a clear, "
             "calibration-guideline-consistent decision (or an honest "
             "escalated_needs_human) when they don't.",
        backstory="You play the role stage_b_prefilter.py and Cowork verification play "
                  "today: given several Annotator Agent outputs for the same sentence plus "
                  "the current calibration_guidelines.md text (already in your prompt), you "
                  "compute agreement, pick/polish the best translation among agreeing "
                  "candidates, and apply the guidelines to break ties consistently rather "
                  "than by raw vote count alone -- a guideline-backed minority can outweigh "
                  "an unexplained majority, same as the human calibration overrides in "
                  "batches 2 and 3 did.",
        llm=build_llm_by_role("stage_b_prefilter_model", temperature=0.1),
        verbose=True,
    )


def build_guideline_agent() -> Agent:
    return Agent(
        role="Guideline Agent",
        goal="Turn a recurring disagreement pattern across several sentences into a "
             "precise, generalizable calibration rule -- or honestly say no clear pattern "
             "exists rather than forcing one.",
        backstory="You formalize what used to happen ad hoc: a human (or Claude, during "
                  "Cowork verification) noticing 'we keep overriding this same kind of "
                  "case' and writing a rule down. The current calibration_guidelines.md "
                  "text is already in your prompt so you don't propose a duplicate of an "
                  "existing rule. You only ever propose in your structured output -- you "
                  "never write to the guidelines file yourself; the calling script does "
                  "that after reading your answer, and only when you said a real pattern "
                  "was found (direction != 'clarify_only').",
        llm=build_llm_by_role("guideline_agent_model", temperature=0.3),
        verbose=True,
    )


def build_auditor_agent() -> Agent:
    return Agent(
        role="Auditor Agent",
        goal="Given the real dataset_category_stats numbers already computed for you, "
             "name underrepresented patterns and recommend what the next batch's seeds "
             "should target.",
        backstory="You play the role config/models.json's stage_c_coverage_model was "
                  "reserved for. The category counts, has_subtext ratio, and near-duplicate "
                  "clusters are already computed and in your prompt -- use those numbers "
                  "exactly rather than re-estimating them, and make your recommendation "
                  "specific enough to hand directly to generate_seeds.py's next prompt "
                  "(e.g. 'target more emotional_subtext rows involving apology or "
                  "gratitude, currently only 21 of 487 rows').",
        llm=build_llm_by_role("stage_c_coverage_model", temperature=0.3),
        verbose=True,
    )
