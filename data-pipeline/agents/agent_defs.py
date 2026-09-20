"""
The six CrewAI Agent definitions. Each pulls its LLM from config/models.json
via llm_factory so a roster swap (like the 2026-09-20 mistral->glm-5.3-flash,
gemini-3.1-flash-lite->minimax-m3 change) updates these agents automatically --
nothing here hard-codes a model slug.
"""

from crewai import Agent

from .llm_factory import build_llm, build_llm_by_role, load_models_config
from .tools import (
    dataset_category_stats,
    flag_ru_address_pronoun,
    lookup_idiom,
    propose_guideline_update,
    read_calibration_guidelines,
    search_ru_en_forums,
)


def build_sourcing_agent() -> Agent:
    return Agent(
        role="Sourcing Agent",
        goal="Find real RU<->EN sentence pairs that already carry a bilingual human's "
             "own nuance commentary -- translator's notes, forum Q&A answers -- rather "
             "than generating sentences from scratch.",
        backstory="You mine public bilingual forums (Russian Language StackExchange) for "
                  "threads where someone asks about a register, idiom, or tone distinction "
                  "and a fluent bilingual explains it. You never invent the commentary "
                  "yourself -- you extract what a real person already wrote, and you always "
                  "keep the source URL so the claim can be checked.",
        tools=[search_ru_en_forums],
        llm=build_llm_by_role("sourcing_agent_model", temperature=0.2),
        verbose=True,
    )


def build_detection_agent() -> Agent:
    return Agent(
        role="Detection Agent",
        goal="Cheaply decide whether a candidate source sentence is worth sending to the "
             "Annotator Agents at all, using grounded rule-based evidence rather than "
             "guessing.",
        backstory="You are the free pre-filter before expensive Annotator attention -- the "
                  "same role deterministic_checks.py plays for Stage A output, but applied "
                  "to raw candidate sentences before generation. You always call "
                  "flag_ru_address_pronoun and lookup_idiom before deciding worth_annotating, "
                  "and you never claim a category is a candidate without tool evidence for it.",
        tools=[flag_ru_address_pronoun, lookup_idiom, read_calibration_guidelines],
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
                      f"pipeline ({generator['role']}: {generator['notes']}). You read "
                      f"config/calibration_guidelines.md before judging, use lookup_idiom "
                      f"to ground any idiom claim instead of inventing one, and never flag "
                      f"has_subtext=true for a pattern the guidelines say survives direct "
                      f"translation. You answer only for the sentence you're given -- you "
                      f"don't see or try to match the other Annotator Agents' answers.",
            tools=[lookup_idiom, flag_ru_address_pronoun, read_calibration_guidelines],
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
                  "today: given several Annotator Agent outputs for the same sentence, you "
                  "compute agreement, pick/polish the best translation among agreeing "
                  "candidates, and apply config/calibration_guidelines.md to break ties "
                  "consistently rather than by raw vote count alone -- a guideline-backed "
                  "minority can outweigh an unexplained majority, same as the human "
                  "calibration overrides in batches 2 and 3 did.",
        tools=[read_calibration_guidelines, lookup_idiom],
        llm=build_llm_by_role("stage_b_prefilter_model", temperature=0.1),
        verbose=True,
    )


def build_guideline_agent() -> Agent:
    return Agent(
        role="Guideline Agent",
        goal="Turn a recurring disagreement pattern across several sentences into a "
             "precise, generalizable calibration rule, proposed for human approval -- "
             "never applied automatically.",
        backstory="You formalize what used to happen ad hoc: a human (or Claude, during "
                  "Cowork verification) noticing 'we keep overriding this same kind of "
                  "case' and writing a rule down. You only ever propose -- "
                  "propose_guideline_update appends to a 'Pending review' section that "
                  "Akshay reads and approves by hand, the same way every calibration rule "
                  "in this project so far was stated in plain language before being "
                  "applied, never assumed silently.",
        tools=[read_calibration_guidelines, propose_guideline_update],
        llm=build_llm_by_role("guideline_agent_model", temperature=0.3),
        verbose=True,
    )


def build_auditor_agent() -> Agent:
    return Agent(
        role="Auditor Agent",
        goal="Periodically scan the accumulated verified dataset for category imbalance, "
             "near-duplicate redundancy, and underrepresented patterns, and recommend what "
             "the next batch's seeds should target.",
        backstory="You play the role config/models.json's stage_c_coverage_model was "
                  "reserved for. You always call dataset_category_stats rather than "
                  "estimating from memory, and your recommendation is specific enough to "
                  "hand directly to generate_seeds.py's next prompt (e.g. 'target more "
                  "emotional_subtext rows involving apology or gratitude, currently only "
                  "21 of 487 rows').",
        tools=[dataset_category_stats],
        llm=build_llm_by_role("stage_c_coverage_model", temperature=0.3),
        verbose=True,
    )
