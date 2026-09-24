# Podtekst's six-agent architecture

Built 2026-09-20 on [CrewAI](https://github.com/crewAIInc/crewAI) (`crewai==1.15.22`
at build time). **This is an experimental parallel path next to the proven
`run_batch.py` pipeline (`stage_a_generate.py` + `stage_b_prefilter.py` + Cowork
verification) -- it does not replace it.** Nothing here has been run against a
live OpenRouter call yet; see "What's verified vs. not" below before trusting
it at volume.

## Role -> file map

| Role | File(s) | Maps to / replaces |
|---|---|---|
| Sourcing Agent | `agents_source.py` | New capability -- mines RU-EN pairs with real human commentary from forums, instead of generating from scratch |
| Detection Agent | `agents/tools.py` (`flag_ru_address_pronoun`, `lookup_idiom`), called from `agents_annotate.py` | A pre-generation analog to `deterministic_checks.py`'s free gate |
| Annotator Agents (x4) | `agents/agent_defs.py`'s `build_annotator_agents()` | `stage_a_generate.py`'s 4-model roster, wrapped as agents given pre-fetched idiom-lexicon + calibration-guidelines evidence in their prompt (see the cost revision below -- no live tool calls) instead of a bare one-shot completion |
| Adjudicator Agent | `agents_annotate.py`'s `adjudicate()` | `stage_b_prefilter.py` + Cowork verification |
| Guideline Agent | `agents_guideline.py` | New capability -- formalizes the ad hoc calibration-rule-writing done during Cowork verification into `config/calibration_guidelines.md` |
| Auditor Agent | `agents_audit.py` | `config/models.json`'s `stage_c_coverage_model` role (never had a script before this) |

Supporting files: `agents/schemas.py` (Pydantic output contracts), `agents/llm_factory.py`
(builds a `crewai.LLM` from a `config/models.json` entry, honoring the same
`provider`/`api_model` discount-routing fields `stage_a_generate.py` reads),
`config/idiom_lexicon_ru_en.json` (starter lexicon, 7 entries seeded from real batch
1-3 idioms), `config/calibration_guidelines.md` (seeded from every calibration rule
established across Stage B batches 1-3 -- see `stage-b-verification-log.md` in the
project for where each one came from).

## Why agents instead of the existing one-shot calls

The distinguishing feature is grounding in real evidence (idiom lexicon, pronoun
detection, calibration guidelines), not the "agent" label, and grounding does NOT
require letting the model call tools live. `stage_a_generate.py`'s calls are
one-shot: sentence in, JSON out, no way to check an idiom claim against a real
dictionary first. Every agent here still gets that grounding -- but as of the
2026-09-20 cost revision below, it's pre-computed in Python and handed to the
agent as text in its prompt, not offered as a CrewAI tool the agent decides to
call mid-reasoning. Whether the grounding itself measurably reduces disagreement
rate vs. the plain roster is still an open question this build doesn't answer;
that's the first thing to check once this runs live (compare `agents_annotate.py`'s
output against a `run_batch.py` run on the same seed file).

## Cost revision -- 2026-09-20 (same day as the initial build)

The first version of this architecture gave every agent CrewAI `tools=` and told
it to call them ("call flag_ru_address_pronoun and lookup_idiom before deciding").
That's real waste: a tool-owning agent has to make an LLM call to *decide* to call
the tool, a second call to read the tool's result and continue, and a third for
the final answer -- up to 3x the LLM calls of a single-shot completion, for
information the orchestrating Python script already knew it needed before it ever
called the agent (the sentence being judged, the guidelines file contents, the
search query, the file paths to audit).

Fixed by removing `tools=` from every one of the six `Agent()` definitions and
moving every tool call into the calling CLI script, run in plain Python *before*
`agent.kickoff()`, with the result embedded directly in the prompt text:

| Agent | Before (tool-owning) | After (pre-fetched) |
|---|---|---|
| Detection | up to 3 calls (decide to call `flag_ru_address_pronoun`, decide to call `lookup_idiom`, final answer) | **1 call** -- both tools run in Python first, evidence embedded |
| Annotator (x4) | up to 3 calls each (guidelines read, idiom lookup, final answer) | **1 call each** -- guidelines read ONCE per batch run (not per sentence), idiom lookup pre-run |
| Adjudicator | up to 2 calls (guidelines read, final answer) | **1 call** -- guidelines text already in prompt |
| Sourcing | up to 2 calls (search, extract) | **1 call** -- search runs in Python first, results embedded |
| Guideline | up to 2 calls (guidelines read, propose) | **1 call** -- guidelines text embedded; the actual write (`propose_guideline_update`) was ALSO removed from the agent's tools entirely and is now only ever called by the script, after reading the agent's structured output |
| Auditor | up to 2 calls (stats, recommend) | **1 call** -- `dataset_category_stats` runs in Python first (the script already has the file paths from its own `--in` args) |

Guidelines-file reads got a second optimization on top of the tool removal:
`agents_annotate.py` and `agents_guideline.py` now read `calibration_guidelines.md`
ONCE per script run and reuse the same text for every sentence, instead of paying
to re-fetch identical content per sentence.

**Bug caught during this revision, never shipped**: the Guideline Agent originally
had `propose_guideline_update` as both an agent tool AND something the calling
script called again after reading the agent's structured output -- a double-write
risk (the same proposal landing in `calibration_guidelines.md` twice). Removing
tools from the agent entirely also removed this risk; the write path is now
single-owner (script only). Caught by a regression test built for this revision,
not found live.

Verified for this revision (mocked, no live OpenRouter calls -- same limitation
as the rest of this build): every agent construction confirms `tools == []`;
`detect()` and `adjudicate()` each make exactly one `kickoff()` call (asserted via
mock call-count); `annotate_all()`'s prompt contains the embedded guidelines text
rather than a tool reference; the Guideline Agent's write-path regression test
confirms exactly one write reaches `propose_guideline_update` per proposal, not
two. Not verified: the actual dollar-cost difference this makes per sentence,
since that needs a live run this environment can't make.

## What's verified vs. not (2026-09-20 build)

Verified, on real data, in this sandboxed environment:
- All 6 `Agent()` objects construct correctly and pull the right model/provider from
  `config/models.json` (checked each `.llm.model` and `.llm.additional_params`
  individually -- confirms the 2026-09-20 roster swap flows through automatically).
- `flag_ru_address_pronoun` correctly parses inflected Russian pronoun forms
  (`тебе` -> NPRO datv) via pymorphy2 -- tested against real sentences.
- `lookup_idiom` correctly matches lexicon entries by substring.
- `dataset_category_stats` run against the real `stage_b_batch1/2/3.jsonl` files
  reproduced the exact known totals (487 rows; formality_shift 63, none 231,
  emotional_subtext 21, sarcasm 91, idiom 81) -- this tool's numbers can be trusted.
- `propose_guideline_update` correctly appends to and doesn't corrupt
  `calibration_guidelines.md` (tested on a backup copy, restored after).
- The full `agents_annotate.py` orchestration logic (Detection gate -> concurrent
  Annotator fan-out via `asyncio.gather` -> Adjudicator) was exercised end-to-end
  with mocked agent responses -- confirms the wiring (field access, JSON
  serialization, async concurrency) has no integration bugs, independent of what
  a real model would actually answer.
- Read crewai's own source (`llms/providers/openai_compatible/completion.py`,
  `llms/providers/openai/completion.py`) to confirm `additional_params` (how
  `provider={"order": [...], "allow_fallbacks": true}` gets passed) is merged
  into the actual chat-completions request body sent to OpenRouter, not silently
  dropped -- crewai's `openrouter` provider is a native OpenAI-compatible
  implementation, not litellm (`llm.is_litellm` is `False` here), so this needed
  checking rather than assuming litellm conventions applied.

NOT verified -- needs a real terminal with OpenRouter access:
- Any actual model response quality, or whether the schema-constrained output
  (`response_format=`) reliably parses for every model in the roster (particularly
  worth checking for `minimax/minimax-m3` and `z-ai/glm-5.3-flash`, added in the
  same session as this build).
- `search_ru_en_forums` -- `api.stackexchange.com` is unreachable from this
  environment too (confirmed with a direct `curl`, same org-egress-proxy block
  that affects `openrouter.ai`), so the Sourcing Agent's core tool is completely
  untested live.
- Cost per sentence. Six agents with tool calls is not directly comparable to
  `stage_a_generate.py`'s flat per-token pricing -- run `agents_annotate.py` with
  `--limit 5` first and check actual OpenRouter usage before a real batch.

## Suggested first real test

```
export OPENROUTER_API_KEY=sk-or-...
python agents_annotate.py --seeds seeds_batch1.txt --limit 10 \
    --out-adjudicated /tmp/agents_test.jsonl \
    --out-candidates /tmp/agents_test_candidates.jsonl \
    --out-skipped /tmp/agents_test_skipped.jsonl
```

Compare `/tmp/agents_test.jsonl` against the real `stage_b_batch1.jsonl` rows for
the same 10 sentences (batch 1's seeds are the same file) -- that's a direct
apples-to-apples check of whether this path's verdicts match what Cowork actually
verified.
