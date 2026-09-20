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
| Annotator Agents (x4) | `agents/agent_defs.py`'s `build_annotator_agents()` | `stage_a_generate.py`'s 4-model roster, wrapped as agents with tool access instead of one-shot completions |
| Adjudicator Agent | `agents_annotate.py`'s `adjudicate()` | `stage_b_prefilter.py` + Cowork verification |
| Guideline Agent | `agents_guideline.py` | New capability -- formalizes the ad hoc calibration-rule-writing that happened in Claude's session memory during Cowork verification into `config/calibration_guidelines.md` |
| Auditor Agent | `agents_audit.py` | `config/models.json`'s `stage_c_coverage_model` role (never had a script before this) |

Supporting files: `agents/schemas.py` (Pydantic output contracts), `agents/llm_factory.py`
(builds a `crewai.LLM` from a `config/models.json` entry, honoring the same
`provider`/`api_model` discount-routing fields `stage_a_generate.py` reads),
`config/idiom_lexicon_ru_en.json` (starter lexicon, 7 entries seeded from real batch
1-3 idioms), `config/calibration_guidelines.md` (seeded from every calibration rule
established across Stage B batches 1-3 -- see `stage-b-verification-log.md` in the
project for where each one came from).

## Why agents instead of the existing one-shot calls

The distinguishing feature is tool access mid-reasoning, not the "agent" label.
`stage_a_generate.py`'s calls are one-shot: sentence in, JSON out, no way to check
an idiom claim against a real dictionary first. Every Annotator Agent here calls
`read_calibration_guidelines` and, when relevant, `lookup_idiom` before answering --
grounded rather than guessed. Whether this measurably reduces disagreement rate vs.
the plain roster is an open question this build doesn't answer; that's the first
thing to check once this runs live (compare `agents_annotate.py`'s output against a
`run_batch.py` run on the same seed file).

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
