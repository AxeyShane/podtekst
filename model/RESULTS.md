# Fine-tuning results

Running log of training runs and what they showed. All numbers are on the same held-out test set
(381 rows) plus a 24-sentence crude-language probe (`probes/crude_probe.jsonl`), unless noted.
Training runs on a Kaggle T4 (fp16, LoRA r16 on all linear layers, answer-only loss, 2 epochs).

*Last updated 2026-10-10.*

## Current best: QVikhr-3-1.7B + LoRA v2b

| | Untrained base | **v2b** |
|---|---|---|
| JSON valid | 99.5% | 100% |
| Translation chrF (all) | 39.2 | **55.9** |
| Subtext precision | 67% | **80%** |
| Subtext recall | 6% | **63%** |
| False alarms on plain messages | 2% | 13.5% |
| Category accuracy | 53% | **73%** |
| Swearing kept in translation | 8% | 12% |

Fine-tuning works: the untrained model almost never flags nuance and barely translates
English→Russian (chrF 8.8); v2b translates both directions and catches most idioms and sarcasm
(F1 ≈ 0.7–0.8). Weak spots: formality (ты/вы) and emotional subtext, and swearing is still
mostly softened or lost.

## What each run taught us

| Run | Change | Result |
|---|---|---|
| v1 | first fine-tune | Detection works (recall 83%), but 33% false alarms, mostly plain ты flagged as formality |
| relabel | flag **only unusual** ты/вы (polite вы to one person, ты to a stranger, switches); plain ты among friends/family → no note | 488 rows relabelled; the training data had contradicted itself |
| v2 | same settings | Accidentally trained on the pre-relabel data; the job now asserts the data version before training |
| v2b | v2 on the relabelled data | False alarms halved (33% → 13.5%), precision up to 80% |
| v3 | formality and emotion rows oversampled ×2 | Formality/emotion F1 up (0.18 → 0.44–0.58), but false alarms back to ~24% — not kept |

Run-to-run spread is several points and the test set has only 35 formality and 19 emotion rows,
so differences under ~5 points are noise.

## Choosing a bigger base (zero-shot, no training)

| | QVikhr-3-1.7B | QVikhr-3-4B | RuadaptQwen3-4B |
|---|---|---|---|
| chrF all / RU→EN / EN→RU | 39.2 / 52.7 / 8.8 | 55.9 / 59.7 / 47.3 | 55.7 / 59.4 / 47.3 |
| Swearing kept | 8% | 27% | **65%** |
| False alarms | 2% | 66% | 33% |

- Both 4B models translate as well **untrained** as the fine-tuned 1.7B.
- Zero-shot nuance detection is unusable for every model; training fixes that, so it doesn't decide.
- **Next: fine-tune RuadaptQwen3-4B-Instruct.** It keeps swearing far better, flags less wildly,
  uses the Qwen3 architecture (converts to LiteRT-LM unchanged) and has a Russian-extended
  tokenizer, so Russian messages take fewer tokens on the phone.
- Gemma 4 E2B is deferred: it needs bf16 for training (fp16 overflows), which a T4 lacks.

## Next

1. RuadaptQwen3-4B LoRA (`launch.py --name rua-v1 --base RefalMachine/RuadaptQwen3-4B-Instruct`), compare with v2b.
2. Crude-language training examples, so swearing is translated, not softened.
3. More unusual ты/вы examples.
4. Convert the winner to `.litertlm` (INT8 first, then INT4) and test on the phone.
