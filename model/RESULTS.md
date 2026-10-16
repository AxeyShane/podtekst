# Fine-tuning results

Running log of training runs and what they showed. All numbers are on the same held-out test set
(380 rows) plus a 24-sentence crude-language probe (`probes/crude_probe.jsonl`), unless noted.
Training runs on a Kaggle T4 (LoRA r16 on all linear layers, answer-only loss, 2 epochs).

*Last updated 2026-10-16.*

## Current best: RuadaptQwen3-4B + QLoRA rua-q3, running on the phone

rua-q3 is RuadaptQwen3-4B-Instruct fine-tuned with QLoRA (4-bit base during training, adapter evaluated
on the fp16 base) on 4,473 rows, including 1,022 swearing/register rows, with formality and emotion
rows oversampled ×2. The same adapter merged and exported to LiteRT-LM was then scored **on a
OnePlus 13 (Snapdragon 8 Elite, GPU backend)** with the test app in [`keyboard/`](../keyboard/):

| | v2b (1.7B) | rua-q3, fp16 (Kaggle) | rua-q3, INT4 on phone | **rua-q3, INT8 on phone** |
|---|---|---|---|---|
| JSON valid | 100% | 100% | 99.7% | **100%** |
| Translation chrF | 55.9 | 61.4 | 58.1 | **60.7** |
| Subtext precision | 80% | 83% | 87% | 85% |
| Subtext recall | 63% | 85% | 72% | **84%** |
| False alarms on plain messages | 13.5% | 13.9% | 9.1% | 12.5% |
| Category accuracy | 73% | 83% | 79% | **83%** |
| F1 formality / idiom / sarcasm / emotion | 0.18 / 0.71 / 0.79 / 0.39 | 0.74 / 0.83 / 0.83 / 0.67 | 0.67 / 0.69 / 0.82 / 0.56 | 0.74 / 0.82 / 0.84 / 0.65 |
| Swearing kept (within one tier) | 12% | 88% | 73% | 81% |
| Model file | — | — | 2.26 GB | 4.04 GB |
| Time per message (median / p90) | — | — | 3.5 s / 5.3 s | 4.6 s / 6.8 s |

- **8-bit on the phone matches the full-precision model** on every detection metric, so the app,
  prompt and chat template reproduce training exactly.
- **4-bit costs real quality**: the model turns cautious, labels 17 more idioms as "nothing lost"
  and softens strong swearing (Сука → "Damn", Да ну нахуй → "Come on"). It never refuses or masks.
- Speed is now the main open problem: ~9 tokens/s at INT8 (~14 at INT4) on the GPU, first token
  ~1.4 s. Next: reuse the system-prompt prefill across messages, then the NPU.
- Phone numbers come from `model/score_device.py`, which scores the app's result files with the
  same code as `evaluate.py`.

## RuadaptQwen3-4B runs

| Run | Change | Result |
|---|---|---|
| rua-v1 | fp16 LoRA | Out of memory on the T4 at batch 4 and 2 |
| rua-q1 | QLoRA, oversampling ×2, 3,451 rows | Recall 86%, false alarms 12.5%, chrF 61.4 — but swearing kept only 35% |
| rua-q2 | + 1,022 swearing/register rows, **no** oversampling | Swearing 88%, but recall fell to 76% and formality recall to 66% |
| rua-q3 | rua-q2 data **with** oversampling ×2 | Best of both: recall 85%, swearing 88%, formality F1 0.74 |

## Earlier runs (QVikhr-3-1.7B)

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

1. Speed on the phone: cache the system-prompt prefill, then try the NPU.
2. Native-speaker review of the slang, emotion and swearing sheets; add the answers as training data.
3. More unusual ты/вы examples (still the most common false alarm).
4. Keyboard: FlorisBoard fork using the `keyboard/engine` module.
