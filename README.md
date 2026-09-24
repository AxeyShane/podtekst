# Podtekst

**Подтекст** — "subtext." An on-device Android keyboard that translates
Russian ↔ English and flags what a literal translation would miss: tone,
formality, sarcasm, idiom, emotional subtext — automatically, without the
user asking.

Fully local. No cloud calls at runtime. Built to help two people from
different language backgrounds understand what's actually being said, not
just the words.

## Status

Early stage — building the training dataset.

| Phase | Status |
| --- | --- |
| Feasibility spike (base Gemma, prompt only) | Done — nuance detection works out of the box |
| Training data pipeline | In progress |
| LoRA fine-tune (Gemma 2B → 4B) | Next |
| On-device conversion + benchmarking | Planned |
| Keyboard (FlorisBoard fork) + nuance UI | Planned |
| Voice input (v1.1) | Planned |

Detailed design/product/execution docs are kept out of this public repo
(tracked locally only) — the code and pipeline tooling here are the
public-facing part of the project.

## How it works

- **One fine-tuned model** (Gemma, LoRA, 4-bit) returns both the
  translation and an optional short nuance note, in a single structured
  output. No note is shown when there's nothing worth flagging.
- **Training data** is synthetic and verified: a multi-model ensemble
  proposes annotated translations, deterministic checks and an automated
  prefilter clear the easy cases, and the contested ones get adjudicated
  and calibrated by hand. Failures feed the next batch's seeds. See
  [`data-pipeline/`](data-pipeline/).
- **Runtime** is fully on-device via LiteRT-LM, chosen because its NPU path
  covers both Qualcomm Snapdragon and MediaTek Dimensity phones.
- **No gates:** the app never refuses or withholds a translation. At worst
  a nuance note is suppressed when the model isn't confident.

## Built on

- [FlorisBoard](https://github.com/florisboard/florisboard) (MIT) — Android
  keyboard base
- [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) (Apache-2.0) —
  on-device LLM inference runtime (Kotlin API, standalone Gradle dependency)
- Planned for voice input (v1.1):
  [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (Apache-2.0) with
  [GigaAM v3](https://github.com/salute-developers/GigaAM) (MIT) for Russian
  speech recognition, and Android's built-in TextToSpeech

See [`NOTICE`](NOTICE) for full attribution.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).

## Why

Translation tools get the words right and the meaning wrong. This exists to
close that gap for one language pair, done well, rather than many language
pairs done shallowly.
