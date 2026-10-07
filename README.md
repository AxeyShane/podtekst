<p align="right"><a href="README.ru.md">Read in Russian</a></p>
<div align="center">

# Podtekst

**Подтекст** — *"subtext"*

An on-device Android keyboard that translates **Russian ↔ English** and flags
what a literal translation would miss: tone, formality, sarcasm, idiom,
emotional subtext. Automatically, without the user asking.

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Status: building the dataset](https://img.shields.io/badge/status-building%20the%20dataset-orange)
![Runs fully on-device](https://img.shields.io/badge/runtime-fully%20on--device-2ea44f)
![Russian ↔ English](https://img.shields.io/badge/languages-RU%20%E2%86%94%20EN-informational)

</div>

---

Fully local, with no cloud calls at runtime. Built to help two people from
different language backgrounds understand what is actually being said, not
just the words.

## What it's for

Illustrative examples of the target behaviour; the fine-tuned model isn't
trained yet.

| You type | Literal translation | Podtekst |
| --- | --- | --- |
| Какого чёрта ты тут делаешь? | What devil are you doing here? | **What the hell are you doing here?**<br>*Idiom: a rough, surprised "what the hell", not about the devil.* |
| Вы не подскажете, где вокзал? | You will not tell me where the station is? | **Excuse me, could you tell me where the station is?**<br>*Formal вы: polite request to a stranger.* |
| Ну ты даёшь! | Well, you give! | **Wow, you're something else!**<br>*Exclamation of surprise or admiration, sometimes teasing.* |

When there's nothing worth flagging, no note is shown. The app never refuses
or withholds a translation; at worst a note is suppressed when the model isn't
confident.

## Progress

*Last updated 2026-10-07.*

| Phase | Status |
| --- | --- |
| Feasibility spike (base Gemma, prompt only) | ✅ Done: nuance detection works out of the box |
| Training data pipeline (synthetic + verified) | ✅ Phase 1 target reached: **3,815 verified rows** (3,802 unique sentences), split into 3,421 train / 381 test with near-duplicates kept on one side |
| Real-dialogue mining (film audio + subtitles) | 🔄 In progress: see below |
| LoRA fine-tune | 🔄 Starting: training and eval scripts in [`model/`](model/); first candidate QVikhr-3-1.7B (a Russian-tuned Qwen3), compared with Gemma 4 E2B/E4B on the test set |
| On-device conversion + benchmarking | 📋 Planned |
| Keyboard (FlorisBoard fork) + nuance UI | 📋 Planned |
| Voice input (v1.1) | 📋 Planned |

Verified rows by label: none 44%, formality shift (ты/вы) 20%, sarcasm 16%, idiom 15%,
emotional subtext 5%. Emotional subtext is the hardest category to collect: most emotions
survive translation, so only culture-specific ones count (обида, тоска, душевный,
умиление). A dedicated emotion-word mining pass over the subtitles raised it from 1% to 5%.

**Real-dialogue mining so far**

- **Film audio:** 76.6 hours of Russian film and series audio processed into
  **29,961 single-speaker clips (27.5 hours of speech)**. Each clip has dialogue
  separated from music, with overlapping speech and noisy stretches filtered out.
  Transcripts are machine-generated (GigaAM v3); a second engine (Whisper
  large-v3) agrees with them on 77% of clips, which serves as a confidence tier,
  not a human check.
- **Subtitles:** the OPUS OpenSubtitles RU-EN corpus is mined for lines where a
  human subtitler departed from a literal translation. Lines are limited to
  Russian-made films and filtered for broken text, and ты/вы cases are kept in
  their own capped bucket. An emotion-word mode collects lines with culture-specific
  emotion words (обида, тоска, душевный, …) for the emotional-subtext category.
- **Creator-subtitled videos:** interviews and podcasts whose creators published
  both Russian and English subtitles. The two tracks are aligned with LaBSE plus
  timing, and the human English is kept as a reference translation. Where only
  one side is subtitled, the other side is transcribed with whisper.cpp first.
- Mined lines and clips are **candidates, not labels**. They seed the same
  verification pipeline as the synthetic data, and none of the media or text
  is committed to this repo.

## How it works

```mermaid
flowchart LR
    S[Seed sentences<br/>synthetic + mined from film dialogue] --> G[Multi-model ensemble<br/>proposes annotated translations]
    G --> C[Deterministic checks<br/>+ automated prefilter]
    C --> V[Contested cases<br/>adjudicated and calibrated by hand]
    V --> D[(Verified dataset)]
    V -. failures feed the next batch .-> S
    D --> F[LoRA fine-tune<br/>small LLM, 4-bit]
    F --> K[On-device keyboard<br/>LiteRT-LM]
```

- **One fine-tuned model** (LoRA on a 1.7–4B model, 4-bit on the phone) returns the
  translation and an optional short nuance note in a single structured output. The base
  is chosen on the held-out test set: QVikhr-3-1.7B first, Gemma 4 E2B/E4B for comparison.
- **Training data** is synthetic and verified. A multi-model ensemble proposes
  annotated translations, deterministic checks and an automated prefilter
  clear the easy cases, and the contested ones are adjudicated and calibrated
  by hand. See [`data-pipeline/`](data-pipeline/), and
  [`data-pipeline/movie_mining/`](data-pipeline/movie_mining/) for the film
  and subtitle mining.
- **Runtime** is fully on-device via LiteRT-LM, chosen because its NPU path
  covers both Qualcomm Snapdragon and MediaTek Dimensity phones.

## Repository layout

| Path | What's there |
| --- | --- |
| [`data-pipeline/`](data-pipeline/) | Dataset generation and verification: seeds, model ensemble, checks, prefilter |
| [`data-pipeline/movie_mining/`](data-pipeline/movie_mining/) | Film audio → clean dialogue clips; OpenSubtitles miner; paired-subtitle aligner |
| [`data-pipeline/voice_eval/`](data-pipeline/voice_eval/) | Speech-recognition scoring: Indian-accented English (Svarah), your own recordings, Russian vs human subtitles |
| [`model/`](model/) | LoRA fine-tuning: shared prompt, SFT data prep, training, evaluation, merge for LiteRT-LM |

The keyboard lands here when that phase starts. Detailed design, product and execution docs are kept out of this public repo.
The code and pipeline tooling here are the public part of the project.

## Built on

- [FlorisBoard](https://github.com/florisboard/florisboard) (MIT): Android keyboard base
- [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) (Apache-2.0): on-device
  LLM inference runtime (Kotlin API, standalone Gradle dependency)
- Data pipeline: [Demucs](https://github.com/facebookresearch/demucs) (MIT)
  dialogue separation, [Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization),
  [GigaAM v3](https://github.com/salute-developers/GigaAM) (MIT) and
  [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (MIT) speech recognition,
  [OPUS OpenSubtitles](https://opus.nlpl.eu/OpenSubtitles.php) (Lison & Tiedemann, 2016)
- Planned for voice input (v1.1): [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
  (Apache-2.0) with GigaAM v3 for Russian speech recognition, and Android's
  built-in TextToSpeech

See [`NOTICE`](NOTICE) for full attribution.

## Why

Translation tools get the words right and the meaning wrong. Podtekst exists
to close that gap for one language pair, done well, rather than many language
pairs done shallowly.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
