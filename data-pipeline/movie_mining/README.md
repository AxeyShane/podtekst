# Movie mining

Build-time tooling that turns film dialogue into Podtekst training material.
It runs locally, and nothing here ships in the app.

There are two tracks:

| Track | Input | Output | Feeds |
| --- | --- | --- | --- |
| **Text** | OPUS OpenSubtitles RU-EN alignments | Pairs where the human subtitle departs from a literal MT: candidate idiom, register or sarcasm handling | Stage A seeds, with the human subtitle kept as reference |
| **Audio** | Your own film files + RU/EN subtitles | Clean single-speaker dialogue clips with subtitle text | ASR domain data and emotion2vec Russian validation (Phase 7) |

**Rights:** mined lines, audio and clips hold verbatim film dialogue. They stay
in the media root (default `raw-media/`, or `D:\podtekst-mm\media`), which is
outside git or gitignored. Never commit or share them.

Run every command from `data-pipeline/`.

## One-time setup (Windows)

```powershell
powershell -ExecutionPolicy Bypass -File data-pipeline\movie_mining\setup_windows.ps1
```

The script:

- checks Python, the GPU, git and ffmpeg (installing ffmpeg via winget if
  it's missing);
- creates a venv and installs CUDA PyTorch plus everything below;
- downloads the whisper.cpp CUDA build (and sets `WHISPER_CPP_BIN`);
- pre-downloads all models (~6 GB, including Whisper large-v3) and the
  OpenSubtitles zip (~1 GB);
- runs the tests.

If C: has less than 40 GB free and there's a D: drive, the venv, model
caches and media all go to `D:\podtekst-mm` instead. Force either drive with
`-Drive C` / `-Drive D`. The locations are saved as user environment variables
(`PODTEKST_MEDIA_ROOT`, `HF_HOME`, `TORCH_HOME`), so every script finds them.
Big files live under the **media root** (`paths.py`): `opensubtitles/`,
`films/`, `work/<film>/`, and `mining/`.

whisper.cpp is optional (only `--asr whisper` uses it). Setup takes the newest
release that ships a Windows x64 zip and just warns if there is none.

## Text track

```powershell
python -m movie_mining.mine_subtitles --name subs1 --max-lines 3000000              # quick trial
python -m movie_mining.mine_subtitles --name subs2 --max-lines 3000000 --origin ru  # Russian-origin films only
```

The miner works in five steps:

0. **Origin (optional, `--origin ru`).** Keep only films whose original
   language is Russian. The OPUS film id is the IMDb number; Wikidata maps it
   (P345 → P364). Answers are cached in `opensubtitles/film_lang.json`. Films
   Wikidata doesn't know, or without a language (many TV episodes), are dropped.
1. **Clean and filter.** Strip markup, drop credits, multi-speaker cues and
   duplicates, and keep lines of 3–25 words. Drop broken Russian: more than one
   word unknown to the pymorphy dictionary, or a doubled capital like «Мможет».
   Drop mis-decoded text (letters outside Russian Cyrillic/ASCII on the RU side,
   outside ASCII/Latin-1 on the EN side, e.g. «Ќа», «ƒа») and lines with no content
   words («О, о-о-о.», «Хе-хе-хе.», «Не-не-не!»).
   The pool is sampled evenly across the corpus.
2. **Check alignment.** A LaBSE cosine of at least 0.75 between RU and EN
   confirms the pair really is a translation; OpenSubtitles alignment is noisy.
3. **Literal MT.** Translate the source with `opus-mt` in the chosen
   direction(s).
4. **Measure divergence.** A pair qualifies if chrF(literal, human) is 45 or
   lower (a whole-line rewrite), or if the human line swaps in a span of 2+
   words inside an otherwise literal line (a local, idiom-sized swap). LaBSE
   also has to agree that the literal and human lines still mean the same
   thing, which drops real mistranslations.
5. **Rank.** Order by alignment × divergence, with a small boost for
   idiom-lexicon hits, in two buckets. English "you" makes every ты/вы line
   diverge from a literal MT, so lines with ты/вы go to an `address` bucket
   capped at 30% of the selection (`--address-share`); everything else is
   `general`. Cap each film at 15 candidates.

It writes three files to `<media root>/mining/` (`--out-dir` to change):

- `subs_candidates_<name>.jsonl`: source, human translation, literal MT,
  scores, bucket, film id
- `subs_stats_<name>.json`: counts for each filter and bucket
- `seeds_subs_<name>.txt`: the top 500 source lines, ready for
  `run_batch.py --seeds <media root>/mining/seeds_subs_<name>.txt`

**These are candidates, not labels.** They go through the normal cycle:
Stage A, checks, prefilter, Cowork, then calibration. A professional
subtitler doing something different is evidence of nuance, not proof.
Fan-made subtitles in OpenSubtitles are noisier still.

The main tuning knobs are `--min-align`, `--max-chrf`, `--min-span`,
`--min-sem`, `--max-per-film` and `--directions ru-en,en-ru`.

## Audio track

```powershell
python -m movie_mining.run_film                          # every video in <media root>\films
python -m movie_mining.run_film D:\films\x.mkv --ru-srt x.ru.srt --en-srt x.en.srt
```

Audio-only files (`.mp3`, `.m4a`, `.opus`, `.flac`, `.wav`, …) work too.
Sidecar subtitles named `film.ru.srt` and `film.en.srt` next to the file are
picked up automatically, and so are embedded text subtitle tracks.
Image-based subtitles (PGS/VobSub) need OCR first.

For each film:

1. `extract_dialogue`: on 5.1 or 7.1 audio, take the **center channel**
   (dialogue) and FL+FR (background); on stereo, fall back to **Demucs**
   vocal separation. The Russian audio track is chosen by language tag.
2. `diarize`: **Nemotron 3 Diarization** (`nvidia/Nemotron-3-Diarization`).
   There are two backends. `transformers` runs natively on Windows: it
   processes 120 s windows and thresholds the model's per-frame speaker
   probabilities into segments. `nemo` is the model card's reference path in
   its most accurate offline profile. Use it inside WSL2 (see
   `requirements-audio.txt`), and add `--chunk-minutes 10` if a whole film
   runs out of GPU memory. `--backend auto` picks NeMo when it's installed.
3. `cut_clips`: remove overlapping speech, keep 1–12 s pieces, and drop
   pieces with a speech-to-background ratio under 8 dB or near-silent ones.
   Attach RU/EN subtitle text when there are subtitles.
4. `transcribe` runs only when there's no Russian subtitle, for example on
   an audio-only file. It uses **GigaAM v3** (`v3_e2e_rnnt`: Russian-specialised,
   punctuated output) on each clip. Clips are at most 12 s, inside GigaAM's
   25 s limit, so no long-form mode, pyannote or HF token is needed. It fills
   the empty `ru_text` fields; human subtitle text is never overwritten.
   `--asr whisper` uses **whisper.cpp** (large-v3, Silero VAD, `-mc 0`)
   instead, on the whole dialogue stem before cutting. `--asr none` skips
   transcription. `ru_text_source` in the manifest records where each line
   came from.

Machine transcripts (GigaAM or Whisper) are **pseudo-labels**. It's fine for making clips
readable, spot-checking emotion2vec, and finding natural spoken lines to use
as seeds. They are *not* a reference for evaluating ASR, especially not GigaAM's own
output, since GigaAM is the ASR planned for the app. That still needs human
transcripts.

The output lands in `<media root>/work/<film>/`: `clips/*.wav` plus
`manifest.jsonl`, with clip, speaker, times, level, ratio, ru_text and en_text.

A few things to know before a big run:

- **Check Russian first.** Russian isn't in Nemotron 3's listed training
  languages, so hand-check one film before trusting a large batch.
- **Pick 2–4 person scenes.** Accuracy drops with 5 or more speakers.
- **Genre.** Prefer dramas, comedies and TV series. Action films are sparse
  on dialogue, and their exaggerated emotion skews the emotion2vec
  validation.
- **Local labels.** Speaker labels only hold within one film (or one chunk).
  They aren't actor identities.

Each step also runs on its own:
`python -m movie_mining.extract_dialogue film.mkv`,
`python -m movie_mining.diarize <media root>/work/<film>`,
`python -m movie_mining.cut_clips <media root>/work/<film>` and
`python -m movie_mining.transcribe <media root>/work/<film>`.

## Tests

```bash
python -m unittest discover -s movie_mining/tests -t .
```

The tests need no models. They cover the text filters, divergence and
ranking logic (with fake models), and a synthetic 5.1 film run through the
center-channel extraction, overlap removal, noise filter and subtitle
alignment.
