# Movie mining

Build-time tooling that turns film dialogue into Podtekst training material.
It runs locally, and nothing here ships in the app.

There are two tracks:

| Track | Input | Output | Feeds |
| --- | --- | --- | --- |
| **Text** | OPUS OpenSubtitles RU-EN alignments | Pairs where the human subtitle departs from a literal MT: candidate idiom, register or sarcasm handling | Stage A seeds, with the human subtitle kept as reference |
| **Audio** | Your own film files + RU/EN subtitles | Clean single-speaker dialogue clips with subtitle text | ASR domain data and emotion2vec Russian validation (Phase 7) |

**Rights:** mined lines, audio and clips hold verbatim film dialogue. They stay
under `raw-media/`, `movie_mining/out/` and `seeds_subs_*.txt`, which are all
gitignored. Never commit or share them.

Run every command from `data-pipeline/`.

## Text track

```powershell
python -m venv .venv-mm; .\.venv-mm\Scripts\Activate.ps1
# install CUDA PyTorch first: https://pytorch.org/get-started/locally/
pip install -r movie_mining/requirements-text.txt

python -m movie_mining.fetch_opensubtitles                       # ~1 GB zip, resumable
python -m movie_mining.mine_subtitles --name subs1 --max-lines 3000000   # quick trial
python -m movie_mining.mine_subtitles --name subs2                        # whole corpus
```

The miner works in five steps:

1. **Clean and filter.** Strip markup, drop credits, multi-speaker cues and
   duplicates, and keep lines of 3–25 words. The pool is sampled evenly
   across the corpus.
2. **Check alignment.** A LaBSE cosine of at least 0.75 between RU and EN
   confirms the pair really is a translation; OpenSubtitles alignment is noisy.
3. **Literal MT.** Translate the source with `opus-mt` in the chosen
   direction(s).
4. **Measure divergence.** A pair qualifies if chrF(literal, human) is 45 or
   lower (a whole-line rewrite), or if the human line swaps in a span of 2+
   words inside an otherwise literal line (a local, idiom-sized swap). LaBSE
   also has to agree that the literal and human lines still mean the same
   thing, which drops real mistranslations.
5. **Rank.** Order by alignment × divergence, with small boosts for ты/вы
   address and idiom-lexicon hits. Cap each film at 15 candidates.

It writes three files:

- `movie_mining/out/subs_candidates_<name>.jsonl`: source, human translation,
  literal MT, scores, film id
- `movie_mining/out/subs_stats_<name>.json`: counts for each filter
- `seeds_subs_<name>.txt`: the top 500 source lines, ready for
  `run_batch.py --seeds seeds_subs_<name>.txt`

**These are candidates, not labels.** They go through the normal cycle:
Stage A, checks, prefilter, Cowork, then calibration. A professional
subtitler doing something different is evidence of nuance, not proof.
Fan-made subtitles in OpenSubtitles are noisier still.

The main tuning knobs are `--min-align`, `--max-chrf`, `--min-span`,
`--min-sem`, `--max-per-film` and `--directions ru-en,en-ru`.

## Audio track

```bash
# ffmpeg on PATH. NeMo (diarization) is best on Linux: on Windows use WSL2 Ubuntu + CUDA.
pip install -r movie_mining/requirements-audio.txt

python -m movie_mining.run_film raw-media/films/          # every video in the folder
python -m movie_mining.run_film raw-media/films/x.mkv --ru-srt x.ru.srt --en-srt x.en.srt
```

Sidecar subtitles named `film.ru.srt` and `film.en.srt` next to the video are
picked up automatically, and so are embedded text subtitle tracks.
Image-based subtitles (PGS/VobSub) need OCR first.

For each film:

1. `extract_dialogue`: on 5.1 or 7.1 audio, take the **center channel**
   (dialogue) and FL+FR (background); on stereo, fall back to **Demucs**
   vocal separation. The Russian audio track is chosen by language tag.
2. `diarize`: **Nemotron 3 Diarization** (`nvidia/Nemotron-3-Diarization`)
   in its most accurate offline profile. Use `--chunk-minutes 10` if a whole
   film runs out of GPU memory.
3. `cut_clips`: remove overlapping speech, keep 1–12 s pieces, and drop
   pieces with a speech-to-background ratio under 8 dB or near-silent ones.
   Attach RU/EN subtitle text.

The output lands in `raw-media/work/<film>/`: `clips/*.wav` plus
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
`python -m movie_mining.diarize raw-media/work/<film>` and
`python -m movie_mining.cut_clips raw-media/work/<film>`.

## Tests

```bash
python -m unittest discover -s movie_mining/tests -t .
```

The tests need no models. They cover the text filters, divergence and
ranking logic (with fake models), and a synthetic 5.1 film run through the
center-channel extraction, overlap removal, noise filter and subtitle
alignment.
