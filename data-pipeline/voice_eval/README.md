# Voice evaluation

Scores speech recognition for the voice-input phase on the accents and speaking styles the
keyboard's users actually have. Nothing here ships in the app, and no audio is committed.

| Script | Data | Answers |
| --- | --- | --- |
| `svarah.py` | AI4Bharat Svarah (Indian-accented English, gated on Hugging Face) | How well a model handles Indian English accents, per speaker first language |
| `own_voice.py` | Your own recordings with verbatim transcripts | How well it handles *your* speech: accent, fillers, code-switching, restarts |
| `../movie_mining/asr_eval.py` | A video's Russian audio + human Russian subtitles | GigaAM word error rate on natural Russian speech |

Run everything from `data-pipeline/` with the movie-mining venv active.

## Svarah

1. Accept the terms on <https://huggingface.co/datasets/ai4bharat/Svarah>.
2. If offline mode is on (the setup script turns it on), clear it in the current window:
   `Remove-Item Env:HF_HUB_OFFLINE, Env:TRANSFORMERS_OFFLINE, Env:HF_DATASETS_OFFLINE -ErrorAction SilentlyContinue`
3. `pip install datasets`, then `hf auth login`.
4. `python -m voice_eval.svarah --model openai/whisper-large-v3 --languages hindi`

The first run downloads the dataset (~1.1 GB) and the model. Svarah's paper reports 7.2% WER for
Whisper large overall, which is a useful check that the scoring here is sane. Run a smaller model
(`--model openai/whisper-small`) too: large-v3 is the accuracy ceiling, not something a phone runs.

## Your own voice

Svarah covers accents but not code-switching, fillers or how one person actually talks.
About 10 minutes of your own speech closes that gap.

**Record.** Use your phone's voice recorder in a quiet room, holding the phone as you would
while dictating. Make 10–15 clips of 30–60 seconds each. Talk, don't read: pick a prompt and
answer it the way you'd tell a friend. Suggested prompts:

1. What you did yesterday, start to finish.
2. Explain your job to someone who has never seen a factory.
3. A trip that went wrong.
4. Give directions from your home to the nearest supermarket.
5. Something you're annoyed about this week.
6. Describe a film or series you watched recently and whether it was worth it.
7. Plan a weekend with a friend out loud, as if leaving them a voice note.
8. Explain how something you built works.
9. Ask a shop for a refund on something that broke.
10. Leave a voice message for a family member, the way you normally would, mixing languages.
11. Dictate a short work message: a delay, a request, a follow-up.
12. Read one paragraph from any article (one read clip is a useful contrast with the rest).

Include the way you normally talk at home: "matlab", "haan", "yaar", "accha", "umm", "aah",
starting a sentence again. That is the point of the exercise.

**Transcribe.** Next to each clip (`talk01.m4a`), save `talk01.txt` with exactly what you said:

- keep fillers ("umm", "uh", "hmm", "aah") as you heard them;
- keep repeats and restarts ("I was, I was thinking");
- write Hindi or other non-English words in Latin script ("matlab", "theek hai");
- write numbers as words if you said them as words;
- leave out anything you can't make out rather than guessing.

**Score.** Put the clips in a folder on the media drive, e.g. `D:\podtekst-mm\voice\me\`, then:

```powershell
python -m voice_eval.own_voice D:\podtekst-mm\voice\me --model openai/whisper-large-v3
python -m voice_eval.own_voice D:\podtekst-mm\voice\me --model openai/whisper-small
```

It reports three word error rates: raw, without hesitation sounds, and English-only (also without
the Hindi words). The gap between the last two shows how much of the error comes from
code-switching rather than accent. Per-clip results go to `own_voice_<model>.tsv` in the same folder.

## Tests

```bash
python -m unittest discover -s voice_eval/tests -t .
```
