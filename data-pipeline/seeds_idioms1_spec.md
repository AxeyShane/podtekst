# Seeds for batch `idioms1`: idioms that lose their meaning word for word

## Why

In the 53-sentence comparison (see `model/RESULTS.md`), rua-q3 scores 83/106 against DeepL's 95. Most
of the gap is idioms: 27/34 vs 34/34 on RU->EN idiom/sarcasm/formality, and 15/24 vs 20/24 on EN->RU.
Failures are literal renderings («родился в рубашке» -> "born in a shirt") or a wrong guess at the
meaning. The training set has 825 idiom rows, but they cover a narrow set of idioms: of the 495 train
idioms below, only 105 appear in `train.jsonl` even once.

## Targets

About 1,000 train rows and about 100 held-out test rows after Stage B. The idioms are listed by hand in
`idioms1_list.txt`; `generate_idiom_seeds.py --check` validates the list and `--dry-run` prints the grid.

### Train, 1,242 seeds (`seeds_idioms1.txt`)

| Language | idiom uses | equivalent idioms ("none") | literal uses ("none") | total |
|---|---:|---:|---:|---:|
| RU (RU->EN) | 290 idioms x 2 = 580 | 35 x 3 = 105 | 15 x 2 = 30 | 715 |
| EN (EN->RU) | 205 idioms x 2 = 410 | 33 x 3 = 99 | 9 x 2 = 18 | 527 |

- **Idiom uses:** the idiom in its idiomatic sense; the reference translation carries the meaning, the
  nuance note says what a literal translation would lose.
- **Equivalent idioms (negatives):** the other language has the same image (играть с огнём / play with
  fire, капля в море / a drop in the ocean). Nothing is lost, so the label is `none`. This teaches the
  model not to flag every idiom.
- **Literal uses (negatives):** the idiom's words in their plain sense (сесть в лужу about an actual
  puddle). Label `none`.

About 250 negatives in all, so the batch does not push the false-alarm rate (12.5% on the phone) up.

### Test, 102 seeds (`seeds_idioms1_test.txt`)

25 RU + 25 EN idioms, 2 uses each, none of them in the train list, `train.jsonl` or any existing seed
file (`--check` enforces this). After Stage B they become `model/probes/idiom_test.jsonl`, an unseen-idiom
test: it measures whether the model learned to translate idioms by meaning, not just these idioms.

## Kept out on purpose

- **The 14 idioms of the 53-sentence comparison** (`BENCHMARK_EXCLUDE` in the script). They are never
  seeded and any line containing one is rejected, so a gain on that comparison comes from generalisation.
- **Train idioms that appear in the 380-row test set or the probes** (19 were dropped), so the main
  test's idiom scores are not inflated.
- Film- and subtitle-derived text: not used. Personal names, real people, lines from films, songs or books:
  not requested, and the same duplicate filters as every batch apply.

## Filters on every generated line

- 4-20 words, natural chat register, right script for the language.
- Contains the idiom (its key regex in `idioms1_list.txt`, any inflected form).
- New against every `seeds_*.txt`, the dataset, the eval probes and the native-review rows (normalised).

## Translation rules

`config/calibration_guidelines.md`, rule 15: an idiom is flagged when a word-for-word rendering is wrong
or misleading, and not flagged when both languages share it. Translations carry the meaning and use a
natural target idiom when one exists.

## Pipeline

    python generate_idiom_seeds.py --check
    python generate_idiom_seeds.py --split train --out seeds_idioms1.txt
    python generate_idiom_seeds.py --split test  --out seeds_idioms1_test.txt
    python stage_a_generate.py --seeds seeds_idioms1.txt --out stage_a_idioms1.jsonl \
        --failures stage_a_idioms1.failures.jsonl
    (retry_failures.py, then deterministic_checks.py)
    python stage_b_prefilter.py --in stage_a_idioms1.clean.jsonl ...  ->  Cowork

Check the OpenRouter balance before the first call. Seeds cost well under a dollar on the default seed
model; Stage A with the four-model roster is the main cost (about $5-15 for ~1,350 rows).
