# Podtekst calibration guidelines

Living document. The **Adjudicator Agent** reads this before resolving contested
candidates; the **Guideline Agent** proposes new entries here when it spots a
recurring disagreement pattern (`agents_guideline.py`) -- proposals are appended
under "Pending review" and only moved into the numbered rules below once a
maintainer approves them. This is the same calibration knowledge that previously
only lived in individual verification sessions -- formalizing it here means future
Annotator Agent runs can see it too, instead of it being rediscovered by whichever
session happens to be verifying that batch.

Seeded 2026-09-20 from the calibration notes established across Stage B batches
1-3 (see the project's `stage-b-verification-log.md` for the full history each
rule came from).

## Formality shift (ты/вы, RU<->EN register)

1. RU->EN ты/вы: flag `formality_shift` only when there's a direct address
   pronoun (ты/вы/тебя/вас) or an informal/formal imperative verb form carrying
   real social stakes. Don't flag when the register already carries through
   cleanly into an equally formal/informal English translation.
2. EN->RU "you"/"your": flag only when the English gives NO cue at all about
   which register fits ("Whatever works for you is fine"). Don't flag when the
   English phrasing itself already resolves the register.
3. Explicit English address/politeness terms resolve register on their own --
   no flag needed: "Ma'am", "Sir", "Dude", "bro", "Excuse me" as a
   request-opener, "Would you be so kind...", "I was wondering if you
   could...", standard customer-service phrasing ("How can I assist you
   today?"), business-register hedges ("I would appreciate it if you
   could...", "I appreciate your patience with this matter").
4. Content-level warmth can make an informal pronoun's closeness redundant:
   an endearment ("дорогая"/"dear") or a sincere vulnerable statement ("I
   really appreciate that you listened to me") already conveys the closeness,
   so the bare pronoun isn't adding separately-hidden information -- false.
   Contrast with plain everyday instructions ("Call me when you're home")
   where the pronoun genuinely is the only signal of closeness -- stays true.
5. A source sentence *about* the ty/vy distinction itself ("Обращайтесь ко
   мне на «ты», мы же коллеги") is an unambiguous true case -- the Russian
   pronoun has no English equivalent and must stay glossed.
6. Genuinely neutral EN->RU instructions with zero register cue either way
   ("Don't forget to lock the door...", "Please send me the file...") stay
   true -- Russian must choose and English gives no basis to predict which.

## Universal pragmatic patterns (survive direct translation -> false)

These patterns carry their meaning through a direct/literal translation
intact, so `has_subtext` is false even when candidate models unanimously (or
near-unanimously) flag them true. The test is always: does the English
reader lose something a Russian reader gets, or does literal translation
already deliver it?

7. Protesting-too-much, backhanded compliments, litotes ("not the worst
   option"), reluctant-gratitude ("I guess"), "of course" as a
   reluctant-concession marker, silver-lining litotes.
8. Resigned-fatalism idioms ("должно было случиться" -> "meant to happen",
   "смириться" -> "live with it") -- see config/idiom_lexicon_ru_en.json.
9. Plain idiomatic emotional-intensity matches ("какого чёрта" -> "what the
   hell", "совсем сдурел" -> "out of your mind") carry their full charge
   through direct idiomatic translation.

## Sourcing rules (unverified claims)

10. Reject a candidate's claim of a specific cultural/literary reference
    (e.g. a Winnie-the-Pooh allusion for "Кажется, дождь собирается") unless
    independently verifiable -- treat as a likely-fabricated model claim and
    go with the plain reading.

## Ambiguous cases (needs_human_review)

11. Standalone sentences with no conversation history where sincere-vs-
    sarcastic tone is a genuine coin-flip (e.g. repeated "He's always so X,
    as usual" pattern, "I'm so lucky that you took on this responsibility")
    -> `needs_human_review`, not a forced guess.

## Pending review

<!-- Guideline Agent proposals land here, newest first. Each entry: the
     pattern, the proposed rule text, 2-4 supporting examples, and the
     direction (which way the correction goes). A maintainer approves by moving
     the rule text up into the numbered list above and deleting the entry
     here; agents_guideline.py never edits the numbered list directly. -->

(none yet)
