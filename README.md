# Podtekst

**Подтекст** — "subtext." An on-device Android keyboard that translates
Russian ↔ English and flags what a literal translation would miss: tone,
formality, sarcasm, idiom, emotional subtext — automatically, without the
user asking.

Fully local. No cloud calls at runtime. Built to help two people from
different language backgrounds understand what's actually being said, not
just the words.

## Status

Early stage. Detailed design/product/execution docs are kept out of this
public repo (tracked locally only) — the code and pipeline tooling here are
the public-facing part of the project.

## Built on

- [FlorisBoard](https://github.com/florisboard/florisboard) (MIT) — Android
  keyboard base
- [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) (Apache-2.0) —
  on-device LLM inference runtime (Kotlin API, standalone Gradle dependency)

See [`NOTICE`](NOTICE) for full attribution.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).

## Why

Translation tools get the words right and the meaning wrong. This exists to
close that gap for one language pair, done well, rather than many language
pairs done shallowly.
