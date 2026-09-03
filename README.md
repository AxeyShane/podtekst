# Podtekst

**Подтекст** — "subtext." An on-device Android keyboard that translates
Russian ↔ English and flags what a literal translation would miss: tone,
formality, sarcasm, idiom, emotional subtext — automatically, without the
user asking.

Fully local. No cloud calls at runtime. Built to help two people from
different language backgrounds understand what's actually being said, not
just the words.

## Status

Early stage — see [`docs/EXECUTION_PLAN.md`](docs/EXECUTION_PLAN.md) for
where the project currently sits.

## Docs

- [`docs/PRODUCT.md`](docs/PRODUCT.md) — what this is, who it's for, scope,
  success metrics, known limitations
- [`docs/DESIGN.md`](docs/DESIGN.md) — architecture, data pipeline, model
  approach, code reuse
- [`docs/EXECUTION_PLAN.md`](docs/EXECUTION_PLAN.md) — phased build plan

## Built on

- [FlorisBoard](https://github.com/florisboard/florisboard) (MIT) — Android
  keyboard base
- [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery)
  (Apache-2.0) — on-device LiteRT model loading/inference

See [`NOTICE`](NOTICE) for full attribution.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).

## Why

Translation tools get the words right and the meaning wrong. This exists to
close that gap for one language pair, done well, rather than many language
pairs done shallowly.
