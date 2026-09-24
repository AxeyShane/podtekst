"""Movie/subtitle mining for Podtekst -- build-time only, never shipped in the app.

Text track: mine OpenSubtitles RU-EN pairs where the human subtitle diverges from
a literal machine translation (candidate nuance handling). Audio track: turn
films into clean single-speaker dialogue clips aligned to subtitles.

All mined text, audio and clips stay local (gitignored). See README.md.
"""
