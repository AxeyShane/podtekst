"""Audio-track tests on a synthetic 5.1 file (needs ffmpeg + soundfile; no models).
Diarization itself is faked with a hand-written segments.json."""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from movie_mining import cut_clips, diarize, extract_dialogue
from movie_mining.srt import parse_srt, text_in_window

SRT_RU = """1
00:00:01,200 --> 00:00:03,000
- <i>Ты где был?</i>

2
00:00:04,200 --> 00:00:06,500
Не твоё дело.

3
00:00:09,200 --> 00:00:10,800
Взрыв!
"""


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
class AudioTests(unittest.TestCase):
    def make_film(self, tmp: Path) -> Path:
        """12 s, 5.1. FC: speaker A (220 Hz) 1-4 s, speaker B (440 Hz) 3.5-7 s,
        speaker A again 9-11 s. FL/FR: quiet noise, then very loud noise 8.5-12 s."""
        film = tmp / "film.mkv"
        fc = ("0.3*sin(2*PI*220*t)*between(t,1,4)+0.3*sin(2*PI*440*t)*between(t,3.5,7)"
              "+0.3*sin(2*PI*220*t)*between(t,9,11)")
        side = "0.003*(random(0)-0.5)+0.9*(random(1)-0.5)*between(t,8.5,12)"
        expr = f"{side}|{side}|{fc}|0|0|0"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        f"aevalsrc=exprs='{expr}':s=48000:d=12:c=5.1", "-c:a", "pcm_s16le",
                        "-metadata:s:a:0", "language=rus", str(film)], check=True)
        return film

    def test_center_extract_and_cut(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            film = self.make_film(tmp)
            (tmp / "film.ru.srt").write_text(SRT_RU, encoding="utf-8")
            wd = extract_dialogue.extract(film, tmp / "work" / "film", ru_srt=tmp / "film.ru.srt")
            meta = json.loads((wd / "meta.json").read_text())
            self.assertEqual(meta["method"], "center_channel")
            self.assertEqual(meta["audio_lang"], "rus")
            self.assertTrue((wd / "ru.srt").exists())

            segs = [diarize.parse_segment(x) for x in ["1.0 4.0 speaker_0", "3.5, 7.0, speaker_1", "9.0 11.0 speaker_0"]]
            self.assertEqual(segs[1], {"start": 3.5, "end": 7.0, "speaker": "1"})
            (wd / "segments.json").write_text(json.dumps(segs))

            stats = cut_clips.cut(wd, min_dur=1.0, min_sbr=8.0)
            rows = [json.loads(l) for l in (wd / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(stats["overlap_regions"], 1)
            self.assertEqual(stats["drop_noisy"], 1)                   # 9-11 s under loud side noise
            self.assertEqual(len(rows), 2)
            a, b = rows
            self.assertAlmostEqual(a["end"], 3.5, places=2)            # overlap 3.5-4 removed
            self.assertAlmostEqual(b["start"], 4.0, places=2)
            self.assertEqual(a["ru_text"], "Ты где был?")
            self.assertEqual(b["ru_text"], "Не твоё дело.")
            self.assertGreater(a["sbr_db"], 20)
            self.assertTrue((wd / a["clip"]).exists())

    def test_overlap_and_srt_helpers(self):
        segs = [{"start": 0, "end": 5, "speaker": "0"}, {"start": 4, "end": 8, "speaker": "1"},
                {"start": 6, "end": 7, "speaker": "0"}]
        self.assertEqual(cut_clips.overlap_regions(segs), [(4, 5), (6, 7)])
        self.assertEqual(cut_clips.subtract((0, 10), [(4, 5), (6, 7)]), [(0, 4), (5, 6), (7, 10)])
        cues = parse_srt(SRT_RU)
        self.assertEqual(len(cues), 3)
        self.assertEqual(text_in_window(cues, 1.0, 3.5), "Ты где был?")
        self.assertEqual(text_in_window(cues, 5.5, 8.0), "")               # only 1 s of a 2.3 s cue


if __name__ == "__main__":
    unittest.main()
