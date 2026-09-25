"""Audio-track tests on a synthetic 5.1 file (needs ffmpeg + soundfile; no models).
Diarization itself is faked with a hand-written segments.json."""
import json
import shutil
import subprocess
import sys
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

    def test_probs_to_segments(self):
        import numpy as np
        p = np.zeros((100, 2))              # 100 frames x 0.1 s = 10 s
        p[10:40, 0] = 0.9                   # spk0 1.0-4.0 s
        p[41:45, 0] = 0.9                   # 0.1 s gap -> bridged
        p[35:70, 1] = 0.8                   # spk1 3.5-7.0 s
        p[90:91, 1] = 0.9                   # 0.1 s blip -> dropped
        segs = diarize.probs_to_segments(p, 0.1, offset=60.0, prefix="w0_")
        self.assertEqual(segs, [{"start": 61.0, "end": 64.5, "speaker": "w0_0"},
                                {"start": 63.5, "end": 67.0, "speaker": "w0_1"}])

    def test_overlap_and_srt_helpers(self):
        segs = [{"start": 0, "end": 5, "speaker": "0"}, {"start": 4, "end": 8, "speaker": "1"},
                {"start": 6, "end": 7, "speaker": "0"}]
        self.assertEqual(cut_clips.overlap_regions(segs), [(4, 5), (6, 7)])
        self.assertEqual(cut_clips.subtract((0, 10), [(4, 5), (6, 7)]), [(0, 4), (5, 6), (7, 10)])
        cues = parse_srt(SRT_RU)
        self.assertEqual(len(cues), 3)
        self.assertEqual(text_in_window(cues, 1.0, 3.5), "Ты где был?")
        self.assertEqual(text_in_window(cues, 5.5, 8.0), "")               # only 1 s of a 2.3 s cue

    def test_merge_same_speaker(self):
        segs = [{"start": 0.0, "end": 0.8, "speaker": "a"}, {"start": 1.1, "end": 1.9, "speaker": "a"},  # gap .3
                {"start": 2.0, "end": 3.0, "speaker": "b"},
                {"start": 3.2, "end": 4.0, "speaker": "a"},                                         # gap 1.3
                {"start": 4.2, "end": 5.0, "speaker": "b"}]                                         # gap 1.2
        out = cut_clips.merge_same_speaker(segs, 0.5)
        self.assertEqual([(s["speaker"], s["start"], s["end"]) for s in out],
                         [("a", 0.0, 1.9), ("b", 2.0, 3.0), ("a", 3.2, 4.0), ("b", 4.2, 5.0)])
        self.assertEqual(segs[0]["end"], 0.8)                                     # input left untouched
        self.assertEqual(len(cut_clips.merge_same_speaker(segs, 0.0)), 5)         # 0 disables merging


if __name__ == "__main__":
    unittest.main()


class DemucsTests(unittest.TestCase):
    def test_runs_in_current_interpreter(self):
        # A bare "python" resolves to whatever is first on PATH, which may lack demucs.
        from unittest import mock
        with tempfile.TemporaryDirectory() as d, mock.patch.object(extract_dialogue, "_run") as run, \
                mock.patch.object(extract_dialogue, "duration", return_value=600.0):
            extract_dialogue.extract_demucs(Path("film.m4a"), 0, Path(d), None)
        demucs = next(c.args[0] for c in run.call_args_list if "demucs" in c.args[0])
        self.assertEqual(demucs[:3], [sys.executable, "-m", "demucs"])

    def test_chunk_spans(self):
        spans = extract_dialogue.chunk_spans
        self.assertEqual(spans(3000, 1200), [(0, 1200), (1200, 1200), (2400, 600)])
        self.assertEqual(spans(2430, 1200), [(0, 1200), (1200, 1230)])          # 30 s tail joins the last chunk
        self.assertEqual(spans(500, 1200), [(0, 500)])

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
    def test_chunked_demucs_concatenates_in_order(self):
        """Real ffmpeg, fake Demucs (copies the mix to both stems): 3.5 s of rising-amplitude tone in
        1 s chunks must come back as one 3.5 s stem, still rising."""
        import numpy as np
        import soundfile as sf
        from unittest import mock
        real_run = extract_dialogue._run
        demucs_calls = []

        def run(cmd):
            if "demucs" in cmd:
                demucs_calls.append(cmd)
                mix = Path(cmd[-1])
                stem_dir = Path(cmd[cmd.index("-o") + 1]) / "htdemucs" / mix.stem
                stem_dir.mkdir(parents=True)
                for stem in ("vocals", "no_vocals"):
                    shutil.copyfile(mix, stem_dir / f"{stem}.wav")
                return ""
            return real_run(cmd)

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            src = tmp / "long.m4a"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                            "aevalsrc='0.2*t*sin(2*PI*440*t)':s=44100:d=3.5:c=stereo", str(src)], check=True)
            with mock.patch.object(extract_dialogue, "_run", side_effect=run):
                extract_dialogue.extract_demucs(src, 0, tmp, None, chunk_minutes=1 / 60)
            expected = extract_dialogue.chunk_spans(extract_dialogue.duration(src), 1.0)  # AAC pads a little
            self.assertGreaterEqual(len(expected), 3)
            self.assertEqual(len(demucs_calls), len(expected))
            for name in ("dialogue", "background"):
                audio, sr = sf.read(tmp / f"{name}.wav")
                self.assertEqual(sr, extract_dialogue.SR)
                self.assertAlmostEqual(len(audio) / sr, 3.5, delta=0.1)
                rms = [float(np.sqrt(np.mean(audio[int(i * sr):int((i + 1) * sr)] ** 2))) for i in range(3)]
                self.assertTrue(rms[0] < rms[1] < rms[2], rms)                   # chunks joined in order


class CrossCheckTests(unittest.TestCase):
    def test_cer_normalisation(self):
        from movie_mining.transcribe import cer
        self.assertEqual(cer("ещё раз, ПРИВЕТ!", "Еще раз привет."), 0.0)   # case, punctuation, ё
        self.assertAlmostEqual(cer("кот", "кит"), 1 / 3)
        self.assertEqual(cer("", ""), 0.0)
        self.assertEqual(cer("шум", ""), 1.0)

    def test_cross_check_flags_disagreement(self):
        from movie_mining import transcribe
        with tempfile.TemporaryDirectory() as d:
            wd = Path(d)
            (wd / "clips").mkdir()
            rows = [{"clip": "clips/a.wav", "ru_text": "Ты где был?", "ru_text_source": "human_subs"},
                    {"clip": "clips/b.wav", "ru_text": "С лёгким паром!", "ru_text_source": "gigaam_x"},
                    {"clip": "clips/c.wav", "ru_text": "Надо выпить.", "ru_text_source": "gigaam_x"}]
            (wd / "manifest.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                                               encoding="utf-8")
            seen = []

            def fake(clips_dir, names):
                seen.extend(names)
                return {"b.wav": "с легким паром", "c.wav": "Продолжение следует"}
            stats = transcribe.cross_check(wd, fake)
            self.assertEqual(seen, ["b.wav", "c.wav"])                           # human rows skipped
            out = [json.loads(l) for l in (wd / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertNotIn("asr_agree", out[0])
            self.assertEqual((out[1]["asr_cer"], out[1]["asr_agree"]), (0.0, True))
            self.assertFalse(out[2]["asr_agree"])
            self.assertEqual(out[2]["ru_text_whisper"], "Продолжение следует")
            self.assertEqual(stats["asr_agree_pct"], 50.0)


class WhisperTests(unittest.TestCase):
    @unittest.skipIf(sys.platform == "win32", "fake whisper-cli is a POSIX shell script")
    def test_command_and_fallback(self):
        import os
        import stat
        from movie_mining import transcribe
        cmd = transcribe.build_command("whisper-cli", "m.bin", Path("d.wav"), Path("wd/ru.whisper"), vad="v.bin")
        self.assertEqual(cmd[:12], ["whisper-cli", "-m", "m.bin", "-f", "d.wav", "-l", "ru", "-osrt", "-of",
                                    str(Path("wd/ru.whisper")), "-t", "8"])
        self.assertIn("--vad", cmd)

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            # fake whisper-cli: writes <-of>.srt like the real one
            fake = tmp / "whisper-cli"
            fake.write_text('#!/bin/sh\nwhile [ "$1" ]; do [ "$1" = "-of" ] && out="$2"; shift; done\n'
                            'printf "1\\n00:00:01,000 --> 00:00:03,000\\nТы где был?\\n" > "$out.srt"\n',
                            encoding="utf-8")
            fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
            wd = tmp / "work"
            wd.mkdir()
            import numpy as np
            import soundfile as sf
            sr = 16000
            t = np.arange(5 * sr) / sr
            sf.write(wd / "dialogue.wav", (0.3 * np.sin(2 * np.pi * 220 * t) * ((t > 1) & (t < 3))).astype("float32"), sr)
            sf.write(wd / "background.wav", np.zeros_like(t, dtype="float32"), sr)
            (wd / "segments.json").write_text(json.dumps([{"start": 1.0, "end": 3.0, "speaker": "0"}]))
            orig = transcribe.model_path
            transcribe.model_path = lambda m: "m.bin"
            try:
                out = transcribe.transcribe(wd, use_vad=False, binary=str(fake))
            finally:
                transcribe.model_path = orig
            self.assertEqual(out.name, "ru.whisper.srt")
            cut_clips.cut(wd)
            row = json.loads((wd / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["ru_text"], "Ты где был?")
            self.assertEqual(row["ru_text_source"], "whisper")


class GigaAMTests(unittest.TestCase):
    def test_transcribe_clips_fills_only_missing_text(self):
        from movie_mining import transcribe

        class FakeASR:
            name = "gigaam_fake"

            def transcribe(self, path):
                return "Привет, как дела?"

        with tempfile.TemporaryDirectory() as d:
            wd = Path(d)
            rows = [{"clip": "clips/a.wav", "ru_text": "Ты где был?", "ru_text_source": "human_subs"},
                    {"clip": "clips/b.wav", "ru_text": "", "ru_text_source": None}]
            (wd / "manifest.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                                                  encoding="utf-8")
            self.assertEqual(transcribe.transcribe_clips(wd, FakeASR()), 1)
            out = [json.loads(l) for l in (wd / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(out[0]["ru_text"], "Ты где был?")
            self.assertEqual(out[1]["ru_text"], "Привет, как дела?")
            self.assertEqual(out[1]["ru_text_source"], "gigaam_fake")
