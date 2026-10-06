"""asr_eval: segment merging, word normalisation and WER, with a fake ASR (no models)."""
import tempfile
import unittest

import numpy as np

from movie_mining import asr_eval


class FakeASR:
    def __init__(self, outputs): self.outputs = list(outputs)
    def transcribe(self, path): return self.outputs.pop(0)


class AsrEvalTest(unittest.TestCase):
    def test_words_normalise_and_drop_fillers(self):
        self.assertEqual(asr_eval.words("Ну, э-э, всё ЁЛКИ!"), ["ну", "э-э", "все", "елки"])
        self.assertEqual(asr_eval.words("Ну, э-э, всё ЁЛКИ!", drop_fillers=True), ["ну", "все", "елки"])

    def test_segments_merge_close_cues_and_skip_short_ones(self):
        cues = [{"start": 0.0, "end": 1.0, "text": "Что ты на меня смотришь?"},
                {"start": 1.2, "end": 2.5, "text": "Юра, я с кем разговариваю?"},
                {"start": 5.0, "end": 5.4, "text": "Да."},
                {"start": 8.0, "end": 10.0, "text": "Зачем вообще в профессию пришел?"}]
        segs = asr_eval.segments(cues)
        self.assertEqual([(s["start"], s["end"]) for s in segs], [(0.0, 2.5), (8.0, 10.0)])
        self.assertIn("Юра", segs[0]["text"])

    def test_evaluate_and_summarize(self):
        try:
            import soundfile  # noqa: F401
        except ImportError:
            self.skipTest("soundfile not installed")
        segs = [{"start": 0.0, "end": 2.0, "text": "Я родился в Москве."},
                {"start": 2.0, "end": 4.0, "text": "Э-э, я говорил на мегрельском."}]
        audio = np.zeros(16000 * 4, dtype="float32")
        asr = FakeASR(["я родился в москве", "я говорил по мегрельски"])
        with tempfile.TemporaryDirectory() as d:
            rows = asr_eval.evaluate(segs, audio, 16000, asr, d)
        s = asr_eval.summarize(rows)
        self.assertEqual(rows[0]["errors"], 0)
        self.assertEqual(s["ref_words"], 9)
        self.assertAlmostEqual(s["wer_no_fillers"], 2 / 8)


if __name__ == "__main__":
    unittest.main()
