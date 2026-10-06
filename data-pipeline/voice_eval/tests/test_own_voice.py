"""own_voice scoring and file pairing (no models, no audio decoding)."""
import os
import tempfile
import unittest

from voice_eval import own_voice


class OwnVoiceTest(unittest.TestCase):
    def test_score_separates_fillers_and_hindi(self):
        ref = "Umm so matlab I was, I was thinking yaar"
        hyp = "So my lab I was thinking yeah"
        s = own_voice.score(ref, hyp, own_voice.HINDI_WORDS)
        self.assertEqual(s["words"], 9)
        self.assertEqual(s["words_nf"], 8)                 # "umm" dropped
        self.assertEqual(s["words_en"], 6)                 # "matlab", "yaar" dropped
        self.assertGreater(s["errors"], s["errors_en"])

    def test_find_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("talk01.m4a", "talk01.txt", "talk02.txt", "talk03.wav"):
                open(os.path.join(d, name), "w").close()
            pairs = own_voice.find_pairs(d)
        self.assertEqual([os.path.basename(a) for a, _ in pairs], ["talk01.m4a"])


if __name__ == "__main__":
    unittest.main()
