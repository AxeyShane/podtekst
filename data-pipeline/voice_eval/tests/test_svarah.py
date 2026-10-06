"""Svarah scoring helpers (no dataset, no models)."""
import unittest

from voice_eval import svarah


class SvarahHelpersTest(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(svarah.normalize("I'm going to the e-mail, OK?"),
                         ["i", "am", "going", "to", "the", "e", "mail", "ok"])

    def test_wer_pieces(self):
        ref, hyp = svarah.normalize("I spoke to him today"), svarah.normalize("I spoke him to day")
        self.assertEqual(svarah.edit_distance(ref, hyp), 3)

    def test_find_column(self):
        cols = ["audio_filepath", "text", "gender", "primary_language", "duration"]
        self.assertEqual(svarah.find_column(cols, "audio"), "audio_filepath")
        self.assertEqual(svarah.find_column(cols, "primary_language", "lang"), "primary_language")
        self.assertIsNone(svarah.find_column(cols, "speaker"))

    def test_summarize(self):
        rows = [{"language": "Hindi", "ref_words": 10, "errors": 1},
                {"language": "Kannada", "ref_words": 10, "errors": 3}]
        s = svarah.summarize(rows)
        self.assertEqual(s["ALL"], (2, 20, 4))
        self.assertEqual(s["Hindi"], (1, 10, 1))


if __name__ == "__main__":
    unittest.main()
