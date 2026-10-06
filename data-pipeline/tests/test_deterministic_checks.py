"""deterministic_checks: empty translations and echo drift (no network)."""
import unittest

import deterministic_checks as dc


def row(**kw):
    r = {"source_lang": "en", "source_text": "And then I, like, would avoid him, like, in school.",
         "translation": "А потом я, типа, избегал его в школе.", "has_subtext": False,
         "category": "none", "nuance_note": "", "_generator_model": "m"}
    r.update(kw)
    return r


class EchoDriftTest(unittest.TestCase):
    def test_unrelated_echo_is_rejected(self):
        r = row(translation="Кот сидел на коврике.", _model_echoed_source_text="The cat sat on the mat.")
        self.assertTrue(dc.check_echo_drift(r))

    def test_partial_echo_is_rejected(self):
        r = row(source_lang="ru", source_text="Ну, у каждого, знаете, свой крест. Аркадий Варламович, к вам пришли.",
                _model_echoed_source_text="Ну, у каждого, знаете, свой крест.")
        self.assertTrue(dc.check_echo_drift(r))

    def test_punctuation_and_typo_echoes_pass(self):
        self.assertEqual(dc.check_echo_drift(row(source_text="It’s fine, I didn’t need it.",
                                                 _model_echoed_source_text="It's fine, I didn't need it.")), [])
        self.assertEqual(dc.check_echo_drift(row(source_lang="ru", source_text="Все зависиттолько от вас.",
                                                 _model_echoed_source_text="Всё зависит только от вас.")), [])

    def test_no_echo_field_passes(self):
        self.assertEqual(dc.check_echo_drift(row()), [])


class TranslationPresentTest(unittest.TestCase):
    def test_empty_translation_is_rejected(self):
        self.assertTrue(dc.check_translation_present(row(translation="  ")))
        self.assertEqual(dc.check_translation_present(row()), [])


if __name__ == "__main__":
    unittest.main()
