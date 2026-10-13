import ast
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kaggle"))
from launch import build_script, metadata  # noqa: E402


class LaunchTest(unittest.TestCase):
    def test_run_line_is_replaced_and_script_parses(self):
        src = build_script({"name": "v9", "epochs": 3, "oversample": ""})
        ast.parse(src)
        self.assertIn("RUN = {'name': 'v9', 'epochs': 3, 'oversample': ''}  # filled in by launch.py", src)
        self.assertEqual(src.count("RUN = {"), 1)

    def test_run_line_executes_with_booleans(self):
        run = {"name": "x", "qlora": False, "bases": [], "base": "Org/Модель"}
        line = next(l for l in build_script(run).splitlines() if l.startswith("RUN = "))
        ns = {}
        exec(line, ns)  # JSON's false/true would raise NameError here, as it did on Kaggle
        self.assertEqual(ns["RUN"], run)

    def test_metadata_attaches_only_the_dataset(self):
        m = metadata("me", "podtekst-train", "podtekst-sft")
        self.assertEqual(m["dataset_sources"], ["me/podtekst-sft"])
        self.assertEqual(m["kernel_sources"], [])
        self.assertTrue(m["enable_gpu"] and m["enable_internet"] and m["is_private"])

    def test_train_command_uses_batch_settings(self):
        src = build_script({"name": "x", "epochs": 2, "oversample": "", "batch": 2, "grad_accum": 8})
        self.assertIn("--batch {RUN.get('batch', 4)} --grad-accum {RUN.get('grad_accum', 4)}", src)

    def test_data_counts_reads_categories_not_substrings(self):
        import json, tempfile
        import train_kaggle
        d = tempfile.mkdtemp()
        rows = [{"category": "formality_shift", "nuance_note": ""},
                {"category": "none", "nuance_note": "not formality_shift"}]
        Path(d, "sft_train.jsonl").write_text("".join(json.dumps({"row": r}) + "\n" for r in rows), encoding="utf-8")
        Path(d, "sft_test.jsonl").write_text('{"row": {}}\n', encoding="utf-8")
        self.assertEqual(train_kaggle.data_counts(d), {"train": 2, "test": 1, "formality_shift": 1})

    def test_convert_metadata_is_cpu_with_adapter_dataset(self):
        m = metadata("me", "podtekst-convert", ["podtekst-lora-v2b"], gpu=False)
        self.assertEqual(m["dataset_sources"], ["me/podtekst-lora-v2b"])
        self.assertFalse(m["enable_gpu"])


if __name__ == "__main__":
    unittest.main()
