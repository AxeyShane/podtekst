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
        self.assertIn('RUN = {"name": "v9", "epochs": 3, "oversample": ""}  # filled in by launch.py', src)
        self.assertEqual(src.count("RUN = {"), 1)

    def test_metadata_attaches_only_the_dataset(self):
        m = metadata("me", "podtekst-train", "podtekst-sft")
        self.assertEqual(m["dataset_sources"], ["me/podtekst-sft"])
        self.assertEqual(m["kernel_sources"], [])
        self.assertTrue(m["enable_gpu"] and m["enable_internet"] and m["is_private"])


if __name__ == "__main__":
    unittest.main()
