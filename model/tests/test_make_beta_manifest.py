"""make_beta_manifest: tier from file name, size and SHA-256, one file per tier."""
import json
import tempfile
import unittest
from pathlib import Path

import make_beta_manifest as mbm


class ManifestTest(unittest.TestCase):
    def test_build(self):
        with tempfile.TemporaryDirectory() as d:
            p4 = Path(d) / "podtekst-rua-q3-int4.litertlm"
            p4.write_bytes(b"abc")
            p8 = Path(d) / "podtekst-rua-q3-int8.litertlm"
            p8.write_bytes(b"abcd")
            m = mbm.build([p4, p8], "rua-q3")
            self.assertEqual([x["tier"] for x in m["models"]], ["int4", "int8"])
            self.assertEqual(m["models"][0]["bytes"], 3)
            self.assertEqual(m["models"][0]["sha256"],
                             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
            self.assertEqual(m["models"][1]["version"], "rua-q3")
            json.dumps(m)

    def test_rejects_unknown_tier_and_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "model.litertlm"
            bad.write_bytes(b"x")
            with self.assertRaises(ValueError):
                mbm.build([bad], "v")
            a = Path(d) / "a-int4.litertlm"
            a.write_bytes(b"x")
            b = Path(d) / "b-int4.litertlm"
            b.write_bytes(b"y")
            with self.assertRaises(ValueError):
                mbm.build([a, b], "v")


if __name__ == "__main__":
    unittest.main()
