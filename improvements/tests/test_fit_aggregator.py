"""J19 fit_aggregator: non-negative least squares and weights-file provenance."""
import hashlib
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "improvements"))
sys.path.insert(0, str(ROOT / "scripts"))

import fit_aggregator  # noqa: E402
from drag_plus.grounding.scorer import SIGNAL_NAMES, load_weights  # noqa: E402


def _synthetic_rows(known, count=12, seed=7):
    rng = random.Random(seed)
    rows = []
    for _ in range(count):
        features = {name: rng.random() for name in SIGNAL_NAMES}
        target = sum(known[name] * features[name] for name in SIGNAL_NAMES)
        rows.append({**features, "target": target})
    return rows


class FitScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def test_recovers_known_weights_on_synthetic_data(self):
        known = {"conf": 0.5, "s_faith": 0.3, "agr": 0.2}
        fitted = fit_aggregator.fit_weights(_synthetic_rows(known), l2=0.0)
        for name in SIGNAL_NAMES:
            self.assertAlmostEqual(fitted[name], known[name], places=6)

    def test_cli_writes_a_loadable_file_with_split_and_manifest(self):
        known = {"conf": 0.5, "s_faith": 0.3, "agr": 0.2}
        rows = _synthetic_rows(known)
        source = Path(self.temp.name) / "rows.jsonl"
        source.write_text("\n".join(json.dumps(
            {"features": {name: row[name] for name in SIGNAL_NAMES},
             "target": row["target"]}) for row in rows) + "\n", encoding="utf-8")
        out = Path(self.temp.name) / "weights.json"
        code = fit_aggregator.main(["--input", str(source), "--out", str(out),
                                    "--split", "dev", "--l2", "0"])
        self.assertEqual(code, 0)
        model = load_weights(out)
        self.assertEqual(model.training_split, "dev")
        self.assertEqual(model.manifest_sha256,
                         hashlib.sha256(source.read_bytes()).hexdigest())
        for name in SIGNAL_NAMES:
            self.assertAlmostEqual(getattr(model, name), known[name], places=6)

    def test_cli_refuses_the_test_split_and_writes_nothing(self):
        source = Path(self.temp.name) / "rows.jsonl"
        source.write_text(json.dumps({"features": {"conf": 1.0, "s_faith": 0.0,
                                                   "agr": 0.0}, "target": 0.5}) + "\n",
                          encoding="utf-8")
        out = Path(self.temp.name) / "weights.json"
        code = fit_aggregator.main(["--input", str(source), "--out", str(out),
                                    "--split", "test"])
        self.assertEqual(code, 2)
        self.assertFalse(out.exists())

    def test_singular_design_and_bad_rows_are_errors(self):
        rows = [{"conf": 0.5, "s_faith": 0.5, "agr": 0.5, "target": 0.5}] * 4
        with self.assertRaisesRegex(fit_aggregator.FitError, "singular"):
            fit_aggregator.fit_weights(rows)
        bad = Path(self.temp.name) / "bad.jsonl"
        bad.write_text('{"features": {"conf": 1.0}, "target": 1}\n', encoding="utf-8")
        with self.assertRaisesRegex(fit_aggregator.FitError, "features"):
            fit_aggregator.read_rows(bad)


if __name__ == "__main__":
    unittest.main()
