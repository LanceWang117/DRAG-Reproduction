"""Offline evidence-recall replay: both policies, the proxy caveat and the dev filter."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT / "scripts"


def passage(doc_id, contents):
    return {"id": doc_id, "contents": contents}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class ReplayTests(unittest.TestCase):
    def make_inputs(self, folder):
        questions = [
            {"dataset": "NQ", "split": "dev", "id": "q1", "question": "q1?",
             "golden_answers": ["Paris"]},
            {"dataset": "StrategyQA", "split": "dev", "id": "s1", "question": "s1?",
             "golden_answers": ["Yes"]},
            {"dataset": "HotpotQA", "split": "dev", "id": "q3", "question": "q3?",
             "golden_answers": ["Nile"]},
        ]
        records = [
            {"id": "q1", "drag": {"final query": [passage("a", "nothing useful")]},
             "naive": [passage("b", "Paris is the capital")]},
            {"id": "s1", "drag": {"final query": [passage("c", "no")]},
             "naive": [passage("d", "Yes")]},
            {"id": "q3", "drag": {"final query": [passage("e", "the Nile river")]},
             "naive": [passage("e", "the Nile river")]},
        ]
        qpath, rpath = folder / "questions.jsonl", folder / "retrieval.jsonl"
        write_jsonl(qpath, questions)
        write_jsonl(rpath, records)
        return rpath, qpath

    def test_both_policies_score_the_175_style_dev_subset(self):
        sys.path.insert(0, str(SCRIPTS))
        import replay_evidence
        with tempfile.TemporaryDirectory() as name:
            rpath, qpath = self.make_inputs(Path(name))
            report = replay_evidence.run(rpath, qpath)
        self.assertEqual(report["baseline"], {"n": 2, "recall_count": 1, "recall": 0.5})
        self.assertEqual(report["anchored"], {"n": 2, "recall_count": 2, "recall": 1.0})

    def test_pool_deduplicates_by_doc_id_before_scoring(self):
        sys.path.insert(0, str(SCRIPTS))
        import replay_evidence
        record = {"drag": {"q1": [passage("x", "a")], "q2": [passage("x", "a")]},
                  "naive": [passage("x", "a"), passage("y", "b")]}
        self.assertEqual([p["id"] for p in replay_evidence.final_pool(record)], ["x"])
        self.assertEqual([p["id"] for p in replay_evidence.anchored_pool(record)],
                         ["x", "y"])

    def test_cli_prints_the_proxy_caveat_and_writes_json(self):
        code = ("import sys; sys.path.insert(0, %r); import replay_evidence; "
                "sys.exit(replay_evidence.main(sys.argv[1:]))" % str(SCRIPTS))
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            rpath, qpath = self.make_inputs(folder)
            out = folder / "report.json"
            done = subprocess.run([sys.executable, "-I", "-c", code, "--retrieval",
                                   str(rpath), "--questions", str(qpath), "--out", str(out)],
                                  capture_output=True, text=True, timeout=60)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertIn("baseline", done.stdout)
            self.assertIn("anchored", done.stdout)
            self.assertIn("Proxy caveat", done.stdout)
            report = json.loads(out.read_text())
            self.assertEqual(report["anchored"]["recall_count"], 2)


if __name__ == "__main__":
    unittest.main()
