"""Retrieval-only metrics, the stored-evidence regression and the CLI."""
import contextlib
import io
import json
from math import log2
from pathlib import Path
import sys
import tempfile
import unittest

IMPROVEMENTS = Path(__file__).resolve().parents[1]
PROJECT = IMPROVEMENTS.parent
sys.path.insert(0, str(IMPROVEMENTS))
sys.path.insert(0, str(PROJECT / "scripts"))
from drag_plus.analysis.retrieval import (containment_at_k, contains_gold, evaluate,  # noqa: E402
                                          ndcg_at_k, percentile, reciprocal_rank,
                                          summarise, unique_title_ratio)
import eval_retrieval  # noqa: E402

RETRIEVAL = PROJECT / "results" / "retrieval-200.jsonl"
QUESTIONS = PROJECT / "results" / "questions-200.jsonl"


def passage(doc_id, title="T", body=""):
    return {"id": doc_id, "contents": f"{title}\n{body or doc_id}"}


class MetricTests(unittest.TestCase):
    def test_containment_sees_the_question_and_the_top_k_passages(self):
        question = "what is the capital of france"
        self.assertEqual(containment_at_k(question, [passage("a", body="Paris")],
                                          ["Paris"], 3), 1.0)
        self.assertEqual(containment_at_k("who wrote it", [passage("a")], ["Paris"], 3), 0.0)
        self.assertEqual(containment_at_k("the answer is Paris", [], ["Paris"], 3), 1.0)

    def test_containment_normalises_gold_and_passage_like_j01(self):
        self.assertEqual(containment_at_k("q", [passage("a", body="The Paris, France.")],
                                          ["paris france"], 1), 1.0)

    def test_containment_at_k_truncates(self):
        passages = [passage("a"), passage("b", body="Paris")]
        self.assertEqual(containment_at_k("q", passages, ["Paris"], 1), 0.0)
        self.assertEqual(containment_at_k("q", passages, ["Paris"], 2), 1.0)

    def test_reciprocal_rank_is_over_the_first_relevant_passage(self):
        self.assertEqual(reciprocal_rank([False, False, True]), 1 / 3)
        self.assertEqual(reciprocal_rank([False, False]), 0.0)
        self.assertEqual(reciprocal_rank([True, True]), 1.0)

    def test_ndcg_at_k_uses_binary_gains(self):
        ideal = 1 / log2(2) + 1 / log2(3)
        self.assertAlmostEqual(ndcg_at_k([True, True], 2), 1.0)
        self.assertAlmostEqual(ndcg_at_k([True, False, True], 3),
                               (1 / log2(2) + 1 / log2(4)) / ideal)
        self.assertAlmostEqual(ndcg_at_k([False, True, True], 3),
                               (1 / log2(3) + 1 / log2(4)) / ideal)
        self.assertEqual(ndcg_at_k([False, False], 5), 0.0)

    def test_unique_title_ratio_counts_distinct_articles(self):
        self.assertEqual(unique_title_ratio([passage("a", "One"), passage("b", "One")]), 0.5)
        self.assertEqual(unique_title_ratio([]), 0.0)

    def test_summarise_reports_means_with_containment_counts(self):
        rows = [evaluate("q", ["Paris"], [passage("a", body="Paris")]),
                evaluate("q", ["Oslo"], [passage("b")])]
        summary = summarise(rows)
        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["containment"], 0.5)
        self.assertEqual(summary["containment_count"], 1)
        self.assertEqual(summarise([]), {"n": 0})

    def test_percentile_is_nearest_rank_and_needs_samples(self):
        values = [5.0, 1.0, 3.0, 2.0, 4.0]
        self.assertEqual(percentile(values, 0.5), 3.0)
        self.assertEqual(percentile(values, 0.99), 5.0)
        with self.assertRaisesRegex(ValueError, "empty"):
            percentile([], 0.5)


class StoredEvidenceTests(unittest.TestCase):
    """B17's 114–115/200, recomputed from the committed passages (read-only)."""

    @classmethod
    def setUpClass(cls):
        cls.questions = eval_retrieval.load_questions(QUESTIONS)
        cls.records = eval_retrieval.read_jsonl(RETRIEVAL)

    def test_each_arm_contains_the_gold_for_114_of_200_questions(self):
        report = eval_retrieval.from_stored(RETRIEVAL, QUESTIONS, ["drag", "naive"], [3], 5)
        for arm in ("drag", "naive"):
            self.assertEqual(report[arm]["n"], 200)
            self.assertEqual(report[arm]["containment_count"], 114)
        self.assertEqual(report["naive"]["containment@3_count"], 114)

    def test_the_union_of_the_arms_passages_is_115_of_200(self):
        """The upper end of the recorded 114–115 range: passages alone, both arms."""
        count = 0
        for record in self.records:
            golds = self.questions[record["id"]]["golden_answers"]
            pool = (eval_retrieval.stored_candidates(record, "naive")
                    + eval_retrieval.stored_candidates(record, "drag"))
            count += any(contains_gold(str(passage.get("contents", "")), golds)
                         for passage in pool)
        self.assertEqual(int(count), 115)

    def test_passages_alone_contain_the_gold_for_105_of_200(self):
        count = sum(containment_at_k("", eval_retrieval.stored_candidates(record, arm),
                                    self.questions[record["id"]]["golden_answers"],
                                    len(eval_retrieval.stored_candidates(record, arm)))
                    for record in self.records for arm in ("naive", "drag"))
        self.assertEqual(int(count), 210)


class CliTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def run_main(self, argv):
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            code = eval_retrieval.main(argv)
        self.assertEqual(code, 0)
        return captured.getvalue()

    def test_the_stored_dump_writes_a_full_report(self):
        out = self.directory / "report.json"
        printed = self.run_main(["--retrieval", str(RETRIEVAL), "--out", str(out)])
        self.assertIn("naive", printed)
        report = json.loads(out.read_text())
        self.assertEqual(report["naive"]["containment_count"], 114)
        self.assertEqual(report["drag"]["containment_count"], 114)
        self.assertIn("ndcg@5", report["naive"])

    def test_candidate_rows_are_scored_without_any_retrieval(self):
        path = self.directory / "candidates.jsonl"
        rows = [{"id": "q1", "question": "capital?", "golden_answers": ["Paris"],
                 "candidates": [passage("a", body="Paris")]},
                {"id": "q2", "question": "river?", "golden_answers": ["Nile"],
                 "candidates": [passage("b")]}]
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        out = self.directory / "candidates-report.json"
        self.run_main(["--candidates", str(path), "--out", str(out), "--ks", "1,2"])
        report = json.loads(out.read_text())["candidates"]
        self.assertEqual(report["n"], 2)
        self.assertEqual(report["containment_count"], 1)
        self.assertEqual(report["containment@1_count"], 1)
        self.assertEqual(report["mrr"], 0.5)

    def test_timings_report_retrieval_percentiles_only(self):
        path = self.directory / "stage-timings.jsonl"
        rows = [{"stage": "retrieval", "variant": "v", "arm": "drag", "dataset": "NQ",
                 "elapsed_ms": value} for value in (10, 20, 30)]
        rows.append({"stage": "query_agent", "variant": "v", "arm": "drag",
                     "dataset": "NQ", "elapsed_ms": 999})
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        out = self.directory / "timing-report.json"
        self.run_main(["--retrieval", str(RETRIEVAL), "--timings", str(path),
                       "--out", str(out)])
        latency = json.loads(out.read_text())["latency"]
        self.assertEqual(latency["v / drag / NQ"], {"count": 3, "p50_ms": 20.0,
                                                    "p99_ms": 30.0})

    def assert_rejected(self, argv):
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                eval_retrieval.main(argv)

    def test_bad_arguments_are_rejected(self):
        self.assert_rejected(["--retrieval", str(RETRIEVAL), "--ks", "0"])
        self.assert_rejected(["--candidates", str(RETRIEVAL), "--timings", str(RETRIEVAL)])
        self.assert_rejected([])


if __name__ == "__main__":
    unittest.main()
