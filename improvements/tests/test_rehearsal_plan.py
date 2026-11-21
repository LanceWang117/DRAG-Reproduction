"""Rehearsal planning: dev-only selection, server-only arms and guards."""
import json
from pathlib import Path
import unittest

import frozen  # noqa: F401
from drag_plus.rehearsal.plan import (MAX_QUESTIONS, RehearsalError, check_manifest,
                                      classify, plan_arms, select_rows)

PROJECT = Path(__file__).resolve().parents[2]
DEV = PROJECT / "results" / "questions-200.jsonl"
DEV_ROWS = [json.loads(line) for line in DEV.read_text().splitlines() if line.strip()]


def e00_subset(rows, per_dataset=10):
    taken, chosen = {}, []
    for row in rows:
        if taken.get(row["dataset"], 0) < per_dataset:
            taken[row["dataset"]] = taken.get(row["dataset"], 0) + 1
            chosen.append(row["id"])
    return chosen


class SelectionTests(unittest.TestCase):
    def test_equal_share_per_dataset_in_manifest_order(self):
        rows = select_rows(DEV_ROWS, 12)
        self.assertEqual(len(rows), 12)
        counts = {}
        for row in rows:
            counts[row["dataset"]] = counts.get(row["dataset"], 0) + 1
        self.assertEqual(set(counts.values()), {2})
        positions = [DEV_ROWS.index(row) for row in rows]
        self.assertEqual(positions, sorted(positions))

    def test_selection_is_a_prefix_of_the_e00_subset(self):
        subset = set(e00_subset(DEV_ROWS))
        for questions in (6, 18, MAX_QUESTIONS):
            self.assertTrue({row["id"] for row in select_rows(DEV_ROWS, questions)} <= subset)

    def test_deterministic(self):
        self.assertEqual(select_rows(DEV_ROWS, 18), select_rows(DEV_ROWS, 18))

    def test_invalid_counts_are_refused(self):
        for questions in (0, 5, 7, 36):
            with self.assertRaises(RehearsalError):
                select_rows(DEV_ROWS, questions)

    def test_a_short_dataset_is_reported(self):
        rows = [{"dataset": "A", "id": str(i)} for i in range(5)] + [{"dataset": "B", "id": "b"}]
        with self.assertRaisesRegex(RehearsalError, "B"):
            select_rows(rows, 6)


class ManifestGuardTests(unittest.TestCase):
    def test_dev_manifest_is_accepted(self):
        check_manifest(DEV, DEV)

    def test_test_set_is_refused(self):
        test_set = PROJECT / "experiments" / "J05-testset" / "questions-test.jsonl"
        with self.assertRaisesRegex(RehearsalError, "test set"):
            check_manifest(test_set, DEV)

    def test_any_other_manifest_is_refused(self):
        with self.assertRaisesRegex(RehearsalError, "dev manifest"):
            check_manifest(PROJECT / "other.jsonl", DEV)


class ArmTests(unittest.TestCase):
    def test_server_only_flags_skip_the_arm(self):
        cases = {"retriever": "hybrid", "bm25_index": "idx", "rerank_k": 20, "faiss_gpu": True,
                 "enable_prefix_caching": True, "quantization": "w4a16", "speculative": "ngram"}
        for flag, value in cases.items():
            plan = classify("v", {"flags": {flag: value}})
            self.assertFalse(plan.runnable, flag)
            self.assertIn(flag, plan.reasons[0])

    def test_inactive_values_do_not_skip(self):
        for flags in ({"retriever": "e5"}, {"faiss_gpu": False}, {"quantization": "none"},
                      {"rerank_k": None}):
            self.assertTrue(classify("v", {"flags": flags}).runnable, flags)

    def test_server_tuning_runs_with_a_note(self):
        plan = classify("v", {"flags": {"gpu_memory_utilization": 0.45, "omp_threads": 16,
                                        "question_batch_size": 8}})
        self.assertTrue(plan.runnable)
        self.assertEqual(len(plan.notes), 3)

    def test_real_variant_files_classify(self):
        configs = PROJECT / "improvements" / "configs"
        plans = {p.name: p for p in plan_arms(
            [json.loads((configs / f).read_text())
             for f in ("baseline.json", "local-smoke.json", "hybrid.json")])}
        self.assertTrue(plans["baseline"].runnable)
        self.assertFalse(plans["hybrid"].runnable)

    def test_flags_unknown_to_the_registry_still_classify(self):
        # faiss_gpu is added by a later job; planning works on the raw document.
        plan = plan_arms([{"name": "gpu", "flags": {"faiss_gpu": True}}])[0]
        self.assertFalse(plan.runnable)

    def test_duplicate_or_missing_names_are_refused(self):
        with self.assertRaisesRegex(RehearsalError, "same"):
            plan_arms([{"name": "same"}, {"name": "same"}])
        with self.assertRaisesRegex(RehearsalError, "name"):
            plan_arms([{"flags": {}}])


if __name__ == "__main__":
    unittest.main()
