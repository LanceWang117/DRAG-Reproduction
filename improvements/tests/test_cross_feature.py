"""J25: feature combinations run together on the fake backend across three datasets.

Each feature has unit tests on its own; these runs prove the committed combined arms
complete end to end, keep the log schema, and answer every question, for a short-answer,
a yes/no and a multi-hop dataset.
"""
from pathlib import Path
import tempfile
import unittest

import frozen  # noqa: F401  (puts improvements/ and the test helpers on sys.path)
from drag_plus.instrument.timing import LOG_FIELDS, STAGES
from drag_plus.runner.run import run_variant
from drag_plus.runner.variant import load_variant
from test_runner_fake import make_request, read_lines

ARMS_DIR = Path(__file__).resolve().parents[2] / "improvements/configs/arms"
DATASETS = ("NQ", "StrategyQA", "2wiki")
ROWS = [{"dataset": dataset, "split": "dev", "id": f"{dataset}-{index}",
         "question": f"{dataset} question {index}: who wrote the {index}th book?",
         "golden_answers": ["answer 1"]}
        for dataset, count in zip(DATASETS, (2, 2, 1)) for index in range(count)]
COMBINED = ("PC/full-stack", "PC/full-stack-generative", "PC/grounded-vote",
            "E02a/D", "E02b/C", "E02b/E", "E03/B4")


class CrossFeatureMixin:
    """Run helpers only: no test methods, so subclasses never inherit generated tests."""

    def run_arm(self, relative):
        variant = load_variant(ARMS_DIR / f"{relative}.json")
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "run"
            manifest = run_variant(make_request(out, variant=variant, rows=ROWS))
            transcripts = read_lines(out / "transcripts.jsonl")
            timings = read_lines(out / "stage-timings.jsonl")
        return variant, manifest, transcripts, timings

    def check_arm(self, relative):
        variant, manifest, transcripts, timings = self.run_arm(relative)
        self.assertEqual(manifest["exit_status"], "complete")
        self.assertEqual(len(transcripts), len(ROWS) * len(variant.arms))
        for row in transcripts:
            self.assertTrue(row["pred"], (relative, row["id"], row["arm"]))
            self.assertIn("pred_native", row)
            self.assertIn("pred_unified", row)
        self.assertEqual({row["dataset"] for row in transcripts}, set(DATASETS))
        for row in timings:
            self.assertEqual(tuple(row), LOG_FIELDS)
            self.assertIn(row["stage"], STAGES)
        return transcripts, timings


def _make_test(relative):
    def test(self):
        self.check_arm(relative)
    test.__name__ = "test_" + relative.replace("/", "_").replace("-", "_")
    test.__doc__ = f"{relative} completes on NQ, StrategyQA and 2Wiki."
    return test


class CrossFeatureTests(CrossFeatureMixin, unittest.TestCase):
    """One generated test per combined arm (below)."""


for _relative in COMBINED:
    setattr(CrossFeatureTests, _make_test(_relative).__name__, _make_test(_relative))


class CombinedBehaviourTests(CrossFeatureMixin, unittest.TestCase):
    def test_the_log_prob_judge_never_retries_in_the_full_stack(self):
        _, timings = self.check_arm("PC/full-stack")
        stages = {row["stage"] for row in timings if row["arm"] == "drag"}
        self.assertIn("query_sufficiency", stages)
        self.assertNotIn("query_judge_retry", stages)

    def test_the_final_only_moderator_runs_once_per_question(self):
        _, timings = self.check_arm("E02a/D")
        for question in {row["id"] for row in timings}:
            moderator = [row for row in timings if row["id"] == question
                         and row["arm"] == "drag" and row["stage"] == "answer_moderator"]
            self.assertEqual(len(moderator), 1, question)

    def test_the_vote_aggregator_needs_no_moderator(self):
        _, timings = self.check_arm("PC/grounded-vote")
        self.assertFalse([row for row in timings if row["stage"] == "answer_moderator"])


if __name__ == "__main__":
    unittest.main()
