"""Rehearsal report, session resume and script, all against fakes (no server, no GPU)."""
import contextlib
import dataclasses
import io
from pathlib import Path
import tempfile
import unittest

import frozen  # noqa: F401
import rehearse_local
from drag_plus.rehearsal.plan import ArmPlan
from drag_plus.rehearsal.report import arm_metrics, differences, variant_report
from drag_plus.rehearsal.report_text import format_report
from drag_plus.rehearsal.session import estimate_minutes, run_rehearsal
from drag_plus.runner.backends import build_fake_backend
from drag_plus.runner.local_generator import DEFAULT_MODEL
from drag_plus.runner.local_smoke import SmokeJob
from drag_plus.runner.variant import parse_variant
from test_local_smoke import GOOD, StubServer
from test_runner_fake import ROWS, DyingGenerator, SimulatedKill

BASELINE = parse_variant({"name": "baseline"})
SHORT = parse_variant({"name": "short", "flags": {"max_query_debate_rounds": 1,
                                                  "max_answer_debate_rounds": 1}})
CONFIGS = frozen.PROJECT / "improvements" / "configs"


def row(stage, role="proponent", rnd=0, qid="q1", prompt=100, out=10, finish="stop"):
    return {"stage": stage, "role": role, "round": rnd, "dataset": "NQ", "id": qid,
            "prompt_tokens": prompt, "output_tokens": out, "finish_reason": finish,
            "elapsed_ms": 1000}


class ArmMetricsTests(unittest.TestCase):
    def test_counts_rounds_tokens_and_statuses(self):
        rows = [row("retrieval", "retriever"), row("query_agent"), row("query_judge", "judge"),
                row("answer_agent", rnd=0), row("answer_agent", rnd=1, prompt=300, finish="length"),
                row("answer_moderator", "moderator", rnd=1, prompt=200)]
        transcripts = [{"entries": [{"key": "QueryStage_Moderator_Round0_DecisionStatus",
                                     "value": "valid_initial"}]}]
        m = arm_metrics(rows, transcripts, questions=1)
        self.assertEqual(m["llm_calls_per_question"], 5)
        self.assertEqual(m["retrieval_calls_per_question"], 1)
        self.assertEqual((m["query_rounds"], m["answer_rounds"]), (1, 2))
        self.assertEqual(m["prompt_tokens_median"], 100)
        self.assertEqual(m["prompt_tokens_p90"], 300)
        self.assertEqual(m["finish_reasons"], {"stop": 4, "length": 1})
        self.assertEqual(m["judge_statuses"], {"valid_initial": 1})

    def test_naive_calls_do_not_count_as_rounds(self):
        m = arm_metrics([row("retrieval", "retriever"), row("answer_agent", "naive")], [], 1)
        self.assertEqual((m["query_rounds"], m["answer_rounds"]), (0, 0))

    def test_empty_arm(self):
        m = arm_metrics([], [], 3)
        self.assertEqual(m["llm_calls_per_question"], 0)
        self.assertIsNone(m["prompt_tokens_median"])


class SessionCase(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def rehearse(self, root, backend=build_fake_backend, variants=(BASELINE, SHORT)):
        job = SmokeJob("tiny:1b", ROWS, list(variants), root, Path("r"), Path("q"))
        return run_rehearsal(job, StubServer(), GOOD, lambda job_: backend(), progress=lambda m: None)


def structural(reports):
    keep = {}
    for report in reports:
        keep[report["variant"]] = {arm: {k: v for k, v in m.items() if k != "local_seconds"}
                                   for arm, m in report["arms"].items()}
    return keep


class SessionTests(SessionCase):
    def test_fewer_rounds_show_up_as_negative_differences(self):
        baseline, short = self.rehearse(self.root / "a")
        self.assertTrue(baseline["ok"] and short["ok"])
        self.assertEqual(short["arms"]["drag"]["answer_rounds"], 1)
        self.assertEqual(baseline["arms"]["drag"]["answer_rounds"], 3)
        self.assertEqual(baseline["arms"]["naive"]["llm_calls_per_question"], 1)
        delta = differences(baseline, short)["drag"]
        self.assertLess(delta["llm_calls_per_question"], 0)
        self.assertLess(delta["answer_rounds"], 0)

    def test_resume_after_a_crash_matches_an_uninterrupted_run(self):
        expected = structural(self.rehearse(self.root / "clean"))
        crashed = self.root / "crashed"
        dying = lambda: dataclasses.replace(build_fake_backend(), generator=DyingGenerator(40))
        with self.assertRaises(SimulatedKill):
            self.rehearse(crashed, backend=dying)
        self.assertEqual(structural(self.rehearse(crashed)), expected)

    def test_finished_variants_are_not_rerun_or_touched(self):
        self.rehearse(self.root / "done")
        manifests = sorted((self.root / "done").glob("runs/*/run-manifest.json"))
        before = [path.read_text() for path in manifests]
        never = lambda: dataclasses.replace(build_fake_backend(), generator=DyingGenerator(0))
        reports = self.rehearse(self.root / "done", backend=never)
        self.assertTrue(all(report["ok"] for report in reports))
        self.assertEqual([path.read_text() for path in manifests], before)

    def test_a_changed_finished_variant_fails_alone(self):
        self.rehearse(self.root / "edit")
        edited = parse_variant({"name": "short", "flags": {"max_query_debate_rounds": 2}})
        baseline, short = self.rehearse(self.root / "edit", variants=(BASELINE, edited))
        self.assertTrue(baseline["ok"])
        self.assertFalse(short["ok"])
        self.assertIn("differs", short["problems"][0])

    def test_a_finished_run_from_another_model_is_not_reused(self):
        self.rehearse(self.root / "model", variants=(SHORT,))
        other = lambda: dataclasses.replace(
            build_fake_backend(), describe=lambda: {"model": "ollama:other"})
        (report,) = self.rehearse(self.root / "model", backend=other, variants=(SHORT,))
        self.assertFalse(report["ok"])
        self.assertIn("ollama:other", report["problems"][0])

    def test_a_corrupt_log_is_reported_not_raised(self):
        (report,) = self.rehearse(self.root / "corrupt", variants=(SHORT,))
        log = Path(report["run_dir"]) / "transcripts.jsonl"
        log.write_text(log.read_text() + "{not json}\n")
        damaged = variant_report(Path(report["run_dir"]))
        self.assertFalse(damaged["ok"])
        self.assertGreater(damaged["schema_errors"], 0)

    def test_estimate_scales_with_rounds(self):
        self.assertGreater(estimate_minutes([BASELINE], 12), estimate_minutes([SHORT], 12))


class ScriptTests(SessionCase):
    def server(self, host, log):
        if any("missing" in arg or "bad.json" in arg or "absent" in arg for arg in self.args):
            raise AssertionError("a refused request must not touch the server")
        return StubServer(models=(DEFAULT_MODEL,))

    def run_main(self, *args):
        self.args = args
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = rehearse_local.main(
                ["--resume", str(self.root), *args], hardware=lambda root: GOOD,
                server_factory=self.server,
                backend_factory=lambda server: (lambda job: build_fake_backend()))
        return code, out.getvalue(), err.getvalue()

    def test_runs_local_arms_and_skips_server_only_ones(self):
        code, out, _ = self.run_main("--questions", "6", "--variant", str(CONFIGS / "local-smoke.json"),
                                     str(CONFIGS / "hybrid.json"))
        self.assertEqual(code, 0)
        self.assertIn("NOT A RESULT", out)
        self.assertIn("RETRIEVAL CAVEAT", out)
        self.assertIn("[PASS] local-smoke: completed 12/12", out)
        self.assertIn("[SKIP] hybrid: retriever='hybrid'", out)
        self.assertIn("Overall: PASS", out)

    def test_only_server_arms_means_fail(self):
        code, out, _ = self.run_main("--questions", "6", "--variant", str(CONFIGS / "hybrid.json"))
        self.assertEqual(code, 1)
        self.assertIn("Overall: FAIL", out)

    def test_test_set_is_refused(self):
        test_set = frozen.PROJECT / "experiments" / "J05-testset" / "questions-test.jsonl"
        code, _, err = self.run_main("--manifest", str(test_set), "--variant",
                                     str(CONFIGS / "baseline.json"))
        self.assertEqual(code, 2)
        self.assertIn("test set", err)

    def test_unrun_arms_are_listed(self):
        plans = [ArmPlan("a", True, (), ()), ArmPlan("b", True, (), ())]
        report = {"variant": "a", "ok": False, "completed": 0, "expected": 2,
                  "parse_failures": 0, "empty_outputs": 0, "schema_errors": 0,
                  "problems": ["timeout"], "run_dir": "x"}
        text = format_report([report], plans, "ollama:x", 6)
        self.assertIn("[NOT RUN] b", text)
        self.assertIn("Overall: FAIL", text)

    def test_finish_reasons_fall_back_to_generator_stats(self):
        plans = [ArmPlan("a", True, (), ())]
        metrics = arm_metrics([row("answer_agent", finish=None)], [], 1)
        report = {"variant": "a", "ok": True, "completed": 1, "expected": 1, "parse_failures": 0,
                  "empty_outputs": 0, "schema_errors": 0, "problems": [], "run_dir": "x",
                  "arms": {"drag": metrics}, "generator": {"finish_reasons": {"stop": 1}}}
        text = format_report([report], plans, "ollama:x", 6)
        self.assertIn("finish not in timing rows", text)
        self.assertIn("generator finish reasons (all arms): {'stop': 1}", text)

    def test_future_server_only_flag_is_skipped_cleanly(self):
        path = self.root / "gpu.json"
        path.write_text('{"name": "gpu", "flags": {"faiss_gpu": true}}')
        code, out, _ = self.run_main("--questions", "6", "--variant", str(path),
                                     str(CONFIGS / "local-smoke.json"))
        self.assertEqual(code, 0)
        self.assertIn("[SKIP] gpu: faiss_gpu=True", out)

    def test_refusals_happen_before_any_server(self):
        bad_json = self.root / "bad.json"
        bad_json.write_text("{not json")
        cases = (["--replay", str(self.root / "missing.jsonl")], ["--variant", str(bad_json)],
                 ["--variant", str(self.root / "absent.json")])
        for extra in cases:
            args = (["--variant", str(CONFIGS / "baseline.json")] if extra[0] == "--replay" else []) + extra
            code, _, err = self.run_main(*args)
            self.assertEqual(code, 2, extra)
            self.assertIn("refused", err)

    def test_resume_of_a_missing_directory_is_refused(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = rehearse_local.main(["--resume", str(self.root / "nope"), "--variant",
                                        str(CONFIGS / "baseline.json")])
        self.assertEqual(code, 2)
        self.assertIn("does not exist", err.getvalue())

    def test_bad_question_count_is_refused(self):
        code, _, err = self.run_main("--questions", "7", "--variant", str(CONFIGS / "baseline.json"))
        self.assertEqual(code, 2)
        self.assertIn("multiple of 6", err)


if __name__ == "__main__":
    unittest.main()
