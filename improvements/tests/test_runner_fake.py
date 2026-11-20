"""End to end with the fake backend: logs, schema, checkpoints, crash and resume."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import frozen  # noqa: F401
from drag_plus.instrument.timing import LOG_FIELDS, STAGES
from drag_plus.runner import fake
from drag_plus.runner.backends import build_fake_backend
from drag_plus.runner.resume import ResumeError
from drag_plus.runner.run import RunRequest, run_variant
from drag_plus.runner.variant import parse_variant
from test_instrument import StepClock

ROWS = [{"dataset": "NQ" if i < 3 else "StrategyQA", "split": "dev", "id": f"q{i}",
         "question": f"question number {i}?", "golden_answers": ["answer 1"]}
        for i in range(5)]


class SimulatedKill(BaseException):
    """Stands in for the process dying: not an Exception, so nothing swallows it."""


class DyingGenerator(fake.FakeGenerator):
    def __init__(self, calls_allowed):
        self.left = calls_allowed

    def generate(self, prompts, **params):
        if self.left <= 0:
            raise SimulatedKill()
        self.left -= 1
        return super().generate(prompts, **params)


def make_request(out_dir, **overrides):
    backend = overrides.pop("backend", None) or build_fake_backend()
    variant = overrides.pop("variant", None) or parse_variant({"name": "t"})
    values = dict(rows=ROWS, variant=variant, backend=backend, out_dir=out_dir, profile=True,
                  clock=StepClock(), now=lambda: "2026-01-01T00:00:00Z",
                  environment=lambda: {"commit": "test"}, command=["run_variant.py"])
    values.update(overrides)
    return RunRequest(**values)


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class RunnerCase(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)


class EndToEndTests(RunnerCase):
    def setUp(self):
        super().setUp()
        self.out = self.root / "run"
        self.manifest = run_variant(make_request(self.out))

    def test_all_three_logs_are_written_with_the_full_schema(self):
        timings = read_lines(self.out / "stage-timings.jsonl")
        transcripts = read_lines(self.out / "transcripts.jsonl")
        self.assertEqual(len(transcripts), 10)
        self.assertTrue(timings)
        for row in timings:
            self.assertEqual(tuple(row), LOG_FIELDS)
            self.assertIn(row["stage"], STAGES)
            self.assertEqual(row["run_id"], "run")
        self.assertEqual({row["arm"] for row in timings}, {"drag", "naive"})
        self.assertEqual(self.manifest["exit_status"], "complete")
        for key in ("flags", "backend", "environment", "variant", "resumes", "command"):
            self.assertIn(key, self.manifest)
        self.assertEqual(self.manifest["environment"]["commit"], "test")

    def test_transcripts_hold_prompts_outputs_hashes_and_both_extractions(self):
        for row in read_lines(self.out / "transcripts.jsonl"):
            self.assertIn("pred_native", row)
            self.assertIn("pred_unified", row)
            self.assertTrue(all(len(e["sha256"]) == 64 for e in row["entries"]))
            keys = [entry["key"] for entry in row["entries"]]
            self.assertIn("pred", keys)
            if row["arm"] == "drag":
                self.assertEqual(keys[0].split("_")[0], "QueryStage")
                self.assertEqual(row["pred"], row["pred_native"])

    def test_summary_scores_each_arm_and_dataset(self):
        results = json.loads((self.out / "summary.json").read_text())["results"]
        seen = {(r["arm"], r["dataset"]): r["count"] for r in results}
        self.assertEqual(seen, {("drag", "NQ"): 3, ("drag", "StrategyQA"): 2,
                                ("naive", "NQ"): 3, ("naive", "StrategyQA"): 2})

    def test_profiling_off_writes_no_timing_log_and_the_same_transcripts(self):
        off = self.root / "off"
        run_variant(make_request(off, profile=False))
        self.assertFalse((off / "stage-timings.jsonl").exists())
        self.assertEqual((off / "transcripts.jsonl").read_text(),
                         (self.out / "transcripts.jsonl").read_text().replace('"run_id": "run"', '"run_id": "off"'))

    def test_interleave_alternates_arms(self):
        out = self.root / "inter"
        run_variant(make_request(out, interleave=True))
        arms = [row["arm"] for row in read_lines(out / "transcripts.jsonl")]
        self.assertEqual(arms, ["drag", "naive"] * 5)

    def test_an_existing_run_directory_is_never_overwritten(self):
        with self.assertRaises(FileExistsError):
            run_variant(make_request(self.out))


class CrashResumeTests(RunnerCase):
    LOGS = ("checkpoints.jsonl", "transcripts.jsonl", "stage-timings.jsonl", "summary.json")

    def crash_then_resume(self, calls_allowed, tear):
        out = self.root / "crashed"
        dying = build_fake_backend()
        dying.generator = DyingGenerator(calls_allowed)
        with self.assertRaises(SimulatedKill):
            run_variant(make_request(out, backend=dying))
        done = len(read_lines(out / "checkpoints.jsonl"))
        if tear:
            with open(out / "checkpoints.jsonl", "ab") as handle:
                handle.write(b'{"order": 99, "arm": "dr')
        final = run_variant(make_request(out, resume=True))
        return out, done, final

    def test_resume_gives_the_same_logs_as_an_uninterrupted_run(self):
        # The run directory name is the run_id, so both runs use the name "crashed".
        reference = self.root / "ref" / "crashed"
        reference.parent.mkdir()
        run_variant(make_request(reference))
        for tear in (False, True):
            with self.subTest(tear=tear):
                out, done, final = self.crash_then_resume(40, tear)
                self.assertTrue(0 < done < 10, done)
                for name in self.LOGS:
                    self.assertEqual((out / name).read_text(), (reference / name).read_text(), name)
                self.assertEqual(final["exit_status"], "complete")
                self.assertEqual(final["resumes"][0]["completed_before"], done)
                self.assertEqual(final["resumes"][0]["repaired_bytes"] > 0, tear)
                shutil.rmtree(out)

    def test_a_killed_run_is_marked_failed_and_keeps_its_checkpoints(self):
        out = self.root / "crashed"
        dying = build_fake_backend()
        dying.generator = DyingGenerator(5)
        with self.assertRaises(SimulatedKill):
            run_variant(make_request(out, backend=dying))
        manifest = json.loads((out / "run-manifest.json").read_text())
        self.assertEqual(manifest["exit_status"], "failed")
        self.assertFalse((out / "summary.json").exists())

    def test_a_keyboard_interrupt_is_marked_interrupted(self):
        out = self.root / "ctrl-c"
        backend = build_fake_backend()
        backend.generator = type("Stop", (), {"generate": lambda s, p, **k: (_ for _ in ()).throw(
            KeyboardInterrupt())})()
        with self.assertRaises(KeyboardInterrupt):
            run_variant(make_request(out, backend=backend))
        manifest = json.loads((out / "run-manifest.json").read_text())
        self.assertEqual(manifest["exit_status"], "interrupted")

    def test_resuming_with_a_different_schedule_is_refused(self):
        out, _, _ = self.crash_then_resume(40, False)
        with self.assertRaises(ResumeError):
            run_variant(make_request(out, resume=True, interleave=True))

    def test_resuming_with_a_different_variant_is_refused(self):
        out, _, _ = self.crash_then_resume(40, False)
        other = parse_variant({"name": "t", "flags": {"max_query_debate_rounds": 1}})
        with self.assertRaisesRegex(ResumeError, "variant"):
            run_variant(make_request(out, resume=True, variant=other))

    def test_resuming_under_different_settings_is_refused(self):
        out, _, _ = self.crash_then_resume(40, False)
        cases = {"profile": dict(profile=False), "questions": dict(rows=ROWS[:3])}
        for name, change in cases.items():
            with self.subTest(change=name), self.assertRaisesRegex(ResumeError, "differs"):
                run_variant(make_request(out, resume=True, **change))
        other = build_fake_backend()
        other.name = "flashrag"
        with self.assertRaisesRegex(ResumeError, "backend"):
            run_variant(make_request(out, resume=True, backend=other))

    def test_a_failing_evaluator_still_finalises_the_manifest_and_logs(self):
        out = self.root / "eval-fails"
        backend = build_fake_backend()
        backend.evaluate = lambda arm, dataset, records: 1 / 0
        with self.assertRaises(ZeroDivisionError):
            run_variant(make_request(out, backend=backend))
        manifest = json.loads((out / "run-manifest.json").read_text())
        self.assertEqual((manifest["exit_status"], manifest["ended"] is not None),
                         ("failed", True))
        self.assertTrue((out / "transcripts.jsonl").exists())
        self.assertFalse((out / "summary.json").exists())
        final = run_variant(make_request(out, resume=True))
        self.assertEqual(final["exit_status"], "complete")

    def test_an_empty_existing_directory_is_accepted_for_a_fresh_run(self):
        out = self.root / "preexisting"
        (out / "drag").mkdir(parents=True)  # FlashRAG's Config creates its save_dir first
        self.assertEqual(run_variant(make_request(out))["exit_status"], "complete")

    def test_resuming_a_missing_directory_is_refused(self):
        with self.assertRaises(ResumeError):
            run_variant(make_request(self.root / "nowhere", resume=True))

    def test_an_ordinary_error_fails_fast_and_is_retried_on_resume(self):
        out = self.root / "failing"
        broken = build_fake_backend()
        broken.generator = type("Broken", (), {"generate": lambda s, p, **k: 1 / 0})()
        with self.assertRaises(ZeroDivisionError):
            run_variant(make_request(out, backend=broken))
        self.assertEqual(json.loads((out / "run-manifest.json").read_text())["exit_status"], "failed")
        final = run_variant(make_request(out, resume=True))
        self.assertEqual(final["exit_status"], "complete")


if __name__ == "__main__":
    unittest.main()
