"""The smoke summary reads a finished run's logs: schema errors, parse failures, verdict."""
import json
from pathlib import Path
import tempfile
import unittest

import frozen  # noqa: F401
from drag_plus.runner import smoke_summary as ss
from drag_plus.runner.backends import build_fake_backend
from drag_plus.runner.run import run_variant
from test_runner_fake import ROWS, make_request


def finished_run(folder, rows=ROWS[:2], profile=True):
    out_dir = Path(folder) / "run"
    run_variant(make_request(out_dir, backend=build_fake_backend(), rows=rows, profile=profile))
    return out_dir


class SummaryTests(unittest.TestCase):
    def test_a_clean_run_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            result = ss.summarise_run(finished_run(folder))
        self.assertEqual((result["expected"], result["completed"], result["schema_errors"]), (4, 4, 0))
        self.assertTrue(result["ok"], result["problems"])

    def edit(self, out_dir, name, change):
        """Rewrite every row of a log with ``change(index, row)``."""
        path = out_dir / name
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        path.write_text("\n".join(json.dumps(change(i, row)) for i, row in enumerate(rows)) + "\n")

    def test_a_transcript_row_missing_a_field_is_a_schema_error(self):
        with tempfile.TemporaryDirectory() as folder:
            out_dir = finished_run(folder)
            self.edit(out_dir, "transcripts.jsonl",
                      lambda i, row: {k: v for k, v in row.items() if k != "pred_unified"})
            result = ss.summarise_run(out_dir)
        self.assertEqual(result["schema_errors"], 4)
        self.assertFalse(result["ok"])

    def test_a_timing_row_missing_a_field_is_a_schema_error(self):
        with tempfile.TemporaryDirectory() as folder:
            out_dir = finished_run(folder)
            self.edit(out_dir, "stage-timings.jsonl",
                      lambda i, row: {k: v for k, v in row.items() if k != "elapsed_ms"})
            self.assertGreater(ss.summarise_run(out_dir)["schema_errors"], 0)

    def test_a_missing_timing_log_in_a_profiled_run_is_a_schema_error(self):
        with tempfile.TemporaryDirectory() as folder:
            out_dir = finished_run(folder)
            (out_dir / "stage-timings.jsonl").unlink()
            self.assertEqual(ss.summarise_run(out_dir)["schema_errors"], 1)

    def test_answers_without_the_marker_are_parse_failures_and_all_of_them_fail_the_run(self):
        with tempfile.TemporaryDirectory() as folder:
            out_dir = finished_run(folder)
            self.edit(out_dir, "checkpoints.jsonl",
                      lambda i, row: dict(row, raw_pred="no marker here"))
            result = ss.summarise_run(out_dir)
        self.assertEqual(result["parse_failures"], 4)
        self.assertIn("no final answer carried", result["problems"][0])

    def test_some_parse_failures_only_warn(self):
        with tempfile.TemporaryDirectory() as folder:
            out_dir = finished_run(folder)
            self.edit(out_dir, "checkpoints.jsonl", lambda i, row: dict(
                row, raw_pred="no marker" if i == 0 else row["raw_pred"]))
            result = ss.summarise_run(out_dir)
        self.assertEqual(result["parse_failures"], 1)
        self.assertTrue(result["ok"], result["problems"])

    def test_a_failed_run_is_reported_with_its_status(self):
        with tempfile.TemporaryDirectory() as folder:
            out_dir = finished_run(folder)
            path = out_dir / "run-manifest.json"
            path.write_text(json.dumps(dict(json.loads(path.read_text()), exit_status="failed")))
            self.assertIn("run failed", ss.summarise_run(out_dir)["problems"])

    def test_the_text_summary_says_it_is_not_a_result(self):
        with tempfile.TemporaryDirectory() as folder:
            text = ss.format_summary([ss.summarise_run(finished_run(folder))], "ollama:tiny:1b", 2)
        self.assertIn("NOT A RESULT", text)
        self.assertIn("[PASS] t: completed 4/4", text)
        self.assertNotIn("accuracy:", text)


if __name__ == "__main__":
    unittest.main()
