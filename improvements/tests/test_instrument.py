"""Timing rows, durable JSONL writers and the manifest builder."""
import importlib.metadata
from pathlib import Path
import subprocess
import tempfile
import unittest

import frozen  # noqa: F401  (puts improvements/ and the test helpers on sys.path)
from drag_plus.flags import Flags
from drag_plus.instrument import logs, manifest
from drag_plus.instrument.timing import ROW_FIELDS, STAGES, CallRecorder
from drag_plus.pipeline import DragPlusPipeline
from fakes import FakeRetriever, IdentityTemplate, RecordingItem, ScriptedGenerator
from t0_scenarios import answer_rounds, query_round


class StepClock:
    """Deterministic clock: every read advances one millisecond."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 0.001
        return self.now


def run_pipeline(profile):
    outputs = (query_round("Opponent", "weak. Query Expansion: second query")
               + query_round("Unclear", retry="Proponent") + answer_rounds(1))
    gen, ret = ScriptedGenerator(outputs), FakeRetriever()
    flags = Flags({"profile_calls": profile, "max_answer_debate_rounds": 1})
    pipeline = DragPlusPipeline(flags, gen, ret, IdentityTemplate(), "NQ",
                                recorder=CallRecorder(StepClock()))
    item = RecordingItem("who wrote it?")
    return pipeline.run_item(item), gen, ret, item


class TimingRowTests(unittest.TestCase):
    def test_one_row_per_generator_and_retriever_call_with_the_full_schema(self):
        state, gen, ret, _ = run_pipeline(True)
        self.assertEqual(len(state.calls), len(gen.calls) + len(ret.calls))
        for row in state.calls:
            self.assertEqual(tuple(row), ROW_FIELDS)
            self.assertIn(row["stage"], STAGES)
            self.assertAlmostEqual(row["elapsed_ms"], 1.0)
        stages = [row["stage"] for row in state.calls]
        for expected in ("retrieval", "query_agent", "query_judge", "query_judge_retry",
                         "answer_agent", "answer_moderator"):
            self.assertIn(expected, stages)

    def test_retrieval_rows_have_no_token_counts_and_generation_rows_do(self):
        state = run_pipeline(True)[0]
        for row in state.calls:
            has_tokens = row["stage"] != "retrieval"
            self.assertEqual(row["prompt_tokens"] is not None, has_tokens)
            self.assertEqual(row["output_tokens"] is not None, has_tokens)

    def test_profiling_off_records_nothing_and_changes_no_call(self):
        off, gen_off, ret_off, item_off = run_pipeline(False)
        on, gen_on, ret_on, item_on = run_pipeline(True)
        self.assertEqual(off.calls, [])
        self.assertEqual(gen_off.calls, gen_on.calls)
        self.assertEqual(ret_off.calls, ret_on.calls)
        self.assertEqual(item_off.calls, item_on.calls)

    def test_rounds_are_recorded(self):
        state = run_pipeline(True)[0]
        query_rounds = {r["round"] for r in state.calls if r["stage"] == "query_agent"}
        self.assertEqual(query_rounds, {0, 1})


class LogTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "log.jsonl"

    def test_append_and_read_round_trip(self):
        logs.append_line(self.path, {"a": 1})
        logs.append_line(self.path, {"b": "é"})
        self.assertEqual(logs.read_lines(self.path), [{"a": 1}, {"b": "é"}])

    def test_a_torn_last_line_is_ignored_and_repaired(self):
        logs.append_line(self.path, {"a": 1})
        with open(self.path, "ab") as handle:
            handle.write(b'{"b": 2')
        self.assertEqual(logs.read_lines(self.path), [{"a": 1}])
        self.assertEqual(logs.repair_tail(self.path), len(b'{"b": 2'))
        logs.append_line(self.path, {"c": 3})
        self.assertEqual(logs.read_lines(self.path), [{"a": 1}, {"c": 3}])
        self.assertEqual(logs.repair_tail(self.path), 0)

    def test_missing_file_is_an_empty_log(self):
        self.assertEqual(logs.read_lines(self.path), [])
        self.assertEqual(logs.repair_tail(self.path), 0)

    def test_atomic_write_replaces_without_leaving_a_temp_file(self):
        logs.write_jsonl_atomic(self.path, [{"x": 1}])
        logs.write_jsonl_atomic(self.path, [{"x": 2}, {"x": 3}])
        self.assertEqual(logs.read_lines(self.path), [{"x": 2}, {"x": 3}])
        self.assertEqual([p.name for p in self.path.parent.iterdir()], ["log.jsonl"])


class ManifestTests(unittest.TestCase):
    def test_environment_uses_only_injected_sources(self):
        def fake_run(args, **kwargs):
            out = {"rev-parse": "abc123\n", "status": " M file\n"}[args[1]]
            return subprocess.CompletedProcess(args, 0, stdout=out)

        def version_of(name):
            if name == "vllm":
                return "0.6.6"
            raise importlib.metadata.PackageNotFoundError(name)

        env = manifest.collect_environment(fake_run, {"OMP_NUM_THREADS": "8"}, version_of)
        self.assertEqual((env["commit"], env["dirty"]), ("abc123", True))
        self.assertEqual(env["packages"]["vllm"], "0.6.6")
        self.assertIsNone(env["packages"]["torch"])
        self.assertEqual(env["threads"], {"OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": None,
                                          "TOKENIZERS_PARALLELISM": None})

    def test_environment_survives_a_missing_git(self):
        def broken(args, **kwargs):
            raise FileNotFoundError("git")
        env = manifest.collect_environment(broken, {}, lambda n: "1")
        self.assertEqual((env["commit"], env["dirty"]), (None, None))

    def test_manifest_carries_every_required_field(self):
        built = manifest.build_manifest(
            run_id="r", variant={"name": "v"}, flags=Flags().to_manifest(),
            backend={"gpu": "none"}, environment={"commit": "c"}, started="s", ended=None,
            exit_status="running", resumes=[], command=["x"], arms=["drag"], questions=1)
        for key in ("run_id", "variant", "flags", "backend", "environment",
                    "started", "ended", "exit_status", "resumes", "command", "arms"):
            self.assertIn(key, built)


if __name__ == "__main__":
    unittest.main()
