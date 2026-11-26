"""J26: the server-check registry, the automatic probes against fakes, and the CLI."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import frozen  # noqa: F401  (puts improvements/ and the test helpers on sys.path)
from drag_plus.flags import Flags
from drag_plus.judge.logprob import SCORED, ScoredOutput
from drag_plus.server import probes
from drag_plus.server.probes import PROBES, ProbeContext
from drag_plus.server.registry import (RegistryError, format_list, load_checks, passed,
                                       run_auto, select, summarise)

PROJECT = Path(__file__).resolve().parents[2]
REGISTRY = PROJECT / "improvements/configs/server-checks.json"
# The Server-unverified list lengths in the J02-J17 status logs (49 items in 14 jobs).
UNVERIFIED = {"J02": 4, "J03": 6, "J04": 3, "J06": 3, "J07": 2, "J08": 2, "J09": 6, "J10": 4,
              "J11": 3, "J12": 4, "J13": 3, "J14": 2, "J16": 3, "J17": 4}
DOCS = [{"id": "11", "contents": "Hamlet\nby Shakespeare"}, {"id": "12", "contents": "Play\ntext"}]


class Template:
    """The Llama-3 chat layout with a generation prompt, as FlashRAG renders it."""

    def get_string(self, messages=None, **_):
        body = "".join(f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>"
                       for m in messages)
        return "<|begin_of_text|>" + body + "<|start_header_id|>assistant<|end_header_id|>\n\n"


class PlainTemplate:
    def get_string(self, messages=None, **_):
        return "\n".join(m["content"] for m in messages)


class ServerGenerator:
    """A vLLM-shaped fake: raw outputs, echoed words, scored labels and reasons."""

    def __init__(self, reason="length", score=-0.5):
        self.reason, self.score, self.last_finish_reasons = reason, score, None

    def generate(self, prompts, **params):
        self.last_finish_reasons = None
        if params.get("return_raw_output"):
            return [SimpleNamespace(outputs=[SimpleNamespace(finish_reason=self.reason)])]
        return [_last_word(p) for p in prompts]

    def generate_with_scores(self, prompts, labels, **params):
        self.last_finish_reasons = [SCORED] * len(prompts)
        return [ScoredOutput(labels[0], {labels[0]: self.score, labels[1]: -1.0}) for _ in prompts]


def _last_word(prompt):
    content = str(prompt).split("<|eot_id|>")[0]
    return content.split()[-1]


class Retriever:
    def search(self, query):
        return list(DOCS)

    def batch_search(self, queries, return_score=False):
        rows = [list(DOCS) for _ in queries]
        return (rows, [[0.9, 0.8] for _ in queries]) if return_score else rows


class Faiss:
    def get_num_gpus(self):
        return 1

    def omp_get_max_threads(self):
        return 12

    def get_num_threads(self):  # stands in for torch too
        return 12


def context(**changes):
    values = dict(backend="flashrag", generator=ServerGenerator(), retriever=Retriever(),
                  make_template=lambda arm, dataset: Template(), flags=Flags(),
                  stored=[("who wrote hamlet", ["11", "12"], DOCS)], index_path=__file__,
                  environ={"OMP_NUM_THREADS": "12", "MKL_NUM_THREADS": "12"},
                  import_module=lambda name: Faiss(), read_gpu_rows=lambda: [79.2])
    values.update(changes)
    return ProbeContext(**values)


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.checks = load_checks(REGISTRY)

    def test_every_server_unverified_item_is_registered(self):
        counts = Counter(check.job for check in self.checks)
        self.assertEqual({job: counts[job] for job in UNVERIFIED}, UNVERIFIED)
        self.assertEqual(sum(UNVERIFIED.values()), 49)
        for job in ("J15", "J19", "J24"):
            self.assertIn(job, counts)

    def test_every_auto_check_names_a_probe_and_every_probe_is_used(self):
        used = {check.probe for check in select(self.checks, tier="auto")}
        self.assertEqual(used, set(PROBES))

    def test_malformed_entries_are_refused(self):
        bad = [{"id": "J02-1", "job": "J03", "tier": "auto", "owner": "E00", "expect": "x", "probe": "nvidia_smi"},
               {"id": "J02-1", "job": "J02", "tier": "later", "owner": "E00", "expect": "x"},
               {"id": "J02-1", "job": "J02", "tier": "smoke", "owner": "E00", "expect": " "},
               {"id": "J02-1", "job": "J02", "tier": "auto", "owner": "E00", "expect": "x"},
               {"id": "J02-1", "job": "J02", "tier": "smoke", "owner": "E00", "expect": "x", "probe": "nvidia_smi"},
               {"id": "J02-1", "job": "J02", "tier": "smoke", "owner": "E00", "expect": "x", "extra": 1}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "r.json"
            for entry in bad:
                path.write_text(json.dumps({"checks": [entry]}))
                with self.assertRaises(RegistryError, msg=entry):
                    load_checks(path)
            good = {"id": "J02-1", "job": "J02", "tier": "smoke", "owner": "E00", "expect": "x"}
            path.write_text(json.dumps({"checks": [good, good]}))
            with self.assertRaisesRegex(RegistryError, "duplicate"):
                load_checks(path)

    def test_the_list_groups_by_owner_and_filters(self):
        text = format_list(select(self.checks, owner="E00", tier="auto"))
        self.assertTrue(text.startswith("## E00"))
        self.assertIn("(probe `faiss_gpus`)", text)
        self.assertNotIn("[smoke]", text)


class ProbeTests(unittest.TestCase):
    def test_every_probe_passes_on_a_healthy_server(self):
        for name, (function, _) in PROBES.items():
            ok, detail = function(context())
            self.assertTrue(ok, f"{name}: {detail}")

    def test_probes_fail_on_the_defects_they_target(self):
        cases = {
            "raw_finish_reason": context(generator=ServerGenerator(reason="stop")),
            "scored_judge_coverage": context(generator=ServerGenerator(score=None)),
            "batch_search_matches_stored_naive": context(stored=[("q", ["99"], DOCS)]),
            "faiss_gpus": context(import_module=lambda name: SimpleNamespace(get_num_gpus=lambda: 0)),
            "thread_env": context(import_module=lambda name: SimpleNamespace(
                omp_get_max_threads=lambda: 12, get_num_threads=lambda: 4)),
            "nvidia_smi": context(read_gpu_rows=lambda: [6.0]),
            "template_renders_messages": context(make_template=lambda a, d: PlainTemplate()),
            "index_identity_fast": context(index_path="/no/such/index.faiss"),
            "b2_search_vs_batch": context(retriever=SimpleNamespace(
                search=lambda q: [{"id": "x"}], batch_search=lambda qs: [list(DOCS)])),
        }
        for name, ctx in cases.items():
            ok, detail = PROBES[name][0](ctx)
            self.assertFalse(ok, f"{name} should fail: {detail}")

    def test_two_cards_summing_to_80_gib_are_not_one_a100(self):
        self.assertFalse(probes.nvidia_smi(context(read_gpu_rows=lambda: [40.0, 40.0]))[0])
        self.assertEqual(probes.parse_gpu_rows("81920\n\n"), [80.0])

    def test_the_template_must_end_with_the_generation_prompt(self):
        class NoGeneration(Template):
            def get_string(self, messages=None, **_):
                return super().get_string(messages).rsplit("<|start_header_id|>assistant", 1)[0]
        ctx = context(make_template=lambda a, d: NoGeneration())
        self.assertFalse(probes.template_renders_messages(ctx)[0])

    def test_b2_divergence_on_the_long_query_is_reported_not_failed(self):
        retriever = SimpleNamespace(
            search=lambda q: [{"id": "short"}] if len(q) < 64 else [{"id": "truncated"}],
            batch_search=lambda qs: [[{"id": "short"}] if len(qs[0]) < 64 else [{"id": "full"}]])
        ok, detail = probes.b2_search_vs_batch(context(retriever=retriever))
        self.assertTrue(ok, detail)
        self.assertIn("B2 reproduced", detail)

    def test_unordered_batches_and_stale_reasons_fail(self):
        shuffled = ServerGenerator()
        shuffled.generate = lambda prompts, **p: list(reversed([_last_word(x) for x in prompts]))
        self.assertFalse(probes.batch_order_preserved(context(generator=shuffled))[0])
        stale = ServerGenerator()
        stale.generate = lambda prompts, **p: ["text"]  # never clears last_finish_reasons
        self.assertFalse(probes.scored_finish_reasons(context(generator=stale))[0])


class RunnerTests(unittest.TestCase):
    def test_each_probe_runs_once_and_maps_to_every_check(self):
        calls = Counter()
        original = PROBES["scored_judge_coverage"]
        PROBES["scored_judge_coverage"] = (lambda ctx: calls.update(["x"]) or (True, "ok"), "any")
        try:
            results = run_auto(load_checks(REGISTRY), context())
        finally:
            PROBES["scored_judge_coverage"] = original
        self.assertEqual(calls["x"], 1)
        self.assertEqual([r["id"] for r in results if r["probe"] == "scored_judge_coverage"],
                         ["J12-1", "J12-2"])
        self.assertTrue(passed(results))

    def test_an_exception_is_an_error_and_server_probes_skip_off_the_server(self):
        broken = context(generator=SimpleNamespace(generate=None, generate_with_scores=None))
        results = run_auto(load_checks(REGISTRY), broken)
        self.assertGreater(summarise(results)["error"], 0)
        self.assertFalse(passed(results))
        fake = run_auto(load_checks(REGISTRY), context(backend="fake"))
        skipped = {r["probe"] for r in fake if r["status"] == "skipped"}
        self.assertEqual(skipped, {n for n, (_, req) in PROBES.items() if req == "server"})
        self.assertTrue(passed(fake))
        self.assertFalse(passed(fake, on_server=True))  # a skip on the server is a failure

    def test_nothing_run_is_never_a_pass(self):
        self.assertFalse(passed([]))
        self.assertFalse(passed([{"status": "skipped"}]))


class ScriptTests(unittest.TestCase):
    def script(self, *args):
        return subprocess.run([sys.executable, str(PROJECT / "scripts/server_checks.py"), *args],
                              capture_output=True, text=True, timeout=60)

    def test_list_runs_anywhere(self):
        result = self.script("--list", "--owner", "E00")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("**J09-1**", result.stdout)
        self.assertNotIn("## E01", result.stdout)

    def test_a_run_that_would_test_nothing_is_refused(self):
        for args in (("--owner", "E01"), ("--owner", "e00"), ("--tier", "smoke")):
            result = self.script("--run", "--backend", "fake", *args)
            self.assertEqual(result.returncode, 2, (args, result.stdout))
            self.assertIn("server_checks:", result.stderr)

    def test_run_on_the_fake_backend_writes_a_passing_report(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.script("--run", "--backend", "fake", "--out", folder)
            report = json.loads((Path(folder) / "server-checks.json").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(report["passed"])
        self.assertEqual(report["summary"]["fail"] + report["summary"]["error"], 0)
        self.assertGreaterEqual(report["summary"]["pass"], 3)
        self.assertIn("versions", report)


if __name__ == "__main__":
    unittest.main()
