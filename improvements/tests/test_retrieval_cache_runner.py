"""Runner integration: one shared proxy, flag off equals the baseline (J04)."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drag_plus.retrieval.cache import CachingRetriever
from drag_plus.runner import run as runner
from drag_plus.runner.backends import build_fake_backend
from drag_plus.runner.run import RunRequest, run_variant
from drag_plus.runner.variant import load_variant, parse_variant

PROJECT = Path(__file__).resolve().parents[2]
CACHED_VARIANT = PROJECT / "improvements" / "configs" / "cached.json"

ROWS = [{"dataset": "NQ", "split": "dev", "id": f"q{i}",
         "question": f"question number {i}?", "golden_answers": ["answer 1"]}
        for i in range(3)]


class CountingRetriever:
    """Records each batch request the proxy forwards to the backend's retriever."""

    def __init__(self, inner):
        self.inner = inner
        self.requests = []

    def search(self, query):
        return self.inner.search(query)

    def batch_search(self, queries):
        self.requests.append(list(queries))
        return self.inner.batch_search(queries)


def request(out_dir, flags=None, backend=None, **overrides):
    backend = backend or build_fake_backend()
    values = dict(rows=ROWS, variant=parse_variant({"name": "t", "flags": flags or {}}),
                  backend=backend, out_dir=out_dir, profile=False,
                  now=lambda: "2026-01-01T00:00:00Z",
                  environment=lambda: {"commit": "test"})
    values.update(overrides)
    return RunRequest(**values)


class SharedProxyTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def test_both_arms_receive_the_same_proxy_instance(self):
        pipelines = runner._Pipelines(request(self.root / "run",
                                              {"retrieval_cache": str(self.root / "c.jsonl")}))
        drag, naive = pipelines.get("drag", "NQ"), pipelines.get("naive", "NQ")
        self.assertIsInstance(drag._ctx.backends.retriever, CachingRetriever)
        self.assertIs(drag._ctx.backends.retriever, naive._ctx.backends.retriever)
        self.assertIs(drag._ctx.backends.retriever, pipelines.retriever)

    def test_every_dataset_shares_that_one_proxy(self):
        pipelines = runner._Pipelines(request(self.root / "run",
                                              {"retrieval_cache": str(self.root / "c.jsonl")}))
        built = {pipelines.get(arm, dataset)._ctx.backends.retriever
                 for arm in ("drag", "naive") for dataset in ("NQ", "StrategyQA")}
        self.assertEqual(len(built), 1)

    def test_the_underlying_retriever_is_called_once_per_unique_query(self):
        backend = build_fake_backend()
        counting = CountingRetriever(backend.retriever)
        backend.retriever = counting
        cache_path = self.root / "cache" / "retrieval-cache.jsonl"
        run_variant(request(self.root / "run", {"retrieval_cache": str(cache_path)},
                            backend=backend, interleave=True))
        asked = [query for call in counting.requests for query in call]
        self.assertEqual(len(asked), len(set(asked)), "no query may be fetched twice")
        self.assertTrue({row["question"] for row in ROWS} <= set(asked))
        lines = [json.loads(line) for line in cache_path.read_text().splitlines()]
        self.assertEqual(sorted(record["query"] for record in lines), sorted(set(asked)))

    def test_a_missing_parent_directory_is_created(self):
        cache_path = self.root / "deep" / "nested" / "retrieval-cache.jsonl"
        run_variant(request(self.root / "run", {"retrieval_cache": str(cache_path)}))
        self.assertTrue(cache_path.exists())

    def test_a_relative_path_resolves_against_the_run_directory(self):
        out = self.root / "run"
        manifest = run_variant(request(out, {"retrieval_cache": "retrieval-cache.jsonl"}))
        self.assertEqual(manifest["flags"]["retrieval_cache"], "retrieval-cache.jsonl")
        self.assertTrue((out / "retrieval-cache.jsonl").exists())

    def test_a_second_run_reuses_the_cache_file(self):
        cache_path = self.root / "shared.jsonl"
        first = build_fake_backend()
        counting = CountingRetriever(first.retriever)
        first.retriever = counting
        run_variant(request(self.root / "one", {"retrieval_cache": str(cache_path)},
                            backend=first))
        second = build_fake_backend()
        reused = CountingRetriever(second.retriever)
        second.retriever = reused
        run_variant(request(self.root / "two", {"retrieval_cache": str(cache_path)},
                            backend=second))
        self.assertEqual(reused.requests, [], "every query was already cached")


class FlagOffTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def test_the_default_passes_the_bare_retriever_through(self):
        backend = build_fake_backend()
        pipelines = runner._Pipelines(request(self.root / "run", backend=backend))
        self.assertIs(pipelines.retriever, backend.retriever)
        self.assertIs(pipelines.get("drag", "NQ")._ctx.backends.retriever, backend.retriever)

    def test_the_default_run_writes_no_cache_file_and_the_same_transcripts(self):
        off, on = self.root / "off", self.root / "on"
        cache_path = self.root / "c.jsonl"
        plain = run_variant(request(off))
        self.assertFalse(cache_path.exists(), "the default run must not create a cache")
        self.assertEqual(list(off.rglob("retrieval-cache.jsonl")), [])
        cached = run_variant(request(on, {"retrieval_cache": str(cache_path)}))
        self.assertTrue(cache_path.exists())
        self.assertIsNone(plain["flags"]["retrieval_cache"])
        self.assertEqual(cached["flags"]["retrieval_cache"], str(cache_path))
        self.assertEqual(
            (off / "transcripts.jsonl").read_text().replace('"run_id": "off"', '"run_id": "on"'),
            (on / "transcripts.jsonl").read_text())

    def test_both_arms_see_identical_passages_for_the_original_query(self):
        out = self.root / "run"
        run_variant(request(out, {"retrieval_cache": str(self.root / "c.jsonl")},
                            interleave=True))
        naive, drag = {}, {}
        for line in (out / "transcripts.jsonl").read_text().splitlines():
            row = json.loads(line)
            pool = next((e["value"] for e in row["entries"]
                         if e["key"] == "QueryStage_QueryPool"), None)
            if row["arm"] == "naive":
                naive[row["id"]] = next(e["value"] for e in row["entries"]
                                        if e["key"] == "retrieval_result")
            elif pool is not None:
                drag[row["id"]] = pool
        self.assertEqual(set(drag), set(naive))
        compared = 0
        for question_id, pool in drag.items():
            question = next(row["question"] for row in ROWS if row["id"] == question_id)
            if question not in pool:  # a rewritten query replaced the original entry
                continue
            self.assertEqual(pool[question], naive[question_id],
                             "the same query must retrieve the same passages in both arms")
            compared += 1
        self.assertTrue(compared, "at least one question kept its original query")

    def test_the_shipped_cached_variant_is_valid(self):
        variant = load_variant(CACHED_VARIANT)
        self.assertEqual(variant.flags.retrieval_cache, "retrieval-cache.jsonl")
        self.assertEqual(variant.arms, ("drag", "naive"))


if __name__ == "__main__":
    unittest.main()
