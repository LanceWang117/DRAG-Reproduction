"""The composed hybrid retriever, its wiring into the backend, and key separation."""
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

IMPROVEMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMPROVEMENTS))
from drag_plus.flags import Flags  # noqa: E402
from drag_plus.retrieval.cache import CacheStore, CacheKey, CachingRetriever  # noqa: E402
from drag_plus.retrieval.hybrid import (HybridRetriever, config_fingerprint,  # noqa: E402
                                        source_results)
from drag_plus.runner.backends import build_retriever  # noqa: E402


def doc(doc_id, title="T"):
    return {"id": doc_id, "contents": f"{title}\nbody {doc_id}"}


class Scripted:
    """A retriever with fixed rows and scores per query."""

    def __init__(self, rows, scores=None, num=True):
        self.rows, self.scores, self.numbered = rows, scores or {}, num
        self.calls = []

    def batch_search(self, queries, **kwargs):
        self.calls.append((list(queries), dict(kwargs)))
        docs = [self.rows.get(q, [])[:kwargs.get("num")] for q in queries]
        scores = [self.scores.get(q, [1.0] * len(row))[:len(row)]
                  for q, row in zip(queries, docs)]
        return (docs, scores) if kwargs.get("return_score") else docs

    def search(self, query, **kwargs):
        return self.batch_search([query], **kwargs)[0]


class SourceResultTests(unittest.TestCase):
    def test_scores_are_only_requested_when_needed(self):
        plain, scored = Scripted({}), Scripted({})
        source_results(plain, ["q"], 2, False)
        source_results(scored, ["q"], 2, True)
        self.assertEqual(plain.calls[0][1], {"num": 2})
        self.assertEqual(scored.calls[0][1], {"num": 2, "return_score": True})

    def test_depth_truncates_documents_and_scores_together(self):
        rows = {"q": [doc("a"), doc("b"), doc("c")]}
        docs, scores = source_results(Scripted(rows, {"q": [0.9, 0.5, 0.1]}), ["q"], 2, True)[0]
        self.assertEqual([d["id"] for d in docs], ["a", "b"])
        self.assertEqual(scores, [0.9, 0.5])

    def test_a_retriever_without_num_still_works(self):
        class Plain:
            def batch_search(self, queries):
                return [[doc("a"), doc("b")] for _ in queries]
        docs, _ = source_results(Plain(), ["q"], 1, False)[0]
        self.assertEqual([d["id"] for d in docs], ["a"])

    def test_a_floor_needs_a_retriever_that_can_score(self):
        class NoScores:
            def batch_search(self, queries):
                return [[doc("a")] for _ in queries]
        with self.assertRaisesRegex(ValueError, "cannot return scores"):
            source_results(NoScores(), ["q"], 2, True)

    def test_result_count_must_match_the_query_count(self):
        class Short:
            def batch_search(self, queries, **kwargs):
                return [[doc("a")]]
        with self.assertRaisesRegex(ValueError, "1 document lists"):
            source_results(Short(), ["q1", "q2"], 1, False)


class SpyReranker:
    def __init__(self, order=None):
        self.seen, self.order = [], order

    def rerank(self, query, docs):
        self.seen.append([d["id"] for d in docs])
        order = self.order or list(range(len(docs)))
        return [(docs[index], 1.0 - rank / 10) for rank, index in enumerate(order)]


class HybridRetrieverTests(unittest.TestCase):
    def build(self, dense_rows, bm25_rows, **kwargs):
        dense = Scripted(dense_rows)
        bm25 = Scripted(bm25_rows)
        return HybridRetriever(dense, bm25, **kwargs), dense, bm25

    def test_both_rankings_are_fused_and_truncated_to_top_k(self):
        hybrid, dense, bm25 = self.build(
            {"q": [doc("a"), doc("b"), doc("c")]}, {"q": [doc("c"), doc("d")]}, top_k=3)
        self.assertEqual([d["id"] for d in hybrid.search("q")], ["c", "a", "b"])
        self.assertEqual(dense.calls[0][1], {"num": 20})
        self.assertEqual(bm25.calls[0][1], {"num": 20})

    def test_num_overrides_the_configured_depth_and_is_validated(self):
        hybrid, _, _ = self.build({"q": [doc("a"), doc("b")]}, {"q": [doc("c")]}, top_k=3)
        self.assertEqual(len(hybrid.search("q", num=1)), 1)
        with self.assertRaisesRegex(ValueError, "num must be >= 1"):
            hybrid.search("q", num=0)

    def test_floors_are_applied_per_retriever(self):
        hybrid, dense, bm25 = self.build(
            {"q": [doc("a"), doc("b")]}, {"q": [doc("c"), doc("d")]},
            dense_floor=0.5, bm25_floor=2.0)
        dense.scores = {"q": [0.9, 0.1]}
        bm25.scores = {"q": [5.0, 1.0]}
        self.assertEqual([d["id"] for d in hybrid.search("q")], ["a", "c"])
        self.assertTrue(dense.calls[0][1]["return_score"])

    def test_rerank_k_bounds_the_cross_encoder_to_the_head(self):
        reranker = SpyReranker(order=[1, 0])
        hybrid, _, _ = self.build({"q": [doc("a"), doc("b")]},
                                  {"q": [doc("c"), doc("d")]},
                                  rerank_k=2, top_k=4, reranker=reranker)
        self.assertEqual([d["id"] for d in hybrid.search("q")], ["c", "a", "b", "d"])
        self.assertEqual(reranker.seen, [["a", "c"]])

    def test_rerank_k_without_a_reranker_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "no reranker"):
            HybridRetriever(Scripted({}), Scripted({}), rerank_k=3)

    def test_the_title_cap_is_applied_before_top_k(self):
        hybrid, _, _ = self.build(
            {"q": [doc("a", "Same"), doc("b", "Same"), doc("c", "Other")]},
            {"q": []}, per_title_cap=1, top_k=2)
        self.assertEqual([d["id"] for d in hybrid.search("q")], ["a", "c"])

    def test_return_score_returns_the_fused_score_with_each_passage(self):
        hybrid, _, _ = self.build({"q": [doc("a")]}, {"q": [doc("a")]})
        docs, scores = hybrid.search("q", return_score=True)
        self.assertEqual([d["id"] for d in docs], ["a"])
        self.assertEqual(scores, [2 / 61])

    def test_batch_search_keeps_request_order(self):
        hybrid, _, _ = self.build({"q1": [doc("a")], "q2": [doc("b")]}, {"q2": [doc("c")]})
        rows = hybrid.batch_search(["q1", "q2"])
        self.assertEqual([[d["id"] for d in row] for row in rows], [["a"], ["b", "c"]])


class BuildRetrieverTests(unittest.TestCase):
    def test_the_default_retriever_is_the_dense_one_untouched(self):
        dense = Scripted({})
        self.assertIs(build_retriever(Flags(), dense), dense)

    def test_hybrid_without_an_index_is_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, "bm25_index"):
            build_retriever(Flags({"retriever": "hybrid"}), Scripted({}))

    def test_hybrid_is_built_from_the_flags(self):
        flags = Flags({"retriever": "hybrid", "bm25_index": "/index", "bm25_top_k": 7,
                       "dense_top_k": 9, "hybrid_top_k": 4, "per_title_cap": 2})
        hybrid = build_retriever(flags, Scripted({}))
        self.assertIsInstance(hybrid, HybridRetriever)
        self.assertEqual((hybrid.dense_k, hybrid.bm25_k, hybrid.top_k), (9, 7, 4))
        self.assertEqual((hybrid.bm25.path, hybrid.bm25.depth), (Path("/index"), 7))
        self.assertIsNone(hybrid.reranker)

    def test_rerank_k_builds_a_cross_encoder_without_importing_it(self):
        flags = Flags({"retriever": "hybrid", "bm25_index": "/index", "rerank_k": 8,
                       "rerank_model": "BAAI/bge-reranker-base"})
        hybrid = build_retriever(flags, Scripted({}))
        self.assertEqual(hybrid.reranker.model_name, "BAAI/bge-reranker-base")
        self.assertEqual(hybrid.rerank_k, 8)


class CacheSeparationTests(unittest.TestCase):
    def test_dense_and_hybrid_configurations_never_share_an_entry(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "cache.jsonl"

        def proxy(flags):
            return CachingRetriever(Scripted({"q": [doc("a")]}), index="idx",
                                    encoder="e5", top_k=3, path=path,
                                    config=config_fingerprint(flags))
        proxy(Flags()).search("q")
        proxy(Flags({"retriever": "hybrid"})).search("q")
        store = CacheStore(path)
        self.assertTrue(store.contains(CacheKey("idx", "e5", 3, "q", config_fingerprint(Flags()))))
        hybrid_config = config_fingerprint(Flags({"retriever": "hybrid"}))
        self.assertTrue(store.contains(CacheKey("idx", "e5", 3, "q", config=hybrid_config)))
        self.assertEqual(len(path.read_text().splitlines()), 2)

    def test_a_different_rrf_k_is_a_different_entry(self):
        base = Flags({"retriever": "hybrid"})
        self.assertNotEqual(config_fingerprint(base),
                            config_fingerprint(base.with_changes(rrf_k=61)))


BLOCK_SCRIPT = """
import sys
HEAVY = set(__HEAVY__)
class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in HEAVY:
            raise ImportError("blocked heavy import: " + name)
        return None
sys.meta_path.insert(0, Blocker())
sys.path.insert(0, __IMPROVEMENTS__)
from drag_plus.flags import Flags
from drag_plus.runner.backends import build_retriever

class Dense:
    def batch_search(self, queries, num=None):
        return [[{"id": "d", "contents": "T\\ndense"}][:num] for _ in queries]

flags = Flags({"retriever": "hybrid", "bm25_index": "/missing", "rerank_k": None})
hybrid = build_retriever(flags, Dense())
assert hybrid.reranker is None
assert not any(name.split(".")[0] in HEAVY for name in sys.modules)
print("OK")
"""


class LazyImportTests(unittest.TestCase):
    def test_building_a_hybrid_without_reranking_imports_no_heavy_library(self):
        script = (BLOCK_SCRIPT.replace("__HEAVY__", repr(
                      ("bm25s", "sentence_transformers", "torch", "transformers", "vllm",
                       "faiss", "flashrag")))
                  .replace("__IMPROVEMENTS__", repr(str(IMPROVEMENTS))))
        done = subprocess.run([sys.executable, "-c", script],
                              capture_output=True, text=True, timeout=30)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("OK", done.stdout)


if __name__ == "__main__":
    unittest.main()
