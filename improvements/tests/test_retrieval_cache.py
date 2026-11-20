"""The caching proxy: one path for both call shapes, once per key, and persistence (J04)."""
import json
from pathlib import Path
import tempfile
import unittest

from drag_plus.retrieval.cache import (CacheKey, CacheStore, CachingRetriever,
                                       index_identity, normalise_query)

DEFAULT_DOCS = 2


def _docs_for(query, num=None):
    return [{"id": f"{query}::{rank}", "contents": f"passage {rank}"}
            for rank in range(num or DEFAULT_DOCS)]


class DivergentRetriever:
    """Answers differ on purpose: the single path returns junk, the batch path is right.

    This is the shape of the pinned FlashRAG defect B2, where ``search`` and
    ``batch_search`` embed different text for the same query. If the proxy ever sends a
    single query down ``search``, these results would show up.
    """

    def __init__(self):
        self.batch_calls = []
        self.search_calls = []

    def search(self, query, num=None, return_score=False):
        self.search_calls.append(query)
        docs = [{"id": f"wrong::{query}", "contents": "single path"}]
        return (docs, [0.0]) if return_score else docs

    def batch_search(self, queries, num=None, return_score=False):
        self.batch_calls.append(list(queries))
        docs = [_docs_for(query, num) for query in queries]
        scores = [[1.0 - 0.5 * rank for rank in range(len(d))] for d in docs]
        return (docs, scores) if return_score else docs


class UnscoredRetriever:
    """The other call shape: no ``return_score`` parameter, documents only."""

    def __init__(self):
        self.batch_calls = []
        self.search_calls = []

    def search(self, query, num=None):
        self.search_calls.append(query)
        return [{"id": f"wrong::{query}", "contents": "single path"}]

    def batch_search(self, queries, num=None):
        self.batch_calls.append(list(queries))
        return [_docs_for(query, num) for query in queries]


def build(inner=None, path=None, warn=None):
    inner = inner or DivergentRetriever()
    kwargs = {} if warn is None else {"warn": warn}
    proxy = CachingRetriever(inner, index="idx", encoder="e5", top_k=3, path=path, **kwargs)
    return proxy, inner


class SinglePathTests(unittest.TestCase):
    def test_search_and_batch_search_agree_on_ids_scores_and_order(self):
        proxy, inner = build()
        single = proxy.search("a question", return_score=True)
        batched = proxy.batch_search(["a question"], return_score=True)[0]
        self.assertEqual(single, batched)
        self.assertEqual([d["id"] for d in single[0]], [d["id"] for d in batched[0]])

    def test_single_search_never_reaches_the_inner_single_path(self):
        proxy, inner = build()
        proxy.search("a question")
        self.assertEqual(inner.search_calls, [])
        self.assertEqual(inner.batch_calls, [["a question"]])

    def test_a_hit_after_a_single_call_is_not_refetched(self):
        proxy, inner = build()
        proxy.search("a question")
        proxy.batch_search(["a question", "a question"])
        self.assertEqual(len(inner.batch_calls), 1)


class OneCallPerKeyTests(unittest.TestCase):
    def test_misses_are_batched_once_and_deduplicated(self):
        proxy, inner = build()
        proxy.batch_search(["one", "two", "one", "three"])
        self.assertEqual(inner.batch_calls, [["one", "two", "three"]])

    def test_repeat_across_calls_hits_the_cache(self):
        proxy, inner = build()
        for _ in range(3):
            proxy.batch_search(["one", "two"])
        self.assertEqual(len(inner.batch_calls), 1)

    def test_request_order_is_preserved_with_duplicates(self):
        proxy, _ = build()
        docs = proxy.batch_search(["b", "a", "b"])
        self.assertEqual([d[0]["id"].split("::")[0] for d in docs], ["b", "a", "b"])

    def test_a_different_top_k_is_a_miss(self):
        proxy, inner = build()
        proxy.batch_search(["one"], num=3)
        proxy.batch_search(["one"], num=5)
        self.assertEqual(inner.batch_calls, [["one"], ["one"]])

    def test_a_different_index_or_encoder_is_a_miss(self):
        inner = DivergentRetriever()
        other = CachingRetriever(inner, index="other", encoder="e5", top_k=3)
        base = CachingRetriever(inner, index="idx", encoder="e5", top_k=3)
        base.search("one")
        other.search("one")
        self.assertEqual(len(inner.batch_calls), 2)
        renamed = CachingRetriever(inner, index="idx", encoder="bge", top_k=3)
        renamed.search("one")
        self.assertEqual(len(inner.batch_calls), 3)


class CopyTests(unittest.TestCase):
    def test_mutating_a_returned_doc_cannot_poison_the_cache(self):
        proxy, _ = build()
        docs = proxy.search("one")
        docs[0]["id"], docs[0]["contents"] = "tampered", "tampered"
        self.assertEqual(proxy.search("one")[0]["id"], "one::0")

    def test_mutating_a_returned_score_list_cannot_poison_the_cache(self):
        proxy, _ = build()
        scores = proxy.search("one", return_score=True)[1]
        scores[0] = -99.0
        self.assertEqual(proxy.search("one", return_score=True)[1], [1.0, 0.5])


class UnscoredRetrieverTests(unittest.TestCase):
    def test_unscored_inner_retriever_still_caches_documents(self):
        proxy, inner = build(UnscoredRetriever())
        self.assertEqual(len(proxy.search("one")), DEFAULT_DOCS)
        proxy.search("one")
        self.assertEqual(len(inner.batch_calls), 1)
        self.assertNotIn("score", proxy.search("one")[0])

    def test_asking_an_unscored_retriever_for_scores_is_an_error(self):
        proxy, _ = build(UnscoredRetriever())
        with self.assertRaisesRegex(ValueError, "cannot return scores"):
            proxy.search("one", return_score=True)


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        handle.close()
        Path(handle.name).unlink()
        self.path = Path(handle.name)
        self.addCleanup(lambda: self.path.unlink(missing_ok=True))

    def test_round_trip_survives_a_new_process(self):
        first, inner = build(path=self.path)
        before = first.batch_search(["one", "two"], return_score=True)
        self.assertEqual(len(self.path.read_text().splitlines()), 2)
        reopened, reopened_inner = build(DivergentRetriever(), path=self.path)
        self.assertEqual(reopened.batch_search(["one", "two"], return_score=True), before)
        self.assertEqual(reopened_inner.batch_calls, [])

    def test_one_line_per_query_with_the_key_and_the_scores(self):
        proxy, _ = build(path=self.path)
        proxy.batch_search(["one"], num=3)
        record = json.loads(self.path.read_text().splitlines()[0])
        self.assertEqual(record["index"], "idx")
        self.assertEqual(record["encoder"], "e5")
        self.assertEqual(record["top_k"], 3)
        self.assertEqual(record["query"], "one")
        self.assertEqual(record["scores"], [1.0, 0.5, 0.0])
        self.assertEqual([d["id"] for d in record["docs"]], ["one::0", "one::1", "one::2"])

    def test_a_corrupt_last_line_is_skipped_with_a_warning(self):
        proxy, _ = build(path=self.path)
        proxy.search("one")
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write('{"index": "idx", "encoder": "e5", "top_k": 3, "que')
        warned = []
        reopened, inner = build(DivergentRetriever(), path=self.path, warn=warned.append)
        self.assertEqual(len(warned), 1)
        self.assertIn("line 2", warned[0])
        self.assertEqual(reopened.search("one")[0]["id"], "one::0")
        self.assertEqual(inner.batch_calls, [])

    def test_nothing_is_written_when_the_inner_call_fails(self):
        class Broken:
            def search(self, query, **kwargs):
                raise AssertionError("unused")

            def batch_search(self, queries, **kwargs):
                raise RuntimeError("index unavailable")
        proxy, _ = build(Broken(), path=self.path)
        with self.assertRaisesRegex(RuntimeError, "index unavailable"):
            proxy.search("one")
        self.assertFalse(self.path.exists())


class KeyTests(unittest.TestCase):
    def test_whitespace_and_unicode_are_normalised_but_case_is_kept(self):
        self.assertEqual(normalise_query("  a\tb\n c "), "a b c")
        self.assertEqual(normalise_query("Cafe\u0301"), "Caf\u00e9")
        self.assertNotEqual(normalise_query("Paris"), normalise_query("paris"))

    def test_key_uses_the_configured_top_k_when_num_is_absent(self):
        key = CacheKey.build({"index": "i", "encoder": "e", "top_k": 3}, " q ")
        self.assertEqual((key.top_k, key.query), (3, "q"))
        self.assertEqual(CacheKey.build({"index": "i", "encoder": "e", "top_k": 3},
                                        "q", num=7).top_k, 7)

    def test_index_identity_tracks_size_and_invalid_paths(self):
        handle = tempfile.NamedTemporaryFile("w", suffix=".idx", delete=False)
        handle.write("x")
        handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        self.assertTrue(index_identity(handle.name).startswith(str(Path(handle.name).resolve())))
        self.assertIn(str(Path("/nonexistent/x.idx")), index_identity("/nonexistent/x.idx"))
        self.assertTrue(CacheStore(path=None).contains(CacheKey("i", "e", 1, "q")) is False)


if __name__ == "__main__":
    unittest.main()
