"""The caching proxy: one path for both call shapes, once per key, scores (J04)."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drag_plus.retrieval.cache import (CacheKey, CacheStore, CachingRetriever,
                                       index_identity, normalise_query)

DEFAULT_DOCS = 2
IDENTITY = {"index": "idx", "encoder": "e5", "top_k": 3}


def _docs_for(query, num=None):
    return [{"id": f"{query}::{rank}", "contents": f"passage {rank}"}
            for rank in range(num or DEFAULT_DOCS)]


class DivergentRetriever:
    """The single and batch paths answer differently on purpose.

    This is the shape of the pinned FlashRAG defect B2, where ``search`` and
    ``batch_search`` embed different text for the same query. If the proxy ever sent a
    single query down ``search``, these results would surface.
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

    def search(self, query, num=None):
        return [{"id": f"wrong::{query}", "contents": "single path"}]

    def batch_search(self, queries, num=None):
        self.batch_calls.append(list(queries))
        return [_docs_for(query, num) for query in queries]


class NoNumRetriever(UnscoredRetriever):
    """Neither a ``num`` nor a ``return_score`` parameter."""

    def batch_search(self, queries):
        self.batch_calls.append(list(queries))
        return [_docs_for(query) for query in queries]


class KwargsOnlyRetriever:
    """Accepts ``**kwargs`` but ignores them and never returns scores."""

    def search(self, query, **kwargs):
        return []

    def batch_search(self, queries, **kwargs):
        return [_docs_for(query) for query in queries]


class CountingRetriever:
    """Records the exact requests the proxy forwards, keeping the inner behaviour."""

    def __init__(self):
        self.inner = DivergentRetriever()
        self.requests = []

    def search(self, query, **kwargs):
        return self.inner.search(query, **kwargs)

    def batch_search(self, queries, **kwargs):
        self.requests.append((list(queries), dict(kwargs)))
        return self.inner.batch_search(queries, **kwargs)


def build(inner=None, **kwargs):
    inner = inner or DivergentRetriever()
    return CachingRetriever(inner, index="idx", encoder="e5", top_k=3, **kwargs), inner


class SinglePathTests(unittest.TestCase):
    def test_search_and_batch_search_agree_on_ids_scores_and_order(self):
        # Separate stores, so each call really resolves through the inner retriever.
        single_proxy, _ = build()
        batch_proxy, _ = build()
        single = single_proxy.search("a question", return_score=True)
        batched = batch_proxy.batch_search(["a question"], return_score=True)[0]
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

    def test_the_inner_receives_the_callers_query_text_not_the_normalised_key(self):
        counting = CountingRetriever()
        proxy = CachingRetriever(counting, index="idx", encoder="e5", top_k=3)
        proxy.batch_search(["  a   b  ", "a b"])
        self.assertEqual(counting.requests, [(["  a   b  "], {"return_score": True})],
                         "one fetch, using the first spelling")
        self.assertEqual(len(proxy.store.get(CacheKey("idx", "e5", 3, "a b"))[0]), 2)

    def test_a_different_top_k_is_a_miss(self):
        proxy, inner = build()
        proxy.batch_search(["one"], num=3)
        proxy.batch_search(["one"], num=5)
        self.assertEqual(inner.batch_calls, [["one"], ["one"]])

    def test_a_different_index_or_encoder_is_a_miss(self):
        store = CacheStore()
        inner = DivergentRetriever()
        base = CachingRetriever(inner, store=store, index="idx", encoder="e5", top_k=3)
        other = CachingRetriever(inner, store=store, index="other", encoder="e5", top_k=3)
        renamed = CachingRetriever(inner, store=store, index="idx", encoder="bge", top_k=3)
        base.search("one")
        other.search("one")
        renamed.search("one")
        self.assertEqual(len(inner.batch_calls), 3)
        other.search("one")
        renamed.search("one")
        self.assertEqual(len(inner.batch_calls), 3, "each identity now has its own entry")

    def test_an_inner_retriever_without_num_refuses_a_requested_depth(self):
        proxy = CachingRetriever(NoNumRetriever(), index="idx", encoder="e5", top_k=3)
        with self.assertRaisesRegex(ValueError, "cannot honour num"):
            proxy.search("one", num=1)
        self.assertEqual(len(proxy.search("one")), DEFAULT_DOCS)

    def test_a_string_batch_is_one_query_not_its_characters(self):
        proxy, inner = build()
        self.assertEqual(len(proxy.batch_search("abc")), 1)
        self.assertEqual(inner.batch_calls, [["abc"]])


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

    def test_asking_an_unscored_retriever_for_scores_is_an_error(self):
        proxy, _ = build(UnscoredRetriever())
        with self.assertRaisesRegex(ValueError, "cannot return scores"):
            proxy.search("one", return_score=True)

    def test_a_kwargs_only_retriever_that_ignores_return_score_is_not_trusted(self):
        proxy = CachingRetriever(KwargsOnlyRetriever(), index="idx", encoder="e5", top_k=3)
        with self.assertRaisesRegex(TypeError, "must return"):
            proxy.search("one", return_score=True)
        self.assertTrue(proxy.store.contains(CacheKey("idx", "e5", 3, "one")) is False)


class KeyTests(unittest.TestCase):
    def test_whitespace_and_unicode_are_normalised_but_case_is_kept(self):
        self.assertEqual(normalise_query("  a\tb\n c "), "a b c")
        self.assertEqual(normalise_query("Cafe\u0301"), "Caf\u00e9")
        self.assertNotEqual(normalise_query("Paris"), normalise_query("paris"))

    def test_key_uses_the_configured_top_k_when_num_is_absent(self):
        key = CacheKey.build(IDENTITY, " q ")
        self.assertEqual((key.top_k, key.query), (3, "q"))
        self.assertEqual(CacheKey.build(IDENTITY, "q", num=7).top_k, 7)

    def test_index_identity_changes_with_the_file_contents(self):
        handle = tempfile.NamedTemporaryFile("w", suffix=".idx", delete=False)
        handle.write("x")
        handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        before = index_identity(handle.name)
        with open(handle.name, "a", encoding="utf-8") as stream:
            stream.write("y")
        after = index_identity(handle.name)
        self.assertNotEqual(before, after)
        self.assertTrue(before.startswith(str(Path(handle.name).resolve())))
        self.assertIn(":1:", before)

    def test_a_missing_index_file_still_gets_a_stable_identity(self):
        self.assertIn("/nonexistent/x.idx", index_identity("/nonexistent/x.idx"))


if __name__ == "__main__":
    unittest.main()
