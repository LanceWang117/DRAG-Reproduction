"""Retrieval cache persistence: JSONL round trip, torn tails, corrupt lines (J04)."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drag_plus.retrieval.cache import CacheKey, CachingRetriever

from test_retrieval_cache import DEFAULT_DOCS, DivergentRetriever, UnscoredRetriever


def build(path, inner=None, warn=None):
    kwargs = {} if warn is None else {"warn": warn}
    inner = inner or DivergentRetriever()
    proxy = CachingRetriever(inner, index="idx", encoder="e5", top_k=3, path=path, **kwargs)
    return proxy, inner


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        handle.close()
        Path(handle.name).unlink()
        self.path = Path(handle.name)
        self.addCleanup(lambda: self.path.unlink(missing_ok=True))

    def test_round_trip_survives_a_new_process(self):
        first, _ = build(self.path)
        before = first.batch_search(["one", "two"], return_score=True)
        self.assertEqual(len(self.path.read_text().splitlines()), 2)
        reopened, reopened_inner = build(self.path)
        self.assertEqual(reopened.batch_search(["one", "two"], return_score=True), before)
        self.assertEqual(reopened_inner.batch_calls, [])

    def test_one_line_per_query_with_the_key_the_documents_and_the_scores(self):
        proxy, _ = build(self.path)
        proxy.batch_search(["one"], num=3)
        record = json.loads(self.path.read_text().splitlines()[0])
        self.assertEqual((record["index"], record["encoder"], record["top_k"]), ("idx", "e5", 3))
        self.assertEqual(record["query"], "one")
        self.assertEqual(record["scores"], [1.0, 0.5, 0.0])
        self.assertEqual([d["id"] for d in record["docs"]], ["one::0", "one::1", "one::2"])

    def test_nothing_is_written_when_the_inner_call_fails(self):
        class Broken:
            def search(self, query, **kwargs):
                raise AssertionError("unused")

            def batch_search(self, queries, **kwargs):
                raise RuntimeError("index unavailable")
        proxy, _ = build(self.path, Broken())
        with self.assertRaisesRegex(RuntimeError, "index unavailable"):
            proxy.search("one")
        self.assertFalse(self.path.exists())

    def test_an_unscored_entry_is_not_served_as_scored(self):
        build(self.path, UnscoredRetriever())[0].search("one")
        scored, _ = build(self.path)
        with self.assertRaisesRegex(ValueError, "stored without scores"):
            scored.search("one", return_score=True)
        self.assertEqual(len(scored.search("one")), DEFAULT_DOCS)


class TornTailTests(unittest.TestCase):
    def setUp(self):
        handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        handle.close()
        Path(handle.name).unlink()
        self.path = Path(handle.name)
        self.addCleanup(lambda: self.path.unlink(missing_ok=True))

    def test_a_torn_last_line_is_dropped_with_a_warning(self):
        build(self.path)[0].search("one")
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write('{"index": "idx", "encoder": "e5", "top_k": 3, "que')
        warned = []
        reopened, inner = build(self.path, warn=warned.append)
        self.assertEqual(len(warned), 1)
        self.assertIn("torn", warned[0])
        self.assertEqual(reopened.search("one")[0]["id"], "one::0")
        self.assertEqual(inner.batch_calls, [])

    def test_the_next_record_is_not_glued_onto_a_torn_tail(self):
        build(self.path)[0].search("one")
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write('{"index": "idx", "encoder": "e5", "top_k": 3, "que')
        write, _ = build(self.path, warn=lambda message: None)
        write.search("two")
        final, inner = build(self.path, warn=lambda message: None)
        self.assertEqual(final.search("one")[0]["id"], "one::0")
        self.assertEqual(final.search("two")[0]["id"], "two::0")
        self.assertEqual(inner.batch_calls, [])

    def test_a_corrupt_line_in_the_middle_is_skipped_and_the_rest_survives(self):
        proxy, _ = build(self.path)
        proxy.search("one")
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write("not json\n")
        proxy.search("two")
        warned = []
        reopened, inner = build(self.path, warn=warned.append)
        self.assertEqual(len(warned), 1)
        self.assertIn("line 2", warned[0])
        self.assertEqual(reopened.search("one")[0]["id"], "one::0")
        self.assertEqual(reopened.search("two")[0]["id"], "two::0")
        self.assertEqual(inner.batch_calls, [])

    def test_invalid_utf8_is_skipped_instead_of_crashing(self):
        with open(self.path, "wb") as handle:
            handle.write(b'{"index": "idx", "encoder": "e5", "top_k": 3, "query": "\xff"}\n')
        warned = []
        reopened, inner = build(self.path, warn=warned.append)
        self.assertEqual(len(warned), 1)
        self.assertEqual(reopened.search("one")[0]["id"], "one::0")
        self.assertEqual(len(inner.batch_calls), 1)

    def test_a_line_separator_inside_a_document_stays_on_one_line(self):
        class Unusual:
            def search(self, query, **kwargs):
                return []

            def batch_search(self, queries, return_score=False):
                docs = [[{"id": "x", "contents": "before\u2028after"}] for _ in queries]
                return (docs, [[0.5] for _ in queries]) if return_score else docs
        build(self.path, Unusual())[0].search("one")
        self.assertEqual(len(self.path.read_text().split("\n")) - 1, 1)
        reopened, _ = build(self.path, warn=lambda message: self.fail(message))
        self.assertEqual(reopened.search("one")[0]["contents"], "before\u2028after")

    def test_malformed_records_are_skipped_with_a_warning(self):
        good = {"index": "idx", "encoder": "e5", "top_k": 3, "query": "good",
                "docs": [{"id": "g"}], "scores": [1.0]}
        bad_scores = dict(good, query="bad_scores", scores="not a list")
        bad_docs = dict(good, query="bad_docs", docs="not a list")
        missing = {"index": "idx", "encoder": "e5", "top_k": 3, "query": "missing"}
        self.path.write_text("\n".join(json.dumps(record) for record in
                                       (good, bad_scores, bad_docs, missing)) + "\n")
        warned = []
        reopened, _ = build(self.path, warn=warned.append)
        self.assertEqual(len(warned), 3)
        self.assertEqual(reopened.search("good")[0]["id"], "g")
        self.assertIsNone(reopened.store.get(CacheKey("idx", "e5", 3, "bad_scores")))
        self.assertIsNone(reopened.store.get(CacheKey("idx", "e5", 3, "bad_docs")))


if __name__ == "__main__":
    unittest.main()
