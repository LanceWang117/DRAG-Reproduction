"""BM25 and cross-encoder adapters: lazy imports, injected fakes, clean errors."""
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

IMPROVEMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMPROVEMENTS))
from drag_plus.retrieval.bm25 import Bm25Retriever  # noqa: E402
from drag_plus.retrieval.rerank import CrossEncoderReranker  # noqa: E402

HEAVY = ("bm25s", "sentence_transformers", "torch", "transformers", "vllm", "faiss",
         "flashrag")


def passage(doc_id, title="T"):
    return {"id": doc_id, "contents": f"{title}\nbody {doc_id}"}


class FakeIndex:
    """A bm25s-shaped index: ``retrieve(tokens, k)`` -> (doc rows, score rows)."""

    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def retrieve(self, tokens, k):
        self.calls.append((list(tokens), k))
        row = list(self.rows[0][:k])
        return ([row for _ in tokens],
                [[1.0 / (rank + 1) for rank in range(len(row))] for _ in tokens])


class Bm25Tests(unittest.TestCase):
    def build(self, rows=None, **kwargs):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.index = FakeIndex(rows if rows is not None else
                               [[passage("a"), passage("b"), passage("c")]])
        self.loaded = []

        def load(path):
            self.loaded.append(path)
            return self.index
        return Bm25Retriever(Path(directory.name), load=load,
                             tokenize=lambda queries: [q.upper() for q in queries],
                             **kwargs)

    def test_loads_once_and_passes_tokenised_queries_to_retrieve(self):
        retriever = self.build()
        docs = retriever.search("first")
        retriever.search("second")
        self.assertEqual(self.loaded, [retriever.path])
        self.assertEqual(self.index.calls, [(["FIRST"], 20), (["SECOND"], 20)])
        self.assertEqual([d["id"] for d in docs], ["a", "b", "c"])

    def test_depth_comes_from_num_or_the_constructor(self):
        retriever = self.build(depth=2)
        self.assertEqual([d["id"] for d in retriever.search("q")], ["a", "b"])
        docs, scores = retriever.search("q", num=1, return_score=True)
        self.assertEqual([d["id"] for d in docs], ["a"])
        self.assertEqual(scores, [1.0])

    def test_batch_search_keeps_request_order(self):
        retriever = self.build()
        rows = retriever.batch_search(["one", "two"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(self.index.calls[0][0], ["ONE", "TWO"])
        with self.assertRaisesRegex(ValueError, ">= 1"):
            retriever.search("q", num=0)

    def test_a_missing_index_fails_before_loading_anything(self):
        loaded = []
        retriever = Bm25Retriever("/does/not/exist", load=lambda path: loaded.append(path))
        with self.assertRaisesRegex(FileNotFoundError, "not found"):
            retriever.search("q")
        self.assertEqual(loaded, [])

    def test_a_string_corpus_is_rejected_with_the_reason(self):
        retriever = self.build(rows=[["a passage", "another"]])
        with self.assertRaisesRegex(ValueError, "corpus of"):
            retriever.search("q")


class RerankTests(unittest.TestCase):
    def test_scorer_order_wins_and_ties_break_on_the_passage_id(self):
        docs = [passage("b"), passage("a"), passage("c")]
        scores = {"b": 0.2, "a": 0.2, "c": 0.9}
        reranker = CrossEncoderReranker(
            "fake/model", scorer=lambda pairs: [scores[doc["id"]] for doc in docs])
        ranked = reranker.rerank("q", docs)
        self.assertEqual([d["id"] for d, _ in ranked], ["c", "a", "b"])
        self.assertEqual([round(s, 3) for _, s in ranked], [0.9, 0.2, 0.2])

    def test_pairs_are_the_query_with_each_passages_contents(self):
        seen = []
        docs = [passage("a")]
        reranker = CrossEncoderReranker(
            "fake/model", scorer=lambda pairs: seen.append(list(pairs)) or [0.5])
        reranker.rerank("the query", docs)
        self.assertEqual(seen, [[("the query", "T\nbody a")]])
        self.assertEqual(reranker.rerank("q", []), [])

    def test_a_short_score_list_is_an_error_not_a_silent_zip(self):
        reranker = CrossEncoderReranker("fake/model", scorer=lambda pairs: [0.1])
        with self.assertRaisesRegex(ValueError, "scored 1 of 2"):
            reranker.rerank("q", [passage("a"), passage("b")])


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
from drag_plus.retrieval.bm25 import Bm25Retriever
from drag_plus.retrieval.rerank import CrossEncoderReranker

class Index:
    def retrieve(self, tokens, k):
        return ([[{"id": "a", "contents": "T\\nbody"}]], [[2.0]])

retriever = Bm25Retriever("/tmp", load=lambda path: Index(),
                          tokenize=lambda queries: queries)
docs, scores = retriever.search("q", return_score=True)
assert docs[0]["id"] == "a" and scores == [2.0]
ranked = CrossEncoderReranker("m", scorer=lambda pairs: [0.25]).rerank(
    "q", [{"id": "a", "contents": "T"}])
assert ranked[0][1] == 0.25
assert not any(name.split(".")[0] in HEAVY for name in sys.modules)
print("OK")
"""


class LazyImportTests(unittest.TestCase):
    def test_importing_and_using_the_adapters_loads_no_heavy_library(self):
        script = (BLOCK_SCRIPT.replace("__HEAVY__", repr(HEAVY))
                  .replace("__IMPROVEMENTS__", repr(str(IMPROVEMENTS))))
        done = subprocess.run([sys.executable, "-c", script],
                              capture_output=True, text=True, timeout=30)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("OK", done.stdout)


if __name__ == "__main__":
    unittest.main()
