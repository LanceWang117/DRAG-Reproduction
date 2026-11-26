"""J30: search_via_batch routes single-query search through batch_search (B2 fix)."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import frozen  # noqa: F401  (puts improvements/ and the test helpers on sys.path)
from drag_plus.flags import Flags
from drag_plus.retrieval.batch_search import BatchSearchRetriever, with_batch_search
from drag_plus.runner.backends import build_fake_backend
from drag_plus.runner.run import run_variant
from drag_plus.runner.variant import parse_variant
from test_runner_fake import make_request, read_lines

ENCODED = 64 - len("query: ")  # FlashRAG's string slice keeps 64 characters with the prefix
ROWS = [{"dataset": "NQ", "split": "dev", "id": "q0", "golden_answers": ["answer 1"],
         "question": "which british biscuit maker won a 1991 vat tribunal arguing that its "
                     "orange sponge snack is legally a cake"},
        {"dataset": "NQ", "split": "dev", "id": "q1", "golden_answers": ["answer 1"],
         "question": "who wrote hamlet"}]


def docs_for(text):
    return [{"id": f"doc::{text}", "contents": f"Title\nAbout {text}"}]


class TruncatingRetriever:
    """FlashRAG 0.1.4's behaviour: search embeds the first characters, batch_search all."""

    def __init__(self):
        self.calls = []

    def search(self, query, num=None, return_score=False):
        self.calls.append(("search", query))
        return docs_for(query[:ENCODED])

    def batch_search(self, queries, num=None, return_score=False):
        self.calls.append(("batch_search", list(queries), num, return_score))
        rows = [docs_for(query)[: num or 5] for query in queries]
        return (rows, [[0.9] * len(r) for r in rows]) if return_score else rows


class WrapperTests(unittest.TestCase):
    def test_search_is_exactly_the_single_element_batch(self):
        inner = TruncatingRetriever()
        wrapped = BatchSearchRetriever(inner)
        long_query = ROWS[0]["question"]
        self.assertEqual(wrapped.search(long_query), docs_for(long_query))
        self.assertNotEqual(inner.search(long_query), docs_for(long_query))  # the bug itself
        self.assertEqual(inner.calls[0], ("batch_search", [long_query], None, False))

    def test_num_and_scores_pass_through(self):
        inner = TruncatingRetriever()
        docs, scores = BatchSearchRetriever(inner).search("q", num=1, return_score=True)
        self.assertEqual((docs, scores), (docs_for("q"), [0.9]))
        self.assertEqual(inner.calls[-1], ("batch_search", ["q"], 1, True))

    def test_a_retriever_without_optional_parameters_still_works(self):
        class Plain:
            def batch_search(self, queries):
                return [docs_for(q) for q in queries]
            index = "kept"
        wrapped = BatchSearchRetriever(Plain())
        self.assertEqual(wrapped.search("q"), docs_for("q"))
        self.assertEqual(wrapped.index, "kept")  # other attributes delegate

    def test_the_flag_is_off_by_default_and_wraps_only_when_set(self):
        inner = TruncatingRetriever()
        self.assertFalse(Flags().search_via_batch)
        self.assertIs(with_batch_search(inner, Flags()), inner)
        self.assertIsInstance(with_batch_search(inner, Flags({"search_via_batch": True})),
                              BatchSearchRetriever)


class EndToEndTests(unittest.TestCase):
    def run_with(self, flags):
        backend = replace(build_fake_backend(), retriever=TruncatingRetriever())
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "run"
            variant = parse_variant({"name": "b2", "flags": flags, "arms": ["drag"]})
            run_variant(make_request(out, backend=backend, variant=variant, rows=ROWS))
            transcripts = read_lines(out / "transcripts.jsonl")
        pools = {}
        for row in transcripts:
            pool = next(e["value"] for e in row["entries"] if e["key"] == "QueryStage_QueryPool")
            pools[row["id"]] = json.dumps(pool)
        return pools, backend.retriever.calls

    def test_the_fix_retrieves_on_the_whole_long_question(self):
        fixed, calls = self.run_with({"search_via_batch": True})
        self.assertIn(f"doc::{ROWS[0]['question']}\"", fixed["q0"])
        self.assertFalse([c for c in calls if c[0] == "search"])

    def test_the_default_keeps_the_recorded_truncated_path(self):
        baseline, calls = self.run_with({})
        self.assertIn(f"doc::{ROWS[0]['question'][:ENCODED]}\"", baseline["q0"])
        self.assertTrue([c for c in calls if c[0] == "search"])

    def test_short_questions_are_unchanged_by_the_fix(self):
        fixed, _ = self.run_with({"search_via_batch": True})
        baseline, _ = self.run_with({})
        self.assertEqual(fixed["q1"], baseline["q1"])


if __name__ == "__main__":
    unittest.main()
