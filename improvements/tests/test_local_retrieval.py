"""The replay retriever: stored passages for stored queries, token overlap otherwise."""
import json
from pathlib import Path
import tempfile
import unittest

import frozen  # noqa: F401
from drag_plus.runner.local_retrieval import ReplayRetriever, load_replay

PROJECT = Path(__file__).resolve().parents[2]


def doc(identifier, text):
    return {"id": identifier, "contents": f"Title {identifier}\n{text}"}


STORED = {
    "who wrote hamlet": [doc("h1", "Shakespeare wrote the play Hamlet"), doc("h2", "Denmark castle")],
    "capital of france": [doc("f1", "Paris is the capital of France"), doc("f2", "river Seine")],
}


class StoredQueryTests(unittest.TestCase):
    def setUp(self):
        self.retriever = ReplayRetriever(STORED, top_k=2)

    def test_a_stored_query_returns_its_stored_passages(self):
        self.assertEqual(self.retriever.search("who wrote hamlet"), STORED["who wrote hamlet"])

    def test_the_query_is_normalised_like_the_retrieval_cache(self):
        self.assertEqual(self.retriever.search("  who   wrote\thamlet "), STORED["who wrote hamlet"])

    def test_case_is_preserved_so_a_different_case_is_a_different_query(self):
        self.retriever.search("Who wrote Hamlet")
        self.assertEqual(self.retriever.stats, {"replayed": 0, "lexical": 1})

    def test_results_are_copies(self):
        self.retriever.search("who wrote hamlet")[0]["contents"] = "changed"
        self.assertEqual(self.retriever.search("who wrote hamlet"), STORED["who wrote hamlet"])

    def test_batch_search_keeps_input_order(self):
        result = self.retriever.batch_search(["capital of france", "who wrote hamlet"])
        self.assertEqual(result, [STORED["capital of france"], STORED["who wrote hamlet"]])


class LexicalFallbackTests(unittest.TestCase):
    def setUp(self):
        self.retriever = ReplayRetriever(STORED, top_k=2)

    def test_an_unseen_query_ranks_every_stored_passage_by_token_overlap(self):
        ids = [d["id"] for d in self.retriever.search("which castle is in Denmark")]
        self.assertEqual(ids, ["h2", "h1"])

    def test_the_best_overlap_comes_first(self):
        ids = [d["id"] for d in self.retriever.search("Paris France capital city")]
        self.assertEqual(ids[0], "f1")

    def test_ties_keep_stored_order_so_the_result_is_deterministic(self):
        first = self.retriever.search("zzz unmatched words")
        self.assertEqual([d["id"] for d in first], ["h1", "h2"])
        self.assertEqual(self.retriever.search("zzz unmatched words"), first)

    def test_stop_words_alone_do_not_decide_the_ranking(self):
        ids = [d["id"] for d in self.retriever.search("the of is Seine")]
        self.assertEqual(ids[0], "f2")

    def test_a_passage_stored_under_two_queries_is_ranked_once(self):
        shared = {"a": [doc("x", "alpha beta")], "b": [doc("x", "alpha beta"), doc("y", "gamma")]}
        ids = [d["id"] for d in ReplayRetriever(shared, top_k=5).search("alpha")]
        self.assertEqual(ids, ["x", "y"])


class LoadReplayTests(unittest.TestCase):
    def write(self, folder, retrieval, questions):
        retrieval_path, questions_path = Path(folder) / "r.jsonl", Path(folder) / "q.jsonl"
        retrieval_path.write_text("\n".join(json.dumps(r) for r in retrieval) + "\n")
        questions_path.write_text("\n".join(json.dumps(q) for q in questions) + "\n")
        return retrieval_path, questions_path

    def rows(self):
        retrieval = [{"dataset": "NQ", "split": "dev", "id": "q1",
                      "drag": {"rewritten query": [doc("d1", "one")]}, "naive": [doc("n1", "two")]},
                     {"dataset": "NQ", "split": "dev", "id": "q2",
                      "drag": {"second question": [doc("d2", "three")]}, "naive": [doc("n2", "four")]}]
        questions = [{"dataset": "NQ", "split": "dev", "id": "q1", "question": "first question"},
                     {"dataset": "NQ", "split": "dev", "id": "q2", "question": "second question"}]
        return retrieval, questions

    def test_the_question_maps_to_the_naive_passages_when_the_drag_map_lacks_it(self):
        with tempfile.TemporaryDirectory() as folder:
            retriever = load_replay(*self.write(folder, *self.rows()), top_k=3)
        self.assertEqual(retriever.search("first question")[0]["id"], "n1")
        self.assertEqual(retriever.search("rewritten query")[0]["id"], "d1")

    def test_a_stored_drag_query_wins_over_the_naive_list_for_the_same_text(self):
        with tempfile.TemporaryDirectory() as folder:
            retriever = load_replay(*self.write(folder, *self.rows()), top_k=3)
        self.assertEqual(retriever.search("second question")[0]["id"], "d2")

    def test_a_row_without_a_known_question_still_loads(self):
        retrieval, _ = self.rows()
        with tempfile.TemporaryDirectory() as folder:
            retriever = load_replay(*self.write(folder, retrieval, []), top_k=3)
        self.assertEqual(retriever.search("rewritten query")[0]["id"], "d1")

    def test_every_recorded_question_replays_stored_passages(self):
        results = PROJECT / "results"
        retriever = load_replay(results / "retrieval-200.jsonl", results / "questions-200.jsonl")
        questions = [json.loads(line) for line in (results / "questions-200.jsonl").read_text().splitlines()]
        for row in questions:
            self.assertEqual(len(retriever.search(row["question"])), 3, row["id"])
        self.assertEqual(retriever.stats, {"replayed": 200, "lexical": 0})


if __name__ == "__main__":
    unittest.main()
