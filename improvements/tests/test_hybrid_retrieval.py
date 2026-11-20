"""Weighted RRF, score floors, the per-title cap and the cache fingerprint."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drag_plus.flags import Flags  # noqa: E402
from drag_plus.retrieval.hybrid import (Source, cap_per_title, config_fingerprint,  # noqa: E402
                                        reciprocal_rank_fusion, retrieval_config,
                                        source_results)


def doc(doc_id, title="T", body="body"):
    return {"id": doc_id, "contents": f"{title}\n{body}"}


class RrfTests(unittest.TestCase):
    def test_score_is_the_weighted_sum_of_reciprocal_ranks(self):
        first = Source("dense", [doc("a"), doc("b")])
        second = Source("bm25", [doc("b"), doc("c")])
        fused = reciprocal_rank_fusion([first, second], k=60)
        scores = {doc["id"]: round(score, 9) for doc, score in fused}
        self.assertEqual(scores["a"], round(1 / 61, 9))
        self.assertEqual(scores["b"], round(1 / 62 + 1 / 61, 9))
        self.assertEqual(scores["c"], round(1 / 62, 9))
        self.assertEqual([d["id"] for d, _ in fused], ["b", "a", "c"])

    def test_k_is_the_rank_constant_and_must_be_positive(self):
        one = Source("dense", [doc("a")])
        self.assertAlmostEqual(reciprocal_rank_fusion([one], k=10)[0][1], 1 / 11)
        with self.assertRaisesRegex(ValueError, "k must be >= 1"):
            reciprocal_rank_fusion([one], k=0)

    def test_weights_scale_a_source_and_zero_drops_it(self):
        dense = Source("dense", [doc("a")], weight=2.0)
        bm25 = Source("bm25", [doc("b")], weight=0.0)
        fused = reciprocal_rank_fusion([dense, bm25], k=60)
        self.assertEqual([d["id"] for d, _ in fused], ["a"])
        self.assertAlmostEqual(fused[0][1], 2 / 61)
        self.assertEqual(reciprocal_rank_fusion([dense, Source("x", [doc("b")], weight=-1)]),
                         reciprocal_rank_fusion([dense]))

    def test_a_source_shorter_than_k_contributes_the_ranks_it_has(self):
        fused = reciprocal_rank_fusion([Source("dense", [doc("only")]),
                                        Source("bm25", [])], k=60)
        self.assertEqual([d["id"] for d, _ in fused], ["only"])

    def test_ties_break_on_the_document_id(self):
        fused = reciprocal_rank_fusion([Source("dense", [doc("z"), doc("a")])], k=60)
        self.assertEqual([d["id"] for d, _ in fused], ["z", "a"])
        both = reciprocal_rank_fusion([Source("one", [doc("b")]),
                                       Source("two", [doc("a")])], k=60)
        self.assertEqual([d["id"] for d, _ in both], ["a", "b"])

    def test_a_repeated_document_inside_one_source_keeps_its_first_rank(self):
        fused = reciprocal_rank_fusion([Source("dense", [doc("a"), doc("b"), doc("a")])])
        self.assertEqual([d["id"] for d, _ in fused], ["a", "b"])
        self.assertAlmostEqual(fused[0][1], 1 / 61)

    def test_the_document_body_comes_from_the_first_source_that_ranked_it(self):
        fused = reciprocal_rank_fusion([Source("dense", [doc("a", body="dense")]),
                                        Source("bm25", [doc("a", body="bm25")])])
        self.assertEqual(fused[0][0]["contents"], "T\ndense")


class FloorTests(unittest.TestCase):
    def test_a_floor_drops_candidates_below_it_per_source(self):
        dense = Source("dense", [doc("a"), doc("b")], scores=[0.9, 0.2], floor=0.5)
        bm25 = Source("bm25", [doc("c"), doc("d")], scores=[12.0, 3.0], floor=5.0)
        fused = reciprocal_rank_fusion([dense, bm25])
        self.assertEqual([d["id"] for d, _ in fused], ["a", "c"])
        self.assertAlmostEqual(fused[1][1], 1 / 61)

    def test_a_floor_needs_scores_of_the_same_length(self):
        with self.assertRaisesRegex(ValueError, "no scores"):
            reciprocal_rank_fusion([Source("dense", [doc("a")], floor=0.5)])
        with self.assertRaisesRegex(ValueError, "scores"):
            reciprocal_rank_fusion([Source("dense", [doc("a")], scores=[], floor=0.5)])

    def test_missing_scores_are_never_silently_admitted(self):
        fused = reciprocal_rank_fusion([Source("dense", [doc("a")], scores=[None], floor=0.5)])
        self.assertEqual(fused, [])


class TitleCapTests(unittest.TestCase):
    def test_cap_keeps_the_first_n_passages_per_title(self):
        ranked = [(doc("a", "One"), 3.0), (doc("b", "One"), 2.0),
                  (doc("c", "Two"), 1.0), (doc("d", "One"), 0.5)]
        self.assertEqual([d["id"] for d, _ in cap_per_title(ranked, 2)], ["a", "b", "c"])
        self.assertEqual([d["id"] for d, _ in cap_per_title(ranked, 1)], ["a", "c"])

    def test_null_cap_is_a_no_op_and_a_negative_cap_is_an_error(self):
        ranked = [(doc("a", "One"), 1.0)]
        self.assertEqual(cap_per_title(ranked, None), ranked)
        with self.assertRaisesRegex(ValueError, "per_title_cap"):
            cap_per_title(ranked, 0)


class Recorder:
    def __init__(self, scores=False, num=True):
        self.calls, self.scored, self.numbered = [], scores, num

    def batch_search(self, queries, **kwargs):
        self.calls.append((list(queries), dict(kwargs)))
        docs = [[doc(f"{query}-1"), doc(f"{query}-2"), doc(f"{query}-3")]
                for query in queries]
        if not kwargs.get("return_score"):
            return docs
        return docs, [[0.9, 0.5, 0.1] for _ in queries]


class SourceResultTests(unittest.TestCase):
    def test_scores_are_only_requested_when_needed(self):
        plain, scored = Recorder(), Recorder()
        source_results(plain, ["q"], 2, False)
        source_results(scored, ["q"], 2, True)
        self.assertEqual(plain.calls[0][1], {"num": 2})
        self.assertEqual(scored.calls[0][1], {"num": 2, "return_score": True})

    def test_depth_truncates_documents_and_scores_together(self):
        docs, scores = source_results(Recorder(), ["q"], 2, True)[0]
        self.assertEqual([d["id"] for d in docs], ["q-1", "q-2"])
        self.assertEqual(scores, [0.9, 0.5])

    def test_a_retriever_without_num_still_works(self):
        docs, _ = source_results(Recorder(num=False), ["q"], 1, False)[0]
        self.assertEqual([d["id"] for d in docs], ["q-1"])

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


class ConfigFingerprintTests(unittest.TestCase):
    def test_dense_and_hybrid_fingerprints_differ(self):
        dense = config_fingerprint(Flags())
        hybrid = config_fingerprint(Flags({"retriever": "hybrid"}))
        self.assertNotEqual(dense, hybrid)
        self.assertEqual(retrieval_config(Flags()), {"retriever": "e5"})

    def test_every_retrieval_setting_reaches_the_fingerprint(self):
        flags = Flags({"retriever": "hybrid", "dense_top_k": 11, "bm25_top_k": 12,
                       "hybrid_top_k": 4, "rrf_k": 61, "dense_weight": 0.5,
                       "bm25_weight": 2.0, "dense_score_floor": 0.3,
                       "bm25_score_floor": 1.5, "bm25_index": "/idx",
                       "rerank_k": 9, "rerank_model": "m", "per_title_cap": 2})
        config = retrieval_config(flags)
        self.assertEqual((config["dense_top_k"], config["bm25_top_k"]), (11, 12))
        self.assertEqual(config_fingerprint(flags),
                         config_fingerprint(Flags(config)))
        changed = flags.with_changes(rerank_model="other")
        self.assertNotEqual(config_fingerprint(flags), config_fingerprint(changed))


if __name__ == "__main__":
    unittest.main()
