"""J18 vote and self-consistency baselines, on scripted outputs."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drag_plus.aggregators import Candidate, select_answer  # noqa: E402
from drag_plus.flags import Flags  # noqa: E402
from drag_plus.pipeline import DragPlusPipeline  # noqa: E402
from fakes import FakeRetriever, IdentityTemplate, RecordingItem, ScriptedGenerator  # noqa: E402

PARIS = [{"id": "d1", "contents": "Paris\nParis is the capital of France."}]
LONDON = [{"id": "d2", "contents": "London\nLondon is the capital of England."}]


def _run(outputs, docs=None, **flags):
    values = {"max_query_debate_rounds": 0, "max_answer_debate_rounds": 1}
    values.update(flags)
    generator = ScriptedGenerator(outputs)
    pipeline = DragPlusPipeline(Flags(values), generator,
                                FakeRetriever({} if docs is None else docs),
                                IdentityTemplate(), "NQ")
    item = RecordingItem("q")
    return pipeline.run_item(item), generator, item


def _result(state):
    return dict(state.outputs)["AnswerStage_Aggregator_Result"]


class SelectAnswerTests(unittest.TestCase):
    def test_plurality_wins_before_any_tie_break(self):
        candidates = [Candidate("a", "Paris", 1), Candidate("b", "London", 1),
                      Candidate("c", "Paris", 1)]
        self.assertEqual(select_answer(candidates, "London"), ("Paris", "plurality"))

    def test_tie_prefers_the_answer_the_evidence_contains(self):
        candidates = [Candidate("a", "Paris", 1), Candidate("b", "London", 1)]
        self.assertEqual(select_answer(candidates, "London is in England"),
                         ("London", "evidence"))

    def test_tie_without_evidence_uses_the_fallback(self):
        candidates = [Candidate("a", "Paris", 1), Candidate("b", "London", 1)]
        self.assertEqual(select_answer(candidates, "no place names", "London"),
                         ("London", "fallback"))

    def test_empty_answers_are_an_empty_result(self):
        candidates = [Candidate("a", "", 2), Candidate("b", "", 3)]
        self.assertEqual(select_answer(candidates, "anything"), ("", "empty"))


class VoteTests(unittest.TestCase):
    def test_vote_uses_evidence_to_break_a_tie(self):
        state, generator, _ = _run(
            ["The answer is: Paris", "The answer is: London"],
            docs={"q": PARIS}, aggregator="vote")
        self.assertEqual(state.pred, "Paris")
        self.assertEqual(len(generator.calls), 2)
        self.assertTrue(all("do_sample" not in params for _, params in generator.calls))
        result = _result(state)
        self.assertEqual(result["tie_break"], "evidence")
        self.assertEqual(result["generated_tokens"], 8)

    def test_vote_falls_back_to_the_proponent_without_evidence(self):
        state, _, _ = _run(["The answer is: Paris", "The answer is: London"],
                           aggregator="vote")
        self.assertEqual(state.pred, "Paris")
        self.assertEqual(_result(state)["tie_break"], "fallback")

    def test_agreement_is_a_plain_plurality(self):
        state, _, _ = _run(["The answer is: **Paris**.", "The answer is: Paris"],
                           aggregator="vote")
        self.assertEqual(state.pred, "Paris")
        self.assertEqual(_result(state)["tie_break"], "plurality")


class SelfConsistencyTests(unittest.TestCase):
    def test_samples_at_temperature_and_votes_with_e_normalisation(self):
        outputs = ["The answer is: Paris", "The answer is: London",
                   "The answer is: **Paris**."]
        state, generator, _ = _run(
            outputs, docs={"q": PARIS}, aggregator="self_consistency",
            self_consistency_samples=3, self_consistency_temperature=0.8,
            agent_max_tokens=64)
        self.assertEqual(state.pred, "Paris")
        prompts, params = generator.calls[0]
        self.assertEqual(len(prompts), 3)
        self.assertTrue(params["do_sample"])
        self.assertEqual(params["temperature"], 0.8)
        self.assertEqual(params["max_tokens"], 64)
        result = _result(state)
        self.assertEqual(result["aggregator"], "self_consistency")
        self.assertEqual(result["tie_break"], "plurality")
        self.assertEqual(result["generated_tokens"],
                         sum(len(output.split()) for output in outputs))

    def test_tie_prefers_the_evidence(self):
        state, _, _ = _run(
            ["The answer is: Paris", "The answer is: London", "The answer is: Berlin"],
            docs={"q": LONDON}, aggregator="self_consistency",
            self_consistency_samples=3)
        self.assertEqual(state.pred, "London")
        self.assertEqual(_result(state)["tie_break"], "evidence")

    def test_tie_without_evidence_uses_the_first_sample(self):
        state, _, _ = _run(
            ["The answer is: Paris", "The answer is: London", "The answer is: Berlin"],
            aggregator="self_consistency", self_consistency_samples=3)
        self.assertEqual(state.pred, "Paris")
        self.assertEqual(_result(state)["tie_break"], "fallback")

    def test_default_moderator_sets_no_sampling_parameters(self):
        _, generator, _ = _run(
            ["proponent\nanswer is Paris", "opponent\nanswer is London",
             "The answer is: Paris"])
        self.assertTrue(all("do_sample" not in params for _, params in generator.calls))


if __name__ == "__main__":
    unittest.main()
