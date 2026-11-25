"""J24: label log-probs from string tokens, the fake scorer and the Ollama scorer."""
import math
import unittest

import frozen  # noqa: F401  (puts improvements/ and the test helpers on sys.path)
from drag_plus.judge.logprob import LABELS, margin
from drag_plus.runner.fake import FakeGenerator
from drag_plus.runner.label_scores import fake_label_logprobs, first_position, label_logprobs
from drag_plus.runner.local_generator import OllamaGenerator
from fake_ollama import FakeOllama

SUFFICIENCY = ("sufficient", "insufficient")


def lse(*values):
    return math.log(sum(math.exp(value) for value in values))


def scored_body(content="Opponent.", finish="stop", top=None):
    top = top if top is not None else [
        {"token": "Opp", "logprob": -0.39}, {"token": "Pro", "logprob": -1.23},
        {"token": "Op", "logprob": -5.19}, {"token": "I", "logprob": -7.08},
        {"token": " Opp", "logprob": -8.22}, {"token": "Prop", "logprob": -9.17}]
    return {"choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": content},
                         "logprobs": {"content": [{"token": "Opp", "logprob": -0.39,
                                                   "bytes": [79, 112, 112],
                                                   "top_logprobs": top}]}}]}


class LabelMatchingTests(unittest.TestCase):
    def test_every_prefix_variant_is_summed_in_probability(self):
        scores = label_logprobs([("Opp", -0.39), ("Op", -5.19), (" Opp", -8.22),
                                 ("Pro", -1.23), ("Prop", -9.17)], LABELS)
        self.assertAlmostEqual(scores["Opponent"], lse(-0.39, -5.19, -8.22))
        self.assertAlmostEqual(scores["Proponent"], lse(-1.23, -9.17))

    def test_matching_ignores_case_and_surrounding_space(self):
        scores = label_logprobs([("OPP", -1.0), ("  pro ", -2.0)], LABELS)
        self.assertEqual(scores, {"Proponent": -2.0, "Opponent": -1.0})

    def test_one_character_tokens_never_count(self):
        scores = label_logprobs([("I", -0.1), ("s", -3.0), ("O", -0.2)], SUFFICIENCY)
        self.assertEqual(scores, {"sufficient": None, "insufficient": None})

    def test_a_token_that_prefixes_two_labels_counts_for_neither(self):
        scores = label_logprobs([("Ye", -0.5), ("Yes", -1.0)], ("Yes", "Yesterday"))
        self.assertIsNone(scores["Yes"])
        self.assertIsNone(scores["Yesterday"])

    def test_a_label_with_no_token_is_none_and_the_margin_is_undefined(self):
        scores = label_logprobs([("Opp", -0.4), ("Neither", -4.0)], LABELS)
        self.assertIsNone(scores["Proponent"])
        self.assertIsNone(margin(scores))

    def test_non_finite_non_numeric_and_repeated_entries_are_skipped(self):
        scores = label_logprobs([("Opp", float("nan")), ("Opp", -1.0), ("Pro", "x"),
                                 ("Pro", True), (None, -1.0), ("Pro", -2.0),
                                 ("Pro", -0.1)], LABELS)
        self.assertEqual(scores, {"Proponent": -2.0, "Opponent": -1.0})

    def test_suffix_tokens_and_longer_tokens_do_not_match(self):
        scores = label_logprobs([("ponent", -1.0), ("Opponents", -1.0)], LABELS)
        self.assertEqual(scores, {"Proponent": None, "Opponent": None})

    def test_first_position_reads_the_openai_shape_and_tolerates_others(self):
        pairs = first_position(scored_body()["choices"][0])
        self.assertEqual(pairs[0], ("Opp", -0.39))
        self.assertEqual(len(pairs), 6)
        no_top = {"logprobs": {"content": [{"token": "Pro", "logprob": -0.2}]}}
        self.assertEqual(first_position(no_top), [("Pro", -0.2)])
        for broken in ({}, {"logprobs": None}, {"logprobs": {"content": []}}, None, "x"):
            self.assertEqual(first_position(broken), [])


class FakeScorerTests(unittest.TestCase):
    def test_fake_scores_are_finite_deterministic_and_prompt_dependent(self):
        first = fake_label_logprobs("prompt one", LABELS)
        self.assertEqual(first, fake_label_logprobs("prompt one", LABELS))
        self.assertTrue(all(math.isfinite(value) and value < 0 for value in first.values()))
        many = {tuple(fake_label_logprobs(f"p{i}", LABELS).values()) for i in range(20)}
        self.assertGreater(len(many), 1)

    def test_fake_generator_scores_each_prompt_and_names_the_best_label(self):
        prompts = [[{"role": "user", "content": f"judge {i}"}] for i in range(3)]
        outputs = FakeGenerator().generate_with_scores(prompts, labels=LABELS)
        self.assertEqual(len(outputs), 3)
        for output in outputs:
            best = max(LABELS, key=lambda label: output.label_logprobs[label])
            self.assertEqual(output.text, best)
            self.assertIsNotNone(margin(output.label_logprobs))
        single = FakeGenerator().generate_with_scores(prompts[0], labels=LABELS)
        self.assertEqual(single, outputs[:1])


class OllamaScorerTests(unittest.TestCase):
    def test_the_scored_request_asks_for_log_probs_and_a_short_answer(self):
        with FakeOllama(script=[("raw", scored_body())]) as server:
            generator = OllamaGenerator("tiny:1b", server.base_url, retries=0)
            outputs = generator.generate_with_scores(
                [{"role": "user", "content": "Who won?"}], labels=LABELS)
            payload = server.requests[-1][1]
        self.assertTrue(payload["logprobs"])
        self.assertEqual(payload["top_logprobs"], 20)
        self.assertEqual(payload["max_tokens"], 8)
        self.assertEqual(outputs[0].text, "Opponent.")
        self.assertAlmostEqual(outputs[0].label_logprobs["Opponent"], lse(-0.39, -5.19, -8.22))
        self.assertAlmostEqual(outputs[0].label_logprobs["Proponent"], lse(-1.23, -9.17))
        self.assertEqual(generator.last_finish_reasons, ["stop"])

    def test_a_reply_without_log_probs_scores_none_and_keeps_the_text(self):
        with FakeOllama(script=[("text", "Proponent")]) as server:
            generator = OllamaGenerator("tiny:1b", server.base_url, retries=0)
            output = generator.generate_with_scores("Who won?", labels=LABELS)[0]
        self.assertEqual(output.text, "Proponent")
        self.assertEqual(output.label_logprobs, {"Proponent": None, "Opponent": None})

    def test_an_explicit_max_tokens_wins_and_reasons_follow_each_prompt(self):
        script = [("raw", scored_body(finish="length")), ("raw", scored_body())]
        with FakeOllama(script=script) as server:
            generator = OllamaGenerator("tiny:1b", server.base_url, retries=0)
            prompts = [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]
            outputs = generator.generate_with_scores(prompts, labels=LABELS, max_tokens=3)
            sent = [body["max_tokens"] for _, body in server.requests]
        self.assertEqual(len(outputs), 2)
        self.assertEqual(sent, [3, 3])
        self.assertEqual(generator.last_finish_reasons, ["length", "stop"])
        self.assertEqual(generator.stats.finish_reasons, {"length": 1, "stop": 1})

    def test_plain_generation_also_reports_its_finish_reasons(self):
        with FakeOllama(script=[("text", "one"), ("empty",)]) as server:
            generator = OllamaGenerator("tiny:1b", server.base_url, retries=0)
            self.assertIsNone(generator.last_finish_reasons)
            texts = generator.generate(["a", "b"])
            payload = server.requests[-1][1]
        self.assertEqual(texts, ["one", ""])
        self.assertEqual(generator.last_finish_reasons, ["stop", "length"])
        self.assertNotIn("logprobs", payload)


if __name__ == "__main__":
    unittest.main()
