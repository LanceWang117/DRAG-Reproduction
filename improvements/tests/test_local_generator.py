"""The Ollama generator against a fake HTTP server: success, failures, retries, order."""
import unittest

import frozen  # noqa: F401
from drag_plus.adapters import Generator
from drag_plus.runner import local_generator as lg
from fake_ollama import FakeOllama

PROMPT = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]


def make(server, **overrides):
    delays = []
    values = dict(timeout=0.3, sleep=delays.append)
    values.update(overrides)
    generator = lg.OllamaGenerator("tiny:1b", server.base_url, **values)
    generator.delays = delays
    return generator


class SuccessTests(unittest.TestCase):
    def test_a_message_list_goes_to_the_chat_endpoint_with_the_baseline_settings(self):
        with FakeOllama([("text", "The answer is: Paris")]) as server:
            output = make(server).generate(PROMPT)
        self.assertEqual(output, ["The answer is: Paris"])
        path, payload = server.requests[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(payload, {"model": "tiny:1b", "messages": PROMPT, "stream": False,
                                   "temperature": 0.0, "max_tokens": 256, "seed": 2024})

    def test_a_plain_string_becomes_one_user_message(self):
        with FakeOllama() as server:
            make(server).generate("hello")
        self.assertEqual(server.requests[0][1]["messages"],
                         [{"role": "user", "content": "hello"}])

    def test_a_batch_comes_back_in_input_order(self):
        prompts = [[{"role": "user", "content": f"question {i}"}] for i in range(4)]
        with FakeOllama(brain=lambda text: f"reply to {text}") as server:
            outputs = make(server).generate(prompts)
        self.assertEqual(outputs, [f"reply to question {i}" for i in range(4)])

    def test_it_satisfies_the_pipeline_generator_protocol(self):
        with FakeOllama() as server:
            self.assertIsInstance(make(server), Generator)

    def test_known_params_are_forwarded_and_unknown_ones_are_recorded(self):
        with FakeOllama() as server:
            generator = make(server)
            generator.generate(PROMPT, max_new_tokens=32, stop=["\n"], temperature=0.2,
                               seed=7, skip_special_tokens=True)
        payload = server.requests[0][1]
        self.assertEqual((payload["max_tokens"], payload["stop"], payload["temperature"],
                          payload["seed"]), (32, ["\n"], 0.2, 7))
        self.assertEqual(generator.stats.ignored_params, {"skip_special_tokens"})

    def test_the_finish_reason_is_counted(self):
        with FakeOllama([("empty",), ("text", "ok")]) as server:
            generator = make(server)
            generator.generate(PROMPT)
            generator.generate(PROMPT)
        self.assertEqual(generator.stats.finish_reasons, {"length": 1, "stop": 1})


class FailureTests(unittest.TestCase):
    def test_a_timeout_is_retried_twice_then_reported(self):
        with FakeOllama([("hang",)] * 3) as server:
            generator = make(server)
            with self.assertRaisesRegex(lg.LocalGeneratorError, "after 3 attempts"):
                generator.generate(PROMPT)
        self.assertEqual(len(server.requests), 3)
        self.assertEqual((generator.stats.retries, generator.delays), (2, [0.5, 1.0]))

    def test_a_timeout_then_success_recovers(self):
        with FakeOllama([("hang",), ("text", "fine")]) as server:
            generator = make(server)
            self.assertEqual(generator.generate(PROMPT), ["fine"])
        self.assertEqual(generator.stats.retries, 1)

    def test_http_500_is_retried_then_reported(self):
        with FakeOllama([("status", 500)] * 3) as server:
            generator = make(server)
            with self.assertRaisesRegex(lg.LocalGeneratorError, "HTTP 500"):
                generator.generate(PROMPT)
        self.assertEqual(len(server.requests), 3)

    def test_http_500_then_success_recovers(self):
        with FakeOllama([("status", 500), ("text", "fine")]) as server:
            self.assertEqual(make(server).generate(PROMPT), ["fine"])

    def test_a_missing_model_is_not_retried_and_says_how_to_pull_it(self):
        with FakeOllama([("status", 404)]) as server:
            with self.assertRaisesRegex(lg.ModelNotFoundError, "ollama pull tiny:1b"):
                make(server).generate(PROMPT)
        self.assertEqual(len(server.requests), 1)

    def test_another_client_error_is_not_retried(self):
        with FakeOllama([("status", 400)]) as server:
            with self.assertRaisesRegex(lg.LocalGeneratorError, "HTTP 400"):
                make(server).generate(PROMPT)
        self.assertEqual(len(server.requests), 1)

    def test_a_dropped_connection_is_retried_then_reported(self):
        with FakeOllama() as server:
            generator = make(server)
        with self.assertRaisesRegex(lg.LocalGeneratorError, "after 3 attempts"):
            generator.generate(PROMPT)
        self.assertEqual(generator.stats.retries, 2)

    def test_a_body_that_is_not_json_is_not_retried(self):
        with FakeOllama([("raw", b"<html>proxy</html>")]) as server:
            with self.assertRaisesRegex(lg.LocalGeneratorError, "not JSON"):
                make(server).generate(PROMPT)
        self.assertEqual(len(server.requests), 1)

    def test_an_unexpected_response_shape_is_reported(self):
        with FakeOllama([("raw", {"choices": []})]) as server:
            with self.assertRaisesRegex(lg.LocalGeneratorError, "unexpected response"):
                make(server).generate(PROMPT)

    def test_retries_can_be_turned_off(self):
        with FakeOllama([("status", 500)]) as server:
            with self.assertRaises(lg.LocalGeneratorError):
                make(server, retries=0).generate(PROMPT)
        self.assertEqual(len(server.requests), 1)

    def test_an_empty_completion_is_returned_and_counted_not_retried(self):
        with FakeOllama([("empty",)]) as server:
            generator = make(server)
            self.assertEqual(generator.generate(PROMPT), [""])
        self.assertEqual((len(server.requests), generator.stats.empty_outputs), (1, 1))


class ContextFitTests(unittest.TestCase):
    def test_a_prompt_inside_the_budget_is_untouched(self):
        fitted, trimmed = lg.fit_messages(PROMPT, 100)
        self.assertEqual((fitted, trimmed), (PROMPT, False))

    def test_the_longest_message_loses_its_middle_and_keeps_both_ends(self):
        messages = [{"role": "system", "content": "HEAD " + "x" * 3000 + " TAIL"},
                    {"role": "user", "content": "Question: why?"}]
        fitted, trimmed = lg.fit_messages(messages, 100)
        text = fitted[0]["content"]
        self.assertTrue(trimmed)
        self.assertTrue(text.startswith("HEAD "))
        self.assertTrue(text.endswith(" TAIL"))
        self.assertIn(lg.TRIM_MARK, text)
        self.assertEqual(fitted[1], messages[1])
        self.assertEqual(sum(len(m["content"]) for m in fitted), 100 * lg.CHARS_PER_TOKEN)
        self.assertEqual(messages[0]["content"][:5], "HEAD ")

    def test_the_generator_trims_before_sending_and_counts_it(self):
        long_prompt = [{"role": "user", "content": "w " * 6000}]
        with FakeOllama() as server:
            generator = make(server, context_tokens=1000, max_tokens=100)
            generator.generate(long_prompt)
        sent = server.requests[0][1]["messages"][0]["content"]
        self.assertLessEqual(len(sent), 900 * lg.CHARS_PER_TOKEN)
        self.assertEqual(generator.stats.trimmed_prompts, 1)

    def test_batches_become_message_lists(self):
        self.assertEqual(lg.as_conversations(["a", PROMPT]),
                         [[{"role": "user", "content": "a"}], PROMPT])
        self.assertEqual(lg.as_conversations(PROMPT), [PROMPT])


if __name__ == "__main__":
    unittest.main()
