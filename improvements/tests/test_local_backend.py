"""``--backend local`` end to end against a fake Ollama server: prompts, runner, CLI."""
import json
from pathlib import Path
import tempfile
import unittest

import frozen  # noqa: F401
from drag_plus.adapters import Generator, PromptTemplate, Retriever
from drag_plus.runner import local_backend as lb
from drag_plus.runner.fake import FakeGenerator
from drag_plus.runner.run import run_variant
from fake_ollama import FakeOllama
from model.prompt import standard_rag_prompt_template
from test_run_variant_cli import BASELINE, run_cli
from test_runner_fake import ROWS, make_request

DOCS = [{"id": "d1", "contents": "Paris\nThe capital of France."},
        {"id": "d2", "contents": "Seine\nA river in Paris."}]
REFERENCE = "Doc 1(Title: Paris) The capital of France.\nDoc 2(Title: Seine) A river in Paris.\n"
DATASETS = ["NQ", "TriviaQA", "PopQA", "2wiki", "HotpotQA", "StrategyQA"]


def write_fixtures(folder):
    """Stored passages for the five test questions, as in results/retrieval-200.jsonl."""
    retrieval = [{"dataset": r["dataset"], "split": r["split"], "id": r["id"],
                  "drag": {}, "naive": [dict(DOCS[0], id=f"{r['id']}-a"), dict(DOCS[1], id=f"{r['id']}-b")]}
                 for r in ROWS]
    questions = [dict(r, position=i) for i, r in enumerate(ROWS)]
    paths = Path(folder) / "retrieval.jsonl", Path(folder) / "questions.jsonl"
    paths[0].write_text("\n".join(json.dumps(r) for r in retrieval) + "\n")
    paths[1].write_text("\n".join(json.dumps(r) for r in questions) + "\n")
    return paths


class TemplateTests(unittest.TestCase):
    def test_finished_messages_pass_through_as_copies(self):
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        out = lb.MessageTemplate("NQ").get_string(messages=messages)
        self.assertEqual(out, messages)
        out[0]["content"] = "changed"
        self.assertEqual(messages[0]["content"], "s")

    def test_the_naive_prompt_matches_the_frozen_template_for_every_dataset(self):
        for dataset in DATASETS:
            frozen_template = standard_rag_prompt_template[dataset](None).kwargs
            expected = [
                {"role": "system", "content": frozen_template["system_prompt"].format(
                    question="Q?", reference=REFERENCE)},
                {"role": "user", "content": frozen_template["user_prompt"].format(
                    question="Q?", reference=REFERENCE)}]
            actual = lb.MessageTemplate(dataset).get_string(question="Q?", retrieval_result=DOCS)
            self.assertEqual(actual, expected, dataset)

    def test_it_satisfies_the_template_protocol_and_never_returns_a_string(self):
        template = lb.MessageTemplate("NQ")
        self.assertIsInstance(template, PromptTemplate)
        self.assertIsInstance(template.get_string(question="Q?", retrieval_result=[]), list)


class BuildTests(unittest.TestCase):
    def test_the_backend_is_named_local_and_describes_itself_as_smoke_only(self):
        with tempfile.TemporaryDirectory() as folder:
            retrieval, questions = write_fixtures(folder)
            backend = lb.build_local_backend(model="tiny:1b", base_url="http://x/v1",
                                             retrieval_path=retrieval, questions_path=questions)
            description, identity = backend.describe(), backend.retrieval_identity()
        self.assertEqual(backend.name, "local")
        self.assertIsInstance(backend.generator, Generator)
        self.assertIsInstance(backend.retriever, Retriever)
        self.assertEqual(description["model"], "ollama:tiny:1b")
        self.assertIn("not comparable", description["note"])
        self.assertEqual(description["generator"]["calls"], 0)
        self.assertEqual((identity["encoder"], identity["top_k"]), ("replay", 3))
        self.assertEqual(backend.count_tokens("abcdefg"), 3)


class RunnerTests(unittest.TestCase):
    def test_both_arms_run_end_to_end_over_http_with_message_lists(self):
        with tempfile.TemporaryDirectory() as folder, \
                FakeOllama(brain=FakeGenerator._reply) as server:
            retrieval, questions = write_fixtures(folder)
            backend = lb.build_local_backend(base_url=server.base_url, retrieval_path=retrieval,
                                             questions_path=questions)
            out_dir = Path(folder) / "run"
            manifest = run_variant(make_request(out_dir, backend=backend))
            checkpoints = (out_dir / "checkpoints.jsonl").read_text().splitlines()
        self.assertEqual(manifest["exit_status"], "complete")
        self.assertEqual(len(checkpoints), 2 * len(ROWS))
        self.assertEqual(manifest["backend"]["name"], "local")
        calls = manifest["backend"]["generator"]["calls"]
        self.assertEqual(calls, len(server.requests))
        for _, payload in server.requests:
            self.assertTrue(all(set(m) == {"role", "content"} for m in payload["messages"]))
        expected = [lb.MessageTemplate(r["dataset"]).get_string(
            question=r["question"], retrieval_result=DOCS) for r in ROWS]
        sent = [payload["messages"] for _, payload in server.requests]
        self.assertEqual(len([prompt for prompt in sent if prompt in expected]), len(ROWS))
        self.assertGreater(manifest["backend"]["retrieval"]["replayed"], 0)


class CommandLineTests(unittest.TestCase):
    def test_validate_only_accepts_the_local_backend_without_a_server(self):
        done = run_cli("--variant", str(BASELINE), "--backend", "local", "--validate-only",
                       "--limit", "2")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("Input validation passed", done.stdout)

    def test_a_missing_replay_file_fails_before_any_work(self):
        done = run_cli("--variant", str(BASELINE), "--backend", "local", "--validate-only",
                       "--replay", "/nonexistent/replay.jsonl")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("Replay file not found", done.stderr)

    def test_the_cli_runs_recorded_questions_against_the_server_url(self):
        with tempfile.TemporaryDirectory() as folder, FakeOllama(brain=FakeGenerator._reply) as server:
            done = run_cli("--variant", str(BASELINE), "--backend", "local", "--local-url",
                           server.base_url, "--limit", "2", "--profile", "--output", folder)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertIn("exit status: complete", done.stdout)
            run_dir = next(Path(folder).iterdir())
            manifest = json.loads((run_dir / "run-manifest.json").read_text())
        self.assertEqual(manifest["backend"]["name"], "local")
        self.assertEqual(manifest["backend"]["generator"]["calls"], len(server.requests))
        self.assertIn("HEAVY=\n", done.stdout + "\n")


if __name__ == "__main__":
    unittest.main()
