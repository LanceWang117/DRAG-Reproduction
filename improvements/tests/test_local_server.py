"""Server lifecycle with a fake process and fake HTTP: nothing real is ever started."""
import json
import subprocess
import unittest
from types import SimpleNamespace

import frozen  # noqa: F401
from drag_plus.runner.local_server import OllamaServer, ServerError


class FakeProcess:
    def __init__(self, exits_with=None, hangs=False):
        self.exits_with, self.hangs = exits_with, hangs
        self.events = []

    def poll(self):
        return self.exits_with

    def terminate(self):
        self.events.append("terminate")

    def kill(self):
        self.events.append("kill")

    def wait(self, timeout=None):
        self.events.append("wait")
        if self.hangs and "kill" not in self.events:
            raise subprocess.TimeoutExpired("ollama", timeout)


class FakeHttp:
    """Answer GETs after ``up_after`` failed attempts, and record every request."""

    def __init__(self, up_after=0, models=None):
        self.up_after, self.failures, self.models = up_after, 0, models or {}
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append((request.get_method(), request.full_url, request.data))
        if self.failures < self.up_after:
            self.failures += 1
            raise ConnectionRefusedError("down")
        body = {"models": [{"name": n, "size": s} for n, s in self.models.items()]}
        return FakeResponse(json.dumps(body if request.full_url.endswith("/api/tags") else {}))


class FakeResponse:
    def __init__(self, text):
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.text.encode()


def make(http=None, process=None, **overrides):
    spawned = []
    http = http or FakeHttp()
    process = process or FakeProcess()

    def popen(command, **kwargs):
        spawned.append((command, kwargs))
        return process

    values = dict(popen=popen, opener=http, sleep=lambda seconds: None, environ={"PATH": "/bin"},
                  wait_seconds=2.0)
    values.update(overrides)
    server = OllamaServer("127.0.0.1:5555", **values)
    return server, spawned, http, process


class ReachTests(unittest.TestCase):
    def test_a_running_server_is_used_and_never_started_or_stopped(self):
        server, spawned, http, process = make()
        server.ensure_running()
        server.close("tiny:1b")
        self.assertEqual((spawned, server.started, process.events), ([], False, []))
        self.assertTrue(all(method == "GET" for method, _, _ in http.requests))

    def test_installed_models_are_listed_with_sizes(self):
        server, *_ = make(FakeHttp(models={"tiny:1b": 1234, "other:latest": 9}))
        self.assertEqual(server.models(), {"tiny:1b": 1234, "other:latest": 9})
        self.assertTrue(server.has_model("tiny:1b"))
        self.assertTrue(server.has_model("other"))
        self.assertFalse(server.has_model("missing:3b"))


class StartTests(unittest.TestCase):
    def test_a_missing_server_is_started_with_one_model_one_request_and_4096_context(self):
        server, spawned, _, _ = make(FakeHttp(up_after=3))
        server.ensure_running()
        command, kwargs = spawned[0]
        self.assertEqual(command, ["ollama", "serve"])
        self.assertEqual({k: kwargs["env"][k] for k in (
            "OLLAMA_HOST", "OLLAMA_CONTEXT_LENGTH", "OLLAMA_NUM_PARALLEL", "OLLAMA_MAX_LOADED_MODELS")},
            {"OLLAMA_HOST": "127.0.0.1:5555", "OLLAMA_CONTEXT_LENGTH": "4096",
             "OLLAMA_NUM_PARALLEL": "1", "OLLAMA_MAX_LOADED_MODELS": "1"})
        self.assertEqual(kwargs["env"]["PATH"], "/bin")
        self.assertTrue(server.started)

    def test_close_unloads_the_model_then_stops_the_process_it_started(self):
        server, _, http, process = make(FakeHttp(up_after=1))
        server.ensure_running()
        server.close("tiny:1b")
        method, url, data = http.requests[-1]
        self.assertEqual((method, url), ("POST", "http://127.0.0.1:5555/api/generate"))
        self.assertEqual(json.loads(data), {"model": "tiny:1b", "keep_alive": 0})
        self.assertEqual(process.events, ["terminate", "wait"])
        self.assertFalse(server.started)

    def test_a_process_that_ignores_terminate_is_killed(self):
        server, _, _, process = make(FakeHttp(up_after=1), FakeProcess(hangs=True))
        server.ensure_running()
        server.close()
        self.assertEqual(process.events, ["terminate", "wait", "kill", "wait"])

    def test_a_server_that_exits_at_once_is_reported_and_still_cleaned_up(self):
        server, _, _, process = make(FakeHttp(up_after=99), FakeProcess(exits_with=1))
        with self.assertRaisesRegex(ServerError, "exited with code 1"):
            server.ensure_running()
        server.close()
        self.assertIn("terminate", process.events)

    def test_a_server_that_never_answers_times_out_and_is_stopped_by_close(self):
        sleeps = []
        server, _, _, process = make(FakeHttp(up_after=99), sleep=sleeps.append)
        with self.assertRaisesRegex(ServerError, "did not answer"):
            server.ensure_running()
        self.assertEqual(len(sleeps), 4)
        server.close()
        self.assertIn("terminate", process.events)

    def test_a_missing_ollama_binary_is_reported_and_nothing_is_marked_started(self):
        def popen(command, **kwargs):
            raise FileNotFoundError(command[0])

        server, *_ = make(FakeHttp(up_after=99), popen=popen)
        with self.assertRaisesRegex(ServerError, "cannot run `ollama serve`"):
            server.ensure_running()
        self.assertFalse(server.started)

    def test_close_without_a_model_skips_the_unload_request(self):
        server, _, http, _ = make(FakeHttp(up_after=1))
        server.ensure_running()
        count = len(http.requests)
        server.close()
        self.assertEqual(len(http.requests), count)


class PullTests(unittest.TestCase):
    def test_a_pull_returns_the_installed_size(self):
        calls = []

        def run(command, **kwargs):
            calls.append((command, kwargs["env"]["OLLAMA_HOST"]))
            return SimpleNamespace(returncode=0, stderr="")

        server, *_ = make(FakeHttp(models={"tiny:1b": 2_000_000_000}), run=run)
        self.assertEqual(server.pull("tiny:1b"), 2_000_000_000)
        self.assertEqual(calls, [(["ollama", "pull", "tiny:1b"], "127.0.0.1:5555")])

    def test_a_failed_pull_is_reported(self):
        def run(command, **kwargs):
            return SimpleNamespace(returncode=1, stderr="pull model manifest: file does not exist")

        server, *_ = make(run=run)
        with self.assertRaisesRegex(ServerError, "file does not exist"):
            server.pull("nope:1b")


if __name__ == "__main__":
    unittest.main()
