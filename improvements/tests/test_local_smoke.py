"""The smoke session: refusals, cleanup and the script, all against fakes."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import frozen  # noqa: F401
import smoke_local
from drag_plus.runner import local_smoke as ls
from drag_plus.runner.fake import FakeGenerator
from drag_plus.runner.local_preflight import PreflightError, Readings
from drag_plus.runner.local_server import OllamaServer
from drag_plus.runner.variant import parse_variant
from fake_ollama import FakeOllama
from test_runner_fake import ROWS

GOOD = Readings(vram_gib=5.0, ram_gib=8.0, disk_gib=50.0)


class StubServer:
    """Records what the session asks of a server; ``models`` is what is installed."""

    def __init__(self, models=("tiny:1b",)):
        self.models, self.events = list(models), []

    def ensure_running(self):
        self.events.append("ensure_running")

    def has_model(self, model):
        return model in self.models

    def pull(self, model):
        self.events.append(f"pull {model}")
        self.models.append(model)

    def close(self, model=None):
        self.events.append(f"close {model}")


def job(model="tiny:1b", variants=2, allow_pull=True):
    names = [parse_variant({"name": f"v{i}"}) for i in range(variants)]
    return ls.SmokeJob(model, ROWS[:2], names, Path("out"), Path("r.jsonl"), Path("q.jsonl"),
                       allow_pull=allow_pull)


def ran(job_, variant):
    return {"variant": variant.name, "ok": True}


class RefusalTests(unittest.TestCase):
    def refused(self, readings, **kwargs):
        server = StubServer(**kwargs)
        with self.assertRaises(PreflightError) as caught:
            ls.run_smoke(job(), server, readings, ran)
        return server, caught.exception

    def test_low_vram_refuses_before_any_server_is_touched(self):
        server, error = self.refused(Readings(2.0, 8.0, 50.0))
        self.assertIn("VRAM", str(error))
        self.assertEqual(server.events, ["close tiny:1b"])

    def test_low_ram_refuses(self):
        self.assertIn("RAM", str(self.refused(Readings(5.0, 3.0, 50.0))[1]))

    def test_low_disk_refuses(self):
        self.assertIn("disk", str(self.refused(Readings(5.0, 8.0, 0.2))[1]))

    def test_a_missing_model_is_not_pulled_when_pulling_is_off(self):
        server = StubServer(models=())
        with self.assertRaisesRegex(PreflightError, "--no-pull"):
            ls.run_smoke(job(allow_pull=False), server, GOOD, ran)
        self.assertNotIn("pull tiny:1b", server.events)

    def test_a_missing_large_model_is_never_pulled(self):
        server = StubServer(models=())
        with self.assertRaisesRegex(PreflightError, "never pulled automatically"):
            ls.run_smoke(job(model="llama3.1:8b"), server, GOOD, ran)
        self.assertFalse(any(e.startswith("pull") for e in server.events))

    def test_a_missing_small_model_needs_ten_gib_of_disk_to_pull(self):
        server = StubServer(models=())
        with self.assertRaisesRegex(PreflightError, "to pull"):
            ls.run_smoke(job(model="llama3.2:3b"), server, Readings(5.0, 8.0, 5.0), ran)
        self.assertFalse(any(e.startswith("pull") for e in server.events))


class SessionTests(unittest.TestCase):
    def test_a_missing_small_model_is_pulled_once_then_every_variant_runs_in_order(self):
        server = StubServer(models=())
        results = ls.run_smoke(job(model="llama3.2:3b", variants=3), server, GOOD, ran)
        self.assertEqual([r["variant"] for r in results], ["v0", "v1", "v2"])
        self.assertEqual(server.events, ["ensure_running", "pull llama3.2:3b", "close llama3.2:3b"])

    def test_an_installed_model_is_not_pulled(self):
        server = StubServer()
        ls.run_smoke(job(), server, GOOD, ran)
        self.assertEqual(server.events, ["ensure_running", "close tiny:1b"])

    def test_the_server_is_released_when_a_run_raises(self):
        server = StubServer()

        def boom(job_, variant):
            raise RuntimeError("model crashed")

        with self.assertRaisesRegex(RuntimeError, "model crashed"):
            ls.run_smoke(job(), server, GOOD, boom)
        self.assertEqual(server.events[-1], "close tiny:1b")

    def test_a_model_side_error_skips_the_remaining_variants(self):
        def run_one(job_, variant):
            return {"variant": variant.name, "ok": False, "error": "timeout"}

        results = ls.run_smoke(job(variants=3), StubServer(), GOOD, run_one)
        self.assertEqual([r["variant"] for r in results], ["v0"])

    def test_a_server_the_session_did_not_start_survives_it(self):
        with FakeOllama() as fake:
            def popen(*args, **kwargs):
                raise AssertionError("must not start a server")

            server = OllamaServer(fake.base_url.split("//")[1].split("/")[0], popen=popen)
            ls.run_smoke(job(), server, GOOD, ran)
            self.assertFalse(server.started)
            self.assertEqual(fake.requests, [])


class ScriptTests(unittest.TestCase):
    def run_main(self, folder, *extra, readings=GOOD, host="127.0.0.1:1"):
        argv = ["--questions", "2", "--model", "tiny:1b", "--host", host, "--output", folder, *extra]
        out, err = io.StringIO(), io.StringIO()

        def no_server(host, log_path):
            return OllamaServer(host, popen=self.fail_popen)

        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = smoke_local.main(argv, hardware=lambda path: readings, server_factory=no_server)
        return code, out.getvalue(), err.getvalue()

    @staticmethod
    def fail_popen(*args, **kwargs):
        raise AssertionError("the script must not start a real server in a test")

    def test_two_questions_run_end_to_end_against_the_fake_server(self):
        with tempfile.TemporaryDirectory() as folder, \
                FakeOllama(brain=FakeGenerator._reply) as fake:
            host = fake.base_url.split("//")[1].split("/")[0]
            code, out, _ = self.run_main(folder, host=host)
            runs = sorted(p.name for p in Path(folder).iterdir())
        self.assertEqual(code, 0, out)
        self.assertIn("[PASS] local-smoke", out)
        self.assertIn("NOT A RESULT", out)
        self.assertEqual(len(runs), 1)

    def test_low_vram_exits_2_without_contacting_the_server(self):
        with tempfile.TemporaryDirectory() as folder, FakeOllama() as fake:
            host = fake.base_url.split("//")[1].split("/")[0]
            code, _, err = self.run_main(folder, host=host, readings=Readings(1.0, 8.0, 50.0))
            self.assertEqual((fake.requests, fake.gets), ([], []))
        self.assertEqual(code, 2)
        self.assertIn("free VRAM is 1.0 GiB", err)

    def test_a_model_the_server_lacks_exits_2_with_no_pull(self):
        with tempfile.TemporaryDirectory() as folder, FakeOllama(models=()) as fake:
            host = fake.base_url.split("//")[1].split("/")[0]
            code, _, err = self.run_main(folder, "--no-pull", host=host)
        self.assertEqual(code, 2)
        self.assertIn("--no-pull", err)

    def test_a_missing_replay_file_exits_2_before_any_server_work(self):
        with tempfile.TemporaryDirectory() as folder, FakeOllama() as fake:
            host = fake.base_url.split("//")[1].split("/")[0]
            code, _, err = self.run_main(folder, "--replay", "/nonexistent/replay.jsonl", host=host)
            self.assertEqual((fake.requests, fake.gets), ([], []))
        self.assertEqual(code, 2)
        self.assertIn("replay file not found", err)

    def test_more_than_five_questions_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            smoke_local.parse_args(["--questions", "6"])


if __name__ == "__main__":
    unittest.main()
