"""The package imports on a bare interpreter; adapters reject incomplete objects."""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "improvements"))
from drag_plus.adapters import Backends  # noqa: E402
from drag_plus.state import QuestionState  # noqa: E402

HEAVY = ("flashrag", "vllm", "torch", "faiss")
PROBE = f"""
import sys
sys.path.insert(0, {str(PROJECT)!r})
import improvements.drag_plus
import improvements.drag_plus.pipeline, improvements.drag_plus.prompts
import improvements.drag_plus.flags, improvements.drag_plus.state, improvements.drag_plus.adapters
loaded = [m for m in {HEAVY!r} if m in sys.modules]
print(','.join(loaded))
"""


class ImportTests(unittest.TestCase):
    def test_import_needs_no_heavy_dependency_and_no_stub(self):
        # -I ignores PYTHONPATH, so the test-only flashrag stub cannot satisfy anything.
        done = subprocess.run([sys.executable, "-I", "-c", PROBE],
                              capture_output=True, text=True, timeout=30)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.strip(), "")


class BackendsTests(unittest.TestCase):
    def test_missing_method_is_rejected_at_construction(self):
        good = SimpleNamespace(generate=lambda p: [p], search=lambda q: [],
                               get_string=lambda **k: k)
        with self.assertRaisesRegex(TypeError, "retriever"):
            Backends(good, SimpleNamespace(), good)

    def test_calls_forward_params_only_when_given(self):
        seen = []
        fake = SimpleNamespace(generate=lambda *a, **k: seen.append((a, k)) or ["x"],
                               search=lambda q: [], get_string=lambda **k: k)
        backends = Backends(fake, fake, fake)
        self.assertEqual(backends.complete("p"), "x")
        backends.complete("p", max_tokens=8)
        self.assertEqual(seen, [(("p",), {}), (("p",), {"max_tokens": 8})])


class StateTests(unittest.TestCase):
    def test_emit_records_in_order_and_forwards(self):
        sunk = []
        state = QuestionState("q", sink=lambda k, v: sunk.append(k))
        state.emit("a", 1)
        state.emit("b", 2)
        self.assertEqual(state.outputs, [("a", 1), ("b", 2)])
        self.assertEqual(sunk, ["a", "b"])

    def test_records_do_not_share_mutable_defaults(self):
        first, second = QuestionState("q"), QuestionState("q")
        first.query_pool["x"] = []
        self.assertEqual(second.query_pool, {})


if __name__ == "__main__":
    unittest.main()
