"""Runner wiring: the run composes the retriever and fingerprints the cache key (J17)."""
import sys
import tempfile
from pathlib import Path
import unittest

IMPROVEMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMPROVEMENTS))
from drag_plus.flags import Flags  # noqa: E402
from drag_plus.retrieval.hybrid import HybridRetriever, config_fingerprint  # noqa: E402


class RunnerWiringTests(unittest.TestCase):
    """The runner composes the retriever and fingerprints it into the cache key."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def pipelines(self, flags):
        from drag_plus.runner import run as runner
        from drag_plus.runner.backends import build_fake_backend
        from drag_plus.runner.run import RunRequest
        from drag_plus.runner.variant import parse_variant
        request = RunRequest(
            rows=[{"dataset": "NQ", "split": "dev", "id": "q0", "question": "q?",
                   "golden_answers": ["a"]}],
            variant=parse_variant({"name": "t", "flags": flags}),
            backend=build_fake_backend(), out_dir=self.root / "run",
            now=lambda: "2026-01-01T00:00:00Z", environment=lambda: {})
        return runner._Pipelines(request)

    def test_the_default_run_keeps_the_bare_dense_retriever(self):
        pipelines = self.pipelines({"retrieval_cache": str(self.root / "c.jsonl")})
        self.assertFalse(isinstance(pipelines.retriever.inner, HybridRetriever))
        self.assertEqual(pipelines.retriever.identity["config"],
                         config_fingerprint(Flags()))

    def test_hybrid_composes_the_dense_retriever_and_is_fingerprinted(self):
        flags = {"retriever": "hybrid", "bm25_index": "/index"}
        pipelines = self.pipelines({"retrieval_cache": str(self.root / "c.jsonl"), **flags})
        self.assertIsInstance(pipelines.retriever.inner, HybridRetriever)
        self.assertEqual(pipelines.retriever.identity["config"],
                         config_fingerprint(Flags(flags)))

    def test_hybrid_without_a_cache_still_composes(self):
        pipelines = self.pipelines({"retriever": "hybrid", "bm25_index": "/index"})
        self.assertIsInstance(pipelines.retriever, HybridRetriever)

if __name__ == "__main__":
    unittest.main()
