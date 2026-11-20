"""The CLI validates inputs without heavy imports and runs end to end offline."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT / "scripts"
BASELINE = PROJECT / "improvements/configs/baseline.json"
HEAVY = ("flashrag", "vllm", "torch", "faiss", "transformers")


def run_cli(*args):
    """Run main() in a clean interpreter and report which heavy modules got imported."""
    code = (f"import sys; sys.path.insert(0, {str(SCRIPTS)!r}); import run_variant; "
            f"status = run_variant.main({list(args)!r}); "
            f"print('HEAVY=' + ','.join(m for m in {HEAVY!r} if m in sys.modules)); "
            f"sys.exit(status)")
    return subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True,
                          timeout=120, cwd=PROJECT)


class ValidateOnlyTests(unittest.TestCase):
    def test_validate_only_imports_no_heavy_library(self):
        done = run_cli("--variant", str(BASELINE), "--validate-only", "--limit", "3")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("Input validation passed", done.stdout)
        self.assertIn("questions: 3", done.stdout)
        self.assertIn("HEAVY=\n", done.stdout + "\n")

    def test_a_bad_variant_fails_before_any_work(self):
        with tempfile.TemporaryDirectory() as folder:
            bad = Path(folder) / "bad.json"
            bad.write_text(json.dumps({"name": "x", "flags": {"nope": 1}}))
            done = run_cli("--variant", str(bad), "--validate-only")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("unknown flag", done.stderr)

    def test_the_fake_backend_runs_end_to_end_from_the_command_line(self):
        with tempfile.TemporaryDirectory() as folder:
            done = run_cli("--variant", str(BASELINE), "--backend", "fake", "--limit", "2",
                           "--profile", "--output", folder)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertIn("exit status: complete", done.stdout)
            self.assertIn("HEAVY=\n", done.stdout + "\n")
            run_dir = next(Path(folder).iterdir())
            self.assertTrue(run_dir.name.endswith("-baseline"))
            for name in ("checkpoints.jsonl", "transcripts.jsonl", "stage-timings.jsonl",
                         "run-manifest.json", "summary.json"):
                self.assertTrue((run_dir / name).is_file(), name)
            again = run_cli("--variant", str(BASELINE), "--backend", "fake", "--limit", "2",
                            "--profile", "--resume", str(run_dir))
            self.assertEqual(again.returncode, 0, again.stderr)
            manifest = json.loads((run_dir / "run-manifest.json").read_text())
            self.assertEqual(manifest["resumes"][0]["completed_before"], 4)


if __name__ == "__main__":
    unittest.main()
