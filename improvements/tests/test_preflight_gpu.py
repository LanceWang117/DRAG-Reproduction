"""The GPU-memory preflight: pure arithmetic, nvidia-smi parsing and the offline CLI.

Every hardware reading is injected; the real card is never queried on the PC.
"""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

PROJECT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))
import preflight_gpu  # noqa: E402


class PreflightMathTests(unittest.TestCase):
    def test_fp16_index_on_an_80_gib_card(self):
        result = preflight_gpu.check_fit(80, 0.45, 64.5)
        self.assertAlmostEqual(result["required_gib"], 71.75)
        with self.assertRaises(preflight_gpu.PreflightError):
            preflight_gpu.check_fit(80, 0.85, 64.5)

    def test_the_error_names_the_shortfall_and_the_fix(self):
        with self.assertRaises(preflight_gpu.PreflightError) as caught:
            preflight_gpu.check_fit(80, 0.85, 64.5)
        message = str(caught.exception)
        self.assertIn("Insufficient GPU memory", message)
        self.assertIn("103.75", message)
        self.assertIn("gpu_memory_utilization", message)

    def test_an_exact_fit_passes_and_one_mib_over_fails(self):
        preflight_gpu.check_fit(80, 0.5, 80.0, e5_gib=0, margin_gib=0)
        with self.assertRaises(preflight_gpu.PreflightError):
            preflight_gpu.check_fit(80, 0.5, 80.0 + 2 / 1024, e5_gib=0, margin_gib=0)

    def test_impossible_inputs_are_rejected(self):
        cases = ((0, 0.5, 1), (80, 0, 1), (80, 1.1, 1), (80, float("nan"), 1),
                 (80, 0.5, -1))
        for total, utilization, index in cases:
            with self.assertRaises(ValueError, msg=(total, utilization, index)):
                preflight_gpu.check_fit(total, utilization, index)

    def test_bytes_convert_to_gib(self):
        result = preflight_gpu.check_fit_bytes(80, 0.45, int(64.5 * preflight_gpu.GIB))
        self.assertAlmostEqual(result["index_fp16_gib"], 32.25)

    def test_nvidia_smi_output_parses_and_sums(self):
        self.assertAlmostEqual(preflight_gpu.parse_smi_memory("81920\n"), 80.0)
        self.assertAlmostEqual(preflight_gpu.parse_smi_memory("81920\n40960\n"), 120.0)
        for text in ("", "not-a-number\n", "0\n"):
            with self.assertRaises(preflight_gpu.PreflightError, msg=text):
                preflight_gpu.parse_smi_memory(text)

    def test_read_gpu_total_uses_the_injected_runner(self):
        fake = SimpleNamespace(returncode=0, stdout="81920\n", stderr="")
        total = preflight_gpu.read_gpu_total_gib(runner=lambda *a, **k: fake)
        self.assertEqual(total, 80.0)

    def test_nvidia_smi_failure_is_reported(self):
        fake = SimpleNamespace(returncode=9, stdout="", stderr="boom\n")
        with self.assertRaisesRegex(preflight_gpu.PreflightError, "exit 9.*boom"):
            preflight_gpu.read_gpu_total_gib(runner=lambda *a, **k: fake)

    def test_missing_nvidia_smi_is_reported(self):
        def missing(*args, **kwargs):
            raise FileNotFoundError("nvidia-smi")

        with self.assertRaisesRegex(preflight_gpu.PreflightError, "GPU host"):
            preflight_gpu.read_gpu_total_gib(runner=missing)


class PreflightCliTests(unittest.TestCase):
    @staticmethod
    def run_cli(*args):
        return subprocess.run([sys.executable, str(SCRIPTS / "preflight_gpu.py"), *args],
                              capture_output=True, text=True, timeout=30)

    def test_fits_and_refuses_from_the_command_line(self):
        ok = self.run_cli("--index-gib", "64.5", "--gpu-total-gib", "80",
                          "--gpu-memory-utilization", "0.45")
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertIn("Preflight passed", ok.stdout)
        bad = self.run_cli("--index-gib", "64.5", "--gpu-total-gib", "80",
                           "--gpu-memory-utilization", "0.85")
        self.assertEqual(bad.returncode, 1)
        self.assertIn("Insufficient GPU memory", bad.stderr)

    def test_an_index_size_is_required(self):
        done = self.run_cli("--gpu-total-gib", "80", "--gpu-memory-utilization", "0.45")
        self.assertNotEqual(done.returncode, 0)


if __name__ == "__main__":
    unittest.main()
