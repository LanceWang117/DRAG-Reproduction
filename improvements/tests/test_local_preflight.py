"""Preflight floors on injected readings; the real GPU, memory and disk are never read."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace

import frozen  # noqa: F401
from drag_plus.runner import local_preflight as pf

MEMINFO = "MemTotal:       24400000 kB\nMemFree:         1000000 kB\nMemAvailable:    7340032 kB\n"
GIB = 1024 ** 3


def readings(vram=5.0, ram=8.0, disk=50.0):
    return pf.Readings(vram, ram, disk)


class ParseTests(unittest.TestCase):
    def test_available_memory_is_converted_from_kib_to_gib(self):
        self.assertAlmostEqual(pf.parse_meminfo(MEMINFO), 7.0)

    def test_a_report_without_memavailable_is_an_error(self):
        with self.assertRaises(ValueError):
            pf.parse_meminfo("MemTotal: 1 kB\n")

    def test_free_vram_is_read_from_the_first_gpu_in_mib(self):
        self.assertAlmostEqual(pf.parse_nvidia_smi("5120\n2048\n"), 5.0)

    def test_unusable_gpu_output_is_none(self):
        self.assertIsNone(pf.parse_nvidia_smi(""))
        self.assertIsNone(pf.parse_nvidia_smi("No devices were found\n"))


class ReadHardwareTests(unittest.TestCase):
    def read(self, run, meminfo_text=MEMINFO, free=20 * GIB):
        with tempfile.TemporaryDirectory() as folder:
            meminfo = Path(folder) / "meminfo"
            meminfo.write_text(meminfo_text)
            seen = []

            def disk_usage(path):
                seen.append(path)
                return SimpleNamespace(free=free)

            result = pf.read_hardware(Path(folder), run=run, meminfo=meminfo, disk_usage=disk_usage)
            return result, seen, folder

    def test_every_source_is_injected(self):
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            return SimpleNamespace(returncode=0, stdout="4096\n")

        result, seen, folder = self.read(run)
        self.assertEqual(result, pf.Readings(4.0, 7.0, 20.0))
        self.assertEqual(calls[0][0], "nvidia-smi")
        self.assertEqual(seen, [Path(folder)])

    def test_a_missing_nvidia_smi_gives_no_vram_reading(self):
        def run(command, **kwargs):
            raise FileNotFoundError("nvidia-smi")

        self.assertIsNone(self.read(run)[0].vram_gib)

    def test_a_failing_nvidia_smi_gives_no_vram_reading(self):
        def run(command, **kwargs):
            return SimpleNamespace(returncode=9, stdout="")

        self.assertIsNone(self.read(run)[0].vram_gib)

    def test_a_timed_out_nvidia_smi_gives_no_vram_reading(self):
        def run(command, **kwargs):
            raise subprocess.TimeoutExpired(command, 15)

        self.assertIsNone(self.read(run)[0].vram_gib)


class ResourceTests(unittest.TestCase):
    def test_enough_of_everything_passes(self):
        self.assertEqual(pf.check_resources(readings()), [])

    def test_the_floors_are_inclusive(self):
        self.assertEqual(pf.check_resources(readings(vram=3.0, ram=4.0, disk=1.0)), [])

    def test_low_vram_is_refused(self):
        self.assertIn("free VRAM is 2.9 GiB", pf.check_resources(readings(vram=2.9))[0])

    def test_no_gpu_reading_is_refused(self):
        self.assertIn("no NVIDIA GPU reading", pf.check_resources(readings(vram=None))[0])

    def test_low_ram_is_refused(self):
        self.assertIn("available RAM is 3.9 GiB", pf.check_resources(readings(ram=3.9))[0])

    def test_low_disk_is_refused(self):
        self.assertIn("free disk is 0.5 GiB", pf.check_resources(readings(disk=0.5))[0])

    def test_every_problem_is_listed(self):
        self.assertEqual(len(pf.check_resources(readings(vram=1, ram=1, disk=0))), 3)


class PullTests(unittest.TestCase):
    def test_a_small_tag_with_enough_disk_may_be_pulled(self):
        for model in ("llama3.2:3b", "llama3.2:1b", "qwen2.5:0.5b", "gemma2:2b-instruct-q4_0"):
            self.assertEqual(pf.check_pull(model, readings(disk=10.0)), [], model)

    def test_an_eight_billion_model_is_never_pulled_automatically(self):
        problems = pf.check_pull("llama3.1:8b", readings(disk=500))
        self.assertIn("never pulled automatically", problems[0])
        self.assertIn("ollama pull llama3.1:8b", problems[0])

    def test_a_tag_without_a_size_is_never_pulled_automatically(self):
        self.assertTrue(pf.check_pull("llama3.2", readings()))
        self.assertTrue(pf.check_pull("llama3.2:latest", readings()))

    def test_low_disk_blocks_a_pull(self):
        problems = pf.check_pull("llama3.2:3b", readings(disk=9.9))
        self.assertIn("need 10 GiB to pull", problems[0])


if __name__ == "__main__":
    unittest.main()
