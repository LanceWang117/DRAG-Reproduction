"""Retrieval acceleration: flag flow, thread-env ordering, GPU build order, GPU checks.

The PC has a 6 GB GTX 1660 Ti, so every GPU reading is injected. No torch, FlashRAG
or FAISS import happens here; the preflight arithmetic has its own test module.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(PROJECT / "improvements"))
sys.path.insert(0, str(SCRIPTS))
from drag_plus.flags import FlagError, Flags  # noqa: E402
from drag_plus.runner import backends  # noqa: E402
from drag_plus.runner.variant import load_variant  # noqa: E402
import preflight_gpu  # noqa: E402
import run_variant  # noqa: E402


def resource_config(root):
    """A config whose model/index/corpus paths exist, so build_backend gets past checks."""
    for name in ("e5", "llama", "index", "corpus"):
        (root / name).write_bytes(b"x")
    path = root / "config.json"
    path.write_text(json.dumps({"model2path": {"e5": "e5", "llama": "llama"},
                                "index_path": "index", "corpus_path": "corpus"}))
    return path


class AcceleratorFlagTests(unittest.TestCase):
    def test_defaults_keep_the_baseline(self):
        flags = Flags()
        self.assertFalse(flags.faiss_gpu)
        self.assertEqual(flags.gpu_memory_utilization, 0.85)
        self.assertEqual(flags.omp_threads, 12)

    def test_gpu_ratio_is_in_the_open_unit_interval(self):
        for value in (0, 0.0, -0.1, 1.01, float("nan"), float("inf")):
            with self.assertRaises(FlagError, msg=value):
                Flags({"gpu_memory_utilization": value})
        self.assertEqual(Flags({"gpu_memory_utilization": 1}).gpu_memory_utilization, 1.0)
        self.assertEqual(Flags({"gpu_memory_utilization": 0.45}).gpu_memory_utilization, 0.45)

    def test_threads_must_be_a_positive_int(self):
        for value in (0, -1, True, "8"):
            with self.assertRaises(FlagError, msg=value):
                Flags({"omp_threads": value})

    def test_gpu_faiss_is_a_bool(self):
        for value in (1, "yes"):
            with self.assertRaises(FlagError, msg=value):
                Flags({"faiss_gpu": value})

    def test_the_s2_variant_carries_the_acceleration_flags(self):
        variant = load_variant(PROJECT / "improvements/configs/s2-gpu-faiss.json")
        self.assertTrue(variant.flags.faiss_gpu)
        self.assertEqual(variant.flags.gpu_memory_utilization, 0.45)
        self.assertEqual(variant.flags.omp_threads, 12)


class ThreadEnvironmentTests(unittest.TestCase):
    def test_thread_variables_come_from_the_flag(self):
        environ = {}
        backends.apply_thread_environment(Flags({"omp_threads": 3}), environ)
        self.assertEqual(environ, {"OMP_NUM_THREADS": "3", "MKL_NUM_THREADS": "3"})

    def test_threads_work_with_faiss_gpu_off(self):
        environ = {}
        backends.apply_thread_environment(Flags({"faiss_gpu": False, "omp_threads": 5}),
                                          environ)
        self.assertEqual(environ["OMP_NUM_THREADS"], "5")

    def test_build_backend_sets_threads_before_the_flashrag_builder(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = SimpleNamespace(backend="flashrag", config=resource_config(root))
            seen = {}

            def fake_build(source, overrides, rows, out_dir, flags):
                seen["omp"] = os.environ.get("OMP_NUM_THREADS")
                seen["mkl"] = os.environ.get("MKL_NUM_THREADS")
                seen["flags"] = flags
                return "backend"

            with patch.dict(os.environ, {"OMP_NUM_THREADS": "99",
                                         "MKL_NUM_THREADS": "99"}), \
                 patch.object(run_variant, "build_runtime", return_value=root / "runtime"), \
                 patch.object(backends, "build_flashrag_backend", fake_build):
                result = run_variant.build_backend(args, Flags({"omp_threads": 3}), [], root)
        self.assertEqual(result, "backend")
        self.assertEqual((seen["omp"], seen["mkl"]), ("3", "3"))
        self.assertFalse(seen["flags"].faiss_gpu)

    def test_gpu_faiss_runs_the_preflight_before_building(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = SimpleNamespace(backend="flashrag", config=resource_config(root))
            index_size = (root / "index").stat().st_size
            seen = {}

            def fake_check(**kwargs):
                seen.update(kwargs)
                return {"required_gib": 1.0}

            def fake_build(source, overrides, rows, out_dir, flags):
                seen["built"] = True
                return "backend"

            flags = Flags({"faiss_gpu": True, "gpu_memory_utilization": 0.45})
            with patch.object(preflight_gpu, "read_gpu_total_gib", return_value=80.0), \
                 patch.object(preflight_gpu, "check_fit_bytes", fake_check), \
                 patch.object(run_variant, "build_runtime", return_value=root / "runtime"), \
                 patch.object(backends, "build_flashrag_backend", fake_build):
                run_variant.build_backend(args, flags, [], root)
        self.assertEqual(seen["total_gib"], 80.0)
        self.assertEqual(seen["gpu_memory_utilization"], 0.45)
        self.assertEqual(seen["index_bytes"], index_size)
        self.assertTrue(seen["built"])


class BuildOrderTests(unittest.TestCase):
    @staticmethod
    def order(faiss_gpu):
        """Record the factory call order for one faiss_gpu setting."""
        order = []

        def generator(config):
            order.append("generator")
            return "generator"

        def retriever(config):
            order.append("retriever")
            return "retriever"

        built = backends.build_rag_objects({}, faiss_gpu, generator, retriever)
        return order, built

    def test_gpu_places_the_index_before_vllm_reserves_its_pool(self):
        order, (generator, retriever) = self.order(True)
        self.assertEqual(order, ["retriever", "generator"])
        self.assertEqual((generator, retriever), ("generator", "retriever"))

    def test_cpu_keeps_the_baseline_generator_first_order(self):
        order, _built = self.order(False)
        self.assertEqual(order, ["generator", "retriever"])


class FaissSupportTests(unittest.TestCase):
    def test_a_cpu_build_fails_with_the_install_pointer(self):
        with self.assertRaisesRegex(RuntimeError, "faiss-gpu.*REPRODUCTION"):
            backends.ensure_faiss_gpu(SimpleNamespace(get_num_gpus=lambda: 0))

    def test_zero_visible_gpus_fails_instead_of_falling_back_to_cpu(self):
        fake = SimpleNamespace(index_cpu_to_all_gpus=lambda *a, **k: None,
                               get_num_gpus=lambda: 0)
        with self.assertRaisesRegex(RuntimeError, "no CUDA GPU"):
            backends.ensure_faiss_gpu(fake)

    def test_a_gpu_build_with_one_card_passes(self):
        fake = SimpleNamespace(index_cpu_to_all_gpus=lambda *a, **k: None,
                               get_num_gpus=lambda: 1)
        backends.ensure_faiss_gpu(fake)

    def test_a_driver_error_is_reported(self):
        def boom():
            raise RuntimeError("driver missing")

        fake = SimpleNamespace(index_cpu_to_all_gpus=lambda *a, **k: None,
                               get_num_gpus=boom)
        with self.assertRaisesRegex(RuntimeError, "driver missing"):
            backends.ensure_faiss_gpu(fake)


class FlashragSettingsTests(unittest.TestCase):
    def test_variant_flags_flow_into_the_config_override(self):
        flags = Flags({"faiss_gpu": True, "gpu_memory_utilization": 0.45})
        settings = backends.flashrag_settings({"index_path": "/i"}, flags, "NQ", "dev",
                                              "drag", Path("/out"))
        self.assertTrue(settings["faiss_gpu"])
        self.assertEqual(settings["gpu_memory_utilization"], 0.45)
        self.assertEqual(settings["split"], ["dev"])
        self.assertEqual(settings["index_path"], "/i")

    def test_defaults_write_the_baseline_values(self):
        settings = backends.flashrag_settings({}, Flags(), "NQ", "dev", "naive",
                                              Path("/out"))
        self.assertFalse(settings["faiss_gpu"])
        self.assertEqual(settings["gpu_memory_utilization"], 0.85)


if __name__ == "__main__":
    unittest.main()
