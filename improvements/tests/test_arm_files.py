"""J25: the committed E-job arm files match their spec, parse, and plan for rehearsal."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import frozen  # noqa: F401  (puts improvements/ and the test helpers on sys.path)
from drag_plus.rehearsal.plan import plan_arms
from drag_plus.runner.arms import ArmSpecError, build_all, build_arm, check, layers_from, load, write
from drag_plus.runner.variant import load_variant

PROJECT = Path(__file__).resolve().parents[2]
ARMS_DIR = PROJECT / "improvements/configs/arms"
EXPECTED = {"E01": {"A", "B", "C", "D", "E"},
            "E02a": {"A", "B", "C", "D", "opt-r2", "opt-tokens128"},
            "E02b": {"A", "B", "C", "D", "E", "F", "opt-no-b2fix"},
            "E03": {"B1", "B2", "B4", "B8", "B16", "B32"},
            "E04": {"e5", "hybrid"}, "E05": {"naive", "drag"},
            "PC": {"full-stack", "full-stack-generative", "grounded-vote"}}


def committed():
    return {path: load_variant(path) for path in sorted(ARMS_DIR.glob("*/*.json"))}


class CommittedArmTests(unittest.TestCase):
    def test_the_committed_files_match_the_spec(self):
        self.assertEqual(check(ARMS_DIR, build_all(*load(ARMS_DIR))), [])

    def test_every_e_job_arm_has_a_file_that_parses(self):
        found = {}
        for path in committed():
            found.setdefault(path.parent.name, set()).add(path.stem)
        self.assertEqual(found, EXPECTED)

    def test_the_arm_tables_are_encoded_exactly(self):
        flags = {f"{p.parent.name}/{p.stem}": dict(v.raw["flags"]) for p, v in committed().items()}
        self.assertEqual(flags["E01/A"], {})
        self.assertEqual(flags["E01/D"], {"moderator_final_only": True, "batch_agents_in_round": True,
                                          "faiss_gpu": True, "gpu_memory_utilization": 0.45})
        self.assertEqual(flags["E02a/D"]["response_rounds"], 1)
        self.assertEqual(flags["E02a/D"]["moderator_max_tokens"], 32)
        self.assertTrue(flags["E02b/E"]["opponent_action_schema"])
        self.assertEqual(flags["E02b/E"]["judge_scoring"], "logprob")
        self.assertEqual(flags["E02b/E"]["evidence_store"], "anchored")
        self.assertNotIn("refill_window", flags["E03/B1"])
        self.assertEqual((flags["E03/B32"]["question_batch_size"], flags["E03/B32"]["refill_window"]),
                         (32, True))

    def test_every_arm_after_e01_carries_the_b2_fix_except_its_control(self):
        for path, variant in committed().items():
            fixed = variant.flags.search_via_batch
            if path.parent.name == "E01" or path.stem == "opt-no-b2fix":
                self.assertFalse(fixed, path)
            else:
                self.assertTrue(fixed, path)
        control = dict(load_variant(ARMS_DIR / "E02b/opt-no-b2fix.json").raw["flags"])
        arm_a = dict(load_variant(ARMS_DIR / "E02b/A.json").raw["flags"])
        self.assertEqual(arm_a, {**control, "search_via_batch": True})

    def test_arms_built_on_p_say_they_are_provisional(self):
        for path, variant in committed().items():
            uses_p = "layers P" in variant.raw["description"]
            if uses_p:
                self.assertIn("PROVISIONAL P", variant.raw["description"], path)
        self.assertNotIn("PROVISIONAL", load_variant(ARMS_DIR / "E01/C.json").raw["description"])

    def test_route_arms_run_one_pipeline_each(self):
        self.assertEqual(load_variant(ARMS_DIR / "E05/naive.json").arms, ("naive",))
        self.assertEqual(load_variant(ARMS_DIR / "E05/drag.json").arms, ("drag",))

    def test_the_rehearsal_skips_exactly_the_server_only_arms(self):
        plans = plan_arms([variant.raw for variant in committed().values()])
        skipped = {plan.name for plan in plans if not plan.runnable}
        self.assertEqual(skipped, {"E01-D", "E02b-F", "E04-hybrid"})


class SpecTests(unittest.TestCase):
    LAYERS = layers_from({"layers": {"a": {"response_rounds": 1}, "b": {"response_rounds": 2}}},
                         {"P": {"provisional": True, "source": "test",
                                "flags": {"moderator_final_only": True}}})

    def test_a_later_layer_wins(self):
        document = build_arm("E9", "X", {"layers": ["a", "b"]}, self.LAYERS)
        self.assertEqual(document["flags"], {"response_rounds": 2})

    def test_unknown_layers_bad_flags_and_bad_constraints_are_refused(self):
        with self.assertRaisesRegex(ArmSpecError, "unknown layer 'zz'"):
            build_arm("E9", "X", {"layers": ["zz"]}, self.LAYERS)
        layers = layers_from({"layers": {"typo": {"respnse_rounds": 1},
                                         "bal": {"judge_order_balance": True}}}, {})
        with self.assertRaisesRegex(ArmSpecError, "E9 arm X"):
            build_arm("E9", "X", {"layers": ["typo"]}, layers)
        with self.assertRaisesRegex(ArmSpecError, "judge_order_balance"):
            build_arm("E9", "Y", {"layers": ["bal"]}, layers)

    def test_measured_layers_need_a_source_a_boolean_and_known_keys(self):
        with self.assertRaisesRegex(ArmSpecError, "needs a 'source'"):
            layers_from({}, {"P": {"provisional": True, "flags": {}}})
        for bad in ("false", None, 0):
            with self.assertRaisesRegex(ArmSpecError, "true or false"):
                layers_from({}, {"P": {"provisional": bad, "source": "s", "flags": {}}})
        with self.assertRaisesRegex(ArmSpecError, "provisonal"):
            layers_from({}, {"P": {"provisonal": False, "provisional": True, "source": "s",
                                   "flags": {}}})
        with self.assertRaisesRegex(ArmSpecError, "both spec and measured"):
            layers_from({"layers": {"P": {}}},
                        {"P": {"provisional": False, "source": "s", "flags": {}}})

    def test_a_misspelt_or_missing_layers_key_is_refused(self):
        for entry in ({"layer": ["a"]}, {"Layers": ["a"]}, {}, {"layers": ["a"], "nots": "x"}):
            with self.assertRaises(ArmSpecError, msg=entry):
                build_arm("E9", "X", entry, self.LAYERS)
        self.assertEqual(build_arm("E9", "Z", {"layers": []}, self.LAYERS)["flags"], {})

    def test_route_arms_carry_the_batch_and_promotion_layers(self):
        description = load_variant(ARMS_DIR / "E05/drag.json").raw["description"]
        for name in ("PROVISIONAL P", "promoted_drag", "best_batch"):
            self.assertIn(name, description)
        self.assertEqual(load_variant(ARMS_DIR / "E05/drag.json").flags.question_batch_size, 4)

    def test_check_reports_missing_changed_and_stale_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            files = {"E9/A.json": "{}\n", "E9/B.json": "{\"x\": 1}\n"}
            write(root, files)
            self.assertEqual(check(root, files), [])
            (root / "E9/A.json").write_text("{\"edited\": true}\n")
            (root / "E9/old.json").write_text("{}\n")
            (root / "E9/B.json").unlink()
            self.assertEqual(check(root, files), ["differs from the spec: E9/A.json",
                                                  "missing: E9/B.json",
                                                  "not produced by the spec: E9/old.json"])
            write(root, files)
            self.assertFalse((root / "E9/old.json").exists())


class ScriptTests(unittest.TestCase):
    def run_script(self, folder, *args):
        return subprocess.run([sys.executable, str(PROJECT / "scripts/make_arms.py"),
                               "--dir", str(folder), *args],
                              capture_output=True, text=True, timeout=60)

    def test_editing_the_spec_without_regenerating_fails_the_check(self):
        with tempfile.TemporaryDirectory() as folder:
            copy = Path(folder) / "arms"
            shutil.copytree(ARMS_DIR, copy)
            self.assertEqual(self.run_script(copy, "--check").returncode, 0)
            spec = json.loads((copy / "spec.json").read_text())
            spec["layers"]["r1"] = {"response_rounds": 2}
            (copy / "spec.json").write_text(json.dumps(spec))
            failed = self.run_script(copy, "--check")
            self.assertEqual(failed.returncode, 1)
            self.assertIn("differs from the spec: E02a/C.json", failed.stderr)
            self.assertEqual(self.run_script(copy).returncode, 0)
            self.assertEqual(self.run_script(copy, "--check").returncode, 0)

    def test_a_malformed_spec_exits_2(self):
        with tempfile.TemporaryDirectory() as folder:
            copy = Path(folder) / "arms"
            shutil.copytree(ARMS_DIR, copy)
            (copy / "measured.json").write_text("{not json")
            result = self.run_script(copy, "--check")
            self.assertEqual(result.returncode, 2)
            self.assertIn("not valid JSON", result.stderr)


if __name__ == "__main__":
    unittest.main()
