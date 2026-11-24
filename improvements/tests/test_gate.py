"""J14: label-free gate features, model loading, risk decisions and lazy Opponent.

The tests cover the feature maths, the leakage property, every loader rejection,
threshold behaviour, the answer-stage call accounting and the lazy path with a
scored generator. Defaults stay baseline (T0 covers that separately).
"""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drag_plus.adapters import prompt_count  # noqa: E402
from drag_plus.flags import FlagError, Flags  # noqa: E402
from drag_plus.gates.escalation import (GateModelError, calibrated_risk, decide,  # noqa: E402
                                        load_model, resolve_sufficiency)
from drag_plus.gates.features import (FEATURE_NAMES, cluster_entropy,  # noqa: E402
                                      compute_features, disagreement, evidence_support,
                                      s_bar_o)
from drag_plus.judge.logprob import ScoredOutput  # noqa: E402
from drag_plus.pipeline import DragPlusPipeline  # noqa: E402
from fakes import FakeRetriever, IdentityTemplate, RecordingItem, ScriptedGenerator  # noqa: E402
from t0_scenarios import query_round  # noqa: E402

QUESTION = "who wrote it?"
PARIS = "The answer is: Paris"

def model_data(**overrides):
    data = {"feature_names": list(FEATURE_NAMES), "coefficients": [40.0, 0.0, 0.0, 0.0],
            "intercept": -10.0, "calibration": {"kind": "identity"},
            "training_split": "train", "manifest_hash": "abc"}
    data.update(overrides)
    return data


def write_model(directory, data=None):
    path = Path(directory) / "gate.json"
    path.write_text(json.dumps(model_data() if data is None else data), encoding="utf-8")
    return str(path)


def state_with(answers, judge_score=None, passages=None):
    messages = {f"Agent {index}": [{"role": "assistant", "content": answer}]
                for index, answer in enumerate(answers)}
    return SimpleNamespace(answer_messages=messages, judge_score=judge_score,
                           query_pool={"q": passages or []})


class ScoredScriptedGenerator(ScriptedGenerator):
    """Text calls from the baseline queue, scored calls from a second queue."""

    def __init__(self, outputs, scored):
        super().__init__(outputs)
        self.scored = list(scored)
        self.score_calls = []

    def generate_with_scores(self, prompts, labels, **params):
        self.score_calls.append((copy.deepcopy(prompts), list(labels), dict(params)))
        if not self.scored:
            raise AssertionError("scripted scored outputs exhausted")
        text, scores = self.scored.pop(0)
        return [ScoredOutput(text, scores) for _ in range(prompt_count(prompts))]


def run_gate(outputs, model_path, **flags):
    values = {"max_query_debate_rounds": 1, "response_rounds": 3, "moderator_final_only": True,
              "escalation_gate": "risk", "escalation_model": model_path, "escalation_tau": 0.5,
              **flags}
    generator = ScriptedGenerator(outputs)
    pipeline = DragPlusPipeline(Flags(values), generator, FakeRetriever(),
                                IdentityTemplate(), "NQ")
    item = RecordingItem(QUESTION)
    return pipeline.run_item(item), generator, item

def run_lazy(outputs, scored=None, generator=None, **flags):
    values = {"max_query_debate_rounds": 1, "response_rounds": 0, "lazy_query_opponent": True,
              "escalation_tau": 0.5, **flags}
    generator = generator or ScoredScriptedGenerator(outputs, scored or [])
    pipeline = DragPlusPipeline(Flags(values), generator, FakeRetriever(),
                                IdentityTemplate(), "NQ")
    item = RecordingItem(QUESTION)
    return pipeline.run_item(item), generator, item

class FeatureTests(unittest.TestCase):
    def test_disagreement_and_entropy(self):
        self.assertEqual(disagreement(["Paris", "paris."]), 0.0)
        self.assertEqual(disagreement(["Paris", "London"]), 0.5)
        self.assertAlmostEqual(disagreement(["Paris", "paris", "London"]), 1 / 3)
        self.assertEqual(disagreement(["only"]), 0.0)
        self.assertEqual(cluster_entropy(["Paris"]), 0.0)
        self.assertAlmostEqual(cluster_entropy(["Paris", "London"]), 1.0)

    def test_evidence_support_and_hook(self):
        self.assertEqual(evidence_support("Paris", ["The capital is Paris"]), 1.0)
        self.assertEqual(evidence_support("Berlin", ["The capital is Paris"]), 0.0)
        self.assertEqual(evidence_support("x", ["y"], scorer=lambda a, t: 0.25), 0.25)

    def test_opponent_pressure_is_a_probability(self):
        self.assertEqual((s_bar_o(None), s_bar_o(0.0)), (0.5, 0.5))
        self.assertGreater(s_bar_o(100.0), 0.99)
        self.assertLess(s_bar_o(-100.0), 0.01)

    def test_features_ignore_gold_labels(self):
        state = state_with([PARIS, "The answer is: London"], judge_score=1.0)
        state.gold_answers = ["Paris"]
        first = compute_features(state)
        state.gold_answers = ["London"]
        self.assertEqual((first, tuple(first)), (compute_features(state), FEATURE_NAMES))
        self.assertEqual(first["disagreement"], 0.5)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def test_loader_and_threshold(self):
        model = load_model(write_model(self.temp.name))
        zero = {name: 0.0 for name in FEATURE_NAMES}
        low, high = decide(zero, model, 0.5), decide({**zero, "disagreement": 1.0}, model, 0.5)
        self.assertEqual((low.stop, low.risk < 0.5), (True, True))
        self.assertEqual((high.stop, high.risk > 0.5), (False, True))

    def test_loader_refuses_test_split_and_feature_drift(self):
        cases = ((model_data(training_split="test"), "test split"),
                 (model_data(feature_names=["other"]), "do not match"))
        for data, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(GateModelError, message):
                load_model(write_model(self.temp.name, data))

    def test_loader_refuses_missing_and_unknown_calibration(self):
        with self.assertRaises(GateModelError):
            load_model(str(Path(self.temp.name) / "missing.json"))
        with self.assertRaisesRegex(GateModelError, "unknown calibration"):
            load_model(write_model(self.temp.name, model_data(calibration={"kind": "magic"})))
    def test_isotonic_and_platt_are_applied(self):
        identity = load_model(write_model(self.temp.name))
        platt = load_model(write_model(self.temp.name, model_data(
            calibration={"kind": "platt", "a": 0.0, "b": 0.0})))
        features = {name: 1.0 for name in FEATURE_NAMES}
        self.assertEqual(calibrated_risk(features, platt), 0.5)
        self.assertNotEqual(calibrated_risk(features, identity), 0.5)
        isotonic = load_model(write_model(self.temp.name, model_data(
            calibration={"kind": "isotonic", "x": [0.0, 1.0], "y": [0.2, 0.8]})))
        self.assertAlmostEqual(calibrated_risk(features, isotonic), 0.8)

    def test_sufficiency_resolution(self):
        good = ScoredOutput("sufficient", {"sufficient": -1.0, "insufficient": -4.0})
        missing = ScoredOutput("unknown", {"sufficient": None, "insufficient": -1.0})
        self.assertEqual((resolve_sufficiency(good, 0.5).skip,
                          resolve_sufficiency(good, 0.99).skip), (True, False))
        self.assertEqual(resolve_sufficiency(missing, 0.5).status, "unavailable")


class FlagTests(unittest.TestCase):
    def test_defaults_are_baseline(self):
        flags = Flags()
        self.assertEqual((flags.escalation_gate, flags.escalation_model,
                          flags.escalation_tau, flags.lazy_query_opponent),
                         ("none", None, None, False))

    def test_gate_and_lazy_require_their_thresholds(self):
        with self.assertRaises(FlagError):
            Flags({"escalation_gate": "risk", "escalation_tau": 0.5})
        with self.assertRaises(FlagError):
            Flags({"escalation_gate": "risk", "escalation_model": "x.json"})
        with self.assertRaises(FlagError):
            Flags({"lazy_query_opponent": True})
        with self.assertRaises(FlagError):
            Flags({"escalation_tau": 1.5})
        self.assertEqual(Flags({"escalation_tau": 1}).escalation_tau, 1.0)


class PipelineGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model = write_model(self.temp.name)

    def test_low_risk_stops_after_round_one(self):
        outputs = query_round("Proponent") + [PARIS, PARIS, PARIS]
        state, generator, _item = run_gate(outputs, self.model)
        self.assertEqual(state.answer_messages["Proponent Agent 0"][-1]["content"], PARIS)
        self.assertEqual(len(state.answer_messages["Opponent Agent 0"]), 3)
        gate = [entry for entry in state.decisions
                if entry["decision"] == "escalation_gate"]
        self.assertEqual(gate[0]["status"], "stop")
        self.assertEqual(len(generator.calls), 6)

    def test_high_risk_runs_one_more_round_then_stops(self):
        outputs = query_round("Proponent") + [PARIS, "The answer is: London", PARIS, PARIS, PARIS]
        state, generator, _item = run_gate(outputs, self.model)
        statuses = [entry["status"] for entry in state.decisions
                    if entry["decision"] == "escalation_gate"]
        self.assertEqual(statuses, ["continue", "stop"])
        self.assertEqual(len(generator.calls), 8)

    def test_cap_is_three_response_rounds(self):
        outputs = query_round("Proponent") + [PARIS, "The answer is: London"] * 3 + [PARIS]
        state, generator, _item = run_gate(outputs, self.model)
        self.assertEqual(len(state.answer_messages["Proponent Agent 0"]), 7)
        self.assertEqual(len(generator.calls), 10)
        statuses = [entry["status"] for entry in state.decisions
                    if entry["decision"] == "escalation_gate"]
        self.assertEqual(statuses, ["continue", "continue", "continue"])

    def test_missing_model_fails_loudly(self):
        values = {"escalation_gate": "risk", "escalation_tau": 0.5,
                  "escalation_model": str(Path(self.temp.name) / "no.json")}
        with self.assertRaises(GateModelError):
            DragPlusPipeline(Flags(values), ScriptedGenerator([]), FakeRetriever(),
                             IdentityTemplate(), "NQ")


class LazyOpponentTests(unittest.TestCase):
    def test_sufficient_evidence_skips_every_query_agent(self):
        scored = [("sufficient", {"sufficient": -0.1, "insufficient": -5.0})]
        generator = ScoredScriptedGenerator([PARIS], scored)
        state, generator, _item = run_lazy([PARIS], generator=generator)
        self.assertGreater(state.sufficiency_score, 0.5)
        self.assertEqual(state.query_messages, {})
        self.assertEqual((len(generator.calls), len(generator.score_calls)), (1, 1))
        self.assertEqual(state.decisions[0]["status"], "sufficient")

    def test_insufficient_evidence_runs_the_normal_opponent(self):
        scored = [("insufficient", {"sufficient": -5.0, "insufficient": -0.1})]
        state, generator, _item = run_lazy(query_round("Proponent") + [PARIS], scored)
        self.assertIn("Opponent Agent 0", state.query_messages)
        self.assertEqual(len(generator.score_calls), 1)
        self.assertEqual(state.decisions[0]["status"], "insufficient")

    def test_unavailable_score_falls_back_to_the_opponent(self):
        scored = [("unknown", {"sufficient": None, "insufficient": -1.0})]
        state, _generator, _item = run_lazy(query_round("Proponent") + [PARIS], scored)
        self.assertIn("Opponent Agent 0", state.query_messages)
        self.assertEqual(state.decisions[0]["status"], "unavailable")

    def test_generator_without_scoring_keeps_the_baseline_path(self):
        outputs = query_round("Proponent") + [PARIS]
        generator = ScriptedGenerator(outputs)
        state, _generator, _item = run_lazy(outputs, generator=generator)
        self.assertIn("Opponent Agent 0", state.query_messages)
        self.assertEqual(state.decisions[0]["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
