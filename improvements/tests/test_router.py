"""J20: pre-debate router features, fitted model loading, cost decisions and routing.

The tests cover the feature maths and the leakage whitelist, every loader rejection, the
cost/accuracy trade-off, the deterministic tie-break, and the three route paths through
the real pipeline. The router is off by default; T0 covers the baseline separately.
"""
import copy
import json
import sys
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drag_plus.adapters import prompt_count  # noqa: E402
from drag_plus.flags import FlagError, Flags  # noqa: E402
from drag_plus.gates.router import (ROUTE_NAMES, RouterModelError, decide_route,  # noqa: E402
                                    load_router)
from drag_plus.gates.router_features import (FEATURE_INPUTS, ROUTER_FEATURE_NAMES,  # noqa: E402
                                            RouterInputs, answer_agreement, compute_features,
                                            extractive_answer, inputs_from_state,
                                            naive_entropy, retrieval_margin)
from drag_plus.judge.logprob import ScoredOutput  # noqa: E402
from drag_plus.pipeline import DragPlusPipeline  # noqa: E402
from fakes import FakeRetriever, RecordingItem, ScriptedGenerator  # noqa: E402

QUESTION = "who wrote it?"
DOCS = {"who wrote it?": [
    {"id": "d1", "contents": "Paris\nbody text", "score": 3.0},
    {"id": "d2", "contents": "Other\nbody text", "score": 1.0}]}


def model_data(**overrides):
    routes = {route: {"coefficients": [0.0, 0.0, 0.0, 0.0], "intercept": -100.0}
              for route in ROUTE_NAMES}
    routes["naive"]["intercept"] = 0.0
    data = {"format_version": 1, "feature_names": list(ROUTER_FEATURE_NAMES),
            "routes": routes, "costs": {"naive": 1.0, "refine_only": 4.0, "full_drag": 10.0},
            "feature_cost": 2.0, "cost_weight": 0.0, "training_split": "train",
            "manifest_hash": "abc"}
    data.update(overrides)
    return data


def write_model(directory, data=None):
    path = Path(directory) / "router.json"
    path.write_text(json.dumps(model_data() if data is None else data), encoding="utf-8")
    return str(path)


class RouterTemplate:
    """Serve both the Naive ``get_string(question=, retrieval_result=)`` and ``render``."""

    def get_string(self, messages=None, question=None, retrieval_result=None, **kwargs):
        if messages is not None:
            return copy.deepcopy(messages)
        return f"PROMPT {question} | {len(retrieval_result or [])} docs"


class ProbeGenerator(ScriptedGenerator):
    """Text calls from the scripted queue; every scored call returns one fixed margin."""

    def __init__(self, outputs, label_scores):
        super().__init__(outputs)
        self.label_scores = dict(label_scores)
        self.score_calls = []

    def generate_with_scores(self, prompts, labels, **params):
        self.score_calls.append((copy.deepcopy(prompts), list(labels), dict(params)))
        return [ScoredOutput("Proponent", dict(self.label_scores))
                for _ in range(prompt_count(prompts))]


def pipeline_for(model_path, outputs, generator=None, docs=None):
    values = {"max_query_debate_rounds": 1, "response_rounds": 1, "router": model_path}
    generator = generator or ScriptedGenerator(outputs)
    pipeline = DragPlusPipeline(Flags(values), generator, FakeRetriever(docs or DOCS),
                                RouterTemplate(), "NQ")
    return pipeline, generator, RecordingItem(QUESTION)


class FeatureTests(unittest.TestCase):
    def test_margin_and_entropy(self):
        self.assertEqual(retrieval_margin(DOCS[QUESTION]), 2.0)
        self.assertEqual(retrieval_margin([{"score": 2.0}]), 0.0)
        self.assertEqual(naive_entropy("Paris"), 0.0)
        self.assertEqual(naive_entropy(""), 0.0)
        self.assertGreater(naive_entropy("a b a c"), 0.0)
        self.assertLessEqual(naive_entropy("a b a c"), 1.0)

    def test_agreement_is_token_containment(self):
        self.assertEqual(answer_agreement("The answer is: Paris", "Paris"), 1.0)
        self.assertEqual(answer_agreement("New York", "New York City"), 1.0)
        self.assertEqual(answer_agreement("Paris", "London"), 0.0)
        self.assertEqual(answer_agreement("", "Paris"), 0.0)

    def test_extractive_answer_is_the_top_title(self):
        self.assertEqual(extractive_answer(DOCS[QUESTION]), "Paris")
        self.assertEqual(extractive_answer([]), "")

    def test_features_are_finite_ordered_and_whitelisted(self):
        inputs = RouterInputs(DOCS[QUESTION], 1.0, "Paris", "Paris")
        features = compute_features(inputs)
        self.assertEqual(tuple(features), ROUTER_FEATURE_NAMES)
        self.assertTrue(all(isinstance(value, float) for value in features.values()))
        self.assertEqual({field.name for field in fields(RouterInputs)}, set(FEATURE_INPUTS))
        self.assertEqual(features["retrieval_margin"], 2.0)
        self.assertGreater(features["s_bar_O"], 0.5)

    def test_inputs_from_state_ignore_gold_and_post_route_fields(self):
        pool = {"who wrote it?": DOCS[QUESTION]}
        state = SimpleNamespace(query_pool=pool, judge_score=0.0,
                                gold_answers=["Wrong"], pred="Wrong", status="done")
        first = compute_features(inputs_from_state(state, "Paris"))
        state.gold_answers, state.pred, state.status = ["Paris"], "Paris", "failed"
        self.assertEqual(first, compute_features(inputs_from_state(state, "Paris")))


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def test_loader_round_trip_and_tie_break(self):
        model = load_router(write_model(self.temp.name))
        zero = {name: 0.0 for name in ROUTER_FEATURE_NAMES}
        self.assertEqual(decide_route(zero, model).route, "naive")
        self.assertEqual(model.costs["full_drag"], 10.0)

    def test_loader_refuses_test_split_feature_drift_and_unknown_route(self):
        bad_routes = model_data()
        bad_routes["routes"].pop("full_drag")
        cases = ((model_data(training_split="test"), "test split"),
                 (model_data(feature_names=["other"]), "do not match"),
                 (bad_routes, "exactly"),
                 (model_data(format_version=2), "format_version"))
        for data, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                    RouterModelError, message):
                load_router(write_model(self.temp.name, data))

    def test_loader_refuses_missing_and_negative_costs(self):
        for data in (model_data(costs={"naive": -1.0, "refine_only": 4.0,
                                       "full_drag": 10.0}),
                     model_data(feature_cost=-1.0), model_data(cost_weight=-0.1)):
            with self.subTest(data=data), self.assertRaises(RouterModelError):
                load_router(write_model(self.temp.name, data))
        with self.assertRaises(RouterModelError):
            load_router(str(Path(self.temp.name) / "missing.json"))

    def test_higher_cost_weight_moves_to_the_cheaper_route(self):
        data = model_data(cost_weight=0.0)
        data["routes"]["naive"]["intercept"] = -1.0
        data["routes"]["full_drag"]["intercept"] = 1.0
        model = load_router(write_model(self.temp.name, data))
        zero = {name: 0.0 for name in ROUTER_FEATURE_NAMES}
        self.assertEqual(decide_route(zero, model, 0.0).route, "full_drag")
        self.assertEqual(decide_route(zero, model, 1.0).route, "naive")
        self.assertGreater(decide_route(zero, model, 0.0).costs["full_drag"],
                           decide_route(zero, model, 0.0).costs["naive"])


class FlagTests(unittest.TestCase):
    def test_router_defaults_keep_the_baseline(self):
        flags = Flags()
        self.assertEqual((flags.router, flags.router_cost_weight), (None, None))
        self.assertEqual(Flags({"router_cost_weight": 0}).router_cost_weight, 0.0)
        with self.assertRaises(FlagError):
            Flags({"router_cost_weight": -0.1})


class PipelineRouteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def test_naive_route_reuses_the_feature_answer(self):
        model = write_model(self.temp.name)
        pipeline, generator, item = pipeline_for(model, ["The answer is: Paris"])
        state = pipeline.run_item(item)
        self.assertEqual(state.router_route, "naive")
        self.assertEqual((state.pred, item.pred), ("Paris", "Paris"))
        self.assertEqual((state.query_messages, state.answer_messages), ({}, {}))
        self.assertEqual(len(generator.calls), 1)
        self.assertIn("Router_Route", dict(item.calls))
        self.assertEqual(state.router_features["retrieval_margin"], 2.0)
        self.assertEqual(state.router_features["answer_agreement"], 1.0)

    def test_refine_only_runs_the_query_debate_then_one_answer(self):
        data = model_data()
        data["routes"]["naive"]["intercept"] = -100.0
        data["routes"]["refine_only"]["intercept"] = 0.0
        model = write_model(self.temp.name, data)
        outputs = ["The answer is: Paris", "proponent argument", "no action here",
                   "Proponent", "The answer is: refined"]
        pipeline, generator, item = pipeline_for(model, outputs)
        state = pipeline.run_item(item)
        self.assertEqual(state.router_route, "refine_only")
        self.assertEqual((state.pred, state.answer_messages),
                         ("The answer is: refined", {}))
        self.assertIn("Opponent Agent 0", state.query_messages)
        self.assertEqual(len(generator.calls), 5)

    def test_full_drag_runs_both_debates(self):
        data = model_data()
        data["routes"]["naive"]["intercept"] = -100.0
        data["routes"]["full_drag"]["intercept"] = 0.0
        model = write_model(self.temp.name, data)
        outputs = ["The answer is: Paris", "proponent argument", "no action here",
                   "Proponent", "proponent answer", "opponent answer",
                   "The answer is: moderator"]
        pipeline, generator, item = pipeline_for(model, outputs)
        state = pipeline.run_item(item)
        self.assertEqual(state.router_route, "full_drag")
        self.assertEqual(state.pred, "The answer is: moderator")
        self.assertIn("Proponent Agent 0", state.answer_messages)
        self.assertEqual(len(generator.calls), 7)

    def test_decision_and_features_are_in_the_state_record(self):
        pipeline, _generator, item = pipeline_for(write_model(self.temp.name),
                                                  ["The answer is: Paris"])
        state = pipeline.run_item(item)
        entry = next(entry for entry in state.decisions if entry["decision"] == "route")
        self.assertEqual(entry["status"], "naive")
        self.assertEqual(set(entry["features"]), set(ROUTER_FEATURE_NAMES))
        self.assertEqual(set(entry["probabilities"]), set(ROUTE_NAMES))
        self.assertGreater(entry["cost"], 0.0)

    def test_scored_probe_sets_s_bar_o(self):
        model = write_model(self.temp.name)
        generator = ProbeGenerator(["The answer is: Paris"],
                                   {"Proponent": -0.1, "Opponent": -2.0})
        pipeline, _generator, item = pipeline_for(model, [], generator=generator)
        state = pipeline.run_item(item)
        self.assertEqual(len(generator.score_calls), 1)
        self.assertLess(state.router_features["s_bar_O"], 0.5)
        self.assertIsNotNone(state.decisions[0]["judge_probe"])


if __name__ == "__main__":
    unittest.main()
