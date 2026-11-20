"""T0: with default flags, drag_plus matches the frozen v1.0.1 pipeline call for call."""
import unittest

import frozen
from drag_plus.flags import Flags
from drag_plus.pipeline import DragPlusPipeline
from fakes import FakeRetriever, IdentityTemplate, RecordingItem, ScriptedGenerator
from t0_scenarios import SCENARIOS


class FrozenBaseline(frozen.DebateAugmentedRAG):
    def evaluate(self, dataset, do_eval=True):
        return dataset


def run_baseline(scenario, backends):
    gen, ret, template = backends
    pipeline = FrozenBaseline(
        {"dataset_name": scenario.dataset}, prompt_template=template,
        max_query_debate_rounds=scenario.query_rounds,
        max_answer_debate_rounds=scenario.answer_rounds,
        generator=gen, retriever=ret)
    items = [RecordingItem(q, f"q{i}") for i, q in enumerate(scenario.questions)]
    pipeline.run(items)
    return items


def run_drag_plus(scenario, backends):
    gen, ret, template = backends
    flags = Flags({"max_query_debate_rounds": scenario.query_rounds,
                   "max_answer_debate_rounds": scenario.answer_rounds})
    pipeline = DragPlusPipeline(flags, gen, ret, template, scenario.dataset)
    items = [RecordingItem(q, f"q{i}") for i, q in enumerate(scenario.questions)]
    pipeline.run(items)
    return items


def fresh_backends(scenario):
    return ScriptedGenerator(scenario.outputs), FakeRetriever(), IdentityTemplate()


def transcript(items, backends):
    gen, ret, template = backends
    return {
        "generator_inputs": gen.calls,
        "retriever_calls": ret.calls,
        "template_calls": template.calls,
        "update_output": [item.calls for item in items],
        "pools": [item.output["QueryStage_QueryPool"] for item in items],
        "preds": [item.pred for item in items],
        "unused_outputs": gen.outputs,
    }


class GoldenTranscriptTests(unittest.TestCase):
    def check(self, scenario):
        baseline_backends = fresh_backends(scenario)
        baseline = transcript(run_baseline(scenario, baseline_backends), baseline_backends)
        plus_backends = fresh_backends(scenario)
        plus = transcript(run_drag_plus(scenario, plus_backends), plus_backends)
        self.assertEqual(baseline["unused_outputs"], [], "scenario script has spare outputs")
        for part in baseline:
            with self.subTest(scenario=scenario.name, part=part):
                self.assertEqual(plus[part], baseline[part])
        return plus


def _add_test(scenario):
    def test(self):
        plus = self.check(scenario)
        for fragment in scenario.expect_pool_keys:
            self.assertIn(fragment, plus["pools"][-1])
    return test


for _scenario in SCENARIOS:
    setattr(GoldenTranscriptTests, f"test_{_scenario.name}", _add_test(_scenario))


if __name__ == "__main__":
    unittest.main()
