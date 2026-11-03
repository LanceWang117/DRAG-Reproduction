"""Exercise the real query-stage method with fake inference and retrieval."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / 'scripts'))
from prepare_source import build_runtime
sys.path.insert(0, str(build_runtime()))

try:
    from model.drag import DebateAugmentedRAG
    IMPORT_ERROR = None
except ImportError as error:
    IMPORT_ERROR = error


class QueryJudgeIntegrationTests(unittest.TestCase):
    def setUp(self):
        # These tests exercise the real method, but its module still needs the
        # FlashRAG imports. CI without the Linux runtime may skip this group.
        if IMPORT_ERROR is not None:
            if os.environ.get('DRAG_REQUIRE_INTEGRATION') == '1':
                self.fail(str(IMPORT_ERROR))
            self.skipTest(f'Full DRAG runtime required: {IMPORT_ERROR}')

    def make_pipeline(self, outputs, rounds=1):
        # Bypass model/index construction. Mock outputs follow the real call
        # order: Proponent argument, Opponent argument, Judge, optional retry.
        pipeline = object.__new__(DebateAugmentedRAG)
        pipeline.max_query_debate_rounds = rounds
        pipeline.agents_messages_query_stage = {'Proponent Agent 0': [], 'Opponent Agent 0': []}
        pipeline.generator = SimpleNamespace(generate=Mock(side_effect=[[v] for v in outputs]))
        pipeline.retriever = SimpleNamespace(search=Mock(return_value=['evidence']))
        pipeline.prompt_template = SimpleNamespace(get_string=lambda **kwargs: kwargs['messages'])
        pipeline.format_query_pool = lambda pool: 'Evidence summary'
        pipeline.maintain_query_pool = Mock(return_value={'refined': ['new evidence']})
        records = {}
        item = SimpleNamespace(question='question', update_output=records.__setitem__)
        return pipeline, item, records

    def test_valid_proponent_stops_without_retry_or_update(self):
        # Two arguments plus one valid Judge reply: no extra model call.
        p, item, records = self.make_pipeline(['argument', 'argument', 'Proponent'])
        self.assertEqual(p.query_stage_debate(item), {'question': ['evidence']})
        self.assertEqual(p.generator.generate.call_count, 3)
        p.maintain_query_pool.assert_not_called()
        self.assertEqual(records['QueryStage_Moderator_Round0_DecisionStatus'], 'valid_initial')

    def test_ambiguous_verdict_does_not_stop_before_retry(self):
        # Reproduce the original failure; only the retry's Opponent verdict
        # should authorize the proposed query-pool update.
        p, item, records = self.make_pipeline([
            'argument', 'proposal', 'Proponent Agent 1:\nOpponent Agent 1:', 'Opponent'
        ])
        self.assertEqual(p.query_stage_debate(item), {'refined': ['new evidence']})
        p.maintain_query_pool.assert_called_once_with({'question': ['evidence']}, 'proposal')
        self.assertEqual(p.generator.generate.call_count, 4)
        self.assertEqual(records['QueryStage_Moderator_Round0_Decision'], 'Opponent')

    def test_retry_can_select_proponent(self):
        # Retrying does not force refinement if the Judge selects Proponent.
        p, item, records = self.make_pipeline(['argument', 'argument', 'invalid', 'Proponent'])
        self.assertEqual(p.query_stage_debate(item), {'question': ['evidence']})
        p.maintain_query_pool.assert_not_called()
        self.assertEqual(records['QueryStage_Moderator_Round0_RetryOutput'], 'Proponent')

    def test_persistent_invalid_verdicts_are_bounded_and_never_update_pool(self):
        # Three rounds with one retry each must terminate without changing
        # the evidence pool or treating an invalid answer as convergence.
        p, item, records = self.make_pipeline(['argument', 'argument', 'invalid', 'invalid'] * 3, rounds=3)
        self.assertEqual(p.query_stage_debate(item), {'question': ['evidence']})
        p.maintain_query_pool.assert_not_called()
        self.assertEqual(p.generator.generate.call_count, 12)
        self.assertEqual(records['QueryStage_Moderator_Round2_DecisionStatus'], 'unresolved')


if __name__ == '__main__':
    unittest.main()
