import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('drag_judge_decision', ROOT / 'model/judge_decision.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
parse = module.parse_judge_decision
resolve = module.resolve_judge_decision


class JudgeDecisionTests(unittest.TestCase):
    def test_single_role_formats(self):
        for text, expected in (
            ('Proponent', 'Proponent'), ('Opponent', 'Opponent'),
            ('  opponent agent 0\n', 'Opponent'),
            ('Proponent Agent', 'Proponent'),
            ('Proponent Agent 1.', 'Proponent'),
        ):
            with self.subTest(text=text):
                self.assertEqual(parse(text), expected)

    def test_invalid_outputs_do_not_choose_either_side(self):
        for text in ('Proponent Agent 1: \nOpponent Agent 1:',
                     'Proponent or Opponent', 'Opponent beats Proponent',
                     'Not Proponent', 'I choose Proponent',
                     'Proponent\nProponent', 'Proponent Agent 0:', '', None):
            with self.subTest(text=text):
                self.assertIsNone(parse(text))

    def test_valid_output_does_not_add_a_model_call(self):
        retry = Mock()
        result = resolve('Opponent Agent 0', retry)
        self.assertEqual((result.decision, result.status), ('Opponent', 'valid_initial'))
        retry.assert_not_called()

    def test_invalid_output_retries_once_and_accepts_either_valid_side(self):
        for choice in ('Proponent', 'Opponent'):
            retry = Mock(return_value=choice)
            result = resolve('Proponent Agent 1:\nOpponent Agent 1:', retry)
            self.assertEqual((result.decision, result.status), (choice, 'valid_retry'))
            self.assertEqual(result.retry_output, choice)
            retry.assert_called_once_with()

    def test_second_invalid_output_is_explicitly_unresolved(self):
        retry = Mock(return_value='Neither is convincing')
        result = resolve('Proponent and Opponent', retry)
        self.assertIsNone(result.decision)
        self.assertEqual(result.status, 'unresolved')
        retry.assert_called_once_with()

    def test_generation_errors_are_not_hidden_as_decisions(self):
        with self.assertRaises(RuntimeError):
            resolve('invalid', Mock(side_effect=RuntimeError('generation failed')))


if __name__ == '__main__':
    unittest.main()
