"""Offline checks for the submission runner and bundled evidence."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from run_evaluation import MANIFEST, resolve_paths, select_questions
from verify_evidence import verify
from prepare_source import ORIGINAL, CHANGES, build_runtime


class SubmissionTests(unittest.TestCase):
    def test_fixed_sample(self):
        rows = select_questions(MANIFEST)
        self.assertEqual(len(rows), 200)
        self.assertEqual(len(select_questions(MANIFEST, '2wiki')), 40)
        self.assertEqual(len(select_questions(MANIFEST, 'NQ', 1)), 1)

    def test_duplicate_and_missing_answers(self):
        row = dict(dataset='NQ', split='dev', id='one', question='Question?', golden_answers=['Answer'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'questions.jsonl'
            path.write_text(json.dumps(row) + '\n' + json.dumps(row) + '\n')
            with self.assertRaises(ValueError):
                select_questions(path)
            row['golden_answers'] = []
            path.write_text(json.dumps(row) + '\n')
            with self.assertRaises(ValueError):
                select_questions(path)

    def test_path_resolution(self):
        result = resolve_paths(dict(model2path={'e5': 'models/e5'},
                                   index_path='index', corpus_path='/data/corpus'), Path('/project'))
        self.assertEqual(result['model2path']['e5'], '/project/models/e5')
        self.assertEqual(result['corpus_path'], '/data/corpus')

    def test_evidence(self):
        self.assertEqual(len(verify()), 6)

    def test_overlay_does_not_edit_original(self):
        # The runnable copy should use our fix, while the authors' files keep
        # their original hashes. This also catches a mistaken source path.
        paths = [p for p in ORIGINAL.rglob('*') if p.is_file()]
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        runtime = build_runtime()
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        for name in ['drag.py', 'judge_decision.py']:
            self.assertEqual((runtime / 'model' / name).read_bytes(), (CHANGES / name).read_bytes())


if __name__ == '__main__':
    unittest.main()
