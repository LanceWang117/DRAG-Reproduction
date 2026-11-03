import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('multiset', Path(__file__).with_name('cloud_multiset_batch.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MultisetTests(unittest.TestCase):
    def test_selection_has_100_fixed_questions_and_preserves_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nq = root / 'nq/fixed-dataset/NQ/dev.jsonl'
            nq.parent.mkdir(parents=True)
            rows = [dict(id=str(i), question=f'Q{i}', golden_answers=['A']) for i in range(50)]
            nq.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            for name, split in module.SPECS:
                path = root / f'repo/dataset/{name}/{split}.jsonl'
                path.parent.mkdir(parents=True)
                path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            first = module.prepare(root / 'run1', root / 'repo', root / 'nq')
            second = module.prepare(root / 'run2', root / 'repo', root / 'nq')
            self.assertEqual([e['ordered_ids'] for e in first['entries']],
                             [e['ordered_ids'] for e in second['entries']])
            self.assertEqual([e['count'] for e in first['entries']], [50, 10, 10, 10, 10, 10])
            self.assertEqual(first['entries'][-1]['split'], 'train')
            self.assertEqual(first['entries'][2]['split'], 'test')
            self.assertEqual(len((root / 'run1/question-manifest-100.jsonl').read_text().splitlines()), 100)
            self.assertEqual((root / 'run1/fixed-datasets/NQ/dev.jsonl').read_text(), nq.read_text())

    def test_timing_and_checkpoints_happen_only_during_first_traversal(self):
        class Item:
            id = 'id1'
            output = {}
            def update_output(self, key, value):
                self.output[key] = value
            def to_dict(self):
                return dict(id=self.id, output=self.output)
        with tempfile.TemporaryDirectory() as directory:
            tracked = module.TimedDataset([Item()], Path(directory), 'dataset', 'dev')
            list(tracked)
            list(tracked)
            self.assertEqual(len(tracked.elapsed), 1)
            self.assertEqual(len((Path(directory) / 'question-checkpoints.jsonl').read_text().splitlines()), 1)
            self.assertEqual(json.loads((Path(directory) / 'progress.json').read_text())['completed'], 1)


if __name__ == '__main__':
    unittest.main()
