import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import cloud_multiset_batch as multiset

spec = importlib.util.spec_from_file_location('naive100', Path(__file__).with_name('cloud_naive_100.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class NaiveInputTests(unittest.TestCase):
    def fixture(self, root):
        rows = [dict(id=str(i), question=f'Q{i}', golden_answers=['A']) for i in range(50)]
        nq = root / 'nq/fixed-dataset/NQ/dev.jsonl'
        nq.parent.mkdir(parents=True)
        nq.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        for name, split in multiset.SPECS:
            path = root / 'repo/dataset' / name / (split + '.jsonl')
            path.parent.mkdir(parents=True)
            path.write_text(nq.read_text())
        multiset.prepare(root / 'paired', root / 'repo', root / 'nq')
        return root / 'paired'

    def test_exact_100_inputs_are_copied_without_resampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paired = self.fixture(root)
            manifest = module.prepare(root / 'naive', paired)
            self.assertEqual(manifest['questions_to_run'], 100)
            self.assertFalse(manifest['resampled'])
            self.assertTrue(all(e['action'] == 'run_now' for e in manifest['entries']))
            self.assertEqual((root / 'naive/question-manifest-100.jsonl').read_bytes(),
                             (paired / 'question-manifest-100.jsonl').read_bytes())

    def test_tampered_subset_is_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paired = self.fixture(root)
            (paired / 'fixed-datasets/NQ/dev.jsonl').write_text('{}\n')
            with self.assertRaises(ValueError):
                module.prepare(root / 'naive', paired)

    def test_question_gold_mismatch_is_rejected_even_with_updated_manifest_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paired = self.fixture(root)
            canonical = paired / 'question-manifest-100.jsonl'
            rows = [json.loads(line) for line in canonical.read_text().splitlines()]
            rows[0]['golden_answers'] = ['Changed']
            canonical.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            manifest_path = paired / 'sample-manifest.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['question_manifest_sha256'] = module.sha(canonical)
            module.write_json(manifest_path, manifest)
            with self.assertRaises(ValueError):
                module.prepare(root / 'naive', paired)


if __name__ == '__main__':
    unittest.main()
