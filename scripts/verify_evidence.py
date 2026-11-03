"""Check the compact results package without a GPU, network, or model calls."""
import csv
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / 'results'


def load(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify():
    # Hashes detect accidental edits to the extracted records; they are not
    # proof of authorship. The full raw runs remain in the local archive.
    hashes = load(RESULTS / 'checksums.json')
    for name, expected in hashes.items():
        path = RESULTS / name
        require(path.is_file(), f'Missing file: {name}')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                f'Checksum differs: {name}')
    comparison = load(RESULTS / 'paired-200.json')
    pairs = comparison['pairs']
    identity = lambda r: (r['dataset'], r['split'], r['id'])
    by_id = {identity(row): row for row in pairs}
    questions = [json.loads(s) for s in (RESULTS / 'questions-200.jsonl').read_text().splitlines()]
    require(len(pairs) == len(by_id) == len(questions) == 200, 'Expected 200 unique questions')
    require(len({identity(q) for q in questions}) == 200, 'Duplicate input question')
    for row in questions:
        pair = by_id[identity(row)]
        require((row['question'], row['golden_answers']) ==
                (pair['question'], pair['golden_answers']), f'Input mismatch: {identity(row)}')
        for method in ['drag', 'naive']:
            require(isinstance(pair[method + '_answer'], str), 'Missing prediction')
            require(pair[method + '_metrics']['em'] in (0, 1), 'Invalid EM')
            require(0 <= pair[method + '_metrics']['f1'] <= 1, 'Invalid F1')
    retrieval = [json.loads(s) for s in (RESULTS / 'retrieval-200.jsonl').read_text().splitlines()]
    require(len(retrieval) == 200 and {identity(r) for r in retrieval} == set(by_id),
            'Retrieval records do not cover the fixed sample')
    for row in retrieval:
        # DRAG stores one top-3 list per final query, not three passages in total.
        require(bool(row['drag']), 'Empty DRAG query pool')
        groups = [row['naive'], *row['drag'].values()]
        require(all(len(g) == 3 and all('id' in d and 'contents' in d for d in g) for g in groups),
                f'Missing passage ID/text: {identity(row)}')
    summary = []
    for dataset in comparison['datasets']:
        rows = [r for r in pairs if r['dataset'] == dataset['dataset']]
        record = dict(dataset=dataset['dataset'], questions=len(rows))
        for method in ['naive', 'drag']:
            for metric in ['em', 'f1']:
                value = sum(r[method + '_metrics'][metric] for r in rows) / len(rows)
                require(abs(value - dataset[method + '_metrics'][metric]) < 1e-12,
                        'Aggregate differs from per-question scores')
                record[f'{method}_{metric}'] = str(
                    (Decimal(str(value)) * 100).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
        summary.append(record)
    with (RESULTS / 'summary.csv').open(newline='') as source:
        expected = list(csv.DictReader(source))
    require([{k: str(v) for k, v in row.items()} for row in summary] == expected, 'Summary CSV differs')
    print(f'Verified {len(hashes)} files: 200 paired answers, selected inputs, retrieval records, and summary.')
    return summary


if __name__ == '__main__':
    verify()
