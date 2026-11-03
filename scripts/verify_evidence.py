"""Check saved evidence locally and recompute the published aggregate scores.

Uses only Python's standard library. No downloads, model calls, or GPU access.
"""
import csv
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / 'results'
RUNS = ['20261017T010415Z_drag_50q', '20261017T015401Z_drag_extra50_v1.0.1',
        '20261017T030736Z_naive_rag_fixed100', '20261017T033818Z_paired_extension100']


def load(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify():
    checksums = load(RESULTS / 'checksums.json')
    for name, expected in checksums.items():
        path = RESULTS / name
        require(path.is_file(), f'Missing evidence: {name}')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                f'Evidence checksum differs: {name}')
    comparison = load(RESULTS / 'fixed200-drag-naive-comparison.json')
    pairs = comparison['pairs']
    key = lambda r: (r['dataset'], r['split'], r['id'])
    by_id = {key(r): r for r in pairs}
    require(len(pairs) == len(by_id) == 200, 'Expected 200 unique paired questions')
    manifest_path = RESULTS / RUNS[-1] / 'question-manifest-200.jsonl'
    manifest = [json.loads(s) for s in manifest_path.read_text().splitlines() if s.strip()]
    require(len(manifest) == 200, 'Wrong manifest length')
    for row in manifest:
        pair = by_id[key(row)]
        require((row['question'], row['golden_answers']) ==
                (pair['question'], pair['golden_answers']), f'Manifest mismatch: {key(row)}')

    seen = {'drag': set(), 'naive': set()}
    for run in RUNS:
        for path in (RESULTS / run).rglob('intermediate_data.json'):
            method = 'naive' if run == RUNS[2] or 'Naive' in path.parts else 'drag'
            for row in load(path):
                # Historical files do not always include the dataset name in
                # each item. Resolve it using the run path, not ID alone.
                candidates = [p for p in pairs if p['id'] == row['id']
                              and p['question'] == row['question']
                              and p['golden_answers'] == row['golden_answers']]
                require(len(candidates) == 1, f'Ambiguous saved question: {path}, {row["id"]}')
                pair = candidates[0]
                identity = key(pair)
                require(identity not in seen[method], f'Duplicate raw result: {method}, {identity}')
                seen[method].add(identity)
                output = row['output']
                require(output['pred'] == pair[method + '_answer'], f'Answer mismatch: {identity}')
                for metric in ['em', 'f1']:
                    require(abs(output['metric_score'][metric] - pair[method + '_metrics'][metric]) < 1e-12,
                            f'Metric mismatch: {identity}, {metric}')
                documents = (output['QueryStage_QueryPool'].values() if method == 'drag'
                             else [output['retrieval_result']])
                require(all(len(group) == 3 and all('id' in d and 'contents' in d for d in group)
                            for group in documents), f'Incomplete retrieval records: {identity}')
    require(all(len(ids) == 200 for ids in seen.values()), 'Missing raw results for one method')
    summary = []
    for dataset in comparison['datasets']:
        rows = [p for p in pairs if p['dataset'] == dataset['dataset']]
        record = dict(dataset=dataset['dataset'], questions=len(rows))
        for method in ['naive', 'drag']:
            for metric in ['em', 'f1']:
                value = sum(p[method + '_metrics'][metric] for p in rows) / len(rows)
                require(abs(value - dataset[method + '_metrics'][metric]) < 1e-12,
                        'Aggregate metric differs from saved per-question scores')
                record[f'{method}_{metric}'] = str(
                    (Decimal(str(value)) * 100).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
        summary.append(record)
    # The summary CSV is an independently recomputed view, not new model output.
    with (RESULTS / 'summary.csv').open(newline='') as source:
        expected = list(csv.DictReader(source))
    require([{k: str(v) for k, v in r.items()} for r in summary] == expected, 'Summary CSV differs')
    print(f'Verified {len(checksums)} evidence files, 200 paired questions, and top-3 passage records.')
    for row in summary:
        print(row)
    return summary


if __name__ == '__main__':
    verify()
