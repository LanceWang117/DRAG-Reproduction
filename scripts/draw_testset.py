"""Draw the seeded, stratified test set from pinned FlashRAG splits.

Only identity, question text and gold answers are written. Passages and the
other source metadata stay in the ignored ``dataset/`` download. Within each
dataset, candidates are ranked by ``SHA-256("<seed>:<dataset>:<id>")``; the
scan then keeps the first rows with unseen normalised question text, so the
bytes are independent of the Python version's random-module implementation. No
question text is ever printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REVISION = 'bcafb8dd07d453be3cbeeeb3f78be1841bddf92c'
DEFAULT_DATA = PROJECT / 'dataset' / f'flashrag-{REVISION[:8]}'
DEFAULT_DEV = PROJECT / 'results' / 'questions-200.jsonl'
DEFAULT_OUT = PROJECT / 'experiments' / 'J05-testset' / 'questions-test.jsonl'
DEFAULT_SEED = '20261121'
DEFAULT_PER_DATASET = 100


@dataclass(frozen=True)
class Source:
    dataset: str
    path: str
    split: str


SOURCES = (
    Source('2wiki', '2wikimultihopqa/dev.jsonl', 'dev'),
    Source('HotpotQA', 'hotpotqa/dev.jsonl', 'dev'),
    Source('NQ', 'nq/test.jsonl', 'test'),
    Source('PopQA', 'popqa/test.jsonl', 'test'),
    Source('StrategyQA', 'strategyqa/train.jsonl', 'train'),
    Source('TriviaQA', 'triviaqa/test.jsonl', 'test'),
)


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding='utf-8') as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f'{path}:{number}: invalid JSON') from error
    return rows


def normalise_question(text):
    """NFKC, case-fold, keep alphanumerics, collapse whitespace."""
    folded = unicodedata.normalize('NFKC', text).casefold()
    return ' '.join(''.join(char if char.isalnum() else ' ' for char in folded).split())


def dev_exclusions(path):
    """Dev identities and normalised question texts that no test row may match."""
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f'dev question file is empty: {path}')
    keys, texts = set(), set()
    for row in rows:
        if (not row.get('dataset') or not row.get('split') or row.get('id') is None
                or not isinstance(row.get('question'), str)):
            raise ValueError(f'dev row lacks dataset, split, id or question: {path}')
        keys.add((row['dataset'], row['split'], str(row['id'])))
        text = normalise_question(row['question'])
        if text:
            texts.add(text)
    return keys, texts


def has_reference_answer(row):
    answers = row.get('golden_answers')
    return (isinstance(answers, list) and bool(answers)
            and all(isinstance(answer, str) and answer.strip() for answer in answers))


def candidate(row, source):
    return {
        'dataset': source.dataset,
        'split': source.split,
        'id': str(row['id']),
        'question': row['question'],
        'golden_answers': list(row['golden_answers']),
    }


def eligible_candidates(rows, source, dev_keys, dev_texts):
    """Rows passing the identity, question-text and answer filters, sorted by id."""
    selected = []
    for row in sorted(rows, key=lambda item: str(item.get('id'))):
        question, raw_id = row.get('question'), row.get('id')
        if raw_id is None or not has_reference_answer(row):
            continue
        if not isinstance(question, str):
            continue
        identity = (source.dataset, source.split, str(raw_id))
        text = normalise_question(question)
        if identity in dev_keys or not text or text in dev_texts:
            continue
        selected.append(candidate(row, source))
    return selected


def distinct_texts(candidates):
    return {normalise_question(row['question']) for row in candidates}


def rank_key(seed, dataset, question_id):
    material = f'{seed}:{dataset}:{question_id}'.encode('utf-8')
    return hashlib.sha256(material).hexdigest()


def draw(candidates, per_dataset, seed, seen=None):
    """Rank first, then keep rows whose normalised text is not already selected."""
    seen = set() if seen is None else seen
    chosen = []
    for row in sorted(candidates, key=lambda item: rank_key(seed, item['dataset'], item['id'])):
        text = normalise_question(row['question'])
        if text in seen:
            continue
        seen.add(text)
        chosen.append(row)
        if len(chosen) == per_dataset:
            break
    return chosen


def build_testset(data_dir, dev_path, per_dataset=DEFAULT_PER_DATASET, seed=DEFAULT_SEED):
    """Return (rows, report) for every configured source, in SOURCES order."""
    dev_keys, dev_texts = dev_exclusions(dev_path)
    rows, report, seen = [], [], set()
    for source in SOURCES:
        path = Path(data_dir) / source.path
        if not path.is_file():
            raise FileNotFoundError(f'missing source split: {path}')
        candidates = eligible_candidates(read_jsonl(path), source, dev_keys, dev_texts)
        selected = draw(candidates, per_dataset, seed, seen)
        report.append({
            'dataset': source.dataset,
            'split': source.split,
            'rows': len(candidates),
            'distinct': len(distinct_texts(candidates)),
            'drawn': len(selected),
            'shortfall': max(0, per_dataset - len(selected)),
        })
        rows.extend(selected)
    for position, row in enumerate(rows):
        row['position'] = position
    return rows, report


def render(rows):
    """One sorted-key JSON object per line; stable across runs and versions."""
    return ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--dev', type=Path, default=DEFAULT_DEV)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--per-dataset', type=int, default=DEFAULT_PER_DATASET)
    parser.add_argument('--seed', default=DEFAULT_SEED)
    return parser.parse_args(argv)


def print_report(report, text, out):
    print(f'python {sys.version.split()[0]} unicode {unicodedata.unidata_version}')
    print('dataset split rows distinct drawn shortfall')
    for entry in report:
        print(f"{entry['dataset']} {entry['split']} {entry['rows']} {entry['distinct']} "
              f"{entry['drawn']} {entry['shortfall']}")
    payload = text.encode('utf-8')
    print(f'total {sum(entry["drawn"] for entry in report)} rows, {len(payload)} bytes, '
          f'sha256 {hashlib.sha256(payload).hexdigest()}, out {out}')


def main(argv=None):
    args = parse_args(argv)
    if args.per_dataset < 1:
        raise SystemExit('--per-dataset must be positive')
    rows, report = build_testset(args.data_dir, args.dev, args.per_dataset, args.seed)
    text = render(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(text.encode('utf-8'))
    print_report(report, text, args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
