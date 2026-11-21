"""Draw the seeded, stratified test set from pinned FlashRAG splits.

Only identity, question text and gold answers are written. Passages and the
other source metadata stay in the ignored ``dataset/`` download. The selection
ranks candidates by ``SHA-256("<seed>:<dataset>:<id>")`` so the bytes are
independent of the Python version's random-module implementation, then keeps
the first ``--per-dataset`` rows. No question text is ever printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
    keys, texts = set(), set()
    for row in read_jsonl(path):
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
    """Usable rows sorted by id, unique by normalised text, dev-excluded."""
    selected, seen = [], set()
    for row in sorted(rows, key=lambda item: str(item.get('id'))):
        question = row.get('question')
        identity = (source.dataset, source.split, row.get('id'))
        if identity in dev_keys or not has_reference_answer(row):
            continue
        if not isinstance(question, str):
            continue
        text = normalise_question(question)
        if not text or text in dev_texts or text in seen:
            continue
        seen.add(text)
        selected.append(candidate(row, source))
    return selected


def rank_key(seed, dataset, question_id):
    material = f'{seed}:{dataset}:{question_id}'.encode('utf-8')
    return hashlib.sha256(material).hexdigest()


def draw(candidates, per_dataset, seed):
    ranked = sorted(candidates, key=lambda row: rank_key(seed, row['dataset'], row['id']))
    return ranked[:per_dataset]


def build_testset(data_dir, dev_path, per_dataset=DEFAULT_PER_DATASET, seed=DEFAULT_SEED):
    """Return (rows, report) for every configured source, in SOURCES order."""
    dev_keys, dev_texts = dev_exclusions(dev_path)
    rows, report = [], []
    for source in SOURCES:
        path = Path(data_dir) / source.path
        if not path.is_file():
            raise FileNotFoundError(f'missing source split: {path}')
        candidates = eligible_candidates(read_jsonl(path), source, dev_keys, dev_texts)
        selected = draw(candidates, per_dataset, seed)
        report.append({
            'dataset': source.dataset,
            'split': source.split,
            'eligible': len(candidates),
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
    print('dataset split eligible drawn shortfall')
    for entry in report:
        print(f"{entry['dataset']} {entry['split']} {entry['eligible']} "
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
    args.out.write_text(text, encoding='utf-8')
    print_report(report, text, args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
