"""Retrieval-only evaluation: Containment@k, MRR, nDCG@k, title diversity, latency.

Nothing is retrieved and no model is loaded; the script scores candidate lists that
already exist. Two inputs are accepted:

* ``--retrieval`` a stored dump shaped like ``results/retrieval-200.jsonl`` (``naive`` is
  a passage list, ``drag`` maps each query to one), scored per arm;
* ``--candidates`` one record per question with ``question``, ``golden_answers`` and
  ``candidates``.

``--timings`` adds p50/p99 retrieval latency from a run's ``stage-timings.jsonl`` (rows
whose ``stage`` is ``retrieval``). The metrics themselves are defined in
``drag_plus/analysis/retrieval.py``.
"""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'improvements'))

from drag_plus.analysis.retrieval import evaluate, latency_summary, summarise  # noqa: E402
from drag_plus.instrument.logs import write_json_atomic  # noqa: E402

QUESTIONS = PROJECT / 'results' / 'questions-200.jsonl'
ARMS = ('drag', 'naive')


def read_jsonl(path):
    text = Path(path).read_text(encoding='utf-8')
    return [json.loads(line) for line in text.split('\n') if line.strip()]


def load_questions(path):
    return {row['id']: row for row in read_jsonl(path)}


def unique(passages):
    """First occurrence of each passage id, so a pooled arm is not double counted."""
    seen, ordered = set(), []
    for passage in passages:
        key = str(passage.get('id'))
        if key not in seen:
            seen.add(key)
            ordered.append(passage)
    return ordered


def stored_candidates(record, arm):
    """The committed dump's shape: Naive is a list, DRAG is a query -> list mapping."""
    if arm == 'naive':
        return list(record.get('naive') or [])
    pool = [passage for query in (record.get('drag') or {}).values() for passage in query]
    return unique(pool)


def score_rows(rows, ks, ndcg_k):
    return [evaluate(question, golds, passages, ks, ndcg_k)
            for _, question, golds, passages in rows]


def from_stored(retrieval_path, questions_path, arms, ks, ndcg_k):
    questions = load_questions(questions_path)
    records = read_jsonl(retrieval_path)
    report = {}
    for arm in arms:
        rows = [(record['id'], questions[record['id']]['question'],
                 questions[record['id']]['golden_answers'], stored_candidates(record, arm))
                for record in records if record['id'] in questions]
        report[arm] = summarise(score_rows(rows, ks, ndcg_k))
    return report


def from_candidates(path, ks, ndcg_k):
    rows = [(row['id'], row['question'], row['golden_answers'], row.get('candidates') or [])
            for row in read_jsonl(path)]
    return {'candidates': summarise(score_rows(rows, ks, ndcg_k))}


def latency(path):
    """p50/p99 retrieval latency per (variant, arm, dataset) from the timing log."""
    groups = {}
    for row in read_jsonl(path):
        if row.get('stage') != 'retrieval':
            continue
        key = ' / '.join(str(row.get(field, '')) for field in ('variant', 'arm', 'dataset'))
        groups.setdefault(key, []).append(float(row['elapsed_ms']))
    return {key: latency_summary(values) for key, values in sorted(groups.items())}


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--retrieval', type=Path,
                        help='Stored dump with drag/naive passages per question')
    source.add_argument('--candidates', type=Path,
                        help='JSONL with id, question, golden_answers, candidates')
    parser.add_argument('--questions', type=Path, default=QUESTIONS,
                        help='Question file with golden_answers (default: results/questions-200.jsonl)')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--ks', default='3,5', help='Comma-separated containment depths')
    parser.add_argument('--ndcg-k', type=int, default=5)
    parser.add_argument('--timings', type=Path, help='stage-timings.jsonl from a run')
    parser.add_argument('--out', type=Path, help='Write the full report as JSON here')
    args = parser.parse_args(argv)
    args.ks = [int(value) for value in args.ks.split(',') if value.strip()]
    if not args.ks or min(args.ks) < 1 or args.ndcg_k < 1:
        parser.error('depths must be positive integers')
    if args.candidates and args.timings:
        parser.error('--timings belongs to a run, not to a candidate dump')
    return args


def print_table(report):
    metrics = [key for key in next(iter(report.values())) if key != 'n']
    print('arm'.ljust(12), 'n'.rjust(4), *[key.rjust(14) for key in metrics])
    for arm, values in report.items():
        cells = [f'{values[key]:.3f}'.rjust(14) for key in metrics]
        print(arm.ljust(12), str(values['n']).rjust(4), *cells)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.retrieval:
        report = from_stored(args.retrieval, args.questions, args.arms, args.ks, args.ndcg_k)
    else:
        report = from_candidates(args.candidates, args.ks, args.ndcg_k)
    print_table(report)
    if args.timings:
        report['latency'] = latency(args.timings)
        for key, values in report['latency'].items():
            print('retrieval latency', key, values)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(args.out, report)
        print('Saved:', args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
