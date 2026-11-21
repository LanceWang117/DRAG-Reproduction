"""Approximate the recall effect of the anchored evidence store from stored passages.

The committed dump has the final DRAG pool only. An Optimization that deleted the
original query's passages is invisible there, so this replay substitutes the Naive
arm's passages as a proxy for the deleted original retrieval. That proxy is imperfect
(B2: Naive and DRAG retrieve different passages), so the numbers are an estimate, not
a rerun. The script reports gold-string recall for:

* ``baseline``: the final DRAG pool exactly as stored;
* ``anchored``: the same pool plus the Naive proxy unioned by passage id.

No model, tokenizer or index is loaded. Only the 175 non-StrategyQA dev questions are
scored, because StrategyQA answers are yes/no and gold-string containment is not
meaningful there.
"""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'improvements'))
from drag_plus.analysis.retrieval import contains_gold  # noqa: E402

QUESTIONS = PROJECT / 'results' / 'questions-200.jsonl'
RETRIEVAL = PROJECT / 'results' / 'retrieval-200.jsonl'
CAVEAT = ('Proxy caveat: the Naive arm\'s passages stand in for the original retrieval '
          'that Optimization may have deleted (B2); this is an approximation, not a rerun.')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def load_questions(path):
    return {row['id']: row for row in read_jsonl(path)}


def unique(passages):
    """First occurrence of each passage id, preserving deterministic order."""
    seen, ordered = set(), []
    for passage in passages:
        key = str(passage.get('id'))
        if key not in seen:
            seen.add(key)
            ordered.append(passage)
    return ordered


def final_pool(record):
    """The stored DRAG pool, flattened across its final queries."""
    return unique([passage for query in (record.get('drag') or {}).values()
                   for passage in query])


def anchored_pool(record):
    """The final pool plus the Naive proxy for the deleted original retrieval."""
    return unique(final_pool(record) + list(record.get('naive') or []))


def recalled(passages, golden_answers):
    """True when any single passage contains some gold answer as a string."""
    return any(contains_gold(str(passage.get('contents', '')), golden_answers)
               for passage in passages)


def score(policy, records, questions):
    rows = [(record, questions[record['id']]) for record in records
            if record['id'] in questions
            and questions[record['id']].get('dataset') != 'StrategyQA']
    hits = sum(recalled(policy(record), question.get('golden_answers') or [])
               for record, question in rows)
    return {'n': len(rows), 'recall_count': hits,
            'recall': hits / len(rows) if rows else 0.0}


def run(retrieval_path=RETRIEVAL, questions_path=QUESTIONS):
    """Score both policies; returns ``{policy: {n, recall_count, recall}}``."""
    questions = load_questions(questions_path)
    records = read_jsonl(retrieval_path)
    return {'baseline': score(final_pool, records, questions),
            'anchored': score(anchored_pool, records, questions)}


def print_report(report):
    print('policy      n  recall_count  recall')
    for policy, values in report.items():
        print(f"{policy:<10} {values['n']:>3} {values['recall_count']:>13}  "
              f"{values['recall']:.4f}")
    print(CAVEAT)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--retrieval', type=Path, default=RETRIEVAL,
                        help='Stored dump with drag/naive passages per question')
    parser.add_argument('--questions', type=Path, default=QUESTIONS,
                        help='Question file with golden_answers')
    parser.add_argument('--out', type=Path, help='Write the report as JSON here')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = run(args.retrieval, args.questions)
    print_report(report)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n',
                            encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
