"""Rebuild ``zero_novelty.json`` from the frozen 200-question retrieval record.

The fixture keeps only query strings and passage IDs, never passage text. It holds the
three multi-query zero-novelty steps named by B15 and five single-replacement pools whose
replacement adds no ID to Naive's pool. The original question's own passage IDs are not
recorded for a replaced pool, so those five use Naive's IDs as the pre-action pool and
are marked approximate (B2).

Usage: python improvements/tests/fixtures/build_zero_novelty.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RETRIEVAL = ROOT / "results" / "retrieval-200.jsonl"
QUESTIONS = ROOT / "results" / "questions-200.jsonl"
OUTPUT = Path(__file__).resolve().with_name("zero_novelty.json")
SINGLE_LIMIT = 5


def _load(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _ids(passages: list) -> list:
    return [passage["id"] for passage in passages]


def _multi_case(row: dict):
    steps, seen, exit_step = [], set(), None
    for query, passages in row["drag"].items():
        ids = _ids(passages)
        steps.append({"query": query, "ids": ids})
        if exit_step is None and not set(ids) - seen:
            exit_step = len(steps) - 1
        seen |= set(ids)
    if exit_step is None:
        return None
    return {"case": f"{row['dataset']}/{row['id']}", "kind": "multi",
            "steps": steps, "exit_step": exit_step}


def _single_case(row: dict, question: str):
    query, passages = next(iter(row["drag"].items()))
    steps = [{"query": question, "ids": _ids(row["naive"])},
             {"query": query, "ids": _ids(passages)}]
    return {"case": f"{row['dataset']}/{row['id']}", "kind": "single_replacement",
            "steps": steps, "exit_step": 1, "approximate": "pre-action IDs from Naive (B2)"}


def _is_single_replacement(row: dict, question: str) -> bool:
    if len(row["drag"]) != 1:
        return False
    query, passages = next(iter(row["drag"].items()))
    if query == question:
        return False
    return not (set(_ids(passages)) - set(_ids(row["naive"])))


def build(retrieval: Path = RETRIEVAL, questions: Path = QUESTIONS) -> dict:
    """Return the fixture object; deterministic for a given recorded results file."""
    questions_by_id = {row["id"]: row["question"] for row in _load(questions)}
    rows = _load(retrieval)
    multi, single = [], []
    for row in rows:
        question = questions_by_id[row["id"]]
        if len(row["drag"]) > 1:
            case = _multi_case(row)
            if case is not None:
                multi.append(case)
        elif _is_single_replacement(row, question) and len(single) < SINGLE_LIMIT:
            single.append(_single_case(row, question))
    return {
        "source": "results/retrieval-200.jsonl",
        "note": "IDs only; no passage text. Multi cases are exact; single replacements "
                "use Naive as the pre-action pool (B2).",
        "cases": multi + single,
    }


def main() -> None:
    OUTPUT.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(f"wrote {OUTPUT} ({len(build()['cases'])} cases)")


if __name__ == "__main__":
    main()
