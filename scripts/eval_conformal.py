"""Evaluate conformal selective answering on logged runs (J21).

Input is JSONL with one record per answered question: ``dataset``, ``gold``, ``answers``
(the agent answers), an optional aggregated ``pred`` and ``judge_score``, and either
``support`` (answer -> score) or ``passages`` for the J14 containment score. The script
builds ``p_hat`` with the same code the pipeline uses, cross-fits the calibration five
ways, and reports for one alpha: empirical coverage with an exact Clopper-Pearson 95%
interval, mean set size, abstention rate, full-coverage accuracy, selective accuracy
(answered only), selective accuracy at 50/70/90% coverage, and coverage per dataset so a
pooled fallback is visible.

The risk-coverage curve is a dependency-free SVG, matching J20: ``requirements-ci.txt``
has no Matplotlib and this job does not authorize a new dependency; ``report.json``
holds the numbers.

Usage: ``python scripts/eval_conformal.py --records run.jsonl --alpha 0.1 --out reports/``
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "improvements"))
from drag_plus.gates.conformal import (calibrate, cluster_key,  # noqa: E402
                                       prediction_set, probability_distribution)
from drag_plus.gates.conformal_metrics import (TARGETS, risk_coverage_svg,  # noqa: E402
                                               summarise)
from drag_plus.gates.features import evidence_support  # noqa: E402


def read_records(path: Any) -> List[dict]:
    """Parse JSONL objects, rejecting a blank-only file and non-object lines."""
    records = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"line {number}: expected a JSON object")
            records.append(record)
    if not records:
        raise ValueError("input has no records")
    return records


def _supports(record: Mapping[str, Any], texts: Mapping[str, str], yes_no: bool) -> dict:
    """Per-cluster evidence support from a recorded map, else from recorded passages."""
    raw = record.get("support")
    if raw is not None:
        if not isinstance(raw, dict):
            raise ValueError("support must be an object mapping answer to score")
        support = {cluster_key(str(key), yes_no): float(value)
                   for key, value in raw.items()}
        if any(not 0.0 <= score <= 1.0 for score in support.values()):
            raise ValueError("support scores must be in [0, 1]")
        return support
    passages = [str(text) for text in record.get("passages") or []]
    return {key: evidence_support(text, passages) for key, text in texts.items()}


def record_distribution(record: Mapping[str, Any]) -> Tuple[str, str, Dict[str, float]]:
    """Validate one record and return ``(dataset, gold cluster, p_hat)``."""
    dataset, gold = str(record.get("dataset") or ""), str(record.get("gold") or "")
    if not dataset or not gold:
        raise ValueError("every record needs a non-empty dataset and gold")
    yes_no = dataset == "StrategyQA"
    answers = [str(answer) for answer in record.get("answers") or []]
    if record.get("pred"):
        answers.append(str(record["pred"]))
    votes: Dict[str, int] = {}
    texts: Dict[str, str] = {}
    for answer in answers:
        key = cluster_key(answer, yes_no)
        if key:
            votes[key] = votes.get(key, 0) + 1
            texts.setdefault(key, answer)
    distribution = probability_distribution(votes, _supports(record, texts, yes_no),
                                             record.get("judge_score"))
    if not distribution:
        raise ValueError("a record needs at least one non-empty answer or support")
    return dataset, cluster_key(gold, yes_no), distribution


def _row(dataset: str, gold: str, distribution: Mapping[str, float], qhat: float) -> dict:
    answer_set = prediction_set(distribution, qhat)
    best = max(distribution, key=distribution.get)
    return {"dataset": dataset, "gold": gold, "covered": gold in answer_set,
            "set_size": len(answer_set), "abstained": len(answer_set) != 1,
            "answered_correct": len(answer_set) == 1 and answer_set[0] == gold,
            "argmax_correct": best == gold, "confidence": distribution[best]}


def _parts(examples: Sequence[tuple], folds: int) -> List[List[tuple]]:
    """Round-robin folds, stratified by dataset, so every fold keeps every dataset."""
    per_dataset: Dict[str, List[tuple]] = {}
    for example in examples:
        per_dataset.setdefault(example[0], []).append(example)
    parts: List[List[tuple]] = [[] for _ in range(folds)]
    for dataset in sorted(per_dataset):
        for index, example in enumerate(per_dataset[dataset]):
            parts[index % folds].append(example)
    return parts


def threshold_for(document: Mapping[str, Any], dataset: str) -> float:
    """The dataset threshold when it has ``min_n`` rows, otherwise the pooled one."""
    entry = document["datasets"].get(dataset)
    if entry is not None and entry["n"] >= document["min_n"]:
        return entry["qhat"]
    return document["pooled"]["qhat"]


def _top_threshold(pairs: Sequence[Tuple[float, bool]], target: float) -> float:
    ordered = sorted((confidence for confidence, _ in pairs), reverse=True)
    rank = max(1, math.ceil(target * len(ordered)))
    return ordered[rank - 1]


def cross_fit(examples: Sequence[tuple], alpha: float, folds: int = 5,
              min_n: int = 20) -> Tuple[List[dict], Dict[Tuple[int, float], float]]:
    """Five-way cross-fitting; returns held-out rows and per-fold selective thresholds."""
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if len(examples) < folds:
        raise ValueError("more records than folds are needed")
    parts = _parts(examples, folds)
    rows: List[dict] = []
    thresholds: Dict[Tuple[int, float], float] = {}
    for index, held_out in enumerate(parts):
        train = [example for other, part in enumerate(parts) if other != index
                 for example in part]
        if not held_out or not train:
            continue
        document = calibrate([(d, g, dist) for d, g, dist in train], alpha, min_n)
        pairs = [(max(dist.values()),
                  max(dist, key=dist.get) == gold) for _, gold, dist in train]
        for target in TARGETS:
            thresholds[(index, target)] = _top_threshold(pairs, target)
        for dataset, gold, distribution in held_out:
            row = _row(dataset, gold, distribution, threshold_for(document, dataset))
            row["fold"] = index
            rows.append(row)
    return rows, thresholds


def _write(directory: Path, alpha: float, report: dict, rows: Sequence[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"report-alpha-{alpha}.json").write_text(
        json.dumps({"alpha": alpha, **report}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    risk_coverage_svg(rows, directory / f"risk-coverage-{alpha}.svg")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-n", type=int, default=20)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--write-calibration")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    examples = [record_distribution(record) for record in read_records(args.records)]
    if args.write_calibration:
        document = calibrate(examples, args.alpha, args.min_n, args.split)
        Path(args.write_calibration).write_text(json.dumps(document, indent=2) + "\n",
                                                encoding="utf-8")
    rows, thresholds = cross_fit(examples, args.alpha, args.folds, args.min_n)
    report = summarise(rows, thresholds)
    print(json.dumps({"alpha": args.alpha, **report}, sort_keys=True))
    if args.out:
        _write(Path(args.out), args.alpha, report, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
