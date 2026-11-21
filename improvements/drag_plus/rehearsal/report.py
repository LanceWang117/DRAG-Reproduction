"""Structural report for a rehearsal: what each arm did, never how well it did.

Built from the J03 logs of one run directory per variant (timing rows need
``profile_calls``, which rehearsals always set). Per variant and arm it reports model
calls and retrieval calls per question (by stage), query and answer rounds, prompt and
output tokens, finish reasons and Judge decision statuses. It then adds the difference
from the first variant. No EM, F1 or latency is reported as a result.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional

from ..instrument import logs
from ..runner.smoke_summary import summarise_run

NOT_LLM_STAGES = ("retrieval", "rerank")
RETRIEVAL_CAVEAT = ("RETRIEVAL CAVEAT: passages are replayed from results/; refined queries fall "
                    "back to lexical matching, so retrieval-dependent behaviour is not representative.")


def _p90(values: List[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))]


def _rounds(rows: Iterable[dict], prefix: str) -> float:
    """Mean number of rounds per question for stages starting with ``prefix``."""
    per_question: Dict[tuple, int] = {}
    for row in rows:
        if row["stage"].startswith(prefix) and row["role"] not in ("naive", "answer_only"):
            key = (row["dataset"], row["id"])
            per_question[key] = max(per_question.get(key, 0), row["round"] + 1)
    return sum(per_question.values()) / len(per_question) if per_question else 0.0


def _statuses(transcripts: Iterable[dict]) -> Dict[str, int]:
    counts: Counter = Counter()
    for row in transcripts:
        counts.update(e["value"] for e in row["entries"] if e["key"].endswith("_DecisionStatus"))
    return dict(counts)


def arm_metrics(rows: List[dict], transcripts: List[dict], questions: int) -> Dict[str, Any]:
    """Structural metrics of one arm from its timing rows and transcripts."""
    llm = [r for r in rows if r["stage"] not in NOT_LLM_STAGES]
    prompt = [r["prompt_tokens"] for r in llm if r["prompt_tokens"] is not None]
    output = [r["output_tokens"] for r in llm if r["output_tokens"] is not None]
    per_q = max(questions, 1)
    return {
        "llm_calls_per_question": len(llm) / per_q,
        "retrieval_calls_per_question": sum(r["stage"] == "retrieval" for r in rows) / per_q,
        "calls_by_stage": {stage: n / per_q for stage, n in sorted(Counter(r["stage"] for r in rows).items())},
        "query_rounds": _rounds(rows, "query_"),
        "answer_rounds": _rounds(rows, "answer_"),
        "prompt_tokens_median": median(prompt) if prompt else None,
        "prompt_tokens_p90": _p90(prompt),
        "output_tokens_median": median(output) if output else None,
        "finish_reasons": dict(Counter(r["finish_reason"] for r in llm)),
        "judge_statuses": _statuses(transcripts),
        "local_seconds": sum(r["elapsed_ms"] for r in rows) / 1000,
    }


def _read_rows(path: Path) -> List[dict]:
    """Rows of a log, or none if it is damaged (``summarise_run`` counts that as a schema error)."""
    try:
        return logs.read_lines(path)
    except ValueError:
        return []


def variant_report(run_dir: Path) -> Dict[str, Any]:
    """The smoke summary of one variant's run plus its per-arm structural metrics."""
    run_dir = Path(run_dir)
    summary = summarise_run(run_dir)
    manifest = json.loads((run_dir / "run-manifest.json").read_text(encoding="utf-8"))
    timings = _read_rows(run_dir / "stage-timings.jsonl")
    transcripts = _read_rows(run_dir / "transcripts.jsonl")
    by_arm_rows: Dict[str, List[dict]] = defaultdict(list)
    by_arm_transcripts: Dict[str, List[dict]] = defaultdict(list)
    for row in timings:
        by_arm_rows[row["arm"]].append(row)
    for row in transcripts:
        by_arm_transcripts[row["arm"]].append(row)
    summary["arms"] = {arm: arm_metrics(by_arm_rows[arm], by_arm_transcripts[arm],
                                        manifest["questions"]) for arm in manifest["arms"]}
    return summary


def differences(first: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Per shared arm: other minus first, for the numeric structural metrics."""
    keys = ("llm_calls_per_question", "retrieval_calls_per_question", "query_rounds",
            "answer_rounds", "prompt_tokens_median")
    result = {}
    for arm in sorted(set(first["arms"]) & set(other["arms"])):
        a, b = first["arms"][arm], other["arms"][arm]
        result[arm] = {k: b[k] - a[k] for k in keys if a[k] is not None and b[k] is not None}
    return result
