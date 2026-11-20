"""Read a finished smoke run's logs and say whether the path worked.

PASS means the run completed, its logs follow the J03 schema, and no model call came back
empty. It says nothing about answer quality: no accuracy is reported, because the local
model is a different family and size than the A100's Llama-3.1-8B BF16. Parse failures
(a final answer without "The answer is:") are counted and shown; they fail a variant only
when every answer failed, because a small model may ignore the format now and then.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..answers import FINAL_ANSWER_PREFIX
from ..instrument import logs
from .records import row_has_full_schema

TRANSCRIPT_FIELDS = ("run_id", "variant", "arm", "dataset", "split", "id", "entries",
                     "pred", "pred_native", "pred_unified")
NOT_A_RESULT = ("NOT A RESULT: a small local model, not comparable with the A100's "
                "Llama-3.1-8B BF16 runs; no accuracy is reported.")


def _read(run_dir: Path, name: str) -> tuple:
    """Rows of one log and the number of problems met while reading it."""
    try:
        return logs.read_lines(run_dir / name), 0
    except ValueError:
        return [], 1


def _schema_errors(manifest: Dict[str, Any], checkpoints: List[dict], transcripts: List[dict],
                   timings: List[dict]) -> int:
    errors = sum(1 for row in transcripts if any(f not in row for f in TRANSCRIPT_FIELDS))
    errors += sum(1 for row in timings if not row_has_full_schema(row))
    errors += abs(len(transcripts) - len(checkpoints))
    if manifest.get("flags", {}).get("profile_calls") and checkpoints and not timings:
        errors += 1
    return errors


def summarise_run(run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "run-manifest.json").read_text(encoding="utf-8"))
    (checkpoints, bad_a), (transcripts, bad_b), (timings, bad_c) = (
        _read(run_dir, "checkpoints.jsonl"), _read(run_dir, "transcripts.jsonl"),
        _read(run_dir, "stage-timings.jsonl"))
    backend = manifest.get("backend", {})
    expected = manifest["questions"] * len(manifest["arms"])
    result = {
        "variant": manifest["variant"]["name"], "run_dir": str(run_dir),
        "exit_status": manifest["exit_status"], "expected": expected,
        "completed": len(checkpoints),
        "parse_failures": sum(1 for r in checkpoints if r["raw_pred"].strip()
                              and FINAL_ANSWER_PREFIX not in r["raw_pred"]),
        "empty_outputs": backend.get("generator", {}).get("empty_outputs", 0),
        "schema_errors": bad_a + bad_b + bad_c + _schema_errors(
            manifest, checkpoints, transcripts, timings),
        "generator": backend.get("generator", {}), "retrieval": backend.get("retrieval", {})}
    result["problems"] = _problems(result)
    result["ok"] = not result["problems"]
    return result


def _problems(result: Dict[str, Any]) -> List[str]:
    problems = []
    if result["exit_status"] != "complete":
        problems.append(f"run {result['exit_status']}")
    if result["completed"] < result["expected"]:
        problems.append(f"only {result['completed']} of {result['expected']} answers")
    if result["schema_errors"]:
        problems.append(f"{result['schema_errors']} schema error(s)")
    if result["empty_outputs"]:
        problems.append(f"{result['empty_outputs']} empty model output(s)")
    if result["completed"] and result["parse_failures"] == result["completed"]:
        problems.append("no final answer carried 'The answer is:'")
    return problems


def format_summary(results: List[Dict[str, Any]], model: str, questions: int) -> str:
    lines = [f"Local smoke summary: {model}, {questions} question(s), {len(results)} variant(s)",
             NOT_A_RESULT]
    for r in results:
        gen, ret = r["generator"], r["retrieval"]
        lines += [
            f"[{'PASS' if r['ok'] else 'FAIL'}] {r['variant']}: completed "
            f"{r['completed']}/{r['expected']}, parse failures {r['parse_failures']}, "
            f"empty outputs {r['empty_outputs']}, schema errors {r['schema_errors']}",
            f"       generator: {gen.get('calls', 0)} calls, {gen.get('retries', 0)} retries, "
            f"{gen.get('trimmed_prompts', 0)} trimmed prompts, finish "
            f"{gen.get('finish_reasons', {})}; retrieval: {ret.get('replayed', 0)} replayed, "
            f"{ret.get('lexical', 0)} lexical",
            f"       run directory: {r['run_dir']}"]
        lines += [f"       problem: {problem}" for problem in r["problems"]]
    return "\n".join(lines)
