"""Plain-text rendering of a rehearsal report (see ``report.py`` for the numbers)."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from ..runner.smoke_summary import NOT_A_RESULT
from .plan import ArmPlan
from .report import RETRIEVAL_CAVEAT, differences


def _fmt(value: Any, spec: str = "{:.2f}") -> str:
    return "n/a" if value is None else spec.format(value)


def _finish(reasons: Dict[Any, int]) -> str:
    # Timing rows may carry no finish reason (the local backend counts them only in its
    # generator stats, shown once per variant), so an all-None tally is not printed.
    known = {key: count for key, count in reasons.items() if key is not None}
    return str(known) if known else "not in timing rows"


def _arm_lines(arm: str, m: Dict[str, Any]) -> List[str]:
    return [
        f"       {arm}: {m['llm_calls_per_question']:.2f} model calls/q, "
        f"{m['retrieval_calls_per_question']:.2f} retrievals/q, query rounds "
        f"{m['query_rounds']:.2f}, answer rounds {m['answer_rounds']:.2f}",
        f"         prompt tokens median {_fmt(m['prompt_tokens_median'], '{:.0f}')} / p90 "
        f"{_fmt(m['prompt_tokens_p90'], '{:.0f}')}, output median "
        f"{_fmt(m['output_tokens_median'], '{:.0f}')}; finish {_finish(m['finish_reasons'])}; "
        f"judge {m['judge_statuses'] or '-'}; local time {m['local_seconds']:.0f}s (not comparable)"]


def _variant_lines(report: Dict[str, Any], plan: ArmPlan) -> List[str]:
    lines = [f"[{'PASS' if report['ok'] else 'FAIL'}] {report['variant']}: completed "
             f"{report['completed']}/{report['expected']}, parse failures "
             f"{report['parse_failures']}, empty outputs {report['empty_outputs']}, "
             f"schema errors {report['schema_errors']}"]
    for arm, metrics in report.get("arms", {}).items():
        lines += _arm_lines(arm, metrics)
    generator = report.get("generator") or {}
    if generator.get("finish_reasons"):
        lines.append(f"       generator finish reasons (all arms): {generator['finish_reasons']}")
    lines += [f"       note: {note}" for note in plan.notes]
    lines += [f"       problem: {problem}" for problem in report["problems"]]
    return lines + [f"       run directory: {report['run_dir']}"]


def _difference_lines(reports: Sequence[Dict[str, Any]]) -> List[str]:
    if len(reports) < 2 or "arms" not in reports[0]:
        return []
    lines = [f"Differences from {reports[0]['variant']} (structural only):"]
    for report in reports[1:]:
        for arm, deltas in differences(reports[0], report).items():
            text = ", ".join(f"{key} {value:+.2f}" for key, value in deltas.items())
            lines.append(f"  {report['variant']} [{arm}]: {text}")
    return lines


def overall_ok(reports: Sequence[Dict[str, Any]], runnable: int = 0) -> bool:
    """PASS needs at least one arm run, every runnable arm reported, and all passing."""
    return bool(reports) and len(reports) >= runnable and all(r["ok"] for r in reports)


def format_report(reports: Sequence[Dict[str, Any]], plans: Sequence[ArmPlan],
                  model: str, questions: int) -> str:
    by_name = {plan.name: plan for plan in plans}
    skipped = [plan for plan in plans if not plan.runnable]
    lines = [f"Local rehearsal: {model}, {questions} dev question(s), {len(reports)} arm(s) run, "
             f"{len(skipped)} skipped", NOT_A_RESULT, RETRIEVAL_CAVEAT]
    for report in reports:
        lines += _variant_lines(report, by_name[report["variant"]])
    for plan in skipped:
        lines.append(f"[SKIP] {plan.name}: " + "; ".join(plan.reasons))
    reported = {report["variant"] for report in reports}
    for plan in plans:
        if plan.runnable and plan.name not in reported:
            lines.append(f"[NOT RUN] {plan.name}: stopped after an earlier model-side error")
    lines += _difference_lines(reports)
    runnable = sum(plan.runnable for plan in plans)
    lines.append(f"Overall: {'PASS' if overall_ok(reports, runnable) else 'FAIL'}")
    return "\n".join(lines)
