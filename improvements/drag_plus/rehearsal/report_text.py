"""Plain-text rendering of a rehearsal report (see ``report.py`` for the numbers)."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from ..runner.smoke_summary import NOT_A_RESULT
from .plan import ArmPlan
from .report import RETRIEVAL_CAVEAT, differences


def _fmt(value: Any, spec: str = "{:.2f}") -> str:
    return "n/a" if value is None else spec.format(value)


def _arm_lines(arm: str, m: Dict[str, Any]) -> List[str]:
    return [
        f"       {arm}: {m['llm_calls_per_question']:.2f} model calls/q, "
        f"{m['retrieval_calls_per_question']:.2f} retrievals/q, query rounds "
        f"{m['query_rounds']:.2f}, answer rounds {m['answer_rounds']:.2f}",
        f"         prompt tokens median {_fmt(m['prompt_tokens_median'], '{:.0f}')} / p90 "
        f"{_fmt(m['prompt_tokens_p90'], '{:.0f}')}, output median "
        f"{_fmt(m['output_tokens_median'], '{:.0f}')}; finish {m['finish_reasons']}; "
        f"judge {m['judge_statuses'] or '-'}; local time {m['local_seconds']:.0f}s (not comparable)"]


def _variant_lines(report: Dict[str, Any], plan: ArmPlan) -> List[str]:
    lines = [f"[{'PASS' if report['ok'] else 'FAIL'}] {report['variant']}: completed "
             f"{report['completed']}/{report['expected']}, parse failures "
             f"{report['parse_failures']}, empty outputs {report['empty_outputs']}, "
             f"schema errors {report['schema_errors']}"]
    for arm, metrics in report.get("arms", {}).items():
        lines += _arm_lines(arm, metrics)
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


def overall_ok(reports: Sequence[Dict[str, Any]]) -> bool:
    """PASS needs at least one arm run and every run arm passing."""
    return bool(reports) and all(report["ok"] for report in reports)


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
    lines += _difference_lines(reports)
    lines.append(f"Overall: {'PASS' if overall_ok(reports) else 'FAIL'}")
    return "\n".join(lines)
