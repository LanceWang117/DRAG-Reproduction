"""Validate retrieval Judge decisions without inferring a winner from prose."""
from dataclasses import dataclass
import re
from typing import Callable, Optional


@dataclass(frozen=True)
class JudgeDecision:
    decision: Optional[str]
    status: str
    retry_output: Optional[str] = None


def parse_judge_decision(output: str) -> Optional[str]:
    if not isinstance(output, str):
        return None
    match = re.fullmatch(
        r"(Proponent|Opponent)(?:\s+Agent(?:\s+\d+)?)?[.!]?",
        output.strip(), re.IGNORECASE,
    )
    return match.group(1).capitalize() if match else None


def resolve_judge_decision(output: str, retry: Callable[[], str]) -> JudgeDecision:
    decision = parse_judge_decision(output)
    if decision is not None:
        return JudgeDecision(decision, 'valid_initial')
    retry_output = retry()
    decision = parse_judge_decision(retry_output)
    status = 'valid_retry' if decision is not None else 'unresolved'
    return JudgeDecision(decision, status, retry_output)
