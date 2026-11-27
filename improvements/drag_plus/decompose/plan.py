"""Typed multi-hop plan: planner prompt, strict parser and validator.

A planner answer has one line per hop, ``qN: <sub-question> -> #EN``. The parser is
deliberately strict: blank or malformed lines, duplicate or out-of-order ids, more than
three hops, a wrong output placeholder and a forward/self/undefined reference all raise
``PlanError``. The route logs the reason and falls back before spending a retrieval,
which is the edge-case contract from the J22 spec.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

MAX_STEPS = 3
_LINE = re.compile(r"^q([1-9][0-9]*):\s*(.+?)\s*->\s*#E([1-9][0-9]*)$")
_PLACEHOLDER = re.compile(r"#E([0-9]+)")

PLANNER_SYSTEM = (
    "You plan multi-hop question answering. Decompose the question into at most three "
    "ordered sub-questions. Return one line per hop and nothing else, in this exact "
    "form:\nq1: <sub-question> -> #E1\nq2: <sub-question that may use #E1> -> #E2\n"
    "A placeholder may refer only to an earlier hop's answer. Keep every entity and "
    "number from the original question. Do not explain, answer, or add a fourth line."
)


class PlanError(ValueError):
    """The planner text is not an executable, acyclic plan of at most three hops."""


@dataclass(frozen=True)
class SubQuestion:
    """One hop: its stable id, text and the placeholder its answer binds."""

    qid: int
    text: str
    output: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {"qid": self.qid, "text": self.text, "output": self.output}


@dataclass(frozen=True)
class Plan:
    """An ordered, acyclic decomposition; ``steps`` is never empty."""

    steps: Tuple[SubQuestion, ...]

    @property
    def max_depth(self) -> int:
        return len(self.steps)

    def as_dict(self) -> Dict[str, object]:
        return {"steps": [step.as_dict() for step in self.steps],
                "max_depth": self.max_depth}


def reference_ids(text: str) -> List[int]:
    """Every ``#E<n>`` referenced by ``text``, including malformed zero ids."""
    return [int(match.group(1)) for match in _PLACEHOLDER.finditer(text)]


def _check_references(step: SubQuestion, defined: set) -> None:
    """Reject a self/forward reference or one to a hop that is not defined yet."""
    for ref in reference_ids(step.text):
        if ref >= step.qid:
            raise PlanError(
                f"q{step.qid} references #E{ref}; cycles/forward references are invalid")
        if ref not in defined:
            raise PlanError(f"q{step.qid} references undefined #E{ref}")


def _check_step(step: SubQuestion, position: int, defined: set) -> None:
    """Validate one hop's id, text, output and backward-only references."""
    if not isinstance(step, SubQuestion):
        raise PlanError("every step must be a SubQuestion")
    if step.qid != position:
        raise PlanError(f"step {position} has id q{step.qid}; ids must be 1..n")
    if not isinstance(step.text, str) or not step.text.strip():
        raise PlanError(f"q{step.qid} has an empty sub-question")
    if step.output and step.output != f"#E{step.qid}":
        raise PlanError(f"q{step.qid} binds {step.output}, expected #E{step.qid}")
    _check_references(step, defined)


def validate_plan(plan: Plan) -> Plan:
    """Reject every invalid shape; return the plan unchanged when it is executable."""
    if not isinstance(plan, Plan) or not plan.steps:
        raise PlanError("the plan is empty")
    if len(plan.steps) > MAX_STEPS:
        raise PlanError(f"depth {len(plan.steps)} exceeds {MAX_STEPS}")
    defined = set()
    for position, step in enumerate(plan.steps, start=1):
        _check_step(step, position, defined)
        defined.add(step.qid)
    return plan


def _parse_line(line: str, position: int) -> SubQuestion:
    """Parse one canonical ``qN: text -> #EN`` line."""
    match = _LINE.match(line)
    if match is None:
        raise PlanError(f"malformed line {line!r}; expected 'qN: text -> #EN'")
    qid, output = int(match.group(1)), int(match.group(3))
    if qid != position:
        raise PlanError(f"line {position} has id q{qid}; ids must be 1..n")
    if output != qid:
        raise PlanError(f"q{qid} binds #E{output}; expected #E{qid}")
    return SubQuestion(qid, match.group(2).strip(), f"#E{output}")


def parse_plan(raw: object) -> Plan:
    """Parse one strict planner answer into a validated plan, or raise ``PlanError``."""
    if not isinstance(raw, str):
        raise PlanError("the planner output is not text")
    lines = [line.strip() for line in raw.strip().splitlines()]
    lines = [line for line in lines if line and not line.lower().startswith("plan:")]
    if not lines:
        raise PlanError("the plan is empty")
    if len(lines) > MAX_STEPS:
        raise PlanError(f"depth {len(lines)} exceeds {MAX_STEPS}")
    steps = [_parse_line(line, position) for position, line in enumerate(lines, start=1)]
    return validate_plan(Plan(tuple(steps)))


def single_hop_plan(question: str) -> Plan:
    """A valid one-hop plan, used when the planner is unavailable or unparseable."""
    text = str(question).strip()
    if not text:
        raise PlanError("a single-hop plan needs a question")
    return Plan((SubQuestion(1, text, "#E1"),))


def planner_messages(question: str) -> List[dict]:
    """Planner chat messages: the system contract plus one user question."""
    text = str(question).strip()
    if not text:
        raise PlanError("the planner needs a question")
    return [{"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": f"Question: {text}\nPlan:"}]
