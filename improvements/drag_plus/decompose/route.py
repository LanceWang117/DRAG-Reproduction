"""Planner-to-execution integration and the fallback contract for ``decompose``.

A planner call is spent first; an invalid plan falls back before any retrieval. Execution
returns None for an ungrounded binding or an exhausted call cap. On any fallback the
route logs the reason and clears the partial anchored pool, so the J20 router or the
normal DRAG pipeline starts from the same state it would have seen without the flag.
"""
from __future__ import annotations

from ..adapters import StageContext
from ..state import QuestionState
from .execute import CallBudget, execute_plan
from .plan import PlanError, parse_plan, planner_messages


def _fallback(state: QuestionState, reason: str, budget: CallBudget) -> None:
    state.emit("Decompose_Fallback", reason)
    state.decisions.append({"stage": "decompose", "decision": "fallback",
                            "status": reason, "calls": budget.used,
                            "call_cap": budget.limit})


def _reset(state: QuestionState) -> None:
    """Leave the fallback pipeline the clean pool it would have had without the route."""
    state.evidence = None
    state.query_pool = {}
    state.queries = []
    state.round = 0


def _record_success(ctx: StageContext, state: QuestionState, plan, result,
                    budget: CallBudget) -> None:
    state.emit("Decompose_Plan", plan.as_dict())
    state.emit("Decompose_Ledger", [entry.as_dict() for entry in result.ledger])
    state.emit("Decompose_Calls", {"used": budget.used, "limit": budget.limit})
    state.emit("Decompose_Beam", {"beams": result.beams, "paths": len(result.ledger)})
    state.decisions.append({"stage": "decompose", "decision": "answer",
                            "status": "grounded", "calls": budget.used,
                            "call_cap": budget.limit, "beams": result.beams})
    state.raw_pred = result.composition.text
    state.pred = result.composition.answer
    state.stage = "answer"
    if ctx.flags.answer_extractor == "native":
        state.emit("pred", state.pred)


def run_decomposed(ctx: StageContext, state: QuestionState) -> bool:
    """Plan and execute the route; False means the normal pipeline must run instead."""
    if not ctx.flags.decompose:
        return False
    budget = CallBudget()
    try:
        if not budget.spend():
            raise PlanError("the call budget is too small for a planner call")
        raw = ctx.backends.complete(
            ctx.backends.render(planner_messages(state.question)),
            call=state.call("query_agent", "planner"))
        state.emit("Decompose_PlanRaw", raw)
        plan = parse_plan(raw)
    except PlanError as error:
        _fallback(state, str(error), budget)
        return False
    result = execute_plan(ctx, state, plan, budget=budget)
    if result is None:
        _fallback(state, "call_cap" if budget.exhausted else "ungrounded", budget)
        _reset(state)
        return False
    _record_success(ctx, state, plan, result, budget)
    return True
