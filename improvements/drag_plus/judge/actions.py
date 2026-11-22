"""Transactional apply for structured Opponent actions (J13).

``action_schema.py`` owns the schema, parser and validator. This module builds the
candidate pool, validates it, and commits only after every check passes. Any failure
leaves the pool untouched and appends ``{raw, parsed, error}`` to ``state.decisions``;
it counts as ``no_op`` for control flow (an invalid action ends the query stage like a
text no-op). The anchored store is cloned before the search, so a retriever error cannot
leave a partial commit. Imports no FlashRAG, vLLM, torch or FAISS.
"""
from __future__ import annotations

from typing import Any, Optional

from ..adapters import StageContext
from ..evidence.actions import (ACTION_APPLIED, ACTION_NONE, EXIT_ZERO_NOVELTY,
                                has_new_id, pool_ids, search, sync_pool)
from ..evidence.store import register_query
from ..state import QuestionState
from .action_schema import (ActionError, OpponentAction, active_queries, as_dict,
                            parse_action, validate_action)


def _apply_dict(ctx: StageContext, state: QuestionState, action: OpponentAction,
                source: str) -> str:
    """Build a candidate baseline dict, then replace the live pool in one commit."""
    candidate = dict(state.query_pool)
    if action.action == "optimize":
        candidate.pop(source, None)
    before = pool_ids(state.query_pool)
    passages = search(ctx, state, action.new_query)
    if ctx.flags.zero_novelty_exit and not has_new_id(passages, before):
        return EXIT_ZERO_NOVELTY
    candidate[action.new_query] = passages
    state.query_pool.clear()
    state.query_pool.update(candidate)
    return ACTION_APPLIED


def _apply_anchored(ctx: StageContext, state: QuestionState, action: OpponentAction,
                    source: str) -> str:
    """Apply to a cloned anchored store; assign it only after the search succeeds."""
    store = state.evidence
    clone = store.copy()
    before = store.ids()
    passages = search(ctx, state, action.new_query)
    if ctx.flags.zero_novelty_exit and not has_new_id(passages, before):
        return EXIT_ZERO_NOVELTY
    previous = state.evidence
    try:
        clone.add_query(action.new_query, passages, state.round)
        if action.action == "optimize":
            clone.supersede(source)
        state.evidence = clone
        sync_pool(ctx, state)
    except Exception:
        state.evidence = previous
        raise
    return ACTION_APPLIED


def _apply(ctx: StageContext, state: QuestionState, action: OpponentAction,
           source: str) -> str:
    if state.evidence is not None:
        return _apply_anchored(ctx, state, action, source)
    return _apply_dict(ctx, state, action, source)


def _register_after(state: QuestionState, source: str) -> None:
    """Give every current plan query a stable ID, with the action as the new source."""
    for query in active_queries(state):
        register_query(state.queries, query, source, state.round)


def _log(state: QuestionState, raw: str, parsed: Optional[Any], status: str,
         error: Optional[Exception] = None) -> None:
    row = {"stage": "query", "round": state.round, "decision": "opponent_action",
           "status": status, "raw": raw, "parsed": parsed}
    if error is not None:
        row["error"] = str(error)
    state.decisions.append(row)


def apply_opponent_action(ctx: StageContext, state: QuestionState,
                          opponent_output: str) -> str:
    """Parse, validate and commit one structured action; never mutates on failure.

    Returns ``ACTION_APPLIED``, ``EXIT_ZERO_NOVELTY`` (the pool was restored by not
    committing), or ``ACTION_NONE`` for a valid ``no_op`` and for any invalid action.
    Every call appends an ``opponent_action`` decision row with the raw output, the
    parsed object when available, and the error on failure.
    """
    parsed = None
    try:
        action = parse_action(opponent_output)
        parsed = as_dict(action)
        source = validate_action(action, state)
        if action.action == "no_op":
            _log(state, opponent_output, parsed, "no_op")
            return ACTION_NONE
        outcome = _apply(ctx, state, action, source)
        if outcome == ACTION_APPLIED:
            _register_after(state, action.action)
        _log(state, opponent_output, parsed, outcome)
        return outcome
    except ActionError as error:
        _log(state, opponent_output,
             error.parsed if error.parsed is not None else parsed, "invalid", error)
        return ACTION_NONE
    except Exception as error:
        _log(state, opponent_output, parsed, "invalid", error)
        return ACTION_NONE
