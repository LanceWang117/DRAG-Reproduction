"""Schema, parser and validator for structured Opponent actions (J13).

The Opponent ends its reasoning with one JSON object. This module takes the last
balanced object (tolerating code fences and trailing text), rejects a second action
object, unknown/missing fields, wrong types and an empty non-``no_op`` query, then
validates query length, duplication and entity preservation against the stable query
trail. The transactional apply lives in ``actions.py`` so each file stays within the
repository's 250-line limit.

No grammar engine is used: the prompt plus this strict parser is the default because the
pinned vLLM 0.6.6 guided-decoding stack has known slowdown/degradation reports (J13
research note). The module imports no FlashRAG, vLLM, torch or FAISS.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from ..evidence.audit import extract_entities
from ..evidence.store import latest_refined, query_text
from ..instrument.timing import whitespace_tokens
from ..query_norm import is_duplicate_query
from ..state import QuestionState

ACTIONS = ("optimize", "expand", "no_op")
FIELDS = frozenset({"action", "source_query_id", "new_query", "reason"})
MIN_QUERY_TOKENS = 3
MAX_QUERY_TOKENS = 30


class ActionError(ValueError):
    """An Opponent action failed schema, validation or application."""

    def __init__(self, message: str, parsed: Optional[Any] = None):
        super().__init__(message)
        self.parsed = parsed


@dataclass(frozen=True)
class OpponentAction:
    """One validated schema action; ``source_query_id`` is an int or None."""
    action: str
    source_query_id: Optional[int]
    new_query: str
    reason: str


def _string_end(text: str, index: int) -> int:
    """Index just past the closing quote, honouring backslash escapes."""
    index += 1
    escape = False
    while index < len(text):
        char = text[index]
        if escape:
            escape = False
        elif char == "\\":
            escape = True
        elif char == '"':
            return index + 1
        index += 1
    return index


def _balanced_objects(text: str) -> Tuple[List[str], int]:
    """Top-level ``{...}`` spans and the final unclosed depth, ignoring strings."""
    spans, start, depth, index = [], None, 0, 0
    while index < len(text):
        char = text[index]
        if char == '"':
            index = _string_end(text, index)
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    spans.append(text[start:index + 1])
                    start = None
        index += 1
    return spans, depth


def _json_objects(text: str) -> List[dict]:
    """Parse every balanced object; unparsable spans are skipped for counting."""
    objects = []
    spans, _ = _balanced_objects(text)
    for span in spans:
        try:
            value = json.loads(span)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    return objects


def _require_fields(value: Any) -> dict:
    if not isinstance(value, dict):
        raise ActionError("the last JSON value is not an object")
    unknown = sorted(set(value) - FIELDS)
    if unknown:
        raise ActionError(f"unknown fields: {unknown}")
    missing = sorted(FIELDS - set(value))
    if missing:
        raise ActionError(f"missing fields: {missing}")
    return value


def _action_field(value: Any) -> str:
    if not isinstance(value, str) or value not in ACTIONS:
        raise ActionError(f"action must be one of {list(ACTIONS)}")
    return value


def _source_field(value: Any) -> Optional[int]:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise ActionError("source_query_id must be an integer or null")
    return value


def _query_field(value: Any, action: str) -> str:
    if value is None and action == "no_op":
        return ""
    if not isinstance(value, str):
        raise ActionError("new_query must be a string")
    text = value.strip()
    if action != "no_op" and not text:
        raise ActionError("new_query must not be empty")
    return text


def _reason_field(value: Any) -> str:
    if not isinstance(value, str):
        raise ActionError("reason must be a string")
    return value.strip()


def _shape(value: Any) -> OpponentAction:
    """Validate the four fields and their types; returns the action or raises."""
    try:
        value = _require_fields(value)
        action = _action_field(value["action"])
        source = _source_field(value["source_query_id"])
        new_query = _query_field(value["new_query"], action)
        reason = _reason_field(value["reason"])
    except ActionError as error:
        error.parsed = value if isinstance(value, dict) else None
        raise
    return OpponentAction(action, source, new_query, reason)


def parse_action(raw: str) -> OpponentAction:
    """Take the last JSON object from ``raw``; tolerate fences and trailing text.

    The last balanced object is the only candidate, so a malformed final action never
    falls back to an earlier example. More than one object carrying an ``action`` key is
    a multiple-action error. Unknown fields, missing fields, wrong types and an empty
    non-``no_op`` query are rejected.
    """
    spans, depth = _balanced_objects(str(raw))
    if depth:
        raise ActionError("unclosed JSON object after the last action")
    if not spans:
        raise ActionError("no JSON object in the Opponent output")
    try:
        value = json.loads(spans[-1])
    except json.JSONDecodeError as error:
        raise ActionError(f"last JSON object is invalid: {error.msg}") from error
    objects = _json_objects(str(raw))
    if sum("action" in obj for obj in objects) > 1:
        raise ActionError("multiple action objects", value if isinstance(value, dict)
                          else None)
    return _shape(value)


def active_queries(state: QuestionState) -> List[str]:
    """Current plan keys: the anchored store's active view, else the baseline pool."""
    if state.evidence is not None:
        return list(state.evidence.active_pool())
    return list(state.query_pool)


def _resolve_source(action: OpponentAction, state: QuestionState) -> str:
    """Resolve the source by stable ID; optimize requires an active source."""
    if action.action == "no_op":
        return state.question.strip()
    active = active_queries(state)
    if action.source_query_id is None:
        if action.action == "optimize":
            raise ActionError("optimize needs a source_query_id")
        return latest_refined(state.queries, active) or state.question.strip()
    text = query_text(state.queries, action.source_query_id)
    if text is None or text not in {str(query).strip() for query in active}:
        raise ActionError(f"unknown source_query_id {action.source_query_id}")
    return text


def validate_action(action: OpponentAction, state: QuestionState) -> str:
    """Check length, duplication and entity preservation; returns the source text."""
    source = _resolve_source(action, state)
    if action.action == "no_op":
        return source
    tokens = whitespace_tokens(action.new_query)
    if tokens < MIN_QUERY_TOKENS or tokens > MAX_QUERY_TOKENS:
        raise ActionError(
            f"new_query has {tokens} tokens, expected {MIN_QUERY_TOKENS}-"
            f"{MAX_QUERY_TOKENS}")
    if is_duplicate_query(action.new_query, active_queries(state)):
        raise ActionError("new_query duplicates an active query")
    dropped = extract_entities(source) - extract_entities(action.new_query)
    if dropped and not action.reason:
        raise ActionError(f"new_query drops entities {sorted(dropped)} without a reason")
    return source


def as_dict(action: OpponentAction) -> dict:
    """The parsed action as a plain dict for decision logging."""
    return {"action": action.action, "source_query_id": action.source_query_id,
            "new_query": action.new_query, "reason": action.reason}
