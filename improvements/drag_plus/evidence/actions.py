"""Query-pool maintenance for the baseline ``dict`` and the anchored store.

The baseline path reproduces v1.0.1 exactly, including the Levenshtein "closest key"
Optimization and its delete-then-insert order. The anchored path replaces the deletion
with an append and a supersede flag, so no passage the original question retrieved can
be evicted.
"""
from __future__ import annotations

from typing import Optional

import Levenshtein

from ..adapters import StageContext
from ..query_norm import is_duplicate_query
from ..state import QuestionState
from .audit import audit_rewrite
from .store import AnchoredEvidenceStore, EvidencePool

ACTION_APPLIED = "applied"
ACTION_NONE = "no_action"
EXIT_DUPLICATE = "duplicate"
EXIT_ZERO_NOVELTY = "zero_novelty"


class QueryPoolError(ValueError):
    """The Opponent's proposal could not be applied to the query pool."""


def find_most_similar_key(pool: dict, target: str) -> Optional[str]:
    best_key, best = None, float("inf")
    for key in pool:
        distance = Levenshtein.distance(key, target)
        if distance < best:
            best, best_key = distance, key
    return best_key


def split_optimization(instruction: str):
    if "->" in instruction:
        parts = instruction.split("->")
        return parts[0].strip(), parts[1].strip()
    return instruction, instruction


def pool_ids(pool: dict) -> set:
    return {passage.get("id") for passages in pool.values() for passage in passages
            if passage.get("id") is not None}


def has_new_id(passages: list, before: set) -> bool:
    found = {passage.get("id") for passage in passages if passage.get("id") is not None}
    return bool(found - before)


def is_pool_duplicate(query: str, pool: dict) -> bool:
    return is_duplicate_query(query, pool)


def prepare_pool(ctx: StageContext, state: QuestionState) -> None:
    """Attach the anchored store, or a delimiter-aware view of the baseline dict."""
    if ctx.flags.evidence_store == "anchored":
        state.evidence = AnchoredEvidenceStore()
    elif ctx.flags.evidence_delimiters:
        state.query_pool = EvidencePool(delimiters=True)


def sync_pool(ctx: StageContext, state: QuestionState) -> None:
    """Expose the store as the baseline query -> passages mapping for saved output."""
    if state.evidence is not None:
        state.query_pool = EvidencePool(state.evidence.to_query_pool(),
                                        entries=state.evidence.entries,
                                        delimiters=ctx.flags.evidence_delimiters)


def search(ctx: StageContext, state: QuestionState, query: str) -> list:
    return ctx.backends.search(query, call=state.call("retrieval", "retriever"))


def log_rewrite_audit(ctx: StageContext, state: QuestionState, source: str,
                      rewrite: str) -> None:
    """Log entities/numbers the rewrite invents; never changes control flow."""
    if not ctx.flags.rewrite_novelty_audit:
        return
    state.decisions.append({"stage": "query", "round": state.round,
                            "decision": "rewrite_audit", **audit_rewrite(source, rewrite)})


def _anchored_optimize(ctx: StageContext, state: QuestionState, original: str,
                       new_query: str) -> str:
    store = state.evidence
    active = store.active_pool()
    old_key = find_most_similar_key(active, original)
    if ctx.flags.skip_duplicate_query and is_pool_duplicate(new_query, active):
        return EXIT_DUPLICATE
    before = store.ids()
    passages = search(ctx, state, new_query)
    if ctx.flags.zero_novelty_exit and not has_new_id(passages, before):
        return EXIT_ZERO_NOVELTY
    store.add_query(new_query, passages, state.round)
    store.supersede(old_key)
    sync_pool(ctx, state)
    return ACTION_APPLIED


def _anchored_expand(ctx: StageContext, state: QuestionState, new_query: str) -> str:
    store = state.evidence
    active = store.active_pool()
    if ctx.flags.skip_duplicate_query and is_pool_duplicate(new_query, active):
        return EXIT_DUPLICATE
    before = store.ids()
    passages = search(ctx, state, new_query)
    if ctx.flags.zero_novelty_exit and not has_new_id(passages, before):
        return EXIT_ZERO_NOVELTY
    store.add_query(new_query, passages, state.round)
    sync_pool(ctx, state)
    return ACTION_APPLIED


def _optimize(ctx: StageContext, state: QuestionState, instruction: str) -> str:
    original, new_query = split_optimization(instruction)
    log_rewrite_audit(ctx, state, original, new_query)
    if state.evidence is not None:
        return _anchored_optimize(ctx, state, original, new_query)
    pool = state.query_pool
    old_key = find_most_similar_key(pool, original)
    if ctx.flags.skip_duplicate_query and is_pool_duplicate(new_query, pool):
        return EXIT_DUPLICATE
    before = pool_ids(pool)
    saved = list(pool.items()) if ctx.flags.zero_novelty_exit else None
    pool.pop(old_key)
    passages = search(ctx, state, new_query)
    if ctx.flags.zero_novelty_exit and not has_new_id(passages, before):
        pool.clear()
        pool.update(saved)
        return EXIT_ZERO_NOVELTY
    pool[new_query] = passages
    return ACTION_APPLIED


def _expand(ctx: StageContext, state: QuestionState, new_query: str) -> str:
    log_rewrite_audit(ctx, state, state.question, new_query)
    if state.evidence is not None:
        return _anchored_expand(ctx, state, new_query)
    pool = state.query_pool
    if ctx.flags.skip_duplicate_query and is_pool_duplicate(new_query, pool):
        return EXIT_DUPLICATE
    before = pool_ids(pool)
    passages = search(ctx, state, new_query)
    if ctx.flags.zero_novelty_exit and not has_new_id(passages, before):
        return EXIT_ZERO_NOVELTY
    pool[new_query] = passages
    return ACTION_APPLIED


def _apply_action(ctx: StageContext, state: QuestionState, text: str) -> str:
    if "Query Optimization:" in text:
        return _optimize(ctx, state, text.split("Query Optimization:")[1].strip())
    if "Query Expansion:" in text:
        return _expand(ctx, state, text.split("Query Expansion:")[1].strip())
    return ACTION_NONE


def maintain_query_pool(ctx: StageContext, state: QuestionState,
                        opponent_output: str) -> str:
    """Apply the Opponent's action in place; returns an action or exit outcome.

    ``EXIT_DUPLICATE`` means the duplicate guard skipped the search and the stage must
    end; ``EXIT_ZERO_NOVELTY`` means the search added no passage ID and the pool is
    restored to its pre-action contents. ``ACTION_NONE`` means no action was proposed.
    """
    try:
        return _apply_action(ctx, state, opponent_output)
    except Exception as error:
        raise QueryPoolError(
            f"Error in maintaining the query pool for Opponent output {opponent_output!r}"
        ) from error
