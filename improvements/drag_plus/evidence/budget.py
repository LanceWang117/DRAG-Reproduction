"""Token-budgeted evidence selection for the prompt context.

Selection is greedy MMR-style: relevance per token (retrieval rank is the signal the
pipeline carries) minus a redundancy penalty based on token-set overlap with the
passages already chosen, with at most ``max_per_article`` chunks per article. Anchored
passages are never dropped; if they alone exceed the budget the lowest-rank anchored
passage is truncated and the caller logs the warning. Counting and rendering use the
same formatter, so a budget check cannot disagree with the text the model sees.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Sequence, Tuple

from .format import EvidenceEntry, entries_from_pool, format_entries

DEFAULT_PER_ARTICLE = 2
DEFAULT_REDUNDANCY = 0.5


class BudgetError(ValueError):
    """The budget cannot hold the fixed context and the anchored passages."""


@dataclass(frozen=True)
class Selection:
    """The passages a budget admits, with the labels it dropped or truncated."""
    entries: Tuple[EvidenceEntry, ...]
    dropped: Tuple[str, ...] = ()
    truncated: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()
    context_tokens: int = 0


@dataclass(frozen=True)
class BudgetedEvidence:
    """A rendered evidence block; ``format_query_pool`` calls ``format_evidence``."""
    text: str
    selection: Selection

    def format_evidence(self) -> str:
        return self.text


def _words(text: str) -> set:
    return set(str(text).casefold().split())


def _jaccard(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _article(entry: EvidenceEntry) -> str:
    """Article identity for the cap: the passage's first line (its FlashRAG title)."""
    title = entry.contents.split("\n", 1)[0].strip().casefold()
    return title or f"doc:{entry.doc_id}"


def _render(entries: Sequence[EvidenceEntry], delimiters: bool) -> str:
    return format_entries(entries, delimiters)


def _cost(entries: Sequence[EvidenceEntry], count_tokens: Callable[[str], int],
          delimiters: bool) -> int:
    return count_tokens(_render(entries, delimiters))


def _truncate(entry: EvidenceEntry, selected: List[EvidenceEntry], index: int,
              available: int, count_tokens: Callable[[str], int],
              delimiters: bool) -> EvidenceEntry:
    """Shrink one passage's words until the whole anchored block fits."""
    words = entry.contents.split()
    while words:
        candidate = replace(entry, contents=" ".join(words))
        trial = list(selected)
        trial[index] = candidate
        if _cost(trial, count_tokens, delimiters) <= available:
            return candidate
        words.pop()
    return replace(entry, contents="")


def _fit_anchored(anchored: List[EvidenceEntry], available: int,
                  count_tokens: Callable[[str], int], delimiters: bool,
                  budget: int) -> Tuple[List[EvidenceEntry], List[str], List[str]]:
    """Keep every anchored passage; truncate lowest-rank ones until the block fits."""
    selected = list(anchored)
    if _cost(selected, count_tokens, delimiters) <= available:
        return selected, [], []
    order = sorted(range(len(selected)),
                   key=lambda i: (-selected[i].rank, -selected[i].qid))
    truncated, warnings = [], []
    for index in order:
        if _cost(selected, count_tokens, delimiters) <= available:
            break
        before = selected[index]
        selected[index] = _truncate(before, selected, index, available, count_tokens,
                                    delimiters)
        if selected[index].contents == before.contents:
            continue
        truncated.append(before.label)
        warnings.append(
            f"truncated anchored passage {before.label} to fit context_token_budget={budget}")
    if _cost(selected, count_tokens, delimiters) > available:
        raise BudgetError(
            f"context_token_budget={budget} cannot hold the anchored passages")
    return selected, truncated, warnings


def _utility(entry: EvidenceEntry, chosen: List[EvidenceEntry], cost: int,
             redundancy: float) -> float:
    words = _words(entry.contents)
    similarity = max((_jaccard(words, _words(other.contents)) for other in chosen),
                     default=0.0)
    density = (1.0 / max(entry.rank, 1)) / max(cost, 1)
    return density * (1.0 - redundancy * similarity)


def _select_free(free: List[EvidenceEntry], anchored: List[EvidenceEntry],
                 available: int, count_tokens: Callable[[str], int], delimiters: bool,
                 max_per_article: int, redundancy: float) -> List[EvidenceEntry]:
    """Greedy MMR over the non-anchored passages; anchored passages seed the penalty."""
    chosen: List[EvidenceEntry] = list(anchored)
    pending = list(free)
    counts = Counter(_article(entry) for entry in chosen)
    current = _cost(chosen, count_tokens, delimiters)
    while True:
        best, best_score, best_cost, best_index = None, -1.0, 0, -1
        for index, entry in enumerate(pending):
            if counts[_article(entry)] >= max_per_article:
                continue
            cost = _cost(chosen + [entry], count_tokens, delimiters) - current
            if cost > available - current:
                continue
            score = _utility(entry, chosen, cost, redundancy)
            if score > best_score:
                best, best_score, best_cost, best_index = entry, score, cost, index
        if best is None:
            break
        pending.pop(best_index)
        chosen.append(best)
        counts[_article(best)] += 1
        current += best_cost
    return chosen[len(anchored):]


def _ordered(entries: Sequence[EvidenceEntry],
             chosen: Sequence[EvidenceEntry]) -> Tuple[EvidenceEntry, ...]:
    """Render in first-seen order, replacing a truncated entry with its shorter text."""
    by_id = {entry.doc_id: entry for entry in chosen}
    return tuple(by_id[entry.doc_id] for entry in entries if entry.doc_id in by_id)


def select_entries(entries: Sequence[EvidenceEntry], count_tokens: Callable[[str], int],
                   budget: int, fixed_text: str = "", *, delimiters: bool = False,
                   max_per_article: int = DEFAULT_PER_ARTICLE,
                   redundancy: float = DEFAULT_REDUNDANCY) -> Selection:
    """Select passages that fit ``budget`` after the fixed header, question and history.

    ``fixed_text`` is every non-evidence token the prompt will carry (excluding role
    instructions, which are not part of the budgeted context). Anchored entries are
    always kept; non-anchored entries are chosen by utility and can be dropped.
    """
    entries = list(entries)
    available = budget - count_tokens(fixed_text)
    if available < 0:
        raise BudgetError(
            f"context_token_budget={budget} is smaller than the fixed context")
    anchored = [entry for entry in entries if entry.anchored]
    free = [entry for entry in entries if not entry.anchored]
    anchored, truncated, warnings = _fit_anchored(anchored, available, count_tokens,
                                                  delimiters, budget)
    chosen = anchored + _select_free(free, anchored, available, count_tokens,
                                     delimiters, max_per_article, redundancy)
    ordered = _ordered(entries, chosen)
    chosen_ids = {entry.doc_id for entry in chosen}
    dropped = tuple(entry.label for entry in entries if entry.doc_id not in chosen_ids)
    context_tokens = count_tokens(fixed_text) + _cost(ordered, count_tokens, delimiters)
    if context_tokens > budget:
        raise BudgetError(f"context_token_budget={budget} would be exceeded")
    return Selection(ordered, dropped, tuple(truncated), tuple(warnings), context_tokens)


def entries_of(pool) -> Tuple[List[EvidenceEntry], bool]:
    """Entries and delimiter mode for an anchored store, a pool wrapper or a dict."""
    entries = getattr(pool, "entries", None)
    if entries is None:
        entries = entries_from_pool(pool)
    delimiters = bool(getattr(pool, "delimiters", False))
    return list(entries), delimiters


def budgeted_evidence(pool, limit: int, count_tokens: Callable[[str], int],
                      fixed_text: str = "",
                      delimiters: Optional[bool] = None) -> BudgetedEvidence:
    """Apply the budget to one prompt-facing evidence block."""
    entries, pool_delimiters = entries_of(pool)
    if delimiters is None:
        delimiters = pool_delimiters
    selection = select_entries(entries, count_tokens, limit, fixed_text,
                               delimiters=delimiters)
    return BudgetedEvidence(_render(selection.entries, delimiters), selection)
