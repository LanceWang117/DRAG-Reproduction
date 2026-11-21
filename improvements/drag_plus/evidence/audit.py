"""Log-only rewrite audit: entities and numbers a rewrite introduces.

Capitalisation and digit runs are cheap signals that need no tokenizer or model, so the
audit can run in CI. They are precision-first, not complete: sentence-initial words and
common function words are ignored, and a missed lowercase entity is accepted. The
result is a decision-log signal only; it never changes control flow.
"""
from __future__ import annotations

import re
import unicodedata

_TOKEN = re.compile(r"[^\W\d_][\w'’-]*", re.UNICODE)
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?%?")
_STOP_ENTITIES = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "i", "it", "its", "we",
    "you", "they", "he", "she", "his", "her", "their", "our", "and", "but", "or",
    "if", "when", "what", "who", "where", "which", "how", "why", "yes", "no",
    "question", "however", "also", "given", "from", "in", "on", "for", "of",
})
_BREAKS = ".!?;:\n"
_TRAILING = ",;:"


def _capitalized(word: str) -> bool:
    """Title case (``Jaffa``, ``Pelé``) or an acronym of at least two letters."""
    if not word[:1].isupper():
        return False
    return word.isupper() and len(word) >= 2 or word[1:].islower()


def _add_group(found: set, group: list, at_start: bool) -> None:
    """Record one run of capitalised tokens; a lone word must be mid-sentence."""
    if not group:
        return
    words = [word.casefold() for word in group]
    if len(group) >= 2:
        if not all(word in _STOP_ENTITIES for word in words):
            found.add(" ".join(words))
        return
    if not at_start and words[0] not in _STOP_ENTITIES:
        found.add(words[0])


def extract_entities(text: str) -> set:
    """Case-folded proper-noun candidates from capitalisation heuristics."""
    source = unicodedata.normalize("NFKC", str(text))
    found, group, at_start, previous_end = set(), [], True, 0
    for match in _TOKEN.finditer(source):
        gap = source[previous_end:match.start()]
        if any(char in _BREAKS for char in gap):
            _add_group(found, group, at_start)
            group, at_start = [], True
        word = match.group(0)
        if _capitalized(word):
            group.append(word)
        else:
            _add_group(found, group, at_start)
            group, at_start = [], False
        previous_end = match.end()
    _add_group(found, group, at_start)
    return found


def _normalise_number(token: str) -> str:
    """Canonical form so ``1,200``/``1200`` and ``40%``/``40 %`` compare equal."""
    token = token.replace(",", "").rstrip("." + _TRAILING)
    percent = token.endswith("%")
    try:
        value = float(token.rstrip("%"))
    except ValueError:
        return token
    rendered = str(int(value)) if value.is_integer() else f"{value:g}"
    return f"{rendered}%" if percent else rendered


def extract_numbers(text: str) -> set:
    """Normalised number-like tokens: integers, decimals, grouped digits, percents."""
    return {_normalise_number(match.group(0)) for match in _NUMBER.finditer(str(text))
            if any(char.isdigit() for char in match.group(0))}


def audit_rewrite(source: str, rewrite: str) -> dict:
    """Entities and numbers in ``rewrite`` that ``source`` does not contain."""
    return {
        "invented_entities": sorted(extract_entities(rewrite) - extract_entities(source)),
        "invented_numbers": sorted(extract_numbers(rewrite) - extract_numbers(source)),
    }
