"""Label log-probabilities for the backends that have no tokenizer (J24).

The vLLM scorer (``judge.logprob.ScoredGenerator``) maps a label to token IDs. The local
Ollama server returns token *strings* instead (OpenAI shape, verified on 0.34.4): a
position's ``top_logprobs`` holds entries such as ``Opp``, `` Opp``, ``Pro`` and
``Prop``. The scored position is the first one whose chosen token is not bare
punctuation (``**``, a quote, a newline), at most ``MAX_SKIPPED`` in. A label's score is
the log-sum-exp of every entry whose stripped token is a case-insensitive prefix of the
label and at least three characters long, so prose openings such as ``In`` or ``I``
never count for ``insufficient``. An entry that prefixes two labels counts for neither.

Scores are trusted only when the reply itself starts with a label (after the same
punctuation): a reply opening with prose ("The Opponent...") scores None for every
label, so the caller falls back to parsing the text, exactly as for a missing vLLM score.

The fake backend scores labels from a hash of the prompt: finite, deterministic and
meaningless, enough to exercise the log-prob path offline.
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

MIN_PREFIX = 3
MAX_SKIPPED = 3
_PUNCTUATION = " \t\r\n*_\"'`([{<:-."


def _finite(value: object) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _logsumexp(values: List[float]) -> float:
    peak = max(values)
    return peak + math.log(sum(math.exp(value - peak) for value in values))


def _owners(token: str, labels: Sequence[str]) -> List[str]:
    stripped = token.strip().lower()
    if len(stripped) < MIN_PREFIX:
        return []
    return [label for label in labels if label.lower().startswith(stripped)]


def label_logprobs(top: Iterable[Tuple[str, object]],
                   labels: Sequence[str]) -> Dict[str, Optional[float]]:
    """``[(token, logprob), ...]`` at one position -> ``{label: logprob or None}``."""
    found: Dict[str, List[float]] = {label: [] for label in labels}
    seen = set()
    for token, value in top:
        number = _finite(value)
        if not isinstance(token, str) or number is None or token in seen:
            continue
        seen.add(token)
        owners = _owners(token, labels)
        if len(owners) == 1:
            found[owners[0]].append(number)
    return {label: (_logsumexp(values) if values else None)
            for label, values in found.items()}


def starts_with_label(text: str, labels: Sequence[str]) -> bool:
    """True when the reply, after leading punctuation, begins with one of the labels."""
    head = text.lstrip(_PUNCTUATION).lower() if isinstance(text, str) else ""
    return any(label and head.startswith(label.lower()) for label in labels)


def first_position(choice: object) -> List[Tuple[str, object]]:
    """``(token, logprob)`` pairs at the first position that is not bare punctuation."""
    try:
        content = choice["logprobs"]["content"]  # type: ignore[index]
        for position in content[:MAX_SKIPPED + 1]:
            if str(position.get("token", "")).strip(_PUNCTUATION):
                entries = position.get("top_logprobs") or [position]
                return [(entry.get("token"), entry.get("logprob")) for entry in entries]
        return []
    except (KeyError, IndexError, TypeError, AttributeError):
        return []


def scored_labels(choice: object, text: str,
                  labels: Sequence[str]) -> Dict[str, Optional[float]]:
    """Label scores for one reply, or all None when the reply does not open with a label."""
    if not starts_with_label(text, labels):
        return {label: None for label in labels}
    return label_logprobs(first_position(choice), labels)


def fake_label_logprobs(text: str, labels: Sequence[str]) -> Dict[str, Optional[float]]:
    """A deterministic finite log-probability per label, from a hash of the prompt."""
    scores = {}
    for label in labels:
        digest = hashlib.sha256(f"{label}\n{text}".encode("utf-8")).hexdigest()
        scores[label] = -0.05 - (int(digest, 16) % 400) / 100.0
    return scores
