"""Label log-probabilities for the backends that have no tokenizer (J24).

The vLLM scorer (``judge.logprob.ScoredGenerator``) maps a label to token IDs. The local
Ollama server returns token *strings* instead (OpenAI shape, verified on 0.34.4): the
first position's ``top_logprobs`` holds entries such as ``Opp``, `` Opp``, ``Op``,
``Pro`` and ``Prop``. A label's score is the log-sum-exp of every entry whose stripped
token is a case-insensitive prefix of that label and at least two characters long, so
``I`` never counts for ``insufficient`` and ``O`` never counts for ``Opponent``. An
entry that prefixes two labels counts for neither; a label with no entry scores None, so
the caller falls back to the generated text, exactly as for vLLM.

The fake backend scores labels from a hash of the prompt: finite, deterministic and
meaningless, enough to exercise the log-prob path offline.
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

MIN_PREFIX = 2


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


def first_position(choice: object) -> List[Tuple[str, object]]:
    """The ``(token, logprob)`` pairs of an OpenAI-shaped choice's first position."""
    try:
        content = choice["logprobs"]["content"]  # type: ignore[index]
        entries = content[0].get("top_logprobs") or [content[0]]
        return [(entry.get("token"), entry.get("logprob")) for entry in entries]
    except (KeyError, IndexError, TypeError, AttributeError):
        return []


def fake_label_logprobs(text: str, labels: Sequence[str]) -> Dict[str, Optional[float]]:
    """A deterministic finite log-probability per label, from a hash of the prompt."""
    scores = {}
    for label in labels:
        digest = hashlib.sha256(f"{label}\n{text}".encode("utf-8")).hexdigest()
        scores[label] = -0.05 - (int(digest, 16) % 400) / 100.0
    return scores
