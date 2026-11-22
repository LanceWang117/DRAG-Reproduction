"""Label log-probability Judge scoring (J12).

The generative path parses the Judge's text and retries once when no single agent is
named. The log-prob path instead scores two explicit labels, ``Proponent`` and
``Opponent``, at the first generated position: ``s_O = log p(Opponent) - log
p(Proponent)``. A verdict is then always defined, so the round is never wasted and no
retry call is made.

Edge cases: a missing or non-finite label score is never treated as zero; the round
falls back to the generated text from the same call and logs ``judge_fallback``. A
margin exactly equal to ``judge_tau`` goes to Proponent (the conservative "stop
retrieving" side). Order balancing needs both presentations to score; if either one is
missing, the round falls back instead of averaging one order.

The vLLM adapter at the bottom is lazy: this module imports no FlashRAG, vLLM, torch or
FAISS, so it is safe to import on the laptop and from ``drag_plus`` itself.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..judge_decision import parse_judge_decision

LABELS: Tuple[str, str] = ("Proponent", "Opponent")
DEFAULT_LOGPROB_K = 32


@dataclass(frozen=True)
class ScoredOutput:
    """One generated text plus one optional log-probability per label."""

    text: str
    label_logprobs: Mapping[str, Optional[float]]


@dataclass(frozen=True)
class ScoredVerdict:
    """The log-prob verdict, the recorded margin and whether text parsing was used."""

    decision: Optional[str]
    status: str
    score: Optional[float]


def _finite(value: Any) -> Optional[float]:
    """Return a finite float, or None for None, bools, strings, NaN and infinity."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def margin(label_logprobs: Mapping[str, Optional[float]]) -> Optional[float]:
    """``s_O = log p(Opponent) - log p(Proponent)``; None unless both are finite."""
    proponent = _finite(label_logprobs.get("Proponent"))
    opponent = _finite(label_logprobs.get("Opponent"))
    if proponent is None or opponent is None:
        return None
    return opponent - proponent


def decide(score: float, tau: float) -> str:
    """Opponent wins iff the margin exceeds ``tau``; an exact tie goes to Proponent."""
    return "Opponent" if score > tau else "Proponent"


def _mean_margin(margins: Sequence[Optional[float]]) -> Optional[float]:
    """Average both orders only when every order has a valid margin."""
    if not margins or any(value is None for value in margins):
        return None
    return sum(value for value in margins if value is not None) / len(margins)


def resolve_scored(orders: Sequence[ScoredOutput], tau: float,
                   order_balance: bool) -> ScoredVerdict:
    """Turn one or two scored presentations into a verdict.

    ``order_balance`` averages the margins of both orders; otherwise the first order
    decides. On a missing score the generated text from the first order is parsed, with
    no second call. ``status`` is ``valid_logprob``, ``fallback_generative`` or
    ``unresolved``.
    """
    margins = [margin(order.label_logprobs) for order in orders]
    score = _mean_margin(margins) if order_balance else (margins[0] if margins else None)
    if score is not None:
        return ScoredVerdict(decide(score, tau), "valid_logprob", score)
    fallback = parse_judge_decision(orders[0].text) if orders else None
    status = "fallback_generative" if fallback is not None else "unresolved"
    return ScoredVerdict(fallback, status, None)


class ScoredGenerator:
    """FlashRAG's ``VLLMGenerator`` plus ``generate_with_scores`` for the Judge.

    ``generate`` is the unchanged baseline path. ``generate_with_scores`` asks vLLM for
    the top-k log-probabilities of the first generated position (``max_tokens=1``) and
    maps each label's first token to its log-probability. The label is tried with and
    without a leading space; a label whose first token is absent, or two labels that
    share it and cannot be summed over the returned positions, maps to ``None`` so the
    pipeline falls back to the generated text.

    ``outputs`` must be raw vLLM ``RequestOutput`` objects, which FlashRAG returns for
    ``return_raw_output=True``.
    """

    def __init__(self, generator: Any, logprob_k: int = DEFAULT_LOGPROB_K):
        self._generator = generator
        self._tokenizer = generator.tokenizer
        self._logprob_k = max(int(logprob_k), 2)

    def generate(self, prompts: Any, **params: Any) -> List[str]:
        return self._generator.generate(prompts, **params)

    def generate_with_scores(self, prompts: Any, labels: Sequence[str],
                             **params: Any) -> List[ScoredOutput]:
        """One vLLM call for all prompts; one scored text per prompt, in order."""
        batch = prompts if isinstance(prompts, list) and prompts and not isinstance(
            prompts[0], dict) else [prompts]
        options = {"return_raw_output": True, "temperature": 0.0, "max_tokens": 1,
                   "logprobs": max(self._logprob_k, 2 * len(labels) + 4)}
        options.update(params)
        raw = self._generator.generate(batch, **options)
        return [self._score_one(output, labels) for output in raw]

    def _score_one(self, output: Any, labels: Sequence[str]) -> ScoredOutput:
        completion = output.outputs[0]
        positions = [self._token_map(position)
                     for position in (completion.logprobs or [])]
        return ScoredOutput(completion.text, self._label_scores(positions, labels))

    @staticmethod
    def _token_map(position: Mapping[int, Any]) -> Dict[int, float]:
        """``{token_id: Logprob}`` -> ``{token_id: finite logprob}``."""
        mapped = {}
        for token, item in position.items():
            value = _finite(getattr(item, "logprob", item))
            if value is not None:
                mapped[int(token)] = value
        return mapped

    def _candidates(self, label: str) -> Iterable[Tuple[int, ...]]:
        for text in (label, " " + label):
            ids = tuple(self._tokenizer.encode(text, add_special_tokens=False))
            if ids:
                yield ids

    def _label_scores(self, positions: List[Dict[int, float]],
                      labels: Sequence[str]) -> Dict[str, Optional[float]]:
        scores: Dict[str, Optional[float]] = {label: None for label in labels}
        picked = {}
        for label in labels:
            for ids in self._candidates(label):
                if positions and ids[0] in positions[0]:
                    picked[label] = ids
                    break
        if len(picked) != len(labels):
            return scores
        firsts = [ids[0] for ids in picked.values()]
        if len(set(firsts)) == len(firsts):
            return {label: positions[0][ids[0]] for label, ids in picked.items()}
        return {label: self._sequence_sum(positions, ids)
                for label, ids in picked.items()}

    @staticmethod
    def _sequence_sum(positions: List[Dict[int, float]],
                      ids: Tuple[int, ...]) -> Optional[float]:
        """Sum a label's log-probs when every token position was returned."""
        if len(ids) > len(positions):
            return None
        total = 0.0
        for token, position in zip(ids, positions):
            if token not in position:
                return None
            total += position[token]
        return total
