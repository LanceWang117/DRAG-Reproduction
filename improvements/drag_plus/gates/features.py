"""Label-free escalation features for the answer stage (J14).

Every feature is a finite number computable from the transcript and the retrieved
evidence alone: no gold answer, no reference passage and no test label is read. A
missing optional signal uses a neutral value, never an unavailable sentinel, so the
gate model always receives the four names it was trained on.

Features (``FEATURE_NAMES`` order):

* ``disagreement`` - ``1 - modal cluster share`` under ``E(.)`` extraction;
* ``cluster_entropy`` - Shannon entropy over answer clusters, divided by ``log k``;
* ``s_bar_O`` - ``sigmoid(judge_score)``, the J12 Opponent-minus-Proponent margin as
  a probability; ``0.5`` when the query stage recorded no score;
* ``evidence_support`` - fraction of the extracted answer's tokens that occur in the
  retrieved passages. This is the J14 stub; ``support_scorer`` is the J19 hook, supplied
  by ``grounding.build_scorer`` when ``faithfulness_scorer`` is configured.
"""
from __future__ import annotations

import math
import re
from typing import Callable, Iterable, List, Mapping, Optional, Sequence

from ..analysis.extract import extract_answer

FEATURE_NAMES = ("disagreement", "cluster_entropy", "s_bar_O", "evidence_support")
_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    """Lowercase alphanumeric tokens; punctuation and case never split a cluster."""
    return _WORD.findall(text.lower())


def _cluster_key(answer: str) -> str:
    """Normalise an answer so ``Paris`` and ``paris.`` land in one cluster."""
    return " ".join(_tokens(answer)) or answer.strip().lower()


def agent_answers(state, yes_no: bool = False) -> List[str]:
    """The latest answer each agent wrote, in first-seen agent order."""
    answers = []
    for history in state.answer_messages.values():
        if history:
            answers.append(extract_answer(history[-1].get("content", ""), yes_no))
    return answers


def answer_clusters(answers: Sequence[str]) -> List[List[str]]:
    """Group answers by normalised string; empty answers form their own cluster."""
    groups = {}
    for answer in answers:
        groups.setdefault(_cluster_key(answer), []).append(answer)
    return list(groups.values())


def disagreement(answers: Sequence[str]) -> float:
    """``1 - (largest cluster / all answers)``; fewer than two answers is zero."""
    if len(answers) < 2:
        return 0.0
    sizes = [len(group) for group in answer_clusters(answers)]
    return 1.0 - max(sizes) / len(answers)


def cluster_entropy(answers: Sequence[str]) -> float:
    """Normalised entropy over answer clusters, in ``[0, 1]``; one cluster is zero."""
    if not answers:
        return 0.0
    clusters = answer_clusters(answers)
    if len(clusters) <= 1:
        return 0.0
    counts = [len(group) for group in clusters]
    total = sum(counts)
    entropy = -sum((count / total) * math.log(count / total) for count in counts)
    return entropy / math.log(len(counts))


def passage_texts(query_pool) -> List[str]:
    """Every retrieved passage's text, in pool order (the J19 scorers' evidence view)."""
    texts = []
    for passages in query_pool.values():
        texts.extend(str(passage.get("contents", "")) for passage in passages)
    return texts


def evidence_support(answer: str, passages: Iterable[str],
                     scorer: Optional[Callable[[str, Iterable[str]], float]] = None) -> float:
    """Answer-token containment in the passages, or the injected J19 scorer's value."""
    texts = list(passages)
    if scorer is not None:
        return float(scorer(answer, texts))
    tokens = _tokens(answer)
    if not tokens or not texts:
        return 0.0
    available = set(_tokens("\n".join(texts)))
    return sum(token in available for token in tokens) / len(tokens)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def s_bar_o(judge_score: Optional[float]) -> float:
    """The query-stage Opponent margin as a probability; no score is neutral."""
    if judge_score is None or not math.isfinite(judge_score):
        return 0.5
    return _sigmoid(float(judge_score))


def compute_features(state, yes_no: bool = False,
                     support_scorer: Optional[Callable] = None) -> Mapping[str, float]:
    """Return every feature for one question; the result order follows ``FEATURE_NAMES``."""
    answers = agent_answers(state, yes_no)
    answer = next((value for value in answers if value), "")
    values = {
        "disagreement": disagreement(answers),
        "cluster_entropy": cluster_entropy(answers),
        "s_bar_O": s_bar_o(getattr(state, "judge_score", None)),
        "evidence_support": evidence_support(answer, passage_texts(state.query_pool),
                                             support_scorer),
    }
    return {name: float(values[name]) for name in FEATURE_NAMES}
