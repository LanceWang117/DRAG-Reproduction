"""Answer scoring identical to FlashRAG 0.1.4.dev20250111.

Mirrors ``flashrag/evaluator/utils.py::normalize_answer`` and the EM, acc (sub-exact
match) and token F1/precision/recall metrics in ``flashrag/evaluator/metrics.py``, so
scores computed here equal the ones stored in ``results/``.
"""
from collections import Counter
import re
import string

_PUNCTUATION = set(string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_SPECIAL_ANSWERS = ("yes", "no", "noanswer")


def normalize_answer(text):
    """Lowercase, remove punctuation, remove articles, collapse whitespace."""
    lowered = text.lower()
    without_punctuation = "".join(ch for ch in lowered if ch not in _PUNCTUATION)
    return " ".join(_ARTICLES.sub(" ", without_punctuation).split())


def _as_list(golden_answers):
    if isinstance(golden_answers, str):
        return [golden_answers]
    return list(golden_answers)


def exact_match(prediction, golden_answers):
    """1.0 if the normalised prediction equals any normalised gold answer."""
    prediction = normalize_answer(prediction)
    golds = (normalize_answer(gold) for gold in _as_list(golden_answers))
    return float(any(gold == prediction for gold in golds))


def substring_match(prediction, golden_answers):
    """FlashRAG's ``acc``: some normalised gold answer occurs in the prediction."""
    prediction = normalize_answer(prediction)
    golds = (normalize_answer(gold) for gold in _as_list(golden_answers))
    return float(any(gold in prediction for gold in golds))


def _overlap_scores(prediction, gold):
    """Token overlap for one normalised pair, or None when FlashRAG skips the pair."""
    special = prediction in _SPECIAL_ANSWERS or gold in _SPECIAL_ANSWERS
    if special and prediction != gold:
        return None
    prediction_tokens, gold_tokens = prediction.split(), gold.split()
    same = sum((Counter(prediction_tokens) & Counter(gold_tokens)).values())
    if same == 0:
        return None
    precision = same / len(prediction_tokens)
    recall = same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return {"f1": f1, "precision": precision, "recall": recall}


def token_scores(prediction, golden_answers):
    """Token F1, precision and recall, each maximised over the gold answers."""
    best = {"f1": 0, "precision": 0, "recall": 0}
    prediction = normalize_answer(prediction)
    for gold in _as_list(golden_answers):
        scores = _overlap_scores(prediction, normalize_answer(gold))
        if scores is not None:
            best = {key: max(scores[key], best[key]) for key in best}
    return best


def score_answer(prediction, golden_answers):
    """All five metrics FlashRAG stores per question: em, acc, f1, precision, recall."""
    return {
        "em": exact_match(prediction, golden_answers),
        "acc": substring_match(prediction, golden_answers),
        **token_scores(prediction, golden_answers),
    }
