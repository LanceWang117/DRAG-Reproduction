"""Optional MiniCheck adapter (J19), imported only when configured.

The external ``minicheck`` package and its weights are never imported on the default
path or in CI. Construction is cheap: the package is imported and the model loaded on
the first ``score`` call. A missing package raises ``FaithfulnessConfigError`` with the
install command; there is no silent fallback to the heuristic.

MiniCheck is sentence-level (Tang et al., EMNLP 2024). The adapter normalises the
answer with E(.), scores one ``(passage, answer)`` pair per cited passage, lets
MiniCheck chunk long passages, and returns the maximum support probability: the best
cited passage. Empty answers or no passages score 0.0. Model loads are cached per
``(model_name, cache_dir, loader)`` so repeated rounds do not reload weights.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

from ..analysis.extract import extract_answer
from .scorer import FaithfulnessConfigError

DEFAULT_MODEL = "flan-t5-large"
_INSTALL = ('pip install "minicheck @ '
            'git+https://github.com/Liyan06/MiniCheck.git@main"')
_MODELS = {}


def _load_minicheck(model_name: str, cache_dir: Optional[str]):
    try:
        from minicheck.minicheck import MiniCheck
    except ImportError as exc:
        raise FaithfulnessConfigError(
            "faithfulness_scorer='minicheck' needs the optional 'minicheck' "
            f"package; install it with: {_INSTALL}") from exc
    return MiniCheck(model_name=model_name, cache_dir=cache_dir)


class MiniCheckFaithfulness:
    """Lazy adapter around ``MiniCheck.score``; one soft score per answer."""

    def __init__(self, yes_no: bool = False, model_name: str = DEFAULT_MODEL,
                 cache_dir: Optional[str] = None,
                 loader: Optional[Callable] = None):
        self.yes_no = yes_no
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._loader = loader or _load_minicheck

    def _model(self):
        key = (self.model_name, self.cache_dir, self._loader)
        if key not in _MODELS:
            try:
                _MODELS[key] = self._loader(self.model_name, self.cache_dir)
            except ImportError as exc:
                raise FaithfulnessConfigError(
                    "faithfulness_scorer='minicheck' could not import the optional "
                    f"package: {exc}; install it with: {_INSTALL}") from exc
        return _MODELS[key]

    def score(self, answer: str, passages: Sequence[str]) -> float:
        """Maximum MiniCheck support probability over the cited passages."""
        claim = extract_answer(answer, yes_no=self.yes_no)
        docs = [str(passage) for passage in passages if str(passage).strip()]
        if not claim or not docs:
            return 0.0
        _, probabilities, _, _ = self._model().score(
            docs=docs, claims=[claim] * len(docs))
        values = [float(value) for value in probabilities or []]
        if not values:
            return 0.0
        return min(max(max(values), 0.0), 1.0)

    def __call__(self, answer: str, passages: Sequence[str]) -> float:
        return self.score(answer, passages)
