"""Paired statistics for comparing DRAG variants (docs/ledger/protocol.md section 4).

Standard library plus numpy only. Methods and sources are recorded in the J01 research
note. In brief:

* McNemar: exact conditional two-sided test (default) or mid-p (Fagerland et al. 2013).
* Sample size and MDE: Connor (1987) for paired binary outcomes.
* TOST: equivalence when the (1 - 2*alpha) normal-approximation CI lies inside the margin
  (Schuirmann 1987).
* Wilcoxon signed-rank: normal approximation with tie correction; zeros dropped.
"""
from fractions import Fraction
from math import ceil, comb, sqrt
from statistics import NormalDist

import numpy as np

_NORMAL = NormalDist()


def discordant_counts(new_correct, base_correct):
    """(b, c): questions only the new arm got right, and only the base arm got right."""
    pairs = list(zip(new_correct, base_correct))
    only_new = sum(1 for new, base in pairs if new and not base)
    only_base = sum(1 for new, base in pairs if base and not new)
    return only_new, only_base


def _binomial_tail(n, k):
    """P(X <= k) for X ~ Bin(n, 1/2), as an exact fraction."""
    return Fraction(sum(comb(n, i) for i in range(k + 1)), 2 ** n)


def mcnemar_p(b, c, midp=False):
    """Two-sided McNemar p-value from the discordant counts b and c."""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    if midp:
        point = Fraction(comb(n, k), 2 ** n)
        lower = _binomial_tail(n, k - 1) if k > 0 else Fraction(0)
        return min(1.0, float(2 * (lower + point / 2)))
    return min(1.0, float(2 * _binomial_tail(n, k)))


def holm(p_values, alpha=0.05):
    """Holm-Bonferroni step-down; returns reject flags in the input order."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    reject = [False] * len(p_values)
    for rank, index in enumerate(order):
        if p_values[index] > alpha / (len(p_values) - rank):
            break
        reject[index] = True
    return reject


def _z(probability):
    return _NORMAL.inv_cdf(probability)


def _pairs_needed(psi, delta, alpha, power):
    """Connor (1987) required number of pairs, before rounding up."""
    if not 0 < abs(delta) < sqrt(psi) <= 1:
        raise ValueError("need 0 < |delta| < sqrt(psi) <= 1")
    numerator = _z(1 - alpha / 2) * sqrt(psi) + _z(power) * sqrt(psi - delta ** 2)
    return (numerator / delta) ** 2


def required_pairs(psi, delta, alpha=0.05, power=0.80):
    """Pairs needed to detect an accuracy difference delta at discordance psi."""
    return ceil(_pairs_needed(psi, delta, alpha, power))


def minimum_detectable_effect(n, psi, alpha=0.05, power=0.80):
    """Smallest |delta| detectable with n pairs (bisection on Connor's formula)."""
    low, high = 1e-9, sqrt(psi) - 1e-9
    for _ in range(100):
        middle = (low + high) / 2
        if _pairs_needed(psi, middle, alpha, power) > n:
            low = middle
        else:
            high = middle
    return high


def paired_bootstrap_ci(differences, n_resamples=10_000, confidence=0.95, seed=2026):
    """Mean of paired differences with a seeded percentile bootstrap CI."""
    values = np.asarray(differences, dtype=float)
    if values.size == 0:
        raise ValueError("no differences to resample")
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    for start in range(0, n_resamples, 1000):
        stop = min(start + 1000, n_resamples)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    return float(values.mean()), float(low), float(high)


def tost_paired(differences, margin, alpha=0.05):
    """Equivalence of paired differences within +/- margin (normal approximation)."""
    values = np.asarray(differences, dtype=float)
    if values.size < 2:
        raise ValueError("TOST needs at least two paired differences")
    mean = float(values.mean())
    half_width = _z(1 - alpha) * float(values.std(ddof=1)) / sqrt(values.size)
    low, high = mean - half_width, mean + half_width
    return {"mean": mean, "ci_low": low, "ci_high": high,
            "equivalent": -margin < low and high < margin}


def _signed_ranks(nonzero):
    magnitudes = np.abs(nonzero)
    order = magnitudes.argsort(kind="mergesort")
    ranks = np.empty(nonzero.size)
    ranks[order] = np.arange(1, nonzero.size + 1)
    _, inverse, counts = np.unique(magnitudes, return_inverse=True, return_counts=True)
    for group in np.flatnonzero(counts > 1):
        ranks[inverse == group] = ranks[inverse == group].mean()
    return ranks, counts


def wilcoxon_signed_rank(differences):
    """Two-sided Wilcoxon signed-rank test (normal approximation, ties corrected)."""
    values = np.asarray(differences, dtype=float)
    nonzero = values[values != 0]
    n = nonzero.size
    if n == 0:
        return {"n": 0, "w_plus": 0.0, "z": 0.0, "p": 1.0}
    ranks, counts = _signed_ranks(nonzero)
    w_plus = float(ranks[nonzero > 0].sum())
    mean = n * (n + 1) / 4
    variance = n * (n + 1) * (2 * n + 1) / 24 - float((counts ** 3 - counts).sum()) / 48
    z = (w_plus - mean) / sqrt(variance) if variance > 0 else 0.0
    p = min(1.0, 2 * (1 - _NORMAL.cdf(abs(z))))
    return {"n": int(n), "w_plus": w_plus, "z": z, "p": p}
