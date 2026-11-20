"""Rebuild the recorded baseline comparison from results/paired-200.json.

Every number is recomputed from the stored per-question answers with the FlashRAG-exact
scorer and the paired statistics in this package; nothing is copied from summaries.
"""
import json

from .score import score_answer
from .stats import (discordant_counts, holm, mcnemar_p, minimum_detectable_effect,
                    paired_bootstrap_ci, required_pairs)

DATASET_ORDER = ("NQ", "TriviaQA", "PopQA", "2wiki", "HotpotQA", "StrategyQA")
SEED = 2026


def load_pairs(path):
    """Paired records (one per question) from a paired-results JSON file."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["pairs"]


def _scores(pairs, method):
    return [score_answer(p[f"{method}_answer"], p["golden_answers"]) for p in pairs]


def compare(pairs):
    """Accuracy, discordance and McNemar p for one group of paired questions."""
    naive, drag = _scores(pairs, "naive"), _scores(pairs, "drag")
    only_drag, only_naive = discordant_counts([s["em"] for s in drag], [s["em"] for s in naive])
    n = len(pairs)
    return {
        "n": n,
        "naive_em": 100 * sum(s["em"] for s in naive) / n,
        "naive_f1": 100 * sum(s["f1"] for s in naive) / n,
        "drag_em": 100 * sum(s["em"] for s in drag) / n,
        "drag_f1": 100 * sum(s["f1"] for s in drag) / n,
        "only_drag": only_drag,
        "only_naive": only_naive,
        "p": mcnemar_p(only_drag, only_naive),
        "p_midp": mcnemar_p(only_drag, only_naive, midp=True),
    }


def by_dataset(pairs):
    """Per-dataset comparison rows in the order used by the ledger."""
    rows = []
    for name in DATASET_ORDER:
        group = [p for p in pairs if p["dataset"] == name]
        if group:
            rows.append({"dataset": name, "split": group[0]["split"], **compare(group)})
    return rows


def format_only_counts(pairs):
    """EM wins whose loser's answer already contained the gold string (acc = 1)."""
    counts = {"drag_wins": 0, "drag_wins_format_only": 0,
              "naive_wins": 0, "naive_wins_format_only": 0}
    for pair in pairs:
        naive = score_answer(pair["naive_answer"], pair["golden_answers"])
        drag = score_answer(pair["drag_answer"], pair["golden_answers"])
        if drag["em"] > naive["em"]:
            counts["drag_wins"] += 1
            counts["drag_wins_format_only"] += int(naive["acc"] == 1)
        elif naive["em"] > drag["em"]:
            counts["naive_wins"] += 1
            counts["naive_wins_format_only"] += int(drag["acc"] == 1)
    return counts


def pooled_summary(pairs):
    """Pooled comparison, F1 bootstrap CI, Holm decisions and power figures."""
    pooled = compare(pairs)
    f1_diffs = [d["f1"] - n["f1"] for d, n in zip(_scores(pairs, "drag"), _scores(pairs, "naive"))]
    mean, low, high = paired_bootstrap_ci(f1_diffs, seed=SEED)
    psi = (pooled["only_drag"] + pooled["only_naive"]) / pooled["n"]
    rows = by_dataset(pairs)
    return {
        **pooled,
        "delta_f1": 100 * mean, "delta_f1_ci": (100 * low, 100 * high),
        "holm_rejects": [r["dataset"] for r, rej in zip(rows, holm([r["p"] for r in rows])) if rej],
        "discordance": psi,
        "mde_pp": 100 * minimum_detectable_effect(pooled["n"], psi),
        "pairs_for_5pp": required_pairs(psi, 0.05),
        "pairs_for_3pp": required_pairs(psi, 0.03),
    }


def _row_line(row):
    return (f"| {row.get('dataset', '**Pooled**')} | {row['n']} "
            f"| {row['naive_em']:.2f} / {row['naive_f1']:.2f} "
            f"| {row['drag_em']:.2f} / {row['drag_f1']:.2f} "
            f"| {row['drag_em'] - row['naive_em']:+.2f} "
            f"| {row['only_drag']} / {row['only_naive']} | {row['p']:.4f} |")


def render(pairs):
    """Markdown report: the accuracy table, then the pooled statistics."""
    summary, counts = pooled_summary(pairs), format_only_counts(pairs)
    lines = ["| Dataset | n | Naive EM / F1 | DRAG EM / F1 | dEM | DRAG-only / Naive-only | McNemar p |",
             "| --- | ---: | --- | --- | ---: | ---: | ---: |"]
    lines += [_row_line(row) for row in by_dataset(pairs)] + [_row_line(summary)]
    low, high = summary["delta_f1_ci"]
    lines += ["",
              f"Pooled McNemar p = {summary['p']:.4f} (mid-p {summary['p_midp']:.4f}); "
              f"Holm rejects: {', '.join(summary['holm_rejects']) or 'none'}.",
              f"dF1 = {summary['delta_f1']:+.2f} pp, bootstrap 95% CI [{low:+.2f}, {high:+.2f}] "
              f"(seed {SEED}).",
              f"Format-only: {counts['drag_wins_format_only']} of {counts['drag_wins']} DRAG wins, "
              f"{counts['naive_wins_format_only']} of {counts['naive_wins']} Naive wins.",
              f"Discordance {summary['discordance']:.2f}; MDE at n = {summary['n']}: "
              f"{summary['mde_pp']:.1f} pp; pairs for 5 pp: {summary['pairs_for_5pp']}, "
              f"for 3 pp: {summary['pairs_for_3pp']}."]
    return "\n".join(lines)
