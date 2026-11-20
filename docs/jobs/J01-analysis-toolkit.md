---
id: J01
title: Analysis toolkit - extractor and statistics
status: todo
env: cpu
size: S
depends: [J00]
---
# J01 Analysis toolkit: extractor and statistics

**Goal.** Build `improvements/drag_plus/analysis/`. It must:
- re-score answers exactly as FlashRAG does;
- provide the unified extractor `E(·)`;
- implement every statistical test in protocol.md §4.

Then reproduce baseline.md's accuracy table from `results/` with a script, and pin the
numbers with regression tests.

**Why.** B1 and B12. Every later claim depends on this code being right.

## Research questions (save to `docs/research/J01-analysis-toolkit.md`)
1. FlashRAG's EM, F1 and acc normalisation (`flashrag/evaluator/metrics.py` at the
   pinned version if possible, otherwise `main`; say which). Reproduce it exactly.
2. Exact two-sided McNemar: the binomial on discordant pairs, with p capped at 1. Mid-p
   is not needed. How to handle zero discordant pairs.
3. Paired bootstrap for a mean difference: percentile CI, seeded resampling.
4. TOST for paired differences, for both binary and continuous scores.
5. The Connor sample-size formula for McNemar, and the minimum detectable effect.

## Scope
- **In:**
  - `analysis/score.py` (normalisation, EM, F1, acc);
  - `analysis/extract.py` (`E(·)`);
  - `analysis/stats.py` (McNemar, bootstrap, Holm, TOST, Wilcoxon, sample size and MDE);
  - `scripts/report_baseline.py`;
  - tests.
- **Out:** model runs; changes to `results/`.

## Files
- **Create:**
  - `improvements/drag_plus/__init__.py`
  - `improvements/drag_plus/analysis/{__init__,score,extract,stats}.py`
  - `scripts/report_baseline.py`
  - `improvements/tests/test_score.py`
  - `improvements/tests/test_stats.py`
  - `improvements/tests/test_report_baseline.py`
- **Modify:** `requirements-ci.txt`, only if `scipy` is truly needed (prefer numpy plus
  exact combinatorics).

## Design notes and edge cases
- **Unified extractor `E(·)`.** Text after the **last** "The answer is:" or "Final
  answer:" (case-insensitive); otherwise the first non-empty line. Strip trailing
  punctuation and whitespace. It must handle:
  - empty strings;
  - a marker with nothing after it;
  - several markers;
  - StrategyQA yes/no normalisation.
- **Scoring check.** Recomputed EM/F1/acc for all 400 stored answers must equal the
  stored per-question metrics **exactly**. Any mismatch means the normaliser is wrong.
- **Script determinism.** `report_baseline.py` prints the baseline.md accuracy table,
  the format-only counts (5/24 and 3/14), the pooled and per-dataset McNemar p, and the
  MDE at n = 200 and required n for 5 pp and 3 pp. Seed 2026 for any resampling.
- **Data handling.** Do not load the JSONL into memory twice. Keep functions ≤ 40 lines.

## Acceptance
- [ ] Recomputed metrics match all 400 stored per-question metrics (test).
- [ ] Regression tests pin these values:
  - pooled 24/14, p = 0.1433 ± 1e-4;
  - NQ p = 0.0625;
  - 2Wiki p = 0.0386;
  - the Holm decision is "none significant";
  - MDE at n = 200 is 8.6 ± 0.2 pp;
  - n for 5 pp is 590 ± 15;
  - format-only counts 5 and 3.
- [ ] Extractor unit tests cover every edge case above, plus `dev_8509` and
  `dev_7633`-style outputs.
- [ ] Bootstrap results are identical across two runs with the same seed.
- [ ] architecture.md flag table: the `answer_extractor` row is marked implemented.
- [ ] findings.md B12: "Addressed by J01" note.

## Status log
- PR:
- First pushed SHA:
- Evidence:
- Deviations:
- Follow-ups:
