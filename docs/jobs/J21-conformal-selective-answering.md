---
id: J21
title: Conformal selective answering
status: todo
env: cpu
size: M
depends: [J01, J20]
---
# J21 Conformal selective answering

**Goal.** Add `conformal_alpha`: split-conformal calibration, done per dataset.
- The nonconformity score is `s(x, y) = 1 − p̂(y | x)`, where `p̂` comes from logged
  agent votes, the Judge margin and evidence support.
- At answer time, build the conformal prediction set: answer if it has one element,
  otherwise abstain (or escalate, if configured).

Add a coverage-evaluation script reporting empirical coverage with Clopper–Pearson
intervals, set size, abstention rate, and selective accuracy at 50/70/90% coverage.

**Why.** Hallucination control: the system currently always answers. This adds
calibrated abstention. Agreement-based confidence is manufactured by the debate (Dai,
2026), so the confidence must come from an exogenous, calibrated certificate.

## Research questions (save to `docs/research/J21-conformal-selective-answering.md`)
1. Split conformal prediction for classification and open-ended QA: answer clustering,
   and finite-sample correction with the ⌈(n+1)(1−α)⌉/n quantile.
2. Conformal-RAG (Feng et al., SIGIR 2025): what transfers to an answer-level setting.
3. Exchangeability across heterogeneous datasets: per-dataset versus pooled calibration.

## Scope
- **In:**
  - `gates/conformal.py` (calibrate, predict set, decide);
  - integration as a final step in `pipeline.py`;
  - `scripts/eval_conformal.py` (5-fold cross-fitting, Clopper–Pearson, Matplotlib
    risk–coverage curve);
  - tests.
- **Out:** reporting raw EM as if abstentions were answers. The evaluation must report
  both full-coverage EM and selective metrics.

## Files
- **Create:**
  - `improvements/drag_plus/gates/conformal.py`
  - `scripts/eval_conformal.py`
  - `improvements/tests/test_conformal.py`
- **Modify:** `pipeline.py`, `flags.py`, architecture.md §7.

## Design notes and edge cases
- **Small calibration sets.** When n is small, the quantile can equal 1, which means
  abstaining always. Report that honestly.
- **Per-dataset calibration.** A dataset with fewer than the minimum n falls back to
  pooled calibration, with a logged warning.
- **Answer clusters.** Clustering uses `E(·)` plus J01 normalisation. Yes/no datasets
  have exactly 2 labels.
- **Calibration file.** Records its split; the loader refuses `test`.

## Acceptance
- [ ] On synthetic exchangeable data, empirical coverage is within the Clopper–Pearson
  interval of nominal for α ∈ {0.1, 0.2}, over 200 seeds (test).
- [ ] The small-n behaviour and the pooled fallback are tested.
- [ ] Abstentions are marked in the state record and excluded from selective accuracy
  only (test).
- [ ] T0 still passes with the defaults.

## Status log
- PR:
- First pushed SHA:
- Evidence:
- Deviations:
- Follow-ups:
