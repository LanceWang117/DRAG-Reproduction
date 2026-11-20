---
id: J14
title: Calibrated escalation gate and lazy Opponent
status: todo
env: cpu
size: M
depends: [J08, J12, J01]
---
# J14 Calibrated escalation gate and lazy Opponent

**Goal.** Add three flags:
- `escalation_gate = "risk"`: after response round 1, compute label-free features and a
  calibrated risk `r̂ = P(wrong | features)`. Stop if `r̂ ≤ escalation_tau`; otherwise run
  another round, capped at 3.
- `lazy_query_opponent`: score query-stage sufficiency with J12's log-prob scorer in a
  single prefill pass, and invoke the Opponent only when evidence looks insufficient.
- `escalation_tau`: the stopping threshold.

Also provide a fit and evaluation script that produces the gate model from logged
transcripts.

**Why.** B6 and B16. Agreement-based stopping is unsafe: it commits confident errors at
18–47% miscoverage (Dai, 2026). So the gate must use calibrated, partly exogenous
features, never raw agreement alone.

## Research questions (save to `docs/research/J14-escalation-gate-lazy-opponent.md`)
1. Calibrated stopping and selective prediction for multi-agent debate. Ledger: Dai
   2026; arXiv 2510.12697; 2605.19193; Moskvoretskii et al. 2025. Which features work?
2. Logistic regression with calibration (Platt or isotonic) on small data: risks and
   cross-validation.
3. Semantic-entropy approximations when there are only 2–3 candidate answers.

## Scope
- **In:**
  - `gates/features.py`: disagreement under `E(·)`; answer-cluster entropy; `s̄_O`; an
    evidence-support stub (answer-string containment in passages), with a hook for J19;
  - `gates/escalation.py`: load model JSON, decide;
  - lazy-Opponent logic in the query stage;
  - `scripts/fit_gate.py`: fit from transcripts, cross-validate, write model JSON and a
    reliability figure;
  - tests.
- **Out:** fitting on real data. That happens in E05 (train split) or on E02a dev
  transcripts. Never on the test set.

## Files
- **Create:**
  - `improvements/drag_plus/gates/{__init__,features,escalation}.py`
  - `scripts/fit_gate.py`
  - `improvements/tests/test_gate.py`
  - `improvements/tests/test_fit_gate.py`
- **Modify:** `answer_stage.py`, `query_stage.py`, `flags.py`, architecture.md §7,
  findings.md B6 and B16.

## Design notes and edge cases
- **Model format.** The gate model is small JSON: feature names, coefficients,
  intercept, calibration map, and the training split and manifest hash.
  - The loader refuses a model whose recorded training split is `test`.
  - It refuses a model whose features don't match the code.
- **Missing features.** Features must be computable at decision time with no gold
  labels. Add a leakage test that proves features ignore gold answers.
- **Missing model.** With `escalation_gate = "risk"` and no model file, fail loudly; do
  not default silently.
- **Lazy Opponent.** It records the sufficiency score and decision. If the score is
  unavailable (fallback), run the normal Opponent.
- **Call accounting.** Expected calls: `3·Q_r + 2(1 + p_esc) + 1`. Tests check the
  accounting.

## Acceptance
- [ ] On synthetic data with a known signal, the fit script recovers a useful model
  (AUROC above a stated threshold) and writes the reliability figure (test).
- [ ] Gate decisions honour τ; the call counts match the formula (tests).
- [ ] The leakage test passes; the loader refuses test-split models (tests).
- [ ] With the lazy Opponent on and sufficiency above τ, there are 0 Opponent calls
  (test).
- [ ] T0 still passes with the defaults.

## Status log
- PR:
- First pushed SHA:
- Evidence:
- Deviations:
- Follow-ups:
