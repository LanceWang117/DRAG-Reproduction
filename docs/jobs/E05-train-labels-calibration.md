---
id: E05
title: Train-split labels and calibration
status: todo
env: gpu
size: L
depends: [E03, J14, J20]
budget: 8 h wall clock with the batched engine
---
# E05 Train-split labels and calibration

**Goal.** Generate labelled, logged runs on benchmark **train** splits, disjoint from dev
and test. Run each question through every route arm: Naive, refine-only, the promoted
DRAG configuration, and (if J22 is merged) decompose. Fit the J14 escalation gate and
the J20 router on these runs; evaluate them on dev; commit the fitted model JSON files.

**Why.** Only 24/200 dev questions gained from DRAG, which is far too few to fit
anything (B16). Fitting on dev and then reporting on dev would be circular.

## Research questions (save to `docs/research/E05-train-labels-calibration.md`)
1. Sample size for calibrating a binary risk model: events per variable, and what ECE
   resolution n gives.
2. How to sample train questions so the label mix is useful without biasing calibration
   (stratify by dataset; do not oversample on outcomes).

## Steps
1. **Preflight.** Run `scripts/draw_testset.py` in train mode (extend it only if J05's
   design allows; otherwise write a small new sampler with tests) to draw about 1,500
   train questions, stratified by dataset. Exclude dev and test keys and texts, and prove
   it.
2. Run all arms with the batched engine (E03's promoted B), checkpointed and resumable.
3. Fit the gate (`scripts/fit_gate.py`) and the router (`scripts/fit_router.py`) with
   LODO.
4. Evaluate on dev: router Pareto curve against random routing; gate stop rate,
   wrong-stop rate and ECE.
5. Commit the model JSON files to `improvements/configs/models/` (small) and the evidence
   to `experiments/E05/`.
6. **Ledger:** B16 measured; gate and router promoted or rejected.

## Acceptance
- [ ] Disjointness proof (keys and texts) for the train sample.
- [ ] Model JSON files record their training split and manifest hash.
- [ ] Dev evaluation: Pareto figure with the random-routing comparison and CI; gate
  reliability diagram.
- [ ] A promotion decision for each model.

## Status log
- PR:
- First pushed SHA:
- Evidence:
- Deviations:
- Follow-ups:
