---
id: J20
title: Cost-aware router
status: todo
env: cpu
size: M
depends: [J01, J03, J12]
---
# J20 Cost-aware router

**Goal.** Add `router`, a fitted model that sends each question to one route in
{`naive`, `refine_only`, `full_drag`} (plus `decompose` once J22 exists). It uses
**pre-debate** features only:
- the retrieval score margin;
- `s̄_O`;
- whether the model's own answer agrees with the retrieval-based answer;
- the entropy of the Naive answer.

The rule is `argmax_r [P̂(correct | r, q) − λ·cost(r)]`. Add fitting and Pareto-evaluation
scripts, compared against random routing at an equal DRAG share.

**Why.** B16: 124 of 200 pools never changed, yet each paid for a debate round. The
unchanged pools tied 60–60 on EM. The router is the only lever that can bring DRAG near
Naive's latency.

## Research questions (save to `docs/research/J20-cost-aware-router.md`)
1. Adaptive-RAG routing (Jeong et al., 2024; currently unverified: verify it) and
   uncertainty-based adaptive retrieval (Moskvoretskii et al., 2025): which signals
   work, and where routers fail under distribution shift.
2. How to evaluate a cost-aware router: Pareto frontiers, a random-routing baseline at
   equal cost, and leave-one-dataset-out (LODO) validation.

## Scope
- **In:**
  - `gates/router_features.py` (pre-debate features only);
  - `gates/router.py` (load and decide);
  - routing in `pipeline.py`;
  - `scripts/fit_router.py` (per-route correctness models from logged multi-arm runs,
    LODO);
  - `scripts/eval_router.py` (Pareto curve with Matplotlib, random-routing baseline);
  - tests.
- **Out:** real fitting (E05).

## Files
- **Create:**
  - `improvements/drag_plus/gates/{router_features,router}.py`
  - `scripts/fit_router.py`
  - `scripts/eval_router.py`
  - `improvements/tests/test_router.py`
  - `improvements/tests/test_router_eval.py`
- **Modify:** `pipeline.py`, `flags.py`, architecture.md §7, findings.md B16.

## Design notes and edge cases
- **Leakage guard.** An explicit feature whitelist, and a test proving no feature reads
  gold answers or any post-routing output.
- **Feature cost.** Computing features costs calls; record that cost in the route's cost.
  The Naive answer is computed once and reused if the route is `naive`.
- **Model file.** Records the training split; the loader refuses `test`.
- **Random-routing baseline.** Route the same fraction of questions to each route
  uniformly at random, averaged over 1,000 seeds, with a CI.

## Acceptance
- [ ] On synthetic logged data with a known signal, the router beats random routing on
  the Pareto curve, and the curve is plotted (test, figure written to a temp
  directory).
- [ ] The leakage test passes; the loader refuses test-split models.
- [ ] Routing selects the right path and records the decision and features in the state
  record (test).
- [ ] T0 still passes with the defaults.

## Status log
- PR:
- First pushed SHA:
- Evidence:
- Deviations:
- Follow-ups:
