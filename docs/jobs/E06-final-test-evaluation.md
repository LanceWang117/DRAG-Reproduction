---
id: E06
title: Final frozen-configuration test evaluation
status: todo
env: gpu
size: L
depends: [J05]
budget: 10 h wall clock; may be split across resumed sessions (RULES.md §10)
---
# E06 Final frozen-configuration test evaluation

**Goal.** Evaluate the configurations promoted by the stage gates on the pre-registered
test set (J05), **once each**, following `experiments/J05-testset/PREREGISTRATION.md`.
Produce the final tables and Matplotlib figures for the team's report.

**Depends on** J05, plus **every job whose configuration is promoted.** Before starting,
list the promoted configurations and verify that each one's job and E-job are merged.
This job's `depends` cannot be known in advance; that check replaces it.

## Research questions (save to `docs/research/E06-final-test-evaluation.md`)
1. Reporting standards for a pre-registered evaluation: deviations, null results, and
   multiplicity.
2. Nothing else. This job is execution, not design.

## Steps
1. **Freeze first.** Write `experiments/E06/FROZEN.md` listing each configuration (its
   variant file and sha256), its source job and E-job evidence, and the commit. Commit it
   and push it **before** any test-set run. This is the commitment point.
2. **Preflight.** Also verify the test-set checksum against `experiments/J05-testset/SHA256SUMS`.
3. Run the baseline arm and each frozen arm on the full test set: paired, interleaved,
   checkpointed. Re-runs only resume; they never restart, and no arm runs twice.
4. **Analyse exactly as pre-registered:**
   - primary: pooled EM McNemar;
   - secondary: per-dataset with Holm, F1 bootstrap, latency and throughput, the
     abstention metrics if J21 is promoted;
   - report every arm, including the ones that lose.
5. Write `experiments/E06/`: README, summary.json, tables, figures (EM by dataset with
   CIs; EM vs seconds Pareto; latency CDF) and SHA256SUMS.
6. **Ledger.** baseline.md gets a "Final results" section with [M] links;
   corrections.md records any earlier claim the test contradicts.

## Acceptance
- [ ] FROZEN.md was committed before the first test-set run (commit order is the
  evidence).
- [ ] Every frozen arm has been run exactly once (manifests prove it).
- [ ] The analysis matches the pre-registration; any deviation is listed and justified.
- [ ] Figures come from committed scripts.

## Status log
- PR:
- First pushed SHA:
- Evidence:
- Deviations:
- Follow-ups:
