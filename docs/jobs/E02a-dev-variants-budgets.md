---
id: E02a
title: Dev variants - novelty exit and budgets
status: todo
env: gpu
size: M
depends: [E01, J07, J08]
budget: 5 h wall clock
---
# E02a Dev variants: novelty exit and budgets

**Goal.** On all 200 dev questions, measure accuracy and latency for the stage-2 budget
variants against the promoted E01 configuration. Decide promotion under protocol.md §7.

**Arms** (paired; the base is E01's promoted configuration "P"):

| Arm | Configuration |
| --- | --- |
| A | P |
| B | P + `skip_duplicate_query` + `zero_novelty_exit` |
| C | P + `response_rounds = 1` |
| D | P + `response_rounds = 1` + `agent_max_tokens = 128` + `moderator_max_tokens = 32` + `concise_agents` |

Optional arms, only if the time budget allows: `response_rounds = 2`;
`agent_max_tokens = 128` alone.

## Research questions (save to `docs/research/E02a-dev-variants-budgets.md`)
1. How to report a non-inferiority result honestly when n = 200 is underpowered.
   State the margin, the CI and the conclusion.
2. Per-dataset heterogeneity reporting without multiplicity abuse: Holm; descriptive
   versus confirmatory.

## Steps
1. Preflight. Confirm P's variant file from `experiments/E01/summary.json`.
2. Run arms with `--interleave` and checkpointing. Resume if interrupted.
3. **Analyse (J01 toolkit):**
   - pooled EM McNemar and TOST (1 pp margin) against A;
   - per-dataset results, descriptive with Holm;
   - F1 bootstrap;
   - calls, tokens, latency (median, p90), and `finish_reason = length` rate;
   - B's exit rate and its EM on the 76 changed-pool questions.
4. **Log the gate features for J14.** Keep per-question round-1 agent answers, the
   disagreement and any Judge scores in the transcripts. This is used for calibration
   later.
5. Write `experiments/E02a/` with figures (EM vs seconds per arm; per-dataset ΔEM with
   CIs).
6. **Ledger:** findings B6 and B15 measured, and promoted arms recorded in summary.json.

## Acceptance
- [ ] Every arm has pooled EM, a TOST verdict, latency and call metrics.
- [ ] A promotion decision per arm, with its reason. 2Wiki gets explicit attention for
  the R = 1 arms (the paper's exception).
- [ ] Transcripts with gate features are archived; the README gives their location and
  checksums.

## Status log
- PR:
- First pushed SHA:
- Evidence:
- Deviations:
- Follow-ups:
