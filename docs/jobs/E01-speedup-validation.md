---
id: E01
title: Validate answer-preserving speedups
status: todo
env: gpu
size: M
depends: [E00, J06, J09]
budget: 4 h wall clock on one A100 80 GB
---
# E01 Validate answer-preserving speedups

**Goal.** Measure the latency gain of the answer-preserving speed changes on the dev
subset. Show that their answers stay within the E00 A/A floor. Decide which ones go
forward (protocol.md §7, stage 1).

**Arms** (all paired, on E00's 60-question stratified dev subset, with interleaved run
order):

| Arm | Configuration |
| --- | --- |
| A | Baseline (`baseline.json`) |
| B | `moderator_final_only` |
| C | B + `batch_agents_in_round` |
| D | C + `faiss_gpu` with `gpu_memory_utilization` 0.45 |
| E | C + the best CPU thread count from E00 (fallback if D fails) |

## Research questions (save to `docs/research/E01-speedup-validation.md`)
1. TOST equivalence for paired binary outcomes at a ±0.5 pp margin: the procedure and
   the sample-size caveat at n = 60. Equivalence claims at this n are weak; say so.
2. Wilcoxon signed-rank for paired latency, and its effect-size reporting.

## Steps
1. Preflight (RULES.md §10). Run `scripts/preflight_gpu.py` for arm D.
2. Run arms A–E.
3. **Retrieval-only overlap.** For arm D, compare top-3 IDs against exact CPU fp32 on
   all 200 dev questions' original queries (target ≥ 99%).
4. **Analyse:**
   - mismatches against the A/A floor;
   - TOST on EM;
   - median and p90 latency with bootstrap CIs, and Wilcoxon;
   - calls and sequential steps per question;
   - peak VRAM.
5. Write `experiments/E01/`, with a latency CDF figure and a per-arm bar chart.
6. **Ledger:**
   - baseline.md: the measured speedups;
   - findings.md B3, B4 and B5: measured;
   - jobs.md: if D fails, note it for E03 and E04.

## Acceptance
- [ ] Each arm's mismatches are reported against the A/A floor, with a pass or fail.
- [ ] Paired latency: median and p90 differences with CIs, and the Wilcoxon p.
- [ ] Arm D: overlap ≥ 99% and no OOM, or a documented failure with VRAM numbers.
- [ ] "Promoted configuration" stated in summary.json; later E-jobs use it as their
  base.

## Status log
- PR:
- First pushed SHA:
- Evidence:
- Deviations:
- Follow-ups:
