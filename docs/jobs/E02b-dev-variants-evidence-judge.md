---
id: E02b
title: Dev variants - evidence and Judge
status: todo
env: gpu
size: M
depends: [E01, J10, J11, J12, J13]
budget: 5 h wall clock
---
# E02b Dev variants: evidence and Judge

**Goal.** On all 200 dev questions, measure the evidence and Judge variants against
E01's promoted configuration P, and decide promotion. Also test whether prefix caching
stays within the A/A floor.

**Arms** (paired):

| Arm | Configuration |
| --- | --- |
| A | P |
| B | P + `evidence_store = anchored` + `evidence_delimiters` |
| C | B + `context_token_budget = 2500` + `prompt_layout = evidence_first` |
| D | P + `judge_scoring = logprob` + `judge_order_balance` |
| E | B + D + `opponent_action_schema` |
| F | C + `enable_prefix_caching`, run on the E00 60-question subset **twice**, to compare its A/A floor with C's |

## Research questions (save to `docs/research/E02b-dev-variants-evidence-judge.md`)
1. How to measure Judge position consistency, and how to report it (Shi et al., 2025).
2. Prefix-cache hit-rate measurement in vLLM 0.6.6 (metrics or logs). Otherwise use
   time to first token as a proxy, marked as a proxy.

## Steps
1. Preflight; confirm P.
2. Run arms A–E, interleaved and checkpointed; then run F.
3. **Analyse:**
   - pooled EM McNemar and TOST against A;
   - gold-in-context recall per arm (needs the passages logged in transcripts);
   - EM on the 53 evidence-loss questions (B7);
   - the invalid-action rate (E);
   - Judge position consistency (D: the share of rounds where the PO and OP verdicts
     agree);
   - calls and latency;
   - for F, mismatches against the A/A floor.
4. Write `experiments/E02b/`.
5. **Ledger:**
   - B7–B10 and B14 measured;
   - promoted arms recorded;
   - prefix caching approved or rejected for later E-jobs.

## Acceptance
- [ ] All arms analysed as above, with figures.
- [ ] A prefix-caching decision, backed by the A/A comparison.
- [ ] Promotion decisions with reasons in summary.json.

## Status log
- PR:
- First pushed SHA:
- Evidence:
- Deviations:
- Follow-ups:
