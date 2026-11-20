---
id: E03
title: Throughput sweep and model-level variants
status: todo
env: gpu
size: M
depends: [E01, J16]
budget: 5 h wall clock
---
# E03 Throughput sweep and model-level variants

**Goal.** Measure questions/hour and per-question latency for the batched engine,
starting from E01's promoted configuration P, across B ∈ {1, 2, 4, 8, 16, 32} with refill
on. Show that per-question answers stay within the A/A floor. Then, only within the time
budget, evaluate two optional model-level variants: INT4 weights (W4A16) and speculative
decoding.

## Research questions (save to `docs/research/E03-throughput-model-variants.md`)
1. W4A16 for Llama-3.1-8B-Instruct on an A100 with vLLM 0.6.6:
   - Which pre-quantised checkpoints exist? Give the exact Hugging Face IDs and
     licences, or say there is none.
   - Is the Marlin kernel used on Ampere?
   - What accuracy loss is expected (Kurtic et al., 2025)?
2. vLLM 0.6.6 speculative decoding (n-gram prompt lookup or a draft model): the
   configuration keys, and the slowdowns at batch size > 1 (ledger issues #8439,
   #16258).

## Steps
1. Preflight.
2. **Sweep.** B ∈ {1, 2, 4, 8, 16, 32}, refill on, on E00's 60-question subset, 3 repeats
   each. Record questions/hour, p50 and p95 latency, GPU utilisation, KV usage and peak
   VRAM. Check per-question transcripts against B = 1, compared with the A/A floor.
3. **INT4 (optional).** Only if a suitable checkpoint exists and fits the budget. Run on
   all 200 dev questions, paired with P at the best B. Report McNemar (pooled and on
   2Wiki/HotpotQA), latency and throughput.
4. **Speculative decoding (optional).** At B = 1 only: the acceptance rate and the
   latency change; then a check at B = 8 to document the slowdown or speed-up.
5. Write `experiments/E03/`: a throughput-vs-B figure, a latency CDF, and the variant
   tables.
6. **Ledger:**
   - baseline.md: throughput numbers [M];
   - findings.md B4 measured;
   - corrections.md: the earlier draft's "8–15×" claim, confirmed or corrected;
   - the promoted B, and whether INT4 or speculation is adopted.

## Acceptance
- [ ] Throughput and latency per B, with medians and IQRs over 3 repeats.
- [ ] Batching equivalence within the A/A floor, or a documented divergence.
- [ ] Each optional variant either measured with an accuracy test, or explicitly
  skipped with its reason (no checkpoint, budget, or incompatibility).

## Status log
- PR:
- First pushed SHA:
- Evidence:
- Deviations:
- Follow-ups:
