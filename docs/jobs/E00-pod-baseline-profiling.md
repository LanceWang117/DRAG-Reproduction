---
id: E00
title: Pod baseline rerun, A/A floor and profiling
status: todo
env: gpu
size: M
depends: [J01, J03, J04]
budget: 3 h wall clock on one A100 80 GB
---
# E00 Pod baseline rerun, A/A floor and profiling

**Goal.** On the A100 Pod:
1. Prove the new variant runner works.
2. Re-establish the baseline with the current code.
3. Measure the nondeterminism (A/A) floor.
4. Measure where DRAG's time goes.

Then replace the [E] latency estimates in the ledger with [M] measurements.

**Why.** The portable runner was never run on a GPU (B1). Every latency claim in
baseline.md is an estimate. Every later E-job compares against this rerun, not against
`results/`.

## Research questions (save to `docs/research/E00-pod-baseline-profiling.md`)
1. FAISS CPU `IndexFlatIP` search cost versus thread count and batch size; vendor
   guidance on `omp_set_num_threads`.
2. How to read GPU memory and utilisation during a run (`nvidia-smi --query-gpu`, NVML).
3. Anything the pinned FlashRAG version does differently between `search` and
   `batch_search` (follow up on J04's finding).

## Steps (each a committed, passing step; evidence goes to `experiments/E00/`)
1. **Preflight** (RULES.md §10). Record GPU, driver, CUDA, conda environment and package
   differences from `ENVIRONMENT.md`.
2. **Smoke test.** Run 1 NQ question through both arms with `run_evaluation.py`
   (legacy) **and** with `run_variant.py --variant improvements/configs/baseline.json`.
   Both must finish.
3. **Reproduction check.** Run `run_variant` with the baseline configuration on 60 dev
   questions, stratified at 10 per dataset by the first 10 IDs in manifest order. Compare
   per-question EM and answers with `results/paired-200.json` and report agreement. Some
   disagreement is expected (nondeterminism, B2): quantify it, don't hide it.
4. **A/A floor.** Repeat step 3 for the same 60 questions and arms, then compute EM
   flips, output-hash mismatches and passage mismatches (protocol.md §3).
5. **Profiling.** Run with `--profile` on 30 of those questions (5 per dataset). Produce
   a stacked-bar Matplotlib figure of seconds by stage and role, and a table of median
   tokens and milliseconds per call type.
6. **Retriever micro-benchmark.** 50 dev queries with `OMP_NUM_THREADS` ∈ {12, 16, 24}
   × batch ∈ {1, 8, 32}; time E5 encoding and the FAISS scan separately.
7. **Ledger update.**
   - baseline.md: mark measured lines [M] with links, and add the measured breakdown.
   - findings.md: settle B2 (root cause, if found), B3 (the dead-call cost) and B5 (the
     retrieval cost).
   - corrections.md: record any estimate that turned out wrong.

## Scope
- **In:** runs, analysis scripts (`scripts/analyze_e00.py` and similar, within limits),
  `experiments/E00/`, ledger updates.
- **Out:** any pipeline change. If the runner is broken, fix it in this job **only** if
  the fix is small and covered by tests; otherwise end BLOCKED-TECHNICAL, naming the
  responsible J-job.

## Acceptance
- [ ] `experiments/E00/` contains README, summary.json, the figures, tables and
  SHA256SUMS (protocol.md §6).
- [ ] summary.json contains:
  - reproduction agreement (EM and answer match);
  - the A/A floor (three counts);
  - per-stage seconds (median and p90);
  - the dead-moderator cost;
  - retrieval ms by thread count and batch size;
  - startup seconds.
- [ ] Every [E] in baseline.md "Where the time likely goes" is now either [M] or
  explicitly "still [E]: reason".
- [ ] No GPU process is left running. `config.local.json` is not committed.

## Status log
- PR:
- First pushed SHA:
- Evidence:
- Deviations:
- Follow-ups:
