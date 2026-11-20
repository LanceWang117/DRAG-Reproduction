---
id: J05
title: Test-set draw and pre-registration
status: todo
env: cpu + network
size: S
depends: [J01]
---
# J05 Test-set draw and pre-registration

**Goal.** Draw a seeded, stratified test set from the FlashRAG benchmark splits,
disjoint from the 200 dev questions. Commit it with checksums and a pre-registration
document **before any model ever sees it**.

**Why.** B1. The 200 dev questions are post-pilot and underpowered. Claims need about
590 fresh paired questions to detect 5 pp (protocol.md §4).

## Research questions (save to `docs/research/J05-testset-preregistration.md`)
1. Which splits FlashRAG publishes per dataset on Hugging Face (`RUC-NLPIR/FlashRAG_datasets`,
   revision `bcafb8dd07d453be3cbeeeb3f78be1841bddf92c` for five datasets, per
   `DATA_DOWNLOADS.md`), the row counts, and the license notes. Also: which revision NQ
   should use (the original was "then-current main"; pick and record a revision).
2. Whether StrategyQA has anything other than train. If not, the test rows must come
   from train IDs disjoint from dev.
3. Pre-registration good practice for ML evaluation: what to fix in advance (primary
   endpoint, analysis plan, stopping rules).

## Scope
- **In:** a sampling script; `experiments/J05-testset/questions-test.jsonl`;
  `PREREGISTRATION.md`; `SHA256SUMS`; a README.
- **Out:** running any model; touching `results/`.

## Files
- **Create:**
  - `scripts/draw_testset.py`
  - `improvements/tests/test_draw_testset.py`
  - `experiments/J05-testset/{README.md,PREREGISTRATION.md,questions-test.jsonl,SHA256SUMS}`

## Design notes and edge cases
- **Allocation.** Default 100 questions per dataset (600 total, ≈ 5 pp power pooled).
  If a split has fewer eligible rows, take all of them and record the shortfall.
- **Disjointness.**
  - Exclude dev keys by `(dataset, split, id)`.
  - **Also** exclude rows whose normalised question text equals any dev question.
    Duplicates across splits exist.
  - Exclude rows without gold answers.
- **Seed.** Commit the seed and the exact source revision. Re-running the script must
  reproduce the file byte for byte.
- **Size.** Keep the committed file ≤ 1 MB (question, gold answers and identity only;
  no passages). Download the raw splits to the ignored `dataset/` folder.
- **PREREGISTRATION.md must fix:**
  - the primary endpoint: pooled EM, exact McNemar, α = 0.05;
  - the secondary endpoints: per-dataset with Holm, F1 bootstrap, latency;
  - the rule that configurations to test are frozen in `experiments/E06/FROZEN.md`
    before the run;
  - "one run per frozen configuration";
  - no post-hoc exclusions;
  - how null results are reported.
- **Do not print** test questions into logs or PR text beyond counts.

## Acceptance
- [ ] Disjointness test: 0 key overlaps and 0 normalised-text overlaps with dev.
- [ ] The determinism test re-draws identical bytes.
- [ ] Per-dataset counts are recorded in the README and the status log.
- [ ] PREREGISTRATION.md and SHA256SUMS are committed, and the status log records the
  commit SHA as the pre-registration timestamp.

## Status log
- PR:
- First pushed SHA:
- Evidence:
- Deviations:
- Follow-ups:
