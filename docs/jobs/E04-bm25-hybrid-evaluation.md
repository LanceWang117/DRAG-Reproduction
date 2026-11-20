---
id: E04
title: BM25 index build and hybrid retrieval evaluation
status: todo
env: gpu-pod (CPU and RAM heavy)
size: L
depends: [E00, J17]
budget: 8 h wall clock; may need the Pod's full RAM and 100+ GB free disk
---
# E04 BM25 index build and hybrid retrieval evaluation

**Goal.**
1. Build a BM25 index over `wiki18_100w` (21,015,324 passages).
2. Evaluate retrieval alone on the 200 dev questions for E5, BM25, RRF, and RRF + rerank
   with K ∈ {10, 20, 30}.
3. If retrieval improves, run the 2 × 2 {E5, best hybrid} × {Naive, P} end to end on
   dev.

## Research questions (save to `docs/research/E04-bm25-hybrid-evaluation.md`)
1. `bm25s` large-corpus indexing: memory, time and memory-mapped loading; tokenisation
   and stemming choices (PyStemmer is installed).
2. Whether a prebuilt BM25 index for `wiki18_100w` exists in FlashRAG's resources. If it
   does, prefer it and record its checksum.

## Steps
1. Preflight: disk, RAM, the corpus checksum against `DATA_DOWNLOADS.md`.
2. Build or download the index in the background, writing to an ignored folder (for
   example `wiki_corpus/bm25/`). Record the build time, size and checksum. It is
   resumable: if the build tool cannot resume, document how long a restart takes.
3. **Retrieval-only evaluation** (`scripts/eval_retrieval.py`): Containment@3 and @5,
   MRR, nDCG@5, unique-title ratio, and p50/p99 latency per configuration, per dataset.
4. **End to end.** Only if the best hybrid beats E5 on Containment@5 by a margin you
   state in advance (write it in the README before running): run the 2 × 2 on dev with
   the interaction analysis.
5. Write `experiments/E04/`. **Ledger:** B17 measured; hybrid promoted or rejected.

## Acceptance
- [ ] Index build time, size and checksum recorded; the index itself is not committed.
- [ ] A retrieval-only table per configuration and dataset, with a recall@k figure.
- [ ] Either the 2 × 2 result with its interaction term, or a documented reason it was
  not run (the margin was not met).

## Status log
- PR:
- First pushed SHA:
- Evidence:
- Deviations:
- Follow-ups:
