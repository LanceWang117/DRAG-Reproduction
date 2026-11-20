---
id: J09
title: Retrieval acceleration configuration
status: todo
env: cpu
size: S
depends: [J03]
---
# J09 Retrieval acceleration configuration

**Goal.** Make retrieval acceleration a validated, testable variant setting:
- `faiss_gpu`, `gpu_memory_utilization` and `omp_threads` flow from the variant JSON
  into FlashRAG's config and the process environment, in the right order;
- a preflight memory check refuses configurations that cannot fit;
- a documented, reproducible procedure installs FAISS GPU on the Pod.

**Why.** B5: the CPU exact scan reads about 64.5 GB per query. FlashRAG's GPU path
clones an fp16 index (~32 GB), which should fit next to vLLM if vLLM's reservation is
lowered from its default of 0.85. E01 runs the measurement; this job makes it safe and
reproducible.

## Research questions (save to `docs/research/J09-retrieval-acceleration-config.md`)
1. How to install FAISS GPU for CUDA 12 alongside `torch==2.5.1+cu124` in a Conda
   environment: conda-forge or pytorch channel, the `faiss-gpu` package versions, and
   known conflicts. Give exact commands and version pins. Mark anything untested **[E]**.
2. At the pinned FlashRAG version: when does `index_cpu_to_all_gpus` run relative to
   generator construction? Does `gpu_memory_utilization` come from config?
3. `OMP_NUM_THREADS` and `MKL_NUM_THREADS` must be set before `faiss` and `torch` are
   imported. Confirm this for FAISS.

## Scope
- **In:**
  - flag plumbing;
  - environment ordering (threads set before any heavy import; retriever built
    **before** the generator when `faiss_gpu` is true, so the index is placed before
    vLLM reserves its pool);
  - `scripts/preflight_gpu.py` (reads GPU total memory through `nvidia-smi` when
    present; pure computation is unit-tested);
  - a Pod setup note in `REPRODUCTION.md`.
- **Out:** running on a GPU (E01).

## Files
- **Modify:** `improvements/drag_plus/runner/backends.py`, `flags.py`, `REPRODUCTION.md`
  (append a "GPU retrieval" section), architecture.md §7.
- **Create:**
  - `scripts/preflight_gpu.py`
  - `improvements/tests/test_retrieval_accel_config.py`
  - `improvements/configs/s2-gpu-faiss.json`

## Design notes and edge cases
- **Memory model** (the inputs come from the manifest or the variant):
  - required = `gpu_memory_utilization × total` + `index_bytes / 2` (fp16) + E5 (~0.5 GB)
    + a 3 GB margin;
  - fail with a clear message if required exceeds total.
- **Error message.** If `faiss_gpu` is true but FAISS has no GPU support, fail early
  with the install pointer. Never fall back to CPU silently.
- **Thread-only variants.** `omp_threads` must also work with `faiss_gpu` false.

## Acceptance
- [ ] Tests prove that thread variables are set before the backend imports heavy
  modules, and that the build order puts the retriever first when `faiss_gpu` is on
  (fake backend records the order).
- [ ] Preflight-check tests: 0.45 on an 80 GB card with a 64.5 GB fp32 index passes
  (fp16 ≈ 32 GB); 0.85 fails; correct error messages.
- [ ] The REPRODUCTION.md section gives exact install commands, each marked tested or
  [E].
- [ ] findings.md B5: "Addressed by J09 (measurement: E01)".

## Status log
- PR:
- First pushed SHA:
- Evidence:
- Deviations:
- Follow-ups:
