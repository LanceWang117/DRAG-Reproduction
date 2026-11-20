---
id: J03
title: Instrumentation and variant runner
status: todo
env: cpu
size: M
depends: [J02]
---
# J03 Instrumentation and variant runner

**Goal.** Build the variant runner, `scripts/run_variant.py --variant <json>`. It runs
`drag_plus` (DRAG) and the Naive arm with FlashRAG objects built lazily, and writes the
§5 logs: `stage-timings.jsonl`, `transcripts.jsonl` and `run-manifest.json`. It must
also support per-question checkpoints with `--resume`, and a fake backend so the whole
path is tested offline.

**Why.** The F0 profiling work, and every E-job, needs measured per-call timings and
resumable runs. The documented reproduction runner `scripts/run_evaluation.py` must stay
unchanged; it is over the size limit and is the reference.

## Research questions (save to `docs/research/J03-instrumentation-runner.md`)
1. How the existing runner builds FlashRAG `Config`, `Dataset`, the generator,
   retriever and `SequentialPipeline` (`scripts/run_evaluation.py:60-145`). Which
   helpers can be imported rather than copied?
2. Which vLLM 0.6.6 request metrics (`RequestOutput.metrics`) are exposed through
   FlashRAG's generator, if any. Otherwise wall-clock timing around `generate` is
   enough; mark the rest **[IC]**.
3. How to count tokens with the Llama-3.1 tokenizer through `transformers`, lazily, and
   how to fall back to a whitespace counter in tests.

## Scope
- **In:**
  - `instrument/timing.py` (wrappers adding timing rows to the state record);
  - `instrument/logs.py` (JSONL writers with fsync, as the existing runner does);
  - `instrument/manifest.py`;
  - `runner/` (load variant, build backends lazily, run both arms, checkpoint and
    resume, evaluate with FlashRAG's evaluator);
  - `scripts/run_variant.py` (thin CLI);
  - `improvements/configs/baseline.json`;
  - tests with the fake backend.
- **Out:** the retrieval cache (J04); any pipeline behaviour change.

## Files
- **Create:**
  - `improvements/drag_plus/instrument/{__init__,timing,logs,manifest}.py`
  - `improvements/drag_plus/runner/{__init__,backends,run,resume}.py`
  - `scripts/run_variant.py`
  - `improvements/configs/baseline.json`
  - `improvements/tests/test_instrument.py`
  - `improvements/tests/test_runner_fake.py`
- **Modify:** `AGENTS.md` (Commands), `REPRODUCTION.md` (add a "Variant runner"
  section; do not change the existing instructions).

## Design notes and edge cases
- **Imports.** FlashRAG and vLLM are imported only inside the backend builder that runs
  them. `--backend fake` uses `improvements/tests/fakes.py`-style scripted adapters (or
  a fake backend module under `drag_plus/runner/`).
- **Resume.** Resuming skips question keys already in the checkpoint file and appends
  to the same run directory. Its manifest records each resume. A crash leaves no
  partial line: write each line whole, then fsync.
- **Timing rows** follow the architecture.md §5 schema exactly. A missing field is a
  test failure.
- **Profiling.** `--profile` sets the `profile_calls` flag. When it is off,
  instrumentation overhead must be negligible.
- **Run order.** Arms run in the order given; the CLI accepts `--interleave` to
  alternate questions between arms (protocol.md §2).
- **Outputs.** Go to `new-runs/<UTC>-<variant>/`, which git ignores. The runner never
  writes into `results/` or `experiments/`.

## Acceptance
- [ ] With the fake backend, an end-to-end run over 5 questions × 2 arms writes all
  three logs with the full schema (test).
- [ ] Killing the run mid-way and then running with `--resume` gives the same final logs
  as an uninterrupted run (test simulates the crash).
- [ ] T0 still passes. `scripts/run_evaluation.py` is unchanged, and
  `test_submission.py` passes.
- [ ] `run_variant.py --validate-only` checks inputs without importing heavy libraries
  (test).

## Status log
- PR:
- First pushed SHA:
- Evidence:
- Deviations:
- Follow-ups:
