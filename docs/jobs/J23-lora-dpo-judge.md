---
id: J23
title: LoRA-DPO Judge (optional)
status: todo
env: gpu
size: L
depends: [J12, E05]
budget: 8 h wall clock
---
# J23 LoRA-DPO Judge (optional)

**Goal.** Train a LoRA adapter for the query-stage Judge (and optionally the moderator)
with length-normalised preference optimisation on E05's **train-split** logs, and serve
it through `judge_adapter` using vLLM's LoRA support. Evaluate on dev against J12's
log-prob Judge.

**Why.** B10, and slide 20's "efficient fine-tuning" topic. Optional: do this only if
the team decides the fine-tuning theme is worth the compute.

## Research questions (save to `docs/research/J23-lora-dpo-judge.md`)
1. DPO and length-normalised variants (SimPO; currently unverified: verify it) and their
   known failures (length inflation, rewarding fluent hallucination).
2. Libraries available in the pinned environment:
   - `peft` 0.14.0 is installed; is `trl` installed? If not, can it be pinned without
     breaking `torch`, `transformers` and `vllm`?
   - Does FlashRAG's generator support LoRA (`generator_lora_path`) on vLLM 0.6.6?
3. Preference-pair construction for a Judge: what counts as the preferred and rejected
   answer, without using test data.

## Scope
- **In:**
  - `scripts/build_preferences.py` (from E05 logs: preferred = a verdict that led to
    improved gold recall; rejected = the opposite; train split only);
  - `scripts/train_judge_lora.py`;
  - `judge_adapter` plumbing;
  - evaluation on dev;
  - `experiments/J23/`.
- **Out:** committing adapter weights (store them outside Git; record their checksum).

## Design notes and edge cases
- **Splits.** The preference builder refuses anything but train-split logs.
- **Order bias.** Randomise argument order in training pairs.
- **Regularisation and stopping.** KL β ≈ 0.1 (tune on a held-out train fold). Early
  stopping on held-out train data.
- **Out-of-distribution check.** Hold out PopQA and TriviaQA entirely from training.
- **Dependencies.** If adding `trl` would break the pinned environment, end
  BLOCKED-ENV with the conflict details. Do not upgrade core packages.

## Acceptance
- [ ] Preference data statistics, and a proof that it contains only train-split data.
- [ ] Dev evaluation against J12: Judge AUROC against the oracle label, position
  consistency, and EM (McNemar). Out-of-distribution results reported separately.
- [ ] The adapter checksum and storage location are recorded; no weights in Git.

## Status log
- PR:
- First pushed SHA:
- Evidence:
- Deviations:
- Follow-ups:
