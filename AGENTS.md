# DRAG Reproduction: agent guide

This is a capstone reproduction of Debate-Augmented RAG (ACL 2025), plus the team's
improvements. Start here, then follow the links. Keep context small: read only what
your task needs.

## If you were given a job prompt

`docs/prompts/RULES.md` is **binding**. Then read your job file in `docs/jobs/`. The
repository is the only memory: verify state with `git` and `gh`, never with chat history
or notes.

## Map

```
src/original_drag/          FROZEN  authors' code (commit c035ee1)
src/reproduction_changes/   FROZEN  recorded v1.0.1 baseline (Judge fix) + its tests
results/                    FROZEN  recorded 200-question evidence and checksums
scripts/                    runner, overlay builder, offline checks (gate tooling from J00)
improvements/               team code: drag_plus package, tests, configs (from J02)
experiments/<JOB-ID>/       compact committed evidence from E-jobs
docs/ledger/                facts, findings, architecture, protocol, job index
docs/jobs/                  one spec + status file per job
docs/prompts/               RULES.md, TEMPLATE.md, one paste-ready prompt per job
docs/research/              per-job research notes with sources
new-runs/, .runtime/, dataset/, models/, wiki_corpus/   ignored local data
```

## Commands

Before J00 has merged, the offline checks are:

```bash
python scripts/verify_evidence.py
python -m unittest discover -s scripts -p test_submission.py -v
python scripts/run_evaluation.py --validate-only
```

Run them under Python 3.11 (see below). After J00, the single local gate is
`bash scripts/gate.sh` (identical to CI), or `bash scripts/gate-docker.sh`. J00 updates
this section.

## Machine constraints

| Machine | What it has | Use it for | Never on it |
| --- | --- | --- | --- |
| **Team laptop** (CachyOS) | Ryzen 7 3750H, 8 threads; 23 GiB RAM shared with a desktop (often ~6–7 GiB free); GTX 1660 Ti Max-Q, **6 GB VRAM**, compute 7.5, no BF16; ~97 GB free disk (80% full); system Python **3.14**; pyenv (3.12 only); Docker without an NVIDIA runtime | J-jobs: code, offline tests, the gate, docs, analysis of committed evidence | Loading Llama-3.1-8B (needs ~16 GB VRAM); the FAISS index (64.5 GB) or corpus (14 GB); torch/vLLM environment builds; index builds; running more than 2 agents at once |
| **A100 Pod** (RunPod, `ENVIRONMENT.md`) | A100 80 GB; ~116 GiB RAM; 27 CPU-equivalents; 250 GB network volume | E-jobs and J23: model runs, profiling, index builds, training | Committing `config.local.json`, tokens or weights; leaving GPU processes running |
| **GitHub CI** (from J00) | ubuntu-latest, Python 3.11, no GPU | The gate on every PR and push to `main` | Anything that needs FlashRAG, vLLM, torch or data downloads |

Rules that follow from this:
- **Python 3.11 only for the gate.** Use `.venv` from `scripts/setup-venv.sh` (pyenv
  3.11) or `scripts/gate-docker.sh` (`python:3.11-slim`). The system 3.14 may fail to
  install the pinned wheels, or behave differently from CI.
- **The gate must fit the laptop:** ≤ 5 minutes, ≤ 2 GB peak RAM, ≤ 10 s per test.
- **Disk.** Downloads go to ignored folders and stay small. Remove your worktree after
  merging.
- **If the machine is busy** (under 3 GiB free, or another gate running), wait rather
  than thrash.

## Commit hygiene

- Commit messages are `<ID>: <what changed>` and nothing else.
- No `Co-Authored-By`, `Assisted-by` or other attribution trailers, no "Generated with"
  footers, no emoji, whatever your tool's default.
- Each PR description has exactly one line `Provenance: produced with <agent> (<model>)
  from prompt <ID>`; that is the only disclosure.

## Never

- Edit frozen paths.
- Force-push, push to `main`, or merge with red CI.
- Commit secrets, weights, the index, the corpus or files over 1 MB.

Full rules: `docs/prompts/RULES.md`. Course policy on AI use: `docs/ledger/README.md`.
