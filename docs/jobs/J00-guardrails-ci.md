---
id: J00
title: Guardrails and CI
status: todo
env: cpu
size: S
depends: []
---
# J00 Guardrails and CI

**Goal.** Create the offline gate that every later job relies on:
- one local command, `scripts/gate.sh`, that CI also runs;
- a code-limit checker;
- a frozen-path guard;
- a no-skip rule;
- a large-file guard;
- a test-only `flashrag` stub, so the existing Judge integration tests actually run
  instead of skipping.

**Why.** There is no CI at all today (`.github/` is absent). The existing integration
tests skip whenever FlashRAG is missing, which is a false green. architecture.md §5–6.

## Research questions (save to `docs/research/J00-guardrails-ci.md`)
1. Current pinned versions of `actions/checkout` and `actions/setup-python`; how to cache
   pip; how to cancel superseded runs with `concurrency`.
2. How to make `unittest` fail when tests are skipped (count `result.skipped`), and how
   to run the frozen tests with `DRAG_REQUIRE_INTEGRATION=1`, which turns skips into
   failures.
3. How to measure function length and cyclomatic complexity for Python. Choose
   between `ast` plus a small McCabe count and `ruff` (rule `C901`); justify the choice.
4. Repo merge settings (squash allowed? delete branch on merge?) and whether branch
   protection is available (`gh api repos/{owner}/{repo}/branches/main/protection`).
   Read only: do not change anything.

## Scope
- **In:** CI workflow, gate script, checker and its tests, the stub, CI requirements,
  the PR template, and an `AGENTS.md` commands update.
- **Out:** any pipeline code; any change to frozen paths; any repository setting.

## Files
- **Create:**
  - `.github/workflows/ci.yml`
  - `scripts/gate.sh`
  - `scripts/setup-venv.sh`
  - `scripts/gate-docker.sh`
  - `scripts/check_code_limits.py`
  - `scripts/code_limits_allowlist.json`
  - `requirements-ci.txt`
  - `improvements/tests/__init__.py`
  - `improvements/tests/stubs/flashrag/{__init__,pipeline,utils,prompt}.py`
  - `improvements/tests/test_code_limits.py`
  - `.github/pull_request_template.md`
- **Modify:** `AGENTS.md` (Commands section only), `.gitignore` (add `.venv/`).

## Design notes and edge cases
- **One gate.** CI runs `bash scripts/gate.sh`, nothing else, so local equals CI.
- **What the gate runs:**
  - `scripts/verify_evidence.py`;
  - `scripts/test_submission.py`;
  - `scripts/run_evaluation.py --validate-only`;
  - the frozen Judge tests with `PYTHONPATH` including `improvements/tests/stubs` and
    `DRAG_REQUIRE_INTEGRATION=1`;
  - `improvements/tests` with the skip check;
  - the limits checker;
  - the frozen-path guard and the large-file guard.
- **Guard comparison base.** On a PR, compare against the merge base with
  `origin/main`. On a push to `main`, compare against `HEAD~1`. When run locally,
  compare against `origin/main`.
- **Stub scope.** Provide only what the frozen modules import:
  - `flashrag.pipeline.BasicPipeline` (stores config and prompt_template; `evaluate`
    returns the dataset);
  - `flashrag.utils.get_generator` and `get_retriever` (raise if called);
  - `flashrag.prompt.PromptTemplate` (accepts arguments, does nothing).

  The stub must never be importable from runtime code paths.
- **Allowlist.** Existing over-limit code is grandfathered by name and current size,
  and may never grow: `scripts/run_evaluation.py::run` (85 lines) and
  `scripts/verify_evidence.py::verify` (50 lines). Frozen paths are excluded entirely.
- **Python.** Pin 3.11 in CI to match `ENVIRONMENT.md`. The team laptop has only Python
  3.14 (system) and pyenv 3.12, so provide two local routes to 3.11:
  - `scripts/setup-venv.sh`: `pyenv install -s 3.11.<patch>`, then create `.venv`, then
    `pip install -r requirements-ci.txt`;
  - `scripts/gate-docker.sh`: runs `gate.sh` in `python:3.11-slim` with the repo mounted
    and a named pip-cache volume. CPU only; Docker on the laptop has no NVIDIA runtime.

  `gate.sh` must exit with a clear message if `python --version` is not 3.11. Add
  `.venv/` to `.gitignore`.
- **Resource budget.** The gate must run in ≤ 5 min with ≤ 2 GB peak RSS on the laptop:
  - Measure it with `/usr/bin/time -v bash scripts/gate.sh` and record wall time and
    maximum RSS in the status log.
  - `gate.sh` prints its own duration.
  - The test runner fails any single test that takes over 10 s, using a timing wrapper;
    don't depend on a pytest plugin.
- **CI requirements.** `requirements-ci.txt` holds only light packages (for example
  `python-Levenshtein`, `tqdm`, `numpy`), pinned to the versions in `ENVIRONMENT.md`.
- **Permissions.** `contents: read`.

## Acceptance
- [ ] CI runs on `pull_request` and on `push` to `main`, and is green on this PR.
- [ ] The frozen Judge integration tests **run** (not skip) in CI.
- [ ] Checker unit tests prove that each of these fails the check:
  - a 41-line function;
  - a 251-line file;
  - complexity 11;
  - growth of an allowlisted function;
  - a skipped test.
- [ ] Locally demonstrated (in a scratch copy, not committed) that touching a frozen
  path, or adding a > 1 MB file, fails the gate. Note this in the status log.
- [ ] The PR template contains the RULES.md §8 final-gate checklist.
- [ ] The gate passes through all three routes: CI, `.venv` from `setup-venv.sh`, and
  `gate-docker.sh`. The status log records wall time and maximum RSS for the local runs,
  both within the budget.
- [ ] `gate.sh` refuses Python ≠ 3.11 with a clear message (checked by running it under
  system Python 3.14).
- [ ] The status log records the merge settings and branch-protection result from
  research question 4. If protection is unavailable, write "main unprotected; gate
  self-enforced by RULES.md §0" and list the human action in `docs/prompts/README.md`.

## Status log
- PR:
- First pushed SHA:
- Evidence:
- Deviations:
- Follow-ups:
