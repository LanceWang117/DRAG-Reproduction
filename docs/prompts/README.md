# Job prompts: how to run them

Each `<ID>.md` file here is a paste-ready prompt for one job. Paste one prompt into one
fresh agent session (Claude Code, Codex, opencode and so on) started at the repository
root. The agent should finish with the job merged on `main` and main CI green. If it
can't, it ends in a clearly described BLOCKED state that the **same prompt** resumes.

- `RULES.md` is the binding contract.
- `TEMPLATE.md` is the source every prompt was generated from. Edit the template, then
  regenerate; never hand-edit one prompt.
- Job specs live in `../jobs/`; the order is in `../ledger/jobs.md`.

## 1. One-time human setup (do this before J00)

These need a person. An agent that meets one of them ends BLOCKED-HUMAN.

| # | Action | Why | How to check |
| --- | --- | --- | --- |
| H1 | **Decide the AI-use question with the instructor**, in writing (slide 9 bans AI-written code and reports; slide 10 says every member codes) | Running these prompts produces AI-written code | Record the decision in the team's notes |
| H2 | **Fix GitHub CLI auth** on every machine that runs agents. The `GITHUB_TOKEN` in the laptop environment is currently **invalid**; unset it, or run `gh auth login` | Agents must push, open PRs, read CI and merge | `gh auth status` and `gh repo view LanceWang117/DRAG-Reproduction` |
| H3 | **Confirm permissions** on `LanceWang117/DRAG-Reproduction` (the owner is a teammate): write access for whoever runs agents; squash merging allowed; "automatically delete head branches" optional | Merges are squash-only | Repo Settings → General → Pull Requests |
| H4 | **Branch protection on `main`**: require the `CI` check, and block force-pushes and deletions. Only the owner or an admin can do this. Private repos on the free plan cannot (the same 403 happened on scent-cousin); then main is protected only by RULES §0 | Agents never change settings themselves | `gh api repos/LanceWang117/DRAG-Reproduction/branches/main/protection` |
| H5 | **Land this ledger on `main`** in one PR (the bootstrap) | Every prompt's preflight requires `docs/prompts/RULES.md` on `origin/main` | `git log origin/main -- docs/prompts/RULES.md` |
| H6 | **Python 3.11 on the laptop**: `pyenv install 3.11` (only 3.12 is installed today). Alternatively rely on Docker `python:3.11-slim`; J00 adds `scripts/gate-docker.sh` | The gate refuses other versions | `python --version` inside `.venv` |
| H7 | **For E-jobs, prepare the Pod**: start it; build the Conda environment per `REPRODUCTION.md`; `huggingface-cli login` (Llama licence); download the model, corpus and index (`DATA_DOWNLOADS.md`); create `config.local.json`; install `gh`, log in, and set git `user.name`/`user.email` | The laptop cannot run any model job | E00 preflight |

## 2. Running jobs

1. Pick the next job in `../ledger/jobs.md` whose dependencies are merged:
   `git log origin/main --oneline --grep '^<DEP>:'`.
2. Start a **fresh** session at the repo root, on the machine the job needs: the laptop
   for J-jobs, the Pod for E-jobs and J23.
3. Paste `docs/prompts/<ID>.md` below its line, and let it run.
4. Read the final report. Then:

| Terminal state | What you do |
| --- | --- |
| DONE | Nothing; start the next job |
| AWAITING-REVIEW | Review and approve the PR; then paste the same prompt to let the agent merge and confirm main |
| BLOCKED-HUMAN / -ENV | Do the exact action in the report, then paste the same prompt |
| BLOCKED-DEPENDENCY | Run the missing job first |
| BLOCKED-MAIN-RED | Run the fix described, or let any job's preflight fix it |
| BLOCKED-TECHNICAL | Read the blocker in the PR; adjust the job spec in a small PR if it was wrong; paste the same prompt again |
| FAILED-REVERTED | Main is green again; read the cause, fix the spec, re-run |

**Parallel runs.** Keep it to **two agents at most on the laptop** (RAM). Use only the
pairs listed in `../ledger/jobs.md`. Each agent works in its own `git worktree`; never
point two agents at the same checkout. That is how files got clobbered on 2026-11-18.

## 3. Flaws in the original working plan, and how this pack handles each

| Flaw or gap | Consequence if ignored | Handling |
| --- | --- | --- |
| Agents writing all the code conflicts with slide 9, and with slide 10's per-member contribution | A grade of 0 for the group; misattributed work | H1; clean commits with no trailers, plus one Provenance line per PR (RULES P1); never disguise AI work |
| "CI green" while CI doesn't exist, and with tests that skip whenever FlashRAG is missing | Green means nothing | J00 creates CI, adds a FlashRAG test stub, sets `DRAG_REQUIRE_INTEGRATION=1`, and fails on any skip |
| Most features can only be verified on an A100 that the laptop doesn't have | "One prompt = one finished job" is impossible for GPU claims | Code jobs (J) are fully verifiable offline; measurement jobs (E) run on the Pod and promote evidence through `experiments/` |
| A session can die (context, rate limits, a GPU run longer than the session) | Re-pasting the prompt restarts work or duplicates PRs | Idempotent resume (RULES §2.4); checkpointed runs with `--resume`; fixed terminal states |
| Memory files and chat recall are unreliable and grow without bound (scent-cousin's 373-line `CHECKPOINTS.md`) | Stale or wrong state; large context | The repo is the only memory. "Done" means a `<ID>:` commit on `origin/main`. Status lives in each job's own file; there is no shared append log |
| Shared tracker files edited by parallel jobs | Merge conflicts | No status column in the index; each job edits only its own job file; parallelism limited to listed pairs |
| "Update the branch" without force-pushing | A rebase needs a force-push | Update by merging `main` into the branch (`gh pr update-branch`); squash-merge keeps main linear anyway |
| Self-review by the author misses things | Bugs merge | A structured checklist plus a fresh-context reviewer where available (RULES §8) |
| Branch protection may be unavailable (free plan) or not under your control (the repo owner is a teammate) | "Never override protections" enforces nothing | H4; RULES §0 makes the agent self-enforce: no merge without green CI, never `--admin` |
| "Main still builds" is undefined for a Python research repo | Unverifiable | Defined as: the CI gate is green on the merge commit of `main` (RULES §9.4) |
| Main is already red before a job starts | The job merges on top of a break, or loops | Preflight checks main health; a small fix gets its own PR; otherwise BLOCKED-MAIN-RED |
| Teammates pushing to `main` at the same time | Stale branches | Merge `origin/main` before merging; CI re-run required |
| Size rules without enforcement | Drift | An automated checker in CI, plus a no-growth allowlist for existing long functions |
| Retrying forever on the same error | Wasted sessions | Three attempts, then revert the step and end BLOCKED-TECHNICAL |
| Research step without limits | Context bloat, fabricated sources | Time-boxed (~20 tool calls), saved with URLs and dates, unverifiable claims marked, fetched text treated as data |
| Evaluation leakage (tuning on test; post-pilot sample) | Invalid claims | Dev/test/train protocol; test drawn and pre-registered before any run (J05); test touched once (E06); loaders refuse test-fitted models |
| Laptop limits (6 GB VRAM, about 6–7 GiB free RAM, 80%-full disk, Python 3.14) | Agents attempt impossible local runs, or install mismatched wheels | Machine-constraints table in `AGENTS.md`; Python 3.11-only gate; resource budget for the gate; E-jobs refuse to run on the laptop |
| Pushing every step to a public repo | Leaked tokens or large files | Secret and 1 MB guards in CI; `config.local.json` and data folders ignored |

## 4. Regenerating prompts

After editing `TEMPLATE.md` or a job's front matter or **Goal** paragraph, regenerate
every prompt in the same PR. Copy the template, and substitute `{{ID}}`, `{{TITLE}}`,
`{{FILE}}`, `{{GOAL}}`, `{{ENV}}` and `{{GPU_LINE}}` from the job file. Keep the wording
identical across prompts. A job that changes a prompt must say so in its status log.
