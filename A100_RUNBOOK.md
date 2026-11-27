# A100 runbook: run the server jobs (E00–E06)

This is the operator guide for the team member who runs the measurement jobs on the
A100 pod. It is written to be read once, then kept open in a second terminal.

You do **not** need to know the research design to run this. You need to start a pod,
copy two things onto it, paste one prompt per job, and check the result. The job specs
and the agent prompts do the rest.

**The short version**

1. Get access: GitHub write access, a Hugging Face account, the pod, and the private
   pack bundle from the owner.
2. Day 1: build the pod (2–4 h), then run **E00** (3 h).
3. Day 2: **E01** (4 h), then **E02a** (5 h).
4. Day 3: **E02b** (5 h), then **E03** (5 h).
5. Day 4: **E06** (10 h).

That is the four-day plan. If you only have three days, use §4.3. If you have two
days, use §4.2: setup + **E00** + **E01** on Day 1, then **E06** with only the baseline
and `P` on Day 2. Skip **E04** and **E05** in every plan; they do not fit a two- to
four-day booking.

---

## 1. Before the clock starts

### 1.1 What you need

| Item | Why | How to check |
| --- | --- | --- |
| GitHub **write** access to `LanceWang117/DRAG-Reproduction` | The agent pushes branches, opens PRs and merges | `gh auth status`; then `gh repo view LanceWang117/DRAG-Reproduction` |
| A Hugging Face account with **Llama 3.1 access** | Only if you download the model yourself | The model page shows "Access granted" |
| The pod: **A100 80 GB**, 1 GPU, a **250 GB network volume** | The corpus and index are about 115 GB before cleanup | Provider console |
| The private **pack bundle** (`drag-pack-YYYYMMDD.tar.gz`) and its SHA-256 | The agent prompts, job specs and ledger are local-only; they are not in git | The owner sends both out of band |
| About **3–4 days** and a spending cap | The jobs are long; the GPU bills by the hour | Agree the cap with the owner |
| The team's AI-use decision recorded (course rule H1) | You will run AI agents on graded work | The owner confirms it in writing |

**Cost.** The plan bills about 36 hours (setup plus the six jobs). At a typical
A100 80 GB rate of about $2–3.50/h, that is roughly **$70–130**. Rates change; check the
provider's live price and agree a cap with the owner before you start. Stopping the pod
between sessions stops the meter; the network volume keeps billing a small amount.

### 1.2 Owner checklist (the owner does this before handoff)

- [ ] Every `J` job that an `E` job depends on is merged and `main` CI is green.
- [ ] The required **local rehearsals (L01)** pass and each E-job status log has a
      `Local rehearsal:` line. These run on the PC, not the pod; you cannot do them.
- [ ] `python scripts/make_arms.py --check` passes (31 arm files match the spec).
- [ ] The partner is added as a repository collaborator with **Write**.
- [ ] This runbook is merged on `main`, so the partner gets it from `git clone`.
- [ ] The pack bundle is built, its SHA-256 is recorded, and it is sent privately.
- [ ] The model/corpus plan is agreed: either the partner downloads with their own
      Hugging Face account, or the owner pre-loads the volume and the partner skips
      the downloads.
- [ ] The pod template and network volume are agreed; the partner knows the budget cap.
- [ ] The test set (`experiments/J05-testset/`) is untouched and the partner knows
      never to use it before E06.

If a rehearsal line is missing, the partner will discover the bug on paid time. Do not
skip this checklist to save an hour.

E02a–E03 arms are generated from the **provisional** `P` (E01 arm C) until E01 promotes
it. Their pre-handoff L01 is therefore a structural check of the flags, not of the final
`P`; the owner can refresh it after E01 if they are available. The E-job agent refuses
an arm whose description still says `PROVISIONAL`.

For a **two-day run**, the L01 set is **E00 + E01 only**: E06 freezes those two
configurations and nothing else is run.

---

## 2. Day 1, part 1: build the pod (2–4 h, once)

Do this in a `tmux` session so a dropped browser tab does not kill it:

```bash
tmux new -s setup
```

### 2.1 Create the pod

- One **A100 80 GB** GPU (SXM4 or PCIe).
- At least **8 vCPU** and **64 GB RAM**. E04 would want more, but it is optional.
- A **250 GB network volume** mounted at `/workspace`.
- A PyTorch/CUDA 12.4 template, or plain Ubuntu 22.04/24.04.
- Keep the volume when the pod stops. The container disk is disposable.

Put **everything that must survive a restart** on `/workspace`: the clone, the conda
environment, the models, the corpus and `new-runs/`.

### 2.2 Clone and identity

```bash
cd /workspace
git clone https://github.com/LanceWang117/DRAG-Reproduction.git
cd DRAG-Reproduction
git config user.name  "Your Name"
git config user.email "you@example.com"
```

The GitHub CLI is installed with the environment in §2.3, so do that next. A stale
`GITHUB_TOKEN` is the usual cause of a 401 on the pod; unset it before `gh auth login`.

### 2.3 Python environment (on the volume)

Use a **prefix on the volume**. A named conda env in the container image disappears
when the pod restarts.

```bash
cd /workspace
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p /workspace/miniconda
rm miniconda.sh
eval "$(/workspace/miniconda/bin/conda shell.bash hook)"
conda create -y -p /workspace/envs/drag python=3.11
conda activate /workspace/envs/drag
conda install -y -c conda-forge gh rsync unzip
unset GITHUB_TOKEN
gh auth login
gh auth status
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The last line must print a version and `True`. If it prints `False`, you are on a CPU
pod or a broken CUDA image; stop and fix that before anything else.

Add the activation to your shell (or paste it after every reconnect):

```bash
echo 'eval "$(/workspace/miniconda/bin/conda shell.bash hook)"' >> ~/.bashrc
echo 'conda activate /workspace/envs/drag' >> ~/.bashrc
```

### 2.4 FAISS GPU

The pinned route is in `REPRODUCTION.md` ("Install FAISS GPU on the Pod"). Do not use
PyPI `faiss-gpu-cu12`: it conflicts with the pinned `numpy==1.26.4`.

```bash
conda activate /workspace/envs/drag
pip uninstall -y faiss-cpu
conda install -y -c conda-forge "faiss=1.8.0=py311cuda120hebc9808_1_cuda"
python -c "import faiss; print(faiss.__version__, faiss.get_num_gpus())"
```

Expect `1.8.0 1`. If it prints `0`, the GPU build is not active.

### 2.5 Models, corpus and index

These are large. Keep them on `/workspace`. The checksums and links are in
`DATA_DOWNLOADS.md`.

```bash
cd /workspace/DRAG-Reproduction
huggingface-cli login          # paste your token; never put it in git
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  --revision 0e9e39f249a16976918f6564b8830bc894c89659 \
  --local-dir models/Llama-3.1-8B-Instruct
huggingface-cli download intfloat/e5-base-v2 \
  --revision f52bf8ec8c7124536f0efb74aca902b2995e5bcd \
  --local-dir models/e5-base-v2
mkdir -p wiki_corpus
# Download wiki18_100w.jsonl and wiki18_100w_e5_index.zip from the
# ModelScope retrieval_corpus folder linked in DATA_DOWNLOADS.md.
unzip -o wiki_corpus/wiki18_100w_e5_index.zip -d wiki_corpus/
sha256sum wiki_corpus/wiki18_100w.jsonl wiki_corpus/wiki18_100w_e5_index.zip
stat -c '%s %n' wiki_corpus/e5_flat_inner.index
```

The two SHA-256 values must match `DATA_DOWNLOADS.md`. The extracted index should be
about 64.6 GB. You may delete the 37 GB ZIP after extraction.

If you cannot get Llama access quickly, the owner can pre-load the volume. You can
still log in to Hugging Face with any account; the model files are what the pipeline
reads.

### 2.6 `config.local.json`

```bash
cd /workspace/DRAG-Reproduction
cp config.example.json config.local.json
# Edit paths only if the data is not under the repo root.
# Relative paths resolve from the config file's directory.
```

Never commit this file. It is already in `.gitignore`.

### 2.7 Unpack the private pack

The pack is **not in git**. Upload the bundle to `/workspace` first (provider file
upload, `scp`, or a private link).

```bash
cd /workspace
sha256sum drag-pack-YYYYMMDD.tar.gz
# Compare with the SHA-256 the owner sent.
tar -xzf drag-pack-YYYYMMDD.tar.gz -C /workspace/DRAG-Reproduction
rm drag-pack-YYYYMMDD.tar.gz
cd /workspace/DRAG-Reproduction
cat >> .git/info/exclude <<'EOF'
docs/prompts/
docs/jobs/
docs/ledger/
docs/research/
AGENTS.md
CLAUDE.md
EOF
git check-ignore -v AGENTS.md docs/prompts/RULES.md
```

The last command must print matching exclude rules. If it prints nothing, the pack is
not ignored; stop, because an agent could commit it.

The pack contains the paste-ready prompts, the job specs, the ledger and the research
notes. It travels privately and is updated only by the owner, except for the files the
job rules say you may edit (see §3).

### 2.8 Readiness checks

```bash
cd /workspace/DRAG-Reproduction
bash scripts/pod_setup.sh --write-config
# Fix every MISSING line it prints, then rerun it until it reports 0 missing.
python scripts/server_checks.py --list --owner E00
python scripts/server_checks.py --run --config config.local.json
python scripts/preflight_gpu.py \
  --index-bytes "$(stat -c%s wiki_corpus/e5_flat_inner.index)" \
  --gpu-memory-utilization 0.45
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
```

`server_checks.py --run` takes a few minutes. It writes a JSON report under
`new-runs/server-checks/`; the E00 agent copies it into `experiments/E00/`.
`pod_setup` checks that *some* Hugging Face account is logged in; it does not test
Llama access. If the model is pre-loaded, any account is fine.

End the setup with a clean teardown:

```bash
bash scripts/pod_setup.sh --teardown
```

It must report that none of your processes hold the GPU. If it names a process you
did not start, do not kill it; note it and tell the owner.

**Checkpoint.** Record these in a local note: `nvidia-smi`, `python -V`,
`git rev-parse HEAD`, the data checksums and the `pod_setup` result. If something
breaks later, this is the "known good" state.

---

## 3. The loop for every job

Run **one job at a time**, in **one agent session**, with **one GPU job**. Do not run a
second agent while the first is running.

1. Start a `tmux` session named after the job:

   ```bash
   tmux new -s E00
   ```

2. Start a **fresh** agent session (Claude Code, Codex, opencode, …) at the **repo
   root**, the checkout that holds the unpacked pack:

   ```bash
   cd /workspace/DRAG-Reproduction
   ```

3. Paste the whole of `docs/prompts/E00.md` (then `E01.md`, and so on). Do not
   summarise it, edit it, or add to it. The prompt is the complete instruction.

4. Let it run. It will create its own worktree (`../DRAG-wt-E00`), work in small
   commits, open a draft PR, run the gate, and merge when CI is green.

5. When it stops, read its final report and use the table below.

| Final state | What you do |
| --- | --- |
| **DONE** | Check the job (§3.1), then start the next job |
| **AWAITING-REVIEW** | The owner must approve the PR; then paste the **same** prompt to let it merge |
| **BLOCKED-HUMAN / BLOCKED-ENV** | Do the exact action in the report (login, missing file, permission); paste the same prompt |
| **BLOCKED-DEPENDENCY** | A `J` job is not merged; tell the owner; do **not** run later jobs |
| **BLOCKED-TECHNICAL** | Read the PR's blocker; send the failure tail to the owner; do not retry blindly |
| **FAILED-REVERTED** | `main` is green again; send the cause to the owner; wait for a fixed spec before re-pasting |

If the same failure happens three times, the prompt's stop rule says to stop. Let it.
Do not "just try once more" on a paid GPU.

### 3.1 Check the agent's work before starting the next job

Five minutes here saves a wasted day.

```bash
cd /workspace/DRAG-Reproduction
git fetch origin
git log origin/main --oneline --grep '^E00:' | head -1     # a merged commit exists
gh pr list --state merged --search 'E00' --limit 3          # the PR is merged
ls experiments/E00                                          # README, summary, figures, SHA256SUMS
(cd experiments/E00 && sha256sum -c SHA256SUMS)             # evidence is intact
cat docs/ledger/inbox/E00.md                                # proposed ledger edits exist
git show --stat "$(git log origin/main --format=%H --grep '^E00:' -1)" | head -30
```

In the last command, the changed paths must **not** include `docs/jobs/`,
`docs/prompts/`, `docs/ledger/` (except the inbox), `config.local.json`, `models/`,
`wiki_corpus/` or `new-runs/`. The PR body must carry the single
`Provenance: produced with …` line.

Also read the job file's status log at `docs/jobs/E00-*.md`: it must have the PR link,
the merged SHA, the evidence and any deviations.

If any of these is missing, the job is not done. Tell the owner; do not start the next
job on top of it.

Before any job that builds on the promoted configuration (E02a, E02b, E03, E06), also
check that `improvements/configs/arms/measured.json` has `P` with
`"provisional": false` and a source citing `experiments/E01/summary.json`. After E00,
`threads` must also be non-provisional before arm E is used. If either is still
provisional when you start E02a, E01 did not finish its promotion step; the generated
arms are not ready to book.

### 3.2 End of session

```bash
bash scripts/pod_setup.sh --teardown    # no GPU process may remain
```

If you are stopping for the day, **stop the pod** (do not terminate it) to stop
billing. The network volume keeps the clone, the environment, the data and the
evidence. On restart, reactivate the environment and re-login to `gh` and Hugging
Face; those credentials live on the container disk.

---

## 4. Order, time and cut lines

The jobs are ordered by dependency. Do not reorder them. E00 always runs first: it
re-establishes the baseline and measures the noise floor that every later comparison
uses.

### 4.1 Four-day plan (recommended)

| When | Job | Budget | Finished when | If it slips |
| --- | --- | ---: | --- | --- |
| Day 1, morning | Pod setup (§2) | 4 h | `pod_setup` reports 0 missing; `server_checks --run` passes | Do not start E00 |
| Day 1, afternoon | **E00** baseline, A/A, profiling | 3 h | `experiments/E00/summary.json`; A/A floor; baseline agreement | Stop and fix; do not start E01 |
| Day 2, morning | **E01** answer-preserving speedups | 4 h | Promoted configuration `P`; `measured.json` updated | Run E02a's core arms A–D only |
| Day 2, afternoon | **E02a** novelty exit and budgets | 5 h | Promotion decision per arm | Skip the optional arms |
| Day 3, morning | **E02b** evidence and Judge | 5 h | Promotion decisions; prefix-cache decision | Skip arm F (prefix cache) |
| Day 3, afternoon | **E03** throughput sweep | 5 h | Best batch size; optional variants measured or skipped | Run one repeat of the sweep; skip INT4/spec |
| Day 4 | **E06** final test evaluation | 10 h | `FROZEN.md` committed, 600 questions per frozen config, final tables | Freeze fewer arms (see below) |

This is about 32 GPU hours plus setup. The days are long; if your working day is
8 hours, the plan slips to five days.

Setup is the most variable step. If it runs past 4 hours, start E00 on Day 2 morning
and switch to the three-day plan (drop E03). Do not compress E00.

### 4.2 Two-day minimum (only if the booking cannot be extended)

This produces one defensible result: the pre-registered test comparison of the
baseline against E01's promoted configuration `P`. It does **not** test the E02a/E02b
method changes or E03 throughput.

| When | Job | Budget | Finished when |
| --- | --- | ---: | --- |
| Day 1, start | Setup (§2) | 2–4 h | `pod_setup` 0 missing; checks pass |
| Day 1, middle | **E00** (full: baseline, A/A, profiling) | 3 h | `experiments/E00/summary.json`; A/A floor |
| Day 1, end | **E01** (arms A–E) | 4 h | `P` promoted; `measured.json` updated |
| Day 2, all day | **E06**: baseline + `P` only | 10 h | `FROZEN.md` committed; 600 questions × 2 configs |

- About **17–21 hours** including setup. Two long days; **there is no slack**.
- E06 freezes exactly **two** configurations: `baseline.json` and `P`. Do not add an
  E02a/E02b/E03 arm; the day cannot absorb it.
- **Go/no-go before E06:** compute `600 × (baseline seconds + P seconds)` from E00 and
  E01. If it is over **9 hours**, the two-day plan cannot finish both full 600-question
  runs. Report the E00/E01 dev results and mark the test evaluation as not run; an
  incomplete test run is not analysed (pre-registration §6).
- If setup takes more than 4 hours, or E00/E01 slip past the end of Day 1, apply the
  same rule: do not start E06 with less than 10 hours left.
- If the owner can pre-build the pod volume, do it. That is the difference between a
  comfortable and a risky two days.
- Do **not** shorten E00 (skip profiling) or E01 (skip arms) to make it fit. Those cuts
  break the job's acceptance and leave nothing to compare.

### 4.3 Three-day minimum

If you have three days, protect the final result:

| When | Jobs |
| --- | --- |
| Day 1 | Setup + **E00** |
| Day 2 | **E01** + **E02a** |
| Day 3 | **E02b** + **E06** |

Skip **E03** in this plan. E06 then freezes at most **two configurations**:
the baseline and `P` (or the best single promotion from E02a/E02b). Without E03's
question batching, E06 is slower; two configs is what fits.

### 4.4 Cut lines (apply these without asking)

- **No green E00 by the end of Day 1** → stop. Do not start E01. Send the owner the
  blocker. Every later number depends on E00's baseline.
- **No promoted `P` by the end of Day 2** → run E02a's core arms A–D only; skip the
  optional `response_rounds = 2` and `agent_max_tokens = 128` arms.
- **No E03 by the end of Day 3** → skip the INT4 and speculative-decoding variants;
  they are explicitly optional. Keep the batch sweep.
- **Less than 10 hours left** → do not start E04. Go to E06.
- **Less than 10 hours left and E06 has not started** → freeze at most two
  configurations and run. A complete two-arm test result beats an incomplete four-arm
  one.
- **Two-day plan:** if setup + E00 + E01 are not finished by the end of Day 1, do not
  start E06. Report the dev results and mark the test evaluation as not run.
- **A run is interrupted** → resume it; never restart it, and never analyse a partial
  run. Every runner has `--resume`.

### 4.5 What is skipped, and why

| Job | Status in this plan | Reason |
| --- | --- | --- |
| **E04** BM25 / hybrid | Optional; only with a 5th day | An 8 h CPU/RAM-heavy index build; it would push E06 to Day 5 |
| **E05** train labels / calibration | Optional; only with 5+ days | Another 8 h; it needs a train-split draw and refits the gate/router |
| **J23** LoRA-DPO Judge | Do not run | It depends on E05 |
| **J27–J29** Jev backends | Do not run | Not merged on `main`; their E-track follow-ups are not booked |

E05 is no longer blocked by J20 (the router is merged), but it is still a large extra.
Do not squeeze it into a four-day plan.

---

## 5. Manual fallback: the exact run commands

Use this section only if the agent cannot run, or if you need to reproduce a run by
hand. The agent remains the main path: it writes the analysis, the figures and the
evidence files. The commands below produce the raw runs; the job file lists the
analysis each job must write.

Every command assumes:

```bash
conda activate /workspace/envs/drag
cd /workspace/DRAG-Reproduction
```

`--output` is a **parent** directory: the runner writes a timestamped subdirectory
under it (for example `new-runs/E01-A/20261126T101500Z-E01-A/`) and prints the exact
path. Use that printed path with `--resume`; do not pass the parent.

### 5.0 Make the dev subsets (E00, E01, E03, E02b arm F)

E00's subset is the **first 10 questions per dataset in manifest order** (60 total).
E00's profiling subset is 5 per dataset (30). This snippet writes one subset; it reads
the committed dev manifest and never touches the test set.

```bash
mkdir -p new-runs
python - 10 <<'PY'
import json, sys
from collections import Counter
per = int(sys.argv[1])
rows = [json.loads(line) for line in open("results/questions-200.jsonl") if line.strip()]
taken, out = Counter(), []
for row in rows:
    if taken[row["dataset"]] < per:
        taken[row["dataset"]] += 1
        out.append(row)
path = f"new-runs/dev{per * len(taken)}.jsonl"
with open(path, "w") as fh:
    for row in out:
        fh.write(json.dumps(row) + "\n")
print(path, len(out), dict(taken))
PY
```

Repeat with `5` instead of `10` for `new-runs/dev30.jsonl`. The output should name six
datasets and 60 (or 30) rows.

### 5.1 E00: baseline, A/A floor and profiling

```bash
# 1. Smoke: both runners, one NQ question.
python scripts/run_evaluation.py --method both --dataset NQ --limit 1 --config config.local.json
python scripts/run_variant.py --variant improvements/configs/baseline.json \
  --config config.local.json --dataset NQ --limit 1 --output new-runs/E00-smoke

# 2. Reproduction check: 60 dev questions, baseline.
python scripts/run_variant.py --variant improvements/configs/baseline.json \
  --config config.local.json --manifest new-runs/dev60.jsonl --output new-runs/E00-repro

# 3. A/A floor: the same 60 questions again, in a new directory.
python scripts/run_variant.py --variant improvements/configs/baseline.json \
  --config config.local.json --manifest new-runs/dev60.jsonl --output new-runs/E00-aa

# 4. Profiling: 30 questions (5 per dataset), with per-call timings.
python scripts/run_variant.py --variant improvements/configs/baseline.json \
  --config config.local.json --manifest new-runs/dev30.jsonl --profile \
  --output new-runs/E00-profile
```

The reproduction comparison, the three A/A counts, the stage figure and the retriever
micro-benchmark are E00's analysis deliverables. If the analysis script is not
committed, the job is not finished; do not hand-write the ledger numbers.

### 5.2 E01: arms A–E

Each arm is a separate process. Add about 5 minutes of model/index startup per arm.
`--profile` is required if you will report latency, because `summary.json` holds
metrics, not per-call timings.

```bash
for arm in A B C D E; do
  python scripts/run_variant.py --variant "improvements/configs/arms/E01/$arm.json" \
    --config config.local.json --manifest new-runs/dev60.jsonl --profile --interleave \
    --output "new-runs/E01-$arm"
done
```

`--interleave` alternates `drag`/`naive` within each arm. The CLI has no
"interleave all arms" flag; running A→E in order is the supported path. Record the
order in the job status log, and note that cross-arm drift is a caveat.

Arm D needs the GPU preflight:

```bash
python scripts/preflight_gpu.py \
  --index-bytes "$(stat -c%s wiki_corpus/e5_flat_inner.index)" \
  --gpu-memory-utilization 0.45
```

If D fails or OOMs, use arm E (CPU FAISS with the best thread count from E00). The job
spec says so; do not invent a new GPU setting.

### 5.3 E02a: all 200 dev questions

```bash
for arm in A B C D; do
  python scripts/run_variant.py --variant "improvements/configs/arms/E02a/$arm.json" \
    --config config.local.json --profile --interleave --output "new-runs/E02a-$arm"
done
# Optional arms only if the time budget allows:
for arm in opt-r2 opt-tokens128; do
  python scripts/run_variant.py --variant "improvements/configs/arms/E02a/$arm.json" \
    --config config.local.json --profile --interleave --output "new-runs/E02a-$arm"
done
```

### 5.4 E02b: all 200 dev questions, then arm F twice on the 60

```bash
for arm in A B C D E; do
  python scripts/run_variant.py --variant "improvements/configs/arms/E02b/$arm.json" \
    --config config.local.json --profile --interleave --output "new-runs/E02b-$arm"
done
for run in 1 2; do
  python scripts/run_variant.py --variant improvements/configs/arms/E02b/F.json \
    --config config.local.json --manifest new-runs/dev60.jsonl --profile --interleave \
    --output "new-runs/E02b-F-run$run"
done
```

### 5.5 E03: batch sweep

```bash
for run in 1 2 3; do
  for b in B1 B2 B4 B8 B16 B32; do
    python scripts/run_variant.py --variant "improvements/configs/arms/E03/$b.json" \
      --config config.local.json --manifest new-runs/dev60.jsonl --profile \
      --output "new-runs/E03-$b-r$run"
  done
done
```

That is 18 runs. If the budget is tight, run one repeat of all six first, then repeat
only `B1` and the best B. INT4 and speculative decoding have no arm files; they are
model-level options the agent handles, and each may be explicitly skipped with a
reason (no checkpoint, no fit, or unsupported in vLLM 0.6.6).

### 5.6 E06: final test run

The agent writes `experiments/E06/FROZEN.md` and one variant file per frozen
configuration, commits them, **then** runs. The commit order is the evidence: no test
question may be run before `FROZEN.md` is on `main`.

```bash
# Baseline, once.
python scripts/run_variant.py --variant improvements/configs/baseline.json \
  --config config.local.json --manifest experiments/J05-testset/questions-test.jsonl \
  --profile --interleave --output new-runs/E06-baseline

# Each frozen configuration, once.
python scripts/run_variant.py --variant improvements/configs/arms/E06/<name>.json \
  --config config.local.json --manifest experiments/J05-testset/questions-test.jsonl \
  --profile --interleave --output "new-runs/E06-<name>"
```

- A run that finishes fewer than 600 questions for infrastructure reasons is
  **incomplete**. Resume it with `--resume <the run directory the runner printed>`; do
  not analyse the completed subset.
- Do not inspect partial outputs. The pre-registration forbids peeking.
- **Time:** one config costs roughly `600 × (DRAG seconds + Naive seconds)` plus
  startup. At 20 s + 5 s per question, that is about 4 h per config. Three configs is
  about 12 h, over the 10 h budget; two configs fit. Measure `P`'s seconds in E01 and
  choose the number of frozen configs accordingly.
- If a frozen configuration affects only DRAG, the agent may run the `drag` arm alone
  for it and keep both arms in the baseline run. Record that choice in `FROZEN.md`.

### 5.7 E04 and E05 (optional, only with more days)

Use the agent. There is no committed "build the BM25 index" script; that build and the
candidate dump are E04's deliverables. E05 needs a train-split draw and refits the
gate/router. Both are specified in their job files; do not improvise them in a
four-day run.

---

## 6. If something goes wrong

| Symptom | What to do |
| --- | --- |
| `nvidia-smi` shows a process you did not start | Do not kill it. Note the PID and ask the owner. |
| Arm D OOMs at `gpu_memory_utilization = 0.45` | Use arm E (CPU FAISS) as the job says; record the VRAM numbers. Do not raise the ratio. |
| `faiss.get_num_gpus()` is 0 | Reinstall the conda-forge GPU build (§2.4); check `conda list faiss`. |
| Hugging Face returns 401 or "gated" | Request Llama access on the model page with the same account; or ask the owner to pre-load the volume. |
| `git push` returns 403 | `gh auth status`; `unset GITHUB_TOKEN`; ask the owner for Write access. |
| Disk full during a run | `df -h /workspace`; delete the extracted ZIP after checking the index; remove old `new-runs/` only after the job's evidence is committed. |
| The pod restarts | Reactivate the env (`conda activate /workspace/envs/drag`), re-login `gh`/HF, `git status`; resume the run with `--resume`. |
| The agent says BLOCKED-ENV | Do the specific missing step it names, then paste the **same** prompt. |
| The agent loops on one failure | Let the three-attempt stop rule fire; send the failure tail to the owner. Do not restart it a fourth time. |
| CI is red on someone else's PR | Do not merge around it. Tell the owner; wait for `main` to be green. |

---

## 7. Handback

When the last job is merged, the committed evidence is already on GitHub. The owner
pulls `main` and gets `experiments/`, the code and the runbook.

The **local pack** is different: it lives only on the pod. Send back only the files the
rules let you edit.

Option A, from the owner's PC with SSH to the pod:

```bash
bash scripts/pack_sync.sh pull <pod>:<repo>
```

Option B, from the pod, a small tarball:

```bash
cd /workspace/DRAG-Reproduction
mkdir -p docs/ledger/inbox
tar -czf /workspace/pack-back-$(date +%Y%m%d).tar.gz docs/jobs docs/research docs/ledger/inbox
sha256sum /workspace/pack-back-*.tar.gz
```

Send the tarball and the hash to the owner. On the PC, extract it at the main checkout:

```bash
tar -xzf pack-back-YYYYMMDD.tar.gz -C /path/to/DRAG-Reproduction
```

Do **not** send `new-runs/` unless the owner asks. It is large and ignored; the
committed `experiments/` records are the evidence.

Before you stop the pod:

```bash
bash scripts/pod_setup.sh --teardown
```

**Stop** the pod to stop billing. **Terminating** deletes the network volume; ask the
owner before you terminate anything.

---

## 8. Words you will see

| Word | Meaning here |
| --- | --- |
| **PACK** | The local-only folder with `AGENTS.md`, `docs/prompts/`, `docs/jobs/`, `docs/ledger/` and `docs/research/`. It is not in git. |
| **Arm** | One variant of the pipeline, described by a JSON file under `improvements/configs/arms/`. |
| **A/A floor** | How much a run changes when you repeat the *same* configuration. Later differences smaller than this are noise. |
| **`P`** | The configuration E01 promotes; later jobs use it as their base. |
| **`experiments/<ID>/`** | The compact, committed evidence for a job: README, `summary.json`, figures, tables, `SHA256SUMS`. |
| **Inbox** | `docs/ledger/inbox/<ID>.md`: the ledger edits an E-job proposes; the owner applies them on the PC. |
| **Provenance line** | One line in each PR body naming the agent and model used. No commit trailers. |
