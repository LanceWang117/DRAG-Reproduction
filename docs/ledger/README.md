# Implementation ledger

The ledger is the shared, durable record that every agent and teammate builds
from. It replaces the earlier `AUDIT.md` drafts. Nothing here depends on chat
history or any agent's memory.

> **Course policy (slide 9 of `L01-Intro_FAll 2026.pptx`).** AI is not allowed
> in the reports and code the team submits. AI may be used for research,
> understanding and debugging. Running the job prompts in `docs/prompts/` with
> an AI agent produces AI-written code. Get written instructor approval before
> doing that for graded work. Commits stay free of attribution trailers; each
> agent PR carries one Provenance line instead (`docs/prompts/RULES.md`, rule P1). Slide 10 also requires every member to
> code; do not let one account carry all the commits.

## Source of truth, in order

1. **Git `origin/main`.** A job is done only if a squash commit whose subject
   starts with its ID (for example `J06:`) is on `origin/main`. Its CI run must
   also be green.
2. **The job's own file** in `docs/jobs/`: spec, status and evidence.
3. **These ledger files**: facts, findings, architecture and protocol.
4. **Everything else** (chat, vault notes, an agent's memory, old drafts) is a
   hint. Re-verify it before relying on it.

If two sources disagree, the higher one wins. Fix the lower one in your PR.

## Files

| File | Holds | Change it when |
| --- | --- | --- |
| [baseline.md](baseline.md) | Frozen baseline numbers: accuracy, latency, call counts | E00 replaces an estimate with a measurement |
| [findings.md](findings.md) | Bottleneck register B1–B17 and minor defects | A job proves, refutes or fixes a finding |
| [architecture.md](architecture.md) | Code layout, feature flags, state record, log schemas, test strategy | A job adds a flag, module or schema |
| [protocol.md](protocol.md) | Experiment, statistics and promotion rules; stage gates | Only by team decision |
| [jobs.md](jobs.md) | Job index, dependencies and execution order | A job is added or split |
| [corrections.md](corrections.md) | Claims that turned out wrong, and their resolution | A claim is corrected |
| [references.md](references.md) | Verified and unverified sources | A source is checked or added |

Per-job specs and status live in [`../jobs/`](../jobs/). Per-job research notes
live in [`../research/`](../research/). The binding agent rules are in
[`../prompts/RULES.md`](../prompts/RULES.md).

## Evidence tags

| Tag | Meaning |
| --- | --- |
| **[D]** | Recomputed from data in the repo (say which file and script) |
| **[C]** | Read from code at the cited `path:line` |
| **[P]** | From a paper or another fetched source (listed in references.md) |
| **[M]** | Measured on hardware by an E-job (link its `experiments/<ID>/` record) |
| **[E]** | Estimate or hypothesis, waiting for measurement |
| **[IC]** | Insufficient context to verify |

**Rules for changing a fact:**
- Never upgrade a tag without evidence.
- An [E] becomes [M] only through an E-job record.
- A wrong claim is not silently deleted: move it to corrections.md with its
  resolution.

## Line-number convention

`drag.py:N` means `src/reproduction_changes/model/drag.py`, the file the runner
executes. The upstream `src/original_drag/model/drag.py` is identical except
that its line numbers are 3 lower up to the Judge block and 25 lower after it.
Line numbers drift as files change, so quote the function name as well as the
line.

## Frozen paths

These are never edited by any job. CI enforces it from J00 onward.

- `src/original_drag/`: the authors' code at commit `c035ee1`.
- `src/reproduction_changes/`: the recorded v1.0.1 baseline (Judge fix).
- `results/`: the recorded 200-question evidence and its checksums.

New code goes in `improvements/`. New experiment evidence goes in
`experiments/<job-id>/`.
