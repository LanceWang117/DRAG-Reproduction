# Job index and execution order

One job = one session = one PR = one squash commit on `main` whose subject starts with
the job ID. Each job's spec, status and evidence live in `docs/jobs/<ID>-*.md`; its
paste-ready prompt is `docs/prompts/<ID>.md`.

- **J-jobs** run anywhere with Python 3.11 and need no GPU. CI verifies them.
- **E-jobs** run on the A100 Pod and produce experiment evidence.

**Status is not tracked in this table** (to avoid merge conflicts).
- **Is a job merged?** `git log origin/main --oneline --grep '^J06:'`
- **What does its file say?** `grep -H '^status:' docs/jobs/*.md`

If the two disagree, git wins.

## Order (easiest to hardest, respecting dependencies)

| # | ID | Title | Env | Size | Depends on | Addresses |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | J00 | Guardrails and CI | cpu | S | none | architecture §6 |
| 2 | J01 | Analysis toolkit: extractor and statistics | cpu | S | J00 | B1, B12 |
| 3 | J02 | Pipeline skeleton, flags and golden transcript | cpu | M | J00 | architecture §2–5 |
| 4 | J03 | Instrumentation and variant runner | cpu | M | J02 | F0 profiling |
| 5 | J04 | Single-path retrieval cache | cpu | S | J03 | B2 |
| 6 | J05 | Test-set draw and pre-registration | cpu + network | S | J01 | B1 |
| 7 | E00 | Pod baseline rerun, A/A floor and profiling | gpu | M | J01, J03, J04 | B1, B2, B5, all [E] |
| 8 | J06 | Final-only moderator and intra-round batching | cpu | S | J02 | B3, B4 |
| 9 | J07 | Duplicate-query guard and zero-novelty exit | cpu | S | J02 | B15 |
| 10 | J09 | Retrieval acceleration configuration | cpu | S | J03 | B5 |
| 11 | E01 | Validate answer-preserving speedups | gpu | M | E00, J06, J09 | B3, B4, B5 |
| 12 | J08 | Round and output-length budgets | cpu | S | J02 | B6 |
| 13 | J10 | Anchored evidence store | cpu | M | J02 | B7, B8, B13 |
| 14 | J11 | Token budget and evidence-first layout | cpu | M | J10, J03 | B8, B9 |
| 15 | J12 | Log-prob Judge with order balancing | cpu | M | J02 | B10 |
| 16 | J13 | Structured Opponent actions | cpu | M | J10, J12 | B7, B14 |
| 17 | E02a | Dev variants: novelty exit and budgets | gpu | M | E01, J07, J08 | B6, B15 |
| 18 | E02b | Dev variants: evidence and Judge | gpu | M | E01, J10–J13 | B7–B10, B14 |
| 19 | J14 | Calibrated escalation gate and lazy Opponent | cpu | M | J08, J12, J01 | B6, B16 |
| 20 | J15 | Per-question state and barrier scheduler | cpu | M | J06, J04 | B4 |
| 21 | J16 | Refill window and batched retrieval | cpu | M | J15 | B4, B5 |
| 22 | E03 | Throughput sweep and model-level variants | gpu | M | E01, J16 | B4 |
| 23 | J17 | Hybrid retrieval code | cpu | M | J04 | B17 |
| 24 | E04 | BM25 index build and hybrid retrieval evaluation | gpu-pod (CPU/RAM heavy) | L | E00, J17 | B17 |
| 25 | J18 | Grounded answer stage and vote baselines | cpu | M | J02, J01 | B11 |
| 26 | J19 | Faithfulness signal and weighted aggregator | cpu | M | J18 | B11 |
| 27 | J20 | Cost-aware router | cpu | M | J01, J03, J12 | B16 |
| 28 | E05 | Train-split labels and calibration | gpu | L | E03, J14, J20 | B16 |
| 29 | J21 | Conformal selective answering | cpu | M | J01, J20 | — |
| 30 | J22 | Multi-hop decomposition route | cpu | L | J10, J12 | B7, B17 |
| 31 | J23 | LoRA-DPO Judge (optional) | gpu | L | J12, E05 | B10 |
| 32 | E06 | Final frozen-configuration test evaluation | gpu | L | J05 and every job whose configuration is promoted | B1 |

**Size:** S ≈ a short session; M ≈ a full session; L = a full session with long
background runs. All three are designed to finish in one session. If a job cannot finish,
it ends BLOCKED with a resumable draft PR (RULES.md §9), and re-running the same prompt
continues it.

## Running jobs in parallel

Run one job at a time unless you need speed. If you do run two, pick jobs with no
dependency between them and in different areas. These pairs are low-risk:

- J01 ∥ J02
- J05 ∥ J03
- J06 ∥ J07
- J10 ∥ J12
- J17 ∥ J18

Jobs that touch `pipeline.py`, `flags.py` or the architecture flag table **will**
conflict on small hunks. RULES.md §8 says how to resolve that (merge `main` into the
branch, never rebase).

**Never run two jobs in the same checkout.** Each job uses its own `git worktree`
(RULES.md §2).
