# Baseline snapshot

The recorded v1.0.1 baseline, from `results/` (200 paired questions; Llama-3.1-8B-Instruct
BF16 on vLLM 0.6.6.post1; E5-base-v2 with a CPU FAISS FlatIP index over the 21,015,324-
passage `wiki18_100w` corpus; top-3; 4,096-token input; 256-token output; greedy).

**Every improvement is compared against a baseline rerun by E00 with the same runner,
commit and environment, not against these historical numbers.** The historical runs used
cloud wrappers that are not in the repo, and the portable runner has never run on a GPU.

## Accuracy [D]

| Dataset (split) | n | Naive EM / F1 | DRAG EM / F1 | ΔEM | DRAG-only / Naive-only | Exact McNemar p | Paper ΔEM [P] |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| NQ (dev) | 50 | 44.00 / 54.72 | 34.00 / 50.56 | −10.00 | 0 / 5 | 0.062 | −1.40 |
| TriviaQA (dev) | 10 | 60.00 / 67.25 | 70.00 / 72.86 | +10.00 | 1 / 0 | 1.000 | 0.00 |
| PopQA (test) | 40 | 45.00 / 49.47 | 55.00 / 59.76 | +10.00 | 5 / 1 | 0.219 | +1.00 |
| 2Wiki (dev) | 40 | 15.00 / 18.00 | 35.00 / 38.13 | +20.00 | 10 / 2 | 0.039 | +14.00 |
| HotpotQA (dev) | 35 | 28.57 / 36.67 | 34.29 / 46.84 | +5.71 | 5 / 3 | 0.727 | +5.00 |
| StrategyQA (train) | 25 | 64.00 / 64.00 | 64.00 / 64.00 | 0.00 | 3 / 3 | 1.000 | +6.60 |
| **Pooled** | **200** | **39.00 / 44.96** | **44.00 / 52.06** | **+5.00** | **24 / 14** (162 ties) | **0.143** | — |

- **ΔEM is not significant.** 95% CI ≈ [−1, +11] pp. 2Wiki's p = 0.039 fails Holm
  correction over 6 datasets (that needs p < 0.0083).
- **ΔF1 = +7.1 pp.** Bootstrap CI ≈ [+1.3, +13.1], permutation p = 0.017. But Naive's
  answers are 2.3× longer (6.8 vs 2.9 words; 13.2 vs 2.8 on 2Wiki), and F1 penalises
  length.
- **Power.** With discordance 0.19 and 80% power, the minimum detectable effect at
  n = 200 is ≈ 8.6 pp. Required n: 10 pp → 149; **5 pp → ~590**; 3 pp → ~1,650.
- **The sample is uneven** (n = 10–50) and the extra 100 questions were chosen after a
  pilot. Treat the 200 as the **dev set**; claims need the fresh test set from J05.

## Latency [D]

| Quantity | Naive | DRAG |
| --- | --- | --- |
| Mean s/q, reported over 200 | 4.86 | 28.23 (NQ uses its aggregate run time) |
| Per-question records: mean / median / p90 / p99 / max | 4.86 / 4.48 / 5.68 / 17.5 / 21.3 | 27.12 / 24.48 / 42.41 / 58.67 / 63.69 (n = 150; all 50 NQ timings are `null`) |
| Minimum | 3.19 | 13.20 |
| By dataset (s/q) | NQ 5.25 · TQA 5.15 · PopQA 5.00 · Hotpot 4.00 · 2Wiki 4.25 · SQA 5.94 | NQ 31.54 · TQA 28.65 · PopQA 22.32 · Hotpot 27.97 · 2Wiki 31.76 · SQA 25.60 |
| Startup, excluded above | 4–5.5 min per process | same |

**Time model [D].** OLS on the 150 timed DRAG questions:
`t ≈ 20.6 s + 13.4 s·1{pool changed} + 3.7 s·(extra queries)`, R² = 0.54. Group means:

| Final pool | Mean s/q | Timed questions |
| --- | ---: | ---: |
| Unchanged | 20.6 | 86 |
| One-for-one replacement | 34.2 | 37 |
| 2 queries | 37.1 | 22 |
| 3 queries | 42.8 | 5 |

## Call counts

- **Ours [C]:** `C = 3·Q_r + 3·R (+ Judge retries)`, where `Q_r` is the number of query
  rounds executed and `R = 3` response rounds.
- **Paper [P]** (StrategyQA): **10.36 LLM calls (3.36 query phase + 7 response phase) and
  2.08 retriever calls.** The final judge is defined once (Eq. 13), so
  `C_paper = 3·Q_r + 2·R + 1`.
- **Our lower bound [D].** Final pools are 124 unchanged, 46 one-for-one and 30
  multi-query, so `Q_r ≥ 1.38`, `C ≥ 13.14`, and there are ≥ 1.4 retrievals per question.

## Where the time likely goes [E]

E00 must measure each of these.

- **Retrieval ≈ 3 s per query.** Naive never finishes under 3.19 s, its time is unrelated
  to answer length (R² ≈ 0), and each extra DRAG query adds about 3.7 s. CPU
  `IndexFlatIP` reads ≈ 21M × 768 × 4 B ≈ 64.5 GB per query at `OMP_NUM_THREADS=12`.
- **The remaining ~23–24 s of DRAG is LLM time,** dominated by generation: six
  answer-stage agent turns per question at up to 256 tokens each.
- **Prefill is cheap at batch size 1:** ≈ 16 GFLOP/token × 1.5k tokens ≈ 24 TFLOP, about
  0.1–0.2 s.
- **The 2 dead moderator calls cost 0.4–4.3 s per question;** the measured split decides
  where in that range.

**Latency (s/question) vs throughput (questions/hour).** Batching across questions raises
throughput but barely changes one question's latency. Report the two separately.
