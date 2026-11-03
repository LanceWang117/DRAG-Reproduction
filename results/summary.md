# Results and comparison with the paper

EM and F1 below use a percentage scale. The paper values come from Table 1,
not a mean across methods or an appendix variant.

| Dataset | Our N | Paper Naive EM/F1 | Paper DRAG EM/F1 | Our Naive EM/F1 | Our DRAG EM/F1 |
| --- | ---: | --- | --- | --- | --- |
| NQ | 50 | 38.20 / 50.08 | 36.80 / 50.38 | 44.00 / 54.72 | 34.00 / 50.56 |
| TriviaQA | 10 | 60.80 / 69.55 | 60.80 / 69.93 | 60.00 / 67.25 | 70.00 / 72.86 |
| PopQA | 40 | 37.60 / 45.69 | 38.60 / 46.50 | 45.00 / 49.47 | 55.00 / 59.76 |
| 2Wiki | 40 | 14.80 / 24.27 | 28.80 / 36.97 | 15.00 / 18.00 | 35.00 / 38.13 |
| HotpotQA | 35 | 25.80 / 35.80 | 30.80 / 41.74 | 28.57 / 36.67 | 34.29 / 46.84 |
| StrategyQA | 25 | 62.60 / not reported | 69.20 / not reported | 64.00 / 64.00 | 64.00 / 64.00 |

DRAG minus Naive EM, in percentage points:

| Dataset | Paper | Our sample |
| --- | ---: | ---: |
| NQ | -1.40 | -10.00 |
| TriviaQA | 0.00 | +10.00 |
| PopQA | +1.00 | +10.00 |
| 2Wiki | +14.00 | +20.00 |
| HotpotQA | +5.00 | +5.71 |
| StrategyQA | +6.60 | 0.00 |

The multi-hop gains remain after expansion. NQ still favors Naive, while
StrategyQA ties instead of showing the paper's gain. This is partial agreement
in direction, not proof that the paper's quantitative results were reproduced.
Different questions, sample sizes, split choices, and our Judge fix limit the
comparison. For example, TriviaQA has only 10 questions: one answer changes EM
by 10 percentage points.

## Aggregate and timing

| Measure | Naive RAG | DRAG v1.0.1 |
| --- | ---: | ---: |
| EM over this 200-question mixture | 39.00 | 44.00 |
| Mean token F1 | 44.96 | 52.06 |
| Serial inference total, excluding initialization | 972.17s | 5645.53s |
| Mean inference time per question | 4.86s | 28.23s |

Of the 200 paired questions, 24 matched exactly only with DRAG and 14 only with
Naive. The remaining 162 had the same EM outcome. The net gain is 10 answers;
it does not mean DRAG preserves every correct Naive answer. No statistical
significance result is claimed. The overall scores weight NQ more heavily than
TriviaQA and reflect the targeted sampling decisions.

DRAG takes about 5.8 times as long on these serial runs. This supports examining
which questions benefit enough to justify debate, but no adaptive router has
been implemented or evaluated in this repository.

## Reading EM and F1

EM checks normalized string equality with any accepted reference answer.
Token F1 gives partial credit for overlapping words, with special handling for
yes/no/noanswer. It is an average score, not a pass percentage or a probability
of correctness. A longer but reasonable answer can receive a lower score. These
metrics alone do not establish a system-wide hallucination rate.

Source: [original paper, Table 1 and Section 4.1](https://aclanthology.org/2025.acl-long.770.pdf).
The paper sampled 500 questions from each dataset; our total is 200.
