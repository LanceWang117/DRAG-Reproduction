# References

Job research notes (`docs/research/<JOB-ID>-*.md`) cite sources directly. Add a source
here only when the ledger itself relies on it. Move an entry from "not re-verified" to
"verified" only after you fetched it and checked the cited claim; record the date.

## Verified (fetched or searched, 2026-11-18)

**DRAG and tooling**
- Hu et al. *Removal of Hallucination on Hallucination: Debate-Augmented RAG.* ACL 2025. https://aclanthology.org/2025.acl-long.770/ ; https://arxiv.org/abs/2505.18581 (§5.6 call counts, Eq. 13, Table 6, problem drift)
- FlashRAG (`main`): https://github.com/RUC-NLPIR/FlashRAG
- Upstream DRAG issues (4 closed; titles only): https://github.com/Huenao/Debate-Augmented-RAG/issues

**Inference engineering**
- Thinking Machines, *Defeating Nondeterminism in LLM Inference* (summary): https://simonwillison.net/2025/Sep/11/defeating-nondeterminism/
- vLLM batch invariance: https://docs.vllm.ai/en/latest/features/batch_invariance/ , https://github.com/vllm-project/vllm/issues/27433
- vLLM prefix caching: https://github.com/vllm-project/vllm/issues/5543 , https://github.com/vllm-project/vllm/issues/7845 , https://github.com/vllm-project/vllm/pull/65962 , https://github.com/vllm-project/vllm/issues/63743
- vLLM guided decoding: https://github.com/vllm-project/vllm/issues/12005 , https://github.com/vllm-project/vllm/issues/12122 , https://github.com/vllm-project/vllm/issues/11289
- Speculative decoding: https://github.com/vllm-project/vllm/issues/8439 , https://github.com/vllm-project/vllm/issues/16258 , https://github.com/Blackwellboy/model-serving-minefield/issues/174 , https://jarvislabs.ai/blog/speculative-decoding-vllm-faster-llm-inference , https://dev.to/ji_ai/vllm-speculative-decoding-18x-faster-alone-slower-at-32-users-5f16
- Kurtic et al. *"Give Me BF16 or Give Me Death"?* ACL 2025. https://aclanthology.org/2025.acl-long.1304/
- Guo, Zhang, Ren. *ACC-RAG.* Findings of EMNLP 2025. https://aclanthology.org/2025.findings-emnlp.1307/

**Multi-agent debate**
- Dai. *When Consensus Is Not Correctness.* KDD 2026 workshop. https://kdd-eval-workshop.github.io/agenticai-evaluation-kdd2026/assets/papers/49_When_Consensus_Is_Not_Corre.pdf
- Choi et al. *Debate or Vote.* NeurIPS 2025. https://arxiv.org/abs/2508.17536
- *Talk Isn't Always Cheap.* https://arxiv.org/abs/2509.05396
- *Free-MAD.* https://arxiv.org/abs/2509.11035
- *Demystifying Multi-Agent Debate.* https://arxiv.org/abs/2601.19921
- *Unanimity Without Persuasion.* https://arxiv.org/abs/2609.26145
- *Adaptive Stability Detection.* https://arxiv.org/abs/2510.12697
- *Wald-SPRT debates.* https://arxiv.org/abs/2605.19193
- Yao et al. *Peacemaker or Troublemaker.* https://arxiv.org/abs/2509.23055

**Judges, output formats, retrieval**
- Shi et al. *Judging the Judges.* IJCNLP-AACL 2025. https://aclanthology.org/2025.ijcnlp-long.18/
- Tam et al. *Let Me Speak Freely?* EMNLP 2024. https://arxiv.org/abs/2408.02442
- *CRANE.* https://arxiv.org/abs/2502.09061
- Jacob et al. *Drowning in Documents.* https://arxiv.org/abs/2411.11767
- *Not All Queries Need Rewriting.* https://arxiv.org/abs/2603.13301
- *How You Ask Matters!* https://arxiv.org/abs/2604.10745

**Knowledge conflicts, grounding, abstention**
- Wang et al. *Astute RAG.* ACL 2025. https://aclanthology.org/2025.acl-long.1476/
- *Don't Pick a Winner.* https://arxiv.org/abs/2610.36169
- Cattan et al. *DRAGged into Conflicts.* https://arxiv.org/abs/2506.08500
- Tang, Laban, Durrett. *MiniCheck.* EMNLP 2024. https://arxiv.org/abs/2404.10774
- Feng et al. *Conformal-RAG.* SIGIR 2025. https://arxiv.org/abs/2506.20978
- Joren et al. *Sufficient Context.* ICLR 2025. https://github.com/hljoren/sufficientcontext
- Moskvoretskii et al. *Adaptive Retrieval Without Self-Knowledge?* ACL 2025. https://aclanthology.org/2025.acl-long.319/

**Statistics**
- Card et al. *With Little Power Comes Great Responsibility.* EMNLP 2020. https://aclanthology.org/2020.emnlp-main.745/

## Not re-verified (check before citing)

- MAST taxonomy (arXiv 2503.13657)
- *Learning to Debate* (IEEE 2026)
- Self-Preference Bias (arXiv 2410.21819)
- Resolution Diagnostics (arXiv 2605.30315)
- Miller, *Adding Error Bars to Evals* (arXiv 2411.00640)
- RAG evaluation survey (arXiv 2504.14891)
- *Control the Temperature* (arXiv 2510.01218)
- vLLM #60044, #17324 and discussion #2443
- Adaptive-RAG (arXiv 2403.14403)
- Search-R1 (arXiv 2503.09516)
- Knowledgeable-R1 (arXiv 2506.05154)
- *Lost in the Middle*; RRF; MMR; Self-Consistency; Connor (1987); SimPO
