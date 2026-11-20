# Corrections log

Claims made in earlier drafts (AUDIT.md, the speed-track notes, the appended README
review) that turned out wrong or overstated, and how each was resolved. Add new rows
instead of rewriting old ones.

| Date | Claim | Resolution |
| --- | --- | --- |
| 2026-11-18 | 46/200 vs 53/200 questions lost original evidence | Both right: 53 lack the original query; 46 were fully replaced |
| 2026-11-18 | "Exactly 116/200 gold-in-context; caps accuracy at 58%" | 114–115/200 depending on normalisation; not a cap |
| 2026-11-18 | "At least 5 of 14" format-only Naive wins | 3 of 14; 5 is the count among DRAG's 24 wins |
| 2026-11-18 | `dev_1245` "jumped to hop 2 without resolving hop 1" | Hop 1 was resolved (Cassius Longinus); the original evidence was deleted |
| 2026-11-18 | Dead moderators cost ~4.3 s/q | Range 0.4–4.3 s; E00 decides |
| 2026-11-18 | "GPU FAISS OOMs on 80 GB" | Only for fp32 next to the default 0.85 vLLM reservation; FlashRAG's fp16 clone (~32 GB) should fit at ~0.45 [E] |
| 2026-11-18 | "Continuous batching is deterministic for greedy" | False; use the A/A floor (protocol.md §3) |
| 2026-11-18 | "< 3.5 s/q, > 8×, bit-identical"; "18 → 6 calls"; fixed EM uplifts (+2.5–4.5 pp, +6–10 pp on NQ, p < 0.01) | Removed: they mixed throughput with latency or had no support |
| 2026-11-18 | "~85% peer conformity" (Peacemaker or Troublemaker) | Not in the abstract; dropped |
| 2026-11-18 | vLLM issue #24394 | Could not verify; dropped (#63743 verified) |
| 2026-11-18 | FlashRAG `gpu_memory_utilization` default 0.9 | `main` uses 0.85, consistent with the observed 67–68 GiB |
| 2026-11-18 | DRAG paper authors unverified | Verified: Wentao Hu, Wengyu Zhang, Yiyang Jiang, Chen Jason Zhang, Xiao-Yong Wei, Li Qing |
| 2026-11-18 | "Stale query shown to the Judge" is a bug | A design question (B14) |
| 2026-11-18 | "124/200: the Judge picks Proponent at round 0" | Upper bound only (B16) |
| 2026-11-18 | `test_8583` "stores the identical 3 docs under 3 queries" | The 2nd query added 1 new passage; only the 3rd duplicated the 2nd |
| 2026-11-18 | "17/76 pools shared 2 of 3 IDs" | 17–20 depending on comparison method |
| 2026-11-18 | "Prefill of 1–2k tokens takes 30–80 ms on A100" | Arithmetic gives ~0.1–0.2 s |
| 2026-11-18 | "`compressed-tensors` enables W4A16" | It is the format library; a quantised checkpoint or tool is still needed [IC] |
| 2026-11-18 | Deck "could not be opened" (one draft) | Read directly: slides 9, 10, 17, 19–21 |
