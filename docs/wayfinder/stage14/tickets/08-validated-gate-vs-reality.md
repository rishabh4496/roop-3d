---
title: Validated gate versus current reality
label: wayfinder:grilling
status: closed
assignee: rishabh4496
blocked_by: []
---

## Question

The audit (ticket 02) found 0 of 121 Models reach Stage 12 lifecycle READY, no render-profile definition exists, "6gb" evidence is a budget policy on a 12 GB card, no cpu evidence exists, and the Stage 11 composite scores come from a simulated harness. Under the settled "Validated" rule, Balanced and Quality therefore always resolve to the baseline plan. What should Stage 14 do about it: ship the orchestrator as designed (auto modes inert until evidence and governance catch up), redefine what counts as validated evidence (for example receipts as the evidence source and treating the Stage 11 harness as non-evidence), declare prerequisite work (governance review, real benchmarks, a render-profile definition) out of scope but named in the spec, or something else? Also decide whether Manual mode is still permitted when its Models are not READY.

Added after ticket 01: the inventory independently confirms 0 READY and that `resolve_for_inference` refuses every installed neural weight (`LICENSE_UNVERIFIED`), so every neural Engine is refused at construction in this tree, not only excluded from auto selection. Non-neural Engines (pose blend, frontalization, TPS, reenactment, PyAV/FFmpeg decode and encode) are not gated.

## Resolution

Resolved by delegation; the Q1, Q3 and Q4 answers were given by the user during grilling, Q5-Q7 were adopted as recommended.

- Zero-READY is the intended fail-closed state (user, Q1). Governance completion is a named prerequisite, not Stage 14 work.
- Evidence must carry valid 64-hex hashes and `measured_with == real_models`; Stage 11 simulated records are not evidence (`EVIDENCE_NOT_REAL`) (user, Q3).
- Manual mode cannot override the Stage 12 gate; no force flag (user, Q4).
- Render-profile definitions are in the spec as `render_profiles.json`; evidence matches on the plan digest, which covers each Engine's entry digest, Model hashes, precision and provider (refinement of Q5).
- If even the baseline cannot run the status is `INFEASIBLE` with the full report; job submission is not offered (Q6).
- Acceptance is proven on fixture registries (shipped manifests with the lifecycle gate stubbed, synthetic evidence), plus one real-registry test for fail-closed and a byte-identical report (Q7).
