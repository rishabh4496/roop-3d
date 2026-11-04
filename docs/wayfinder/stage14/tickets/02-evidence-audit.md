---
title: Evidence audit
label: wayfinder:research
status: closed
assignee: rishabh4496
blocked_by: []
---

## Question

What performance and quality evidence exists today? Audit `.runtime/benchmarks/benchmark_database.json`, Stage 11 evaluation outputs, golden dataset coverage, precision parity reports, and `ModelRecord.benchmark_status` values. For each Engine-relevant Model/render profile and each hardware tier (cpu/6gb/12gb), state whether an exact-match record would exist under the settled staleness rule (all Model sha256, precision, provider, tier, render-profile definition). Identify which fields the benchmark record lacks (execution provider, render-profile definition, peak VRAM, latency) and where each could come from. Report the gap per Engine per tier. Facts only.

## Resolution

Closed by research agent; full findings in `research/02-evidence-audit.md` (facts only, file:line cited). Gist:

- The benchmark DB has 2 records (12gb, CUDA). Their sha256 values are 49/50 chars (not valid digests) and match no registry hash; they equal `record_run`'s keyword defaults. No production code reads the DB.
- The Stage 11 harness never loads a Model: each render profile is a pixel tint plus `time.sleep`. Its composite scores, latencies and vetoes describe simulated profiles, not Models, providers or tiers. The golden dataset is procedurally drawn.
- The only sha-keyed real evidence is per-Model qualification receipts in `.runtime/acceleration/{6gb,12gb}/` (10 Models per tier; provider, precision, parity, p50/p95 on one fixed golden input, 5 iterations). All passing records are FP32; FP16 was rejected for inswapper, w600k_r50 and liveportrait_appearance. `liveportrait_warping_spade` resolves to CPU.
- "6gb" evidence is a budget policy on a 12 GB RTX 4070, not a 6 GB card. No record anywhere is labelled cpu.
- No store holds a render-profile definition, so no record can satisfy the full staleness key.
- All 121 `ModelRecord`s are `benchmark_status: NOT_RUN` and nothing writes the field. The Stage 12 scan, best case, yields 0 READY (113 LICENSE_RESTRICTED, 8 FAILED), so the settled "Validated" precondition fails for every Model today. There is no per-Engine peak-VRAM measurement.
- Consequence for the map: under the settled rules, no Engine can be selected by Balanced or Quality today. Raised as ticket 08.
