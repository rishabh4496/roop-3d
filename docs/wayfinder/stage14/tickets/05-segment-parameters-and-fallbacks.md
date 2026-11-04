---
title: Segment parameters and fallbacks
label: wayfinder:grilling
status: closed
assignee: rishabh4496
blocked_by: [01-engine-inventory.md]
---

## Question

What are the numeric and structural parameters for scene-aware stability: minimum dwell length, hysteresis rule, what forces a segment boundary beyond scene cuts, the initial scene-trait thresholds (frontal vs large yaw, low confidence), and the declared conservative fallback for each Engine category (including what a low-confidence frame does)?

## Resolution

Resolved by delegation: the Q1-Q4 defaults put to the user during grilling were adopted without confirmation and implemented.

- Segment = one `scene_id` shot. Dwell and hysteresis dropped: plans change only between shots. Refinement during implementation: all tracks in a shot share one plan (shared frames, residency and runner invocation).
- Low-confidence frames: the existing tracker withhold (original frame passes through), identical for every plan.
- Manifest `fallback`: `none` | `withhold` | `degrade_to` (target Engine in the same or an earlier category, closed reason codes). Current in-code degradations declared: CUDA allocation/OOM retry, NVDEC/NVENC to software, pose/asset/TPS/LivePortrait/reenactment to the proven swap crop, XSeg to BiSeNet. Degradation to a non-Engine code path (geometric hull mask) is described in text only.
- Trait: `large_yaw` at |yaw| extent >= 30 degrees (outer pose-cell nominal centre), from the inspection min/max, labelled a candidate estimate. A robust statistic (for example p95) would need an inspection change; left as fog.
