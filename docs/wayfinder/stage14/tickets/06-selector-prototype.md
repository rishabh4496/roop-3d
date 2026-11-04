---
title: Selector prototype on real data
label: wayfinder:prototype
status: closed
assignee: rishabh4496
blocked_by: [01-engine-inventory.md, 02-evidence-audit.md, 03-registry-manifest-shape.md]
---

## Question

Run a throwaway selector sketch against the real Engine inventory and benchmark evidence. What do Manual, Balanced and Quality actually choose per profile (cpu/6gb/12gb)? Does anything beyond the baseline qualify, and does the decision report read clearly with real rejections? Link the prototype as an asset; do not merge it.

## Resolution

Superseded by implementation: the selector exists in `app/roop_3d/orchestrator/selector.py` and ran against the real registry. Result on the shipped data: Manual, Balanced and Quality are all `INFEASIBLE` on every tier because no Model is `READY` and no real evidence exists; the report names the cause per Engine. Nothing beyond the baseline can qualify until the prerequisites in the spec are met. Throwaway prototype not needed.
