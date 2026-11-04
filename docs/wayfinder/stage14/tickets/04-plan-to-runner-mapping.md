---
title: Plan-to-runner mapping
label: wayfinder:grilling
status: closed
assignee: rishabh4496
blocked_by: [01-engine-inventory.md]
---

## Question

How does each Engine in a Pipeline plan compile onto the existing execution path (`render_mode`, enhancer selection, swapper choice in `WorkstationParameters`, `ReferencePipeline`/`MasterPipelineRunner`)? Which plan requests are not expressible and go into the `unsupported` list? What do the dry-run endpoint and orchestrated job submission look like at the API level?

## Resolution

Resolved by delegation ("complete stage 14, do what you think best"): the recommended defaults were adopted and implemented, not individually confirmed.

A plan compiles to `render_backend`, `face_restoration_policy` and `WorkstationParameters` fields (`render_mode`, `pose_mode`, `occlusion_mode`, `restorer_model`, `tps_expression_weight`, `restoration_blend`) in `orchestrator/compile.py`. Requests the runner would alter silently go into `unsupported` and make the plan structurally invalid, so they are never selected (reference runner always attaches the parser and retargets by default; stream backend lacks 3D, expression and occlusion; TensorRT is stream-only). Dry-run: `POST /api/v1/orchestrator/plan`, `GET /api/v1/orchestrator/engines`. Wiring job submission is not done (runner takes one `WorkstationParameters` per job); listed as a prerequisite in the spec.
