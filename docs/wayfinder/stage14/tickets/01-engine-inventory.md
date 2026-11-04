---
title: Engine inventory
label: wayfinder:research
status: closed
assignee: rishabh4496
blocked_by: []
---

## Question

For each of the ten Stage 14 categories (face detection, face recognition, dense landmarks, identity swapping, 3D reconstruction, expression transfer, occlusion segmentation, restoration, video decoding, video encoding): which real, adapter-backed Engines exist today in `app/roop_3d/`? For each, record the Models it uses (ModelRecord ids / filenames), inputs and outputs expressed in a candidate closed vocabulary of data kinds, supported precisions and execution providers, known resolution limits, any declared or measured peak VRAM, heavy/light classification (adaptive_runtime), exclusivity hazards, and what fallback behavior exists in code today. List what is NOT backed by a real adapter (so it is excluded per the settled scope rule). Report facts only; no design decisions.

## Resolution

Closed by research agent; full findings in `research/01-engine-inventory.md` (284 lines, file:line cited, per-category tables for all ten categories). Gist:

- No per-capability Engine registry exists. The only "engine" registry is the job-runner one (`reference_onnx` plus two unavailable entries). Two render backends choose the adapters: the reference runner and the stream runner (no occlusion, no LivePortrait, no 3D modes in the stream path).
- Zero Models are Stage 12 READY (113 LICENSE_RESTRICTED, 8 FAILED). `ModelCatalog.resolve_for_inference` refused every installed weight probed with `LICENSE_UNVERIFIED`, so every neural Engine is "wired but refused by the gate" in this tree. Non-neural Engines (identity asset builder, pose blend, frontalization, TPS expression, reenactment, PyAV decode/encode, FFmpeg stream source/writer) are not subject to the gate.
- `ModelRecord` carries almost none of the registry fields: VRAM, precision requirements and supported precision are empty on all 121 records; providers set on only 4 (CPU). There is no role for dense landmarks, video codecs or 3D reconstruction. 16 sha256 groups have multiple record ids.
- No measured per-Engine peak VRAM exists. TensorRT receipts cover 10 Models and `AcceleratorFactory` is not called by any server or job code. `HEAVY_MODELS`/`LIGHT_MODELS` and `AdaptiveModelPool` have no production effect in jobs.
- Provides a candidate closed vocabulary of 19 data kinds (input to ticket 07) and four "not backed by a real adapter" groups: A no adapter code; B stand-ins that fabricate output; C real code nothing calls; D wired but refused by the READY gate.
- Undetermined: any measured per-Engine peak VRAM, production-environment parity with this `.venv`, physical 6 GB behaviour, HEVC intent, whether a Model can reach READY other than by editing manifests.
