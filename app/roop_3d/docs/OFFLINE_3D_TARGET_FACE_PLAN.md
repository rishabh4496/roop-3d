# Offline 3D-Aware Target Face Replacement — Implementation Plan

Status: design specification only. No production inference behavior is changed by this document.

## Goal

Provide an offline, local-video workflow that replaces one explicitly selected target identity with a user-authorized source FaceSet, preserves the target's motion and expression, and uses 3D-aware geometry/visibility/occlusion where available. Never infer that the LiveFaceSwap.ai service's private model architecture is known or copy proprietary weights/code.

## Reference-site findings

The public LiveFaceSwap.ai description documents a live camera + portrait-reference workflow whose main AI inference runs in the cloud and returns a live output. Its public materials do not specify a 3D reconstruction model, mesh representation, Gaussian splatting, FLAME fitting, or offline video export pipeline. This project should reproduce desired user-facing behavior—not claim compatibility with an undocumented internal implementation.

Reference: https://livefaceswap.ai/
Related public description: https://livefaceswap.ai/what-is-live-face-swap

## Existing Roop 3D components to reuse

- `engine/detection/face_tracker.py`: multi-face tracking and stable track IDs.
- `engine/volumetric/identity_swapper.py`: local source identity transfer with pose-aware FaceSet view selection.
- `engine/volumetric/reference_adapter.py`: optional LivePortrait motion/appearance retargeting.
- `engine/volumetric/session.py`: frame-level orchestration and selected track filtering.
- `reconstruction/geometry_types.py`, `flame_fitter.py`, `tps_fallback.py`, `pose_blend.py`, `uv_synthesizer.py`: geometry, optional license-gated parametric fitting, 2.5D fallback and UV texture support.
- `compositor/pipeline.py`, `occlusion.py`, `temporal.py`, `stream_restoration.py`: blending, occlusion handling and temporal stabilization.
- `media/service.py`, `engine/pipeline_runner.py`, `jobs/worker.py`: inspection, local video I/O and queued processing.
- Existing UI API: `POST /api/v1/media/inspect-faces`, `POST /api/v1/media/detect-frame-faces`, and saved inspection / face-mapping job contracts.

These modules must be inspected and their actual call paths verified before writing production code. A file's existence is not proof that its model weights are installed or that its path is used for final video rendering.

## Product workflow

1. Select an offline video file.
2. Run an inspection pass and show each detected identity with representative thumbnails, first/last visible timestamps, face count, confidence and track continuity.
3. User explicitly selects one or more target tracks; no automatic identity selection. Require a source FaceSet mapping for every selected track.
4. Persist the selection against the inspected video's SHA-256. Reject stale selections if the video changes.
5. Show a short, user-approved preview range before full rendering.
6. Process locally with pause/resume/cancel and checkpoint recovery.
7. Export video + audio with source timing, frame count and audio synchronization preserved.
8. Make synthetic-media provenance visible and require the operator to confirm rights/consent for the source identity.

## Proposed architecture

### A. Target identity lock and tracking
- Persist an immutable mapping: target track ID -> source FaceSet ID.
- Use detector + tracker association with embeddings, landmarks, pose and motion; do not use frame-local detection order as identity.
- Add track confidence and explicit LOST / REACQUIRED states.
- On low confidence, do not swap a different face. Hold the last valid track state briefly, then leave the frame unchanged and report the gap.
- Handle cuts by resetting temporal state and re-identifying selected identities only against the saved target reference, then require a confidence threshold.
- Support multiple selected targets by independent source mapping, session state and resource budget.

### B. 3D-aware reconstruction and reenactment
- Always-available baseline: PnP pose + dense landmarks + depth/shape prior + TPS/piecewise-affine 2.5D deformation.
- Optional parametric path: FLAME only when the exact licensed model asset and license state are verified. Never bundle restricted FLAME weights.
- Estimate target pose, expression coefficients, jaw/eye motion, camera intrinsics and per-frame visibility.
- Build a source identity texture/UV atlas only from observed, high-confidence source views. Mark unseen regions as unknown, never fabricate them as observed.
- Retarget target expression and pose to source identity. Use LivePortrait as an optional learned reenactment adapter, not as a substitute for explicit target identity tracking.
- Keep reconstruction state temporally stable with robust landmark losses and bounded smoothing. Reset on scene cuts and identity reacquisition.
- Do not promise physically correct hidden-side reconstruction from a single portrait. Show confidence and use conservative fallback.

### C. Occlusion-aware compositing
- Combine semantic face parsing, dense landmarks, depth ordering, temporal occluder masks and boundary confidence.
- Preserve foreground objects (hands, hair, glasses, microphones) in front of the face.
- Stabilize mask edges temporally without smearing mouth/eyes or retaining stale masks after cuts.
- Match local illumination and color in linear/light-aware space where supported; preserve grain and background pixels.
- Fall back safely to current compositor or no replacement where masks/geometry are uncertain.

### D. Offline video pipeline
- Decode with PyAV/FFmpeg and use NVDEC only when verified. Encode with NVENC only when available; otherwise use software.
- Cache detection/tracking/landmarks/pose/visibility in bounded frame-aligned checkpoints.
- Run inference in batches where graph contracts support it; avoid repeatedly initializing ONNX sessions or recomputing source features.
- Keep decode, inference, compositor and encode queues bounded with backpressure.
- Preserve VFR timestamps, audio, color metadata and rotation; validate output duration/frame timestamps/audio sync.
- Resume only from a checkpoint bound to video hash, source FaceSet hashes, model hashes, settings hash and pipeline version.
- Enforce explicit VRAM budgets for RTX 4070 12 GB and RTX 3060 Laptop 6 GB; use lower-resolution/temporal modes on constrained hardware.

## Performance gate

The checked-in Stage 15 benchmark artifact reports, for an RTX 4070 run, 0.07 completed output FPS and a retarget p50 near 7.14 seconds per frame, with restoration and compositing adding further substantial latency. Treat this as a regression to diagnose, not as a validated real-world target. Before adding heavier 3D inference:
- Profile LivePortrait model calls individually (motion, appearance, warp, stitching), device/provider, shape conversions, synchronization and copies.
- Verify ONNX execution providers are actually selected and no unexpected CPU fallback occurs.
- Avoid repeated dense landmark inference and per-frame feature/session preparation.
- Benchmark warm and steady-state separately on representative 1080p clips (front, profile, expressions, occlusions, cuts, multiple faces).
- Report p50/p95 latency, output FPS, VRAM/RAM peak, quality metrics and hardware/provider. Do not compare unlike clips or claim improvement without measured data.

## Milestones and acceptance

### Milestone 1 — Audit and baseline (no behavior change)
- Trace UI -> inspection -> target mapping -> JobRequest -> worker -> pipeline -> frame compositor.
- Identify whether current job path actually calls ReferencePipeline and whether LivePortrait weights/model contracts are present.
- Add integration tests proving only explicitly mapped target track(s) are replaced.
- Record reproducible 4070 baseline and 3060 Laptop plan.

### Milestone 2 — Target lock UX/API
- Face gallery and track selection; source FaceSet mapping; identity lock persisted against video hash.
- Tests: two similar-looking faces, crossing paths, face disappears/reappears, scene cut, selected ID absent, stale inspection.

### Milestone 3 — Temporal 3D-aware inference
- Integrate validated geometry/pose/expression/visibility data into the actual offline render path.
- Optional learned reenactment; deterministic 2.5D fallback.
- Tests: yaw/pitch/roll extremes, mouth/eye expression, profile, occlusion, missing landmarks, uncertain depth.

### Milestone 4 — High-fidelity compositing
- Occlusion/depth-aware masks and temporally stable face boundary; foreground object preservation.
- Golden clips include hand-over-face, glasses, hair, microphone, two people interacting and low light.

### Milestone 5 — Optimization and release
- Profile and optimize without quality regression.
- Full suite, model SHA verification, legacy isolation audit and unmocked golden video validation.
- Produce separate RTX 4070 and RTX 3060 Laptop results.
- Keep new mode feature-flagged until acceptance gates pass.

## Safety, privacy and compliance
- Local-only processing by default; no remote upload or hidden network transfer.
- Require the operator to confirm permission to use source and target media.
- Do not enable impersonation for fraud, deception, harassment or non-consensual intimate imagery.
- Preserve provenance/watermark metadata and a clear synthetic-media disclosure option.
- Respect model licenses, attribution and commercial restrictions. Do not reuse or redistribute third-party weights without rights.

## Non-goals
- Do not claim to reverse engineer LiveFaceSwap.ai.
- Do not scrape or copy private website code/models.
- Do not use Gaussian splatting merely as a marketing label; only add it if the input capture, multi-view data, rendering quality and compute budgets justify it. For ordinary monocular video, a robust 2.5D/parametric face model is usually the more practical starting point.
- Do not silently replace the existing identity-swap path or modify Roop Ultimate.
