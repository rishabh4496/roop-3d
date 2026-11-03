# Comprehension Audit Report

**Target:** `docs/designs/roop_3d_workstation_modernization.md` (Lines 1..320)
**Reviewer:** `egm-comprehension` (Naive Reader Persona)
**Date:** 2026-11-03

---

## What this proposes

The document proposes modernizing the Roop 3D workstation frontend and its server coordination by introducing a dark glassmorphic UI, interactive 3D spatial tools (Head Pose Gizmo, Spherical Harmonics lighting probe, and FLAME mesh viewer), and a 5-domain categorized Settings Inspector with client-side constraint validation. It addresses timeline scrub preview failures by replacing HTTP 409 cancellation errors with graceful HTTP 204 responses and debounced low-resolution proxy scrubbing, while eliminating HTTP 422 extraction errors via centroid-based spatial matching and relaxed payload validation. Furthermore, it establishes dedicated memory management profiles—a clamped 4,600 MiB sequential pipeline for 6 GB RTX 3060 GPUs and an unthrottled resident-engine pipeline for 12 GB RTX 4070 GPUs—to eliminate UI frame drops and WebGL context evictions.

## How the existing system works, per this document

Currently, the system uses a monolithic `SettingsInspector` that renders an unorganized, flat JSON schema list without functional grouping, real-time validation, or clear persistence feedback. During timeline scrubbing or parameter adjustments, the client uses `AbortController` to cancel in-flight `/api/preview` requests, which causes the backend to raise `DomainError("PREVIEW_CANCELLED", 409)` and display false user error banners, while media switching causes HTTP 422 schema rejections. Face extraction and bank discovery endpoints (`/faces/extract_from_frame` and `/faces/bank/discover`) fail with HTTP 422 errors due to rigid Pydantic validators (such as mandatory `faceset_name`) and face index desynchronization across decoded video frames. In addition, unconstrained CSS backdrop blurs and persistent WebGL draw loops run concurrently with memory-intensive PyTorch/TensorRT inference without hardware-specific VRAM tier budgets, leading to frame rate drops and GPU memory exhaustion on 6 GB consumer cards.

---

## Detailed Findings

### Finding 1: Unresolved `/faces/bank/discover` Contract Discrepancy
* **Severity**: Blocker
* **Affected Scope**: Section "The Problem" (line 16), Section "Goals" (line 23), Section 3.2 "Face Extraction & Bank Alignment" (lines 153-156), and Section Detailed Plan #4 (lines 235-241).
* **Discrepancy Description & Rationale**:
  The document identifies `/faces/bank/discover` contract discrepancies as a primary cause of HTTP 422 failures in Problem #2 and commits in the Goals section to "Resolve all HTTP 422 validation errors across `/api/preview`, `/faces/extract_from_frame`, and `/faces/bank/discover`". However, `/faces/bank/discover` is completely missing from the technical architecture diagram, Technical Plan Section 3.2, and the Detailed Implementation Plan (which modifies only `face_extract.py`). The document does not describe what the contract discrepancy is, what schema changes or Pydantic adjustments are needed for bank discovery, or which file implements the fix. A fresh reader is left with an unresolved blocker in the core problem statement.

### Finding 2: Undefined Architecture and Synchronization Contract for `WorkstationParameters`
* **Severity**: Major
* **Affected Scope**: Section CUJ 2 (line 54), Section 2.2 "Functional 3D Spatial Tools" (line 115), and Section Detailed Plan #6 (line 253).
* **Discrepancy Description & Rationale**:
  The design states that the 3D Head Pose Gizmo converts Euler rotations to Quaternions and synchronizes them bidirectionally with `WorkstationParameters.pose` and `useFaceSelectionStore`. However, `WorkstationParameters` is never defined. It is unclear whether `WorkstationParameters` is a TypeScript interface, a React context, a global Zustand store, an in-memory object, or a payload structure dispatched to `/api/preview`. Furthermore, the document does not explain how bidirectional synchronization between `useFaceSelectionStore` and `WorkstationParameters` avoids infinite update loops, or which entity acts as the single source of truth.

### Finding 3: Target Media Transition Contradiction and Undefined `inspection_id`
* **Severity**: Major
* **Affected Scope**: Section "Open Questions" (lines 40-41) vs. Section Detailed Plan #3 (`useFaceSelectionStore.ts`, lines 230-234).
* **Discrepancy Description & Rationale**:
  Open Question 2 asks whether switching target videos should attempt nearest-neighbor spatial matching on the first frame or reset completely. Yet, the Detailed Implementation Plan specifies modifying `useFaceSelectionStore.ts` to "drop tracks belonging to prior videos" whenever `inspection_id` changes, effectively deciding on an unconditional reset without acknowledging it in Open Questions. Additionally, `inspection_id` is introduced without definition: the document does not specify what generates or invalidates this identifier, what it scopes (a video file, an active session, or an extraction job), or how the frontend tracks its changes.

### Finding 4: Undefined FLAME 3D Asset Store & "Pose Coverage Confidence"
* **Severity**: Major
* **Affected Scope**: Section 1 Component Diagram (line 89), Section 2.2 (lines 118-121), and Section Detailed Plan #8 (lines 263-269).
* **Discrepancy Description & Rationale**:
  The component diagram introduces a "FLAME 3D Reconstruction Asset Store" and references `/api/v1/facesets/{id}/identity-asset/mesh` supplying a 5,023-vertex FLAME geometry with "per-vertex pose coverage confidence heatmaps". For an engineer without prior project knowledge, it is not explained how, when, or by what subsystem this FLAME mesh is generated (e.g. fitted during face extraction or computed asynchronously). Furthermore, "pose coverage confidence" is completely undefined: the document does not specify what mathematical metric or distribution this heatmap represents, what data schema the endpoint returns, or how the frontend component is expected to map those scalar values onto vertex colors.

### Finding 5: Missing Operational Mechanism for Hardware Profile Switching
* **Severity**: Major
* **Affected Scope**: Section "Open Questions" (lines 38-39), Section 2.3 (line 124), Section 3.3 (lines 157-161), and Section Detailed Plan #11 (lines 284-290).
* **Discrepancy Description & Rationale**:
  The technical plan defines two distinct hardware profiles (`6gb_low_vram` with sequential execution/cache eviction and `12gb_standard` with resident TensorRT engines). However, the document does not specify how profile transitions occur at runtime. It is unstated whether switching profiles requires an application/server restart, whether the inference engine can dynamically evict resident TRT models to switch to sequential execution, or what happens if a user selects `auto` in `ComputeSettingsSection`. The exact coordination protocol between `SettingsInspector.tsx`, `/api/v1/config`, and `server/engine_gate.py` remains unexplained.

### Finding 6: Undefined "2.5D TPS Fallback"
* **Severity**: Nit
* **Affected Scope**: Section 2.3 "Categorized Settings Inspector" (line 125).
* **Discrepancy Description & Rationale**:
  The document lists "2.5D TPS fallback thresholds" as a configurable setting under `Spatial3DSettingsSection` without defining TPS (Thin Plate Spline) or explaining what pipeline failure or condition triggers fallback from 3D FLAME fitting to 2.5D TPS warping.

### Finding 7: Unspecified Telemetry Source for TensorRT Build Status
* **Severity**: Nit
* **Affected Scope**: Section 2.3 (line 129) and Section Detailed Plan #5 (line 246).
* **Discrepancy Description & Rationale**:
  The plan specifies that client-side validation will disable `TensorrtExecutionProvider` when "TRT engines are unbuilt". However, while Section Detailed Plan #11 indicates `/api/v1/system/capabilities` will expose the active hardware profile, it does not confirm whether this endpoint (or another API) exposes the compiled/unbuilt status of TensorRT engines. The source of truth for this validation check is omitted.

### Finding 8: Ambiguous `--perf-mode` CSS Selector Syntax
* **Severity**: Nit
* **Affected Scope**: Section 2.4 (lines 143-145) and Section Detailed Plan #9 (line 275).
* **Discrepancy Description & Rationale**:
  The document states that `--glass-blur: none` is activated "when `--perf-mode: high`". In standard CSS, custom properties cannot be queried directly in media queries or selectors without container style queries. The document should clarify whether performance mode is set via a DOM attribute (e.g. `[data-perf-mode="high"]`) or a root class name.

### Finding 9: Inconsistent API Route Prefix Conventions
* **Severity**: Nit
* **Affected Scope**: Section 1 (lines 79-82), Section 2.2 (line 119), Section 3.2 (line 153), and Detailed Plan (lines 222, 236, 266, 288).
* **Discrepancy Description & Rationale**:
  Multiple disparate routing prefix conventions are mixed throughout the document: unversioned endpoints under `/api/preview`, unversioned root endpoints under `/faces/extract_from_frame`, and versioned endpoints under `/api/v1/config`, `/api/v1/facesets/...`, and `/api/v1/system/capabilities`. A brief note explaining whether this is legacy preservation or an intentional API partitioning strategy would prevent confusion for new developers.
