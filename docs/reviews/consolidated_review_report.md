# Consolidated Review Report: Roop 3D Workstation Modernization

**Target Document:** `docs/designs/roop_3d_workstation_modernization.md` (Lines 1..320)
**Execution Preset:** Consolidated report with all review rubrics (Uninterrupted Full Flow)
**Date:** 2026-11-03
**Review Swarm:** `egm-zoom-out`, `egm-comprehension`, `egm-design`, `egm-structural`, `egm-style`, `egm-implementation-readiness`, `egm-codebase-details`

---

## Executive Summary & Review Statistics

The uninterrupted multi-lens review evaluated `docs/designs/roop_3d_workstation_modernization.md` across high-altitude simplicity, naive reader comprehension, hostile system design, structural symmetry, stylistic discipline, implementation readiness, and pedantic codebase compatibility.

* 🔴 **Blockers:** 5
* 🟡 **Major Issues:** 13
* 🟢 **Nits / Quality Polish:** 7
* **Total Evaluated Findings:** 25

The design provides a compelling functional modernization strategy but contains critical implementation blockers in threadpool cancellation mechanics, missing Pydantic schema declarations, multi-context WebGL VRAM depletion, and performance test suite invariants.

---

## Prioritized Actionable Tasks

### 🔴 Blockers (Must Resolve Before Implementation)

#### B1: Synchronous Threadpool Cancellation Protocol (`_acquire_slot`)
* **Reviewer Lens:** `egm-codebase-details`
* **Affected Scope:** Lines 150–152 & Lines 223–225
* **Issue:** `_acquire_slot` is a synchronous helper executed inside Starlette's `run_in_threadpool`. Returning a FastAPI `Response(status_code=204)` from inside a worker thread does not send an HTTP response; `_compute` will continue into execution without acquiring `_slot` or throw a type error treating `Response` as `FrameOutcome`.
* **Action Required:** Raise a dedicated `PreviewCancelled` exception from `_acquire_slot` (or return `FrameOutcome(status="cancelled")`), caught by the async route handler `_single()`, which returns `Response(status_code=204, headers={"X-Preview-Status": "cancelled"})`.

#### B2: Missing `faceset_id` Field on Pydantic `ExtractRequest`
* **Reviewer Lens:** `egm-codebase-details`
* **Affected Scope:** Line 154 & Line 238
* **Issue:** In `app/roop_3d/server/face_extract.py`, `ExtractRequest` currently contains no `faceset_id` attribute. When the client sends `faceset_id`, Pydantic will fail validation or drop the field.
* **Action Required:** Explicitly declare `faceset_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")` on `ExtractRequest`, and update the `@model_validator` to verify `faceset_id or faceset_name` when `destination == "active_faceset"`.

#### B3: Unresolved `/faces/bank/discover` Contract Discrepancy
* **Reviewer Lens:** `egm-comprehension`
* **Affected Scope:** Line 16, Line 23, Lines 153–156, and Lines 235–241
* **Issue:** The document names `/faces/bank/discover` as a primary source of HTTP 422 failures in the Problem and Goals statements, but completely omits it from the architecture diagram, technical plan, and file modification list (which modifies only `face_extract.py`).
* **Action Required:** Specify the exact schema mismatch for bank discovery, enumerate `app/roop_3d/server/face_bank.py` in the file modification plan, and document the required validator relaxation.

#### B4: Concurrent Three.js WebGL Canvases & VRAM Depletion
* **Reviewer Lens:** `egm-design`
* **Affected Scope:** Lines 111–121 & Lines 249–268
* **Issue:** Concurrently mounting three separate `@react-three/fiber` viewports (`PoseGeometryViewer`, `LightProbeSphere`, `FlameMeshViewer`) alongside `WorkstationCanvas` / `WebGLViewport` allocates multiple active WebGL2 contexts. On an RTX 3060 (6 GB) during CUDA/TensorRT execution, this risks exceeding the 1,500 MiB compositor safety budget and triggering browser WebGL context loss.
* **Action Required:** Restructure the studio 3D tools into a unified, tabbed spatial inspector with lazy mount/unmount lifecycles, ensuring at most **one** auxiliary Three.js context is active at any time with explicit WebGL disposal (`renderer.forceContextLoss()`, `renderer.dispose()`) on tab transitions.

#### B5: Telemetry Stream Re-Render Invariant Breach in `SettingsInspector`
* **Reviewer Lens:** `egm-implementation-readiness`
* **Affected Scope:** Line 128 & Line 245
* **Issue:** Embedding live telemetry HUD toggles and active engine monitoring in `SettingsInspector.tsx` risks subscribing it to `telemetryStore`. Per `docs/UI_PERF_STAGE11.md`, `SettingsInspector` is instrumented with `renderProbe('SettingsInspector')`, and the automated test gate (`stage11Perf.test.tsx` and `telemetry.e2e.ts`) streams 90 telemetry frames at 30 Hz and strictly asserts **0 renders** of `SettingsInspector`.
* **Action Required:** Keep `SettingsInspector` strictly isolated from telemetry event sockets. Confine telemetry visualization to dedicated leaves (`GpuReadout`, `PipelinePill`), using settings only to update stored preferences.

---

### 🟡 Major Issues (Architectural & Technical Rigor)

#### M1: Non-Existent `WorkstationParameters.pose` Field
* **Reviewer Lens:** `egm-codebase-details`
* **Affected Scope:** Line 114 & Line 253
* **Action Required:** `WorkstationParameters` has fields for `pose_strength`, `pose_mode`, and `pose_frontalization`, but no `pose` vector. Update the plan to route gizmo rotation updates to `useFaceSelectionStore.focusPose` or a dedicated pose override store, avoiding invalid property assignments.

#### M2: `hardware_profile` Enum Renaming Breaks Schema Backward Compatibility
* **Reviewer Lens:** `egm-codebase-details`
* **Affected Scope:** Line 124, Lines 158–161, and Line 287
* **Action Required:** `app/roop_3d/core/settings.py` defines `hardware_profile: Literal["auto", "cpu", "6gb", "12gb"]`. Introducing `"6gb_low_vram"` and `"12gb_standard"` breaks stored configs. Retain canonical values `"6gb"` and `"12gb"`, using UI labels ("6 GB Low-VRAM" / "12 GB Standard") for display, or implement backward-compatible Pydantic alias normalization.

#### M3: Contract Mismatch on `GET /facesets/{id}/identity-asset/mesh`
* **Reviewer Lens:** `egm-codebase-details` & `egm-comprehension`
* **Affected Scope:** Lines 118–120 & Lines 266–267
* **Action Required:** `/identity-asset/mesh` returns vertices, faces, uvs, and texture, but no confidence values. Clarify that `FlameMeshViewer` fetches coverage metrics from `/facesets/{faceset_id}/coverage` and maps them onto vertices, or explicitly define the extension to the `/identity-asset/mesh` endpoint.

#### M4: Incompatible State Management Framing ("Zustand")
* **Reviewer Lens:** `egm-codebase-details`
* **Affected Scope:** Line 114 & Line 228
* **Action Required:** Zustand is not installed in the repository. Replace all mentions of Zustand with `useSyncExternalStore` / `createAtomicStore` to maintain dependency integrity.

#### M5: Ineffective Memory Eviction via `torch.cuda.empty_cache()`
* **Reviewer Lens:** `egm-implementation-readiness`
* **Affected Scope:** Line 160 & Line 287
* **Action Required:** PyTorch cache clearing does not affect ONNX Runtime or TensorRT BFCArena memory allocations. Enforce the 4,600 MiB limit using ONNX Runtime session options (`gpu_mem_limit`, `arena_extend_strategy=kSameAsRequested`) and explicit session deallocation.

#### M6: Wrong Concurrency Verification Test Target
* **Reviewer Lens:** `egm-codebase-details`
* **Affected Scope:** Line 314
* **Action Required:** `test_stage14_workstation.py` is for checkpoint/chunk uploads and does not test live preview. Update the verification plan to target `app/roop_3d_ui/src/liveSwapPreview.test.tsx` and a dedicated test in `tests/test_live_preview.py`.

#### M7: Unresolved Architectural Open Questions
* **Reviewer Lens:** `egm-zoom-out`
* **Affected Scope:** Lines 37–41
* **Action Required:** Open Questions 1 and 2 must be formally resolved in the design: (1) Default to automatic hardware detection at startup with manual override in Settings; (2) Automatically purge stale track mappings upon target media transition.

#### M8: High Coupling of Concurrency Fixes with Aesthetic Styling
* **Reviewer Lens:** `egm-zoom-out`
* **Affected Scope:** Lines 9–19 & Lines 70–100
* **Action Required:** Re-align the design document with strict phased staging: Phase 1: Concurrency & Contract Fixes; Phase 2: Memory & Hardware Profiles; Phase 3: Settings De-monolithing; Phase 4: 3D Spatial Tools; Phase 5: Glassmorphic Styling.

#### M9: Undefined Kelvin Blackbody-to-SH Relighting Pipeline
* **Reviewer Lens:** `egm-design`
* **Affected Scope:** Line 117 & Line 260
* **Action Required:** The relighting engine accepts `manual_light_yaw` and `manual_light_pitch`, but no Kelvin parameter. Specify the blackbody-to-RGB conversion algorithm and detail how Kelvin temperature modulates the ambient $L_0^0$ and directional SH coefficients.

#### M10: Missing Settings Section-to-Schema Mapping Table
* **Reviewer Lens:** `egm-structural`
* **Affected Scope:** Lines 122–130 & Lines 242–248
* **Action Required:** Provide an explicit table mapping each of the 28 `Settings` schema keys to its designated domain section to prevent dropping configuration options during tab restructuring.

#### M11: Unbounded Spatial Centroid Distance Matching Thresholds
* **Reviewer Lens:** `egm-implementation-readiness`
* **Affected Scope:** Lines 154–156 & Lines 239–240
* **Action Required:** Define a maximum normalized distance cutoff radius ($d_{\max} \le 0.15 \times \text{frame diagonal}$) beyond which a face is considered unmapped, preventing incorrect swaps in crowded scenes.

#### M12: Undefined Architecture and Synchronization Contract for `WorkstationParameters`
* **Reviewer Lens:** `egm-comprehension`
* **Affected Scope:** Line 54, Line 115, and Line 253
* **Action Required:** Define `WorkstationParameters` as the typed parameter object in `components/workstation/pipeline.ts` and document the unidirectional flow from UI manipulators to parameters to avoid infinite update loops.

#### M13: Missing Operational Mechanism for Hardware Profile Switching
* **Reviewer Lens:** `egm-comprehension`
* **Affected Scope:** Line 38, Line 124, Lines 157–161, and Lines 284–290
* **Action Required:** Specify whether switching between `6gb` and `12gb` requires a server restart or dynamically reloads engine providers, and document the coordination protocol with `server/engine_gate.py`.

---

### 🟢 Nits & Quality Polish

#### N1: Inaccurate File Modification Status for `LightProbeSphere.tsx`
* **Scope:** Line 256. `LightProbeSphere.tsx` does not exist in `components/studio/`; mark as `(Create)`.

#### N2: Dark-Biased Glass Tokens Degrade Readability in Light Themes
* **Scope:** Lines 134–142 & Lines 271–283. Update `deriveThemeVars` to compute light-glass tokens (`rgba(255, 255, 255, 0.75)`) when `mode === 'light'`.

#### N3: Missing Asynchronous Preview Scrub Cancellation Sequence Diagram
* **Scope:** Lines 70–100. Add a Mermaid sequence diagram modeling client scrub dispatch, `AbortController.abort()`, backend slot release, and HTTP 204 receipt.

#### N4: Ambiguous `--perf-mode` CSS Selector Syntax
* **Scope:** Lines 143–145 & Line 275. Specify that performance mode is set via a root DOM attribute (`[data-perf-mode="high"]`).

#### N5: Undefined "2.5D TPS Fallback"
* **Scope:** Line 125. Define TPS (Thin Plate Spline) and explain that fallback triggers when FLAME 3D landmark optimization fails to converge.

#### N6: Unspecified Telemetry Source for TensorRT Build Status
* **Scope:** Line 129 & Line 246. Confirm that `/api/v1/system/capabilities` exposes `caps.models[*].state` and TRT build status.

#### N7: Inconsistent API Route Prefix Conventions
* **Scope:** Lines 79–82, 119, 153, 222, 236, 266, 288. Add a note clarifying legacy unversioned routes (`/api/preview`, `/faces/...`) vs. versioned routes (`/api/v1/...`).

---

## Traceability Matrix

| ID | Title | Priority | Reviewer Lens | Target Section in Design |
| :--- | :--- | :--- | :--- | :--- |
| **B1** | Synchronous Cancellation Return Type in `_acquire_slot` | 🔴 Blocker | `egm-codebase-details` | Section 3.1 & Detailed Plan #2 |
| **B2** | Missing `faceset_id` on Pydantic `ExtractRequest` | 🔴 Blocker | `egm-codebase-details` | Section 3.2 & Detailed Plan #4 |
| **B3** | Unresolved `/faces/bank/discover` Contract Discrepancy | 🔴 Blocker | `egm-comprehension` | Section 3.2 & Detailed Plan #4 |
| **B4** | Concurrent Three.js WebGL Canvases & VRAM Depletion | 🔴 Blocker | `egm-design` | Section 2.2 & Detailed Plan #6, #7, #8 |
| **B5** | Telemetry Stream Re-Render Invariant Breach | 🔴 Blocker | `egm-implementation-readiness` | Section 2.3 & Detailed Plan #5 |
| **M1** | Non-Existent `WorkstationParameters.pose` Field | 🟡 Major | `egm-codebase-details` | Section 2.2 & Detailed Plan #6 |
| **M2** | `hardware_profile` Enum Renaming Backward Compatibility | 🟡 Major | `egm-codebase-details` | Section 2.3, 3.3 & Detailed Plan #11 |
| **M3** | Contract Mismatch on `GET /facesets/{id}/identity-asset/mesh` | 🟡 Major | `egm-codebase-details`, `egm-comprehension` | Section 2.2 & Detailed Plan #8 |
| **M4** | Incompatible State Management Framing ("Zustand") | 🟡 Major | `egm-codebase-details` | Section 2.2 & Detailed Plan #3 |
| **M5** | Ineffective Memory Eviction via `torch.cuda.empty_cache()` | 🟡 Major | `egm-implementation-readiness` | Section 3.3 & Detailed Plan #11 |
| **M6** | Wrong Concurrency Verification Test Target | 🟡 Major | `egm-codebase-details` | Verification Plan |
| **M7** | Unresolved Architectural Open Questions | 🟡 Major | `egm-zoom-out` | Open Questions |
| **M8** | High Coupling of Concurrency Fixes with Aesthetic Styling | 🟡 Major | `egm-zoom-out` | Overview, The Problem, Staging |
| **M9** | Undefined Kelvin Blackbody-to-SH Relighting Pipeline | 🟡 Major | `egm-design` | Section 2.2 & Detailed Plan #7 |
| **M10** | Missing Settings Section-to-Schema Mapping Table | 🟡 Major | `egm-structural` | Section 2.3 & Detailed Plan #5 |
| **M11** | Unbounded Spatial Centroid Distance Matching Thresholds | 🟡 Major | `egm-implementation-readiness` | Section 3.2 & Detailed Plan #4 |
| **M12** | Undefined Architecture for `WorkstationParameters` | 🟡 Major | `egm-comprehension` | CUJ 2, Section 2.2 & Detailed Plan #6 |
| **M13** | Missing Mechanism for Hardware Profile Switching | 🟡 Major | `egm-comprehension` | Open Questions, Section 2.3, 3.3, #11 |
| **N1** | Inaccurate File Modification Status for `LightProbeSphere.tsx` | 🟢 Nit | `egm-codebase-details` | Detailed Plan #7 |
| **N2** | Dark-Biased Glass Tokens Degrade Light Themes | 🟢 Nit | `egm-design` | Section 2.4 & Detailed Plan #9, #10 |
| **N3** | Missing Preview Scrub Cancellation Sequence Diagram | 🟢 Nit | `egm-structural` | Architecture Component Diagram |
| **N4** | Ambiguous `--perf-mode` CSS Selector Syntax | 🟢 Nit | `egm-comprehension` | Section 2.4 & Detailed Plan #9 |
| **N5** | Undefined "2.5D TPS Fallback" | 🟢 Nit | `egm-comprehension` | Section 2.3 |
| **N6** | Unspecified Telemetry Source for TensorRT Build Status | 🟢 Nit | `egm-comprehension` | Section 2.3 & Detailed Plan #5 |
| **N7** | Inconsistent API Route Prefix Conventions | 🟢 Nit | `egm-comprehension` | Section 1, 2.2, 3.2 & Detailed Plan |

---

*Consolidated Review complete. Deliverable ready for remediation via `/egm-resolve`.*
