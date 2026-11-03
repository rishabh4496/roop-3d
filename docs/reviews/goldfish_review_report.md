# Goldfish Annotated Review Report

**Target:** `docs/designs/roop_3d_workstation_modernization.md`
**Evaluator:** Elephant (Primary Session Context)
**Date:** 2026-11-03
**Status:** resolved

> **Resolution Summary:** All 25 findings (5 Blockers, 13 Majors, 7 Nits) have been accepted and incorporated into [`roop_3d_workstation_modernization.md`](file:///g:/pinokio/api/roop-3d/docs/designs/roop_3d_workstation_modernization.md) and [`design_draft.md`](file:///g:/pinokio/api/roop-3d/design_draft.md). The design status has transitioned from `Proposed` to `Approved`.

---

## Prioritized Findings & AI Suggestions

### B1: Synchronous Cancellation Return Type in `_acquire_slot` [`egm-codebase-details`] - Severity: Blocker
* **Affected Scope:** Lines 150–152 (Section 3.1) & Lines 223–225 (Detailed Plan Item #2)
* **Discrepancy Description:** Returning `Response(status_code=204)` from `_acquire_slot` inside Starlette's `run_in_threadpool` does not send an HTTP response; `_compute` will continue into execution or throw a type error treating `Response` as `FrameOutcome`.
* **Rationale:** `_acquire_slot` is synchronous. Route responses must originate in asynchronous FastAPI route handlers.
* ***AI Suggestion:*** PROPOSE FIX
* ***Justification:** The critique is accurate. `_acquire_slot` should raise a dedicated `PreviewCancelled` exception caught by async `_single()`, which returns `Response(status_code=204)`.
* ***Proposed Edit:***
  * Location: Section 3.1 and Detailed Plan Item #2
  * Change: Specify that `_acquire_slot()` raises `PreviewCancelled` on client disconnect. The async route handler `_single()` in `server/live_preview.py` catches `PreviewCancelled` and returns `Response(status_code=204, headers={"X-Preview-Status": "cancelled"})`.

### B2: Missing `faceset_id` Field on Pydantic `ExtractRequest` [`egm-codebase-details`] - Severity: Blocker
* **Affected Scope:** Line 154 (Section 3.2) & Line 238 (Detailed Plan Item #4)
* **Discrepancy Description:** `app/roop_3d/server/face_extract.py` does not declare `faceset_id` on `ExtractRequest`.
* **Rationale:** Pydantic will drop or fail validation when the client supplies `faceset_id`.
* ***AI Suggestion:*** PROPOSE FIX
* ***Justification:** Codebase verification confirms `ExtractRequest` lacks `faceset_id`. Adding it is required for contract alignment.
* ***Proposed Edit:***
  * Location: Section 3.2 and Detailed Plan Item #4
  * Change: Explicitly declare `faceset_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")` on `ExtractRequest` in `server/face_extract.py`, updating `@model_validator` to verify `faceset_id or faceset_name` when `destination == "active_faceset"`.

### B3: Unresolved `/faces/bank/discover` Contract Discrepancy [`egm-comprehension`] - Severity: Blocker
* **Affected Scope:** Line 16 (The Problem), Line 23 (Goals), Lines 153–156 (Section 3.2), and Lines 235–241 (Detailed Plan Item #4)
* **Discrepancy Description:** `/faces/bank/discover` was committed as a goal to fix 422 errors, but omitted from technical plan and file changes.
* **Rationale:** Leaves a core problem statement unaddressed in the implementation plan.
* ***AI Suggestion:*** PROPOSE FIX
* ***Justification:** Server logs show `/api/v1/faces/bank/discover` returning 422. We must document the fix in `server/face_bank.py`.
* ***Proposed Edit:***
  * Location: Section 3.2 and Detailed Plan
  * Change: Add `app/roop_3d/server/face_bank.py` to target files. Relax discovery request validation: when `target_path` is provided, if `inspection_id` is omitted or stale, look up the latest inspection for `target_path` rather than throwing HTTP 422.

### B4: Concurrent Three.js WebGL Canvases & VRAM Depletion [`egm-design`] - Severity: Blocker
* **Affected Scope:** Lines 111–121 (Section 2.2) & Lines 249–268 (Detailed Plan Items #6, #7, #8)
* **Discrepancy Description:** Concurrently mounting three separate `@react-three/fiber` canvases risks WebGL context loss on 6 GB RTX 3060 during inference.
* **Rationale:** Multiple active WebGL2 framebuffers exhaust browser GPU memory headroom.
* ***AI Suggestion:*** PROPOSE FIX
* ***Justification:** Consolidating spatial tools into a single tabbed inspector ensures only one auxiliary WebGL context is active.
* ***Proposed Edit:***
  * Location: Section 2.2 and Detailed Plan Items #6–#8
  * Change: Unify spatial tools into `components/studio/SpatialInspector.tsx` with tabbed subcomponents (`HeadPoseTab`, `LightingProbeTab`, `FlameMeshTab`). Ensure lazy mounting and call `renderer.forceContextLoss()` / `renderer.dispose()` on tab unmount.

### B5: Telemetry Stream Re-Render Invariant Breach in `SettingsInspector` [`egm-implementation-readiness`] - Severity: Blocker
* **Affected Scope:** Line 128 (Section 2.3) & Line 245 (Detailed Plan Item #5)
* **Discrepancy Description:** Embedding live telemetry HUD toggles and active engine monitoring in `SettingsInspector.tsx` threatens Stage 11 zero-render invariants.
* **Rationale:** `stage11Perf.test.tsx` strictly asserts 0 re-renders of `SettingsInspector` during 30 Hz telemetry streaming.
* ***AI Suggestion:*** PROPOSE FIX
* ***Justification:** The critique is accurate. `SettingsInspector` must not subscribe to telemetry streams.
* ***Proposed Edit:***
  * Location: Section 2.3 and Detailed Plan Item #5
  * Change: Explicitly specify that `SettingsInspector` remains decoupled from `telemetryStore`. Telemetry displays remain confined to leaf components (`GpuReadout`, `PipelinePill`). Settings toggles only read/write static preferences.

---

### M1: Non-Existent `WorkstationParameters.pose` Field [`egm-codebase-details`] - Severity: Major
* **Affected Scope:** Line 114 (Section 2.2) & Line 253 (Detailed Plan Item #6)
* **Discrepancy Description:** `WorkstationParameters` has no `pose` vector.
* **Rationale:** Attempting to assign `WorkstationParameters.pose` causes TypeScript errors.
* ***AI Suggestion:*** PROPOSE FIX
* ***Justification:** Rotation angles should route to `useFaceSelectionStore.focusPose` and pass via `pose_override` in preview requests.
* ***Proposed Edit:***
  * Location: Section 2.2 and Detailed Plan Item #6
  * Change: Replace `WorkstationParameters.pose` with `useFaceSelectionStore.focusPose` and document passing `pose_override: { yaw, pitch, roll }` in preview payloads.

### M2: `hardware_profile` Enum Renaming Breaks Schema Backward Compatibility [`egm-codebase-details`] - Severity: Major
* **Affected Scope:** Line 124, Lines 158–161, and Line 287
* **Discrepancy Description:** `settings.py` defines `hardware_profile: Literal["auto", "cpu", "6gb", "12gb"]`.
* **Rationale:** Renaming values to `"6gb_low_vram"` and `"12gb_standard"` breaks existing stored configs.
* ***AI Suggestion:*** PROPOSE FIX
* ***Justification:** Retaining canonical values `"6gb"` and `"12gb"` in the schema preserves backward compatibility while allowing rich UI labels.
* ***Proposed Edit:***
  * Location: Sections 2.3, 3.3, and Detailed Plan Item #11
  * Change: Retain canonical schema enum `Literal["auto", "cpu", "6gb", "12gb"]` in `settings.py` and `/api/v1/config`. Map display labels ("6 GB Low-VRAM" / "12 GB Standard") in the UI.

### M3: Contract Mismatch on `GET /facesets/{id}/identity-asset/mesh` [`egm-codebase-details`, `egm-comprehension`] - Severity: Major
* **Affected Scope:** Lines 118–120 & Lines 266–267
* **Discrepancy Description:** `/identity-asset/mesh` returns mesh geometry but no confidence scores.
* **Rationale:** Coverage metrics reside on `/facesets/{faceset_id}/coverage`.
* ***AI Suggestion:*** PROPOSE FIX
* ***Justification:** Documenting the dual-endpoint fetch ensures clear implementation.
* ***Proposed Edit:***
  * Location: Section 2.2 and Detailed Plan Item #8
  * Change: Clarify that `FlameMeshViewer` retrieves geometry from `/identity-asset/mesh` and pose coverage metrics from `/facesets/{id}/coverage`, mapping coverage cell scores onto corresponding facial mesh vertex regions.

### M4: Incompatible State Management Framing ("Zustand") [`egm-codebase-details`] - Severity: Major
* **Affected Scope:** Line 114 & Line 228
* **Discrepancy Description:** References to Zustand contradict repository dependency rules.
* **Rationale:** State stores use `useSyncExternalStore` and `createAtomicStore`.
* ***AI Suggestion:*** PROPOSE FIX
* ***Justification:** Replace all occurrences of "Zustand" with "Custom `useSyncExternalStore` Atomic Store".
* ***Proposed Edit:***
  * Location: Line 114 and Line 228
  * Change: Replace "Zustand/Atomic State Store" with "Atomic State Store (`useSyncExternalStore`)".

### M5: Ineffective Memory Eviction via `torch.cuda.empty_cache()` [`egm-implementation-readiness`] - Severity: Major
* **Affected Scope:** Line 160 & Line 287
* **Discrepancy Description:** PyTorch cache clearing does not affect ONNX Runtime or TensorRT memory arenas.
* **Rationale:** ONNX Runtime allocates device memory through its own BFCArena.
* ***AI Suggestion:*** PROPOSE FIX
* ***Justification:** Enforce memory limits through ONNX Runtime session options (`gpu_mem_limit`, `arena_extend_strategy`).
* ***Proposed Edit:***
  * Location: Section 3.3 and Detailed Plan Item #11
  * Change: Specify configuring ONNX Runtime session options with `gpu_mem_limit: 4823449600` and `arena_extend_strategy: kSameAsRequested` alongside explicit session unbinding on the 6 GB profile.

### M6: Wrong Concurrency Verification Test Target [`egm-codebase-details`] - Severity: Major
* **Affected Scope:** Line 314
* **Discrepancy Description:** `test_stage14_workstation.py` does not test live preview scrub.
* **Rationale:** Pointing to the wrong test file creates false verification.
* ***AI Suggestion:*** PROPOSE FIX
* ***Justification:** Update the verification plan to target `app/roop_3d_ui/src/liveSwapPreview.test.tsx` and `app/roop_3d/tests/test_live_preview.py`.
* ***Proposed Edit:***
  * Location: Verification Plan (Line 314)
  * Change: Replace `test_stage14_workstation.py` with `app/roop_3d_ui/src/liveSwapPreview.test.tsx` and `app/roop_3d/tests/test_live_preview.py`.

### M7: Unresolved Architectural Open Questions [`egm-zoom-out`] - Severity: Major
* **Affected Scope:** Lines 37–41
* **Discrepancy Description:** Questions 1 and 2 are unresolved.
* **Rationale:** Leaving key architectural questions open blocks implementation.
* ***AI Suggestion:*** PROPOSE FIX
* ***Justification:** Formally resolve both questions in the design doc.
* ***Proposed Edit:***
  * Location: Section "Open Questions"
  * Change: Mark questions as resolved: (1) Default to automatic hardware detection on startup (`auto`) with manual override in Settings (`6gb` vs `12gb`); (2) Automatically purge stale track mappings upon target media transition.

### M8: High Coupling of Concurrency Fixes with Aesthetic Styling [`egm-zoom-out`] - Severity: Major
* **Affected Scope:** Lines 9–19 & Lines 70–100
* **Discrepancy Description:** Stability fixes are bundled with styling in an unstructured release.
* **Rationale:** Increases regression risk.
* ***AI Suggestion:*** PROPOSE FIX
* ***Justification:** Re-align the document with strict phased staging.
* ***Proposed Edit:***
  * Location: Overview and Staging
  * Change: Explicitly structure the design into 5 sequential phases: Phase 1 (Concurrency & Contract Fixes), Phase 2 (Memory & Hardware Profiles), Phase 3 (Settings De-monolithing), Phase 4 (Functional 3D Spatial Tools), Phase 5 (Glassmorphic Styling).

### M9: Undefined Kelvin Blackbody-to-SH Relighting Pipeline [`egm-design`] - Severity: Major
* **Affected Scope:** Line 117 & Line 260
* **Discrepancy Description:** Relighting engine accepts yaw and pitch, but no Kelvin color temperature.
* **Rationale:** Missing blackbody-to-RGB conversion math.
* ***AI Suggestion:*** PROPOSE FIX
* ***Justification:** Specify Tanner Helland blackbody conversion math to compute normalized RGB factors modulating the SH coefficients.
* ***Proposed Edit:***
  * Location: Section 2.2 and Detailed Plan Item #7
  * Change: Define the Tanner Helland algorithm converting Kelvin (2000K–10000K) to normalized RGB color factors `(r, g, b)`, which modulate the directional $L_1$ and ambient $L_0^0$ Spherical Harmonics coefficients.

### M10: Missing Settings Section-to-Schema Mapping Table [`egm-structural`] - Severity: Major
* **Affected Scope:** Lines 122–130 & Lines 242–248
* **Discrepancy Description:** Missing structured mapping of schema keys to domain tabs.
* **Rationale:** Moving to static tabs risks dropping configuration keys.
* ***AI Suggestion:*** PROPOSE FIX
* ***Justification:** Add a structured mapping table covering all 28 configuration keys.
* ***Proposed Edit:***
  * Location: Section 2.3
  * Change: Add a Markdown mapping table assigning every configuration key in `legal_parameter_bounds` to its designated domain section with a catch-all fallback.

### M11: Unbounded Spatial Centroid Distance Matching Thresholds [`egm-implementation-readiness`] - Severity: Major
* **Affected Scope:** Lines 154–156 & Lines 239–240
* **Discrepancy Description:** Missing maximum cutoff radius for Euclidean centroid matching.
* **Rationale:** Can swap wrong faces in crowded scenes.
* ***AI Suggestion:*** PROPOSE FIX
* ***Justification:** Define a normalized cutoff radius $d_{\max} = 0.15 \times \text{frame diagonal}$.
* ***Proposed Edit:***
  * Location: Section 3.2 and Detailed Plan Item #4
  * Change: Define normalized cutoff radius $d_{\max} = 0.15 \times \text{frame diagonal}$. If no face falls within $d_{\max}$, return `FACE_NOT_FOUND`.

### M12: Undefined Architecture for `WorkstationParameters` [`egm-comprehension`] - Severity: Major
* **Affected Scope:** Line 54, Line 115, and Line 253
* **Discrepancy Description:** `WorkstationParameters` is introduced without definition or synchronization flow.
* **Rationale:** Ambiguity in frontend state flow.
* ***AI Suggestion:*** PROPOSE FIX
* ***Justification:** Define `WorkstationParameters` in Section 2.1 as the typed parameter object in `components/workstation/pipeline.ts`.
* ***Proposed Edit:***
  * Location: Section 2.1
  * Change: Define `WorkstationParameters` as the typed parameter object in `pipeline.ts` passed to `/api/preview`, with unidirectional update flow from UI manipulators to store drafts.

### M13: Missing Mechanism for Hardware Profile Switching [`egm-comprehension`] - Severity: Major
* **Affected Scope:** Line 38, Line 124, Lines 157–161, and Lines 284–290
* **Discrepancy Description:** Unclear whether switching profiles requires restart or rebinds dynamically.
* **Rationale:** Operational ambiguity.
* ***AI Suggestion:*** PROPOSE FIX
* ***Justification:** Specify runtime re-binding for subsequent preview dispatches.
* ***Proposed Edit:***
  * Location: Section 3.3
  * Change: Document that updating `hardware_profile` via `/api/v1/config` applies immediately to subsequent preview dispatches; active render jobs retain their launch profile until completion.

---

### Nits (N1–N7):
* **N1 [`egm-codebase-details`]:** Change `LightProbeSphere.tsx` status to `(Create)`. -> PROPOSE FIX
* **N2 [`egm-design`]:** Compute light-glass tokens (`rgba(255, 255, 255, 0.75)`) when `mode === 'light'`. -> PROPOSE FIX
* **N3 [`egm-structural`]:** Add Mermaid sequence diagram for preview scrub cancellation. -> PROPOSE FIX
* **N4 [`egm-comprehension`]:** Clarify performance mode set via DOM attribute `[data-perf-mode="high"]`. -> PROPOSE FIX
* **N5 [`egm-comprehension`]:** Define Thin Plate Spline (TPS) fallback when 3D optimization fails to converge. -> PROPOSE FIX
* **N6 [`egm-comprehension`]:** Clarify that `/api/v1/system/capabilities` exposes `caps.models[*].state`. -> PROPOSE FIX
* **N7 [`egm-comprehension`]:** Add note clarifying legacy unversioned vs. versioned routes. -> PROPOSE FIX
