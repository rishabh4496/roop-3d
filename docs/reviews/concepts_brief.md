# Roop 3D Workstation Modernization & UI Repair

* **Author:** rishabh4496
* **Date:** 2026-11-03

## Overview

A comprehensive overhaul of the Roop 3D workstation frontend (`app/roop_3d_ui`) and related server coordination contracts. This initiative pairs a dark glassmorphic visual language and functional 3D spatial tools with direct engineering fixes for active runtime breakages: live swap preview request collisions (HTTP 409 and 422 errors), FaceSet extraction contract failures, and monolithic settings layout. The architecture enforces hardware-specific operational boundaries, supporting an RTX 3060 (6 GB VRAM) low-memory fallback profile alongside an RTX 4070 (12 GB VRAM) high-performance workstation tier.

## Analysis

### 1. Visual Glassmorphic Design & Theme Tokenization
* **CSS Glass Token System:** Glass surfaces must not rely on indiscriminate CSS blurs that force costly full-frame compositing recalculations in Chromium. Instead, tokenized glass surfaces (`--surface-glass`, `--surface-glass-raised`, `--glass-blur`, `--glass-border`, `--glass-specular`) will sit exclusively on floating panels and modals above static elements. Active WebGL video canvases and 3D viewports remain blur-free to avoid GPU rasterization overhead.
* **Theme System Continuity:** Roop 3D supports 37 existing themes, including AMOLED pure black (`#000000`), Midnight Violet, and Dracula. Glassmorphic tokens must compute relative to the active theme palette rather than hardcoding static rgba tints, preserving high-contrast legibility across all dark and light modes.
* **Studio Panel Hierarchy:** The workstation uses a 4-panel studio layout (Target, Identity, Live Preview, and Studio Inspector over a bottom timeline dock). Acrylic panel styling will visually distinguish active focus areas while demoting secondary tools.

### 2. Concurrency Control & Live Preview Contract Repair
* **Benign Cancellation vs. Conflict (HTTP 409):** In `useLiveSwapPreview`, rapid playhead scrubbing or slider drags trigger `AbortController.abort()`. On the server, `_acquire_slot()` detects cancellation and raises `DomainError("PREVIEW_CANCELLED", 409)`. This triggers false error flashes in the UI. The server must acknowledge cancelled in-flight requests as non-error cancellations (HTTP 204 or suppressed payload), while the client preserves the last valid frame as stale until a fresh frame arrives.
* **Stale Media Mapping Pruning (HTTP 422):** When switching target videos, `useFaceSelectionStore` previously retained track mappings from prior media. Posting invalid track IDs triggers `DomainError("TRACK_NOT_FOUND", 422)`. The frontend must sanitize mapping payloads against current target inspection data before submission.
* **Extraction Contract Sanitization (HTTP 422):** In `useFaceExtraction.ts`, assigning a face to an active FaceSet when no active source ID exists causes missing field errors in the backend Pydantic schema. Pre-flight validation in `FaceContextMenu` must guarantee valid IDs or fall back to named FaceSet creation.
* **Centroid-Based Spatial Face Matching:** Detector rank desynchronization between video decoders and frontend inspections will be resolved by submitting normalized bounding box centroids (`norm_cx, norm_cy`) alongside left-to-right indices.

### 3. VRAM Profiling & Memory Budgets (6 GB vs. 12 GB)
* **RTX 3060 6 GB (Low-VRAM Fallback Profile):**
  * Maximum VRAM allocation capped at 4,600 MiB to leave 1.5 GB for OS window compositing and browser WebGL.
  * Sequential model execution: intermediate feature maps and tensor buffers must be explicitly released between detection, alignment, swapping, and enhancement passes.
  * Preview throttling: scrub interactions run at 0.5x resolution with post-processing face restoration bypassed until the playhead settles (250ms debounce).
  * WebGL canvas clamping: Three.js device pixel ratio clamped to `1.0` with disabled antialiasing buffers to prevent VRAM starvation.
* **RTX 4070 12 GB (Standard Performance Profile):**
  * Persistent resident models in VRAM (~8.5 GB allocated) for zero-latency engine switches.
  * Real-time preview rendering at full native resolution with live restoration active during playback.
  * High-fidelity 3D viewport with dynamic device pixel ratios (`[1, 2]`) and hardware MSAA.

### 4. Categorized & Live-Validated Settings Architecture
* **De-monolithing the Settings Screen:** The single flat list rendered by `SettingsInspector.tsx` will be replaced with five organized, validated domains:
  1. *Compute & Acceleration:* Execution provider selection (`TensorrtExecutionProvider`, `CUDAExecutionProvider`, `CPUExecutionProvider`), hardware profile (`auto`, `6gb`, `12gb`), VRAM cap, safety margins.
  2. *3D Spatial & Volumetric Synthesis:* FLAME 3D mesh fit parameters, camera projection mode, 2.5D TPS fallback thresholds, pose hysteresis, spherical harmonic relighting.
  3. *Engine & Stream Pipeline:* Execution thread pools, ring buffer slot allocation, batch sizing, chunk segment frame counts.
  4. *Finishing & Post-Production:* Color transfer algorithms (`lab`, `histogram`), restoration engine selection (`gpen`, `codeformer`), lip-sync configuration, C2PA provenance signing.
  5. *Workstation UX & Display:* Glassmorphic blur intensity, hardware telemetry HUD visibility, viewport antialiasing, theme selection.
* **Real-Time Consistency Checking:** Reactive client-side validation prevents incompatible settings configurations (such as choosing TensorRT without supported hardware engines, or requesting a 12 GB profile on a 6 GB device), providing contextual remediation text directly beside controls.

### 5. Functional 3D Spatial Utilities
* **Interactive Head Pose Gizmo:** Replaces static numeric readouts with a Three.js / React Three Fiber canonical 3D low-poly head mesh. Supports direct drag manipulation of Yaw, Pitch, and Roll angles with Euler/Quaternion synchronization writing directly into `WorkstationParameters`.
* **Spherical Harmonics (SH L2) Lighting Probe:** Replaces static vector indicators with an interactive WebGL sphere rendering 9-coefficient irradiance. Features direct draggable key-light positioning for azimuth and elevation alongside a Kelvin color temperature slider (2000K–10000K).
* **FLAME Parametric Mesh Inspection:** Renders the 5,023-vertex FLAME mesh reconstructed from multi-view FaceSets (`GET /api/v1/facesets/{id}/identity-asset/mesh`). Displays a pose-coverage confidence heatmap (mint for high confidence, amber for sparse, dim for blind spots) with strict WebGL resource cleanup (`dispose()`) to prevent browser memory leaks.

## Goals vs. Non-Goals

* **In-Scope Goals:**
  * Eliminate HTTP 409 `PREVIEW_CANCELLED` errors by implementing benign cancellation handling on both client and server.
  * Fix HTTP 422 validation errors across `/api/preview`, `/faces/extract_from_frame`, and `/faces/bank/discover` through payload sanitation and centroid matching.
  * Restructure `SettingsInspector.tsx` into 5 domain-specific tabs with reactive constraint validation.
  * Implement an explicit hardware profile switch (6 GB Low-VRAM vs. 12 GB Standard) with automated fallback safeguards.
  * Build interactive Three.js 3D spatial tools: Head Pose Gizmo, Spherical Harmonics Lighting Probe, and FLAME Mesh Viewer.
  * Deploy a theme-tokenized dark glassmorphic design system that preserves existing contrast and passes all 37 Vitest suites (317 tests).

* **Out-of-Scope Non-Goals:**
  * Unbounded global CSS blurs over active video playback canvases.
  * Non-functional 3D background animations or decorative camera swoops that delay user interaction.
  * Modifications to underlying PyTorch/TensorRT model weight files or breaking external Pinokio launcher lifecycles.

## Alternatives Weighed

* **Full CSS Backdrop Blur on All Viewport Panels:** Rejected due to GPU compositing bottlenecks. Benchmarks show full-screen `backdrop-filter` reduces video playback from 60 FPS to under 28 FPS on 6 GB GPUs running concurrent CUDA tasks.
* **Monolithic Settings Modal (Status Quo):** Rejected. A flat list of unorganized schema properties prevents users from identifying which engine or hardware settings are active and valid.
* **Continuous Background Three.js Scene:** Rejected. Persistent WebGL draw loops consume dedicated VRAM and GPU compute cycles needed by neural network models.
* **Client-Only Error Masking for Live Preview:** Rejected. Simply swallowing 409 and 422 errors in the browser leaves the backend preview queue clogged with dead tasks; the server must coordinate proper slot release.

## Key Decisions & Staging for Next Steps

* **Staging Order:**
  * *Phase 1 (Blocker Resolution):* Concurrency handling for `/api/preview` (HTTP 409 suppression) and payload contract alignment for face extraction (HTTP 422).
  * *Phase 2 (Memory Architecture):* Implementation of the 6 GB Low-VRAM profile with sequential pipeline execution and preview resolution throttling.
  * *Phase 3 (Settings De-monolithing):* Reorganizing `SettingsInspector.tsx` into 5 validated domain panels.
  * *Phase 4 (Functional 3D Spatial Tools):* WebGL Head Pose Gizmo, SH Lighting Probe, and FLAME Mesh Viewer with strict resource disposal.
  * *Phase 5 (Glassmorphic Styling):* Theme-tokenized glass CSS surfaces applied across studio panels and modals, validated against all test suites.
