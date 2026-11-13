# 04 — Final upgrade verification report

Scope: what changed in `roop-3d` after the audits (reports 01-03), what was verified, and what was not. Every number below was observed in this session; nothing is estimated. No throughput or quality benchmark was run, so **this report makes no measured speed or fidelity claim versus roop-ultimate**; where an improvement is architectural it is labelled so.

Execution note: the requested model/effort bindings (`gpt-6-luna`, `gpt-6.1-sol`) were not selectable. Workers ran on whatever the harness offered, and the first Wave 2 workers failed on a provider rate limit; the orchestrator reviewed their partial edits, fixed a regression one of them introduced (dropped `expression_energy`/`expression_held` in `video_tracker._frame`), and completed the rest directly.

## 1. Gate results

| Gate | Command (from `app/`) | Result |
|---|---|---|
| Test suite | `./roop_3d/.venv/Scripts/python.exe -B -m pytest -c roop_3d/pyproject.toml --confcutdir=roop_3d roop_3d/tests -q` | **2873 passed, 0 failed, 47 skipped**, 13 subtests passed, 46m03s, on the final code (an earlier run of the same suite had 3 failures caused by this work; they were fixed and this re-run is clean). `test_e2e_video_pipeline.py` has ~8 min of fixture setup, which looks like a hang |
| Model registry (read-only) | `... roop_3d/audit/local_models.py --verify` | exit 0; 95/95 checked, 0 violations, no unavailable scopes |
| Isolation | `... roop_3d/audit/verify_isolation.py --report` | exit 0; 5/5 checks pass (sibling imports 7819 checked / 651 files, sibling writes 855, interpreter inside `app/roop_3d/.venv`, git origin `roop-3d.git`, 1224 tracked files free of venv/cache/weight artefacts). Counts are computed; the audit is a static AST heuristic and says so in its output |
| Frontend | `tsc --noEmit`; eslint on touched dirs; Vitest `trackingTelemetry` + `stage11Perf` | clean; 13 tests pass. The full Vitest/Playwright suites were not run |

## 2. Angle tolerance (Axis 1/2)

| Aspect | roop-ultimate (report 01) | roop-3d now |
|---|---|---|
| Live pose source | 5-pt weak-perspective solve, "measured 15-20 deg off per person" | Dense-landmark PnP (`reconstruction/pose_solver.py`, wired in `engine/detection/dense_landmarks.py` and `face_tracker.py`) when 68 landmarks exist; 5-pt estimate only as an explicitly labelled fallback (`pose_source`, `pose_fallback_reason`). Test: synthetic head recovered with reprojection RMS < 1 px, tz within 3 cm |
| Self-occlusion | none (2D profile guard changes scale only) | `mesh_raycaster.py` rewritten as a real depth-buffered visibility pass (the old file claimed Embree/BVH and was a hard-coded index mask); `use_simd`/`bvh_enabled` removed; bare vertex arrays raise `VISIBILITY_FACES_REQUIRED`. Tests (`test_w2_3d_visibility.py`): canonical head frontal 0.974 visible; MakeHuman asset >= 0.75 of front-facing vertices (it contains interior eye/mouth geometry); at +/-60, 75, 90 deg the turned-away cheek is hidden and the near cheek visible, mirror-symmetric |
| Pose bins | coarse (profile = abs(yaw) > 45) | shared spherical bin names (`reconstruction/pose_bins.py`) attached to PnP output |
| Identity transfer | 2D network | **unchanged**: still INSwapper/HifiFace-class 2D networks conditioned by 3D pose/geometry/texture |

Also done: `parametric` block on the identity-asset mesh endpoint (null + reason for unavailable parts), `spherical_bin(s)` on FaceSet coverage, `NeuralRelighter3D` docstring corrected and SH coefficients exposed. Not done: consolidation of the legacy 3x3 / 5x3 / `classify_pose_bin` grids (kept deliberately, reason in `pose_bins.py`). No real-footage +/-75 deg evidence exists (fixtures absent); all angle evidence is synthetic geometry.

## 3. Visual quality (Axis 3)

* `compositor/frequency_blend.py`: Laplacian-pyramid split, `retain_hf`, masked Lab Reinhard base; selectable as `BlendMode.FREQUENCY` in `boundary_fusion`. Test: with `retain_hf=1` fine-detail energy of the result is within 10% of the target's, with 0 it falls below 60%; pixels outside the matte are unchanged. Limit: `retain_hf`/`color_strength` are not yet exposed through `CompositeOptions` (defaults only); not enabled by default.
* `pipeline/subpixel.py`: float affine/perspective warp with Lanczos4 now used by crop fusion (test: differs from the old bilinear warp, matte stays linear).
* XSeg default `xseg_3.onnx` already exists in the manifest; a named `VISIBLE_FACE_MODEL_NOT_IN_MANIFEST` error now guards it.
* Not done: SH relighting changes (existing SH-9 fit retained), XSeg-intersect-with-mesh-silhouette, face-restorer changes. No image-quality metric was measured.

## 4. Temporal stability (Axis 4)

* `tracking/scene_cut.py`: HSV-histogram Bhattacharyya + SSIM drop (+ pose velocity), cut vs flash. Wired into `FaceTracker.process` (flash flushes filters but keeps tracks) and optionally `FrameCompositor`. Test: different scene -> cut, exposure jump with preserved structure -> flash.
* `compositor/mask_flow.py`: forward-backward-confidence dense-flow matte stabiliser, one shared `FlowEstimator` for occlusion tracker, detail stabiliser, stream restoration. Test: temporal flicker of the static background falls below 0.7x raw, edge stays within 1 px, no ghost after a cut.
* ByteTrack-style second association pass in `face_tracker.py` (low-score detections only continue existing tracks).
* One `OneEuroFilter` implementation (`temporal.py`).
* Known limits: `engine/volumetric/session.py` only forwards `tracker.scene_cut`, so flashes reach the compositor only when it owns a detector; the previously documented temporal-detail stabiliser shortfall (28% vs 70% target) was not re-measured.

## 5. Telemetry, terminal, compute, queue (Axes 5-8)

* `tracking.telemetry` frames (schema version 2, additive): unit quaternion, pixel translation/scale, and, when the observation carries a PnP fit, `translation_m`, `reproj_rms_px`, `pose_source`, `spherical_bin`. Unmeasured fields are null. Temporal fields (`jitter_px`, `stability_score`, flash, mask-flow confidence) exist on the schema and in `tracking/temporal_telemetry.py` but the live render path does not yet publish them, so the UI graph shows `n/a` until it does.
* UI: `stores/trackingStream.ts` + `components/hud/TrackingTelemetryPanel.tsx` in the pipeline pill: yaw/pitch/roll, quaternion, translation, source, bin, reprojection error, stability/jitter graph with cut and flash markers. The existing 3D inspector already had lit/wire/albedo/normal/depth/cavity/SH views and `BatchMatrixScreen` already existed; neither was duplicated.
* CLI: `roop_3d/cli.py` (`--diagnostics`, `render` against a running local API with `--json-logs`), `telemetry/tui.py`; tests in `test_w2_ops_{cli_tui,media,trt_queue,audit}.py` pass (the Rich dashboard test skips if Rich is absent; plain output is the tested path). `render` imports the FaceSet, uploads the video, submits one job through the app's own job store and polls it in-process; it was **not run end to end** (needs models), only its missing-input guard is tested.
* Compute: shared TRT optimisation level between `core/ort_session.py` and `trt_session_builder.py`; INT8 only as guarded opt-in. Queue-state mapping: `SpecState` has 11 members (9 states plus failed and cancelled); `spec_slot()` collapses failed/cancelled into the spec's 10th slot. TRT level 5 chosen because the strict builder is the canary-validated path; there is no on-device A/B against 3. INT8 raises `INT8_CALIBRATION_MISSING` without a calibration cache and `INT8_FP32_LOCKED_MODEL` for locked models. All-stream remux and these changes pass their tests. No GPU benchmark was run.
* Integrity: `verify_isolation.py` is now a real check (was a hard-coded pass); `local_models.py --verify` no longer writes files.

## 6. Open items

Selectable frequency-blend parameters, publishing temporal telemetry from the live path, standalone render CLI, real-footage extreme-yaw fixtures, TensorRT INT8 calibration / RTX 3060 validation, soak runs. The `rich` dependency was installed into the local venv by hand; `requirements.lock` was not regenerated.
