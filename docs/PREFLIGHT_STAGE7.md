# Stage 7: pre-flight linter, quick tour, checkpoints and the E2E gate

Path mapping (the stage prompt names `app/api.py`, `app/roop/*.py`, `react-ui/`; none exist):
backend `app/roop_3d/` (`server/`, `engine/`, `jobs/`, `core/`), frontend `app/roop_3d_ui/`,
benchmark endpoint `POST /api/v1/settings/benchmark_threads` (the prompt's `/api/settings/...`).

## Stage 27 Pre-Flight Render Linter

`POST /api/v1/system/preflight-lint` keeps its four original checks and adds four. A check may carry an `action`
that the modal renders as a button; the lint re-runs after an action so the fix is verified, not assumed.

| id | warns when | action |
|---|---|---|
| `trt_dynamic_models` | TensorRT is selected (or `auto` on a verified NVIDIA GPU that advertises it) and a registry swapper model has neither `<stem>_dynamic.onnx` nor a verified `<stem>.batch.onnx` | **Convert & Optimize Now** → `POST /system/optimize-models` (runs `models.dynamic_shapes.convert`, reports `CONVERTED` / `STATIC_BATCH_1` / `FAILED` per model) |
| `frontalization_vs_pose_matrix` | `pose_frontalization` > 0.5 and a mapped FaceSet's 3×3 pose matrix has ≥ 5 populated cells | lower to 0.5 |
| `ultramax_crop_bypass` | restorer policy is `ultramax`, the ≤ 320 px native bypass is off, and the target is ≥ 1080 px shortest edge with > 1 simultaneous face | enable bypass at 320 |
| `vram_threads` | free VRAM < 4 GiB and `execution_threads` ≥ 4 | set the recommended thread count |

Thread recommendation: the measured value from the `benchmark_threads` snapshot when its state is `DONE`
(capped below the current count); otherwise a conservative VRAM bound, and the message says it is not measured.
The benchmark itself measures a synthetic affine/transfer workload, not full-pipeline FPS.

Limits: the converted `_dynamic.onnx` is written and verified, but the streaming runner still builds its TensorRT
profile from the registry's `inswapper_128.batch.onnx`; it does not yet prefer the local `_dynamic` file.
A check whose evidence cannot be read (no probe, unreadable FaceSet, no inspection) reports PASS rather than guessing.

## Measured per-frame flamegraph

`JobProgress.stage_timings_ms` (also carried by `job.progress` WebSocket events) holds mean milliseconds for
`decode, detection, pose_tps, inswapper, masking, restorer, encode`. A stage that was not timed is **absent**,
and the UI shows "not measured"; the old hard-coded 9-stage defaults were removed.

* Reference runner: decode wait, detector/tracker, retarget + projection/TPS, swap, compositor masking (everything in
  `compose` except the restorer), restorer, and per-frame PNG checkpoint write as "encode".
* Streaming runner: decode (ring wait), detection (prepass detect/track calls), "inswapper" (whole batch incl. any
  decoupled restorer, which is not separable there) and writer encode. Masking, pose/TPS and restorer show "not measured".

Shown on `#/processing` (running job preferred) and inside the expanded diagnostics HUD. Preview-capture latency is
used only when no job timings exist, and covers detection, pose/TPS and swap.

## Quick Tour and layouts

"How to Use" in the header and on `#/home` opens a four-step dialog (source face → target & 3D pose → live preview &
buffer soloing → render queue). Each step navigates to `#/faceswap` and outlines a `data-tour` element when present;
`Esc`, arrow keys and dots navigate. It does not auto-open; the header button is highlighted until the first finish.
The prompt's "Face Banking" is not a feature in this UI; the tour describes the real identity clusters
(Swap / Reference / Exclude per cluster in the Face Tray). Presets are now `Standard`, `3D Pose & Geometry Pro`,
`Masking & VFX Compositor` and `Dual Monitor Floating Preview`; custom layouts (up to 30) were already saveable.

## Checkpoints

Every job writes `checkpoint.json`, `checkpoint.bin` and now `checkpoint.npz` (float64 pose cache plus the JSON
manifest, loaded without pickles via `read_checkpoint_npz`). New checkpoint fields: `inputs`, `pose_settings`
(frontalization, TPS weight, per-FaceSet pose-matrix fingerprint), `hardware_state` and the existing
`execution_provider`. Old checkpoints load unchanged. The work directory, NPZ included, is removed when a job
completes or is cancelled, as before; the queue and project resume paths are unchanged.

## Verification gate

```
cd app
roop_3d/.venv/Scripts/python.exe -m pytest roop_3d/tests
roop_3d/.venv/Scripts/python.exe ../verify_integration.py
cd roop_3d_ui && npm run build && npx vitest run
cd .. && roop_3d/.venv/Scripts/python.exe -m roop_3d.tools.verify_e2e_workstation
```

`verify_e2e_workstation` serves the built `dist/` through the real app and checks assets, zero remote references, SPA
fallback, the API 404 boundary, all eight lint checks, strict TensorRT profile binding, the three render stages with
measured timings, the JSON/NPZ checkpoint, then runs the Stage 2–5 and 7 test modules (preview swap, Use Face From
Frame). No browser driver is installed, so "zero console errors" is asserted in the Vitest suite (component and
full-`App` mounts with a `console.error` guard), not in a real browser.
