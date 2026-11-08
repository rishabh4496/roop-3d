# Final workstation stage: curves, sweep, linter, flamegraph, passes, certification

This stage was written against a prompt that described four feature areas. Most of the surface already existed, so the work was an
audit followed by the real gaps, not a rebuild. Paths in the prompt (`app/api.py`, `react-ui/`) do not exist here: the backend is
`app/roop_3d/`, the frontend `app/roop_3d_ui/`, and tests live in `app/roop_3d/tests/` plus the certification suite in `app/tests/`.

## What was already there (verified in code, not rebuilt)

| Area | Existing implementation |
|---|---|
| Timeline | filmstrip, measured timecode ruler, in/out trim, chapter markers, `0.25 / 0.5 / 1 / 2 / 4x` playback (`components/timeline/TimelineWorkstation.jsx`) |
| Curve editor | collapsible Bezier / linear / hold keyframe editor with undo/redo (`CurveEditorTrack.jsx`, `stores/useHistoryStore.js`); curves are validated and evaluated per frame by the backend (`core/animation.py`) |
| Snapshots | A/B/C/D slots with split wipe and full-workspace restore (`SnapshotSlots.jsx`) |
| 3x3 sweep | any two settings, nine real `POST /api/preview` frames, click a tile to apply (`ContactSheetModal.tsx`) |
| Linter | modal plus `POST /system/preflight-lint` with VRAM, encoder, models, TensorRT dynamic-shape and thread checks |
| Flamegraph | per-stage means from the job's measured timings, 14-section diagnostics report, NVML PCIe throughput |
| Interchange | FCPXML 1.10, OpenTimelineIO, CMX 3600 EDL (one file per track), Premiere XML |
| Passes | alpha, occluder, uv-warped face, normal, depth as 16-bit PNG sequences plus lossless FFV1 |

## What this stage added

### 1. Parameter curves for the five animatable controls
`CURVE_PARAMETERS` now lists, first in the dropdown: **3D mesh geometry weight** (`pose_strength`), **UltraMax enhancer fidelity**
(`fidelity_weight`, the `w` input UltraMax feeds its network), **SH relighting intensity** (`relight_strength`), **Mask feather radius**
(`feather_px`) and **Occlusion threshold** (`occlusion_threshold`, new).

* `relight_strength` and `occlusion_threshold` are registered in `core/animation.py` (bounds 0..1) and evaluated per frame into the real
  `WorkstationParameters` fields.
* `occlusion_threshold` is a new opt-in field. It feeds `CompositeOptions.semantic_confidence_threshold`, the confidence the semantic
  face parser must reach before its mask is trusted over the landmark outline (both the standard and robust compositor paths).
  The default is the compositor's long-standing `0.55`, and the field is omitted from the wire until set, so earlier requests keep
  their serialisation, idempotency fingerprint and preview cache key.
* **It is not an XSeg control.** This build has no XSeg network (see `POST_PRODUCTION.md`); the control drives the installed parser
  (BiSeNet). The label reads "Occlusion threshold (XSeg / BiSeNet)" and the hint under the curve says so.
* The relight curve only has an effect while 3D Relighting is switched on; its hint says so.

### 2. 3x3 sweep: enhancer and relight axes, native-pixel zoom
Two axes join the continuous settings: **Enhancer** (`None / GPEN Realistic / UltraMax`, the Engine & Model Rack's own mapping, where
"None" is a restoration blend of 0) and **SH relighting strength** (`25% / 60% / 100%`). `applySweepPatch` is the single function that
turns a tile into parameters, used both for the tile's preview request and for applying it, so what you click is what you get.
A restorer at blend 0 would look identical to "None", so a swept restorer keeps your blend or falls back to `0.7`.

Each ready tile has a **Zoom** button: wheel / `+` `-` / 400% / double-click zoom about the pointer up to 16x, drag or arrow keys to pan,
`0` to fit, then **Apply these settings**. Tiles are the full preview frames shrunk by CSS, so zoom shows native preview pixels (not
more than the preview endpoint renders).

**A layout defect found by looking at it.** The sweep dialog was rendering inside the workstation's centre column: the animated screen
wrappers create a containing block for `position: fixed`, so the backdrop only dimmed that column and the sheet's third column of
tiles was clipped away. `Modal` now renders through a portal to `<body>` (which also repairs the other dialogs that use it), and tile
height is capped so all nine tiles fit the dialog without scrolling. `e2e/finalWorkstation.e2e.ts` drives the sweep (checking that the
nine preview requests carry the right per-tile restorer and relight values) and the Stream Overlap tab, and saves screenshots.

### 3. Pre-flight linter: four render-readiness checks
All run server-side in `server/diagnostics.py`; the request optionally carries `swapper_model_id` and `export_passes` so the estimates
are about *this* render.

| Check id | Status | Logic |
|---|---|---|
| `fsz_extreme_yaw` | WARN | the target's per-track yaw range (from the face inspection) against the yaw of every real view in the mapped `.fsz`. A turn beyond 40 deg needs a view within one pose cell (15 deg) of it; otherwise the check names the track, the angle and the missing side. An unreadable archive is reported as unknown, not as a pass. (The older `3d_pose_coverage` view-count proxy now only fires when the `.fsz` cannot be read.) |
| `trt_engines_built` | WARN | only when TensorRT is the selected provider on a verified NVIDIA GPU: swapper and restorer this render will load, split by the catalog's `trt_cached` flag. First-run engine builds look like a stalled render. |
| `storage_disk_space` | PASS / WARN / FAIL | free space on the output volume against an estimate of this render: per-frame PNG checkpoints, each selected pass (16-bit, 1 or 3 channels), and the baked video at 1.5x the source. PNG is assumed to be 60% of raw size, so the report calls it an estimate. FAIL below the estimate, WARN below 1.5x (or 2 GB). |
| `ultramax_on_4k` | WARN + one-click fix | target short edge >= 2160 and the effective restorer is not UltraMax. A chosen restorer at blend 0 counts as off. The fix selects UltraMax and gives it a visible blend. |

The modal blocks on FAIL (the existing "skip for this session" escape remains).

**Batch submission is linted too** (`components/diagnostics/batchLint.js`, used by `BatchMatrixScreen`): each *distinct target* in the
batch is linted once with the batch's rack parameters, swapper and passes. A FAIL stops the whole batch before any job is created;
warnings are appended to the "queued N jobs" message; an unreachable linter never blocks queueing. The same session opt-out applies.
Batch payloads are full job requests, so each target is linted with its own inspection and FaceSet mappings, including the yaw check.

### 4. Flamegraph: measured lanes, ORT-boundary copies, ring buffers
New tab **Stream Overlap & PCIe** (`StreamOverlapPanel.jsx`, `lanesModel.js`).

* **Lanes.** `AsyncPipeline` now records `(frame, start, end)` per stage and `snapshot()` returns the last 48 frames on one time axis,
  so overlap is visible. Real stage names are `decode`, `track` (Stage 1 pose), `swap` (Stage 2), `composite` (Stage 3: mask, relight,
  restorer incl. UltraMax) and `write`. **`write` is the per-frame PNG checkpoint, not NVENC**; the final video encode runs after the
  last frame, outside the overlapped graph, and is labelled that way.
* **CUDA stream overlap is not measured in this environment.** Stream windows need `torch.cuda` events and PyTorch is not installed
  here. The snapshot returns `overlap_gpu_streams: null` plus a note, and the UI prints "not measured" with the reason. Host-thread
  overlap (>=2 / >=3 stages active) is measured and shown.
* **PCIe copies: a ledger, and an honest verdict.** `core/transfer_ledger.py` wraps the ONNX Runtime calls that perform a memcpy
  (`ortvalue_from_numpy` / `update_inplace` to a device, `OrtValue.numpy()` from a device, `InferenceSession.run` on CUDA/TensorRT)
  and attributes each to the pipeline stage that issued it. Verified on the RTX 4070: one 64-byte upload and one download are counted
  exactly, CPU-only traffic is not counted, and a device-resident chain reads as zero. **The current pipeline is not zero-copy**:
  every model wrapper calls `.run(...)` then `.numpy()`, so each model call is one H->D and one D->H. The panel reports that
  (`N tensor copies per frame cross PCIe between models`) instead of the "0 intermediate copies" the prompt hoped to confirm.
  "Nothing observed yet" is `null` (unknown), never zero. Not counted, and listed in the response: torch / CV-CUDA / OpenCV-CUDA
  copies, NVENC/NVDEC internals, copies inside ORT between graph partitions.
* **Ring buffer.** The frame pools are page-locked *host* memory, not a VRAM ring; the panel shows slots in flight, size and how often
  decode waited for a slot, and says "host memory".

### 5. Clean relit face pass
`relit` joins the pass list (`alpha, occluder, uv, normal, depth, relit`). It is the swapped face after SH relighting, before blending
with the target: straight RGB, no background or occluders, to be paired with the alpha pass, captured from the compositor's `color`
stage. A job that requests it with relighting off is rejected at validation instead of exporting black frames; the Extras panel
disables Queue and explains.

### 6. NLE interchange
Unchanged: FCPXML 1.10, OTIO, EDL, Premiere XML. DaVinci Resolve imports FCPXML, OTIO and EDL through *File > Import > Timeline*.
Import acceptance in Final Cut, Resolve, Nuke or After Effects has **not** been performed on this host (see `POST_PRODUCTION.md`).

## Certification (`app/tests/test_stage27_certification.py`)

```
app/roop_3d/.venv/Scripts/python.exe -m pytest app/tests/test_stage27_certification.py -c app/tests/pytest.ini -v -rs
```
Run from the repository root. For the UI half run `npx playwright test e2e/certification.e2e.ts` in `app/roop_3d_ui` first; it writes
`app/roop_3d/.runtime/certification/ui_coverage.json`, and the Python test skips (with that instruction) when it is absent or stale.

| Claim | How it is certified | Measured here |
|---|---|---|
| (a) UltraMax native high-res sharpness | calls the compositor regression functions in `test_hires_handoff.py`; reports the ratio | native detail **7.7x** the legacy shrunk path (threshold 3x) |
| (b) full-head replacement from `.fsz` | calls the `test_full_head.py` regressions: 3D route never calls the inner-face swapper, forehead and jaw kept, dense layouts round-trip through the `.fsz` | pass |
| (c) >85% GPU through TensorRT FP16 + IOBinding + micro-batching | `roop_3d/tools/gpu_saturation.py`, best of two 8 s runs, NVML sampled at 10 Hz | **88.7% mean, 97% peak**, 8,000 samples/s, 0 intermediate copies; batch 1 alone is ~52%, batch 8 on one worker ~83% |
| (d) 100% UI route and WebSocket binary-canvas coverage | Playwright in real Chrome: every route in `AppRouter.jsx` mounts and is reachable one at a time; binary JPEG frames paint the live canvas | 9/9 routes, 3/3 frames |
| standalone / offline | `test_standalone_install.py`: `app/env`, `app/models`, `app/facesets` are real directories (no symlink, no junction, nothing inside linked), gitignored, untracked; no external host in 100+ production UI source files or the built bundle; runtime check that no request leaves 127.0.0.1 | pass |

Read the (c) number for what it is: a **synthetic FP16 convolution stack** run through the same provider, binding and batching path.
No model weights are installed in this repository, so end-to-end swap-pipeline utilization cannot be certified here and is not claimed.

Two environment facts the certification exposed:

* `test_hires_handoff.py` skipped *entirely* here (a module-level `importorskip("torch")`), so the CPU sharpness evidence never ran.
  The guard is now per test; 11 CPU tests execute and 9 CUDA tests stay skipped with an explicit reason.
* The development machine's endpoint-security web filter (Kaspersky) injects a script into every HTTP page; the UI e2e therefore sees
  11 requests to `*.kaspersky-labs.com`. They are counted in the report (`injected_by_security_software`) and any other off-machine
  request fails the test. No source file or bundle references that host.

## Not done, and why

* **Per-frame D2H/H2D elimination.** Making the stages hand over device tensors is real engine work. The ledger now measures it, so
  that work has a number to move.
* **CUDA stream overlap and a VRAM ring buffer** need a torch-backed pipeline.
* **NVENC encode timing in the lanes**: the encode is a separate phase in the current runner and is not instrumented.
* **Pipeline-level GPU utilization** needs model weights.
