# Stage 6 regression sweep - 2026-11-15

Verdict: **FAIL; release acceptance is not met.** This is verification of the current checkout, not evidence that earlier visual defects were fixed.

## Issue / action / evidence

| Issue | Action and result |
|---|---|
| Source hygiene | Initial `git status --short` showed 20 modified and 4 untracked project files; `git diff` was inspected, not treated as clean. The pre-edit AST pass parsed 641 tracked Python files with 0 syntax errors; post-edit policy and GLB tests imported the modified modules. No tracked raster/video files or media paths appeared in the status output. All generated test outputs were left intact. |
| Stale 2.5D descriptions | The reconstruction package and `tps_fallback.py` docstrings now describe perspective PnP, not a TPS/Delaunay warper. Corrected the tracker and pipeline package summaries during this sweep. |
| 2D/2.5D exclusion | `test_zero_2d_and_25d_policy.py`: **20 passed**. Active render path and tracking GLB export have no crop-paste or planar fallback. `HeadTrack.from_tracked_frames` now requires a measured perspective `pose_6dof`, preserves x/y/z translation, and rejects missing/invalid pose records. |
| Import resolution / GLB regressions | Changed-module imports resolved. Job-history + perspective tests: **5 passed**; focused 6-DoF GLB tests: **4 passed**. Full `test_glb_export.py`: **56 passed, 3 failed**. Failures: tests expected 422 for `shape_keys`/`jaw_width` but observed 200; missing identity asset expected 404 but returned 503. |
| Full backend suite | The preceding Stage 6 log records collection errors (16 errors, 1 skipped) and an extended run that stalled at 18%; the broad suite was not rerun in this pass. Logs remain under `.runtime/stage6-*.log`. |
| Focused backend suite | The preceding Stage 6 log records 76 passed, 24 failed, 1 skipped across its selected modules. Those results were not rerun or overwritten. |
| UI | `npm run build` passes TypeScript, ESLint and production build. The Vite UI loaded in the browser and settled to **3D renderer LIVE**. Existing saved project validations returned HTTP 422 `PROJECT_INVALID`; therefore the UI did not meet a strict no-error acceptance criterion. |
| CLI | `python -B -m roop_3d.cli --help` and `--headless queue list` exit 0; queue reports zero jobs. This does not certify video rendering. |
| Main service | `serve.py` launched on `127.0.0.1:8030`; `/api/v1/true3d/options` returned HTTP 200 with renderer `wired=true`, state `ready`, and `frames_rendered=0`. The runtime report is **ATTENTION** with 28 startup-reconciliation findings unresolved. |
| Original coverage / flicker / pose defects | Preserved Stage 5 report fails all eight clips on coverage and/or flicker. Target frames left unchanged: s1=15, s2=1, s3=243, s7=79, s9=100, s10=195, s11=80, s=26. Those issues remain unresolved; no new rendering claim is made. |
| Identity splits / media integrity | Existing Stage 5 evidence reports one canonical identity, no splits, successful full decoding, matching frame count/resolution/fps and preserved audio where present. These are prior results, not newly rerendered videos. |
| Temporary debug frames | `git ls-files` found zero tracked raster/video files under `app/roop_3d` and `app/roop_3d_ui`; `git status --untracked-files=all` contained no image/video paths. Ignored runtime artifacts were not deleted. |

## Preserved MP4 inventory

All files are in `outputs/test 1/`; the recursive inventory contains these eight MP4s. No new videos were generated during this sweep.

| File | Bytes |
|---|---:|
| s.mp4 | 4,866,256 |
| s1.mp4 | 1,616,125 |
| s2.mp4 | 2,462,905 |
| s3.mp4 | 13,981,365 |
| s7.mp4 | 24,605,984 |
| s9.mp4 | 7,527,823 |
| s10.mp4 | 7,752,301 |
| s11.mp4 | 2,218,939 |

## Tuning reference

These are code/schema defaults, not new CLI flags. Workstation fields live in `core/workstation.py`; global settings in `core/settings.py`; tracker options in `tracking/video_tracker.py:TrackerSettings`; renderer parameters in `pipeline/true3d_renderer.py`, `engine/volumetric/session.py`, and `reconstruction/stage4_head.py`. The API now reports the renderer ready. A serialized schema field is not necessarily an active UI control: legacy manual mask/fusion sliders are disabled, while semantic protection is automatic.

### 3D head reconstruction

- Live head: MakeHuman CC0 is installed and implemented. FLAME is unsupported by this renderer. The implemented shader is perspective SH Lambertian; the Gaussian avatar and PBR/SSS claims are unsupported.
- Camera: perspective pin-hole, auto-only: `f = 35/36 * max(width,height)` px and centered principal point. Manual intrinsics are not supported. Schema ranges (not runtime controls): FOV 5-170 degrees, focal length 50-100000 px, normalized principal point 0-1.
- Renderer internal constants: `SOURCE_FOV_DEGREES=45`, `MAX_DENSE_REPROJECTION_FRACTION=0.08`, `MIN_DENSE_REPROJECTION_PX=5`, `SILHOUETTE_FEATHER_PX=1.5`. Dense-68 target pose is required; do not loosen rejection thresholds to conceal missing measurements.
- Separate `6gb` and `12gb` hardware profiles remain supported. Global `vram_budget_mb=0` follows the profile; reserve defaults to 768 MB; concurrent jobs default to 1. No new GPU qualification was performed.

### UV texture

- Renderer `texture_size=1024`, integer range 64-4096. Higher resolution increases memory and synthesis cost.
- `restoration_weight=0.35`, range 0-1, controls canonical UV detail restoration before projection; it does not select a screen-space face restorer.
- Internal `BOUNDARY_ERODE_FRACTION=0.018` rejects contaminated source boundaries; `TONE_DECAY_FRACTION=0.12` controls completed-skin tone relaxation; `DELIGHT_MAX_LOG_SHADING=0.75` bounds delighting. Atlas completion uses Poisson seams and feathering scaled with texture resolution. Hair-covered or unseen skin remains inferred rather than measured.

### Hair masking / fusion

| Workstation schema field | Default | Range / effect |
|---|---:|---|
| hair_matting / neck_protection / preserve_ears_glasses | true | Semantic protection switches |
| occlusion_mode | standard | standard or robust; robust needs trusted parser/occluder models |
| strand_matte | false | Directional strand matte, robust mode only |
| occlusion_threshold | 0.55 | 0-1 parser confidence threshold |
| xseg_occlusion_strength | 1 | 0-1, robust mode visible-face subtraction |
| hairline_offset_px | 0 | -32 to 32 |
| feather_px | 5 | 0-64 |
| erosion_px | 0 | -32 to 32 |
| jaw_feather_px | 6 | 2-24 |
| skin_texture_retention | 0.25 | 0-1 |
| color_strength | 0.65 | 0-1 |

The active renderer requires automatic semantic protection (BiSeNet, XSeg/visible-face and occluder masks); it does not reconstruct scalp-hair geometry. The fields above remain in the workstation schema, but current UI controls for manual masking/fusion are disabled and must not be treated as effective tuning controls.

### Tracking

- Global temporal filter: `temporal_min_cutoff=0.15` (0-120, exclusive zero), `temporal_beta=0.04` (0-10), `temporal_d_cutoff=1` (0-120, exclusive zero), `pose_hysteresis_degrees=2.5` (0-10, exclusive 10). More smoothing trades responsiveness for stability.
- Tracker: `max_abs_yaw_deg=85` (10-90), `expression_max_yaw_deg=60` (10-85), `max_pose_rms_iod=0.12`, `max_pose_rate_deg_s=1400`.
- Missing detections: `coast_s=0.25` (0-0.5), `reacquire_s=1`. Calibration: 8 frames (2-120), window 120 (8-1000). Identity lock: minimum 10 frames (2-200), lock 30 frames (2-400), break run 12 (3-200).
- Cut detection: image difference 0.45; joint image difference 0.18 and histogram distance 0.7; jump 2.5 interocular distances; rotation 2700 degrees/s; max gap 0.2 s.
- Gaze and lighting default enabled; iris fallback enabled; lighting stride 3 (1-60), assumed IPD 0.063 m (0.03-0.1, exclusive endpoints).
- CLI target selection: `--track` default 0, stable `--track-id` overrides it; `--full-scan` requests every frame with a 15-minute inspection budget. No threshold change substitutes for validating coverage and flicker on the preserved clips.
