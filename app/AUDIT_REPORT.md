# Roop 3D Read-Only Repository Audit

**Scope:** repository source and configuration under `app/`, active Pinokio runtime metadata, the requested FaceSet/model/output inventories, and the supplied MP4 directory. Paths and source line numbers below are relative to the `roop-3d` repository root unless marked absolute.

**Evidence labels:** **Source** = current code/configuration; **saved evidence** = an existing file under `outputs/test 1` or `.runtime`, not a fresh run; **not tested** = no app launch, model inference, or video decode was performed. No source file, model, FaceSet, video, or output was changed. This report is the sole requested write. Video/image contents and the `.fsz` archive bytes were not opened. No external `roop-ultimate` model directory was read.

## Executive findings

1. **Render is unavailable in the current source.** `true3d_renderer.is_wired()` returns false and render entry points raise `RENDERER_NOT_WIRED`/503. The default runner and the opt-in FFmpeg runner gate before a frame can be rendered; no 2D fallback exists (`app/roop_3d/pipeline/true3d_renderer.py:1-53`, `app/roop_3d/optimized_processor.py:226-242`, `app/roop_3d/pipeline/video_job_runner.py:359-384`, `app/roop_3d/core/true3d_guard.py:588-602`).
2. **Pinokio startup has a likely fatal factory-call mismatch.** `server.__main__` prefers `app.api.create_unified_app`, then always passes `share=...`; that function does not accept `share`. Its fallback catches only `ImportError`, not the resulting `TypeError`. This is statically evident but was not reproduced by launching the app (`app/roop_3d/server/__main__.py:28-32,121-128`; `app/api.py:41-46`). If the import path is reached, `create_unified_app` also supplies `repo_root`, which makes `create_app` default `worker_enabled=False` (`app/api.py:43-45`; `app/roop_3d/server/app.py:74-80`).
3. **The job gate is inconsistent with capability reporting.** `/system/capabilities` advertises `local_video_render` and an enabled queue based on worker existence while separately reporting `true_3d_render.enabled=false`; `/jobs` validates the engine ID and inspection, but does not check `EngineRegistry.selectable()` or renderer readiness. A request can therefore be accepted and fail in the worker at the render guard (`server/routes.py:60-153,629-671`; `engine/registry.py:55-69`).
4. **No active classic 5-point crop/paste or named 2.5D swapper call was found.** The crop-swap path is explicitly described as removed. Existing affine calls are input alignment, transformed masks/buffers, or post-render crop placement—not a five-point swapper paste-back. Some dated test artifacts contain old engine/detection results and are not current inference evidence.
5. **The scanner is not sparse by default, but the UI fabricates intermediate detections from summaries.** `/media/inspect-faces` runs detection/tracking on every decoded frame. It stores one first-seen box per track, then the UI reuses that box and synthetic landmarks/confidence across the track interval. Frame-locking also mixes IDs from independent tracking registries. These are concrete causes of misleading/missing/misbound faces even when scan inference succeeds (`media/service.py:174-284`; `ui/src/useFaceSelectionStore.ts:312-409`; `ui/src/utils/frameFaces.ts:35-65`).
6. **The local venv path is consistent; provider availability is not the main blocker.** The active Pinokio venv is `app/roop_3d/.venv`; it contains ONNX Runtime GPU and advertises TensorRT/CUDA/CPU EPs. No current model session was run. PyTorch, InsightFace, and MediaPipe package metadata is absent; PyTorch is optional in this project. The unwired renderer remains a blocker regardless.
7. **The saved `harjot` asset is not a complete full-head reconstruction.** Existing JSON reports five 5-point views, only 11.07° pitch span, no up/down views, `limited_multiview`, no depth triangulation, and face-only geometry; the cached mesh has 1,129 vertices, below the full-head guard minimum of 1,500. This is archived build evidence, not a fresh reconstruction (`outputs/test 1/phase1_verify/faceset_result.json`; guard policy in `core/true3d_guard.py:97-120,350-425`).

## 1. End-to-end execution map

| Stage | Current path and key entry points | Audit result |
|---|---|---|
| Pinokio startup | `start_roop3d.js:33-47` selects `app/roop_3d/.venv`, runs `uv pip sync --require-hashes`, then `python -B app/serve.py`. `app/serve.py:16-24` bootstraps CUDA/TensorRT DLL paths before importing `server.__main__`. | **Source:** venv path matches the installed venv; no path mismatch found. **Risk:** factory mismatch described above may stop startup before Uvicorn. `start.js`/`install.js` also target the project venv; `install.js:13-18,35-47`. |
| App/model startup | `server/app.py:74-114` constructs the catalog, model hub, media service, and worker; lifespan rescans local models and starts the worker at `116-146`. `ModelCatalog` and registries handle discovery, verification, and lifecycle gates (`models/catalog.py:84-114,214-283,546-560`; `models/registry.py:274-386`; `models/governance.py:19-110`). | Local model paths are rescanned at startup. A malformed universal manifest disables its model-management endpoints instead of crashing the server (`server/app.py:56-71`). Registry artifacts use pinned digest/size and governance checks; files placed directly in `app/models` use the explicit local-trust path, which does **not** establish upstream provenance or license approval (`catalog.py:90-97,214-243,285-294,546-560`). |
| Offline/readiness checks | `installer/offline_cache.py:34-65,68-80,134-150`; `/offline-health` route in `server/routes.py:282-287`. Pinokio runs a normal `uv pip sync --require-hashes`, without an offline-only flag (`start_roop3d.js:45-47`). | The offline dependency check calls a venv “present/verified” when `.venv/pyvenv.cfg` exists, and `offline_ready` only requires that venv-presence flag plus required local models. It does not prove installed distributions are complete (`offline_cache.py:47-56,134-149`). A cold cache may require package-index access; this is a source-level risk, not a tested network event. Model-download routes remain available (`server/routes.py:510-517`), so the application does not globally prohibit network access. |
| FaceSet loading | `faceset_loader.LandmarkEngine`, `FaceSetReader`, and reconstruction helpers parse `.fsz` or image inputs, resolve ArcFace/landmark models, align samples, estimate pose, and enrich views (`faceset_loader.py:1-16,62-103,139-170,173-181`; `faceset/faceset_reader.py:74-128,230-244,370-437`; `faceset/validator.py:19-109`). | This is source-image identity/reconstruction input, not a 2D swapper. The saved `harjot` asset uses only five landmarks/view. **External-path risk:** `faceset_loader._model_dirs` includes sibling `roop-ultimate` landmark-model candidates and `_session` opens found files directly with ONNX Runtime, rather than via the hashed model registry (`faceset_loader.py:49-59,85-103`). No such external path was accessed during this audit. |
| Video decode and face inspection | `MediaService.inspect` hashes the video and iterates `VideoReader(path).frames()`, calling `FaceTracker.process` on every decoded frame (`media/service.py:174-204`). `VideoReader` handles color/HDR/rotation (`media/video_reader.py:34-78,92-130`). | **Source:** UI requests `full_video: true`, `time_budget_seconds: 900` (`ui/src/App.tsx:482-489`). The scan is full-frame inference, not a fixed-stride scan; the 900 s limit and 64-track cap still apply (`media/service.py:205-240`). `max_samples` counts keyframes and is a stop bound, not an inference stride. |
| Face detection and cross-frame identity | `MediaService` wraps `Detector` with `RobustDetector` and `FaceAnalyzer` (`media/service.py:40-64`); inspection constructs `FaceTracker` with `TrackPolicy` when `identity_lock=true` (`service.py:174-199`). | The main inspection returns serialized `FaceTrack` summaries and persists JSON (`service.py:261-284`). Frame scrubbing/angle capture uses `FaceAnalyzer` and the separate `FaceTrackRegistry`, not the inspection tracker (`service.py:40-64`; `server/target_lock.py:145-151`; `identity/target_lock.py:175-195`). |
| 3D reconstruction and UV | FaceSet reconstruction/build operations include multiview fitting, identity assets, UV synthesis, and full-head rasterization (`reconstruction/identity_asset.py:49-74,296-310,450-555`; `reconstruction/full_head.py:70-158`; `reconstruction/uv_synthesizer.py:89-153,372-603`). | These are real reconstruction building blocks. The cached `harjot` asset is facial-region/template geometry, not complete head geometry. `ReferencePipeline` has no active neural face replacement (`engine/volumetric/session.py:124-127`). |
| Renderer, mask, blend | `true3d_renderer.is_wired()` is false and renderer functions raise (`pipeline/true3d_renderer.py:1-53`). `require_true3d_renderer` rejects before render (`core/true3d_guard.py:588-602`). The source has face parsing, coverage/occlusion masks, LAB/color/illumination, multiband fusion, and SDF/boundary feathering (`compositor/pipeline.py:48-63,391-417,494-527,674-762,829-832`; `compositor/boundary_fusion.py:52-74,86-118`). | The planned mask/blend work is not evidence of an executed swap: the renderer gate prevents an end-to-end face layer from reaching it. |
| Encode and publish | `video_job_runner.stream_video` pipes FFmpeg decode → frame callback → NVENC/libx264 writer → audio remux → output verification (`pipeline/video_job_runner.py:129-178,291-355`). The standard pipeline has a writer handoff (`engine/pipeline_runner.py:1023-1047`; `media/video_writer.py:129-224,379-400,540-618`). | Both the default stream runner and `photoreal_3d_ffmpeg` fail the true-3D readiness guard before their render loop (`optimized_processor.py:226-242`; `video_job_runner.py:359-384`). The encoder is present but unreachable for a valid swap in the current wired path. |

### Startup and job-admission defects

- `server.__main__` imports `app.api.create_unified_app` when import succeeds (`server/__main__.py:28-32`), then calls `create_app(settings, share=share)` (`:121-128`). `create_unified_app` accepts only `settings` and keyword-only `repo_root` (`app/api.py:41-46`). Thus the preferred import branch raises a keyword-argument `TypeError`; fallback is not attempted. This is a source-derived launch blocker, **not runtime reproduced**.
- If that signature issue were absent, the unified factory passes an explicit `repo_root` (`app/api.py:43-45`), while `create_app` defaults `worker_enabled = repo_root is None` (`server/app.py:74-80`). The unified route would therefore have no worker unless it explicitly enables one.
- Capabilities report `execution_mode.swap: None`, advertise `local_video_render`, and set queue availability based on worker existence; renderer readiness is a separate false feature (`server/routes.py:112-147`). `/jobs` calls `engines_registry.validate` plus path/inspection/mapping validation, not renderer readiness (`routes.py:629-671`). `App.tsx:985` disables queue based on `reason('queue')`, not true-3D readiness. The UI/API can present a render as queueable even though the runner is guaranteed to fail.

## 2. Legacy 2D / 2.5D code and mode inventory

### Swap implementation search

A repository-wide source scan for `cv2.warpAffine` found **71 occurrences across 37 Python files**, including tests and tools. No production call path was found that takes a five-point-aligned identity-swap model output and pastes it into the target frame. Exact-name search found no production invocation of `inswapper_128`, `simswap`, `ghost`, or `hyperswap`, and no production `fallback_2d` key. This matches the source’s explicit removal statements (`optimized_processor.py:226-242`; `reconstruction/neural_face.py:1`; `core/workstation.py:19-41`). The installed model files are not proof those engines are used.

`warpAffine` hits fall into these non-swap categories:

- **Crop/alignment for model inputs:** `faceset/recognizer.py:112`; `faceset_builder.py:143`; `faceset_loader.py:146,235`; `engine/detection/dense_landmarks.py:53,120`; `engine/volumetric/preprocess.py:62,92`; `expression_restorer.py:44`; `reconstruction/full_head.py:417,422`; `reconstruction/morphable_backbone.py:235,259`; `reconstruction/native_backbone.py:161,213-215`; `reconstruction/neural_face.py:172,243`.
- **Input transforms:** display rotation in `media/video_reader.py:56-78`; rotation recovery in `engine/detection/robust_detector.py:226`.
- **Masks, coverage, compositing buffers:** `compositor/boundary_fusion.py:45`; `compositor/pipeline.py:888,954,1181,1209`; `compositor/robust_fusion.py:86`; `engine/edit_plan.py:115,121`; `engine/volumetric/angle_diagnostics.py:138`; `media/vfx.py:69-96`; `pipeline/photorealistic_swap_core.py:327,378,380`; `pipeline/session_adapter.py:205,222,235`; `vfx_export.py:165,175`.
- **UV support registration, not swap paste-back:** `faceset_synthesizer.py:151-165` aligns projected 3D mesh UV/silhouette buffers to generated view landmarks; pixel placement is subsequently through mesh visibility/UV logic. `_run` generates pose variants and registers them into UV (`faceset_synthesizer.py:181-252`); it is a FaceSet synthesis tool, not target-video identity swapping.
- **Tests/tools only:** `tests/identity_world.py:119`; `tests/test_extreme_pose_3d_swap.py:160,327`; `tests/test_occlusion_compositor.py:515`; `tests/test_occlusion_real_models.py:237-238`; `tests/test_resampling_borders.py:49,67`; `tests/test_stage06_pipeline.py:177`; `tests/test_stage09_compositor.py:115`; `tests/test_subpixel_compositor.py:19`; `tests/test_w2_temporal_core.py:82`; `tests/test_zero_2d_and_25d_policy.py:617-623` (synthetic policy fixtures); `tools/occlusion_acceptance.py:241-267`; `tools/prepare_relighting_case.py:49`; `tools/temporal_stability.py:51,53`; `app/tests/test_pipeline_audit.py:136` (audit event string).

Other likely-to-confuse code:

- `reconstruction/tps_fallback.py:1` has a stale module docstring mentioning a “2.5D Thin-Plate-Spline / Delaunay mesh warper.” Its implementation is canonical 3D landmark generation and PnP (`:28-168,175-267`); there is no TPS image-warp function there. Its exports are `get_canonical_3d_landmarks` and `solve_pose_pnp` (`reconstruction/__init__.py:71-74`). `reconstruction/reenactment.py:1-7` says the former 2.5D rig/painter renderer was removed; remaining code measures expression only.
- `rendering/hybrid_renderer.py:553-653` names a `low_vram_uv_warp`, but it projects a 3D mesh, samples its UV atlas, and rasterizes with perspective/depth testing. It is not a 2D crop swapper. The renderer hook remains unwired.
- `heterogeneous_scheduler.py:205-255` retains a dormant swap stage whose default is `lambda img: img`; it can count a swap while returning the unchanged image if called without a `swap_func`. No production caller was found in the current app path. This is not a working swap engine.
- `models/` contains `inswapper_128_fp16.onnx` and three Hyperswap weights, but current source does not route these through a target-frame render path. Existing references in UI tests and old result artifacts do not establish a live engine.
- **Direct generic image-layer helper:** `bridges/comfyui_bridge.py:433-441` fetches an external ComfyUI image crop, wraps it as a `Replacement` with an affine inverse, and passes it to the compositor. This is a possible 2D patch-composite helper if explicitly invoked; it is not a mesh projection or named identity-swap model. No production route/caller was found, and it is outside the current job runner.


### Render-mode/API controls

- The only render mode is `full_3d_head`; old IDs are accepted and normalized to it (`server/unified_state.py:25-55`; `ui/src/components/workstation/renderModes.ts:1-29`). Legacy IDs include `standard_2d`, `estimated_2_5d`, and older 3D labels (`unified_state.py:35-50`).
- `App.tsx:128,961-995` has a `centerMode: '2d' | '3d'` switch. Its labels are **Target 2D Canvas** and **3D Identity Viewport**; it changes the displayed viewer, not the render engine. `SwapScopeSelector.tsx:33-36` has stale copy referring to a “plain 2D swap.”
- `execution_mode` in `core/settings.py:64-68` still lists `swap`, `volumetric`, `hybrid`, `diffusion_vfx`; `/system/capabilities` still marks `swap` selectable, while `hybrid`/`diffusion_vfx` say no verified diffusion loader (`server/routes.py:120-123`). These are stale/other execution settings, not a 2.5D render toggle. `fallback_2d` has no production source match.
- Old workstation settings such as `render_mode`, swapper/restorer/fidelity/pose-frontalization fields are accepted for migration and dropped (`core/workstation.py:19-39,107-123`; `server/true3d.py:1-18,94-130`). The true-3D guard explicitly bans screen-space/2D/2.5D projections and swapper/restorer keys (`core/true3d_guard.py:97-104`).

## 3. Issue 1 — engine availability, manifests, governance, venv, providers

### Current blockers

- **Renderer not wired:** `pipeline/true3d_renderer.py:1-53` returns false from `is_wired()` and raises from preview/live/render calls. `core/true3d_guard.py:35-37,89-104,588-602` translates this into `RENDERER_NOT_WIRED`, HTTP 503, and explicitly refuses 2D/2.5D fallback. `server/true3d.py:54-60,94-130` and `/api/v1/health` report unavailable readiness (`server/app.py:258-265`). `EngineRegistry.selectable()` includes renderer readiness (`engine/registry.py:55-69`).
- **Default and opt-in jobs both stop before render:** `make_stream_runner` calls the guard at the start of `run_prepared` (`optimized_processor.py:226-242`); `EndToEndVideoJobRunner.run_prepared` does the same before FFmpeg/decode setup (`video_job_runner.py:359-388`). The session also raises for renderable faces rather than returning a fake frame (`engine/volumetric/session.py:488-544`).
- **Potential launch blocker:** factory mismatch above takes precedence over every engine check if the preferred import branch is used.

### Model governance and integrity

- `ModelCatalog` validates supported registry/catalog structure on construction (`models/catalog.py:84-114`), rescans local files (`server/app.py:122-125`), verifies registry artifacts by expected SHA-256 and size (`catalog.py:251-283`; `models/verifier.py:21-65`), and applies lifecycle/license/compatibility checks in `resolve_for_inference` (`catalog.py:546-560`).
- Files copied directly into `app/models` are classified as local-trusted and can bypass the registry lifecycle/license gate (`catalog.py:90-97,285-294,546-560`). That is the local-use policy, not proof the bytes are the upstream pinned artifact or commercially licensed. The recorded license review states InsightFace model use is noncommercial/research-only (`models/governance_reviews.json:11-31`).
- A malformed universal manifest logs and disables its model-manager endpoints (`server/app.py:56-71`); it is not the same as the true-3D render gate. A missing candidate without a pinned checksum/source produces `MODEL_NOT_INSTALLED` with an explicit reason (`catalog.py:319-342`).
- FLAME is not available/used for the saved `harjot` asset: the cached result says registry status `blocked`, weights unavailable, and academic license not accepted (`outputs/test 1/phase1_verify/faceset_result.json`, `build.flame`). This is a FaceSet reconstruction limitation; source supports other reconstruction paths, so do not treat FLAME alone as the global render blocker.
- The `LandmarkEngine` candidate path under a sibling `roop-ultimate` model directory bypasses `ModelCatalog` hash verification (`faceset_loader.py:49-59,85-103`). This audit did not inspect that directory or any file there.

### Runtime environment and actual provider evidence

The active interpreter observed at `app/roop_3d/.venv/Scripts/python.exe` is Python 3.10.20. Installed distribution metadata and runtime ORT provider query:

| Package/runtime | Observed |
|---|---|
| `onnxruntime-gpu` | 1.23.2 |
| ONNX Runtime available providers | `TensorrtExecutionProvider`, `CUDAExecutionProvider`, `CPUExecutionProvider` |
| TensorRT CUDA 12 bindings/libs | 10.9.0.34 |
| `onnx` / `scipy` / `numpy` | 1.16.0 / 1.15.3 / 1.26.4 |
| OpenCV / PyAV | `opencv-python-headless` 4.11.0.86 / `av` 17.1.0 |
| PyTorch / torchvision | Not installed |
| InsightFace / MediaPipe | Not installed |

`pyproject.toml:10-35` makes ORT GPU, SciPy, PyAV, and OpenCV core dependencies; PyTorch appears only in optional extras. Missing `torch` disables code paths that explicitly require Torch tensors (`rendering/hybrid_renderer.py:24-40`), but its NumPy low-VRAM mesh renderer exists (`:553-653`). Current source’s ONNX detector/recognizer/landmark paths do not require the Python `insightface` or `mediapipe` packages; local ONNX models are used instead. **Provider list is availability, not successful model-session/inference proof.** No ONNX model was loaded or executed during this audit.

**Archived provider diagnostic (not current runtime):** `audit/STAGE07_VALIDATION.json:444-468` records a LivePortrait `warp_spade` graph requested on TensorRT, then failing on CUDA because its 5-D `GridSample` input is unsupported; the captured fallback provider is CPU and the record shows one execution. `audit/STAGE07_REPORT.md:3-5,25-39` likewise says the original warp/SPADE path remains CPU. This is a historical GPU-acceleration limitation with a recorded CPU fallback, not evidence that current ORT inference fails globally. The same report states the `inswapper_128.onnx` FP16 candidate failed numerical parity gates (`:25-31`). Treat these stored results as historical; another archived engine probe is not proof of current session health.


`app/roop_3d/.venv` matches the Pinokio venv path (`start_roop3d.js:37-47`; `.venv/pyvenv.cfg`). No root-level venv mismatch was found. There are TensorRT cache entries on disk, but no current cache/model session was validated. Existing `outputs/test 1/phase1_verify/engines_result.json` is an archived engine probe only; it cannot establish current provider/session health.

## 4. Issue 2 — target scanning, face serialization, UI/API binding

### Scan coverage and detector gates

- The Workstation’s `inspect()` request sets `identity_lock=true`, `full_video=true`, `time_budget_seconds=900` (`ui/src/App.tsx:482-489`). `MediaService.inspect` calls `tracker.process` for each decoded frame (`media/service.py:174-204`). The time check remains active even with `full_video=true`; scan stops after the 900-second budget or when the 64-track limit is reached (`service.py:205-240`).
- `InspectRequest` defaults to threshold 0.5, 32 keyframe samples, a 10,000-frame cap, and a 30-second budget (`media/schema.py:13-23`). `full_video=true` bypasses the frame/keyframe stop bounds, not the time limit. The scanner runs every frame until a stop condition; it does not sample only the `max_samples` frames (`service.py:227-240`).
- Main inspection uses `RobustDetector`. Recovery runs only when the primary frame has no confident face; scales 1.5×/2×/3×, tiles, and rotations ±30/60/90° are configured with geometry/profile checks (`engine/detection/robust_detector.py:14-20,44-68,97-167,169-253`). **Risk:** one ≥0.5 face suppresses recovery for the entire frame, so another missed small/profile face may not be recovered (`:97-120`).
- `FaceAnalyzer` has primary/tolerant score tiers (0.50/0.30), a 24-pixel minimum, and aspect/landmark checks. Sub-0.50 profile candidates need plausible 5-point geometry and, when available, dense 106-point agreement; narrow profiles are rejected if the dense landmark model is unavailable (`media/face_analysis.py:12-20,29-65,82-178`). `/media/detect-frame-faces` instead calls detector directly with the request threshold and does not use the same tolerant tier (`media/service.py:286-354`). This route-to-route difference can make a face visible during inspection/scrubbing but unavailable during locking.

### Serialization/UI binding defects

- The backend stores a track’s box, thumbnail, and initial frame when first seen; later frames update only last-seen/count/pose fields (`media/service.py:205-226`). The response is a `FaceTrack` summary, not a per-frame box/landmark trajectory (`:261-282`; schema `media/schema.py:115-133`). `identity_continuity_verified` remains false; warnings call out uncalibrated identity thresholds and possible splits/merges (`service.py:254-282`).
- `populateFromInspection` synthesizes `sampled_detections` time points between first/last frame, reuses the same bounding box at every point, generates five landmark dots from fixed fractions of that box, and sets confidence to `0.95` (`ui/src/useFaceSelectionStore.ts:312-409`). These are approximations, not observed frame detections; moving faces may show stale positions, and empty intervals can look detected. `getFacesAtTimestamp` chooses the nearest synthetic point (`:416-437`).
- `/media/detect-frame-faces` searches the inspection directory and takes the first JSON with the same target digest; it does not receive the active `inspection_id` (`media/service.py:315-325`). With multiple inspections of the same video, glob order can make frame locking mutate/update an older inspection (`:447-469`).
- Inspection IDs and frame/target-lock IDs come from separate registries. They both use `track_NNNN`, but their allocation/order is independent (`media/service.py:174-199`; `identity/target_lock.py:175-195`; `media/face_analysis.py:167-178`). `matchFrameFaces` matches any non-null frame ID only by exact equality and returns without IoU fallback on mismatch; IoU≥0.3 is used only when the frame ID is absent (`ui/src/utils/frameFaces.ts:35-65`). A same-string collision can bind the wrong stored identity; a mismatch can leave a detected face unbound. This is a source-proven failure mode, not confirmed on a supplied clip.
- Raw one-frame `face_id` is frame-local (`frame_index_rank`), with `track_id/cluster_id` initially null (`media/face_analysis.py:181-203`). Main video inspection correctly serializes track IDs; the problem is the distinction among these ID domains and the synthetic UI representation, not absence of an inspection response.

**Saved detector evidence:** `outputs/test 1/detect_eval/dense.log` is a prior detector-only run. It reports hits of 349/400 on `s5` (1,135 total), 353/400 on `s6` (1,335 total), 307/400 on `s7` (600 total), 329/381 on `s9`, and 320/381 on `s10`; other listed clips have 100% in the sampled subset. Several long videos were sampled only to 400 frames, so these are neither full-video recall nor current-run results. They show the detector path has historically missed some sampled frames; no video was decoded for this audit.

## 5. Issue 3 — target identity fragmentation (`face 2, 3, 4, 5…`)

- The UI’s full inspection uses `TrackPolicy` with `FaceTracker` (`media/service.py:184-199`). New unmatched detections receive new IDs; tracks can expire after 15 missed frames or 45 uncertain frames. Low-score detections down to 0.1 can continue an existing track by IoU, but cannot start one; the high threshold is 0.5 (`engine/detection/face_tracker.py:105-173,243-338,457-590`).
- Policy thresholds are heuristics: appearance/continuity/weak-cosine tiers, stricter reacquisition around cosine 0.50 plus margin 0.08, and continuity IoU constraints (`engine/detection/track_policy.py:1-17,40-78`). Profile, occlusion, blur, lighting, or a scene cut can lower embeddings/break IoU; once a track expires, a subsequent observation is a new numbered ID. This explains a plausible split but does not prove that a given person is split in the supplied videos.
- Separate target-angle registry uses Hungarian matching with cosine minimum 0.42, strong ≥0.55, and creates a new cluster for unmatched faces (`identity/target_lock.py:12-15,175-284`). This registry is not the inspection tracker. Both allocate labels like `track_0001`, so equal IDs are not evidence of equal identity.
- FaceSet quality’s identity clustering uses another ArcFace cosine threshold of 0.30 (`reconstruction/faceset_quality.py:27-35,178-215`). These three thresholds govern different workflows and should not be treated as one global identity policy.
- The UI’s synthetic tracks reuse a static box and assign `0.95` confidence (`ui/src/useFaceSelectionStore.ts:334-387`), which can visually suggest persistence despite no per-frame evidence. There is no supplied-video identity report establishing actual fragmentation.

## 6. Issue 4 — angled/profile frames and pose failures

- The target scanner does not apply a global yaw/pitch/roll rejection gate. `FaceInspection` records pose ranges and labels yaw/pitch as uncalibrated (`media/service.py:261-282`). `FaceAnalyzer` uses score, face size, aspect, and landmark consistency instead (`media/face_analysis.py:29-65,82-178`).
- Profile risk is concentrated in the tolerant path: low-score face candidates need dense-106 agreement; narrow aspect-ratio candidates are rejected if the dense model is unavailable. Robust recovery is helpful but only runs when no confident primary face exists and stops on the first recovery transform returning valid faces (`robust_detector.py:97-167,169-253`). This can miss a profile face in a multi-person frame where another face already passes.
- `FaceSet` quality imposes an absolute yaw ceiling of 88° and rejects pose/landmark disagreement >25° (`reconstruction/faceset_quality.py:27-35,131-166`). The target-angle bank only records bins for |yaw|≤75° and |pitch|≤45° (`identity/target_lock.py:24-28`); out-of-grid poses lose coverage, not target detection.
- PnP can fail on near-profile/self-occluded landmarks; `faceset_3d._pose` may fall back to stored pose (`reconstruction/faceset_3d.py:25-46,60-78`). Inspection serializes `PoseRange.method="five_point_geometric_estimate"` regardless of whether dense PnP contributed (`media/service.py:267-276`). A reported zero reprojection error in fallback data is not proof PnP succeeded (saved-source caveat; `faceset_3d.py:60-78`).
- `PoseFade` declares a 65°–82° fade but the reference session only retains/resets the state; it does not call `level()` in the frame path (`engine/volumetric/presence.py:1-67`; `engine/volumetric/session.py:132-135,514-520,664-666`). It is not a target-scanner rejection gate.
- **Saved `harjot` coverage:** yaw approximately −78.8°…+78.92° (157.72° span), pitch span 11.07°, no looking-up/down views, and only 5-point landmarks (`outputs/test 1/phase1_verify/faceset_result.json`, `build.coverage`). Source profile views exist, but vertical expression/forehead/chin and back-of-head support are weak or absent. The archived report calls the asset `limited_multiview`, `depth_triangulated=false`, `full_head_geometry=false`, with FLAME unavailable/unlicensed.

## 7. Issue 5 — partial coverage, hair/forehead/jaw/chin, seams, and lighting

- **Full-head topology gate:** `GuardPolicy.min_vertices=1500`, min depth/height 0.45, back-of-head fraction 0.18, azimuth coverage 13/16, and required anatomical regions/parts (`core/true3d_guard.py:97-120,350-425`). The saved harjot mesh has 1,129 vertices/2,208 faces and declares itself facial-region geometry; skull, ears, and back of head are not reconstructed (`outputs/test 1/phase1_verify/faceset_result.json`, `build.asset`). If supplied as the full-head mesh, it is structurally below the configured minimum and incomplete.
- The cached asset has `vertex_observed=0`, `vertex_interpolated=1110`, `vertex_unseen=19`, and a 512² texture with 0.989 observed texel ratio. Texel coverage is not geometry coverage: most mesh vertices are interpolated from the template, while texture lighting/specular residuals may remain baked in (`faceset_result.json`, `build.asset.statistics` and `build.asset.limitations`). The report’s `readiness="full_coverage"` means its sampled pose bins, not a complete full-head surface; the same object lists missing up/down views and false full-head claims.
- `FullHeadUVCompleter` can fill/symmetry-project unseen texture but marks these values generated/display-only, not observed (`reconstruction/uv_completion.py:1-15,188-203`). `UVSynthesizer` uses surface visibility/z-buffering and reports observed/interpolated/unseen coverage (`reconstruction/uv_synthesizer.py:89-153,372-603`). Neither can invent reliable hairline, ears, skull or neck detail absent from source geometry/views.
- The render crop is intended to be derived from projected mesh extents with padding (`reconstruction/full_head.py:46-55`). `Photorealistic3DSwapCore._rectify` warps a **rendered crop and its coverage** to frame space with border replication; it is not a five-point source crop (`pipeline/photorealistic_swap_core.py:369-392`). Its edit geometry below the brow line assumes source photographs already cover the forehead (`:345-359`). Coverage masks cannot add unseen pixels beyond their support; tight source geometry therefore remains a likely clipping/edge risk.
- The compositor does include color and seam handling: LAB/skin-isolated matching, local illumination, multiband fusion, jaw/SDF feathering and hairline/occlusion controls (`compositor/pipeline.py:48-63,391-417,494-527,674-762,829-832`; `compositor/boundary_fusion.py:52-74,86-118`). Defaults include `blend_mode=multi_band`, four pyramid levels, 5 px feathering, color strength 0.65, jaw feather 6 px, standard occlusion mode, and hair/neck/ear protections (`core/workstation.py:28-95`; compositor options `pipeline.py:80-165`). Color matching becomes unavailable with fewer than two usable skin samples (`pipeline.py:494-527`).
- Parser failure is a concrete mask risk: `compute_dual_bisenet_forbidden_mask` can return an empty forbidden mask when parser is missing/fails, so semantic hair/neck/ear exclusion is absent on that path (`compositor/masks.py:41-89`; `compositor/pipeline.py:689-707,1017-1020,1129-1135`). Standard composition can still fall back to a landmark hull, but that does not restore the missing semantic exclusion. More robust XSeg/hair parsing is optional.
- **No visible seam was reproduced.** Current renderer gating prevents a target face from reaching these compositor stages. Findings above are code/asset-based seam risks, not visual measurements from the supplied videos.

## 8. Environment and requested assets

### Pinokio interpreter/packages

The active venv is `G:\pinokio\api\roop-3d\app\roop_3d\.venv` (Python 3.10.20). Package versions and provider availability are as listed in §3. No `torch`, `torchvision`, `insightface`, or `mediapipe` distribution metadata was found. No package was installed or changed. The project dependencies distinguish core ORT from optional Torch extras (`app/roop_3d/pyproject.toml:10-35`).

### Local model files

`G:\pinokio\api\roop-3d\app\models` contains the following 19 ONNX files (plus `inswapper_128_emap.json/.npy`):

- `GFPGANv1.4.onnx`, `bisenet_resnet_34.onnx`, `GPEN-BFR-512.onnx`, `gpen_bfr_512.onnx`, `xseg.onnx`, `occluder.onnx`, `ultramax.onnx`;
- `hyperswap_1a_256.onnx`, `hyperswap_1b_256.onnx`, `hyperswap_1c_256.onnx`, `inswapper_128_fp16.onnx`;
- `w600k_r50.onnx`, `arcface_w600k_r50.onnx`, `face_recognition_sface_2021dec.onnx`, `face_detection_yunet_2023mar.onnx`, `det_10g.onnx`, `scrfd_2.5g.onnx`, `2d106det.onnx`, `1k3d68.onnx`.

`app/models/.trt_cache` contains TensorRT engine/profile/cache artifacts. `app/roop_3d/models/cache` contains model-cache artifacts. This is a filesystem inventory; I did not hash/load each large model. File presence does not establish that the current runtime can create an inference session or that a model is selected by the active pipeline.

### Harjot FaceSet

- `G:\pinokio\api\roop-3d\app\facesets\harjot.fsz` exists (about 1.2 MB). A directory named `facesets/harjot/` was not found.
- `.runtime/facesets/797577eca6ab4cf88c7b3e075b590979.json` identifies `name="harjot"`, `item_count=5`.
- `outputs/test 1/phase1_verify/faceset_result.json` is archived build metadata for the same ID. It reports five usable views, five-point landmarks, limited multiview/face-only geometry, missing vertical views, and no verified/accepted FLAME weights. The `.fsz` itself was not unpacked or visually inspected.

### Input videos

`G:\pinokio\api\roop-keep\single` contains 12 `.mp4` files. Sizes below are the directory listing’s rounded display values; clips were not decoded.

| File | Displayed size |
|---|---:|
| `s.mp4` | 1.4 MB |
| `s1.mp4` | 483 KB |
| `s2.mp4` | 709 KB |
| `s3.mp4` | 5.2 MB |
| `s4.mp4` | 39.3 MB |
| `s5.mp4` | 31.8 MB |
| `s6.mp4` | 42.2 MB |
| `s7.mp4` | 5.6 MB |
| `s8.mp4` | 41.7 MB |
| `s9.mp4` | 1.2 MB |
| `s10.mp4` | 2.0 MB |
| `s11.mp4` | 435.7 KB |

### Output directory

`G:\pinokio\api\roop-3d\app\roop_3d\outputs\test 1` exists and contains roughly 360 files / 15 MB of **pre-existing** logs and verification artifacts. Relevant saved files include `phase1_verify/faceset_result.json`, `phase1_verify/engines_result.json`, and `detect_eval/dense.log`. These are historical artifacts, not fresh output from this audit or a guarantee of current model/provider behavior.

## Audit limits

This was read-only. No app startup, UI session, model inference, output render, test suite, or video decode was run. The binary FaceSet, input videos, and model weights were not opened; package status came from the configured venv’s distribution/provider metadata. Static call-path findings are distinguished from archived results. The startup-factory mismatch and hypothetical face-identity collisions are not runtime-reproduced.

## Stage 0.5 Purge Report: pure-3D full-head reconstruction only

**Date:** 2026-11-15. **Scope:** `app/roop_3d/`, `app/roop_3d_ui/src/`, `app/api.py`. Evidence labels: **ran** = executed in this session; **not run** = not executed.

### Mandate and result

No 2D planar swap, 2.5D landmark-warp, crop paste-back, mode toggle or silent fallback remains in the active code. The one required frame path is `3D Head Pose & Volumetric Mesh Fitting -> 3D UV Identity Texture Projection -> 3D Mesh Rasterization & Lighting -> Hair-Excluding Semantic Mask -> Soft Feathered Edge Blend`. **Blocker unchanged: `true3d_renderer.is_wired()` is still false, so every render request still fails closed (HTTP 503 `RENDERER_NOT_WIRED`). This stage removed the alternatives; it did not wire the renderer.**

### 1. 2D / 2.5D code paths removed

| Area | Removed |
|---|---|
| Compositor | `compositor/pipeline.py` (crop-based `FrameCompositor`, `CompositeOptions`, `_remove_old_silhouette`, expression/LivePortrait hooks, affine paste-back). `compositor/__init__.py` exports only mask/fusion primitives. |
| Image-warp silhouette stack | `rendering/silhouette_replacement.py`, `rendering/silhouette_inpaint.py`, `rendering/perimeter_cage.py`, their lazy exports, and the `swap_scope` / `skull_blend` fields. |
| Expression retargeting | `expression_restorer.py`, `engine/volumetric/reference_adapter.py`, `liveportrait_trt.py` (LivePortrait warp/stitch crop retargeting). |
| Swap core | `pipeline/photorealistic_swap_core.py` now accepts only full-frame rasters that carry the ordered stage trace; geometry/rectify branches, source-photo patching and unprojected photo fill are gone. `pipeline/session_adapter.py` builds a full-frame `FacePacket` from the real 3D layer. |
| Session | `ReferencePipeline` has no swapper, portrait/LivePortrait, TPS or compositor state; `infer_frame` fails closed through `require_true3d_renderer`. `retarget`, `render_mode`, `pose_blend` and `session_options()` / `inference_options()` plumbing are gone from the runners. |
| Tools | `tools/occlusion_acceptance.py` (exercised the removed compositor). |
| Dead helpers | `angle_tracker.normalize_render_mode` / `RENDER_MODE_TO_ENGINE`, runtime-registry swap filters (`_REMOVED_SWAP_ROLES`, `_REMOVED_SWAP_TOKENS`), `execution_mode`, `swap_weight` / `swap_opacity` curves, batch-recipe `swap_weight`. |

### 2. Mode switches, config keys, schema fields, UI controls removed

Backend: `Settings.execution_mode` (`swap` / `volumetric` / `hybrid` / `diffusion_vfx`), `render_mode`, `active_processors` and its `/api/v1/processors` routes, `swap_scope`, `skull_blend`, `expression_restore_factor`, `face_expression_overrides`, `lip_sync_*`, `supports_3d_hybrid`, `adapter_supported` plumbing, and the `fallback_2d` / `use_3d=False` flags. Settings, workstation and live-preview contracts are `extra="forbid"`; removed fields fail validation instead of being silently dropped. `unified_state` is now sources, target media and `true3d` settings only (`POST /api/v1/state`).

Frontend: the target-2D / identity-3D center-mode toggle in `App.tsx`, `renderModes.ts` / `RenderModeId`, `SwapScopeSelector`, `FaceExpressionEditor`, `arkit52`, the swap-weight (identity weight) setting and curve, the expression/gaze edit gizmos (`withFaceEdit` / `faceEditOf`), the telemetry swap-mode badge (now `PURE 3D HEAD`), `supports_3d_hybrid` and `mesh_2_5d` types. `migrateParameters` rejects unsupported keys instead of dropping them.

### 3. Model registry pruned to pure-3D components

Removed from `models/model_manifest.json`, `stage16_models.json`, the orchestrator `engines_manifest.json` / `render_profiles.json`, `supported_models.json` and `model_registry.py`: the LivePortrait appearance, motion, warp and stitch checkpoints (12 manifest records), `liveportrait_retarget`, the Stage 16 LivePortrait / generator records, and the `liveportrait_*` `Role` literals. Retained: detectors (SCRFD / RetinaFace / YOLOFace), ArcFace recognizer, 1k3d68 / 2d106 / 203-point landmarks, MakeHuman / FLAME mesh assets, `bisenet_resnet_34` and XSeg semantic parsers, RealESRGAN UV/skin detail upscalers. `rg -i "inswapper|simswap|hyperswap|ghost"` finds no match in any manifest. No file in `roop-ultimate` was touched.

### 4. Hard runtime guard

`core/true3d_guard.py` defines `FRAME_PIPELINE_STAGES` and `require_true3d_frame_pipeline(stages, through=...)`, which raises `True3DFramePipelineError` (`TRUE3D_FRAME_PIPELINE_INVALID`, HTTP 422) with `expected`, `observed` and `next_stage` if a frame lacks or reorders a stage. It is asserted at three points: replacement attestation (`SessionFrameProcessor._packet`, stages 1-3), semantic-mask completion (stage 4) and the blend (`Photorealistic3DSwapCore.process_face`, stage 5). A fit failure (`HEAD_POSE_FAILED`, `HEAD_CORRESPONDENCE_*`, `DENSE_HEAD_LANDMARKS_REQUIRED`, `HEAD_BEHIND_CAMERA`, `TRUE3D_SEMANTIC_*` and related codes in `SessionFrameProcessor.FIT_SKIP_CODES`) logs a structured `true3d_frame_skipped` warning, records it in `FrameReport.skipped` and returns the frame untouched. Every other error (including `RENDERER_NOT_WIRED`) propagates. **ran:** throwaway probe, then `tests/test_pure_3d_frame_guard.py` (new, passes), covering ordering rejection, skip diagnostic with an untouched frame, and no swallowing of non-fit errors.

### 5. Verification gate

| Check | Result |
|---|---|
| **ran** `rg -i` for `fallback_2d`, `use_3d=False`, `estimated_2_5d`, `standard_2d`, `inswapper`, `simswap`, `hyperswap`, `face_swapper`, `swap_scope`, `skull_blend`, `swap_weight`, `execution_mode`, `LivePortraitAdapter`, `liveportrait_*` roles, `expression_restorer`, `reference_adapter`, `perimeter_cage`, `silhouette_replacement` over `app/roop_3d`, `app/roop_3d_ui/src`, `app/api.py` (excluding tests, docs, audit history, venvs) | zero matches |
| Remaining `2d` / `2.5d` text | Guard vocabulary in `core/true3d_guard.py` (the banned-token list that rejects those shapes), explanatory docstrings, and HTML canvas `getContext('2d')` for mask-brush overlays; none is a swap path |
| **ran** import of every `roop_3d.*` module | clean except `reconstruction.lbs` and `reconstruction.monocular_regressors`, which need `torch` (not installed in the venv; optional, pre-existing) |
| **ran** `tests/test_zero_2d_and_25d_policy.py` (AST import-graph gate: swapper ids, 5-point paste, 2.5D, post-crop GAN, OpenAPI/`contracts.ts`) | 20 passed |
| **ran** `tests/test_pure_3d_frame_guard.py` | 7 passed |
| **ran** `test_async_runner`, `test_async_pipeline`, `test_stage12_worker`, `test_identity_lock`, `test_stage02_jobs`, `test_stage02_settings`, `test_stage13_bridges`, `test_batch_matrix_and_checkpoint_history` | all pass |
| **ran** `npx tsc --noEmit` and `npx vitest run` in `app/roop_3d_ui` | no type errors; 89 files, 869 tests passed |
| **not run** | the full backend suite to completion (it exceeds the per-run budget here), ESLint, any GPU / model inference, and any real render (renderer is unwired) |

### 6. Tests removed or changed

Deleted because they exercised only deleted code: `test_silhouette_replacement`, `test_dense_compositor`, `test_occlusion_compositor`, `test_occlusion_real_models`, `test_stage09_compositor`, `test_stage10_audio_sync`, `test_stage15_linear_light`, `test_temporal_consistency`, `test_visual_quality_regressions`, `test_hair_neck_protection_controls`, `test_photometric_head`, `test_photorealistic_swap_core`, `test_workstation3d_controls`, `test_workstation_preview`, `test_animation_history_roles`, `test_expression_and_synthesis`, `test_stage06_pipeline`, `test_extreme_pose_3d_swap`, `test_full_head`. **Coverage cost:** these also held checks on retained primitives (matte, hairline, linear-light blend, temporal stabilizers) that were only reachable through the deleted `CompositeOptions` / `FrameCompositor`; those primitives now have no direct test and should get new tests against the 3D compositor.

Rewritten: the identity-lock session gate tests (now assert withholding and fail-closed behavior), the policy allow-list, session stubs, and the frontend tests that referenced removed controls.

### 7. Known issues left (not caused by this stage)

- Many backend test files fail or error at collection from the earlier Phase 0 purge (examples: `test_stage14_orchestrator`, `test_stage03_registry`, `test_stage06_model_hub`, `test_head_tracking_3d`, `test_head_tracking_stream`, `test_identity_asset_session`, `test_remediation_stage2`, `test_flame_and_liveportrait`); they reference deleted swapper roles, engines and modules and need a separate test-suite rewrite. I did not baseline them against the pre-stage tree.
- `reconstruction/reenactment.py` lines 155-157 use an undefined name `group` (pyflakes); it was not edited here.
- Compositor modules `boundary_fusion`, `robust_fusion`, `masks` and others remain and are used only by the 3D path; `reconstruction/tps_fallback.py` is kept for its canonical landmark set and PnP solver (the 2.5D warper is gone; the docstring now says so).

## Stage 1 — Pure-3D model availability and execution providers

**Scope:** model inputs, real standalone component forwards, provider initialization, and component-status API/UI. This does **not** wire the production renderer or certify a finished video render.

### Repairs

- Added `models/pure3d_models.json` and `models/pure3d.py` for explicit required-component identities, verified local resolution, acquisition instructions, and head-asset integrity checks. The existing mathematical `UVSynthesizer` needs no neural texture-generator checkpoint.
- Unified model identities with the authoritative registry; removed fabricated universal hashes/URLs. Local placement, exact registry-byte identity, license review, successful initialization, and executed inference are separate facts. Local-use policy does not grant commercial rights.
- Prefer the resolvable Buffalo `det_10g.onnx` artifact rather than a same-byte SCRFD alias with incomplete governance metadata. Detector selection now checks inference eligibility, not only file presence. FaceSet landmark loading uses the registry instead of opening sibling model paths directly.
- Default ONNX sessions try CUDA, DirectML, then CPU. TensorRT remains explicit opt-in. Actual initialization/forward exceptions retain their text; advisory preload/conversion warnings do not turn a successful load into an unavailable engine. Fixed scalar fidelity-input binding and FP16 image/DOUBLE scalar inputs for the real UltraMax graph.
- Added lazy optional PyTorch CUDA/DirectML/CPU tensor-operation probing in `core/torch_device.py`; wired automatic splat device selection and explicit UV GPU sampling to it. Explicit sampling failure raises a structured diagnostic instead of silently executing CPU. **PyTorch is not installed in this venv:** its structured `PYTORCH_NOT_INSTALLED` diagnostic was exercised, not successful PyTorch GPU execution. Required verified modules below use ONNX Runtime or NumPy/OpenCV/SciPy.
- Hair-exclusion parser absence/failure now raises a structured error instead of returning an empty forbidden mask.
- `GET /api/v1/models/components` and the workstation Engine & Model Rack display resolved file, integrity/license evidence, actual resident provider/load/error diagnostics, fallback warnings, and file-identity-bound standalone forward receipts. Saved forwards never imply resident sessions or production render readiness.
- Fixed the launcher factory mismatch. Removed 13 retired Stage-0.5 fields from the existing ignored runtime config so current Settings can load; retained remaining user settings. Normal startup then exposed a separate `JOB_STORE_INVALID` failure. Persisted jobs were preserved, not rewritten or deleted.

### Stage 1 Verification Report

**Ran from `app/`, using the active Pinokio interpreter:**

```bat
"roop_3d/.venv/Scripts/python.exe" -B -m roop_3d.tools.verify_stage1
```

Test image: actual `0.png` in `app/facesets/harjot.fsz`. Each row executed its real adapter/mathematical implementation. No fake session or replacement engine was used.

| Component | Result | Actual execution backend |
|---|---|---|
| Buffalo detector | PASS; 1 face, confidence 0.7776 | CUDAExecutionProvider |
| ArcFace embedder | PASS; finite unit 512-vector | CUDAExecutionProvider |
| 1k3d68 landmarker | PASS; 68 x 3 | CUDAExecutionProvider |
| Dense contour landmarker | PASS; 106 x 2 | CUDAExecutionProvider |
| MakeHuman head reconstructor | PASS; fitted 4,289 vertices / 8,524 triangles; RMS 1.8784 px | CPU, NumPy/SciPy |
| UV texture synthesis | PASS; 128 x 128 atlas with observed texels | CPU, NumPy/OpenCV |
| Perspective UV mesh rasterization | PASS; 3,866 visible triangles / 39,134 covered pixels | CPU, NumPy/OpenCV |
| Hair-excluding semantic parser | PASS; normalized 19-class probabilities | CUDAExecutionProvider |
| XSeg mask | PASS; finite 1 x 256 x 256 x 1 tensor | CUDAExecutionProvider |
| UltraMax canonical-UV detailer | PASS; 1024 x 1024 atlas, 325,479 changed channel values | CUDAExecutionProvider |

**10/10 passed.** Machine: RTX 4070. Structured receipts: `app/roop_3d/.runtime/stage1_verification.json`; reproducible verifier: `app/roop_3d/tools/verify_stage1.py`. The reconstruction pass is a single-view template fit, not proof of observed back-of-head identity. UltraMax normalization remains the existing CodeFormer RGB [-1,1] convention, not independently verified upstream preprocessing or photorealism.

### Additional observed checks and limits

- Focused receipt/governor/hardware tests: **13 passed**.
- Frontend `npm run typecheck` and `npm run lint`: **passed**.
- Provider-order/DirectML-admission smoke and semantic missing-parser refusal: **passed**. DirectML is not installed in this ORT build; no DirectML hardware execution is claimed.
- Browser inspection and screenshots exercised the actual workstation diagnostic panel. API returned HTTP 200, all ten top-level standalone receipts verified, and `render_ready=false`. This diagnostic-only launch explicitly disabled jobs and lifespan because normal startup stops at the preserved invalid job store; it is not proof of normal application startup.
- No packages, model weights, FaceSets, or sibling-application files were changed. No new download was needed: all required inputs were present.
- **Production blockers remain:** `RENDERER_NOT_WIRED` and persisted `JOB_STORE_INVALID`. A passing component canary does not make an unwired render engine selectable.
- Real missing-DirectML -> CPU detector canary: **passed**, nine finite output tensors; exact DirectML-unavailable diagnostic retained without marking the CPU-loaded component unavailable.

## Stage 2 — Target face detection, multi-angle recall and the `harjot` reference bank

**Scope:** target scanning, detector calibration, UI/API binding, FaceSet loading and the multi-angle reference bank. The renderer is still unwired (`RENDERER_NOT_WIRED`) and the persisted job store is still `JOB_STORE_INVALID`; neither was touched.

### Root causes found and repaired

| Defect | Evidence | Repair |
|---|---|---|
| Default inspection looked only at the head of the file | `InspectRequest` defaults (32 keyframes / 30 s budget) stopped after ~2 s at ~0.5 s per analysed frame | `sampling="adaptive"` is the default: one decode pass over the **whole** timeline (`media/adaptive_sampling.py`). Stride relaxes 1,2,4… to 0.25 s (face present) / 0.5 s (none); a cut, a track appearing or vanishing, a ≥8° pose jump, motion or rescale, or any head past 30° yaw / 22° pitch collapses it (steep frames ≤2 apart). A measured cost per sample stretches the stride so the sweep ends inside the time budget instead of truncating. `every_frame` + `full_video` remain the exhaustive legacy scan. |
| UI invented its detections | `populateFromInspection` replicated one first-seen box, five fixed-fraction landmark dots and `confidence: 0.95` across the track, with ids `track_frame` that could collide when `sampled_detections` exceeded the frame span | `FaceTrack.observations` carries real per-sample box, 5 landmarks, pose and score (≤400 per track, evenly thinned, pose extremes kept). The store builds faces from them; a scan saved before this change shows one real box at its first frame and nothing invented. |
| Frame-lock ids were another id domain | `/media/detect-frame-faces` numbered clusters by registry order and took the first same-digest JSON in directory order | The route takes the workstation's `inspection_id` (404 if unknown, newest scan otherwise) and gives a found face the track whose own nearest observation (≤0.5 s, IoU ≥0.4) it overlaps; real detector scores replace the constant `0.98`; frame size comes from the decoded frame (rotation-safe). |
| Detector operating point too strict for turned heads | `PRIMARY=0.5` | `det_thresh` calibrated to **0.40** (inspection, frame detection, tracker, FaceSet reader 0.35). Scores in [0.40, 0.50) must pass roll-invariant five-point geometry and the independent 2d106 landmark agreement, so the lower cut does not admit clutter; ≥0.50 stands as before. |
| Recovery ran only when no face passed | one 0.5 face suppressed it for the whole frame | The tracker passes the number of live tracks as `expected`; a missing one triggers recovery even when another face passed. New `magnify` stage (640 px windows at 1×/1.5×/2×) joins zoom-out, coarse tiles and ±30/60/90° rotation. Box NMS now also drops nested/partial duplicates (ear-sized box ≥60 % inside a ≥4× larger head). |
| Frontal-leaning landmark vetoes | narrow profiles needed a dense model; sub-threshold profiles were vetoed by 2d106, which is itself unreliable on steep profiles; box aspect ≥0.30 | Aspect floor 0.25; a primary-score profile needs no dense model; a tolerant-band face with valid geometry survives when it scores ≥0.35 even if 2d106 disagrees. No yaw/pitch rejection exists in the scan path. |
| `harjot` not loadable by id | `app/facesets/harjot.fsz` has no `.runtime/facesets/harjot.json`; `/faceset/reconstruct-3d` and the identity-asset routes demanded a catalog manifest | `faceset/locator.py` resolves catalog id → library archive by stem → unique catalog name; used by the reconstruct and identity-asset routes. The job-lock resolver (`identity/lock.py`) was not changed. |
| Reference quality was a whole-image Laplacian × 0.7 + constant 0.95 detector confidence | `harjot` views 0 and 4 scored 0.5015 / 0.5041, one rounding from the `≥0.5` aggregation cut | Reader uses the multi-scale detector, keeps the dominant face, records the **real** score and face-region sharpness; cached v1 metadata is versioned (`CACHE_VERSION=2`) so old numbers are never reused. |
| A configured detector that is not installed | `.runtime/config.json` selected `retinaface`; no `retinaface_*.onnx` exists, so every scan answered 409 `MODEL_INCOMPATIBLE` | Local (git-ignored) config set to `scrfd`, which resolves to the installed `det_10g.onnx`. The UI still lists all three detectors regardless of availability. |

### `harjot` reference bank (`faceset/reference_bank.py`)

`load_faceset` now builds a `ReferenceBank` for every set. Each view gets a measured quality (detector confidence, face sharpness, face size, landmark trust with PnP reprojection error, exposure of the face region). Views with fabricated landmarks, or an embedding below the 0.30 identity-link cosine, are excluded with a reason. The identity centroid is the unit-length quality-weighted mean of the rest. `bank.apply()` writes centroid, medoid and per-view quality into the `FaceSetDocument` that the identity-asset builder reads (`server/identity_assets.load_document`). `bank.uv_views()` orders views by pose coverage for UV reconstruction. `bank.similarity(embedding, yaw, pitch)` scores a target against the centroid **and** the references nearest its pose; it is used by `scan_missing_angles` and exposed at `GET /api/v1/facesets/{id}/reference-bank` and `POST …/reference-bank/match`.

| `harjot` view | yaw | pitch | roll | quality | cosine to centroid | landmarks |
|---|---:|---:|---:|---:|---:|---|
| 2 | −0.7 | +5.5 | −2.3 | 0.800 | 0.886 | 1k3d68 + 2d106 |
| 1 | +30.8 | +0.8 | −2.6 | 0.796 | 0.914 | 1k3d68 + 2d106 |
| 3 | −35.5 | −0.6 | −2.9 | 0.791 | 0.908 | 1k3d68 + 2d106 |
| 0 | +85.8 | −10.3 | +5.1 | 0.664 | 0.833 | archive 5-point |
| 4 | −85.4 | −9.7 | −2.1 | 0.646 | 0.809 | archive 5-point |

5 of 5 usable, none rejected, `|centroid| = 1.0`, UV view order `[2, 0, 4, 3, 1]`. Pitch span is still only 15.85° (no up/down views): the bank reports coverage, it cannot create vertical views.

### Verification gate (ran)

```bat
"roop_3d/.venv/Scripts/python.exe" -B -m roop_3d.tools.verify_stage2
```

Production `MediaService.inspect` (adaptive, identity-lock tracker, ArcFace, 2d106) on every `.mp4` in `G:\pinokio\roop-keep\single`, plus the production FaceSet loader. Receipt: `app/roop_3d/.runtime/stage2_verification.json`. Yaw bins use the existing 5-point geometric estimate: frontal \|yaw\| < 25°, 3/4 25–55°, profile ≥ 55°.

| Video | Frames analysed / decoded | Detections | Tracks | Frontal | 3/4 | Profile | Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| s.mp4 | 49 / 59 | 49 | 1 | 43 | 6 | 0 | 9.6 s |
| s1.mp4 | 99 / 206 | 95 | 1 | 95 | 0 | 0 | 29.9 s |
| s2.mp4 | 189 / 306 | 189 | 1 | 184 | 5 | 0 | 44.7 s |
| s3.mp4 | 220 / 246 | 220 | 1 | 3 | 8 | 209 | 47.2 s |
| s4.mp4 | 286 / 374 | 296 | 3 | 9 | 124 | 163 | 104.8 s |
| s5.mp4 | 576 / 1135 | 556 | 2 | 330 | 56 | 170 | 158.5 s |
| s6.mp4 | 738 / 1335 | 480 | 2 | 295 | 41 | 144 | 190.6 s |
| s7.mp4 | 348 / 600 | 320 | 1 | 84 | 55 | 181 | 83.3 s |
| s8.mp4 | 144 / 569 | 144 | 1 | 139 | 5 | 0 | 46.0 s |
| s9.mp4 | 188 / 381 | 176 | 2 | 93 | 20 | 63 | 42.8 s |
| s10.mp4 | 212 / 381 | 199 | 2 | 119 | 22 | 58 | 53.8 s |
| s11.mp4 | 113 / 181 | 113 | 1 | 113 | 0 | 0 | 21.9 s |

**12 videos, 2,837 real detections, 0 failed (none truncated, none without a detection); `harjot` bank OK.** Tracks start at t = 0 and extend into the clip (e.g. s6 0–53.2 s).

### Observed checks and limits

- **ran** `test_stage2_target_scan.py` (15 new tests: sampler coverage / turn densification / budget stretch, observation thinning, id binding, magnified-window recovery in native pixels, expected-count recovery, band vetting, nested-box dedupe, reference-bank centroid / rejection / pose-matched scoring, library locator) and `test_robust_detection.py`: all pass. Three `test_robust_detection` contracts that pinned the 0.5 operating point, the old pass count and the 0.30 aspect floor were updated; two legacy bounded-scan tests were pinned to `sampling="every_frame"`.
- **ran** a real `create_app` + `TestClient` smoke (lifespan skipped because of `JOB_STORE_INVALID`): `reference-bank` 200, `match` 200, `/faceset/reconstruct-3d` by id `harjot` 200, `inspect-faces` 200 with observations, `detect-frame-faces` bound to the scan's `track_0000` with a real confidence, unknown `inspection_id` 404.
- **ran** `npx tsc --noEmit`, `npx eslint .`, `npx vitest run` in `app/roop_3d_ui`: clean, 89 files / 870 tests; the store test now asserts real observations and the no-observation fallback.
- **Pre-existing, not caused here** (the two `test_identity_asset_build` failures are identical at `HEAD` in a clean worktree): many `create_app`-fixture tests fail with `NameError: free variable 'app'` / `MODEL_REGISTRY_INVALID`; `test_inspection_gpu`, `test_stage8_studio` and `test_workstation_preview` do not import.
- **Pose caveat:** the profile counts for s3, s4, s7 are inflated. Those clips contain heads rolled ≥ 90° (lying down); detection and boxes are correct on visual inspection, but the five-point yaw/pitch estimator saturates at ±90° there and does not separate roll. Detection recall is unaffected; the pose bins are a heuristic, not 3D head pose.
- **Not demonstrated:** none of the 12 clips depicts `harjot` (best-match cosines −0.08 … +0.16), so bank matching was exercised for plumbing and range, not for a positive identification. s4 still yields three tracks (a small persistent box near the ear was reduced, not eliminated). Nothing here renders a frame: `is_wired()` is still false.

## Stage 3 — Persistent cross-frame face tracking and single-ID locking

**Scope:** target tracking, identity consolidation and the binding of the per-frame loop to the locked canonical ID. The renderer is still unwired (`RENDERER_NOT_WIRED`) and the persisted job store is still `JOB_STORE_INVALID`; neither was touched. Design notes: `roop_3d/docs/PERSISTENT_TRACKING_STAGE3.md`.

### Root cause of `Face 2, 3, 4, 5…`

Measured on the 12 clips with the Stage 2 tracker (cached per-frame detections, `tools/stage3_scan.py`): the main subject split whenever (a) the person turned away for longer than the 15-frame bridge (s5: face 0 / face 1, 139-frame gap; s6: 154 frames), (b) the ArcFace embedding changed mode without the head moving (s10, a rolled/rendered head: cosine 0.03 between two tracks whose boxes overlap on consecutive frames), or (c) the detector found parts of the head (s4: an ear/jaw box scoring 0.51-0.61 inside the 2000 px face became `Face 3`, plus stray hits). Online association alone cannot repair (a) or (b) without guessing, so the fix is layered.

### What changed

| Layer | Change |
|---|---|
| Online tracker (`engine/detection/face_tracker.py`, new `motion.py`, `track_policy.py`) | ByteTrack high/low tiers kept; association now Kalman-predicted boxes (`BoxMotion`, frame-number time so sparse adaptive scans predict across gaps) + IoU/GIoU + pose-aware ArcFace cosine, solved with `scipy.optimize.linear_sum_assignment` (the hand-written Hungarian is gone). Per track: EMA gallery, anchors, one prototype per 30° yaw bin, a `PoseTrajectory` and a metric 3D-centre Kalman filter (Mahalanobis gate on IoU-only links). A head that is turning (steep or sharply changing yaw) lowers the continuity cosine floor by `turn_relief` while the predicted box is continuous, and the profile is learned as its own bin. Missed tracks coast on a decaying velocity or optical flow for `bridge_frames` (90) instead of the old 15. Frame-number clock: `FaceTracker.frame_number`. |
| Global consolidation (new `identity/consolidation.py`) | Tracklets are merged once the video is tracked: artifacts (box inside a much larger co-occurring face) and noise (< 5 detections) get no ID; tracklets seen on the same frame or with disagreeing frontal prototypes are cannot-link; links by pose-matched prototype cosine, by continuity (gap ≤ 60 frames, plausible shift/scale, nobody else nearby), by continuity in a steep turn, or by a seamless same-shot hand-over (≤ 3 frames, IoU ≥ 0.6). Union-find, strongest first, never joining co-occurring clusters. Canonical IDs 1, 2, … ordered by trusted-observation count (the main subject is `Face 1`). `MediaService.inspect` runs it after the scan; inspection tracks, the identity report (segments/thumbnails/counters merged) and the lock-visible `persistent_id` are canonical; `FaceTrack.source_track_ids` records which tracker IDs were folded in. |
| Calibration | `tools.verify_stage3 --calibrate`: pose-matched cosine between different people (different videos, visibly identical subjects excluded) p50 0.03, p95 0.15-0.22, p99 ≤ 0.25 in every yaw bin; same person 0.96 frontal and median 0.45 vs the frontal prototype in profile. Thresholds: `merge_cosine` 0.45, `merge_weak_cosine` 0.20, `merge_turn_cosine` 0.05, `veto_cosine` 0.15 (in `TrackPolicy`, so the policy fingerprint changes and old locks/inspections must be redone). The different-video set contains at least one pair of the same person (cosine 0.91-0.95), so the impostor tail is an estimate, not a labelled benchmark. |
| Locked binding (new `identity/locked_target.py`, `session.py`, `pipeline_runner.py`, `video_job_runner.py`) | `LockedTargetBinder` re-labels every frame's faces with the locked canonical IDs: (1) a pre-pass table (`ReferencePipeline.scan_locked_targets`, and the existing angle scan) built by consolidating the whole timeline and matching each identity to a locked signature, looked up by frame + box; (2) sticky raw-ID binding; (3) Hungarian embedding fallback with continuity/turn relief and an ambiguity margin. A face no ID claims is never selectable. `tracking_options()` builds the binder from the inspection's identity report; the video job runner runs the pre-pass before rendering; with a binder the clustering registry no longer renumbers tracks. The report signature now holds the consolidated pose-aware prototypes (up to 8 anchors), and `IdentityGuard` lowers its per-frame floor for steep heads (rivals still veto). |

### Verification gate (ran)

```bat
"roop_3d/.venv/Scripts/python.exe" -B -m roop_3d.tools.verify_stage3 --live
```

Full production stack on every decoded frame of every `.mp4` in `G:\pinokio\roop-keep\single` (`FaceTracker.process`: detector → recovery → ArcFace → association), then global consolidation, then locked binding re-run with a fresh tracker. Ground-truth proxy for the target on a single-subject clip: on every frame the largest detection scoring ≥ 0.5. Receipt: `roop_3d/.runtime/stage3_verification.json`.

| Video | Total frames | Target detected (≥ 0.5) | Tracked target frames (Face 1) | Unique target IDs | ID switches | Online raw IDs on target / switches (before consolidation) | Locked binding: bound to Face 1 / elsewhere / switches |
|---|---:|---:|---:|---:|---:|---|---|
| s.mp4 | 59 | 59 | 59 | 1 | 0 | 1 / 0 | 59 / 0 / 0 |
| s1.mp4 | 206 | 191 | 191 | 1 | 0 | 1 / 0 | 191 / 0 / 0 |
| s2.mp4 | 306 | 284 | 284 | 1 | 0 | 1 / 0 | 284 / 0 / 0 |
| s3.mp4 | 246 | 246 | 246 | 1 | 0 | 1 / 0 | 246 / 0 / 0 |
| s4.mp4 | 374 | 374 | 374 | 1 | 0 | 1 / 0 | 374 / 0 / 0 |
| s5.mp4 | 1135 | 946 | 946 | 1 | 0 | **2 / 1** | 946 / 0 / 0 |
| s6.mp4 | 1335 | 1156 | 1156 | 1 | 0 | **2 / 1** | 1156 / 0 / 0 |
| s7.mp4 | 600 | 432 | 432 | 1 | 0 | 1 / 0 | 432 / 0 / 0 |
| s8.mp4 | 569 | 569 | 569 | 1 | 0 | 1 / 0 | 569 / 0 / 0 |
| s9.mp4 | 381 | 321 | 321 | 1 | 0 | 1 / 0 | 321 / 0 / 0 |
| s10.mp4 | 381 | 279 | 279 | 1 | 0 | **2 / 1** | 279 / 0 / 0 |
| s11.mp4 | 181 | 181 | 181 | 1 | 0 | 1 / 0 | 181 / 0 / 0 |

**12/12 pass: one identity per video (`Face 1`), 0 ID switches, every detected target frame carries it.** The online tracker alone still produced two IDs on s5, s6 and s10 (one switch each); consolidation merged them (s5 and s6 by appearance, cosine 0.93 / 0.99 across a 139 / 154-frame turn-away; s10 by the seamless rule across a 1-frame gap). s4's two extra raw tracks (2 stray hits; an ear-box inside the head) received no ID. Frames without a ≥ 0.5 detection (e.g. s5 frames 279-455: the subject faces away, the detector finds only a 0.10 box on hair) are not tracked frames and are not hidden in the totals.

Session-level check (`ReferencePipeline` with a binder from a real adaptive inspection of s5): inspection `track_0001` (one tracker ID), pre-pass `{tracker_ids: 2, identities: 1, matched_to_lock: 1}`, replay through `track_frame`: 997 frames carry `Face 1` (low-score continuation hits included), 0 faces under any other ID, only ID 1 selected.

### Observed checks and limits

- **ran** `tests/test_persistent_tracking.py` (25 new: GIoU, Kalman gap/coast/decay, pose-trajectory turn detection, Hungarian admissibility, head turning to profile keeps one swappable ID, an embedding collapse without a turn is still withheld, sparse frame-number sampling, two people while one turns, profile split / return / stranger / people-together / third person nearby / artifacts / determinism / seamless / shot change, binder relabel / stranger / turn / pre-pass global decision, inspection record + report folding, guard steep floor) and `test_identity_tracker_policy.py` (20, unchanged): pass. `test_persistent_target_registry` was updated for the new session attribute. `npx tsc --noEmit`: clean (`Track.source_track_ids`).
- Fallback-only binding (a frame the pre-pass missed, strict embedding/continuity only) is reported but not gated: on s6 it leaves 13 frames unbound where a new tracker ID is born at a 90° yaw with cosine 0.20-0.33 to the signature. That is deliberate: impostor cosine in profile (p99 ≈ 0.25) cannot separate people, and the pre-pass covers those frames in the runners.
- The target proxy and the "ID switch" count are properties of single-subject clips; they say nothing about multi-person crossings, which are covered only by the analytic scenarios in the tests. The consolidation has not been run on a multi-person video.
- Replay uses the production detections; `--live` runs them fresh. The adaptive production scan (`sampling="adaptive"`) was exercised end to end on s11 and s5 (one canonical `track_0001`), not on all 12 clips.
- **Pre-existing, not caused here:** `NameError: free variable 'app'` in every `create_app`-fixture test (`server/app.py:90`), `test_d_zero_2d_post_crop_gan_calls` (`ultramax` ids from Stage 1), `test_e2e_video_pipeline` (removed `inswapper_128`). Nothing here renders a frame: `is_wired()` is still false.