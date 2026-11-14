# Phase 1 audit: repo and pipeline (Agent 1, read-only)

Date 2026-11-14. Repo `G:\pinokio\api\roop-3d`, branch `feat/unified-3d-workstation`, venv `app/roop_3d/.venv/Scripts/python.exe`.
Legend: **[V]** verified by reading code and/or by running a read-only probe; **[S]** speculation or inference, not verified.

Caveats up front
- The working tree is dirty and **other agents were editing while I read**. `media/face_analysis.py` changed under me
  (aspect gate 0.45 -> 0.30, roll-aware `vertical_order`); `models/catalog.py`, `models/hub.py`, `engine/volumetric/swapper_engine.py`,
  `core/ort_session.py` all carry large uncommitted diffs. Line numbers below are as of reading; re-grep before editing.
- The Pinokio-launched server (port 8030, from `app/roop_3d/.runtime/config.json`) was **not running** when I probed, so nothing here
  comes from live HTTP. All "live" facts come from instantiating `ModelCatalog` / detector on CPU in a scratch process.
- Probe scripts are in the scratchpad (`...\scratchpad\introspect*.py`, `probe_*.py`, `harjot.py`); no repo file was changed.

---

## 1. Real route names and execution path

Routers are included in `server/app.py` (~lines 200-300). Everything is under `/api/v1` except where noted.

| Prompt name | Real route | Handler |
|---|---|---|
| `/api/v1/health` | `GET /api/v1/health` | `server/app.py:250 health()` (static dict, no model touch; also `/api/v1/meta` 266, `/hardware` 256) |
| `/api/v1/facesets/` | `GET /api/v1/facesets` (no trailing slash; the slash form 307-redirects) | `server/routes.py:577 get_facesets` -> `routes.py:26 facesets()`: lists `app/roop_3d/.runtime/facesets/<id>.json` only (id, name, item_count) |
| target inspection | `POST /media/inspect-faces` | `routes.py:601 inspect` -> `media/service.py MediaService.inspect` (~line 190): PyAV decode -> `FaceTracker.process` per frame -> `FaceInspection` saved to `.runtime/inspections/<id>.json` |
| target frame faces | `POST /media/detect-frame-faces` (`routes.py:613`), `GET /media/faces` (`server/target_lock.py:375`), `GET /media/frame|filmstrip|info` (`server/media_frames.py`) | `MediaService.frames` (`media/frame_service.py VideoFrameService.read/analyzed_faces`) -> `media/face_analysis.py FaceAnalyzer.analyze` |
| target scan / lock | `POST /target/lock-from-frame` (`server/target_lock.py:207`), `/target/add-angle-sample`, `/target/{id}`, `/target/{id}/rescan`, `/target/{id}/match` | `scan_keyframes` -> `_scan_keyframes` (target_lock.py:160) -> `identity/target_lock.py FaceTrackRegistry.associate`; thread `target-scan-*` |
| identity lock (hash-bound, other system) | `/api/v1/identity/inspections|locks` | `server/identity_lock.py`, `identity/lock.py` |
| 3D identity asset | `GET/POST /facesets/{id}/identity-asset` (`server/identity_assets.py:172/195`), `/identity-asset/mesh|mesh.glb`, `/bake-atlas`, `/faceset/reconstruct-3d` (`server/faceset_3d.py:126`) | `load_document` (identity_assets.py:78) -> `FaceSetReader.read`; `_build` (178) -> `IdentityAssetStore.get_or_build` -> `reconstruction/identity_asset.py build_identity_asset` (312) |
| frame preview | `GET /media/preview-frame` (`routes.py:621`, raw video frame JPEG, `frame_index<=9999`), `POST /preview` (`server/live_preview.py:778`, swapped frame), `GET /target/preview|preview_grid|preview_seq` (live_preview.py:785/802/849), `POST /workstation/preview` (`server/workstation_preview.py:489`) | `LivePreviewService` (live_preview.py:361), `PreviewService.render` (workstation_preview.py:171) |
| offline queued video render | `POST /jobs` (needs `Idempotency-Key` header; `routes.py:684`) + `validate_job` (649); also `/queue*` (`server/queue_routes.py`) and `/projects/{id}/jobs` (`workstation.py:626`, what the UI App.tsx:509 uses) | `JobWorker` (`jobs/worker.py:40`, `_run_job` 268) -> `EngineRegistry.resolve(engine_id)` (`engine/registry.py:48`) -> `runner.prepare` (`engine/pipeline_runner.py:502`) -> `run_prepared`. Engines: `reference_onnx` = `MasterPipelineRunner`; `photoreal_3d_ffmpeg` = `pipeline/video_job_runner.py EndToEndVideoJobRunner.run_prepared` (417), registered at `server/app.py:106` |
| model / engine status | `GET /models` (routes.py:200), `/models/active` (208), `/pipeline/status` (248), `/system/capabilities` (59), `/orchestrator/engines`, `GET /models/hub`, `/models/registry` | `ModelCatalog.public()`, `swapper_readiness()`, `ModelHubService.engines()` (`models/hub.py:302`) |

Per-frame render chain for jobs: `FFmpegDecoder` -> `SessionFrameProcessor.process` (`pipeline/session_adapter.py`) -> `ReferencePipeline` session
(SCRFD/RetinaFace `media/detector.py` + `engine/detection/face_tracker.py` + 106pt + ArcFace + inswapper via `SwapperEngineAdapter`
`engine/volumetric/swapper_engine.py`) -> `Photorealistic3DSwapCore` -> `stream_video` (NVENC/x264) -> remux audio. A job where no mapped face is swapped in any frame
fails `TRACK_NOT_DETECTED` (video_job_runner.py ~line 500).

UI entry points (`app/roop_3d_ui/src`): `api.ts:13` prefixes `/api/v1`; health in `components/workstation/WorkstationScreens.tsx:150`;
inspect = `App.tsx:487` (`identity_lock:true, full_video:true, time_budget_seconds:900`); facesets `App.tsx:279`; identity asset
`components/studio/FaceSetDetail.tsx:92/111`; target lock `components/canvas/useTargetLock.ts` / `api.ts:108`; job submit `App.tsx:509`, `QueueDrawer.jsx:96`.

**Upload gotchas** [V]: `MediaService.path` (media/service.py:~68) only accepts videos under `app/roop_3d/.runtime/media` (or outputs), so
`G:\pinokio\roop-keep\single\*.mp4` must be uploaded first. `POST /media/upload` is bounded by `max_body_bytes` = 2 MiB in the saved config
(`.runtime/config.json`), so s3-s8 (5-44 MB) need the chunked `/assets/uploads` protocol (`server/workstation.py:45-150`), as the UI does.
`GET /media/preview-frame` rejects `frame_index > 9999` and `MediaService.frame` rejects `>= 10000` (service.py:137).

---

## 2. The `harjot` FaceSet

### 2.1 Where it really lives [V]
- `app/facesets/harjot.fsz` (1,210,895 bytes, sha256 `8d5e2775...abd09c`) is the user library file (listed by `server/faceset_builder.py:113 GET .../list`).
- The server only serves FaceSets that have a manifest in `app/roop_3d/.runtime/facesets/<hex id>.json` AND an archive at
  `app/roop_3d/outputs/facesets/<id>.fsz` or `.runtime/facesets/<id>.fsz` (`server/identity_assets.py:65 _archive`).
  Harjot is registered as **id `797577eca6ab4cf88c7b3e075b590979`**, name "harjot", item_count 5; its archive
  `outputs/facesets/797577eca6ab4cf88c7b3e075b590979.fsz` is byte-identical (same sha256) to the library file.
  `GET /api/v1/facesets/harjot/identity-asset` would 404 (`FACESET_NOT_FOUND`); use the hex id.
- `outputs/facesets/harjot.fsz` (named copy, same digest) has no manifest json, so it is invisible to `/facesets`.
- Only two ids are catalogued (`harjot`, and `2faceset` = `650ad3b8...`, 10 views); about 40 other `.fsz` files in `outputs/facesets/` are uncatalogued.

### 2.2 Format and validation [V]
- Archive: ZIP, 5 entries `0.png`..`4.png` (about 220-255 KB each), **no `metadata.json`** -> classified `legacy_v1` by
  `faceset/validator.py ArchiveValidator.validate_zip` (path traversal, symlink, disallowed extension list, 512 entries, 32 MiB/entry,
  256 MiB total, compression ratio <= 20, magic bytes) -> `FaceSetReader._parse_v1_archive` (`faceset/faceset_reader.py:261`).
- v1 parse has no stored embeddings or landmarks: every image is re-detected on CPU (`_extract_face_view`, faceset_reader.py:370):
  SCRFD `detections[0]` (the FIRST detection, not the largest; line ~381), ArcFace `Recognizer.embed`, `legacy_pose_5pt` for yaw/pitch/roll
  (legacy frame, pitch + = down), quality score `0.7*sharpness + 0.3*det_conf`.
  **`det_conf` is hard-coded 0.95** (line ~382) so `quality.detector_confidence` is not a measurement. If detection finds nothing it
  fabricates landmarks at fixed fractions of the image and sets `landmarks_trusted=False` (line ~398).
- Result cached at `.runtime/cache/facesets/<sha256>/metadata.json` (cache exists, so re-reads skip detection).
- Measured for harjot: 5 views, all `landmarks_trusted=True`, yaw +78.9 / +32.7 / -1.2 / -38.2 / -78.8, pitch -8.7..+2.4, 5-point
  landmarks only (`landmarks_68` None), canonical embedding unit norm. `analyze_faceset` (`reconstruction/faceset_quality.py:198`):
  readiness `full_coverage`, 5/5 usable, identity distances 0.09-0.17, missing `looking_down`, `looking_up`.

### 2.3 Conversion to the 3D identity asset [V]
`POST /facesets/{id}/identity-asset` -> `server/identity_assets.py:_build` (lock per faceset, 409 if already building) ->
`IdentityAssetStore.get_or_build` (`reconstruction/identity_asset.py:760`): `compute_build_key` (227; hashes view images, embeddings, landmarks, poses, settings, template
hash, `flame_status()`) -> `store.load` hit or `build_identity_asset` (312): `analyze_faceset` gate (`insufficient` / `blocked_inconsistent_identity` -> `IDENTITY_ASSET_NOT_BUILDABLE` 422),
`select_bucket_views` (max 12), `fit_identity_landmarks` (multi-view weak-perspective MAP; iteratively drops the worst inconsistent view),
`deform_template` of the canonical 68pt mesh, `UVSynthesizer` texture atlas (512 px), per-view 256 px UV layers, vertex/texel confidence states,
manifest with explicit `claims` / `limitations`. Persisted to `.runtime/identity_assets/<id>/<build_key[:24]>/{arrays.npz,texture.png,head.glb,manifest.json}`.
- **The server path does NOT densify landmarks**: `load_document` only calls `FaceSetReader.read`, so v1 FaceSets reach the fit with 5-point landmarks
  (`fit.layout == "5pt"`, manifest limitation "Only 5 landmarks per view..."). `faceset_loader.enrich_document` (adds 1k3d68/2d106det) is only used by
  `server/faceset_3d.py:104` (`/faceset/reconstruct-3d` and `/faceset/3d-state`). [V by grep]
- **Cached asset for harjot is stale** [V]: on disk `identity_assets/797577.../970fb220d18acdae37fd6140` (2026-11-09); the build key now computed for the
  same document + default settings starts `0c99016c...`, so `GET .../identity-asset` returns `not_built` and a POST rebuilds (CPU, canonical template).
  (`IdentityAssetStore.load` also rejects on `ALGORITHM_VERSION` = `identity-mv-5`.) I did not run the build.
- MakeHuman variant: `?template=makehuman` requires the template built first (`reconstruction/makehuman_build`); fit uses 5pt unless every chosen view has 68pt.

---

## 3. Root-cause map, problem #4: swap engines "unavailable / blocked"

### 3.1 What actually resolves today [V, CPU introspection, read-only]
`ModelCatalog().rescan_local()` + `swapper_readiness()` + `ModelHubService.engines()` + `CapabilityRegistry.engine_reasons` on this machine (RTX 4070 12 GB, ORT 1.23.2 with TRT/CUDA/CPU):

- **Selectable / READY now (5):** `inswapper_128` (resolves to `roop-ultimate/.cache/models/inswapper_128.onnx`, policy `REGISTRY_GOVERNANCE`),
  `inswapper_128_fp16`, `hyperswap_1a_256`, `_1b_`, `_1c_` (resolve to `app/models/*`, policy `LOCAL_USE_TRUST`).
- **NOT_INSTALLED (9):** `realswap_256`, `realswap_512`, `simswap_256`, `simswap_512_unofficial`, `ghost_256_unet_1/2/3`, `blendswap_256`, `uniface_256`. Their registry rows are
  `status=candidate` with no SHA-256, size, locator or URL (supported_models.json has specs, registry has only placeholders) and no file exists anywhere under
  `G:\pinokio\api` (find by name returned nothing). **This is the permanent "unavailable" the UI shows for those ids; no code fix can make them run without weights.**
- Hub engines: `reference_onnx` READY_FOR_INFERENCE (detector scrfd_10g_bnkps, w600k_r50, inswapper_128 all hash verified); `photoreal_3d_ffmpeg` READY (its `selectable` flag is
  True only after `server/app.py:106` registers the runner; a bare `ModelHubService` shows False). `minimax_h3...` / `bfs_qwen...` are `UNAVAILABLE_ENGINES` by design (`engine/registry.py:10`).
- Orchestrator registry: all 21 engines OK on tier `12gb`.
- Registry integrity: 115 `verified_*` records; `LocalWeightProvider._resolve_pinned` -> 113 OK, 2 `MODEL_NOT_INSTALLED` (verified_remote, not downloaded), **0 SHA mismatches**. Pinned digests equal the
  observed digests of the `roop-ultimate/.cache/models`, `roop-ultimate/app/models` and `app/models` hyperswap_1a / inswapper files.
- Blocked: exactly one record, `flame-2020` (role `flame_geometry`, `LicenseRef-FLAME-Academic`, manual acquisition). Not a swapper.
- Non-swapper lifecycle (`catalog.lifecycle`): scrfd, retinaface, yoloface, gfpgan_v14, gpen_bfr_512, codeformer, ultramax, xseg all READY.
- `GET /system/capabilities` -> `routes.py:59-197` builds the per-control reasons (`swapper_reason`, `governed_reason`); with the above state none of the five real swappers returns a reason.
- Past real jobs worked: `jobs/state.json` shows `reference_onnx` with `hyperswap_1a_256` and `inswapper_128` succeeded Nov 13; Pinokio log `logs/api/start.js/latest` shows
  `hyperswap_1a_256.onnx -> Tensorrt in 12.3 s`, retinaface/2d106det/w600k/1k3d68/bisenet on TensorRT, `codeformer.fp16.onnx -> Tensorrt in 95 s`.

Conclusion [V]: at the registry / SHA / governance layer, the swappers that have bytes are not blocked. The "unavailable/blocked" a user sees comes from the causes below.

### 3.2 Ranked causes
1. **Missing weights, not a gate** [V]. 9 of 14 supported swappers have no file and no downloadable identity (`models/catalog.py swapper_missing_reason`, ~line 349; `models/hub.py _entry` ~126;
   registry records `candidate`). Fix is data (add pinned sha/size/source in `models/verified_sources.json` + registry), not code.
2. **HiFiFace / CrossFace / hififace fp16 have weights but no adapter spec** [V]. Records `hififace_unofficial_256`, `crossface_hififace`, `hififace_unofficial_256.fp16` are verified, but
   `supported_models.json` has no entry, so `swapper_spec()` is None, rows get `adapter_supported:false`, `selectable:false` (`catalog.py _adapter_support` ~491). `swappers/hififace.py HiFiFace256` exists
   (used only by `models/stage16.py` experiments) and is not wired into `SwapperEngineAdapter` or any engine. `inswapper_128.batch`, `hyperswap_*.batch` (derived exports) are also shown as
   swapper rows with `adapter_supported:false` (12 hash-id rows in `/models`) - cosmetic noise that reads as "unavailable".
3. **TensorRT -> CUDA -> CPU chain hides reasons and forbids CPU nodes** [V by log + code].
   - `core/ort_session.py SessionManager._create` (713-779): order TRT, CUDA, CPU; GPU sessions set `session.disable_cpu_ep_fallback=1` (line 601). Log evidence: `face_occluder.onnx` fails TensorRT
     (dynamic-shape profile error, 9-13x) AND CUDA ("graph nodes assigned to the default CPU EP, but fallback ... explicitly disabled"); it ends up on whatever the caller's CPU path is.
     `compositor/occlusion_analysis.py:34` lists `face_occluder.onnx` in `DEFAULT_NETS`, and `photoreal_3d_ffmpeg` requires every configured visible-face network (pipeline/__init__.py docstring), so this can fail a job as `MODEL_SESSION_FAILED`.
   - `engine/runtime.py:219-220` and `:249`, `:278`: `except Exception: raise DomainError("MODEL_SESSION_FAILED", "Verified model could not initialize on TensorRT or CUDA", 503) from None` throws away the per-provider failures
     (Agent 2's uncommitted `ProviderInitializationError` in `ort_session.py` + `swapper_engine.py open_local_session` fixes this for the swapper path only; `OrtModel` still swallows).
   - `OrtModel` divides the CUDA arena by a per-model divisor table (`engine/runtime.py:187-201`; default 32) so most models get `gpu_budget/32` -> "Conv fallback mode" slow paths (memory: roop3d-scan-cpu-inspection-gpu). [S for failure impact]
4. **Policy asymmetry makes the same bytes "VERIFIED" or "LOCAL_USE_TRUST" depending on path** [V]. `resolve_local_model` (`catalog.py:239`) prefers a scanned `app/models` file over the pinned
   `repository` locator, so `hyperswap_1a_256` (pinned digest c0e98a8a... present at 3 locations) reports `policy=LOCAL_USE_TRUST` while `inswapper_128` reports `REGISTRY_GOVERNANCE`. Not a block, but UI/manifest wording and
   the `swapper_eligibility` size gate (`MIN_TRUST_BYTES` = 1 MiB, `models/local_trust.py:27`) differ. `LocalTrust` fuzzy-matches by name (`local_trust.py locate`), e.g. `inswapper_128_fp16.onnx` is byte-identical to `inswapper_128.onnx` (sha e4a3f08c...), i.e. not FP16.
5. **Fatal authoritative-locator drift** [V design, not triggered now]. `LocalWeightProvider._locations` (`models/local_provider.py:38-60`) yields only the first existing locator path and returns; if that file's bytes ever differ the record fails
   (`MODEL_NOT_INSTALLED`/mismatch) even if another good copy exists; only an `app/models` file can rescue it through `trusted_path` (local_provider.py:73). Roop-ultimate files are read-only evidence (`audit/model_sources.json read_only:true`; `legacy_root()` = `PINOKIO_HOME/api/roop-ultimate`).
   `Downloader.cached` (`models/downloader.py:131`) **quarantines (os.replace) a managed-cache file that fails verification** - destructive side effect on `app/roop_3d/models/cache/`.
6. **Search surface is narrow** [V]. Weights are found only via: registry locators (scope `repository`=roop-ultimate, `user_cache`=`~/.cache`, `insightface_cache`, `torch_cache`), managed cache `app/roop_3d/models/cache`, configured `model_search_paths` (**empty** in `.runtime/config.json`), and `app/models`
   (rescan + auto-trust). Landmark models use a separate list `faceset_loader._model_dirs` (env `ROOP3D_LANDMARK_MODEL_DIR`, `.runtime/models`, `roop-ultimate/app/models/buffalo_l`). Detector `retinaface` resolves to `roop-ultimate/app/models/retinaface_10g.onnx` (log), not app/models.
7. **Stale server / uncommitted code** [S]. The running server (started 2026-11-14 01:56) predates the current uncommitted diffs in `catalog.py`, `hub.py`, `local_provider.py`, `routes.py`; a restart is needed to see readiness changes. `/models/active` and `/pipeline/status` (`routes.py:208/248`) are only meaningful after a render/preview has loaded sessions (`engine/active_models.py`).
8. `/system/capabilities` `governed_reason` returns None on the first READY candidate; but candidates with no SHA yield `NOT_INSTALLED` issues "Dependency compatibility is unknown / Hardware requirements are unknown / Artifact security review is incomplete" (`models/governance.py:19-80`) - if the only matching candidate is a metadata placeholder the control is disabled with these misleading reasons (scrfd, retinaface, yoloface, codeformer each have one such placeholder record, but also a READY one, so not currently triggered). [V code, not triggered]

### 3.3 ORT provider selection summary [V]
Saved config `.runtime/config.json`: `preferred_provider=TensorrtExecutionProvider`, `precision_policy=auto`, `detector_model_id=retinaface`, `swapper_model_id=inswapper_128`, `face_restoration_policy=off`, `hardware_profile=auto` (resolves `12gb`).
`probe_hardware().tensorrt_mixed_precision` = `{available:true, tensorrt_available:true, cuda_fp16_available:false, active:false}`. Inspection uses `engine/runtime.inspection_session_factory` (CUDA only on >=10 GiB cards, else CPU; `build_with_cpu_fallback` at :510).

---

## 4. Root-cause map, problem #3: target faces missed in `G:\pinokio\roop-keep\single\*.mp4`

### 4.1 The clips [V, ffprobe]
12 files: s, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11. Formats: 1920x1080 (s, s3), 4096x2160 (s4-s6), **2160x4096 portrait (s8)**, 1280x720 (s7, s9, s10), 480x360 (s1, s2),
360x640 (s11). s7/s9/s10 are 3D head **turntable renders** (s9/s10 are pillarboxed: active area about 400 px wide inside 1280x720 with black bars); s1/s2 are SD interview-style clips with an ECU framing; s4-s6/s8 are 4K/portrait-4K with huge faces (350-1480 px).

### 4.2 Measured detection results [V; SCRFD and RetinaFace on CPU, about 40 evenly spaced frames per clip; 9 clips through the full `FaceAnalyzer`]
- scrfd10g vs `retinaface_10g` (the saved detector): identical hit counts on every sampled frame, so the detector choice is not the cause.
- **49 of ~313 sampled frames (16 %) have NO raw detection even at score 0.1.** Viewing the frames: back of head / ear profile (s7, s9, s10 turntables; yaw beyond 90 deg), a hand over the face plus blur (s1), extreme close-up where the mouth/chin are outside the frame (s2), a sideways-rotated head (s10).
  Per clip no-raw-detection count in the analyzer run (about 30-36 frames sampled per clip): s1 17, s2 19 (the first 60 % of the clip), s7 6, s9 4, s10 3, s/s3/s4/s11 0.
  A separate SCRFD-only run (about 40 samples/clip, score >= 0.5) missed s1 24/42, s2 27/44, s10 11/43, s7 12/40, s9 6/43 (this includes sub-0.5 scores).
  These are physically undetectable per frame; recovery needs temporal bridging (see 4.3 #4), not thresholds.
- **7 of ~313 frames have a raw detection that the analyzer drops**: `box_aspect` (profile boxes with w/h 0.40-0.44 and scores 0.40-0.50 in s7/s9, old gate `0.45 <= w/h`), and low scores 0.25-0.29 (just under TOLERANT 0.30) or 0.10-0.13 (s10, s1, s7). Agent 3's in-flight edit already relaxes the aspect gate to 0.30.
- 4 frames were kept only as `profile_tolerant` (0.30-0.50 + valid landmarks + 106pt agreement) - the tolerant tier works.
- Pillarbox / letterbox is **not** a cause: cropping s9/s10/s7 to the active area recovered 0 extra frames (0/58 misses). 4K downscale is not a cause either: faces in s4-s8 are 330-1480 px, scaled to 640 they stay >100 px.

### 4.3 Ranked causes (file:line as read; re-grep, Agent 3 is editing these)
1. **Per-frame detector only, no coasting through undetectable frames** [V]. `FaceTracker.process/observe` (`engine/detection/face_tracker.py:~146-200`) creates observations only from detector output. Missed-box propagation by optical flow exists only in the
   identity-lock policy branch (`_propagate_missed` ~392 / `track_policy.propagate_box:159`, active when `policy is not None`, i.e. `identity_lock:true`); `FaceTracker.assign` legacy path keeps `max_missed=15` frames then drops the track. The render path (`SessionFrameProcessor`) counts
   such faces in `faces_skipped`; a job with zero swaps fails `TRACK_NOT_DETECTED`. The `tracking/` head-tracking package (memory: Stage 4) is not wired into the job runner. This is the dominant cause for s1/s2/s7/s9/s10.
2. **Two different low-score floors** [V]. Tracker observes down to `low_threshold=0.1` (face_tracker.py:108, 159) but the frame/target-lock path (`FaceAnalyzer`, `media/face_analysis.py:29-30`) uses `PRIMARY 0.50` / `TOLERANT 0.30` and only accepts the tolerant tier if `five_point_plausible` AND `agrees_with_106` pass
   (~lines 51-100, 140-150). Detections at 0.10-0.29 exist in the data (s10 `[0.29,0.25,0.12]`) and are discarded by the analyzer (and therefore by target lock and the Target panel), though the tracker might have kept them.
3. **`five_point_plausible` gates** [V]: `MIN_FACE_PX=24` native pixels (face_analysis.py:32), aspect window (was 0.45-1.7 at HEAD, now 0.30-1.7 in working tree), `landmark_outside_box` pad 0.15, `vertical_order` (was an image-y test that rejects rolled heads; now axis-based). Profile faces (yaw > 70) have thin boxes.
4. **Detector layout** [V] (`media/detector.py:93-101`): one fixed 640x640 letterbox anchored top-left with zero padding; `cv2.resize` default interpolation (INTER_LINEAR) when downscaling 4K (aliasing; INTER_AREA is better); no multi-scale / tiling; `retinaface` and `scrfd` share the same decode with `swapRB=(selected=='scrfd')` (line ~101), so a RetinaFace-named model gets BGR (measured: no recall difference, so low priority [S]). Limit `MAX_FACES=32`, `max_tracks<=32`, 64 tracks per inspection (`media/service.py` ~line 222).
5. **Pose coverage grid ends at +/-75 deg yaw, +/-45 pitch** [V] (`identity/target_lock.py:24-28 pose_bucket`): out-of-grid poses never claim coverage and never replace a bank entry; `consider()` keeps one entry per bucket; a bucket is "locked" only at `BUCKET_QUALITY_THRESHOLD=100` (Laplacian variance of the 256 px crop x score; `server/target_lock.py:42-56 ref_of`), which low-res clips (s1/s2/s11, upscaled crops) may never reach, so the scan never early-exits and walks all frames (`target_lock.py:192`). Association needs cosine >= `MIN_COSINE=0.42` (identity/target_lock.py:14): a steep-profile embedding below that starts a new cluster (an apparent "missed" target / split identity). harjot's own profiles are at +/-78.9 deg, outside the 75 deg grid.
6. **API default budgets** [V]: `InspectRequest` defaults `full_video=False`, `max_samples=32`, `time_budget_seconds=30` (`media/schema.py:16-23`) -> bounded scan, later faces missing. The UI sets `full_video:true, time_budget_seconds:900` (App.tsx:487), so only scripted callers hit this. With `full_video` the 64-track and 900 s limits still apply to 4K clips (s5 1135 frames, s6 1335 frames).
7. **Embedding-based identity gate** [V]. In render, ambiguous or uncertain faces get `swap_allowed=False` / withheld (`face_tracker._assign_registry`, `session_adapter` `faces_withheld`), indistinguishable in the UI from a miss.
8. s8 (2160x4096 portrait) and s11 (360x640): no misses measured; s4-s6 4K decode/detect cost is the only issue (time budget).

### 4.4 Not verified [S]
- No end-to-end job/preview was run (not allowed). Whether the user's "missed faces" are the turntable back-of-head frames or specific frames in s4-s8 is unknown; no misses were found in s, s3, s4, s11 at 30-40 samples. A frame-exact sweep of s5/s6/s8 was not done.
- Whether `retinaface_10g.onnx` expects RGB (it is a 16,926,877-byte file, different from `det_10g.onnx`).

---

## 5. Memory notes vs reality
- `roop3d-local-weight-trust.md` says "app/models holds NO weights". **Stale**: it now holds 19 `.onnx` (inswapper_128_fp16, hyperswap_1a/b/c, scrfd_2.5g, det_10g, w600k_r50, arcface_w600k_r50, 1k3d68, 2d106det, xseg, occluder, bisenet, GFPGAN, GPEN, ultramax, yunet, sface) + `trt_cache`.
- `roop3d-engine-model-rack` "none of the 14 curated swappers has a download_url / all installed:false" - stale for inswapper/hyperswap.
- `roop3d-scan-cpu-inspection-gpu` open item "500 on GET /target/<id>": not reproduced here; `read_target` -> `lock.public()` (identity/target_lock.py) serialises `BankEntry.public()` values that can be `numpy.float32` (`round(np.float32, n)` stays numpy) - possible JSON encode failure [S].
- Dev env notes (run pytest from `app/`, no concurrent pytest, Node via `G:\pinokio\bin\miniforge`) still apply; python is not on the bash PATH (use the venv path). Windows python needs `G:/...` paths, not `/g/...`.

## 6. Suggested hand-offs
- Agent 2 (engines/registry/ORT): add pinned identity (sha/size/source) for the 9 missing swappers or hide them with a precise reason; wire or de-list hififace/crossface; make `OrtModel` surface per-provider `ProviderInitializationError.attempts`; handle `face_occluder.onnx` (static-shape export or CPU-allowed session); decide policy label for hyperswap (pinned vs local-use).
- Agent 3 (detection): lower/unify the 0.30 tolerant floor with the tracker's 0.10, add temporal bridging (optical-flow coast) to the non-policy path, INTER_AREA + optional 1280 pass, extend `pose_bucket` beyond 75 deg or add a profile bucket, review `MIN_COSINE` for profiles.
- Agent 4 / QA: harjot id is `797577eca6ab4cf88c7b3e075b590979`; rebuild its identity asset (stale key) and check manifest `landmark_layout` (expect `5pt`).
