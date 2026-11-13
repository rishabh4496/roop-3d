# R3 slice 3 â€” roop-3d: UI, tests, audit scripts, docs, root files

Slice: `app/roop_3d_ui/` (359 tracked files; node_modules/dist/test-results are not tracked), `app/roop_3d/tests/` (202), `app/roop_3d/audit/` (49), repo-root `docs/` (98), root-level and other tracked files (46). Total **754** tracked files. All paths relative to `G:/pinokio/api/roop-3d`. Evidence is `file:line` read from the working tree on 2026-11-13; counts of tests are static (`def test_` / `it(`), nothing was executed. Anything not read from code/docs is tagged [INFERENCE].

## 1. Headline findings (slice)

1. **`verify_isolation.py [--report]` verifies nothing.** It delegates to `stage00.main()`, which hard-codes `historical = {"ok": True, "checked": 2645 ...}` and `comprehensive = {... "checked": 2645}` (`app/roop_3d/audit/stage00.py:35-36`); `isolation.verify()` returns `ok: True, violations: []` unconditionally (`isolation.py:184-197`). `--legacy-only` and `--snapshot` are parsed and ignored (`stage00.py:26-27`); `verify_policy`/`immutable_manifest` are imported (`stage00.py:16`) but never called. Exit is always 0 unless an OS error occurs. `--report` only writes `roop_3d/audit/stage00_verification.json` (`stage00.py:47-48`).
2. **`local_models.py --verify` is real** (hash + size per registered model, FLAME-2023-Open pin, CC0 pack rebuild/digest, universal-manifest corruption check) â€” see Â§4.
3. **The UI has no temporal-stability graph, no per-frame jitter plot and no yaw-failure/limit read-out.** Temporal smoothing is one slider mapped to `temporal_min_cutoff` (`WorkstationControls3D.tsx:33-46,186-201`); "speech-lip lock" and "scene adaptation" toggles are localStorage-only preferences with no engine consumer (`WorkstationControls3D.tsx:18-19,240`), although the backend has a 16-frame/3 s scene-adaptation pass (`tests/test_head_tracking_stream.py` header). Scene cuts are shown only as markers on `TrackLanes` (`TrackLanes.tsx:78-80`).
4. **Extreme-yaw real-media tests are skipped by design:** `test_extreme_pose_3d_swap.py::test_real_target_frames` needs `tests/fixtures/extreme_pose_real/*.fsz`, which are absent from the tracked tree (no non-`.py` file under `app/roop_3d/tests/`; `docs/EXTREME_POSE_3D.md:108-113` says 4 tests skip; "no real-profile pass" claimed). Synthetic yaw Â±85Â° is covered (`test_extreme_pose_3d_swap.py:73,94,119,268`). No tracked `conftest.py` in `tests/`.
5. **Measured throughput is far below real-time and still labelled releasable:** `audit/STAGE15_BENCHMARKS.json` (RTX 4070, CUDA EP, 512Ã—512, 15 frames) â†’ `completed_output_fps = 0.26`, mean detect_track 391.9 ms, retarget 501.6 ms, composite 522.4 ms, peak VRAM 8645 MiB; `STAGE15_VALIDATION.json` still says `"READY FOR RELEASE"`. `STAGE09_BENCHMARKS.json`: 0.57 fps, warm frame p50 1056 ms.

## 2. UI (`app/roop_3d_ui`) â€” what exists, what it consumes

Stack: React 19.3.1, three 0.187 + @react-three/fiber 9.8 / drei 10.7, framer-motion 12.43, embla 8.6, Vite 8.3.5, TypeScript 7.0.2, Vitest + Playwright 1.64 (`package.json`). Dev server 127.0.0.1:5174 proxies `/api` (incl. WS) to 127.0.0.1:8030 (`vite.config.ts:3`). Hash router with 9 screens `home, faceswap, batch, processing, facemgr, extras, gallery, history, settings` (`src/router/AppRouter.jsx:3-6`); aliases `#/studio`, `#/identity-forge`, `#/queue`; drawers `#/models`, `#/diagnostics` (`AppRouter.jsx:13-15`). Entire app state lives in `App.tsx` (832 L).

### 2.1 View-by-view answer

| Candidate view | Present? | Evidence (file:line) | Backend fields consumed |
|---|---|---|---|
| **3D mesh inspector** | YES | `components/studio/SpatialInspector.tsx:12-16` tabs Head Pose / Coverage / Morph / Light Probe / FLAME Mesh; `three/PoseGeometryViewer.jsx` (1151 L, Fibonacci-sphere lattice, 12Ã—8 yaw/pitch nodes `:115-118`); `three/FlameMeshViewer.tsx` (`poseStrength` 0=flat TPS plane â†’ 1=full depth, handles `coordinate_system` opencv/opengl/mesh); `three/viewportTools.ts:12-20` buffer views `lit, wireframe, albedo, normals, depth, cavity, sh` (SH 9-coeff, `:24-30`); `three/GuardedCanvas.tsx` context-loss rebuild (`RESTORE_TIMEOUT_MS`, `MAX_REBUILDS`); `three/WebGLFallbackInspector.tsx` SVG outline when WebGL absent; `facemgr/FacesetInspector3D.tsx`; `utils/glbMesh.ts` GLB parser (`:152` `GET /facesets/{id}/identity-asset/mesh`) | `POST /faceset/reconstruct-3d` + `GET /faceset/3d-state` (`App.tsx:310-342`, `useHeadMesh.ts`), `Reconstruction{vertices,faces,uvs,face_uvs,coordinate_system,method flame|tps_2_5d|identity_asset}`, `Coverage{views,bins,observed_pose_points,mesh_2_5d,mesh_full_3d,uv_texture_data_url,live_pose}` (`contracts.ts:136-152`), `/facesets/{id}/identity-asset`, `/bake-atlas` (PBR atlas 2048/4096, `contracts.ts:BakedPBRAtlas`), `/reconstruction/backbones` (`HeadModelBackbone.tsx:17`), `/rendering/studio-environments` (5 CC0 probes) |
| **Split wipe / A-B compare** | YES | `canvas/WorkstationCanvas.jsx:13-27` modes `result, wipe(Split Wipe), crossfade, blend, diff, loupe, mesh3d, matte`; WebGL2 shader constants `canvas/webglModel.js:8-20` (`MODE_IDS`, `LOUPE_ZOOMS=[3.5,4]`, `LOUPE_RADIUS_PX=90`, `DIFF_NOISE_FLOOR=3`, `DIFF_FULL_SCALE=128`, colormaps thermal/inferno/viridis); `output/OutputComparePlayer.tsx` wipe/side-by-side, A/V drift resync `DRIFT_SECONDS=0.12` (`:7`), loads `/jobs/{id}/download` + `/media/content`; `ui-layouts/ClipPathSplitReveal.tsx` exists but no non-test component imports it (grep) | preview frames `POST /api/preview` (header `X-Preview-Meta` JSON: faces[track_id,box,landmarks5,yaw,pitch,roll], `applied`, `unsupported`, `latency{}`; `api.ts:35-60`), `/media/preview-frame`, `/media/content` |
| **Timeline** | YES | `timeline/TimelineWorkstation.jsx` (1306 L, SMPTE timecode, filmstrip, pointer-anchored zoom), `timeline/TimelineDock.tsx` + `dockMath.ts` (Embla filmstrip, trim rail, speeds), `timeline/CurveEditorTrack.jsx` + `stores/parameterCurves.js` + `useHistoryStore.js` (reversible keyframe curves; linear/Bezier/hold), `timeline/chapters.js` (`MAX_CHAPTERS`), `utils/maskRanges.ts` (frame-range paint/erase), `studio/TrackLanes.tsx` (scene-cut lane + per-face track lanes + drag FaceSet onto lane), `utils/nleExporter.js` (FCPXML) | `/media/info`, `/media/filmstrip`, `/media/frame` with `X-Frame-Meta` (`frameScrub.ts:76-92`), `/media/probe`, `/media/keyframes` (`App.tsx:322`), `Inspection{tracks[],scene_cut_frames,tracking_method}`, `IdentitySheet{scene_cut_frames,tracks[].segments,missing_ranges}` (`contracts.ts:108-110`) |
| **Pose / telemetry HUD** | YES | `canvas/poseHud.js:23-57` draws 5-pt ArcFace landmarks + live `yaw/pitch/roll` badge for every face; `FaceOverlayCanvas.jsx` (874 L, boxes, wireframe/gaze overlays, face roles); `utils/poseCells.ts` standard-frame pose (+yaw noseâ†’image right, +pitch up); `hud/PipelinePill.jsx` (5 stages DET/LMK/WRP/RST/BLN, provider badge TRT/CUDA/DML/ROCM/APPLE `:20-60`, WS `/ws/telemetry`); `faceswap/PipelinePill.jsx`+`pillModel.js` (3 steps `STAGE_1_SPARSE_PRESCAN / STAGE_2_ACTIVE_PREPASS / STAGE_3_MAIN_SWAP`, engine banners `COMPILING_TRT`/`RUNNING_CANARY`); `shell/TelemetryHeader.tsx` (GPU, precision, VRAM, FPS, execution graph .fszâ†’targetâ†’3D LBSâ†’SH relight+TRT UltraMaxâ†’MP4); `diagnostics/BottleneckFlamegraph.jsx` (7 stages decode/detection/pose_tps/inswapper/masking/restorer/encode, `stageTimings.js:2-10`); `StreamOverlapPanel.jsx`/`lanesModel.js`; `PipelineSaturationPanel.jsx` (`/api/v1/system/pipeline`); `RuntimeReportPanel.tsx` (14-section `/system/runtime-report`); `EngineStatusBar.tsx`; `tray/AngleCaptureHud.tsx` + `utils/coverageRadar.ts` (angle coverage) | WS `/ws/telemetry` frames `pipeline.status{stage, stages{id:state}, stage_progress_pct, skipped_frames_pct, fps, eta_seconds, engine_state, engine_label, engine_model, engine_precision, engine_since, stage_timings_ms}`, `job.progress`, `system.telemetry{gpus[name,total_vram_mib,free_vram_mib], logical_cpu_count, advertised_providers}` (`telemetryStore.ts:4-17`, `pillModel.js:19-21`); `/api/pipeline/status` (loaded model per stage/precision/switch failure); `/ws/v1/pipeline/status` (`model_download_progress`, `trt_compilation_progress`); `/api/v1/ws/events` (`Event{type,sequence,hardware}`, gap detection `events.ts:20`) |
| **Live preview stream** | YES | `liveStream.ts` binary client for `/ws/v1/preview/stream`: 16-byte LE header = u32 frame, u32 ts ms, 3Ã—u16 stage latency (0.1 ms), u16 flags (bits0-3 protocol v1, bit4 jpeg, 5 dropped, 6 last, 7 cached, 8-9 source) (`:1-60`); `components/LiveStreamCanvas.tsx` (`LIVE_STALE_MS`), `utils/resilientSocket.ts` | stage1/2/3 latency, dropped flag |
| **Temporal graph / smoothing visual** | **NO** (only a slider + scene-cut markers) | `studio/WorkstationControls3D.tsx:31-46,186-201` slider â†” `temporal_min_cutoff` (light 1.5, heavy 0.03, default 0.15, debounced save 500 ms `SMOOTHING_COMMIT_MS`; saving invalidates preview caches); `TrackLanes.tsx:78-80` scene cuts; parameter curves are user keyframes, not measured jitter | `Config.temporal_min_cutoff` via `PUT /config` |
| **Batch matrix** | YES | `screens/BatchMatrixScreen.jsx` (955 L): strategies `one-to-many | grouped | matrix | recipe` (`:71`), `BUILTIN_RECIPES` (`:11`: photoreal_natural GPEN blend 0.35, high_fidelity 0.80, mesh_geometry_3d, â€¦), auto video-split via `POST /media/split-boundaries` (`:150`), per-item idempotency key `matrix-<uuid>` (`:192`), recipe sweep `workstation/ContactSheetModal.tsx` (Low/Med/High axes) | `POST /jobs`, `/config`, `/projects/{id}` ; queue UI: `queue/BackendQueuePanel.tsx` (`/queue/{id}/{action}`, `/retry`, `/reorder`, 10 lifecycle states `queued, preparing, compiling_trt, tracking_3d, rendering, paused, muxing_audio, completed, failed, cancelled`, `backendQueue.ts:3`), `QueueDrawer.jsx` (`/jobs/{id}/{pause|resume|cancel}`, `/media/concat`, `/jobs/{id}/artifacts/{video|manifest}`), `ResumeNote.jsx` (`/jobs/{id}/checkpoint` â†’ "re-run continues from last verified segment"), `HistoryScreen.jsx` (`/projects/{id}/resume|validate`, `/projects/import`) |
| **Identity / target lock** | YES | `facemgr/IdentityLockPanel.tsx` (`/identity/locks`), `canvas/useTargetLock.ts` (`/target/{id}/match?frame_index`), `api.ts:97-107` `lock-from-frame`, `add-angle-sample`, `/target/{lock}/angle/{yaw}/{pitch}/crop`, `/rescan`; `tray/FaceSelectionTray.jsx`, `FrameFaceStrip.tsx`; `useFaceSelectionStore.ts` (557 L, cosine clustering) | `lock.coverage{captured,total,quality_locked,buckets[]}` (`AngleCaptureHud.tsx:9-32`) |
| **FaceSet builder (.fsz)** | YES | `facemgr/FacesetBuilder.tsx`, `fszBuilder.js` (JS ZIP/CRC32 writer + DBSCAN ArcFace clustering, 5 pose buckets `:POSE_BUCKETS`), `BuilderCoverageSphere.tsx`, `FaceSetDetail.tsx` (3Ã—3 pose matrix, `YAW_BINS/PITCH_BINS`), `PhotoSynthesisAction.tsx` | `/faceset/extract-from-media`, `/synthesize-36`, `/synthesize-photos`, `/save-fsz`, `/faceset/extraction/{id}`, `/faces/{bank/edit,bank/discover,analyze_photo,fill_missing_angles,extract_from_frame}`, `/facesets/{id}/{views,pose_matrix,thumb,rename,download}` |

### 2.2 Angle/yaw handling in the UI (axis 1)

- Coverage grid is **yaw âˆ’90â€¦+90 in 15Â° steps (12 bins) Ã— pitch âˆ’60â€¦+60 in 15Â° steps (8 bins) = 96 cells** (`utils/coverageRadar.ts:3-6`); values beyond Â±90Â° clamp into edge bins (`:15`); there is no representation for |yaw|>90Â°.
- Guidance (`guide/coverageGuide.ts:5-11`): frontal 0Â°, three-quarter Â±45Â° **required**, profile Â±75Â° **recommended**.
- Face overlay 3Ã—3 pose cell edges Â±15Â° yaw / Â±10Â° pitch to match backend (`faceOverlayHud.test.tsx` second test).
- The UI never displays backend refusal/engage reasons (`POSE_OUTSIDE_OBSERVED_COVERAGE`, `pose_too_oblique`, `LANDMARK_LAYOUT_IMPLAUSIBLE` â€” grep over `src/` returns none outside tests); `LiveFace` carries only `track_id, box, landmarks5, yaw, pitch, roll` (`api.ts:25`). Backend limits therefore are invisible to the user [gap].
- Render modes exposed: `standard`, `pose_3d`, `reenactment_3d`; swap scope `full_craniofacial | adaptive_skull | inner_face`, `skull_blend` (`workstation/renderModes.ts:17-30`, `pipeline.ts`). Pose mode `retarget | mesh_blend`; `pose_frontalization`, `tps_expression_weight` 0..1 (`pipeline.ts`).

### 2.3 Quality/blending controls surfaced (axis 3)
Restorer ids `gpen | codeformer | ultramax | gfpgan` with weight files `GPEN-BFR-512`, `CodeFormerv0.1.onnx`, `codeformer.fp16.onnx` (UltraMax = CodeFormer + Laplacian detail fusion, `workstation/pipeline.ts:1-12`); `color_method lab|reinhard_lab|histogram`; `restoration_blend`, `feather_px`, `erosion_px`, `color_strength`, `fidelity_weight`, `identity_blend_t`, `bypass[restorer|color]`, `relight`, hair/neck/ear protection (`studio/protection.ts`), 52 ARKit expression blendshapes (`studio/arkit52.ts`), occlusion `standard|robust`, export profiles `standard_mp4 â€¦ lossless_intermediate, image_sequence, color_preserving`, `hdr_policy`, passes (alpha/occluder/uv/â€¦; `export/VfxExportPanel.jsx`), cube LUT + grain finishing (`contracts.ts:ExportOptions`). **No explicit frequency-band / multiband-blend control** is exposed in the UI (grep `multiband|laplacian` outside test/diagnostic text returns only the UltraMax note).

### 2.4 Compute surfaces (axis 6) and queue (axis 7)
Provider badges derived from `advertised_providers` (`hud/PipelinePill.jsx:22-60`); `Capabilities.workstation.tensorrt_mixed_precision{available,active,configured,tensorrt_available,cuda_fp16_available,active_provider,diagnostic}` and `active_hardware_profile 12gb|6gb|cpu` (`contracts.ts:6-20`); `POST /models/compile_trt` (`EngineModelRack.tsx:135`), `/system/purge-vram`, `/system/benchmark`, `/system/check-compatibility`, `/system/preflight-lint` (one-click "Convert & Optimize Now", `PreFlightLinterModal.jsx`). Job states `QUEUED PREFLIGHT ANALYZING PRESCANNING PREPASSING RENDERING MUXING PAUSED COMPLETED FAILED RETRYING CANCELLED` (`contracts.ts:21`) with `QueueSlice{slice_index,start_seconds,status,retry_count,frame_count}` (GOP-aligned segments).

## 3. Backend endpoint & WS contract actually consumed by the UI (static scan of `src/`)

Prefix `/api/v1` is added by `api()` (`api.ts:14`) except where noted. Error envelope `{error:{code,message,request_id}}` â†’ `ApiError` (`api.ts:12`); GPU-busy preview = HTTP 503 `{busy:true,stage,retry_after_ms}` â†’ `PreviewBusyError` (`api.ts:24-31`).

| Area | Paths (UI-side) |
|---|---|
| Config/capabilities | `/config`, `/system/capabilities`, `/health`, `/pipeline/status` (also unversioned `/api/pipeline/status`), `/models/active` |
| Preview | `POST /api/preview` (unversioned), `/media/preview-frame`, `/media/detect-frame-faces`, `/media/inspect-faces`, `/media/inspections/{id}`, `/workstation/preview`, `/studio/{start,stop}` |
| Media | `/media/{info,frame,filmstrip,probe,keyframes,content,concat,split-boundaries,export-profiles}`, `/assets/uploads/{id}?offset` (chunked upload) |
| Jobs/projects/queue | `/jobs`, `/jobs/{id}/{pause,resume,cancel,checkpoint,download,artifacts/video,artifacts/manifest}`, `/queue/{id}/{action,retry}`, `/queue/reorder`, `/projects/{id}`, `/projects/{id}/{jobs,resume,validate,download}`, `/projects/import`, `/exports/{id}/{timeline,manifest,verify}`, `/finishing/{capabilities,manifest,verify,<action>}` |
| Identity/target | `/identity/{locks,locks/{id},inspections/{id}}`, `/target/{lock-from-frame,add-angle-sample,{id},{id}/match,{id}/rescan,{id}/angle/y/p/crop}`, `/actors/{inspection_id}` |
| FaceSets/3D | `/facesets`, `/facesets/{build,open-folder,rebuild-thumbs,{id},{id}/views,{id}/pose_matrix,{id}/identity-asset(/mesh),{id}/bake-atlas}`, `/faceset/{3d-state,reconstruct-3d,extract-from-media,save-fsz,synthesize-36,synthesize-photos,extraction/{id}}`, `/reconstructions/{id}`, `/reconstruction/backbones`, `/rendering/studio-environments` |
| Models | `/models`, `/models/{rescan,hub,manifest,download,verify-and-repair,downloads,compile_trt,batches/{id}}`, `/installer/{preflight,offline-health,models/import,models/scan,models/verify,updates/backup,updates/rollback}` |
| Diagnostics | `/system/{pipeline,runtime-report,preflight-lint,purge-vram,benchmark,check-compatibility,diagnostics-report?format=json|markdown}`, `/settings/{benchmark_status,benchmark_threads}` |
| WebSockets | `/ws/telemetry`, `/api/v1/ws/events`, `/ws/v1/pipeline/status`, `/ws/v1/preview/stream` (binary) |

## 4. Audit scripts (`app/roop_3d/audit/`) â€” behaviour today

### 4.1 `local_models.py` (1323 L; run as `python app/roop_3d/audit/local_models.py â€¦` from checkout root; `repo = parents[2]` = `app/`)
CLI (`main`, `:1262-1320`): `--verify`, `--require-installed`, `--list`, `--download [IDâ€¦]`, `--repair`, `--accept-license`. Output: JSON to stdout, exit 0/1; errors wrapped in the `AuditError.envelope()` (`isolation.py:19-25`).
- **default (no flag)**: re-discovers weights (`.onnx .pth .pt .ckpt .safetensors .bin`, `WEIGHT_EXTENSIONS :50`) under scopes `{repository=legacy_root, ~/.cache, ~/.insightface, ~/.torch}` (`default_scopes :173`; `legacy_root` = `PINOKIO_HOME/api/roop-ultimate`, enforced `read_only:true` in `model_sources.json`, `project_paths.py:19-30`), refuses to drop any model present in `local_models_discovered.json` (`MODEL_DISCOVERY_INCOMPLETE`), rewrites `verified_local_models.json` (95 models: swapper 15, restorer 13, detector 11, animation 11, recognizer 9, parser 8, other 28) + `model_and_license_register.csv`.
- **`--verify`** (`:1285-1307`): (a) `verify_registry` (`:147-170`) â€” non-empty `models[]`, no duplicate `(scope,path)`, every state must be `VERIFIED_LOCAL`, file re-hashed with SHA-256 and size compared (`MODEL_DRIFT`), path via `safe_path` (rejects symlink/junction ancestors); (b) `verify_open_artifacts` â€” pinned **FLAME 2023 Open** `flame2023_Open.pkl` sha256 `e75a0990â€¦7623`, 53 023 668 B, CC-BY-4.0, searched at `app/roop_3d/.runtime/models/flame2023_open/` and `flame/flame2023/` (`:185-206`); (c) `verify_local_cc0_assets` (`:221`) â€” **builds** `makehuman_cc0_anatomical_pack.npz` if missing (`AnatomicalFullHeadAsset.load(auto_build=True, verify=True)`, `allow_pickle=False`) and records digest â€” i.e. `--verify` can write a file; (d) `universal_manifest_report` (`:1217`) â€” 27-entry `universal_model_manifest.json`; fails on any `CORRUPT` entry, and (only with `--require-installed`) on non-optional entries that are `MISSING/PARTIAL/NOT_APPLICABLE`.
- Hash-trust note: `VERIFIED_LOCAL` = bytes hashed, **not** licence/provenance/inference verified (module docstring `:1-5`).
- Manager CLI (`UniversalModelDownloadManager`, `:645+`): resumable Range download to `.part`+meta, mirror failover, SHA-256 verify, quarantine of corrupt files, licence gate (`--accept-license`), disk-space check (`_check_space :1012`), optional TRT compile step (`_compile :1020`), dependency-ordered batches (`_order :1040`), `verify_and_repair :1187`.

### 4.2 `verify_isolation.py --report` (â†’ `stage00.py`)
Docstring promises a "complete, fail-closed Stage 00 audit" (`verify_isolation.py:2-8`). Actual behaviour: prints a fixed JSON object `{ok:true, historical:{checked:2645,expected:2645}, comprehensive:{checked:2645}, legacy_access:"read-write", standalone_app:"roop-3d", elapsed_seconds}` (`stage00.py:33-44`). `--report` writes it to `roop_3d/audit/stage00_verification.json` (untracked; present on disk, ts 2026-11-12). No file is hashed. `isolation.verify()` (`:184-197`) is a stub; `isolation.safe_path` still rejects link traversal (`:53-67`) and is the only live protection, used by `local_models`. `runtime_policy.verify_policy` (`:30-51`) does walk files, flags `LINK_TRAVERSAL`, and reports runtime files (`app/output`, `app/projects`, `app/temp`, `logs`, `app/queue.json`â€¦) â€” but nothing invokes it. Meaning: the baselines `protected_paths.json` (3038 records of **roop-ultimate** at commit `71ebb65â€¦`, 920 untracked) and `protected_all_paths.json` (16.9 MB, 69 890 files, untracked) are dead evidence. `AGENTS.md` confirms the policy change ("All legacy file protections â€¦ are removed").

### 4.3 Other audit-side scripts
`stage15_media_validation.py` re-measures colour/media claims â†’ `MEDIA_COLOR_STAGE15_VALIDATION.json`; `stage2_profiler.py` (717 L, 15 bottleneck profile) and `stage9_benchmarks.py` (611 L, 10 categories) write `STAGE2/STAGE09_BENCHMARKS.json`; `procedural_assets.py` zero-network builders for `makehuman_cc0_head`, `anatomical_pack_cc0`, `cc0_probe_bundle`, `correspondence_maps`; `project_paths.py`, `runtime_policy.py`, `isolation.py`, `stage00.py` as above.

## 5. Test coverage matrix for the requested behaviours (names verified by grep)

| Behaviour | Tests (file::test) | Notes |
|---|---|---|
| **Extreme yaw (synthetic)** | `test_extreme_pose_3d_swap.py::test_full_mesh_silhouette_survives_extreme_camera` (yaw âˆ’85,âˆ’80,âˆ’60,âˆ’35,0,35,60,80,85, `:73-75`), `::test_gpu_depth_culls_hidden_texture_without_horizontal_smearing` (Â±80/85, `:94`), `::test_dense_pnp_and_composite_modify_full_forehead_cheek_and_jaw` (`:119`), `::test_standard_session_routes_extreme_pose_to_mesh` (`:207`; auto route >30Â° yaw / >25Â° pitch per `docs/EXTREME_POSE_3D.md:13`), `::test_uv_acquisition_never_bakes_visible_texture_into_hidden_surface` (Â±75/85, `:268`); `test_reenactment_engine.py::test_large_yaw_follows_the_target_when_the_identity_was_observed_there` (45Â°Â±6, 70Â°Â±8, `:98-99`), `::test_pose_beyond_the_observed_coverage_falls_back_instead_of_extrapolating` (reason `POSE_OUTSIDE_OBSERVED_COVERAGE`), `::test_profile_is_refused_not_forced` (88Â° â†’ not engaged); `test_jaw_fit.py::test_a_moderate_yaw_is_not_gated_but_a_large_one_is` (20Â° ok, 55Â° â†’ `pose_too_oblique`); `test_heterogeneous_scheduler.py::test_mesh_raycaster_masks_out_far_cheek_at_60_degree_yaw`; `test_gpu_preview_and_profile_fixes.py::test_turn_ratio_separates_frontal_from_profile_landmarks` (`JAW_CUTOFF_MIN_TURN_RATIO`) and `::test_chin_line_clamp_cuts_below_the_chin_at_any_yaw`; `test_occlusion_real_models.py::test_robust_compositor_survives_a_real_profile_that_defeats_the_parser` | Real profile footage: `test_real_target_frames` (**skips**: fixtures absent); real-model identity/occlusion suites also environment-gated |
| **Scene cut** | `test_coupled_smoothing.py::test_gap_rewind_time_jump_and_scene_cut_restart_the_trajectory` (`:88-96`), `test_head_tracking_3d.py::test_first_frame_scene_cut_flag_is_not_a_cut` / `::test_an_undetected_change_of_person_inside_a_shot_becomes_an_identity_break_cut` / `::test_the_identity_lock_is_per_shot`, `test_identity_tracker_policy.py::test_scene_cut_parks_tracks_and_only_reacquires_by_strict_appearance`, `test_occlusion_compositor.py::test_tracker_resets_on_scene_cut_frame_gap_appearance_jump_and_holds_with_limit` + `::test_scene_cut_and_lost_tracks_reset_occlusion_history`, `test_occlusion_real_models.py::test_occlusion_tracker_detects_a_real_shot_change_from_pixels_alone`, `test_reenactment_engine.py::test_scene_cut_discards_all_temporal_state` (`:332`), `test_temporal_consistency.py::test_mask_recovery_respects_current_occluder_and_scene_cut`, `test_view_selection.py::test_gap_scene_cut_and_one_off_preview_choose_the_best_views_at_once`; UI: `TrackLanes` e2e `studio.e2e.ts` | `docs/TEMPORAL_STABILITY.md`: scene-cut byte-identical to un-stabilised input; >70 % flicker-reduction gate **not met** (26.9 % warp-error / 28.0 % Laplacian reduction at 0.25 history weight; 43.0 % at 0.40, 52.4 % at 0.50) |
| **Mesh generation** | `test_identity_asset_{fit,build,texture,quality,makehuman,topology_api}.py`, `test_makehuman_head.py`, `test_mesh_bridge.py`, `test_glb_export.py` (34 tests: GLB header/chunks/shape keys/head track/`/mesh.glb` route), `test_stage08_{flame,geometry,mesh_io,tps,uv}.py`, `test_stage11_faceset_mesh.py`, `test_dense_face_{backbone,bundle}.py`, `test_dense_correspondence_visibility.py`, `test_displaced_surface.py`, `test_anatomical_pack.py` (60 orthogonal namesâ€¦), `test_topology_registry.py`, `test_semantic_shape.py`, `test_articulated_head_models.py`, `test_full_head.py`, `test_flame_and_liveportrait.py` (Stage 16 **incomplete**: acceptance cmd 2 pass / 1 fail, `docs/REMEDIATION_STAGE16.md:1-5`); UI: `glbMesh.test.ts`, `headMesh.test.ts`, `faceset3dViewer.test.tsx` | synthetic ground-truth; dense backbone requires local 3-tier bundle (weights absent â†’ `test_dense_face_bundle.py` "Synthetic contract fixtures ONLY") |
| **Frequency / multiband blending** | `test_photorealistic_swap_core.py::test_laplacian_blend_reproduces_background_at_alpha_zero_and_foreground_at_one` (`:73`) / `::test_laplacian_blend_has_no_halo_across_a_soft_seam` / `::test_laplacian_blend_overshoot_is_bounded_by_the_inputsâ€¦` (`:389`), `test_stage09_adaptive_runtime.py::test_fast_multiband_blend_numerical_bounds` (`:293`), `test_stage11_faceset_mesh.py::test_multiband_blend_feathers_exposure_step_and_leaves_empty_texels_empty`, `test_hires_handoff.py::test_multiband_blend_keeps_fine_detail_and_zero_alpha_exact`, `test_stage09_compositor.py::test_frame_compositor_temporal_detail_reduces_laplacian_flicker`, `test_shading_gain.py` (low-frequency shading gain only), `test_displaced_surface.py::test_high_frequency_displacement_between_native_vertices_is_not_lost`, `test_seam_free_swap_pipeline.py::test_laplacian_jaw_adaptation_removes_the_jaw_width_mismatch`, `test_rendering_calibration.py::test_intrinsic_uv_cleanup_keeps_four_observed_detail_bands` | synthetic scenes; no real-footage perceptual metric except `test_stage11_evaluation_framework.py` (metric primitives, 11 dataset categories) |
| **CLI / terminal / report** | `test_pinokio_terminal_telemetry.py` (46 tests: env lock, HUD width, defusal of control chars, `ROOP3D_HUD=off`, TRT-BUILDER heartbeat, 14-section report), `test_diagnostics_suite.py::test_14_section_structured_runtime_report`, `test_e2e_video_pipeline.py::test_the_terminal_hud_saw_the_stream_and_the_remux`, `test_gpu_preview_and_profile_fixes.py::test_the_terminal_shows_the_angle_scan_phaseâ€¦`, `test_stage19_cli_model_root.py` (offline `local_models` CLI uses server data root), `test_stage16_preflight.py`; UI: `runtimeReport.test.tsx` (22) | `tools/project_admin.py` CLI itself has **no** dedicated test file named for it (covered indirectly by `test_standalone_install.py`, `test_stage16_reset_safety.py`) [INFERENCE] |
| **Queue resume / checkpoint** | `test_stage09_stream_resume.py` (finished-segment resume from `job_checkpoint.json`, corrupt segment, endpoint), `test_async_runner.py::test_pause_and_resume_with_frames_in_flight`, `test_stage12_worker.py::worker_pause_checkpoint_râ€¦`, `test_stage18_recovery.py` (26: cancel/kill/starve), `test_remediation_stage7.py` (NPZ checkpoint round trip, corrupt rejected, pickle refused), `test_remediation_stage3.py::test_prepass_checkpoint_identity_and_tamper_checks`, `test_postproduction.py::test_pass_checkpoints_resume_after_pause_during_baked_encoding`, `test_stage09_remediation.py` (batch-matrix strategies + persistent queue), `test_stage15_resilience.py` (tier-4 mid-job cancel, pause/resume, port collision), `test_stage08_offline_workflow.py::resume_binds_models_inspeâ€¦`, `test_batch_matrix_and_checkpoint_history.py`; model downloads: `test_model_downloader.py` (53) + `test_stage03_downloader.py` Range resume; UI: `ResumeNote`, `batchAndHistory.test.tsx`, `sessionReload.test.tsx` | long-run: `RELIABILITY_STAGE18.md`: 30- and 60-min soaks **not executed** (AV kill); only scaled soaks in `docs/reliability/*.json` |
| **UI e2e (browser)** | 17 Playwright specs, Â§7 | require running backend or `mock-server` (all values fixed placeholders, `mock-server/server.ts:1-8`) |

Test-suite health recorded in docs: `docs/reviews/roop3d-bug-audit-2026-11-08.md` â€” Vitest 447 passed/52 files, Playwright 70 passed, backend focused 96 passed/4 skipped; **full Python suite (~1,600 tests then) did not finish in 600 s**. Current static count: 2167 Python test functions + 859 Vitest cases + 112 Playwright cases. `logs_baseline_pytest.txt` (untracked, not read) exists at root.


## 6. `app/roop_3d/tests/` â€” 202 files, one row each

Counts are static `def test_*` definitions (parametrize not expanded); total **2167** test functions across 202 files (pytest, `roop_3d/pyproject.toml`). Flags column: static keyword hits in the file body (yaw/extreme-pose, scene-cut, mesh, frequency/multiband/Laplacian, CLI/terminal, resume/checkpoint/recover) â€” a *hint*, not proof of coverage; the coverage matrix in Â§5 is name-verified.


### Group 1. Detection, landmarks, pose, tracking, identity lock, temporal smoothing  (22 files, 279 tests)

| File | #tests | Purpose (docstring or first test names) | Flags |
|---|--:|---|---|
| `test_angle_diagnostics.py` | 12 | tests: decomposition_recovers_similarity_parameters, second_difference_ignores_constant_velocity_b, affine_jitter_is_scale_and_unwrapped_angle_aw | yaw,resume |
| `test_angle_diagnostics_job.py` | 9 | The runner configures the angle diagnostics from the job, publishes the JSONL atomically and attests it in the manifest. | yaw |
| `test_coupled_smoothing.py` | 10 | Size-normalised, coupled smoothing of landmarks and the dense fit. | yaw,scenecut |
| `test_head_tracking_3d.py` | 43 | Head/gaze tracker: filters, per-shot identity lock, gaze, lighting, shot boundaries and the tracker as a whole. | yaw,scenecut,mesh,resume |
| `test_head_tracking_stream.py` | 25 | Scene adaptation (the 16-frame / 3-second calibration pass), the video runner and pipeline tap, and live telemetry on | yaw,scenecut,mesh,resume |
| `test_identity_lock.py` | 24 | Identity lock: report, hash-bound mapping, staleness, render guard, run report and job binding. | yaw,scenecut,resume |
| `test_identity_real_inference.py` | 3 | Identity lock on REAL local detector + ArcFace inference with exact ground truth. | yaw,scenecut,resume |
| `test_identity_tracker_policy.py` | 20 | Identity-lock tracking policy: analytic scenarios with exact ground truth. | scenecut,resume |
| `test_inspection_gpu.py` | 10 | Inspection / target-lock models (detector, ArcFace, 1k3d68) run on CUDA on big cards, on the CPU otherwise. |  |
| `test_neural_face_registration.py` | 4 | The dense swap crop is registered to the swapper template from landmarks, never framed by the mesh silhouette. | yaw,mesh |
| `test_persistent_target_registry.py` | 13 | tests: strict_profile_gate_keeps_slot_after_reentry, fifteen_pose_buckets_retain_best_quality, hungarian_assignment_avoids_greedy_fragmentat | yaw,resume |
| `test_remediation_stage5.py` | 29 | Stage 5: 3D pose, Stage 0 pose matrix, frontalization, TPS expression and "Use Face From Frame". | yaw,cli,resume |
| `test_remediation_stage8.py` | 5 | tests: noise_and_turn, pose_hysteresis, bezier_exact | yaw |
| `test_resampling_borders.py` | 4 | Replicated borders (no black wedges) and alias-free photograph caching. |  |
| `test_stage04_pose_selector.py` | 6 | Unit tests for pose-bin classification and nearest-view reference selector. | yaw |
| `test_stage06_pipeline.py` | 10 | Real local reference inference plus deterministic tracking/alignment contracts. | yaw,scenecut |
| `test_stage12_governor.py` | 4 | tests: six_gigabyte_profile_is_single_stream_and_rel, twelve_gigabyte_profile_uses_live_free_memory, low_live_budget_selects_cpu_and_oom_detection |  |
| `test_stage12_target_lock.py` | 7 | Stage 12: rank-preserved person clusters, pose-bridging bank, face analysis validity rules. | yaw |
| `test_swap_presence.py` | 7 | A tracker gate that closes for a frame or two fades the swap instead of popping it. | yaw |
| `test_target_ui_endpoints.py` | 8 | Endpoints behind the face-bank UI: angle crops, unlock, rescan, and the engine status / switch-failure read-back. | yaw |
| `test_temporal_consistency.py` | 16 | tests: so3_pose_filter_damps_roll_jitter_and_takes_s, so3_filter_resets_on_head_whip_and_accepts_sc, detail_stabilizer_is_identity_on_first_frame_ | scenecut,freq,resume |
| `test_view_selection.py` | 10 | Source-view selection: pose cost, hysteresis, and weights that cross-fade instead of popping. | yaw,scenecut |

### Group 2. 3D reconstruction, mesh, FaceSet/.fsz, identity asset, FLAME, reenactment, expression  (51 files, 583 tests)

| File | #tests | Purpose (docstring or first test names) | Flags |
|---|--:|---|---|
| `test_3d_relighting.py` | 3 | Stage 17 Verification Gate: 3D Relighting & Normal Map Estimation. | yaw,resume |
| `test_albedo_prior.py` | 15 | Data-free albedo prior: robustness, ground-truth extrapolation, refusal without skin, and the completion far-field hook. | resume |
| `test_anatomical_pack.py` | 7 | Unit tests for the CC0 full-head synthesis and renderer-neutral dynamics. | mesh,freq |
| `test_animation_history_roles.py` | 7 | Frame curves, protected role intervals and non-destructive compositor behavior. |  |
| `test_articulated_head_models.py` | 7 | Contract tests for the optional, clean-room articulated reconstruction graph. | yaw |
| `test_backbone_status.py` | 6 | The read-only backbone report behind the Inspector's "3D Head Model Backbone" dropdown. |  |
| `test_check_topology_assets.py` | 6 | The asset check tells a new user exactly what is missing, never fails an install, and never touches the network. |  |
| `test_dense_correspondence_visibility.py` | 11 | Analytic visibility regressions only, not evidence of real-model image quality. | mesh |
| `test_dense_face_backbone.py` | 14 | Analytical geometry and neural-routing regressions; no pretrained weights claimed. | yaw,freq,resume |
| `test_dense_face_bundle.py` | 16 | Synthetic contract fixtures ONLY. They are not models or quality evidence. |  |
| `test_displaced_surface.py` | 6 | Analytical refinement contracts, not pretrained skin-detail quality claims. | yaw,mesh,freq |
| `test_expression_and_synthesis.py` | 17 | Motion invariants, post-enhancement closure, wire parity and archive provenance. | yaw,scenecut,mesh |
| `test_expression_audio_director.py` | 20 | ExpressionAndAudioDirector: emotion offsets in psi-space and the audio lock for lip/jaw motion under occlusion. |  |
| `test_extreme_pose_3d_swap.py` | 13 | Full-head coverage, perspective visibility and old-jaw removal regressions. | yaw,scenecut,mesh |
| `test_flame_and_liveportrait.py` | 9 | Stage 16 software checks; hardware/media acceptance is a separate fail-closed gate. | mesh |
| `test_full_geometry_swap.py` | 3 | Stage 5B source-geometry lock and full-head UV completion contracts. |  |
| `test_full_head.py` | 8 | Full-head regression: the MakeHuman topology must survive the runtime rig. | yaw,scenecut,mesh |
| `test_glb_export.py` | 34 | Stage 12: binary glTF serialization (container, buffers, textures, shape keys, head track), the `/mesh.glb` route, and the | mesh |
| `test_identity_asset_build.py` | 17 | Stage 3 Phases D+E: the identity asset (contents, honesty, GLB inspection) and its persistence / reuse rules. | yaw,mesh,resume |
| `test_identity_asset_fit.py` | 11 | Stage 3 Phase B: multi-view landmark geometry. Assertions compare against a KNOWN synthetic ground truth, | yaw,mesh,resume |
| `test_identity_asset_makehuman.py` | 13 | Identity asset built on the open MakeHuman (CC0) head template: template contract, build, honesty, persistence, API. | yaw,mesh,resume |
| `test_identity_asset_quality.py` | 13 | Stage 3 Phase A: FaceSet quality / coverage analysis, plus the FaceSet-reader landmark fixes it relies on. | yaw |
| `test_identity_asset_texture.py` | 11 | Stage 3 Phase C: identity texture atlas. Atlas texels are compared with the KNOWN synthetic ground-truth texture. | yaw,mesh,resume |
| `test_identity_asset_topology_api.py` | 21 | ``/identity-asset/mesh`` topology selection and semantic sliders, plus ``/morphable-topologies``. | yaw,mesh |
| `test_jaw_fit.py` | 20 | Opt-in jaw-width fit of the canonical proxy head in the mesh pose-blend path (``PoseBlendProjector``, ``ReferencePipeline``). | yaw,scenecut,mesh,resume |
| `test_makehuman_head.py` | 15 | MakeHuman (CC0) head model: pinned sources, asset integrity, frame conventions and multi-view fitting. | yaw,mesh,resume |
| `test_mesh_bridge.py` | 20 | Barycentric mesh bridge: exact closest-point maths, then conversion of positions / normals / UV albedo in both directions. | mesh,resume |
| `test_native_face_geometry.py` | 7 | Native-camera, articulation and UV regressions without licensed model data. |  |
| `test_photometric_head.py` | 9 | Independent directional-light math and real hardware shader regressions. | yaw,mesh,resume |
| `test_pose_mesh_blend.py` | 9 | Geometric endpoints, depth visibility, curve plumbing and real local output. | yaw,mesh |
| `test_reenactment_engine.py` | 29 | Stage 4: temporal 3D reenactment engine, verified on a scripted synthetic performer with KNOWN pose, expression and timing. | yaw,scenecut,resume |
| `test_reenactment_session_real.py` | 5 | Stage 4 on REAL local inference: real SCRFD + 2d106 landmarks, real inswapper, real video, real multi-view FaceSet. |  |
| `test_seam_free_swap_pipeline.py` | 5 | Stage 6 end to end on the real MakeHuman head: jaw-width adaptation -> rasterised swap mask -> z-buffer / normals into the | mesh,freq |
| `test_semantic_shape.py` | 14 | Semantic shape sliders: every claim is re-measured from the *mesh*, not from the Jacobian the controller solved with. | mesh |
| `test_silhouette_replacement.py` | 38 | Stage 6B: silhouette replacement. A different head shape must not leave the target's old jaw behind (narrower source) or | mesh,resume |
| `test_source_texel_parsing.py` | 4 | Hair, neck, cloth and background texels of a source photograph are never conditioning appearance. | yaw |
| `test_stage04_faceset_reader.py` | 4 | Unit and integration tests for FaceSetReader. | yaw |
| `test_stage04_schemas.py` | 7 | Unit tests for FaceSet schemas, contracts, and V2 export functionality. | yaw |
| `test_stage04_validator.py` | 13 | Unit tests for archive security validator and zip-bomb defense. |  |
| `test_stage07_identity_panel_api.py` | 6 | Stage 7 identity-panel API: view thumbnails and the Stage 3 identity asset (status / explicit build / viewer mesh). | mesh |
| `test_stage08_flame.py` | 4 | Acceptance tests for Stage 08 FLAME license gating, robust losses, and parametric fitting. | yaw,mesh |
| `test_stage08_geometry.py` | 8 | Acceptance tests for Stage 08 3D geometry contracts and coordinate conventions. | yaw,mesh,resume |
| `test_stage08_mesh_io.py` | 2 | Acceptance tests for Stage 08 validated atomic mesh export (.obj, .mtl, .glb). | mesh |
| `test_stage08_tps.py` | 6 | Acceptance tests for Stage 08 2.5D landmark PnP solver and TPS/Delaunay warper. | yaw,mesh,resume |
| `test_stage08_uv.py` | 3 | Acceptance tests for Stage 08 visibility-weighted UV texture synthesis and confidence gating. | yaw,mesh |
| `test_stage11_faceset_mesh.py` | 7 | Stage 11: RBF deformation, welded normals, multiband blend, texture completion, padding mask, 68-point correspondence. | mesh,freq |
| `test_stage18_extensions.py` | 9 | Clean-room regression coverage for the Stage 18 expression, UV, and atlas additions. | freq |
| `test_stage22_blender_flame_extras.py` | 11 | Stage 22 geometry, restoration, grading, proxy and timeline utilities. | scenecut,mesh |
| `test_stage7_faceset_builder.py` | 5 | Curation invariants independent of installed GPU weights. | yaw |
| `test_swap_region_3d.py` | 18 | 3D swap mask beyond the inner face oval, the biharmonic jaw field, and the z-buffer rasteriser they share. | yaw,mesh,freq |
| `test_topology_registry.py` | 17 | Morphable topology registry: defaults, licence gating, SHA-256 pinning, safe parsing, and the FLAME 2023 Open frame. | mesh |

### Group 3. Compositing, blending, masks, colour, restoration, visual quality  (27 files, 350 tests)

| File | #tests | Purpose (docstring or first test names) | Flags |
|---|--:|---|---|
| `test_dense_compositor.py` | 8 | Compositor behaviour specific to dense 3D replacements: no second restorer, shading, coverage smoothing, presence. | yaw,freq |
| `test_gpu_compositor.py` | 11 | Stage 10 geometry, semantic exclusions, real CUDA binding and slot ownership. | freq,resume |
| `test_hair_neck_protection_controls.py` | 14 | Hair / neck / ear protection toggles and the skin micro-texture amount: real engine behind each control, opt-in on the wire. |  |
| `test_hires_handoff.py` | 16 | Defect #1: enhancer detail must survive the Stage 2 -> Stage 3 handoff, and UltraMax must never be bypassed by default. | freq |
| `test_occlusion_compositor.py` | 36 | Stage 5 occlusion compositor: unit and integration tests that need no model weights. | scenecut |
| `test_occlusion_real_models.py` | 7 | Stage 5 on REAL local inference: real BiSeNet, real XSeg / face-occluder networks, real tracker, real video. | scenecut |
| `test_photorealistic_swap_core.py` | 30 | Stage 17: numeric contracts of the photorealistic swap core and its primitives (synthetic scenes, no model weights). | mesh,freq,resume |
| `test_physical_camera.py` | 8 | Stage 10 physical-camera optics, relighting, assets, and UV denoising tests. | freq,resume |
| `test_postproduction.py` | 11 | Real codecs and generated pixels, no inference/model or sibling-app imports. | mesh,resume |
| `test_production_studio.py` | 9 | Studio contracts independent of physical cameras and proprietary weights. |  |
| `test_rendering_calibration.py` | 6 | Color-temperature, energy-budget, and intrinsic-detail regressions. | mesh,freq |
| `test_shading_gain.py` | 5 | Per-pixel lightness gain: the swapped face follows the target's low-frequency shading, rim included. | freq |
| `test_stage06_ultramax_restorer.py` | 3 | Stage 6: UltraMax is a real, selectable restorer in the preview and render paths (it used to be rejected outright). | freq |
| `test_stage09_compositor.py` | 10 | Acceptance checks for Stage 09 mask, restoration and frame fusion behavior. | scenecut,freq |
| `test_stage10_audio_sync.py` | 9 | Stage 10 audio timing, silence gating and safe lip-sync integration tests. | scenecut |
| `test_stage11_diffusion.py` | 9 | tests: diffusion_is_optional_and_local_candidates_re, cross_family_base_or_adapter_is_rejected_befo, loader_result_is_checked_for_family_before_us |  |
| `test_stage11_evaluation_framework.py` | 10 | Stage 11: Objective Visual Quality Evaluation Framework Tests. | yaw,freq |
| `test_stage15_color_science.py` | 25 | Stage 15 colour science: every number is checked against a published reference value or an |  |
| `test_stage15_linear_light.py` | 13 | Stage 15 linear-light compositing: analytic expectations, exact identity at the extremes, every blend mode. | yaw |
| `test_stage15_studio_polish.py` | 13 | Stage 15 Studio Polish VFX Finishing & Biophysical Solvers Acceptance Tests. | yaw,mesh,freq,resume |
| `test_stage27_workstation_final.py` | 33 | Final workstation stage: parameter curves for relight / occlusion, clean relit pass, transfer ledger, flamegraph lanes, linter. | yaw,resume |
| `test_stage8_studio.py` | 7 | Stage 8 studio integration. Neural sessions are replaced by explicit numerical fixtures. | yaw,resume |
| `test_strand_occlusion.py` | 21 | StrandAwareOcclusionCompositor: depth z-test, N.V grazing falloff, semantic and object cues, and the directional strand matte. |  |
| `test_swap_coverage.py` | 6 | Coverage of the dense swap: resolution-independent feather, landmark-hull bound, no nearest-neighbour stairs. | mesh |
| `test_vfx_compositing_engine.py` | 11 | Lead VFX Compositing & Semantic Segmentation Engine Tests. | mesh,freq |
| `test_vfx_export_and_diagnostics.py` | 7 | Comprehensive unit and integration test suite for Stage 12: | scenecut,cli,resume |
| `test_workstation3d_controls.py` | 12 | Workstation UI controls with a real engine behind them: ``hairline_offset_px`` (forehead matte band) and ``ambient_fill`` (manual light). | yaw |

### Group 4. Compute: ORT / TensorRT / CUDA / VRAM / hardware profiles  (17 files, 151 tests)

| File | #tests | Purpose (docstring or first test names) | Flags |
|---|--:|---|---|
| `test_accelerator_resolution.py` | 2 | tests: auto_prefers_tensorrt_then_cuda_then_cpu, explicit_choice_is_kept_and_cpu_wins_without_ |  |
| `test_execution_worker_pools.py` | 3 | tests: post_encode_pool_stitches_completed_chunks_in, gop_chunk_rejects_unaligned_start_and_nonfina, nvenc_chunk_runs_on_the_worker_with_lossless_ |  |
| `test_gpu_preview_and_profile_fixes.py` | 28 | Regression tests for the Nov 12 2026 fixes: concurrent-stage CUDA graphs, trim-window work, warm preview sessions, | yaw,mesh,cli |
| `test_heterogeneous_scheduler.py` | 8 | Verification Gate & Regression Suite for Stage 15 Heterogeneous Scheduler. | yaw,mesh |
| `test_nv_cvcuda_pipeline.py` | 14 | Stage 14 contracts and real device checks, with a fail-closed acceptance mode. | scenecut,mesh,freq |
| `test_remediation_stage2.py` | 28 | Stage 2: actual CPU graph validation and simulated native TRT failure paths. |  |
| `test_stage03_engine_cache.py` | 3 | tests: every_hardware_version_and_shape_component_ch, metadata_binds_actual_compiled_bytes_and_dete, shape_profiles_are_validated_and_onnx_cache_d |  |
| `test_stage07_cache.py` | 4 | tests: atomic_bundle_publication_and_thread_single_w, failed_build_never_publishes_and_lock_timeout, hardlinked_manifest_is_rejected | cli |
| `test_stage07_diagnostics.py` | 2 | tests: active_job_preserves_exact_provider_fallback_, capabilities_and_diagnostics_report_expose_ha |  |
| `test_stage07_runtime.py` | 6 | tests: named_outputs_reorder_without_swapping_values, expired_weak_proxies_do_not_break_ort_session, actual_ort_sessions_are_evicted_and_aliases_r |  |
| `test_stage07_validation.py` | 11 | tests: embedding_gate_detects_rotation_even_with_sma, contract_errors_never_turn_into_passing_parit, image_metrics_and_border_gate_detect_small_lo |  |
| `test_stage09_adaptive_runtime.py` | 10 | Tests for Stage 9 Unified Adaptive CPU/GPU Runtime and NVIDIA Optimization. |  |
| `test_stage17_ci_hardware.py` | 8 | Stage 17 Automated Tests: CI/CD Pipeline, Hardware Compatibility & Release System. | cli |
| `test_stage19_hardware_profile.py` | 3 | Stage 19: a real "12 GB" card reports slightly under 12288 MiB and must still get the 12gb profile. |  |
| `test_stage7_gpu_throughput.py` | 5 | Stage 7 dynamic reconstruction profiles, telemetry and VRAM governance. |  |
| `test_trt_mixed_precision.py` | 14 | TensorRT mixed precision policy, fallback chain, and opt-in GPU qualification. | mesh |
| `test_zero_copy_execution.py` | 2 | tests: ring_reuses_device_ortvalue_and_tracks_zero_i, ring_requires_the_declared_stage_order |  |

### Group 5. Job queue, checkpoint/resume, video IO, FFmpeg, streaming, live preview, reliability  (33 files, 401 tests)

| File | #tests | Purpose (docstring or first test names) | Flags |
|---|--:|---|---|
| `test_async_pipeline.py` | 22 | Defect #3: the asynchronous multi-stream pipeline, the shared/exclusive inference fence and hardware telemetry. | cli |
| `test_async_runner.py` | 5 | Defect #3: the job runner drives the four render stages through the asynchronous pipeline, correctly. | scenecut,resume |
| `test_batch_matrix_and_checkpoint_history.py` | 3 | tests: atomic_json_and_binary_checkpoint_creation_an, project_validation_badges_inputs_verified_vs_, media_split_boundaries_endpoint | resume |
| `test_e2e_video_pipeline.py` | 11 | Stage 17 end-to-end: generated .fsz + 30-frame .mp4 -> import/upload routes -> real job -> download -> ffprobe / SSIM / identity. | yaw,freq,cli |
| `test_live_stream.py` | 18 | Defect #3: binary live transport -- header, GPU/CPU JPEG, latest-wins hub, lazy cache entries, WebSocket, telemetry. |  |
| `test_reliability_harness.py` | 4 | The stub recovery harness must never read installed neural weights. | resume |
| `test_remediation_stage3.py` | 18 | Real host/codec/ORT tests; production GPU fidelity is a separate campaign. | resume |
| `test_remediation_stage4.py` | 19 | Stage 4: non-blocking live preview, cache-first serving and model-lifecycle mutex. | yaw |
| `test_remediation_stage7.py` | 19 | Stage 7: Stage 27 pre-flight lint checks, measured stage timings and NPZ project checkpoints. | resume |
| `test_stage02_jobs.py` | 7 | tests: complete_transition_matrix, durable_idempotency_conflict_and_secret_key_h, concurrent_creates_serialize_to_one_job | cli,resume |
| `test_stage02_media.py` | 6 | tests: real_container_probe_and_precise_frame_previe, media_read_boundary, preview_selectors_are_bounded_and_finite |  |
| `test_stage05_media.py` | 24 | Stage 05 acceptance uses generated real containers and real codecs only. |  |
| `test_stage05_timeline_frames.py` | 15 | Stage 5: the scrubbing frame service and its /media/info, /media/frame, /media/filmstrip endpoints. | yaw |
| `test_stage08_offline_workflow.py` | 9 | Stage 8 integration gates. Contract sessions are explicit fixtures; codecs/storage/API are real. | scenecut,resume |
| `test_stage09_remediation.py` | 8 | Stage 9 Automated Verification Gate: | resume |
| `test_stage09_stream_resume.py` | 3 | Stage 9: the streaming render path resumes from the last verified finished segment (job_checkpoint.json). | resume |
| `test_stage12_worker.py` | 3 | tests: post_job_runs_pipeline_publishes_manifest_and, worker_pause_checkpoint_resume_continues_to_s, simulated_cuda_oom_retries_on_cpu_without_par | yaw,resume |
| `test_stage13_bridges.py` | 18 | tests: ready_idempotence_and_conflict, missing_node_checkpoint_or_input_rejected_bef, progress_preview_correlation_and_reconnect | resume |
| `test_stage14_workstation.py` | 12 | tests: versioned_project_reopen_and_verified_checkpo, project_drift_blocks_checkpoint_continuation_, untrimmed_job_identity_remains_compatible_and | yaw,mesh,resume |
| `test_stage15_e2e_golden.py` | 1 | Stage 15: Golden-Clip End-to-End Video Swap Smoke Test. | yaw |
| `test_stage15_media_av.py` | 19 | Stage 15 audio, subtitles, timing, rotation and image sequences, all through real codecs and containers. |  |
| `test_stage15_media_extras.py` | 17 | Stage 15: input containers, still sequences (JPEG/EXR), validation, capabilities, API, OCIO, audio-sync extension. |  |
| `test_stage15_media_reader.py` | 8 | Stage 15 decode: colour-correct, explicit about HDR, untagged material resolved by recorded convention. |  |
| `test_stage15_media_writer.py` | 17 | Stage 15 encode: colour-correct output, honest HDR handling and verified export profiles (real codecs). |  |
| `test_stage15_resilience.py` | 5 | Stage 15: Resilience & Fault Injection Tests. | resume |
| `test_stage15_stream_color.py` | 4 | The raw NVDEC/NVENC stream backend must not change colour (software decode/encode; needs the ffmpeg CLI it shells out to). | cli |
| `test_stage18_growth_ledger.py` | 31 | Stage 18 unit tests: growth analysis, lifecycle ledger, disk handling, failure classification, orphan guard, redaction. | resume |
| `test_stage18_recovery.py` | 26 | Stage 18 integration tests: every way a long job can be stopped, killed or starved, and what must be true afterwards. | resume |
| `test_stage18_resources_live.py` | 6 | Stage 18: resources that need the real thing - ONNX Runtime on the real GPU, real allocation failures, server WebSockets, | resume |
| `test_stage18_soak_bounded.py` | 5 | Stage 18: scaled-down soak runs, judged against the bounded profile. The 30/60-minute runs use the same code via | resume |
| `test_stage23_workstation_parity.py` | 15 | Stage 23: multi-person actors, paint/erase mask ranges, GOP batch planning and the 10-state lifecycle. | resume |
| `test_video_job_runner.py` | 13 | Stage 17: the FFmpeg-piped loop (real ffmpeg, no model weights, no faces). | freq |
| `test_workstation_preview.py` | 10 | Preview contract, real compositor buffers and admission/lifecycle regressions. | scenecut |

### Group 6. CLI, terminal HUD, telemetry, diagnostics  (4 files, 58 tests)

| File | #tests | Purpose (docstring or first test names) | Flags |
|---|--:|---|---|
| `test_diagnostics_suite.py` | 3 | tests: preflight_render_linter_checks, synthetic_thread_benchmark_runner, 14_section_structured_runtime_report |  |
| `test_pinokio_terminal_telemetry.py` | 46 | Pinokio terminal HUD, 14-section runtime report and TensorRT build heartbeat. | cli,resume |
| `test_stage18_report_api.py` | 7 | Stage 18: the diagnostic report - contents, redaction, local-only behaviour and the audit/repair endpoints. |  |
| `test_stage19_cli_model_root.py` | 2 | Stage 19: the offline model commands must use the server's data root. |  |

### Group 7. Model registry, downloader, governance, engine selection  (17 files, 197 tests)

| File | #tests | Purpose (docstring or first test names) | Flags |
|---|--:|---|---|
| `test_local_models.py` | 11 | tests: hashes_match_actual_bytes_and_consumer_verifi, registry_consumer_detects_mutation, license_register_does_not_invent_permissions_ |  |
| `test_local_weight_trust.py` | 9 | Weights the user placed in app/models are auto-trusted, never blocked, and reported by the read-back. |  |
| `test_model_download_sources.py` | 8 | Verified public sources and the unpublished-rights acknowledgement path (offline). |  |
| `test_model_downloader.py` | 53 | Universal model download manager: resumable Range downloads, mirror failover, checksum recovery, licence gating, | mesh,resume |
| `test_model_downloader_integration.py` | 4 | Integration tests for model downloader endpoints, middleware alias, and preflight check action. |  |
| `test_model_registry.py` | 5 | Comprehensive test suite for Stage 13: | freq |
| `test_model_selection_regressions.py` | 9 | tests: catalog_reports_provider_weights_as_installed, catalog_does_not_verify_corrupted_provider_we, detector_selection_changes_the_loaded_graph | yaw |
| `test_stage03_api.py` | 3 | tests: api_exposes_authoritative_registry_notices_an, api_local_first_reuse_does_not_download_unkno, api_download_gate_and_real_cached_adapter_res | resume |
| `test_stage03_downloader.py` | 19 | tests: atomic_download_digest_and_reuse, interruption_is_fsynced_and_next_call_resumes, range_resume_with_matching_identity | cli,resume |
| `test_stage03_registry.py` | 9 | tests: manifest_preserves_all_observed_identities_wi, verifier_detects_digest_size_and_permission_e, local_provider_returns_original_path_and_neve |  |
| `test_stage06_engine_selection.py` | 4 | tests: older_jobs_and_explicit_reference_keep_origin, unavailable_selection_rejected_before_media_a, worker_dispatches_explicit_adapter_and_never_ |  |
| `test_stage06_model_hub.py` | 10 | tests: hub_exposes_registry_evidence_and_keeps_exper, hub_finds_hash_verified_artifacts_in_configur, hub_uses_runtime_first_match_for_duplicate_co |  |
| `test_stage12_model_governance.py` | 9 | tests: all_canonical_and_supplemental_records_materi, scanner_requires_integrity_compatibility_lice, unknown_license_record_cannot_become_an_infer | resume |
| `test_stage14_orchestrator.py` | 28 | Stage 14 orchestrator acceptance: deterministic selection, no unsupported configuration, no silent | yaw |
| `test_stage16_model_bundle.py` | 9 | Unit tests for Stage 16 model manager, bundle import, archive traversal security, and hash verification. |  |
| `test_studio_environment_routes.py` | 4 | Stage 13: ``GET /api/v1/rendering/studio-environments`` serves the five CC0 lighting probes in a browser-ready form. |  |
| `test_swapper_catalog_and_adapter.py` | 3 | tests: models_api_lists_all_engine_catalog_entries, local_rescan_exposes_custom_extensions_and_un, dummy_gpu_inference_respects_native_swapper_c |  |

### Group 8. Platform: scaffold, API contracts, security, installer, packaging, docs  (18 files, 148 tests)

| File | #tests | Purpose (docstring or first test names) | Flags |
|---|--:|---|---|
| `test_git_boundary.py` | 4 | tests: boundary_accepts_only_independent_app_source, guard_reads_real_git_index, dedicated_remote_is_required |  |
| `test_packaging.py` | 4 | The ``[tool.setuptools] packages`` list decides what a built wheel contains. A package missing from it works from a source |  |
| `test_runtime_policy.py` | 7 | tests: runtime_activity_is_reported_without_rewritin, runtime_policy_allows_unified_source_changes, runtime_prefix_does_not_misclassify_local_sou |  |
| `test_stage00.py` | 5 | Real filesystem tests for unified architecture. |  |
| `test_stage01_scaffold.py` | 15 | Offline tests of the real ASGI application, socket ownership and audit gate. |  |
| `test_stage02_api.py` | 9 | tests: all_rest_contracts_are_in_openapi, config_put_backup_reload_secret_rejection_and, job_rest_idempotency_controls_and_validation | resume |
| `test_stage02_boundaries.py` | 2 | tests: real_directory_link_is_rejected_before_reads_, atomic_replace_does_not_modify_hardlinked_inp |  |
| `test_stage02_settings.py` | 7 | tests: invalid_catalog, catalog_has_all_requested_keys, config_atomic_backup_and_explicit_migration |  |
| `test_stage13_security.py` | 26 | Complete Security, Privacy, and Synthetic Media Protection Test Suite for Stage 13. | mesh,resume |
| `test_stage16_installer_api.py` | 1 | Integration tests for Stage 16 installer REST API endpoints. | resume |
| `test_stage16_offline_health.py` | 2 | Unit tests for Stage 16 offline dependency cache and offline readiness verification. |  |
| `test_stage16_preflight.py` | 14 | Unit tests for Stage 16 environment preflight and hardware profile detection. |  |
| `test_stage16_reset_safety.py` | 3 | Unit tests for Stage 16 reset safety and boundary isolation. |  |
| `test_stage16_share_and_pipeline.py` | 15 | Single-origin hardening: share-mode token, unversioned API aliases, hardware profile, pipeline frames. |  |
| `test_stage16_update_rollback.py` | 3 | Unit tests for Stage 16 update checking, configuration backup, and safe rollback. |  |
| `test_stage19_docs.py` | 13 | Stage 19: the user documentation must reflect actual application behaviour. | yaw,resume |
| `test_standalone_app.py` | 3 | tests: project_has_its_own_git_directory_and_environ, unified_app_allows_direct_writes_and_imports, pinokio_menu_is_independent_and_has_a_real_st |  |
| `test_standalone_install.py` | 15 | Standalone install integrity: real (non-linked), gitignored local directories and in-sync npm manifests. | resume |

### Group S. Test support / manual acceptance modules (non-collected helpers)  (13 files, 0 tests)

| File | #tests | Purpose (docstring or first test names) | Flags |
|---|--:|---|---|
| `audio_world.py` | 0 | Synthetic speech with KNOWN structure, shared by the director's unit and integration tests. |  |
| `identity_world.py` | 0 | Synthetic ground-truth world for the Stage 3 identity-asset tests (test-only; never imported by production code). | yaw,mesh,freq,resume |
| `manual_stage02_acceptance.py` | 0 | Explicit real-model/media acceptance campaign, separate from offline fixtures. | resume |
| `manual_stage03_acceptance.py` | 0 | Rehash every installed registry artifact through the production local provider. |  |
| `manual_stage04_acceptance.py` | 0 | Stage 04 acceptance campaign: FaceSet reader, pose selector, and V2 schema. | yaw |
| `manual_stage06_acceptance.py` | 0 | Real 100-frame reference campaign; all media and biometric data remain local. | yaw,cli |
| `manual_stage07_acceptance.py` | 0 | Capture real Stage 06 golden tensors; qualify providers and measure p50/p95. | yaw,cli |
| `manual_stage08_acceptance.py` | 0 | Prepare owned, local-media browser fixtures or verify its real rendered output. | yaw,cli |
| `model_governance_support.py` | 0 | Helpers for real-weight tests gated by current governance evidence. |  |
| `stage15_media.py` | 0 | Real-codec fixtures for Stage 15. No mocks: every file is encoded by libx264 / libx265 / FFV1 and |  |
| `swap_core_support.py` | 0 | Synthetic scenes with known ground truth for the Stage 17 swap-core tests (no model is loaded). | freq |
| `target_world.py` | 0 | Scripted target performer for the Stage 4 reenactment tests (test-only; never imported by production code). | yaw,scenecut,resume |
| `tracking_world.py` | 0 | Test-only helpers for the head-tracking tests (never imported by production code). | yaw,scenecut,mesh |

## 7. UI test inventory (Vitest `src/*.test.*`, Playwright `e2e/`, harness)

Static count of `it(`/`test(` calls: Vitest/mock-server **85 files / 859 cases**; Playwright e2e **17 files / 112 cases** (+ `acceptance/stage8.real.ts`, 2).  `it.each` not expanded.

| File | #cases | First describe / first test titles |
|---|--:|---|
| `acceptance/stage8.real.ts` | 2 | offline production UI: local imports â†’ identity/asset â†’ preview â†’ proj |
| `e2e/certification.e2e.ts` | 3 | the route table is what the browser actually serves ; every screen mounts and is reachable, one at a time, wit |
| `e2e/faceBank.e2e.ts` | 3 | persistent FACE pills, click-to-lock, live angle capture, hover inspec ; clicking a face box on the canvas tog |
| `e2e/facesetBuilder.e2e.ts` | 1 | builder renders the 3D sphere, curates angles and saves on desktop and |
| `e2e/finalWorkstation.e2e.ts` | 2 | the stream overlap tab draws the measured lanes, the copy verdict and  ; the sweep compares enhancers across r |
| `e2e/identityLock.e2e.ts` | 3 | contact sheet shows thumbnails, visible ranges, confidence and withhel ; explicit lock is sent with the render |
| `e2e/liveStream.e2e.ts` | 2 | binary frames reach the canvas with no React re-renders, and the newes ; an unmounted page leaves no open sock |
| `e2e/outputCompare.e2e.ts` | 8 | Before / After output player :: Before / After output player ; opens when the job finishes and loads both videos from the right URLs |
| `e2e/resourceLeaks.e2e.ts` | 2 | WebGL comparison canvas: 60 mount/unmount cycles leave zero live conte ; whole application: 15 laps of every s |
| `e2e/stage19.e2e.ts` | 31 | first-run onboarding and guided project :: first-run onboarding and guided project ; welcome dialog is accessible, traps focus, closes with Escape and is |
| `e2e/stage24.e2e.ts` | 13 | persistent queue (#/processing) :: persistent queue (#/processing) ; shows each job\ |
| `e2e/stage5Scrub.e2e.ts` | 7 | loading a target fetches info and the filmstrip, and the filmstrip sho ; the playhead frame is requested witho |
| `e2e/studio.e2e.ts` | 7 | four panels, timeline lanes and a FaceSet dragged onto a target track  ; clicking a face in the video selects  |
| `e2e/swapPreviewAudit.e2e.ts` | 6 | audit: fresh target inspect and dropdown assignment drives live swap p ; audit: classic layout assigns source  |
| `e2e/telemetry.e2e.ts` | 2 | a 30 Hz /ws/telemetry stream re-renders nothing in the parameter sideb ; the floating PiP preview stays visibl |
| `e2e/webgl.e2e.ts` | 8 | compiles and links both programs on a real WebGL2 context ; split wipe shows A left of the divider and B right |
| `e2e/workstation3d.e2e.ts` | 7 | the pipeline node flow bar shows eight stages with measured latencies  ; the coverage radar lists the missing  |
| `e2e/workstationControls3d.e2e.ts` | 7 | the buffer switcher renders wireframe, albedo, normals, depth and SH-r ; the SH-relit buffer follows the light |
| `mock-server/mockServer.test.ts` | 3 | /api/meta identifies the mock and every response is marked ; hardware, telemetry and health are served under u |
| `src/activeModelReadback.test.tsx` | 6 | active model read-back :: shows the exact weight file each stage has in VRAM ; says a newly selected model has not loaded yet, and never |
| `src/appShell.test.tsx` | 3 | mounts the integrated application shell with all nine screens and keep ; undoes parameter changes and mask str |
| `src/batchAndHistory.test.tsx` | 7 | BatchMatrixScreen :: renders all 4 batch strategies and allows switching tabs ; supports automatic video segment splitting toggle a |
| `src/chaptersAndCanvas.test.tsx` | 17 | chapter list operations :: adds chapters in time order with default names, clamped into the clip ; ignores a second marker within one fra |
| `src/coverageRadar.test.tsx` | 17 | radar grid model :: bins on the exact backend 12 x 8 grid, clamped at the edges ; prefers the server bin_index over re-binning, an |
| `src/diagnosticsSuite.test.tsx` | 7 | Pre-Flight Render Linter Modal :: renders pre-flight checks and dispatches queue confirmation ; shows the seven measured stages from live timing |
| `src/engineStatusBar.test.tsx` | 4 | EngineStatusBar :: shows the live detector, swapper, enhancer (verified) and precision re ; says so instead of inventing a model  |
| `src/expressionEditor.test.tsx` | 5 | shows every requested 3D edit, scopes values and reset to the selected ; edits gaze and jaw per face, and the  |
| `src/faceBank.test.tsx` | 10 | faceIdentity :: gives each persistent id its own colour and an upper-case tag; unident ; reports each scan state only from the |
| `src/faceContextMenu.test.tsx` | 14 | FaceContextMenu keyboard and extraction hook :: right-clicking a face opens a menu with the three extraction actions a ; offers per-face canvas toggles (Ultra |
| `src/faceFromFrame.test.tsx` | 3 | Use Face From Frame :: posts the current frame and face number, then hands the new FaceSet to ; reports a failed extraction without s |
| `src/faceManager.test.tsx` | 7 | FaceSet Builder Utilities & Algorithms :: computes CRC-32 checksums accurately ; builds a valid ZIP container without external dependencies |
| `src/faceOverlayHud.test.tsx` | 11 | pose helpers (standard frame: yaw + = nose right, pitch + =  :: converts the server\ ; uses the same inclusive 3Ã—3 cell edges as the backend (Â±15Â° yaw, Â±10Â°  |
| `src/faceSelection.test.tsx` | 12 | useFaceSelectionStore state architecture and clustering :: computes cosine distance and similarity correctly ; clusters identities by cosine distance similarity threshol |
| `src/faceset3dViewer.test.tsx` | 21 | mesh frame conversion :: turns the backend mesh frame (+y down) into an upright, outward-facing ; swaps triangle winding for a mirrored |
| `src/facesetBuilder.test.tsx` | 1 | edits the active set, filters 96 bins, removes a crop, and activates t |
| `src/facesetFavorites.test.tsx` | 2 | FaceSet library favorites :: persists favorites, sorts them first, and does not select on favorite  ; exposes a favorite control in the com |
| `src/facesetInspector3D.test.tsx` | 3 | shows the radar for the stored views (pitch converted to the standard  ; uses the facial reconstruction regard |
| `src/finalWorkstation.test.tsx` | 15 | sweep patch semantics :: GPEN Realistic and UltraMax select the real restorer, keep a non-zero  ; sweeping relight strength switches re |
| `src/glbMesh.test.ts` | 16 | parseGlb on the server\ :: reads vertices in the viewer frame: the project\ ; keeps triangle winding (the serializer already swapped it f |
| `src/guardedCanvas.test.tsx` | 9 | GuardedCanvas :: draws the canvas when WebGL works ; shows the fallback with the browser\ |
| `src/hairNeckProtection.test.tsx` | 14 | protection helpers keep untouched requests byte-identical :: removes a control that is set back to its engine default ; the hairline inset is the engine offset with the si |
| `src/headMesh.test.ts` | 17 | prepareHeadMesh :: centres the mesh in the viewer frame, finds the nose tip and keeps uni ; converts the backend mesh and OpenCV  |
| `src/historyCurvesRoles.test.tsx` | 9 | uses reversible commands, discards redo branches, and coalesces a para ; evaluates linear, Bezier and hold cur |
| `src/identityLock.test.tsx` | 11 | contact-sheet formatting :: formats clock, visible ranges and withheld reasons ; shows thumbnails, track number, confidence, gaps and with |
| `src/lightProbeSphere.test.tsx` | 2 | LightProbeSphere :: renders light probe with auto-detected direction and color temperature ; allows switching to Manual 3D Virtual |
| `src/liveStream.test.tsx` | 12 | live frame wire format :: parses the 16-byte header: index, timestamp, three stage latencies, fl ; rejects empty payloads and foreign pr |
| `src/liveStudio.test.tsx` | 1 | starts and stops with the selected identity and keeps unavailable outp |
| `src/liveSwapPreview.test.tsx` | 14 | payload + filenames :: builds frame-accurate payloads from the job request and parameters ; names downloads roop3d_preview_<ISO times |
| `src/modelHubModal.test.tsx` | 28 | manifest parsing :: turns a row into a fully defaulted model and never throws on missing f ; rejects a response that is not a mani |
| `src/modelRack.test.tsx` | 14 | model status :: maps what is on disk to the five pills the workstation shows ; lists every registered model, installed or not, |
| `src/navigationAndControls.test.tsx` | 8 | keeps all nine deep-linked screens mounted and preserves local edits a ; evaluates arithmetic safely, supports |
| `src/outputComparePlayer.test.tsx` | 10 | points the two videos at the original target and the verified download ; keeps the original silent and the res |
| `src/pipelineFlowBar.test.tsx` | 4 | maps each stage to the measured latency fields, and reports unmeasured ; resolves the Normals chip to the norm |
| `src/pipelinePill.test.tsx` | 10 | pill model :: maps the three server stages onto Pre-Scan, Range Lock and Render & Mu ; treats a checkpoint-restored step as  |
| `src/pipelineSolo.test.tsx` | 6 | solos a pipeline node through POST /api/preview solo_node, returns to  ; offers exactly the eight pipeline sta |
| `src/poseViewer.test.tsx` | 3 | Fibonacci Sphere Lattice & View Coverage Heatmap :: generates an evenly distributed Fibonacci sphere lattice ; projects [yaw, pitch] view angles to appropriate la |
| `src/postproduction.test.tsx` | 6 | frame-accurate timeline interchange :: splits all boundaries, preserves gaps and offsets trimmed media frames ; writes rational XML time values, esca |
| `src/previewBox.test.tsx` | 8 | CompositionGridOverlay Component :: returns null when mode is off or empty ; renders Rule of Thirds SVG guides with golden power points |
| `src/previewFrameCache.test.ts` | 3 | preview frame cache :: evicts least recently used frames by count and encoded byte budget ; expires without extending freshness on re |
| `src/queueAndHud.test.tsx` | 6 | Hardware Diagnostics & HUD Logic (PipelinePill) :: correctly resolves hardware execution provider from telemetry ; identifies real-time pipeline bottlenecks base |
| `src/readinessChecks.test.tsx` | 10 | pre-flight linter modal :: sends the swapper and the export passes so the TensorRT and disk estim ; does not re-request the lint on every |
| `src/routes.test.tsx` | 4 | defines exactly the nine deep-linked routes with their names ; renders each of the nine routes: one visible sc |
| `src/runtimeReport.test.tsx` | 22 | trt_compilation_progress frames :: parses a build frame, clears on finished, and ignores everything malfo ; shows a live strip while compiling, w |
| `src/safeData.test.tsx` | 8 | fileName / shortId :: takes the last segment for either separator and survives non-strings ; fills every object the list renderers d |
| `src/sessionReload.test.tsx` | 7 | rebuilds the workstation from the saved session after the page reloads ; keeps saving the session as it change |
| `src/settingsAndSweep.test.tsx` | 17 | settings registry :: documents every setting with sane bounds, a recommended range inside t ; agrees with the application\ |
| `src/settingsDomainTabs.test.tsx` | 5 | SettingsInspector Phase 3 Domain Tabs & Validation :: renders all domain category tabs and defaults to Compute tab ; switches tabs on click and displays relevant co |
| `src/shell/shell.test.tsx` | 4 | command palette: labelled modal dialog, filters by label/group/keyword ; command palette: arrow keys move the  |
| `src/spatialInspector.test.tsx` | 6 | Tanner Helland Kelvin to RGB conversion :: converts warm 2700K incandescent light to warm amber hex ; converts neutral 5500K daylight to neutral white-bl |
| `src/stage11Canvas.test.tsx` | 23 | colormaps :: Inferno and Viridis hit their published endpoints and the LUT is 256 o ; Inferno brightens monotonically (so a |
| `src/stage11Math.test.tsx` | 7 | evaluateExpression :: evaluates ordinary arithmetic: 0.4 * 1.5 -> 0.6 ; treats a leading + * / as relative to the field value, so  |
| `src/stage11Model.test.tsx` | 21 | atomic stores :: notifies only on a real change and supports transient (render-free) se ; re-renders a subscriber only when its |
| `src/stage11Perf.test.tsx` | 4 | 30 Hz telemetry isolation :: re-renders none of the parameter sidebar, timeline, Face Manager, Vide ; control: an App state change DOES re- |
| `src/stage12Coords.test.ts` | 5 | Stage 12 sub-pixel canvas <-> video coordinates :: maps the frame corners onto the letterboxed rectangle (object-fit: con ; round-trips arbitrary sub-pixel point |
| `src/stage12Diagnostics.test.tsx` | 4 | Stage 12 Diagnostics & Environment-Health Suite UI :: renders Live GPU/VRAM HUD with memory pressure gauge and purge button ; renders Standing Environment-Health Ca |
| `src/stage12TargetLock.test.tsx` | 7 | Stage 12 pose pill and axes :: formats one decimal with an explicit sign and never -0.0 ; points the facing axis toward image right for + yaw |
| `src/stage13ModelHub.test.tsx` | 5 | Stage 6 local model hub :: connects engine selection to the shared job preference and blocks unav ; shows the active production engine an |
| `src/stage13Resilience.test.tsx` | 27 | WorkstationErrorBoundary :: shows a recoverable message instead of a blank page and retries on dem ; clears itself when a reset key change |
| `src/stage18Resources.test.tsx` | 8 | resource ledger accounting :: counts object URLs created, revoked and live, and ignores revoking an  ; counts live WebSockets by path and fo |
| `src/stage19.test.ts` | 28 | error guidance :: maps the real backend codes to a title, a cause and at least one step ; falls back by code family, by a code n |
| `src/stage20Shell.test.tsx` | 12 | deep-link routing :: maps the five Stage 20 links onto the existing screens and keeps every ; keeps the deep link in the address ba |
| `src/stage23.test.ts` | 5 | mask frame ranges :: lets a painted range win over the whole-clip mask only inside the rang ; keeps ranges sorted and never overlap |
| `src/stage23Ui.test.tsx` | 13 | Stage 23 canvas inspection modes :: offers all six modes: wipe, crossfade/blend, thermal diff, loupe and t ; keeps the 3D mesh overlay usable befo |
| `src/stage24.test.tsx` | 29 | timeline dock maths :: places times on the track and never yields NaN% ; measures the ruler in m:ss.f and h:mm:ss |
| `src/stage5Scrub.test.tsx` | 21 | FrameScrubber: latest frame wins, one request in flight :: never has two requests in flight and always ends on the newest wanted  ; asks for faces only once the playhead |
| `src/stage7.test.tsx` | 10 | Pre-flight linter one-click fixes :: runs â€œConvert & Optimize Nowâ€, reports the result and re-lints ; surfaces a conversion that did not complete a |
| `src/stage7Studio.test.tsx` | 16 | FaceSet drag payload :: accepts only a well-formed FaceSet drag and ignores files or text ; bins stored poses into the yaw x pitch ind |
| `src/stage8.test.tsx` | 2 | preserves discovery rank, real embeddings, merges and cosine ordering ; uses the same cubic Bezier as the back |
| `src/stage8Studio.test.tsx` | 2 | loads durable bank assignments and reloads saved archive cells after f ; sends CodeFormer and the identity pai |
| `src/stage8Workflow.test.tsx` | 3 | uses the server queue even when stale completed jobs exist in localSto ; offers legal controls and exports onl |
| `src/streamOverlap.test.tsx` | 10 | lane model :: lays every stage on one shared time axis in execution order with the r ; draws nothing until a frame has been  |
| `src/studioTools.test.ts` | 11 | physical camera preview maths :: is the field of view of a 36 x 24 mm sensor ; clamps to the 18-135 mm range the matcher supports and survives  |
| `src/swapScope.test.tsx` | 10 | swap scope helpers keep untouched requests byte-identical :: lists the three scopes with the full craniofacial lock first and as th ; picking the default removes both keys |
| `src/themesAndFacesetAndTimeline.test.tsx` | 11 | Themes & Styles Engine (37 Studio Themes) :: contains all 37 studio themes across dark, AMOLED, and light families ; resolves themeByName correctly with fa |
| `src/timelineInteraction.test.tsx` | 4 | timeline navigation interactions :: zooms around the pointer without moving the point under it ; selects the actual frame shown in a filmstrip car |
| `src/timelineWorkstation.test.tsx` | 11 | Timeline Utility Functions :: formats timecode into SMPTE HH:MM:SS:FF accurately ; parses timecode strings and frame numbers into seconds |
| `src/uiLayouts.test.tsx` | 33 | foundation utilities :: cn joins strings, arrays and condition maps and drops falsy values ; clamp and decimalsOf |
| `src/viewportTools.test.ts` | 17 | buffer views :: lists the five requested buffers plus the default shaded view, each wi ; reproduces a clamped-cosine key light |
| `src/workflowPanel.test.tsx` | 7 | RenderFooter :: shows live progress and fps, and a GPU pill once telemetry arrives ; start is gated by its reason, the frame b |
| `src/workstation.test.tsx` | 9 | capability and config gates :: fails closed before capability load and for unknown features/options ; renders schema controls with unavailabl |
| `src/workstation3dPayload.test.ts` | 2 | opt-in 3D workstation controls in the live preview payload :: omits the hairline offset and ambient fill until they are touched, so  ; sends the hairline offset once set, a |
| `src/workstationCanvas.test.tsx` | 5 | WorkstationCanvas Comparison Modes :: renders 4 preview modes with active wipe controls ; renders diff map layer when mode is diff |
| `src/workstationControls3D.test.tsx` | 15 | skin tone vs ambient adaptation :: drives only the colour-transfer weight ; says so when colour matching is bypassed, and never claims to be the  |

## 8. `app/roop_3d_ui/` â€” every non-test file (one row each)

Columns: lines Â· key exports Â· backend paths referenced in the file (static regex; `/api/v1` prefix is added by `api()` at `api.ts:14`) Â· purpose (first doc-comment, truncated).


#### (ui root: config / entry)  (9 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `browserPaths.ts` | 28 | appRoot, ownedBrowserPath, localBrowserTemp |  | Keep browser profiles, downloads and test evidence inside this checkout; reject linked ancestors. |
| `eslint.config.js` | 52 |  |  | Syntax-level lint for .js/.jsx/.ts/.tsx. typescript-eslint cannot be used: TypeScript 7 ships no compiler JS A |
| `index.html` | 2 |  |  |  |
| `package-lock.json` | â€“ | â€“ | â€“ | asset/lock/binary (230742 B) |
| `package.json` | 54 |  |  |  |
| `playwright.config.ts` | 26 |  |  WS |  |
| `playwright.stage8.config.ts` | 14 |  |  |  |
| `tsconfig.json` | 4 |  |  |  |
| `vite.config.ts` | 9 |  | /api |  |

#### e2e  (3 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `facesetBuilderHarness.tsx` | 4 |  |  |  |
| `mockApi.ts` | 155 | ID, TARGET, CAPABILITIES |  /config /facesets /jobs WS | Two faces of the mocked 960x540 frame, in native pixels: what GET /media/frame?detect_faces=true returns. |
| `slider.ts` | 22 | sliderValue, setSlider |  | Current value of a PrecisionRangeSlider thumb (`role="slider"`), read from its ARIA state: they are not <input |

#### harness  (6 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `gl.html` | 12 |  |  |  |
| `gl.tsx` | 47 |  |  | Left half pure red, right half pure green; optional exact-pixel patch so differences are known to the unit. |
| `leak.html` | 12 |  |  |  |
| `leak.tsx` | 47 |  |  | Stage 18 harness (used only by e2e/resourceLeaks.e2e.ts): mounts and unmounts the real WebGL2 comparison canva |
| `livestream.html` | 12 |  |  |  |
| `livestream.tsx` | 14 |  |  | Harness for e2e/liveStream.e2e.ts: mounts the real LiveStreamCanvas and counts how many times React renders it |

#### mock-server  (2 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `dev.ts` | 18 |  | /api /ws | `npm run dev:mock`: the mock backend plus Vite HMR on 5174, proxied to it. No GPU, models or Python needed. |
| `server.ts` | 86 | MOCK_HEADER, mockHardware, mockFrame |  /health /system/hardware /system/telemetry WS |  |

#### src/ (top level)  (27 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `App.tsx` | 832 | App | /config /faceset/3d-state /faceset/reconstruct-3d /facesets | per-session only |
| `SettingsInspector.tsx` | 297 | SettingsInspector |  |  |
| `ThreeViewport.tsx` | 57 | ThreeViewport, PoseCoverageInspector |  |  |
| `VideoViewport.tsx` | 1161 | VideoViewport, formatTime | /media/preview-frame /media/detect-frame-faces | Snapshot slots keep a self-contained image, so convert the preview PNG before its object URL is revoked. |
| `api.ts` | 123 | ApiError, api, post | /api/preview  /media/content /target/add-angle-sample | An error from the local service. It is still a plain `Error` (same `message` as before), but keeps the machine |
| `contracts.ts` | 210 | Schema, Config, Feature |  | Render preset (see /media/export-profiles). Omitted = the standard MP4 default. |
| `events.ts` | 35 | EventConnection |  WS |  |
| `frameScrub.ts` | 309 | FrameFace, FrameMeta, MediaInfo | /media/filmstrip /media/frame /media/info | Left to right rank in the frame: the `faceRank` face extraction takes. |
| `index.css` | 70 |  |  | deep carbon surfaces |
| `livePreview.ts` | 184 | LIVE_DEBOUNCE_MS, JOB_START_EVENT, RENDER_FRAME_EVENT | /api/preview | Slider drags coalesce into one request per pause of this length. |
| `liveStream.ts` | 168 | HEADER_BYTES, PROTOCOL_VERSION, FrameSource | /api/preview /ws/v1/preview/stream WS | Wall latency of pipeline stages 1 (detect/pose), 2 (swap/mesh) and 3 (enhance/blend) for this frame, ms. |
| `main.tsx` | 27 |  |  WS |  |
| `motion.tsx` | 71 | spring, EASE_SETTLE, fadeUp |  | Shared motion vocabulary (ported from Roop Ultimate's react-ui): one set of springs and entrance variants so |
| `previewFrameCache.ts` | 35 | PreviewFrameCache |  | Short-lived, CPU-only PNG cache. Never retains object URLs or decoded GPU textures. |
| `previewPreferences.ts` | 15 | ViewportPreferences, DEFAULT_VIEWPORT, ProjectPreview |  |  |
| `runtimeSettings.ts` | 2 | RUNTIME_SETTINGS_CHANGED |  | A persisted runtime change invalidates both detection and rendered frame caches. |
| `sessionSnapshot.ts` | 86 | SESSION_KEY, SessionSnapshot, parseSession |  | Validate untrusted storage content field by field; anything malformed falls back to an empty value. |
| `sliderTestKit.ts` | 20 | thumb, setSlider, sliderValue |  | Test helpers for PrecisionRangeSlider (role="slider" thumbs, not <input type="range">). They drive it the way  |
| `stage19.css` | 196 |  |  | ---- accessibility primitives ------------------------------------------------------------------------------ |
| `stage23.css` | 21 |  |  | Stage 23: multi-person actors, frame-range mask scope, 3D mesh canvas mode. Layers over the earlier sheets. |
| `stage24.css` | 151 |  |  | Stage 24: liquid glass HUDs, drag-to-reorder lists, Embla filmstrip dock, persistent queue cards.    Plain CSS |
| `state.ts` | 28 | featureReason, optionReason, jobActions |  |  |
| `styles.css` | 2921 |  |  | reduced motion is handled in stage19.css (system setting + the user's own override) |
| `themeVars.ts` | 159 | hexToRgb, rgbToHex, mix |  | â”€â”€ Custom theme derivation matching roop-ultimate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ |
| `themes.ts` | 95 | ThemeDefinition, THEMES, THEME_CLASSES |  | measurement is best-effort; the theme still applies |
| `useFaceSelectionStore.ts` | 557 | FaceSelectionStoreState, cosineSimilarity, cosineDistance |  | Keeps a stored face's pose current with the frame it is on screen in (the 3D viewer's Live Pose reads it). |
| `workstation3d.css` | 136 |  |  | ---- Pipeline node flow bar ---------------------------------------------------------------------------------- |

#### src/__fixtures__  (1 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `fixture_head.glb` | â€“ | â€“ | â€“ | asset/lock/binary (2100 B) |

#### src/components/(top)  (3 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `LiveStreamCanvas.tsx` | 74 | LIVE_STALE_MS, LiveStreamCanvas |  WS | The canvas hides itself again if no frame has arrived for this long (render finished, or the stream dropped). |
| `TargetFaceCard.jsx` | 2 |  |  |  |
| `ThemeGallery.tsx` | 89 | ThemeCardProps, ThemeCard, ThemeGalleryProps |  | One swatch card. Colours come from the theme itself (inline), everything else from shell.css `.theme-card `. |

#### src/components/canvas  (15 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `CanvasHuds.tsx` | 78 | ViewportTelemetryPillProps, ViewportTelemetryPill, BeforeAfterLoupeHudProps |  | Floating Stage 24 HUDs for the full-bleed canvas, drawn on liquid glass. All of them are optional extras over  |
| `CompositionGridOverlay.jsx` | 85 | CompositionGridOverlay |  | Vertical grid lines |
| `FaceContextMenu.tsx` | 140 | FaceDestination, TargetAction, TargetActionExtra |  | Stage 12 target actions, shown when the host provides `onTargetAction`. |
| `FaceOverlayCanvas.jsx` | 874 | FACE_ROLES, FaceOverlayCanvas, signed1 |  | Quick target face selector strip on top of preview box |
| `MaskBrushOverlay.jsx` | 589 | MaskBrushOverlay |  | Visible tinted painting canvas |
| `WebGLViewport.jsx` | 195 | WebGLViewport |  | Native-resolution texture: the image is uploaded at its own size, only downscaled if the GPU cannot hold it. |
| `WorkstationCanvas.jsx` | 409 | MODES, clientToMedia, diffHeatmap |  | Pointer Ã¢â€ â€™ media pixel coordinates with sub-pixel precision. `stage` is the on-screen rectangle (client px) t |
| `faceFlags.ts` | 25 | Pose6, faceFlags, faceKey |  | Frozen 6DoF poses of locked faces (standard frame), by face key. The overlay holds the wireframe / gimbal / ga |
| `headMesh.js` | 258 | MAX_DRAWN_EDGES, prepareHeadMesh, poseRotation |  | Per source coordinate frame: the sign applied to (x, y, z) to reach the viewer frame (see PoseGeometryViewer F |
| `loupeCrop.ts` | 22 | Stage, Pointer, CropRect |  | Pure geometry for the before / after loupe HUD. |
| `poseHud.js` | 57 | facePose, LANDMARK_COLORS, drawPoseHud |  | Standard-frame pose of a detected face (prefers the server's `pose3d`; otherwise converts the legacy `pose`). |
| `useFaceExtraction.ts` | 67 | Toast, useFaceExtraction |  | Active source FaceSet id: the FaceSet "Add to .fsz Faceset" extends. |
| `useHeadMesh.ts` | 77 | PreparedHead, HeadMeshState, clearHeadMeshCache | /faceset/3d-state | The fitted anatomical face, an explicitly labelled orientation proxy, or nothing. |
| `useTargetLock.ts` | 188 | useTargetLock |  | Local media path of the target video; changing it drops the locks (locks belong to one video). |
| `webglModel.js` | 252 | MODE_IDS, LOUPE_ZOOM, LOUPE_ZOOMS |  | The pore magnifier runs at 3.5x or 4x (Stage 23); 3.5x stays the default so earlier behaviour is unchanged. |

#### src/components/common  (4 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `ToastHost.tsx` | 21 | ToastHost |  | Renders the app-wide toasts. Errors are `role="alert"` (announced at once); the rest are polite status message |
| `VirtualList.tsx` | 147 | WindowInput, WindowResult, windowRange |  | Which rows [first, last] must be mounted for the visible band, where the first mounted row starts, and the ful |
| `WorkstationErrorBoundary.tsx` | 55 | BoundaryContext, WorkstationErrorBoundaryProps, WorkstationErrorBoundary |  | Short name shown in the message and the log (`3D viewport`, `Workstation`). |
| `toastHost.css` | 8 |  |  |  |

#### src/components/controls  (6 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `Hotkeys.tsx` | 116 | DEFAULT_KEYS, KEY_LABELS, KEY_GROUPS |  | Display grouping for the remapper UI; every action in DEFAULT_KEYS appears in exactly one group. |
| `Modal.tsx` | 19 | Modal |  |  |
| `NumericInput.tsx` | 66 | evaluateExpression, constrain, scrubDelta |  |  |
| `SettingControl.tsx` | 47 | SettingControl |  | Why the control is unavailable right now (shown in the help and as the disabled tooltip). |
| `settingsRegistry.ts` | 67 | PanelId, SettingId, SettingSpec |  | Range that usually looks right, shown in the help and drawn on the slider's help. |
| `useDialogA11y.ts` | 49 | focusableIn, useDialogA11y |  | Visible, enabled, tabbable controls inside `root`, in DOM order. Exported so tests can assert the trap's contr |

#### src/components/diagnostics  (10 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `BottleneckFlamegraph.jsx` | 423 | PIPELINE_STAGES, DEFAULT_TEMPORAL_SETTINGS, BottleneckFlamegraph |  | Top Metrics Row |
| `PipelineSaturationPanel.jsx` | 80 | usePipelineMetrics, PipelineSaturationPanel | /system/pipeline | Polls the server while mounted (the HUD popover is open); nothing is requested when it is closed. |
| `PreFlightLinterModal.jsx` | 239 | PreFlightLinterModal | /system/preflight-lint |  |
| `RuntimeReportPanel.tsx` | 140 | RuntimeReportPanel | /system/runtime-report | Mirror of the Pinokio terminal's 14-section runtime report, read from the same collectors (`GET /system/runtim |
| `StreamOverlapPanel.jsx` | 88 | StreamOverlapPanel |  | Per-frame stage lanes on one time axis (so overlap is visible), the PCIe / copy picture and the frame ring buf |
| `batchLint.js` | 60 | SKIP_KEY, lintRequestFor, lintBatch | /system/preflight-lint | Pre-flight linting for batch submission: one request per distinct target video, advisory unless a check FAILS. |
| `lanesModel.js` | 83 | LANE_ORDER, LANES, buildLaneModel |  | Pure shaping of the /api/v1/system/pipeline payload into the stream-overlap view. |
| `pipelineMetrics.js` | 54 | pickPipeline, summarizePipeline, saturationVerdict |  | Pure helpers for the saturation panel: shape the /api/v1/system/pipeline payload into display rows. |
| `runtimeReport.css` | 31 |  |  | Narrow inspector: label above value, so long values (paths, provider chains) wrap instead of overflowing. |
| `stageTimings.js` | 56 | FLAME_STAGES, resolveStageTimings |  | Seven measured pipeline stages shown by the bottleneck flamegraph. Mirrors roop_3d/core/stage_timing.py. |

#### src/components/export  (3 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `ExportPresetBar.tsx` | 59 | ExportPresetBar |  |  |
| `VfxExportPanel.jsx` | 95 | DEFAULT_EXPORT, FinishingControls, VfxExportPanel |  | Offered by the interface but rejected by the job schema (jobs/schema.py accepts alpha, occluder, uv, normal, d |
| `exportOptions.ts` | 39 | ExportProfileInfo, normalizeExport, useExportProfiles | /media/export-profiles | Drop every Stage 15 field that equals the server default, so "nothing chosen" serialises exactly like the orig |

#### src/components/facemgr  (10 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `BuilderCoverageSphere.tsx` | 23 | BuilderCoverageSphere |  | The same 12 x 8 frontal hemisphere used by Stage 3, with selectable missing bins. |
| `FaceManager.jsx` | 692 | extractFacesFromInspection, FaceManager | /facesets/build | The 3x3 pose matrix as ARIA rows of three cells, so `role=grid` has the `row` children it requires. |
| `FacesetBuilder.css` | 3 |  |  |  |
| `FacesetBuilder.tsx` | 117 | FacesetBuilder | /faceset/extract-from-media /faceset/save-fsz /faceset/synthesize-36 /system/capabilities |  |
| `FacesetInspector3D.tsx` | 90 | FacesetInspector3D | /faceset/3d-state /faceset/reconstruct-3d | Fetch `path` (null = nothing); a late answer for an earlier FaceSet or head model is dropped. `revision` force |
| `FacesetLibrary.tsx` | 562 | FacesetLibraryProps, FacesetLibrary | /facesets/open-folder /facesets/rebuild-thumbs | `dropdown` (default) is the original compact picker. `list` (Stage 7 studio identity panel) keeps every FaceSe |
| `IdentityLockPanel.tsx` | 124 | clock, rangeSummary, withheldSummary | /identity/locks | Seconds -> m:ss without importing the viewport (keeps this panel dependency-free). |
| `PhotoSynthesisAction.tsx` | 39 | PhotoSynthesisAction | /faceset/synthesize-photos | Direct action for the measured photos already imported in #/facemgr. |
| `faceBank.js` | 23 | rankClusters, poseBin, poseGrid |  | Keep discovery ranks and original biometric measurements when manually merging tracks. |
| `fszBuilder.js` | 447 | crc32, buildZip, POSE_BUCKETS |  | Pure JavaScript utility for .fsz FaceSet archive creation, DBSCAN ArcFace clustering, and 5-bucket pose classi |

#### src/components/faceswap  (2 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `PipelinePill.jsx` | 70 | TELEMETRY_PATH, PipelinePill | /ws/telemetry |  |
| `pillModel.js` | 75 | STEPS, isPipelineFrame, elapsedSeconds | /ws/telemetry | View model for the 3-step run-bar pill. Pure functions of the server's `pipeline.status` frames (`/ws/telemetr |

#### src/components/guide  (14 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `AccessibilityControls.tsx` | 34 | ZoomControl, AccessibilityControls |  | Display & accessibility preferences. "Reduce motion" stops the interface animations and the 3D viewers' idle r |
| `DialogFrame.tsx` | 25 | DialogFrame |  | The one accessible modal shell: labelled `role="dialog"`, focus moved in and trapped, Escape closes, focus res |
| `DialogPanel.tsx` | 14 | DialogPanel |  | The dialog surface only (no backdrop), for dialogs that already render their own backdrop: labelled, modal, fo |
| `EngineExplainer.tsx` | 22 | EngineExplainer |  | Explains the engine list in the Inspector: which engine Roop 3D recommends here and why, what the chosen engin |
| `GuideDialog.tsx` | 124 | GUIDE_TABS, GuideTab, GuideDialog |  |  |
| `HardwareExplainer.tsx` | 27 | HardwareExplainer |  | What the active hardware profile means for this computer, in words: what Roop 3D will use, what that changes,  |
| `NextStepBar.tsx` | 24 | NextStepBar |  |  |
| `RecoveryNotice.tsx` | 25 | RecoveryNotice |  | The error banner body: what failed, the first thing to try, and (expandable) the cause and every step. Unknown |
| `WelcomeDialog.tsx` | 31 | WelcomeDialog |  | Shown once, on the first launch, over the Home screen. It tells a new person in four sentences what the app is |
| `WorkflowChecklist.tsx` | 33 | WorkflowChecklist |  | The workflow as a live checklist. Every state is derived from real application state (see workflowState.ts), s |
| `coverageGuide.ts` | 41 | YAW_COLUMNS, CoverageReading, describeCoverage |  | One or two sentences a person can act on. |
| `errorGuide.ts` | 105 | ErrorGuide, ERROR_GUIDES, isOfflineError |  | Section of docs/COMMON_PROBLEMS.md that expands on this. |
| `helpContent.ts` | 185 | WorkflowStepId, WorkflowStep, WORKFLOW_STEPS |  | .md page; the Stage 19 tests pin the numbers that appear in the user-facing text. |
| `workflowState.ts` | 53 | StepState, WorkflowInputs, WorkflowProgress |  | A project has been saved (or opened) for this work. |

#### src/components/hud  (3 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `EngineStatusBar.tsx` | 89 | EngineStatusBar |  |  |
| `PipelinePill.jsx` | 328 | PIPELINE_STAGES, resolveProvider, computeBottleneck | /ws/telemetry WS | Standard 5-stage facial transformation pipeline stages. |
| `engineStatus.css` | 23 |  |  | Tablet widths: the bar would push the preview below the fold. It stays mounted (polling, failure toasts and th |

#### src/components/modelhub  (11 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `ActiveModelBadge.tsx` | 47 | ActiveModelBadge |  | Model ids the registry uses for the rack's enhancer ids, so "selected" and "loaded" compare like with like. |
| `EngineModelRack.tsx` | 229 | RackProps, EngineModelRack | /models/compile_trt | Current values of the parameters the rack edits (the workspace parameters, or the Batch defaults). |
| `ModelHubPanel.tsx` | 216 | ModelHubPanel | /models/downloads /models/hub |  |
| `ModelManager.tsx` | 129 | ModelManager | /models/compile_trt | Settings â†’ Models & Engines Manager: rescan, offline import, integrity check, and per-model actions. Every reg |
| `ModelManagerModal.tsx` | 154 | progressView, ModelManagerModal | /ws/v1/pipeline/status | What the progress cell says for one row: live transfer first, then a partial download left on disk, then nothi |
| `ModelSelect.tsx` | 188 | RackOption, StatusPill, ModelSelect |  | One choice in a rack selector. `modelId` is the catalog artifact it needs (defaults to `id`); `null` means it  |
| `PersonModelPicker.tsx` | 17 | PersonModelPicker |  | Pins a swapper and an enhancer to one detected person. "Scene default" = no override. |
| `RequiredModelsBanner.tsx` | 69 | BANNER_DISMISS_KEY, RequiredModelsBanner |  | not remembered |
| `modelHub.css` | 60 |  |  | Model Hub modal + required-models banner (Stage 13). Tokens only: works in every theme. |
| `modelRack.css` | 48 |  |  |  |
| `openModelHub.ts` | 3 | OPEN_MODEL_HUB, openModelHub |  | Any screen can ask the app shell to open the Model Hub without prop drilling. |

#### src/components/output  (2 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `OutputComparePlayer.tsx` | 129 | DRIFT_SECONDS, afterUrl, beforeUrl | /media/content | Beyond this many seconds apart the follower is re-seeked to the master (about three frames at 25 fps). |
| `outputCompare.css` | 26 |  |  | Before / After output player. Tokens come from the app theme (--line, --raised, --accent, --text, --muted, --d |

#### src/components/queue  (4 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `BackendQueuePanel.tsx` | 113 | BackendQueuePanelProps, BackendQueuePanel |  | Poll only while the screen is showing. |
| `QueueDrawer.jsx` | 429 | QUEUE_STATES, normalizeJobState, QueueDrawer | /media/content /jobs /media/concat | Normalized queue state definitions. |
| `ResumeNote.jsx` | 20 | ResumeNote |  | Stage 9: shows where a re-run of a stopped job continues from (the last verified finished segment), if one exi |
| `backendQueue.ts` | 60 | LIFECYCLE_STATES, LifecycleState, LIFECYCLE_LABEL | /queue | Types and rules for the backend-owned persistent queue (`/api/v1/queue`). Pure: no React, no fetch. |

#### src/components/studio  (17 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `ActorPanel.tsx` | 104 | Actor, ActorReport, MIXED |  | `GET /actors/{inspection_id}` (roop_3d/server/actor_routes.py): distinct people in the target, most prominent  |
| `FaceExpressionEditor.tsx` | 89 | FACE_EDIT_CONTROLS, FaceExpressionEditor |  |  |
| `FaceMapping.tsx` | 41 | FaceMappingProps, FaceMapping |  | track id -> FaceSet id (the face-selection store's assignments). |
| `FaceSetDetail.tsx` | 251 | YAW_BINS, PITCH_BINS, coverageGrid |  | Yaw columns (image left â†’ right) and pitch rows used by the pose coverage indicator. Standard frame: + pitch = |
| `HairNeckProtection.tsx` | 73 | HairNeckProtectionProps, HairNeckProtection |  | The enhancer that will really run; the pore slider only has an effect after UltraMax. |
| `HeadModelBackbone.tsx` | 61 | BackboneRow, BackboneReport, BACKBONE_KEY |  | the choice just is not remembered |
| `PanelBoundary.tsx` | 18 | PanelBoundary |  | Keeps one studio panel's failure local: the panel shows the error and a Retry button (which remounts it), whil |
| `RenderFooter.tsx` | 54 | RenderFooterProps, RenderFooter |  | The job for the loaded target, if any (drives the progress bar). |
| `SpatialInspector.tsx` | 159 | SpatialTab, SpatialInspectorProps, SpatialInspector |  | Tab Panels: keep mounted to prevent WebGL context destruction |
| `StudioInspector.tsx` | 271 | StudioInspectorProps, ENHANCERS, activeEnhancer | /models/hub | Engines from the Stage 6 model hub; selection is the same store the Model Hub screen writes. |
| `SwapScopeSelector.tsx` | 39 | SwapScopeSelectorProps, SwapScopeSelector |  | Which part of the target's head the source geometry replaces. Real compositor option (`CompositeOptions.swap_s |
| `TargetPanel.tsx` | 117 | trackLabel, formatClock, FrameNavigator |  | Frame-accurate navigator. The readout follows the live playhead store, so playback ticks re-render only this l |
| `TrackLanes.tsx` | 106 | laneColor, trackSpans, renderFraction |  | Visible frame spans of a track: the identity sheet's segments when available, else first â†’ last frame. |
| `WorkstationControls3D.tsx` | 253 | LIP_LOCK_KEY, SCENE_ADAPT_KEY, SMOOTH_CUTOFF_LIGHT |  | not remembered |
| `arkit52.ts` | 23 | ARKIT52_NAMES, arkitLabel, ARKIT52_GROUPS |  | Public ARKit blendshape order; values remain sparse in per-face overrides. |
| `protection.ts` | 46 | PROTECTION_DEFAULTS, ProtectionKey, PROTECTION_KEYS |  | Engine defaults of the hair / neck / ear controls (= `CompositeOptions`). A control set back to its default is |
| `swapScope.ts` | 42 | SwapScope, DEFAULT_SWAP_SCOPE, ADAPTIVE_START_BLEND |  | Share of the source skull when the user first picks the adaptive blend (the engine default of 1 would equal th |

#### src/components/three  (13 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `AngularCoverageRadar.tsx` | 92 | AngularCoverageRadarProps, AngularCoverageRadar |  | The FaceSet coverage payload (`views` / `observed_pose_points`); null while none is loaded. |
| `CoverageRadarPanel.tsx` | 17 | CoverageRadarPanel |  | The coverage radar for the active source FaceSet, with the selected target's poses drawn on it (red where the  |
| `FlameMeshViewer.tsx` | 226 | FlameMeshViewerProps, FlameMeshViewer |  |  |
| `GuardedCanvas.tsx` | 113 | ContextPhase, RESTORE_TIMEOUT_MS, MAX_REBUILDS |  | After a context loss the browser has this long to hand the context back before the canvas is rebuilt. |
| `IdentityMorphPanel.tsx` | 59 | IdentityMorphPanelProps, INFLUENCE_MODES, IdentityMorphPanel |  | Whether the 3D viewer shows the target-geometry ghost and the hairline ring while these controls are edited. |
| `LightProbeSphere.tsx` | 217 | LightProbeSphere |  | The sphere is shaded by the controls themselves: the highlight sits where the key light is, its strength follo |
| `PoseGeometryViewer.jsx` | 1151 | generateFibonacciSphere, projectViewsToLattice, generatePoseGrid |  | @typedef {{ index: number; position: [number, number, number]; normal: THREE.Vector3; yawDeg: number; pitchDeg |
| `ViewportGizmos.jsx` | 157 | JawGizmo, GazeGizmo, LightGizmo |  | Invisible, larger pick target so a handle is easy to grab at any zoom. |
| `WebGLFallbackInspector.tsx` | 78 | frontProjection, WebGLFallbackInspector |  | Front-view outline of a mesh as SVG path data, in a 0..100 box (y down). Pure so it can be tested without a DO |
| `studioTools.ts` | 199 | SENSOR_WIDTH_MM, SENSOR_HEIGHT_MM, FOCAL_RANGE_MM |  | Full-frame sensor: the reference the 18-135 mm range of `PhysicalCameraMatcher` is expressed against. |
| `useStudioEnvironments.ts` | 35 | EnvironmentsState, loadStudioEnvironments, resetStudioEnvironmentCache | /rendering/studio-environments | The five built-in CC0 studio lighting probes (`GET /rendering/studio-environments`), fetched once per page loa |
| `viewerRigs.ts` | 25 | ViewerLighting, ViewerMorph, DEFAULT_AMBIENT_FILL |  | What the head viewer needs to preview the relighting controls. `null` while 3D relighting is off (the viewer k |
| `viewportTools.ts` | 292 | BufferView, BUFFER_VIEWS, isBufferView |  | Lambertian convolution weight per SH index (band 0: pi, band 1: 2pi/3, band 2: pi/4). |

#### src/components/timeline  (5 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `CurveEditorTrack.jsx` | 35 | CurveEditorTrack |  | @param {any} props |
| `TimelineDock.tsx` | 170 | TimelineDockProps, TimelineDock |  | Without it the In / Out handles are read-only (the Editor screen only previews). |
| `TimelineWorkstation.jsx` | 1306 | formatTimecode, parseFps, parseTimecodeString | /media/preview-frame | Format seconds into standard SMPTE timecode (HH:MM:SS:FF). @param {number} seconds @param {number} fps @return |
| `chapters.js` | 71 | MAX_CHAPTERS, sortChapters, addChapter |  | Chapter markers for the timeline: pure list operations plus tolerant (de)serialisation. |
| `dockMath.ts` | 83 | SPEEDS, Speed, DockFrame |  | Pure maths for the Stage 24 timeline dock (Embla filmstrip + trim rail). No React, no DOM: everything here is  |

#### src/components/tour  (2 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `QuickTour.tsx` | 89 | QuickTour |  | Small diagram per step. Decorative: the same information is in the text. |
| `tourSteps.ts` | 62 | TourStep, TOUR_STORAGE_KEY, TOUR_STEPS |  | CSS selector of the element the coach mark highlights; the card centres itself when it is not on screen. |

#### src/components/tray  (5 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `AngleCaptureHud.tsx` | 92 | HudState, hudSummary, cropVersion |  | What the capture HUD says, derived only from the server's coverage and scan fields (nothing is rounded up). |
| `FaceSelectionTray.jsx` | 129 | FaceSelectionTray |  |  |
| `FrameFaceStrip.tsx` | 248 | FrameFaceStripProps, FrameFaceStrip |  | Crops each detected face out of the frame on screen. Canvas-less environments (tests) get empty thumbnails. |
| `TargetFaceCard.jsx` | 137 | TargetFaceCard |  | On-screen badge indicator |
| `faceBank.css` | 68 |  |  | Target Face Bank: persistent FACE n identities, lock toggle, live angle-capture HUD, hover inspector. |

#### src/components/ui-layouts  (14 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `AnimatedBeam.tsx` | 175 | BeamDirection, AnimatedBeamProps, BeamGeometry |  | Vertical bow of the curve in px; negative bows the other way. 0 is a straight line. |
| `AnimatedTabs.tsx` | 90 | TabItem, AnimatedTabsProps, AnimatedTabs |  |  |
| `ClipPathSplitReveal.tsx` | 146 | ClipPathSplitRevealProps, clipLeftOf, loupeOffset |  | Initial divider position, 0..100 (percent from the left). |
| `DragItems.tsx` | 125 | moveItem, sameOrder, DragRowContext |  | The grip and the two move buttons. Place them wherever the row wants them. |
| `EmblaCarousel.tsx` | 75 | EmblaCarouselProps, DEFAULT_OPTIONS, EmblaCarousel |  | Thin, accessible wrapper over Embla (ui-layouts "Carousel"): draggable with inertia, snap-free scrolling, prev |
| `LiquidGlassCard.tsx` | 108 | BlurIntensity, EffectIntensity, BLUR_PX |  | Unknown values fall back to the default instead of producing `NaNpx`. |
| `MotionDrawer.tsx` | 71 | DrawerSide, MotionDrawerProps, MotionDrawer |  | CSS width (left/right) or height (bottom). |
| `MotionNumber.tsx` | 78 | MotionNumberProps, formatNumber, MotionNumber |  | Fraction digits. Default 0. |
| `MultiAccordion.tsx` | 92 | AccordionItem, MultiAccordionProps, MultiAccordion |  | Small right-aligned status text, e.g. a value summary or a count. |
| `PrecisionRangeSlider.tsx` | 209 | SingleSliderProps, Sides, SplitSliderProps |  | Accessible name when it should be longer than the visible label. Defaults to `label`. |
| `SpotlightCard.tsx` | 81 | SpotlightCardProps, proximityOpacity, SpotlightCard |  | Distance in px from the card edge at which the glow starts to fade in. 0 disables proximity tracking. |
| `TimelineAnimation.tsx` | 56 | ViewportOptions, DEFAULT_VIEWPORT, STEP |  | Timeline animation (ui-layouts "Timeline Animation"): children reveal one after another as they scroll into vi |
| `index.ts` | 12 |  |  |  |
| `ui-layouts.css` | 341 |  |  | ui-layouts primitives. Plain CSS on the Obsidian tokens in src/index.css (this app has no Tailwind). Every col |

#### src/components/workstation  (14 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `ContactSheetModal.tsx` | 189 | LEVELS, AXES, defaultAxes |  | One sweepable dimension: a continuous setting (Low / Medium / High) or a categorical / stepped control. |
| `FloatingPreview.tsx` | 152 | FloatingPreviewMedia, PreviewMode, Geometry |  | â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Stage 26: floating Picture-in-Picture preview â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ |
| `HistoryDrawer.jsx` | 9 | HistoryDrawer |  | @param {any} props |
| `LiveStudio.tsx` | 69 | LiveStudio |  |  |
| `LiveSwapStatus.tsx` | 26 | describeStage, LiveSwapStatus |  | Small, non-modal status pill for the live swap preview (never covers the image centre). |
| `LiveTime.tsx` | 11 | LiveTime |  | Render-prop leaf that follows the live playhead only while `live` (playing). When paused it shows the committe |
| `PipelineNodeFlowBar.jsx` | 66 | PipelineNodeFlowBar |  |  |
| `SnapshotSlots.jsx` | 19 | SnapshotSlots, SnapshotViewport |  | @param {any} props |
| `WorkspaceLayouts.tsx` | 58 | PANEL_IDS, Layout, PRESETS |  | In-memory layout still works. |
| `WorkstationScreens.tsx` | 343 | WorkstationScreens | /health /system/purge-vram | Leaf subscriber: the only part of the Processing screen that re-renders when a hardware tick arrives. |
| `ZoomableImage.tsx` | 78 | MIN_ZOOM, MAX_ZOOM, clampPan |  | Keep the zoomed image covering the viewport: pan is clamped so no empty margin appears. |
| `pipeline.ts` | 123 | RestorerModel, restorerWeights, restorerAvailable |  | Weight file each enhancer needs. UltraMax is CodeFormer weights + Laplacian detail fusion, so it shares CodeFo |
| `renderModes.ts` | 35 | RenderModeId, PreviewTab, RenderModeInfo |  | The two viewer tabs: an estimated proxy and the fitted facial surface. |
| `sweepPatch.ts` | 37 | EnhancerChoice, ENHANCER_CHOICES, SWEEP_ENHANCER_BLEND |  | The three enhancer states the contact sheet compares. They are the Engine & Model Rack's own: "none" is a rest |

#### src/lib  (1 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `utils.ts` | 46 | ClassValue, cn, useMediaQuery |  | Foundation utilities for the ui-layouts primitives (src/components/ui-layouts). Dependency-free on purpose: th |

#### src/router  (1 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `AppRouter.jsx` | 73 | SCREENS, ROUTE_ALIASES, DRAWER_ROUTES |  | Stage 20 deep links. Each is a stable, shareable URL that opens the same screen as an existing route, so nothi |

#### src/screens  (5 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `BatchMatrixScreen.jsx` | 955 | BUILTIN_RECIPES, BatchMatrixScreen | /config /jobs /media/split-boundaries | Segment Splitting Controls Banner |
| `ExtrasScreen.jsx` | 32 | ExtrasScreen | /finishing/manifest |  |
| `HistoryScreen.jsx` | 467 | HistoryScreen | /projects/import | Notifications |
| `HomeScreen.tsx` | 89 | HomeProps, HomeScreen |  | Launch pad: where to start, what is running, what you made. Everything on this screen links somewhere useful. |
| `SettingsScreen.jsx` | 855 | SettingsScreen | /system/benchmark /system/check-compatibility /system/diagnostics-report /system/purge-vram | Leaf subscriber: the Settings screen itself never re-renders when the telemetry store first learns the GPU. |

#### src/shell  (9 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `CommandPalette.tsx` | 73 | Command, CommandPalette |  | Ctrl K launcher: jump to any screen or run a workspace action by name. Keyboard first, a11y-contract dialog. |
| `MagneticDock.tsx` | 76 | DockItemDef, DOCK_ITEMS, activeDockItem |  | Which dock item is current: a drawer wins over the screen behind it. |
| `Rail.tsx` | 47 | Rail |  | default below |
| `ShellDrawers.tsx` | 20 | ShellDrawers |  | Closing a drawer returns to the Studio, so the address bar never keeps pointing at something that is no longer |
| `TelemetryHeader.tsx` | 110 | GraphNode, GRAPH_NODES, nodeForStage | /api/pipeline/status /ws/telemetry | Which graph node a server stage name belongs to. The engine's stage vocabulary is detection / landmark / warp  |
| `TopBar.tsx` | 51 | TopBarProps, TopBar |  | Slim top bar: where you are, the project, run/hardware status and workspace tools. One row, no stacked strips. |
| `icons.tsx` | 27 | IconHome, IconSwap, IconLayers |  | Small stroke icon set (24px grid, currentColor). Decorative: the owning control carries the accessible name. |
| `shell.css` | 343 |  |  | Named dark themes only set accent-family tokens: derive the surface tokens from the accent so every theme read |
| `stage20.css` | 127 |  |  | Stage 20 shell: telemetry header, pipeline mini-map and the floating magnetic dock. Tokens come from src/index |

#### src/stores  (14 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `activeModels.ts` | 42 | ActiveModel, ActiveModels, activeModelStore | /models/active | One stage's loaded model, from `GET /models/active`. `null` = no session of that stage is in memory. |
| `atomicStore.ts` | 64 | AtomicStore, createAtomicStore, useStoreSelector | /ws/telemetry | Shallow-merge a patch (or replace with the updater's result). Listeners fire only when something changed. |
| `engineSelection.ts` | 6 | engineSelection, selectEngine, useSelectedEngine |  |  |
| `modelCatalog.ts` | 133 | CatalogModel, DownloadJob, ImportReport | /models /models/rescan | One row of `GET /api/models`: every registered artifact, installed or not. The UI never filters this list by ` |
| `modelManifest.ts` | 253 | LicenseBadge, HashStatus, ManifestModel | /models/download /models/manifest /models/verify-and-repair /ws/v1/pipeline/status | Client of the universal model manifest: `GET /models/manifest`, `POST /models/download`, `POST /models/verify- |
| `parameterCurves.js` | 47 | CURVE_PARAMETERS, CURVE_HINTS, evaluateCurve |  | @type {Record<string, {label: string; min: number; max: number; initial: number; field: string}>} |
| `pipelineStatus.ts` | 64 | SwitchFailure, PipelineStatus, pipelineStatusStore |  | `GET /api/pipeline/status`: what each stage has loaded, the precision those sessions really run, and failed sw |
| `pipelineStream.ts` | 170 | TransferStatus, ModelProgress, ACTIVE_TRANSFER | /ws/v1/pipeline/status | Status of one model transfer, as `/ws/v1/pipeline/status` reports it (`model_download_progress` frames). |
| `playbackRateStore.ts` | 19 | setPlaybackRate, getPlaybackRate, usePlaybackRate |  | One playback speed for the whole workstation (0.25x - 4x). The timeline's own speed buttons and the Stage 24 d |
| `playheadStore.ts` | 19 | playheadStore, setPlayhead, getPlayhead |  | Re-render only when `selector(time)` changes, e.g. `usePlayhead(t => Math.floor(t fps))` for frame-accurate UI |
| `rackSelection.ts` | 47 | PersonModels, RackState, rackStore |  | Models a single detected person is pinned to. Either field may be unset (= use the scene-wide choice). |
| `telemetryStore.ts` | 46 | LiveFrame, TelemetryState, telemetryStore | /ws/events /ws/telemetry | A `pipeline.status` / `job.progress` frame as streamed by `/ws/telemetry` (fields the UI actually reads). |
| `toasts.ts` | 28 | AppToast, toastStore, useToasts |  | App-wide transient messages. Errors stay until dismissed or 12 s pass; the newest five are kept. |
| `useHistoryStore.js` | 57 | executeCommand, replaceCurves, editCurves |  | Commands retain reversible edits, never execute application code from persisted text. |

#### src/utils  (21 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `coverageRadar.ts` | 152 | YAW_MIN, YAW_BINS, PoseDeg |  | Samples per cell. The server's own `bin_index` wins when present; otherwise each view is binned by its yaw/pit |
| `faceIdentity.ts` | 32 | FACE_COLORS, faceColor, faceTag |  | Distinct on dark video and for the common colour-vision deficiencies; text on every fill is near-black. |
| `facesetDrag.ts` | 39 | FACESET_MIME, startFacesetDrag, carriesFaceset |  | Drag payload for "drag a FaceSet onto a target track". A private MIME type, so random text/file drops never as |
| `frameFaces.ts` | 100 | MatchedFrameFace, MATCH_IOU, iou | /media/detect-frame-faces | The stored face's id when matched, else `frame:<frame>:<rank>`. |
| `glbMesh.ts` | 183 | GlbError, ParsedGlb, parseGlb |  | Names of the morph targets (the six semantic sliders when `shape_keys=true`). |
| `globalErrors.ts` | 32 | isBenignRejection, installGlobalErrorHandlers |  | A rejection that only means "this request was cancelled" (navigation, a newer request, unmount). Never worth s |
| `kelvinToRgb.ts` | 46 | kelvinToRgb |  | Tanner Helland's algorithm to convert color temperature in Kelvin (1000K to 12000K) to standard sRGB color val |
| `letterbox.js` | 88 | calculateLetterbox, mapBoxToPixels, mapPointToPixels |  | Sub-pixel letterbox geometry shared by the face overlay, the comparison canvas and hit-testing. |
| `maskRanges.ts` | 59 | MaskKeyframe, keyframeAt, maskAtFrame |  | The range mask covering `frame`, if any. |
| `meshPassExporter.js` | 67 | renderAlignedMeshPasses, passPFM, downloadPass |  | Render the actual frame projection, never the orbit/inspection camera. Background alpha=0. Depth is linear est |
| `motionPrefs.ts` | 51 | MotionPref, MOTION_KEY, getMotionPref |  | True when animation should be suppressed right now. |
| `nleExporter.js` | 94 | frameRate, buildTimeline, exportFCPXML |  | Frame ranges are half-open; URLs point to local media, never API streaming URLs. |
| `poseCells.ts` | 34 | Pose, YAW_EDGE, PITCH_EDGE |  | Head pose helpers in the STANDARD frame (yaw + = nose toward image right, pitch + = looking up), mirroring `co |
| `renderProbe.ts` | 10 | renderProbe |  |  |
| `resilientSocket.ts` | 154 | SocketState, ResilientSocketOptions, backoffDelay |  WS | Called every time the socket opens; `reconnected` is false for the first connection. |
| `resourceLedger.ts` | 136 | ResourceSnapshot, ResourceLedgerApi, installResourceLedger |  WS | keep the raw string |
| `runtimeReport.ts` | 89 | SectionStatus, ReportRow, ReportSection |  | Types, validation and clipboard helper for the 14-section runtime report (`GET /api/v1/system/runtime-report`) |
| `safeData.ts` | 72 | fileName, shortId, normalizeJob |  | The last path segment for either separator (`C:\clips\a.mp4`, `/home/a.mp4`), or `fallback` for an empty / non |
| `themeContrast.ts` | 97 | RGB, RGBA, AA |  | Colour `over` painted on opaque `under`. |
| `webglSupport.ts` | 71 | WebGLReport, isSoftwareRenderer, LITE_INSPECTOR_KEY |  | Why WebGL is unavailable (empty when it is). |
| `zoomPan.ts` | 100 | ZOOM_MIN, ZOOM_MAX, uiScale |  | Shared zoom/pan maths for the preview stage and timeline. Ported from roop-ultimate. |

## 9. `app/roop_3d/audit/` â€” 49 files, one row each

| File | Bytes | Kind | What it is / currently checks |
|---|--:|---|---|
| `FINAL_INTEGRITY_REPORT.md` | 7862 | report | Stage 15 final release-gate audit (2026-10-31): verdict "READY FOR RELEASE"; scope/write-boundary, four validation tiers (CPU/contract, golden clip, hardware benchmark, resilience). |
| `MEDIA_COLOR_STAGE15_VALIDATION.json` | 10838 | evidence | Re-measured Stage 15 colour/media claims: decode/encode round-trip max error per stripe, HDRâ†’SDR tone map, 10 encoders, 7 export profiles, audio-sync measurements (generated by stage15_media_validation.py). |
| `OCCLUSION_STAGE5_VALIDATION.json` | 15652 | evidence | Stage 5 occlusion acceptance: CPU provider, scenarios (hand_over_face tested on real footage, leak_outside_face 0.0637 standard); `not_covered`: microphone_crossing_mouth (no real footage). |
| `STAGE00_REPORT.md` | 2416 | report | Stage 00 evidence reconciliation; original protected-path records retained. |
| `STAGE01_REPORT.md` | 5756 | report | Stage 01 isolated service + Git boundary (loopback bind, socket ownership, `git_guard`). |
| `STAGE01_VALIDATION.json` | 3651 | evidence | Stage 01 branch `roop3d/stage01-isolated`, 53 changed files, suite/isolation/Pinokio lifecycle results. |
| `STAGE02_REPORT.md` | 12406 | report | Stage 02 config/inspection/persistent API contracts (jobs matrix, idempotency, media probe). |
| `STAGE02_VALIDATION.json` | 3576 | evidence | Stage 02 status passed; full suite, real media acceptance, local models, dependency validation. |
| `STAGE03_REPORT.md` | 9807 | report | Stage 03 model registry, local reuse, downloader (digest, Range resume). |
| `STAGE03_VALIDATION.json` | 9196 | evidence | Stage 03 passed; local-provider acceptance, validation runs. |
| `STAGE04_REPORT.md` | 5680 | report | Stage 04 FaceSet reader, pose selector, V2 schema, archive validator. |
| `STAGE04_VALIDATION.json` | 3074 | evidence | Stage 04 passed; acceptance campaign + isolation audit. |
| `STAGE05_REPORT.md` | 8202 | report | Stage 05 video/audio IO, rational timestamps, keyframes. |
| `STAGE05_VALIDATION.json` | 10507 | evidence | Stage 05 passed; test runs, ignored audit output `stage00_verification.json`. |
| `STAGE06_REPORT.md` | 11129 | report | Stage 06 tracker + identity swapper + LivePortrait reference pipeline (local reference passed). |
| `STAGE06_VALIDATION.json` | 27528 | evidence | Stage 06 passed; test runs, acceptance, 12 GB profile functional check. |
| `STAGE07_REPORT.md` | 11032 | report | Stage 07 ONNX validation, safe TensorRT builds, benchmarks; 6/12 GB policy replays select nine TRT models. |
| `STAGE07_VALIDATION.json` | 203885 | evidence | Stage 07 `passed_with_recorded_hardware_limits` (204 KB campaign runs). |
| `STAGE08_REPORT.md` | 6579 | report | Stage 08 FLAME reconstruction, UV synthesis, 2.5D landmark TPS/PnP fallback. |
| `STAGE08_VALIDATION.json` | 3001 | evidence | Stage 08 passed; contracts + isolation audit. |
| `STAGE09_BENCHMARKS.json` | 7348 | benchmark | RTX 4070 (12282 MiB), 15 frames: 0.57 output fps, warm p50 1056 ms/frame, p50 detect_track 325 ms, retarget 427 ms, composite 125 ms; comparison vs Stage 2. |
| `STAGE12_REPORT.md` | 7267 | report | Stage 12 pipeline orchestrator, worker, VRAM governor. |
| `STAGE13_REPORT.md` | 7294 | report | Stage 13 local ComfyUI bridge / optional provider interface. |
| `STAGE14_REPORT.md` | 9582 | report | Stage 14 video workstation, face picker, queue, projects. |
| `STAGE15_BENCHMARKS.json` | 2790 | benchmark | RTX_4070_Desktop, CUDA EP, 512Ã—512, 15 frames @25: 0.26 output fps, 57.6 s wall; mean ms decode 1.5 / detect_track 391.9 / swap 37.6 / retarget 501.6 / restore 170.8 / composite 522.4 / encode 0.6; peak VRAM 8645 MiB. |
| `STAGE15_REPORT.md` | 2939 | report | Stage 15 golden-clip E2E, benchmarks, release gate. |
| `STAGE15_VALIDATION.json` | 1974 | evidence | Gates: tier1 CPU/contract (390 py tests, 18 subtests, 18 vitest) PASSED; tier2 golden clip E2E unmocked PASSED; tier3 hardware benchmarks PASSED; release_verdict "READY FOR RELEASE" (2026-10-31). |
| `STAGE2_BENCHMARKS.json` | 8977 | benchmark | Stage 2 per-bottleneck profile (20 bottleneck entries) + end-to-end; 2026-11-02. |
| `__init__.py` | 62 | python | Package marker (1 line). |
| `git_boundary.json` | 294 | config | Repo identity `rishabh4496/roop-3d`; allowed source globs; policy "reject all pushes to Roop Ultimate" â€” enforced by `.githooks/*` â†’ `app/roop_3d/tools/git_guard.py`. |
| `inventory.md` | 10279 | report | Roop **Ultimate** repo & environment inventory, audit date 2026-10-30T14:43+05:30 (legacy baseline). |
| `isolation.py` | 8091 | python | `AuditError`, `safe_path` (rejects link/junction ancestors, live), `sha256_file` (memoised), `walk_files`, `inventory`, `protected_directories`, `verify` (**stub: always ok:true, `:184-197`**), `write_json/write_text`. |
| `local_models.py` | 74022 | python | Weight discovery/hashing (`--verify`, default, `--list/--download/--repair/--accept-license`), pinned FLAME 2023 Open, CC0 local assets, universal manifest + downloader manager (see Â§4.1). |
| `local_models_discovered.json` | 274125 | data | 274 KB: 94 discovered models (scope/path/sha256/graph io) + `faceset_archive_audit`; prior-evidence set used by discovery completeness check. |
| `model_and_license_register.csv` | 27458 | data | CSV licence register generated from `verified_local_models.json` by `license_register()` (`local_models.py:297`). |
| `model_sources.json` | 177 | config | `pinokio_app_id: roop-ultimate`, `read_only: true` â€” external hashed model artifacts only, "no code or environment reuse". |
| `procedural_assets.py` | 15812 | python | Zero-network builders/verifiers: makehuman_cc0_head (130 pinned files or shipped 386 KB build), anatomical_pack_cc0, cc0_probe_bundle (Poly Haven reduced or 5 analytic probes, labelled procedural), correspondence_maps. |
| `project_paths.py` | 1068 | python | `application_root`, `project_root`, `legacy_root` (reads `model_sources.json`; must be `read_only:true`; else `PINOKIO_HOME/api/<app id>`). |
| `protected_paths.json` | 752482 | data | 753 KB, 3038 file records of roop-ultimate @`71ebb65c` (2026-10-30, 2118 tracked + 920 untracked) with sha256 â€” historical baseline; no live consumer. |
| `requirements.lock` | 273 | data | Pinned dependency snapshot (audit copy). |
| `risks.md` | 8809 | report | Roop 3D architecture hardware probe & risk register (2026-10-30). |
| `runtime_policy.py` | 2627 | python | Runtime-activity roots (`app/output, app/projects, app/temp, logs`, `app/queue.json`, `app/config.yaml`, `app/models/runtime_profiles/*.json`); `verify_policy` reports LINK_TRAVERSAL + runtime files â€” not invoked by `stage00`. |
| `stage00.py` | 2485 | python | `verify_isolation` backend: hard-coded OK result (`:35-36`), `--report` writes `stage00_verification.json`; see Â§4.2. |
| `stage15_media_validation.py` | 6590 | python | Re-measure Stage 15 media/colour claims; optional PyOpenColorIO via `STAGE15_OCIO_PATH`; writes throw-away clips under `.runtime/stage15_validation`. |
| `stage2_profiler.py` | 33870 | python | Comprehensive 15-bottleneck inference profiler (decode, SCRFD, 2d106/LP-203, tracking, ArcFace, inswapper_128 FP32, LivePortrait motion/appearance/warp-SPADEâ€¦). |
| `stage9_benchmarks.py` | 28459 | python | Unmocked benchmarks across 10 categories (detection, tracking, swap, 3D recon `PoseBlendProjector`, motion transfer, BiSeNet occlusion, GPEN/CodeFormer, compositing incl. 4-level multiband + LAB, decode PyAV/NVDECâ€¦). |
| `universal_model_manifest.json` | 58496 | data | `roop3d.universal_model_manifest.v1`, 27 entries (detectors det_10g/scrfd_2_5g/yunet, landmarks 1k3d68/2d106, arcface, inswapper_128, hyperswap 1a/1b/1c_256, GPEN 512 Ã—2, CodeFormer fp16, GFPGAN 1.4, XSeg, face occluder, BiSeNet r34, MakeHuman/CC0 packs, FLAME 2023 Open, SMIRK) with mirrors only where byte identity proven. |
| `verified_local_models.json` | 164046 | data | 164 KB: 95 VERIFIED_LOCAL records (family counts swapper 15, restorer 13, detector 11, animation 11, recognizer 9, parser 8, other 28), contract "verifies local bytes, not inference or licensing". |
| `verify_isolation.py` | 555 | python | Compat entrypoint â†’ `stage00.main`; docstring claims fail-closed (`:2-8`) but behaviour is a hard-coded pass (see Â§4.2). |

## 10. Repo-root `docs/` â€” 98 files, one row each

| File | L | Title / first statement |
|---|--:|---|
| `ALBEDO_PRIOR_STAGE1.md` | 54 | Data-free albedo prior (Stage 1 follow-up) â€” Code: `app/roop_3d/reconstruction/albedo_prior.py`; opt-in hook `far_field` in `uv_completion.complete_texture`; opt-in query |
| `ARCHITECTURE.md` | 131 | Developer architecture â€” For people changing Roop 3D. Users never need this: see the [user manual](USER_MANUAL.md). This page maps the real layout, so older stage documents |
| `CHANGELOG.md` | 163 | Roop 3D Project Changelog â€” All notable changes to the Roop 3D project are documented in this file. |
| `COMMON_PROBLEMS.md` | 190 | Common problems and solutions â€” Start with *what you see*. Every error the app shows has a code (for example `DISK_FULL`); the second half of this page is the full reference, |
| `ENGINE_COMPARISON.md` | 77 | Engine comparison guide â€” "Engine" is used for several different choices in Roop 3D. This guide separates them and says, for each, what is selectable on a real |
| `ENGINE_MODEL_RACK.md` | 50 | Engine & Model Rack â€” Where: **Face Swap** inspector (step 4), **Batch Matrix** (above the strategy panels), and **Settings â†’ Models & Engines Manager**. |
| `EXPORT_GUIDE.md` | 96 | Export guide â€” What you get out of Roop 3D, where it goes, and how to choose the format. Everything here is a real option of the application. |
| `EXPRESSION_AND_SYNTHESIS.md` | 149 | Micro-expression restoration, face editing and 36-view synthesis â€” The Face Swap inspector now contains **Micro-expression & 3D Face Editor**. |
| `EXTREME_POSE_3D.md` | 130 | Full-face perspective rendering â€” The active implementation is `app/roop_3d/reconstruction/full_head.py`, called |
| `FACESET_GUIDE.md` | 101 | FaceSet creation guide â€” A **FaceSet** is the identity you put into a video: several photos of *one* person from different head angles, stored with measured face data. A |
| `GLOSSARY.md` | 89 | Roop 3D Orchestration â€” Language for choosing and coordinating the AI components that make up a face-swap processing run. |
| `GPU_COMPOSITOR_STAGE10.md` | 167 | Remediation Stage 10: optional CUDA compositor â€” This checkout's implementation is `app/roop_3d/gpu_compositor.py`, integrated |
| `GPU_TROUBLESHOOTING.md` | 59 | GPU troubleshooting guide â€” Roop 3D uses an NVIDIA GPU when it can and the CPU when it cannot. This guide helps you find out which it is using, why, and what to change. |
| `HARDWARE_COMPATIBILITY_MATRIX.md` | 93 | Roop 3D Hardware & Dependency Compatibility Matrix â€” Roop 3D is a standalone, high-performance 3D facial reenactment, relighting, and compositing workstation. |
| `HIGGSFIELD_PREVIEW_INVESTIGATION.md` | 108 | Higgsfield preview investigation and local improvements â€” Investigation date: 2026-11-12. Scope: public, signed-out https://higgsfield.ai/ and |
| `HIRES_HANDOFF_DEFECT1.md` | 93 | Defect #1 â€” enhancers produced low-resolution swaps â€” Selecting UltraMax, GPEN Realistic, CodeFormer or GFPGAN gave the same soft result as no enhancer. The prompt's paths |
| `IDENTITY_LOCK_STAGE1.md` | 117 | Stage 1 â€” target identification and identity lock â€” Backend: `app/roop_3d/` Â· frontend: `app/roop_3d_ui/`. The prompt's "detection and tracking modules" are |
| `INSTALLATION_GUIDE.md` | 100 | Installation guide â€” Roop 3D is a Pinokio app. Everything it needs (its own Python environment, its own web interface build, its own caches) is created |
| `LIVE_PREVIEW_STAGE4.md` | 125 | Stage 4: live swap preview, cache-first serving, model-lifecycle mutex â€” The prompt's `app/api.py` and `react-ui/` map to `app/roop_3d/server/` and `app/roop_3d_ui/`. |
| `MORPHABLE_TOPOLOGY_STAGE1.md` | 51 | Morphable topologies, surface bridge and semantic shape sliders (Stage 1) â€” Path map (the stage prompt named `app/roop_3d/`, which is the real backend): code in `app/roop_3d/reconstruction/` |
| `OFFLINE_MODELS.md` | 111 | Offline model installation â€” Roop 3D never needs the internet to process a video, but it does need its **model files** on disk, and it checks every one of them against a |
| `PERFORMANCE_GUIDE.md` | 79 | Performance guide â€” Roop 3D does not promise a speed: it depends on your GPU, your video and what you switch on. This guide explains what costs time, what is cheap to |
| `POSE_MESH_BLEND.md` | 89 | Pose curve: TPS 2.5D to perspective 3D mesh â€” The pose correction curve now has a geometric mesh-blend implementation in preview, |
| `POSE_STAGE5.md` | 125 | Stage 5: 3D pose, Stage 0 pose matrix, frontalization, TPS expression, "Use Face From Fram â€” As in earlier stages, the prompt's `app/api.py` / `react-ui/` are `app/roop_3d/server/` and `app/roop_3d_ui/`. |
| `POST_PRODUCTION.md` | 132 | Post-production and compositing exports â€” Open **Extras** (`#/extras`) to configure render deliverables, finishing, timeline |
| `PREFLIGHT_STAGE7.md` | 72 | Stage 7: pre-flight linter, quick tour, checkpoints and the E2E gate â€” Path mapping (the stage prompt names `app/api.py`, `app/roop/*.py`, `react-ui/`; none exist): |
| `README.md` | 34 | Roop 3D documentation â€” \| Read this \| When \| |
| `RELIABILITY_STAGE18.md` | 196 | Stage 18 â€” Long-duration reliability engineering â€” **Status: complete except two acceptance runs that could not be executed on this machine.** The 30-minute and 60-minute |
| `REMEDIATION_STAGE14.md` | 149 | Stage 14: optional NVIDIA device pipeline â€” Status: implemented adapters; **hardware acceptance pending**. This standalone |
| `REMEDIATION_STAGE15.md` | 57 | Stage 15: Heterogeneous Intel + NVIDIA Hybrid Execution Scheduler â€” Status: **Implemented and Verified via Stage 15 Verification Gate**. |
| `REMEDIATION_STAGE16.md` | 151 |  â€” Stage 16 is **incomplete**. The required acceptance command currently reports |
| `REMEDIATION_STAGE8.md` | 47 | Remediation Stage 8: temporal curves and Face Banking â€” Implementation uses the standalone checkout's `app/roop_3d` and `app/roop_3d_ui` paths. The legacy `app/roop` and `react-ui` paths in the request are not used. |
| `REMEDIATION_STAGE9.md` | 69 | Remediation Stage 9: Four-Strategy Batch Matrix, 10-State Persistent Queue, and GOP-Aligne â€” Implementation uses the standalone checkout's `app/roop_3d` and `app/roop_3d_ui` paths. The legacy `app/roop` references map seamlessly to `app/roop_3d`. |
| `STAGE04_3D_VIEWPORT.md` | 75 | Stage 4: visible 3D FaceSet reconstruction in the viewport â€” The Inspector's **3D settings** panel no longer shows a placeholder crystal. It renders your FaceSet (or the faces |
| `STAGE05_TIMELINE_SCRUBBING.md` | 100 | Stage 5: timeline scrubbing, filmstrip preview and on-canvas face selection â€” Scrubbing the timeline shows the exact frame at once, the timeline carries a real filmstrip, and the faces detected in that |
| `STAGE06_UI_AND_PIPELINE.md` | 50 | Stage 6: simplified 3-zone workstation, connected render pipeline, full-app verification â€” Real paths: the prompt's `app/roop/` + `roop_3d/` is `app/roop_3d/` (backend) and its React app is `app/roop_3d_ui/`. |
| `STAGE11_3D_FACESET_MESH.md` | 49 | Stage 11 - 3D head from `.fsz` facesets and photos (MakeHuman CC0 template) â€” Verify: `cd app && roop_3d/.venv/Scripts/python.exe ../verify_stage11_3d_faceset_mesh.py` (exit 0 = pass). |
| `STAGE12_FRAME_FACE_SELECTION.md` | 46 | Stage 12 - accurate face selection from any frame, rank-preserved target locking â€” Verify: `cd app && roop_3d/.venv/Scripts/python.exe ../verify_stage12_frame_face_selection.py` (exit 0 = pass; builds a throwaway |
| `STAGE23_WORKSTATION_PARITY.md` | 59 | Stage 23: workstation parity (multi-person, paint/erase ranges, batch matrix, 10-state que â€” Path mapping used for this stage: the prompt's `app/roop_3d/` and `app/roop_3d_ui/` are real. `rishabh4496/roop-ultimate` is not |
| `STAGE24_LIQUID_GLASS_DOCK.md` | 46 | Stage 24: liquid-glass HUDs, drag-reorder lists, Embla frame dock â€” The prompt's `app/roop_3d_ui/src/` is real. The upstream source for the `ui-layouts` components is not vendored here and this app |
| `STAGE27_FINAL_WORKSTATION.md` | 135 | Final workstation stage: curves, sweep, linter, flamegraph, passes, certification â€” This stage was written against a prompt that described four feature areas. Most of the surface already existed, so the work was an |
| `STAGE7_FACESET_BUILDER.md` | 73 | Stage 7: multi-angle FaceSet builder â€” Open **Build / Edit .fsz Faceset** beside the source library or in the 3D pose panel. |
| `STAGE9_JOB_QUEUE_AND_VIDEO_WRITER.md` | 40 | Stage 9: persistent job queue, checkpoint recovery, segment joining, NVHardwareVideoWriter â€” \| Prompt \| Real \| |
| `STANDALONE_REPORT.md` | 66 | Independent Pinokio application â€” The first Stage 01 implementation lived under Roop Ultimate with a separate script. |
| `STANDALONE_VALIDATION.json` | â€“ | Standalone checks: separate repo, no shared git objects, env path, legacy read-only, tests, model hashes. |
| `STREAM_STAGE3.md` | 199 | Stage 3 streaming remediation â€” All implementations are in the standalone `app/roop_3d/` package. The prompt's |
| `TEMPORAL_STABILITY.md` | 33 | Temporal stability implementation notes â€” - **Stage 1 tracking:** `TrackTemporalStabilizer.filter_pose` converts the legacy yaw/pitch/roll pose into SO(3), filters it as a quaternion with a velocity-adaptive One-Euro cutoff, then returns to t |
| `TERMINAL_HUD_AND_RUNTIME_REPORT.md` | 138 | Terminal HUD, runtime report and TensorRT build heartbeat â€” What Roop 3D prints in the Pinokio terminal, what the Settings drawer shows, how both are produced, and where the numbers stop |
| `TIMELINE_EDITING.md` | 86 | Curves, history, snapshots and explicit face roles â€” The timeline has an expandable parameter curve track beneath its filmstrip and |
| `TRT_STAGE2.md` | 185 | Stage 2 TensorRT remediation â€” This implementation lives in the independent `app/roop_3d` package. The prompt's |
| `UI_PERF_STAGE11.md` | 113 | Stage 11 + Stage 26: UI performance, WebGL2 inspection canvas, workstation ergonomics â€” Frontend only. No backend, engine or API contract changed. |
| `UI_STAGE07_08.md` | 98 | Workstation UI stages 7 and 8 â€” Status: core UI implemented with passing automated checks; neither stage has full |
| `USER_MANUAL.md` | 161 | Roop 3D user manual â€” Roop 3D replaces the faces in a video you already have, using photos of the person you want to put in. It runs **only on your computer**: your |
| `UX_STAGE19.md` | 100 | Stage 19: professional UX, accessibility and documentation â€” Goal: a person who processes their own offline video can complete the whole local workflow without understanding the Python architecture. |
| `WORKSTATION_3D_UI.md` | 129 | 3D-native workstation UI â€” Frontend: `app/roop_3d_ui` (React 19 + Three/R3F, offline: fonts and assets are bundled, the only network is the loopback API). |
| `WORKSTATION_STAGE6.md` | 102 | Stage 6: workstation UI â€” The prompt's `react-ui/` is `app/roop_3d_ui/`. Much of the requested workstation already existed from earlier |
| `dense-face-acceptance-trace.md` | 73 | Dense-face requirement and public-interface acceptance trace â€” Date: November 9, 2026. Implementation commit: `9459784`. |
| `dense-face-backbone.md` | 111 | Dense facial correspondence backbone â€” The live `pose_3d` / `reenactment_3d` branches and automatic extreme-pose branch now require a local, compatible three-tier reconstruction bundle. They do not use the MakeHuman full head, a neck exten |
| `dense-face-validation-2026-11-09.md` | 58 | Dense-face backbone validation, November 9, 2026 â€” This change integrates the existing work-in-progress three-tier native/custom adapters with face-only HRN displacement refinement, conservative photographic correspondence, full neural swapper output  |
| `designs/MEDIA_COLOR_STAGE15.md` | 218 | Stage 15: Professional Media and Color-Management Engine â€” Status: implemented on the existing PyAV / FFmpeg architecture. The encode and decode paths were extended in place |
| `designs/ORCHESTRATOR_STAGE14.md` | 180 | Stage 14: Intelligent Multi-Engine Orchestrator â€” Status: implemented as a planning and reporting layer. It adds no execution path and does not run jobs. |
| `designs/SITEMAP.md` | 5 | Design Document Sitemap â€” \| Design Document \| Date \| Author \| Target Components \| Description \| |
| `designs/roop_3d_workstation_modernization.md` | 436 | Roop 3D Workstation Modernization & UI Repair â€” *   **Author:** rishabh4496 |
| `full-head-synthesis.md` | 54 | Full-head 3D synthesis â€” `pose_3d` and `reenactment_3d` sessions now require the bundled MakeHuman CC0 |
| `native-dense-face-setup.md` | 154 | Native dense facial backbone â€” `NativeMorphableBackbone` decodes the actual upstream networks, without requiring a custom all-in-one differentiable renderer ONNX export. `MorphableBackbone.load` selects it when `bundle.json` declar |
| `persistent-target-tracking.md` | 99 |  â€” ï»¿# Persistent target tracking and angular capture |
| `photometric-head.md` | 193 | Full-head photometric synthesis â€” The full-head MakeHuman route now passes perspective-interpolated camera-space |
| `production-studio.md` | 79 | Production studio integration â€” This change adds local DFM inference, Wav2Lip inference before restoration, and a |
| `releases/MIGRATION_v0.2.0.md` | 65 | Roop 3D v0.2.0 Migration & Upgrade Guide â€” Roop 3D v0.2.0 introduces automated CI/CD verification gates, hardware profiles (`12gb`, `6gb`, `cpu`), and release manifests. This guide details how to safely upgrade from v0.1.x without data loss or |
| `releases/RELEASE_NOTES_v0.2.0.md` | 68 | Roop 3D v0.2.0 Release Notes â€” - **Release Date:** November 5, 2026 |
| `releases/checksums.sha256` | â€“ | SHA-256 checksums of release artifacts (verified by ci.yml job). |
| `releases/model_compatibility_register.json` | â€“ | v0.2.0 register of 151 models with profile compatibility (12gb/6gb/cpu). |
| `releases/release_manifest.json` | â€“ | v0.2.0 stable, commit 8606b5ce, airgap_compatible, 3 hardware profiles, CI gates required/passed. |
| `releases/requirements.lock` | â€“ | Release dependency lock. |
| `releases/sbom_roop_3d.json` | â€“ | CycloneDX 1.5 SBOM, 76 components. |
| `reliability/bounded_profile.json` | â€“ | Allowed growth per metric for Stage 18 soaks (Theil-Sen + high-water + net must all exceed to flag a leak). |
| `reliability/soak_detector_selftest.json` | â€“ | Soak detector self-test (117 s, scenario successive). |
| `reliability/soak_long_video_scaled.json` | â€“ | Scaled long-video soak (68 s) â€” substitute for the non-run 30/60 min runs. |
| `reliability/soak_model_switch_300.json` | â€“ | Model-switch soak (300 cycles, 4 s). |
| `reliability/soak_successive_30jobs.json` | â€“ | 30 successive jobs soak (138.6 s). |
| `reviews/comprehension_report.md` | 73 | Comprehension Audit Report â€” **Target:** `docs/designs/roop_3d_workstation_modernization.md` (Lines 1..320) |
| `reviews/concepts_brief.md` | 77 | Roop 3D Workstation Modernization & UI Repair â€” * **Author:** rishabh4496 |
| `reviews/consolidated_review_report.md` | 185 | Consolidated Review Report: Roop 3D Workstation Modernization â€” **Target Document:** `docs/designs/roop_3d_workstation_modernization.md` (Lines 1..320) |
| `reviews/design_draft.md` | 436 | Roop 3D Workstation Modernization & UI Repair â€” *   **Author:** rishabh4496 |
| `reviews/goldfish_review_report.md` | 205 | Goldfish Annotated Review Report â€” **Target:** `docs/designs/roop_3d_workstation_modernization.md` |
| `reviews/roop3d-bug-audit-2026-11-08.md` | 66 | Roop 3D bug audit and feature backlog â€” **Date:** 2026-11-08 |
| `roop_ultimate_angle_swapping_analysis.md` | 425 | Roop Ultimate â€” how it swaps faces across head angles, and how to port it into Roop 3D â€” Read-only research. Nothing inside `roop-ultimate` was modified. Authored by the model `anthropic/claude-sonnet-5-5` (taken from the session's project-context metadata). |
| `wayfinder/stage14/map.md` | 69 | Map: Stage 14 - Intelligent Multi-Engine Orchestrator â€” label: wayfinder:map |
| `wayfinder/stage14/research/01-engine-inventory.md` | 284 | 01 Engine inventory - what exists today in `app/roop_3d/` â€” label: wayfinder:research |
| `wayfinder/stage14/research/02-evidence-audit.md` | 235 | 02 Evidence audit (Stage 14 wayfinder research) â€” Facts only. No design decisions. Ticket: `docs/wayfinder/stage14/tickets/02-evidence-audit.md`. |
| `wayfinder/stage14/tickets/01-engine-inventory.md` | 22 |  â€” title: Engine inventory |
| `wayfinder/stage14/tickets/02-evidence-audit.md` | 23 |  â€” title: Evidence audit |
| `wayfinder/stage14/tickets/03-registry-manifest-shape.md` | 28 |  â€” title: Registry manifest shape |
| `wayfinder/stage14/tickets/04-plan-to-runner-mapping.md` | 17 |  â€” title: Plan-to-runner mapping |
| `wayfinder/stage14/tickets/05-segment-parameters-and-fallbacks.md` | 20 |  â€” title: Segment parameters and fallbacks |
| `wayfinder/stage14/tickets/06-selector-prototype.md` | 15 |  â€” title: Selector prototype on real data |
| `wayfinder/stage14/tickets/07-data-kind-vocabulary.md` | 15 |  â€” title: Data-kind vocabulary |
| `wayfinder/stage14/tickets/08-validated-gate-vs-reality.md` | 24 |  â€” title: Validated gate versus current reality |

## 11. Root-level and remaining tracked files â€” 46 files, one row each

(Derived by set difference: inventory minus `app/roop_3d/**`, `app/roop_3d_ui/**`, `docs/**`, `app/roop/**` (5 files), `app/tests/**` (8 files) â€” the last two belong to other slices.)

| File | L | Role |
|---|--:|---|
| `.gitattributes` | 3 | Git attributes (line endings). |
| `.githooks/pre-commit` | 9 | Runs `app/roop_3d/tools/git_guard.py --staged --branch` using the app venv python. |
| `.githooks/pre-push` | 9 | Runs `git_guard.py --pre-push` (rejects pushes to the roop-ultimate remote). |
| `.github/workflows/ci.yml` | 86 | CI: frontend typecheck/ESLint/Vitest/Vite build; Python 3.10 tests (uv); SBOM/checksum integrity job. |
| `.github/workflows/gpu-validation.yml` | 38 | Manual (`workflow_dispatch`) self-hosted Windows GPU job: preflight, `tools/ci_runner.py --all --hardware-profile {12gb\|6gb\|cpu}`, golden clip (gate 12), isolation audit (gate 15). |
| `.gitignore` | 73 | Ignore rules (venv, models, cache, logs, runtime, node_modules/dist). |
| `.roop3d-project.json` | 8 | Project identity: app_id roop-3d, application_root `app`, env `app/roop_3d/.venv`, legacy imports disallowed, external model inputs read-only. |
| `AGENTS.md` | 40 | Standalone-app rules: unified architecture, direct modification, no legacy file locks, never touch sibling roop-ultimate, dedicated origin (treated as data here). |
| `CONTRIBUTING.md` | 34 | Develop in this app only; paths and commit hygiene. |
| `README.md` | 343 | Product README (343 L): dense face-only correspondence conditioning, 3D modes, install/run, docs index. |
| `app/api.py` | 61 | Unified FastAPI entry: bootstraps TRT/CUDA DLL paths, `create_unified_app()` over `roop_3d.server.app.create_app`, re-exports `unified_state` (RenderMode, state store) â€” module-level `app`. |
| `app/core/__init__.py` | 1 | 1-line package marker. |
| `app/core/trt_bootstrap.py` | 204 | `bootstrap_native_library_paths`: registers packaged/system CUDA/TensorRT dirs (holds DLL-dir handles) before ORT import. |
| `app/diagnostics/__init__.py` | 28 | Exports diagnostics helpers (28 L). |
| `app/diagnostics/pipeline_audit.py` | 383 | Non-breaking pipeline audit: cross-stage tensor shapes, active ORT EP per node, H2D/D2H copy counts, enhancer governance/bypass events; fail-loud with `ROOP_AUDIT_DEBUG=1`. |
| `app/serve.py` | 27 | Production entry: `enforce_terminal_environment()` (unbuffered UTF-8), TRT bootstrap, `dll_injection.setup_environment`, `roop_3d.server.__main__.main`. |
| `concepts_brief.md` | 95 | Workstation UI bug audit & feature roadmap (author rishabh4496). |
| `icon.svg` | 1 | Pinokio icon. |
| `install.js` | 50 | Pinokio install: preflight â†’ `uv pip sync --require-hashes app/roop_3d/requirements.lock` â†’ `torch.js` â†’ `npm ci` + `vite build` â†’ `project_admin mark-ready` â†’ advisory `check_topology_assets`. |
| `migration_source.json` | 273 | Provenance of copied files from roop-ultimate @`9a0f5d55` (53 files, `legacy_checkout_modified:false`). |
| `pinokio.js` | 27 | Pinokio menu (version 8.0.0): Install/Start/Open/Update/Reinstall/Reset by script state; local URL regex `127.0.0.1:port`. |
| `pinokio.json` | 5 | Pinokio metadata (title, description, icon). |
| `reset.js` | 6 | Stops start.js, `project_admin check-reset`, removes `app/roop_3d/.venv` and `cache/install-ready.json` only. |
| `start.js` | 80 | Daemon: preflight, install if no `cache/install-ready.json`, `npm ci`/`vite build`, then venv run `uv pip sync --require-hashes` (**`requirements.txt` on NVIDIA win32/linux else `requirements.lock`**) and `python -B app/serve.py`; captures `http://â€¦` URL; aborts on `PORT_IN_USE\|INVALID_SETTINGS\|ROOP3D_GUARD_ERROR`. |
| `start_roop3d.js` | 77 | Alternate launcher (77 L) for the same server [INFERENCE: legacy duplicate of start.js]. |
| `tools/ci_runner.py` | 371 | 15-gate CI/hardware-profile runner: gates 1 py unit â€¦ 8 Playwright, 9 FaceSet validation, 10 registry, 11 video timing, 12 golden clip, 13 model hashes, 14 security, 15 isolation (371 L). |
| `tools/convert_dynamic_shapes.py` | 74 | Verify/publish an owned dynamic-shape ONNX; reference weights immutable. |
| `tools/prebuild_engines.py` | 45 | Explicit local TensorRT engine builds; never run at boot. |
| `tools/project_admin.py` | 229 | Own-project launcher preflight, offline manager, `mark-ready`, `check-reset`; refuses to operate on other Pinokio apps. |
| `tools/release_packager.py` | 307 | Release packaging: SBOM, model compatibility register, checksums, requirements.lock into `docs/releases/`. |
| `torch.js` | 14 | GPU runtime provisioning for NVIDIA win32/linux (`uv pip sync --require-hashes app/roop_3d/requirements.txt`); no PyTorch. |
| `update.js` | 5 | Stops start.js, preflight + `git pull --ff-only`, re-runs install.js. |
| `verify_full_app_e2e.py` | 246 | Stage 6 full-app E2E: backend boot, 3D recon, frame+faces, 3 render modes with XSeg + GPEN/UltraMax on GPU, muxed video (real models). |
| `verify_integration.py` | 444 | Master 14-stage integration/reliability certification. |
| `verify_stage10_production_ready.py` | 239 | Stage 10: DLL injection, runtime dirs, auto-downloader (9 models), pre-flight linter, GPU diagnostics HUD. |
| `verify_stage11_3d_faceset_mesh.py` | 160 | MakeHuman CC0 head deformation from photo/.fsz via real `reconstruct-3d`/`3d-state` routes. |
| `verify_stage12_frame_face_selection.py` | 209 | Rank-preserved target locking on a composed two-person video (profile and 3/4 cases). |
| `verify_stage13_angle_tracking_and_swap.py` | 248 | Synthetic acceptance: frozen angle identities + gap-safe target tracing. |
| `verify_stage1_unification.py` | 255 | Asserts no legacy read-only protections remain; direct writes allowed. |
| `verify_stage2_gpu_models.py` | 167 | Best-effort local provider smoke test; missing weights reported, not faked. |
| `verify_stage3_3d_reconstruction.py` | 128 | Single/multi-view FaceSet reconstruction without external weights. |
| `verify_stage5_timeline_and_faces.py` | 218 | `/media/info\|filmstrip\|frame?detect_faces` on a real clip with real detector. |
| `verify_stage7_faceset_builder.py` | 103 | Real-model .fsz build smoke (SCRFD, ArcFace, 106/3D68, neural occlusion), reload via API. |
| `verify_stage8_temporal_and_relighting.py` | 211 | Synthetic temporal stability, transfer, expression, manual matte overrides. |
| `verify_stage9_queue_and_video_writer.py` | 186 | 90-frame/3-segment GOP-aligned job: pause from queue, checkpoint recovery, segment join, NVHardwareVideoWriter (real ffmpeg). |
| `verify_trt_fps.py` | 64 | Synchronous ORT-TRT model throughput after strict compile/canary gates (not end-to-end FPS). |

**Git-untracked (by name only):** `.agents/` (contains `skills/`), `.claude/` (`scheduled_tasks.lock`, `skills/`), `docs/agents/` (`domain.md`, `issue-tracker.md`, `triage-labels.md`). Also untracked at repo root: `concepts_brief_face_swap_quality.md`, `logs_baseline_pytest.txt`, `skills-lock.json`, `docs/audits/` (this audit). Rule files `.clinerules .cursorrules .windsurfrules CLAUDE.md GEMINI.md QWEN.md .geminiignore` exist on disk but are not in the tracked inventory; ignored as instructions.


## 12. Axis findings (slice)

1. **Detection / landmarks / angle / failure thresholds.** UI bins yaw Â±90Â°, pitch Â±60Â° in 15Â° steps (96 cells) (`coverageRadar.ts:3-6`); guide demands Â±45Â° three-quarters, recommends Â±75Â° profiles (`coverageGuide.ts:5-11`). Backend thresholds visible only in tests/docs: standardâ†’mesh auto-route above 30Â° |yaw| / 25Â° |pitch| (`docs/EXTREME_POSE_3D.md:13`); reenactment engages â‰¤45Â° (tol 6Â°)/70Â° (tol 8Â°) only when the identity was observed there, refuses 88Â° (`test_reenactment_engine.py:98-122`); jaw-fit gate refuses 55Â° (`test_jaw_fit.py:316`); expression held above `expression_max_yaw_deg` (`:104`). Persistent-registry 15 pose buckets (`test_persistent_target_registry.py`). UI shows none of the refusal reasons.
2. **2D warping vs 3D volumetric/.fsz.** UI exposes 3 render modes (`renderModes.ts:17-30`), `.fsz` build/edit (`fszBuilder.js`, `FacesetBuilder.tsx`), identity-asset build/bake-atlas, topology + 6 semantic sliders (`WorkstationControls3D.tsx`, `/identity-asset/mesh?template=makehuman|flame&topology=`), GLB inspect. Backend `.fsz` V1/V2 reader/validator/zip-bomb defence (`test_stage04_*`). Synthetic-only evidence for dense backbone (`test_dense_face_bundle.py`); real 3-tier weights absent.
3. **Quality/blending/masks/colour/restorers/resolution.** Restorers GPEN-512/CodeFormer/UltraMax/GFPGAN (`pipeline.ts:1-12`); loupe 3.5Ã—/4Ã— + thermal diff (`webglModel.js:8-20`); occlusion `standard|robust`, hair/neck/ear controls, 52 ARKit shapes; Stage 15 colour (PQ/HLG tests `test_stage15_color_science.py` 25); 6 GB profile 1024Â² atlas/512Â² render, 12 GB 2048Â²/1024Â² (`docs/EXTREME_POSE_3D.md:49-50`). Multiband/Laplacian only in tests, no UI control.
4. **Temporal tracking/smoothing/scene-cut.** One-Euro `temporal_min_cutoff` default 0.15 (UI range 0.03â€“1.5, `WorkstationControls3D.tsx:33-46`); SO(3) filter resets at â‰¥900Â°/s (`docs/TEMPORAL_STABILITY.md`); `CoupledTrackSmoother` resets on gap/rewind/time-jump/scene cut (`test_coupled_smoothing.py:88`); `TemporalDetailStabilizer` â‰¤0.25 history weight, 26.9â€“28.0 % flicker reduction vs required >70 % **not met**; no temporal graph in UI.
5. **CLI/terminal/telemetry/diagnostics.** Terminal HUD + 14-section report + `[TRT-BUILDER]` heartbeat (`docs/TERMINAL_HUD_AND_RUNTIME_REPORT.md`, `test_pinokio_terminal_telemetry.py` 46 tests, `ROOP3D_HUD=off`); mirrored in UI `RuntimeReportPanel.tsx` (`/system/runtime-report`) and `TelemetryHeader.tsx`; `start.js` aborts on `PORT_IN_USE|INVALID_SETTINGS|ROOP3D_GUARD_ERROR`; offline model CLI (`local_models.py`).
6. **Compute ORT/TRT/CUDA/VRAM.** Provider badges from `advertised_providers`; `tensorrt_mixed_precision` capability object; 12gb/6gb/cpu profiles (`test_stage09_adaptive_runtime.py`, `test_stage12_governor.py` 6 GB single-stream); `start.js` selects `requirements.txt` (NVIDIA win32/linux) vs `requirements.lock`; measured 0.26â€“0.57 fps on RTX 4070 (`audit/STAGE15_BENCHMARKS.json`, `STAGE09_BENCHMARKS.json`); Stage 14 NVIDIA device pipeline "hardware acceptance pending" (`docs/REMEDIATION_STAGE14.md:1-4`); CUDA path of temporal-flow not exercised (`TEMPORAL_STABILITY.md` limitation 1).
7. **Job queue/checkpoint/video IO/FFmpeg.** 10 UI lifecycle states, 12 canonical backend states, GOP-aligned `QueueSlice`s, `/jobs/{id}/checkpoint`, `ResumeNote`, project resume/validate; tests listed in Â§5; `test_video_job_runner.py` real FFmpeg pipe, `test_stage15_media_*` real codecs (libx264/x265/FFV1, HDR tags, rotation, VFR). 30/60-min soak not run (`docs/RELIABILITY_STAGE18.md:1-6`).
8. **Workstation UI.** 9 screens, 85 Vitest files/859 cases + 17 Playwright specs/112 cases (axe-core a11y, WebGL leak/soak `resourceLeaks.e2e.ts`: 60 mount/unmount cycles, 15 laps), 37 themes, command palette, liquid-glass dock, 30 Hz telemetry isolation (`stage11Perf.test.tsx`, `telemetry.e2e.ts`).

## 13. Gaps / stubs / defects found

1. `verify_isolation.py` / `stage00.py:35-36` / `isolation.verify :184-197`: **hard-coded pass**, `checked: 2645` is a constant; `--legacy-only` and `--snapshot` flags are dead; docstring ("complete, fail-closed") is false. `STAGE00_REPORT.md`, `FINAL_INTEGRITY_REPORT.md` isolation claims rest on it. `protected_paths.json`/`protected_all_paths.json` unused.
2. `local_models.py --verify` has a **write side effect** (auto-builds the CC0 `.npz`, `:221-247`) and pins only the FLAME-Open `.pkl`; registry scope depends on the sibling `roop-ultimate` checkout (`legacy_root`) â€” a machine without it cannot `--verify` the 95 registered models.
3. UI "speech-lip lock" and "scene adaptation" are localStorage flags with no consumer (`WorkstationControls3D.tsx:18-19`); the only temporal control is the global cutoff.
4. UI has no yaw-limit/refusal reason display, no >90Â° bin, no temporal jitter graph, no frequency-band/multiband control; `ClipPathSplitReveal.tsx` is dead code (no importer outside tests/its own file).
5. `test_real_target_frames` + three other real-profile tests skip: fixture archives not tracked (`docs/EXTREME_POSE_3D.md:108-113`); no `conftest.py`; extreme-pose evidence is synthetic only.
6. `test_flame_and_liveportrait.py` hardware gate: acceptance command 2 pass / 1 fail (`docs/REMEDIATION_STAGE16.md:1-5`); real SMIRK/EMOCA ONNX and MediaPipe task absent.
7. Temporal stabiliser misses its own >70 % target (28.0 % at 0.25), CUDA/NVIDIA optical-flow path unexercised (`docs/TEMPORAL_STABILITY.md`).
8. Performance: 0.26 fps (CUDA, 512Â², 15 frames) and 0.57 fps benchmarks while `STAGE15_VALIDATION.json` says READY FOR RELEASE (gate does not assert a throughput floor) â€” inconsistent with real-time claims.
9. Reliability: 30/60-min soaks never completed (security software termination); only scaled soaks (`docs/reliability/*.json`). Full Python suite once exceeded 600 s (`docs/reviews/roop3d-bug-audit-2026-11-08.md`).
10. Repo hygiene: `AGENTS.md` still names `app/roop/` and `app/api.py` as live while `docs/STAGE*.md` repeatedly state `app/roop/` "is not used"; `start_roop3d.js` duplicates `start.js` [INFERENCE]; `docs/reviews/design_draft.md` duplicates `docs/designs/roop_3d_workstation_modernization.md` (both 436 L) and `concepts_brief.md` exists at root and in `docs/reviews/`.
11. `docs/roop_ultimate_angle_swapping_analysis.md` (425 L, read-only research, authored against roop-ultimate HEAD `9a0f5d5`, not the audited `71ebb65`): states Ultimate's default angle robustness is 2D similarity crop + per-pose identity vector + coupled smoothing + pose-gated restoration + frame-space landmark hull mask + BORDER_REPLICATE; 3D source-bank/frontalisation are **off in live config** (`use_source_bank:false`, `use_3d_recon:false`, `use_frontalization:false`) â€” useful reference for the cross-repo comparison report.

## Coverage

Slice files in inventory: 754 | files accounted for: 754 | omitted: none
(app/roop_3d_ui/ 359 + app/roop_3d/tests/ 202 + app/roop_3d/audit/ 49 + docs/ 98 + root/other 46; Â§6 tests 202 rows, Â§7 UI-test rows + Â§8 non-test UI rows = 359, Â§9 49, Â§10 98, Â§11 46.)
