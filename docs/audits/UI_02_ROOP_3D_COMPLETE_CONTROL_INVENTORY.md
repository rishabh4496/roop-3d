# UI-02 — Roop 3D UI, Settings, API, and Preview Inventory

**Target:** this standalone `roop-3d` checkout only. No `roop-ultimate` files or sibling directories were modified.

## Audit method and evidence

The local source audit uses the current UI/backend code plus [`02_ROOP_3D_COMPLETE_AUDIT.md`](02_ROOP_3D_COMPLETE_AUDIT.md) and its source-indexed UI/test inventory [`_parts/R3_3_ui_tests_audit_docs.md`](_parts/R3_3_ui_tests_audit_docs.md). The source index accounts for 359 tracked UI files and describes screens, components, API contracts, tests, and backend modules. This report is a source/static inventory, not a runtime visual or hardware-performance measurement. Numeric defaults and bounds below are taken from the authoritative `app/roop_3d/core/settings.py` Pydantic contract and the workstation parameter contract; capabilities can further disable controls.

## Screens, panels, controls

The React 19 / TypeScript / Three.js + React Three Fiber app has nine routes: Home, Face Swap, Batch Matrix, Processing, Face Manager, Editor/Extras, Outputs/Gallery, History, Settings. `#/studio`, `#/identity-forge`, and `#/queue` are aliases; `#/models` and `#/diagnostics` open drawers. The default Face Swap route is the Workstation.

| Surface | Controls and data displayed | Source modules |
|---|---|---|
| Persistent shell | Route rail, project name/save/copy, theme and zoom, palette, connection/pipeline status, hardware/provider telemetry, guide/tour | `App.tsx`, `shell/*`, `TelemetryHeader.tsx`, `hud/*` |
| Source identity / FaceSet | `.fsz` library/import/build; view thumbnails and pose coverage; 12×8 yaw/pitch coverage radar; identity-asset status and 3D mesh; identity source blend; drag a FaceSet onto a target track | `FacesetLibrary`, `FaceSetDetail`, `FacesetBuilder`, `facemgr/*`, `utils/coverageRadar.ts`, `three/*` |
| Target media and face selector | Import/probe video, duration/FPS/frame navigation, inspect target, tracks and thumbnails, face selection/rank, per-track source mapping, role/exclusion, target identity lock and angle samples | `TargetPanel`, `FaceMapping`, `ActorPanel`, `FaceSelectionTray`, `IdentityLockBar`, `canvas/*`, `useFaceSelectionStore.ts` |
| 2D preview canvas | Workstation canvas modes: result, wipe, crossfade, blend, diff, loupe, mesh3d, matte; zoom/pan, overlays, pose and face boxes/landmarks; mask paint/erase and frame-range masks | `VideoViewport.tsx`, `canvas/WorkstationCanvas.jsx`, `canvas/webglModel.js`, `FaceOverlayCanvas.jsx`, `utils/maskRanges.ts` |
| 3D spatial inspector | Head-pose/coverage/morph/light probe/FLAME mesh tabs; lit/wireframe/albedo/normals/depth/cavity/SH buffer modes; source identity asset and pose geometry; FLAME/MakeHuman topology where available | `SpatialInspector.tsx`, `three/PoseGeometryViewer.jsx`, `three/FlameMeshViewer.tsx`, `three/viewportTools.ts`, `ThreeViewport.tsx` |
| Workstation pipeline inspector | Swap engine/mode/scope, pose mode/frontalization/expression, identity, occlusion strategy and thresholds, masks/feather/erosion, enhancer/fidelity/blend, color transfer, relighting/light, hair/neck/ear protection, restoration texture retention, shutter/grain, expression editor | `StudioInspector.tsx`, `WorkstationControls3D.tsx`, `FaceExpressionEditor.tsx`, `HairNeckProtection.tsx`, `workstation/pipeline.ts` |
| Timeline and project | Frame/timecode scrub, filmstrip, trim, chapters, scene-cut and face-track lanes, keyframe curves, mask ranges, export and project controls | `TimelineWorkstation.jsx`, `TimelineDock.tsx`, `TrackLanes.tsx`, `stores/parameterCurves.js`, `export/*` |
| Queue and batch | Four batch strategies, jobs and slices, queue reorder/retry/pause/resume/cancel, progress states, project checkpoints and outputs | `BatchMatrixScreen.jsx`, `queue/*`, `QueueDrawer.jsx`, `App.tsx` |
| Model and diagnostics | Model hub/rack/readiness; model integrity; system hardware and performance; runtime report (14 sections); stage latency/flamegraph; preflight, local model registry and output diagnostics | `modelhub/*`, `diagnostics/*`, `RuntimeReportPanel.tsx`, backend `server/routes.py`, `audit/*` |

Some surfaces are separate routes/drawers by design. The UI does not and cannot place all nine routes, dialogs, and every parameter on one physical display at once. The Face Swap settings inspector is the scope of the all-visible workstation change; other routes remain reachable and no dialog is needed to expose its schema settings.

## Authoritative backend settings schema

`Settings` in `app/roop_3d/core/settings.py` is frozen, rejects unknown fields, validates defaults, and is the API source of truth (`GET /config`, `PUT /config`; capabilities expose the same JSON schema). Numeric ranges are inclusive/exclusive as shown. `schema_version` is fixed at 2. The UI displays capability-filtered schema controls; these are not all individual model- or workstation-parameter controls.

| Key(s) | Default and legal values/range |
|---|---|
| `temporal_min_cutoff`, `temporal_beta`, `temporal_d_cutoff`, `pose_hysteresis_degrees` | `.15` (`0 < x ≤ 120`), `.04` (`0 ≤ x ≤ 10`), `1` (`0 < x ≤ 120`), `2.5` (`0 ≤ x < 10`) |
| `heterogeneous_split_mode`, `qsv_offload_enabled`, `auxiliary_offload_enabled` | `false`, `false`, `true` |
| `openvino_device` | `AUTO:GPU,NPU,CPU` |
| `schema_version` | Literal `2` |
| `execution_mode` | `swap`; `swap`, `volumetric`, `hybrid`, `diffusion_vfx` |
| `hardware_profile` | `auto`; `auto`, `cpu`, `6gb`, `12gb` |
| `preferred_provider` | `auto`; `auto`, `TensorrtExecutionProvider`, `CUDAExecutionProvider`, `CPUExecutionProvider` |
| `precision_policy` | `auto`; `auto`, `fp32`, `fp16` |
| `detector_model_id` | `scrfd`; `scrfd`, `retinaface`, `yoloface` |
| `swapper_model_id` | `inswapper_128`; pattern `^[a-z0-9][a-z0-9_]{0,95}$` |
| `dfm_morph_factor` | `.75`; `0..1` |
| `vram_limit_mb`, `vram_budget_mb`, `memory_safety_reserve_mb` | `0`, `0`, `768`; each `0..131072`, strict integer (`0` means provider/profile-derived budget for the first two) |
| `max_concurrent_jobs` | `1`; integer `1..8` |
| `face_restoration_policy` | `off`; `off`, `auto`, `gpen`, `gfpgan`, `codeformer`, `ultramax` |
| `video_timing_policy` | `preserve`; `preserve`, `cfr` |
| `audio_policy` | `copy`; `copy`, `transcode`, `drop` |
| `lip_sync_mode` | `off`; `off`, `wav2lip`, `wav2lip_gan_fp16`, `musetalk`, `edtalk` |
| `lip_sync_audio_path`, `lip_sync_audio_stream` | `null`; path max 1024 characters; stream `null` or integer ≥0 |
| `model_search_paths` | empty tuple; at most 32 paths, constrained to owned runtime roots |
| `output_root`, `cache_root` | `roop_3d/.runtime/outputs`, `roop_3d/.runtime/cache`; must remain within owned runtime |
| `remote_processing_enabled`, `synthetic_media_notice` | `false`, `true` |
| `diagnostics_level` | `info`; `error`, `warning`, `info`, `debug` |
| `render_backend` | `auto`; `auto`, `reference`, `stream` |
| `execution_threads`, `restorer_threads` | `1`, `1`; integer `1..8` |
| `stream_batch_size`, `stream_ring_slots`, `stream_segment_frames`, `sparse_interval`, `restorer_pool_size` | `4` (`1..8`), `4` (`2..32`), `250` (`1..10000`), `25` (`1..1000`), `2` (`1..32`); all strict integers |
| `enhancer_blend`, `restorer_fidelity` | `1.0`, `.5`; each `0..1` |
| `ultramax_native_bypass` | `false` |
| `ultramax_min_pixels` | `320`; integer `1..4096` |
| `host`, `port`, `max_body_bytes`, `dev_cors` | `127.0.0.1` (literal), `8030` (`1..65535`, port 5174 reserved), 2 MiB (`1..64 MiB`), `false` |

Only `ROOP3D_HOST`, `ROOP3D_PORT`, `ROOP3D_MAX_BODY_BYTES`, and `ROOP3D_DEV_CORS` are environment overrides in `Settings.from_env`; this is not a TensorRT INT8 configuration schema.

### Workstation preview parameters (distinct from `Settings`)

`components/workstation/pipeline.ts:WorkstationParameters` defines frame-preview/render payload fields such as `restoration_blend`, `restorer_model` (`gpen`, `codeformer`, `ultramax`, `gfpgan`), `fidelity_weight`, `color_method` (`lab`, `reinhard_lab`, `histogram`), `color_strength`, `feather_px`, `erosion_px`, `pose_strength`, `pose_mode` (`retarget`, `mesh_blend`), `pose_frontalization`, `tps_expression_weight`, `render_mode` (`standard`, `pose_3d`, `reenactment_3d`), `swap_scope`, `skull_blend`, `relight` / `relight_strength`, `occlusion_mode` (`standard`, `robust`), `occlusion_threshold`, manual light yaw/pitch and ambient fill, hairline offset, hair/neck/ear protection, jaw feather, skin texture retention, shutter angle, grain match, and per-face expression overrides. Optional fields are omitted until touched where payload compatibility requires it. This is not an interchangeable mirror of the Pydantic settings object. The registry also marks `fidelity_weight` as not always engine-active; a visible label alone must not imply the backend consumes it on every path.

## Preview/transport behavior and bottlenecks

- Slider preview lives in `src/livePreview.ts`: changes coalesce through a 150-ms debounce, the previous HTTP request is aborted, and the UI retains a short-lived CPU PNG cache (`previewFrameCache.ts`). `POST /api/preview` returns binary PNG by default; optional JSON mode can carry Base64. The UI does not send each slider preview through the binary preview WebSocket.
- `/ws/v1/preview/stream` is already a binary JPEG frame channel with a 16-byte little-endian header and latest-wins frame handling (`src/liveStream.ts`, `LiveStreamCanvas.tsx`, backend `server/live_stream.py`). It is separate from `/ws/v1/pipeline/status`, which streams JSON status/telemetry.
- The frame socket paints through an imperative canvas/ImageBitmap path and has a tested zero-React-render stream harness. `LiveStudio` uses a separate React state frame path. App telemetry is published to external atomic stores, and the 30-Hz render-isolation test covers selected workstation surfaces; there is no evidence that every high-frequency datum is a typed-array ring.
- `server/live_preview.py` has an LRU of **complete final frame images**, keyed by inspection, exact frame selector, and full configuration digest; capacity is bounded by entry count/bytes. Cache misses call the full live-preview pipeline. This is CPU memory, not a GPU-resident cache of detection, mesh/swap, and restoration intermediates.
- `server/workstation_preview.py` offers a richer single-frame workstation preview but may replay tracking/compositing from frame zero and has an admission/budget lock. The pipeline stages in status are progress phases, not reusable cached intermediate stages.
- Main canvas WebGL2 work and R3F 3D scenes execute on the page thread; no `transferControlToOffscreen()` worker renderer currently exists. Existing workers cover other tasks (e.g. image/mesh utilities), not both workstation canvases.
- No measured <16-ms slider-to-updated-frame guarantee, 60–120-FPS full application guarantee, GPU stage cache, telemetry typed-array ring, 10,000-item virtualized queue, or all-14-section in-workstation diagnostics panel was verified.

## Not present as a wired control

Do not add decorative-only controls for RetinaFace/SCRFD confidence and NMS thresholds, PnP switches, 68/106 landmark selection, TensorRT INT8, FFmpeg CRF/NVENC choices, quaternion One-Euro controls, Kalman covariance, optical-flow lock, scene-cut threshold, or named BiSeNet semantic-class masks unless the backend contract and selected pipeline consume them. The existing settings schema does not contain those fields. Diagnostics expose 14 report sections, but those sections are not equivalent to fourteen independently adjustable engine controls.

## Source pointers

- `app/roop_3d/core/settings.py`, `app/roop_3d/server/routes.py`
- `app/roop_3d/server/live_preview.py`, `app/roop_3d/server/workstation_preview.py`, `app/roop_3d/server/live_stream.py`, `app/roop_3d/server/ws.py`
- `app/roop_3d_ui/src/{App.tsx,SettingsInspector.tsx,livePreview.ts,liveStream.ts,previewFrameCache.ts}`
- `app/roop_3d_ui/src/components/{studio/StudioInspector.tsx,studio/WorkstationControls3D.tsx,studio/SpatialInspector.tsx,workstation/pipeline.ts,canvas/WorkstationCanvas.jsx}`
- [`02_ROOP_3D_COMPLETE_AUDIT.md`](02_ROOP_3D_COMPLETE_AUDIT.md) and [`_parts/R3_3_ui_tests_audit_docs.md`](_parts/R3_3_ui_tests_audit_docs.md)
