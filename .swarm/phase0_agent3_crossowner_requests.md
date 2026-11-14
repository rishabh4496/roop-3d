# Phase 0 - Sub-Agent 3 cross-owner requests (Owner A = backend, Owner C = tests)

Sub-Agent 3 did not edit anything outside `app/roop_3d/server/**` and `app/roop_3d_ui/src/**` (non-test). The server layer was written to
tolerate each item below being absent or present; every item says what the server layer does until it lands.

## A. New module - the true-3D renderer hook (REQUIRED, otherwise every render entry point answers 503 RENDERER_NOT_WIRED)

Path: `app/roop_3d/pipeline/true3d_renderer.py` (module name is `RENDERER_MODULE` in `server/true3d.py`).

| Symbol | Signature | Used by |
|---|---|---|
| `is_wired()` | `-> bool`; True only when a real full-head render path (topology engine + perspective camera + shader) can produce a frame | `server/true3d.py renderer_status()`, `/health`, `/api/v1/true3d/*`, preflight lint, UI banner |
| `describe()` (optional) | `-> dict` merged into the renderer status when wired | `/true3d/options` |
| `render_preview(state, request, cancel, *, raw, final_only)` | `request` is `server.workstation_preview.PreviewRequest` (has `.true3d` = `True3DSettings`, `.request` = `JobRequest`, `.effective_parameters`); returns `{frame_index, timestamp, width, height, profile, provider, notes, faces, buffers{final, detector, pose, source, matte, color, normal_map, sh_lighting, expression}, latency{total_ms, detection_ms, tracking_ms, projection_ms, render_ms}, parameters}` | `PreviewService.render` -> `/workstation/preview`, `/api/preview`, grid, seq |
| `open_live_session(state, config, true3d, admission)` | returns an object with `.render(frame_bgr) -> frame_bgr` and `.close()`; `config` is `live_studio.StudioConfig` | `/studio/start` |

Until it exists: `renderer_module()` is None, `LivePreviewService.frames()` raises `DomainError("RENDERER_NOT_WIRED", "3D renderer not wired", 503)` for any
uncached frame, `/workstation/preview` and `/studio/start` do the same, `/health.render_3d.renderer.wired == false`, `/engines/readiness` reports no READY engine.

## B. `core/workstation.py` (`WorkstationParameters`) - shared pydantic contract
- ADD (names exactly): `head_topology_engine: Literal['FLAME-FullHead-3D','MakeHuman-CC0-Craniofacial-3D'] = 'MakeHuman-CC0-Craniofacial-3D'`,
  `shader_mode: Literal['3D-PBR-Subsurface-Scattering + Neural-Deferred-UV','3D-Gaussian-Head-Avatar'] = <first>`,
  `camera_intrinsics: dict | CameraIntrinsicsSpec` (the server forwards `True3DSettings.camera.model_dump()` under this name; same shape as
  `server/true3d.py CameraIntrinsicsSpec`: mode auto|manual, projection const "perspective", focal_unit px|fov_deg, focal_length_px, fov_deg, principal_point).
  The server only forwards a field if `WorkstationParameters.model_fields` declares it; otherwise it is reported in the frame's `meta.unsupported` as `contract:<name>`.
- REMOVE: `restorer_model`, `fidelity_weight`, `restoration_blend`, `pose_strength`, `pose_mode`, `pose_frontalization`, `tps_expression_weight`, `render_mode`,
  `ultramax_small_face_mode`, `ultramax_min_native_pixels`, `jaw_adaptation`, and `'restorer'` from `bypass` (UI + server now send bypass only `['color']`).
  Add a before-validator that DROPS those keys from stored jobs/projects (`server.true3d.migrate_legacy` is importable for this) - `app/jobs/*/checkpoint.json`
  (23 files) carry them and `Contract` is `extra="forbid"`.

## C. `jobs/schema.py` (`JobRequest`)
- REMOVE `swapper_model_id`; keep `detector_model_id`; same legacy-drop validator for stored requests. `server/live_preview.py` no longer passes `swapper_model_id`.

## D. `core/settings.py` (`Settings`) + persisted `.runtime/config.json`
- REMOVE `swapper_model_id`, `dfm_morph_factor`, `face_restoration_policy`, `enhancer_blend`, `restorer_fidelity`, `restorer_threads`, `restorer_pool_size`,
  `ultramax_native_bypass`, `ultramax_min_pixels`. `Settings` is `extra="forbid"`, so `ConfigStore.load` / `migrate_config` MUST drop these keys from an existing
  config.json or an upgraded install fails to start (the UI also strips them, but the server loads the file first).
- UI `SettingsInspector` renders whatever the Settings JSON schema exposes; no UI change needed after removal.

## E. `queue/batch_matrix.py`
- `BUILTIN_RECIPES` / `restorer` argument / `{restorer}` output-name token / `_RESTORER_FIELD`. The server route (`server/queue_routes.py`) no longer has a `restorer`
  field and drops legacy `restorer` keys; the UI recipes are now true-3D (topology + shader + camera). `generate_per_file_matrix(restorer="gpen")` default would
  still inject a restorer - remove it. `GET /batch/recipes` still returns `BUILTIN_RECIPES` (UI has its own copy in `BatchMatrixScreen.jsx`).

## F. `engine/active_models.py`, `models/catalog.py`, `models/hub.py`, `models/engine_readiness.py`
- `active_models.snapshot()` is now consumed generically (`{stage: model|None}`); stage names are free-form (UI knows `detector`, `head`, `shader`, else prints the key).
  `server/routes.py` no longer calls `catalog.swapper_readiness()`, `load_ultramax_spec`, `swap_engine_report` or `engine_readiness.swap_engine_summary`.
- Model categories: UI Model Manager now groups `head | detector | occluder | other` (was swapper/enhancer/...). `ModelSummary.adapter_supported /
  adapter_unsupported_reason` in `server/contracts.py` were kept only because the catalog still emits them; delete them together with the swapper adapter rows.
- Registry/hub rows for the 2D swapper/restorer weights must go (Owner A table A-S9..A-S17); the server `_resolve_manifest_entry_id` aliases for them are already gone.

## G. Naming leftovers in A-owned telemetry (server/ws.py is mine but only forwards them)
- `STAGE_3_MAIN_SWAP` stage id (`server/pipeline_status.py`, `server/ws.py`) and `core/stage_timing.py` stage `inswapper`. UI now reads `render_ms` / `head_render_ms`
  (latency keys) and the diagnostics stage ids `head_render`, `head_pose`. Renaming `STAGE_3_MAIN_SWAP` touches tests and the WebSocket schema, so I left it.
- `tracking/telemetry.py`: pose source `weak_perspective` -> `perspective_mesh` (UI label added for both).

## H. Reconstruction payloads
- `reconstruction/faceset_3d.py reconstruct_faceset`: stop emitting `mesh_2_5d` (UI no longer reads it). `/reconstructions/{id}` receipt method `tps_2_5d` is now rejected
  (`server/workstation.py` accepts `flame` | `identity_asset`).
- `reconstruction/identity_asset.py`: `IdentityAssetSettings(template=...)` - `canonical` removal. `server/identity_assets.py` now always passes `template="makehuman"`
  and treats `?template=canonical` as an alias.
- `reconstruction/backbone_status.py` default backbone (`3ddfa_v3_hrn`) is still what `HeadModelBackbone.tsx` shows; see manifest section 8 item 2.

## I. Entry points that may disappear
- `server/app.py` now guards `from roop_3d.optimized_processor import make_stream_runner` and `pipeline.video_job_runner` with `ImportError` (worker then builds its own
  reference runner). `server/model_runtime.py` looks for `load_local_session` in `core.ort_session`, `models.session`, then the old `engine.volumetric.swapper_engine`;
  please extract it to `core/ort_session.py` (manifest A-S3) and I will drop the legacy fallback.

## J. Owner C (tests) - things that now fail by design
- Python: any test that posts `restorer`/`swapper_model_id`/`render_mode` to `/api/preview`, reads `health.swap_engines`, `/engines/readiness` swapper rows,
  `RenderMode.STANDARD_2D`/`engine_render_mode` from `/render-mode`, `PreviewRequest` axes `restoration_blend|pose_strength`, `diagnostics` sections 9/11 titles,
  `POST /facesets/.../identity-asset` default `template=canonical`.
- UI: ~90 `tsc` errors, all in 21 `*.test.tsx` files (list in `phase0_agent3_report.md`); e2e specs / mockApi still use `swapper_model_id` and `restoration_blend`.
