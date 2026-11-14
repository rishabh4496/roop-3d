# Phase 0 - Agent 2 cross-owner requests

Agent 2 edited nothing under `app/roop_3d/server/**`, `app/roop_3d/tests/**`, `docs`, `app/roop_3d_ui/**`, `app/roop/**`, `audit/protected_paths*.json`.
At the end of the purge `import roop_3d.server.*` and every other non-test module import cleanly (see report), so the server needs no
edit to *import*. Items below are things the owners must still align with.

## Agent 3 (server routes / schemas)

| Path | Symbol | Why |
|---|---|---|
| `server/*` (any reader of `/health`, `/engines/readiness`) | `roop_3d.models.engine_readiness.render_engine_summary / render_engine_report` (aliases `swap_engine_summary` / `swap_engine_report` kept) | Now returns ONE row, model_id `true3d_head_renderer`, `NONE_AVAILABLE` / `RENDERER_NOT_WIRED` / message `3D renderer not wired` until `pipeline.true3d_renderer.is_wired()` is True. |
| `server/true3d.py` | hook module `roop_3d.pipeline.true3d_renderer` | Provided by Agent 2: `is_wired() -> False`, `describe()`, `render_preview(...)`, `open_live_session(...)`, `render_frame(...)` (the last three raise `True3DRendererUnavailable`, code `RENDERER_NOT_WIRED`, status 503). Same code/message as `server/true3d.NOT_WIRED_*`. |
| `server/live_preview.py`, `workstation_preview.py` | `WorkstationParameters` | Removed fields: `restorer_model, fidelity_weight, restoration_blend, pose_strength, pose_mode, pose_frontalization, tps_expression_weight, render_mode, jaw_adaptation, ultramax_small_face_mode, ultramax_min_native_pixels`; `bypass` now only allows `'color'`. Added optional `head_topology_engine`, `shader_mode`, `camera_intrinsics` (dict) (what `true3d.forward_to` copies). The model drops the removed keys on input (`REMOVED_FIELDS`), so saved jobs still load. |
| any code reading `Settings` | removed fields | `swapper_model_id, dfm_morph_factor, face_restoration_policy, restorer_threads, restorer_pool_size, enhancer_blend, restorer_fidelity, ultramax_native_bypass, ultramax_min_pixels` (old saved config keys are ignored by a before-validator). `render_policy.RenderPolicy` lost `ultramax_*` too. |
| `server/*` readers of job `JobRequest` | `JobRequest.swapper_model_id` | Field removed (dropped on input). |
| `server/*` readers of `/models/active` | `engine.active_models.STAGES` | now `("detector", "enhancer")` (no `"swapper"` stage). |
| `server/routes.py` model rack | `ModelCatalog.swapper_*`, `ultramax_spec`, `.dfm` category, `supported_models.json` swapper/enhancer rows | All removed; rows are plain `custom`/detector/occluder/upscaler now. `catalog._adapter_support` / `_selection` kept as constant shims. |
| `server/orchestrator.py` | orchestrator manifests | Categories `identity_swapping`/`restoration` -> `head_rendering`; data kinds `swapped_crop`/`restored_crop` -> `rendered_head_layer`; new engine `true3d_head_renderer` (adapter intentionally NOT registered in `orchestrator/registry.ADAPTERS`, so it reports `ADAPTER_NOT_REGISTERED` until wired); `content_version` 2026.11.14-1. |
| `server/diagnostics.py` | `core.stage_timing.STAGES` | Stage keys renamed `pose_tps->head_pose`, `inswapper->head_render`, `restorer->expression` (`("decode","detection","head_pose","head_render","masking","expression","encode")`). |
| `server/*` | `compositor.pipeline.CompositeOptions` | Removed fields `restoration_blend, roi_resolution_limit, fidelity_weight, ultramax_small_face_mode, ultramax_min_native_pixels`. |
| `server/*` | `queue.batch_matrix` | `restorer` -> `shader` (`restorer_model` -> `shader_mode`), default template `{target}_{source}_{shader}_{timestamp}.mp4`; recipes replaced by `natural_3d / relit_3d / soft_blend_3d`. |
| `server/engine_gate / pipeline_status` | `ReferencePipeline.capability()` | `identity_swapper`, `pose_projection`, `render_modes.identity_assets`, `hybrid_mesh_renderer`, `dense_face` keys are gone; new `renderer: {wired, message}` and `render_modes: {available: ["true_3d"]}`. |

## UI owner (`app/roop_3d_ui/**`)

* Pipeline stage ids in `/api/v1/pipeline` progress events changed (see `STAGES` above) - `stageTimings.js`, `BottleneckFlamegraph.jsx`, `lanesModel.js`, TUI columns.
* Compositor observer node names: `swapper` -> `head_layer`, `restorer` -> `render`.
* Batch matrix recipe ids / `shader` field.

## Tests owner (Agent 4) - modules removed or moved

* Deleted modules: `swappers/*`, `engine/volumetric/identity_swapper.py`, `engine/volumetric/swapper_engine.py` (generic ONNX loader moved to `core/local_session.py`: `open_local_session`, `load_local_session`), `pose_matrix.py`, `models/emap_cache.py`, `optimized_processor.make_stream_runner` body (now refuses), `vectorized_pipeline.py`, `gpu_compositor.py`, `gpu_transforms.py`, `cvcuda_ops.py`, `hires_handoff.py`, `compositor/restoration.py`, `compositor/stream_restoration.py` (`MaskSeamWatchdog` moved to `compositor/mask_seam_watchdog.py`), `enhancers/{codeformer,gpen_realistic,ultramax,styleganex_age}.py`, `engine/stage16_pipeline.py`, `reconstruction/{pose_blend,jaw_fit,expression_tps,frontalization,tps_warp}.py` (landmark block layout moved to `reconstruction/landmark_layout.py`; CPU perspective rasterizer to `rendering/cpu_perspective_raster.py`), `rendering/super_res_tiles.py`, `run_benchmarks.py`, `audit/stage2_profiler.py`, `audit/stage9_benchmarks.py`, `tools/{benchmark_compositor,verify_stream_hardware,verify_stream_integration}.py` (`create_clip` moved to `tools/synthetic_clip.py`).
* Deleted symbols: `WeakPerspectiveCamera` (-> `PerspectiveCamera`), `build_canonical_head_mesh`, `ThinPlateSplineWarper`, `warp_delaunay_piecewise_affine`, `SyntheticFLAMEModel`, `reenactment.IdentityRig` (-> `reconstruction/head_rig.py::HeadRig`, MakeHuman only) and `reenactment.ReenactmentEngine/ReenactmentFrame/RENDER_MODES`, `rendering.vertex_raster.orthographic_pixels` (-> `perspective_pixels`), `pipeline.image_ops.tps_flow`, `engine.volumetric.session.{Replacement.projection/.reenactment/.restored, swapped_portrait, profile_aware_mask, select_swapper}`, `ModelManifestEntry` rows for swappers/restorers/3ddfa_v2, `FacePacket.swapped_crop` (-> `rendered_crop`).
* Fixtures `identity_world.py`, `quality_world.py`, `model_governance_support.py`, `swap_core_support.py` still import removed symbols.
* New: `core/true3d_guard.py` (API + exceptions), `enhancers/uv_detailer.py::UVTextureDetailer`, `pipeline/true3d_renderer.py`.
* Behaviour change to expect in tests: `core.pose.estimate_pose_5pt` is now a perspective PnP fit (round trip with `project_model5` < 1.2 deg); yaw/pitch of non-synthetic landmarks differ by a few degrees from the old scaled-orthographic fit.
