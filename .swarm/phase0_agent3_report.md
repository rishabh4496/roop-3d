# Phase 0 - Sub-Agent 3 report: UI + FastAPI schema pure-3D refactor

Lane: `app/roop_3d_ui/src/**` (non-test) and `app/roop_3d/server/**`. Nothing committed or pushed. Nothing outside the lane edited (`roop-ultimate`, `app/roop/`,
`core/*`, `jobs/*`, tests, e2e untouched). Cross-owner requests: `.swarm/phase0_agent3_crossowner_requests.md`.
The first run was interrupted by a usage limit; on resume `workstation_preview.py` was verified fully written (compiles, imports) and nothing was redone.

## New typed options (the only render controls left)

| Option | Values | Server (pydantic, `server/true3d.py`) | UI (`src/true3d.ts`, `components/studio/True3DControls.tsx`) |
|---|---|---|---|
| 3D Head Topology Engine | `FLAME-FullHead-3D` -> registry `flame2023_open`; `MakeHuman-CC0-Craniofacial-3D` -> `makehuman_cc0` (default) | `HeadTopologyEngine` literal; field `head_topology_engine`; availability from `reconstruction.topology.default_registry().describe()` (installed + hash-verified, reason text when not) | `HEAD_TOPOLOGY_ENGINES`, `WorkstationParameters.head_topology_engine`; select with unavailable options disabled + server reason (same pattern as the backbone dropdown) |
| 3D Photorealistic Shader Mode | `3D-PBR-Subsurface-Scattering + Neural-Deferred-UV` (default), `3D-Gaussian-Head-Avatar` | `ShaderMode` literal; field `shader_mode`; both reported unavailable while the renderer is not wired | `SHADER_MODES`, `WorkstationParameters.shader_mode` |
| Camera Intrinsics | `mode` auto (dynamic focal-length solver) / manual; `projection` is the single literal `"perspective"` (no orthographic / weak-perspective value exists); `focal_unit` px / fov_deg; `focal_length_px` 50..100000; `fov_deg` (horizontal) 5..170; `principal_point` normalised (cx/width, cy/height) | `CameraIntrinsicsSpec` (validated: manual needs the value its unit names); field `camera`; `POST /api/v1/true3d/camera/resolve` returns fx/fy/cx/cy/FOV (auto = flagged starting prior 55 deg) | `CameraIntrinsics` type, `WorkstationParameters.camera`; auto/manual cards, unit select, FOV slider or focal px input, principal point inputs, live K read-out |

Where the fields travel: `LivePreviewPayload` (`POST /api/preview`: flat `head_topology_engine`, `shader_mode`, `camera`), `PreviewRequest.true3d` (`/workstation/preview`),
`StudioConfig` (`/studio/start`), `PreFlightLintRequest.head_topology_engine` (preflight lint), unified state `true3d` (`/api/v1/state`, `/render-mode`),
`GET /api/v1/true3d/options|status`, `workstation.controls` of `/system/capabilities`, `/engines/readiness`, `/health.render_3d`. The UI sends them as part of the
workspace parameters (so projects, sessions and batch recipes carry them) and mirrors them to `/state`.
Forwarding into `WorkstationParameters` uses introspection (`true3d.forward_to`): a field the shared contract (Owner A) does not declare yet is not dropped silently, it is listed in
`meta.unsupported` as `contract:<name>`.

## Removed (server)
- `live_preview.py`: `swapper_model_id`, `restorer`, `restoration_blend`, `fidelity_weight`, `frontalization_weight`, `pose_frontalization`, `tps_expression_weight`, `pose_strength`,
  `render_mode`, `ultramax_*`, solo nodes `swapper|restorer`, bypass `restorer`, `RestorerName`, `validate_restorer`, `pose_mode` mapping, restorer telemetry, the `2D-swap` HUD path.
- `workstation_preview.py`: rewritten. The MasterPipelineRunner / ReferencePipeline / GpenRestorer / reblend / crop-warp-paste preview is gone; `PreviewService.render` calls the
  connected renderer hook or raises `RENDERER_NOT_WIRED`. Sweep axes now `feather_px|color_strength|relight_strength|specular_sheen`.
- `live_studio.py`: swapper session, `dfm_morph_factor`, `render_mode`; session comes from `renderer.open_live_session`.
- `routes.py`: swapper readiness / governance, UltraMax spec, `face_restoration_policy` + `swapper_model_id` capability controls, `pose_mesh_blend` feature, manifest aliases for the 2D models;
  `/engines/readiness` now reports renderer + topology + shader readiness; `_selected_models` returns detector + 3D choices; `/pipeline/status` precision loop is stage-agnostic.
- `diagnostics.py`: frontalization, UltraMax crop-bypass and UltraMax-on-4K checks, swapper dynamic-shape role, fabricated `inswapper_ms: 9.8`, "Identity Transfer & Inswapper 128" and
  "Face Restoration Adapters" sections (replaced by honest "3D Renderer Readiness" / "Photometric & PBR Shading"); new `renderer_wired` and `head_topology_available` preflight checks.
- `unified_state.py`: `RenderMode` 3-mode taxonomy replaced by the single `full_3d_head`; `face_swapper` processor replaced by `head_reconstruction_3d`.
- `queue_routes.py` (`restorer`), `identity_assets.py` (`canonical` 2.5D template -> always `makehuman`), `workstation.py` (`ESTIMATED_2_5D`, `tps_2_5d` receipts), `contracts.py`
  (`Health.swap_engines` -> `render_3d`), `model_runtime.py` (loader no longer from the swapper engine), `app.py` (no swap-engine scan; guarded optional imports; `true3d` router).

## Removed / replaced (UI)
- Deleted features: swapper rack (14 swapper ids), enhancer/restorer rack + fidelity slider, "Face restoration" workflow step, Restoration model selects (App, StudioInspector), render-mode
  cards (Standard 2D / 3D Facial Warp / 3D Expression Transfer), 3D Influence Mode + Jaw Reshape Weight (`pose_mode`/`pose_strength`), "Pose correction method", pose frontalization /
  TPS / restorer blend / fidelity settings and curves, Small-face UltraMax, UltraMax context-menu toggle, "Estimated 2.5D" viewer tab and landmark overlay, Dual-Endpoint (flat vs 3D) blend slider,
  enhancer contact-sheet axis, restorer pipeline node (replaced by "3D Head Render" + "Colour Match" nodes), inswapper/restorer stage-timing rows, batch presets using GPEN/CodeFormer/UltraMax/TPS.
- Added: `true3d.ts`, `True3DControls.tsx` (used in Studio Inspector step 1, Engine & Model Rack, Live Studio), renderer status banner (`data-testid="renderer-status"`), batch presets
  for MakeHuman/FLAME/Gaussian/wide-angle/telephoto, camera-FOV sweep axis, `RENDERER_NOT_WIRED` error-guide entry.

## Persisted-settings / saved-data migration (old data loads without crashing)
- Server: `true3d.migrate_legacy` drops removed keys (and `solo_node=swapper|restorer`, `bypass=restorer`, non-perspective camera projection) in a `mode="before"` validator on
  `LivePreviewPayload`, `PreviewRequest` (also nested `parameters`, `request`, `request.workstation_parameters`, removed sweep axes), `StudioConfig`, `BatchGenerateRequest`; dropped names are returned in
  `legacy_dropped` and surfaced in `meta.unsupported` as `legacy:<name>`. `unified_state`: legacy render modes (`standard_2d`, `estimated_2_5d`, `standard`, `pose_3d`, `reenactment_3d`,
  `full_3d_reconstructed`) normalise to `full_3d_head` and are listed in `migrated_from`; legacy processors dropped; invalid true3d values fall back to defaults (`migrate_true3d`).
  Not done (Owner A, see requests D/B/C): `Settings` / `WorkstationParameters` / `JobRequest` are `extra="forbid"`, so the stored `config.json` and job/project checkpoints need the same drop there.
- UI: `migrateParameters` (App: project open, job open, session restore), `stripRemovedParameters` (session snapshot), `migrateCurves` (project/job/session curves), rack store drops the
  stored swapper and strips batch parameters, `migrateCamera` / `migrateTrue3D` fall back to defaults.

## Honest readiness
- `GET /api/v1/true3d/options|status`, `/health.render_3d`, `/engines/readiness`, `/system/capabilities.workstation.renderer`, preflight lint and the Studio/Live Studio UI all report
  `wired: false` / "3D renderer not wired" while `roop_3d.pipeline.true3d_renderer` is absent. Preview / grid / seq / studio start / workstation preview return HTTP 503 `RENDERER_NOT_WIRED`; the
  live-preview hook shows the message as an error with Retry, the Queue button toasts and refuses, Live Studio disables Start and prints the reason. No fallback engine is substituted.
- Smoke-tested in-process (no pytest): legacy payload validates and reports `legacy_dropped`, `renderer_status().message == "3D renderer not wired"`, `/render-mode` migration, `/state` migration,
  camera resolve (`fov 60` -> fx 1662.8 px at 1920 wide), orthographic projection value rejected by `CameraIntrinsicsSpec`.

## Verification
- `python -m compileall -q app/roop_3d/server` (venv python): OK. All server modules (`app`, `routes`, `live_preview`, `workstation_preview`, ...) import.
- `tsc --noEmit` (full project): 0 errors in non-test sources. 90 errors, all in 21 `src/*.test.ts(x)` files owned by Sub-Agent 4 (settingsAndSweep 21, finalWorkstation 14, workflowPanel 10,
  stage7Studio 6, coverageRadar 6, activeModelReadback 5, modelRack 4, liveSwapPreview 4, swapScope 3, faceset3dViewer 3, hairNeckProtection 3, compositingControls 2, and 1 each in
  stage8Workflow, workstationControls3D, workstation3dPayload, stage8Studio, readinessChecks, faceContextMenu, expressionEditor, engineStatusBar, stage23Ui). So `npm run build`
  (typecheck + lint + vite) fails on the typecheck step until those tests are updated.
- `eslint src --ignore-pattern "**/*.test.*"`: 0 errors, 0 warnings (one pre-existing exhaustive-deps warning in VideoViewport was fixed as part of removing `parameters`/`setParameters` deps).
- `vite build`: OK (1216 modules, zero warnings); `app/roop_3d_ui/dist/` (git-ignored) regenerated so the served bundle no longer carries the old strings.
- NOT run (by instruction): pytest, vitest, playwright. NOT visually checked in a browser.

## Honest leftovers
1. The renderer does not exist: everything above is the honest "not wired" surface plus the typed controls; no pixel was rendered by this work.
2. `Settings` / `WorkstationParameters` / `JobRequest` still declare the removed fields until Owner A edits them; the server layer tolerates both states but a stale `config.json` cannot load until Owner A migrates it.
3. UI still contains: `STAGE_3_MAIN_SWAP` naming (ws.py contract), `ModelSummary.adapter_*` fields, `HeadModelBackbone` default `3ddfa_v3_hrn` (manifest decision 2), LivePortrait-dependent
   copy in `WorkstationControls3D` (not reviewed line by line; manifest decision 1), `FaceSetDetail`/`WorkstationControls3D` text not grepped for 2D wording beyond the banned-id list.
4. Banned-identifier grep over my lane (non-test) now only matches the intentional migration allow-lists (`true3d.py`, `true3d.ts`, `unified_state.py`, `workstation_preview.py`), comments that say
   "removed", and the `swapper_engine` fallback import path in `model_runtime.py` (goes away when Owner A extracts the loader).
5. `ENGINE`-level naming (`reference_onnx`, `photoreal_3d_ffmpeg`, "Render engine" select) is unchanged; which of them is a true 3D engine is Owner A's decision.
6. e2e specs, `mock-server/`, `acceptance/` still reference the old fields (Sub-Agent 4).
