# Stage 6: simplified 3-zone workstation, connected render pipeline, full-app verification

Real paths: the prompt's `app/roop/` + `roop_3d/` is `app/roop_3d/` (backend) and its React app is `app/roop_3d_ui/`.
Render-mode names map to the saved `render_mode` values: `standard_2d` = `standard`, `estimated_2_5d` = `pose_3d`,
`full_3d_reconstructed` = `reenactment_3d`.

## UI (studio layout, `app/roop_3d_ui/src`)

| Zone | What it holds | Where |
|---|---|---|
| Left: inputs | 01 Source face / FaceSet (library, `.fsz` import, views + coverage), 02 Target media (import, frame navigator, detected faces), 03 Target face mapping (`Source → Target` rows with the angle count of each source) | `App.tsx` studio block, `components/studio/FaceMapping.tsx` |
| Centre | Unchanged: face boxes, split-wipe compare, zoom/pan, filmstrip timeline, `⚡ Live Frame Swap Preview` | `VideoViewport.tsx` |
| Right: pipeline | Step 1 mode cards + live 3D FaceSet viewport (Observed Views, Sphere Coverage X/96, Live Pose); Step 2 masking; Step 3 enhancer + blend; footer | `components/studio/StudioInspector.tsx`, `RenderFooter.tsx` |

* **Step 2** has BiSeNet (always on, shown locked), one `XSeg mask + Smart Occluder` toggle and one `Mask feather / blur` slider.
  The two networks are one control on purpose: they are the compositor's `occlusion_mode: robust`, which runs XSeg-3, the face
  occluder and BiSeNet together; they cannot be switched independently, so the UI does not pretend they can.
* **Step 3** enhancer: None / GPEN Realistic 512 / UltraMax High-Detail / CodeFormer, plus a 0–100 % blend. "None" is a blend
  of 0; picking an enhancer from None starts at 70 %.
* **Footer**: `Render current frame` (asks the viewport to show the live swap through `RENDER_FRAME_EVENT`), `Start full render`
  (the same queue path as the header button), a progress bar and fps from the job, and a GPU/VRAM pill from the telemetry
  stream. The instant output preview is the centre canvas's live swap (not duplicated in the footer).
* **Advanced Diagnostics** (closed by default) holds everything else: target identity facts/role, engine hub, restoration
  model + fidelity, colour transfer, relighting + light probe, the viewport's pose/expression/masking/restorer parameter
  panels, and Processing settings.

## Backend

Before: `ultramax` was refused by the preview (`RESTORER_UNAVAILABLE`) and silently mapped to GPEN for jobs, so choosing it in
the UI could not do what it said. Now `UltraMaxPipelineRestorer` (`compositor/restoration.py`) runs the verified CodeFormer
weights at the 512 px training alignment, then `laplacian_pyramid_fusion` keeps the swap crop's colour and low/mid
frequencies and takes only the finest detail. It is wired into the preview (`workstation_preview.py`), the job runner
(`pipeline_runner.py`), `WorkstationParameters.restorer_model` and the live-preview payload. `gfpgan` is still refused: it has no adapter.

The per-frame order is the existing `ReferencePipeline.infer_frame` → `FrameCompositor.compose` (detect/track → pose-matched
source view → inswapper → 2.5D/3D projection → robust matte → restorer → colour/relight → blend) → `encode_video`
(NVENC when usable, `libx264` otherwise, original audio copied). Live preview and render share it.

## Verification

* `verify_full_app_e2e.py` (real models, GPU, real clip): backend boot + health; 3D reconstruction with 96-bin sphere
  coverage and both meshes; face boxes from an extracted frame; 3 modes × {GPEN Realistic, UltraMax} with the robust XSeg matte,
  each asserting the composite changed, XSeg is in the compositor capability and the restorer actually ran; then a muxed video
  render. Exit code 0 only if every check passes. Outputs: `app/roop_3d/.runtime/e2e_verify/`.
* `npm run build` (`tsc` → `eslint` → `vite build`): exits 0 with **no warnings**. The 88 pre-existing lint warnings were
  mostly false positives (core `no-unused-vars` cannot see `<Canvas />`-style JSX uses; fixed with a local rule in
  `eslint.config.js`) plus real dead variables (removed) and four intentional hook-dependency omissions (annotated with the
  reason). The 63 bundler notices were framer-motion's `"use client"` directive, filtered by code in `vite.config.ts`.
* Tests: `src/workflowPanel.test.tsx`, additions to `stage7Studio.test.tsx`, `roop_3d/tests/test_stage06_ultramax_restorer.py`;
  `e2e/studio.e2e.ts` updated for the radio mode picker and the drawer.
