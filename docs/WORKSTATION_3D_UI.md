# 3D-native workstation UI

Frontend: `app/roop_3d_ui` (React 19 + Three/R3F, offline: fonts and assets are bundled, the only network is the loopback API).
Everything below is built on the existing studio layout, the Three.js viewer and the WebGL2 compare canvas; nothing was rewritten.

## What each control is wired to

| Control | Where | Real engine behind it |
|---|---|---|
| **Pipeline node flow bar** (8 stages, hotkeys `1`–`8`, one-click solo) | above the canvas, always visible (`PipelineNodeFlowBar.jsx`, `pipeline.ts`) | `POST /api/preview` `solo_node`. Latency per stage is the preview's measured `latency` (`detection_ms`, `retarget_ms`, `projection_ms`, `swap_ms`, `total_ms`); a stage the engine does not time shows `—`, never an estimate |
| Stage 3 *Canonical UV Projection* | node 3 | the `source` buffer: the pose-matched canonical source view the identity is projected from. The engine does not export the raw UV atlas as a preview buffer |
| Stage 4 *SH Relighting & Normals* | node 4 + a **Normals** chip | `sh_lighting` buffer; the chip solos `normal_map` |
| Stage 6 *UltraMax High-Res*, stage 7 *XSeg + Hair Occlusion Matte* | nodes 6, 7 | `restorer` and `matte` buffers (XSeg + hair matte is in the buffer when *Occlusion handling* is Robust) |
| **Angular Coverage Radar** | Face Manager sphere panel, studio inspector (Step 1), spatial inspector → *Coverage* tab (`AngularCoverageRadar.tsx`, `utils/coverageRadar.ts`) | the FaceSet's own views on the backend's 12 × 8 grid (yaw −90…+90, pitch −60…+60, 15° bins). Missing profile bands are listed; target poses (track pose range + faces on screen) the FaceSet has no nearby view for are marked red |
| **Jawline / Cranial Volume Blend** | *Identity morph* panel | `pose_strength` in `mesh_blend` mode: target 2.5D proxy (0%) ↔ source 3D mesh (100%). It blends the whole head volume (jaw and cranium together) |
| **Forehead & Hairline Boundary Offset** | *Identity morph* panel | new `hairline_offset_px` (−32…32): grows / pulls in the matte only inside the forehead band (`compositor/hairline.py`); preview and render |
| **SH light probe** key light, ambient fill, specular sheen | Light probe (spatial inspector tab, studio *Colour & lighting*) | `manual_light_yaw/pitch`, **new** `ambient_fill` (the SH band-0 term of the manual light), `specular_sheen`, `relight_strength`. The 3D head in the inspector is lit by the same controls |
| **3D head + UV albedo** | FaceSet panel → *Head model*: Canonical / **MakeHuman CC0** | `GET /facesets/{id}/identity-asset/mesh?template=makehuman` (built from the panel's *Build* button; never built implicitly) |
| **6DoF wireframe, gimbal, gaze** | canvas HUD buttons **Mesh** and **Gaze** (`headMesh.js`) | each face's measured yaw / pitch / roll; the MakeHuman topology when the active FaceSet has one, else the canonical head (the tooltip says which) |
| **Face right-click menu** | canvas | bind a named `.fsz` to the face, UltraMax, lock 6DoF pose, solo mask (see limits below) |
| **Mask brush** | Edit Mask | size, feather (hardness), opacity, include / exclude / erase, and Undo / Redo buttons on the workspace history (Ctrl+Z / Ctrl+Y) |

`ambient_fill` and `hairline_offset_px` are opt-in on the wire: omitted until the user touches them, so earlier requests keep their exact
serialisation, job idempotency fingerprint and preview cache key (pinned by `tests/test_workstation3d_controls.py`).

## 3D head model, influence mode and hair & neck protection

Added on top of the controls above. Every control writes a real engine parameter; nothing is a placeholder.

| Control | Where | Writes | Engine behaviour |
| --- | --- | --- | --- |
| **3D Head Model Backbone** | Step 1 → *3D head model & influence* | browser preference `roop3d-head-backbone` | Options and availability come from `GET /api/v1/reconstruction/backbones` (`reconstruction/backbone_status.py`, read-only, loads nothing). The server loads exactly **one** dense-face bundle, so the choice does not swap bundles: a family is selectable only if its bundle is installed. 3DDFA-V3 + HRN and FLAME 2023 Open + SMIRK are the two `topology` values of `roop_3d/.runtime/models/dense_face/bundle.json`; FaceVerse V4 has no loader and is always unavailable; MakeHuman is selectable only once built and is inspection / export only |
| **3D Influence Mode** | same panel | `pose_mode` | *3D Pose-Guided Neural Swap* = `retarget`; *3D Geometry + Jaw Reshape Weight* = `mesh_blend` |
| **Jaw Reshape Weight** | same panel | `pose_strength` | the former "Jawline / Cranial Volume Blend"; only used in geometry mode |
| **Strict Neck & Throat Protection** | Step 2 → *Hair & neck protection* | `neck_protection` | off drops BiSeNet neck / neck_l / cloth from the forbidden set and skips the jaw clamp |
| **Hair-Strand Guided Matting** | same | `hair_matting` | off replaces the guided-filter hairline matte with a Gaussian feather |
| **Preserve Target Ears & Eyeglasses** | same | `preserve_ears_glasses` | off drops eye_g / ear classes from the forbidden set and from the semantic occluders |
| **Forehead Hairline Inset** (−20…+20 px) | same | `hairline_offset_px`, **sign flipped** | + pulls the swap down off the hair; the engine value is + = grow upward. It moved here from the morph panel |
| **Jaw-to-Neck Feather Falloff** (2…24 px) | same | `jaw_feather_px` | inward falloff of the jaw clamp (default 6) |
| **Real Skin Micro-Texture & Pore Retention** (0…100 %) | same | `skin_texture_retention` | share of the target's high-pass skin residual re-injected after UltraMax (default 25 %) |

All five new parameters are opt-in on the wire and removed again when set back to their default, so untouched requests keep their exact
serialisation, job fingerprint and preview cache key (`tests/test_hair_neck_protection_controls.py`, `src/hairNeckProtection.test.tsx`).

Scope, stated in the UI too: the jaw clamp, hairline matting and jaw feather shape the **3D-projected matte** (a 3D render mode, or Robust
occlusion). The plain 2D swap uses the face-parser matte only, so there they are inert. Pore retention applies to the UltraMax enhancer only. In
Robust occlusion the Smart Occluder still keeps glasses in front regardless of *Preserve Ears & Eyeglasses*.

## Limits worth knowing

- **Wireframe registration is approximate.** The mesh is anchored on the detected nose tip and sized from the detector box (head ≈ 1.4 × box width). It shows head orientation and topology, not a per-vertex photometric fit.
- **Gaze arrows are head-gaze**, the direction the head faces from each eye. The pipeline has no iris tracking.
- **UltraMax from the menu is global.** The render engine applies one enhancer to every swapped face, so the menu item switches UltraMax on (or off) for the whole swap and says so.
- **Pose lock and mask solo are canvas-only.** *Lock 6DoF pose filter* holds that face's wireframe / gimbal / gaze at its current pose; *Solo mask* dims the other faces and opens the matte view. Neither changes the render.
- **Ambient fill applies to the manual light.** The auto light is estimated from the target frame, so the slider is disabled in auto mode.
- **`#/extras` (Editor) is unchanged**: its media tools (LUT, grain, VFX export) have no 3D pose, coverage or buffer data to show.
- The dense MakeHuman topology needs the CC0 head built locally (`reconstruction/makehuman_build.py`); without it the overlay falls back to the canonical head.

## Verify

```
cd app/roop_3d_ui
npm run build                                   # tsc + eslint + vite (zero errors)
npx vitest run                                  # unit tests
npx playwright test e2e/workstation3d.e2e.ts    # real Chrome: flow bar, radar, morph, mesh overlay, menu
cd ..   &&   roop_3d/.venv/Scripts/python.exe -m pytest roop_3d/tests/test_workstation3d_controls.py
```

Pitch conventions: stored / detected-face poses are *legacy* (+ = down); the radar, 3D head, wireframe and rotation maths are *standard* (+ = up).
Conversion happens once, in `targetPoseSamples`, `CoverageRadarPanel` and `facePose`. The wireframe rotation is checked against the 3D inspector's
`poseQuaternion` for yaw, pitch and roll (`headMesh.test.ts`).
