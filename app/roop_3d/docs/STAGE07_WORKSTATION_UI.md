# Stage 7: Professional workstation UI (Studio layout)

**Document path:** `app/roop_3d/docs/STAGE07_WORKSTATION_UI.md`
**Scope:** reorganise the existing React workstation into a four-panel studio (target · identity · preview ·
inspector + bottom timeline), add the missing interactions, and connect every new control to a real backend path.
The frontend was **not** rebuilt: every panel hosts existing components and stores.

## 1. Path mapping (prompt → repository)

| Prompt name | Real location | Stage 7 use |
| :--- | :--- | :--- |
| `App.tsx` | `app/roop_3d_ui/src/App.tsx` | Studio layout + `classic` fallback, shared track selection / assignment |
| `VideoViewport.tsx` | `src/VideoViewport.tsx` | new optional `studio` prop: parameter panels, timeline and tray are portalled into the studio slots (all state stays in VideoViewport) |
| `FaceManager.jsx` | `src/components/facemgr/FaceManager.jsx` | unchanged (own route) |
| `FacesetLibrary.tsx` | `src/components/facemgr/FacesetLibrary.tsx` | new `variant="list"`: always-visible virtualized, draggable rows (the dropdown variant is unchanged) |
| `FaceSelectionTray.jsx` | `src/components/tray/FaceSelectionTray.jsx` | hosted in the bottom dock; `TargetFaceCard` accepts FaceSet drops |
| `FaceOverlayCanvas.jsx` | `src/components/canvas/FaceOverlayCanvas.jsx` | FaceSet drop onto a face box; its click selection now drives the inspector |
| `TimelineWorkstation.tsx` | `src/components/timeline/TimelineWorkstation.jsx` (JSX) | hosted in the bottom dock, unchanged |
| `WorkstationCanvas.tsx` | `src/components/canvas/WorkstationCanvas.jsx` (JSX) | unchanged (wipe / crossfade / blend / diff / loupe) |
| `PoseGeometryViewer.jsx` | `src/components/three/PoseGeometryViewer.jsx` | texture, auto-fit, explicit GPU disposal, auto-switch to a supplied mesh |
| `SettingsScreen.jsx` | `src/screens/SettingsScreen.jsx` | unchanged; the inspector reuses its engine store (`stores/engineSelection.ts`) |
| `QueueDrawer.jsx` | `src/components/queue/QueueDrawer.jsx` | hosted in the bottom dock, unchanged |
| `ThemeGallery.tsx` | `src/components/ThemeGallery.tsx` | unchanged; studio CSS uses only theme tokens |
| *(new)* | `src/components/studio/{TargetPanel,FaceSetDetail,StudioInspector,TrackLanes,PanelBoundary}.tsx`, `src/utils/facesetDrag.ts` | studio panels |
| *(new)* | `app/roop_3d/server/identity_assets.py` | identity panel API (§4) |

## 2. What already existed vs. what was missing (evidence)

Already present and **reused, not rebuilt**: split (side-by-side) / wipe / crossfade / blend / difference / loupe
comparison (WebGL2), zoom 1–4× with pan, frame stepping, face overlay with landmarks + pose HUD, right-click
"Use face from frame" extraction, debounced, 503-aware live swap preview (`useLiveSwapPreview`), undo / redo
history store, `.r3dp` project save / verify / restore, snapshot slots, pipeline solo / bypass, curve editor,
face tray, queue drawer, 37 themes, virtualized lists, telemetry isolation.

Defects / gaps found and fixed:

1. **The Stage 4 3D render mode had no UI at all.** `WorkstationParameters.render_mode` existed in the backend
   and the job path, but no control set it. Added to the inspector ("3D settings").
2. **The live preview ignored `render_mode` and `occlusion_mode`.** `LivePreviewPayload` had neither field, so the
   Stage 5 "Occlusion handling" selector only took effect in full renders. Both are now accepted (optional,
   omitted = previous behaviour, previous cache keys unchanged).
3. **Lighting controls did nothing.** The light probe set `relight_strength`/… but `relight` was never `true`, so
   jobs never relit. The inspector now has an explicit "3D relighting" toggle; the probe controls are shown only
   while it is on, and the preview receives the same values.
4. **The Stage 3 identity asset had no HTTP API**, so it could not be shown or built from the UI (§4).
5. **Clicking a face in the video did not select its target track** (only the overlay's store selection
   changed). App now follows the store's `selectedClusterId`.
6. **No FaceSet drag and drop** existed. Added (§3).
7. **Reopening a project / job lost face-box geometry**: `populateFromInspection(…, null, …)` was called without
   the probe, so every overlay box fell back to one default rectangle and a click could not tell faces apart.
   The probed target info is now passed.
8. **The FaceSet library rendered unstyled**: it used Tailwind class names but Tailwind is not a dependency.
   The new list variant uses project CSS (the dropdown variant is left as it was).
9. `PoseGeometryViewer` created a BufferGeometry per mesh and never disposed it, never applied the supplied
   texture, and drew metre-scale meshes far too small. All three fixed.
10. Missing `/favicon.ico` produced a console 404 on every load; an inline icon was added to `index.html`.

## 3. Layout and interactions

```
┌ 01 TARGET ─────────┐┌ 03 PREVIEW ───────────────────────────────┐┌ 04 INSPECTOR ───────┐
│ import video       ││ snapshots · pipeline solo/bypass · tools  ││ target identity     │
│ frame navigator    ││ original | processed (side by side)       ││ models & engine     │
│ detected tracks ⇦  ││ compare: wipe/crossfade/blend/diff/loupe  ││ 3D settings + viewer│
├ 02 IDENTITY ───────┤│ zoom · overlays · landmarks · pose HUD    ││ geometry/expression │
│ FaceSet library ⇨  ││ live swap · PNG · snapshot · float · PiP  ││ masking · restorer  │
│ views · coverage   │└───────────────────────────────────────────┘│ colour & lighting   │
│ 3D identity asset  │                                             │ processing settings │
└────────────────────┘                                             └─────────────────────┘
┌ lanes: scene cuts · processing range · one lane per face track ⇦ · render progress ─────┐
└ timeline (filmstrip, keyframes, pose curve, parameter curves) · export range · tray · queue ┘
```

At ≥ 1760 px target and identity are separate columns; 1280–1759 px stacks them in one column; below 1280 px the
inspector moves under the preview; below 900 px everything is one column (no horizontal page scroll at 390 px).
The header's **Studio layout / Classic layout** button switches back to the previous stacked workstation (stored
per browser only).

| Interaction | Backend / state path |
| :--- | :--- |
| Click a face in the video | overlay `setActiveSelection` → store `selectedClusterId` → App `selectedTrack` → inspector, track card, lane |
| Right-click a face | existing `FaceContextMenu` → `POST /api/faces/extract_from_frame` |
| Drag a FaceSet onto a track card, lane, tray card or a face in the video | `assignSource` in the face-selection store (one undoable command) → `face_mappings` / `face_segments` of the preview and job |
| Preview without rendering | existing `POST /api/preview`, now also carrying `render_mode`, `occlusion_mode`, relighting |
| Save / restore | existing `.r3dp` projects; the new parameters live in `workstation_parameters` |
| Undo / redo | existing history store (assignments, parameters, curves, masks) |
| Engine | Stage 6 `/models/hub` engines; unselectable engines are disabled with their status |
| 3D identity asset | §4; never built implicitly |

## 4. Identity panel API (`server/identity_assets.py`)

* `GET /facesets/{id}/views/{index}/image?size=` (32–512): JPEG of one stored view. No embeddings.
* `GET /facesets/{id}/identity-asset`: `analyze_faceset` coverage summary + the cached asset manifest
  (`state: built | not_built`). **It never builds.**
* `POST /facesets/{id}/identity-asset`: explicit user action. Uses the exact `IdentityAssetStore()` and default
  `IdentityAssetSettings()` that `engine/volumetric/session.py` uses, so the asset built from the panel is the one a
  `pose_3d` / `reenactment_3d` render reuses (same content-addressed build key). One build per FaceSet at a time
  (409 `IDENTITY_ASSET_BUILD_IN_PROGRESS`). Errors from the reconstruction (e.g. inconsistent identity) are
  returned as their domain errors, never replaced by a fake asset.
* `GET /facesets/{id}/identity-asset/mesh`: the cached asset converted with `to_gltf_frame` (y-up), its UVs and
  the observed atlas with **unseen texels painted neutral grey**; 404 until built; bounded to 100k vertices.

## 5. Performance and resource hygiene

* Telemetry isolation preserved: the existing 30 Hz telemetry gate (`e2e/telemetry.e2e.ts`, `stage11Perf.test.tsx`)
  now runs against the studio layout (the default) and passes; new playhead-following UI (frame navigator readout,
  lane playhead) subscribes to `playheadStore` in leaf components only.
* Virtualized: FaceSet list, multi-angle view gallery (lazy-loaded tiles), target track list.
* Stale requests: FaceSet views / asset status abort on change and also drop late answers explicitly; build /
  mesh results are ignored if the selected FaceSet changed meanwhile; the live preview keeps its existing abort +
  debounce + busy-retry logic.
* WebGL: identity / reconstruction BufferGeometry, its texture and the canonical head geometry are disposed when
  replaced or unmounted. The identity mesh is released when the FaceSet or target changes.
* Object URLs: unchanged existing `PreviewURL` leases; the new panels use plain API URLs and data URLs only.
* Each studio panel sits in a `PanelBoundary`: a failing panel shows its error and a Retry button while the preview
  and timeline keep working. Malformed API responses become visible errors with Retry, never a crash.

## 6. Validation (executed)

See §7 for the numbers from the final run.

* Backend: `tests/test_stage07_identity_panel_api.py`: view images, status-never-builds, explicit build → reuse →
  mesh orientation / atlas size against the stored asset, build conflict, unknown FaceSet, live preview mode
  plumbing into `prepare_session` / `infer_frame` and distinct cache keys, and the same routes on a **real local
  FaceSet** (real reader, asset built into `tmp_path`).
* Vitest `src/stage7Studio.test.tsx`: drag payload validation, coverage binning (pitch frame), response validation,
  stale-answer rejection, build-on-request + error/retry, lanes (spans, progress, seek, drop), frame navigator,
  inspector render mode / relight / engines / hub failure retry, live payload fields, panel boundary.
* Playwright `e2e/studio.e2e.ts` (real Chrome, real HTML5 drag and drop, real WebGL): four panels and lanes; drag onto
  card / lane / face in the video; click-to-select; right-click menu; undo / redo; render mode, occlusion and relight
  reaching the preview and the saved project; asset build → textured 3D view → release; light theme; classic
  fallback; no page overflow at 1920 / 1440 / 1100 / 820 / 390 px.
* **Real-server visual acceptance** (production build served by `serve.py`, real models, RTX 4070 machine): a real
  clip was uploaded, inspected (real SCRFD/ArcFace tracking), the real FaceSet "lori" was dragged onto the detected
  track, the live swap frame rendered (~1.4–1.7 s), the identity asset was built from the real FaceSet and shown
  textured in 3D, `pose_3d` was previewed, the wipe comparison was shown, light theme and 1280 / 820 / 390 px widths
  were captured, with zero console errors. Screenshots were inspected; they are not committed.

## 7. Final run (2026-11-03)

| Gate | Result |
| :--- | :--- |
| Python suite (`roop_3d/tests`) | 924 passed, 10 skipped, 0 failed, 18 subtests (11 min 28 s) |
| Stage 7 backend tests | 5 passed (incl. the real local FaceSet) |
| Vitest (`npx vitest run`) | 37 files, 317 tests passed (14 new in `stage7Studio.test.tsx`) |
| Playwright (`npm run test:e2e`) | 18 passed (5 new in `studio.e2e.ts`; telemetry isolation and WebGL gates unchanged) |
| Production build (`npm run build`: tsc → eslint → vite) | exit 0; 0 lint errors (81 warnings, all in pre-existing code patterns) |
| Legacy integrity audit (`verify_isolation.py --report`) | ok: 68,390 comprehensive + 2,645 historical immutable files, no violations |
| Real-server visual acceptance | completed; 0 console errors; no horizontal overflow at 1680 / 1280 / 820 / 390 px |

An earlier full-suite attempt was cut off by a 25-minute wall-clock cap at ~92 % and had shown two failures in
`test_heterogeneous_scheduler.py` (a timing-sensitive raycast performance test) while browser and server runs had
just been competing for the CPU. They did not reproduce: the complete rerun above and three isolated reruns of that
file (8/8 each) passed. No Stage 7 code touches that module.

## 8. Known limitations

* The identity-asset build runs synchronously in the request (seconds on real FaceSets). The button shows elapsed
  time and the request is finite; there is no server-side progress stream or cancel.
* Track lanes use the identity sheet's segments when available, else first → last frame. Inspections are a bounded
  sample (the server says "truncated"), so lanes show only what was inspected.
* The render-progress lane maps `progress.frame / total_frames` onto the job's trim range; jobs without a frame total
  show the scope only (no invented percentage).
* "Expressions" are the existing TPS expression weight and its curve; there is no per-blendshape expression editor
  because the engine has no such control.
* The phone layout is usable but dense: the pre-existing header, run-step strip and node bar dominate small screens.
* The classic layout keeps its pre-existing visual issues (the unstyled FaceSet dropdown); it is a fallback only.
* Dragging requires the target to be on screen (the lanes are reserved on the first screen at ≥ 1000 px tall).
* `pose_3d` / `reenactment_3d` preview fidelity is whatever Stage 4 provides; Stage 7 only routes the mode.
