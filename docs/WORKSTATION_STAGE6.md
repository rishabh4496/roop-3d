# Stage 6: workstation UI

The prompt's `react-ui/` is `app/roop_3d_ui/`. Much of the requested workstation already existed from earlier
stages (router, canvas modes, node bar, snapshot slots, undo/redo, scrub labels, timeline). This stage audited each
requirement against the code, kept what already satisfied it, and fixed or added what did not.

## What was found wrong, and fixed

* **The comparison modes were facades.** Wipe and Alpha Blend drew a divider and an empty `<div>`; no imagery was
  composited, and the Diff Map checked `Image.complete` synchronously on a just-assigned URL, so it stayed blank.
  `WorkstationCanvas` now composites the real original and swapped frames aligned to the letterboxed media rectangle.
* **Solo and snapshots used the legacy sweep endpoint** (`/workstation/preview`). Both now go through
  `POST /api/preview` (Stage 4): cache-first, 503-aware, one code path.
* Solo disabled itself while a preview was loading, and showed the *previous node's* image while the next loaded.
  Switching now aborts the in-flight request, and a live image is shown only if the server's `solo_node` echo
  matches what is currently requested.
* Settings had no reset, no explanation, and the sweep supported four parameters through a different endpoint.

## Routes

Nine hash routes, unchanged ids: `#/home`, `#/faceswap`, `#/batch`, `#/processing`, `#/facemgr`, `#/extras`,
`#/gallery`, `#/history`, `#/settings`. Nav labels now read Home, Face Swap, Batch Matrix, Processing, Face Manager,
Editor, Outputs, History, Settings. All nine screens stay mounted (hidden + `inert`), so drafts and scroll survive
switching tabs, and a refresh on any hash reopens that screen. Unknown hashes land on Home; `#/workstation` is a
legacy alias for `#/faceswap`.

## Run-bar pill (`components/faceswap/PipelinePill.jsx`, `pillModel.js`)

Subscribes to `/ws/telemetry` through the reconnecting `EventConnection` and renders `pipeline.status` frames:
Step 1 Pre-Scan, Step 2 Range Lock, Step 3 Render & Mux (the server's three stage ids, one-to-one), skipped-frame
share, fps, ETA, and a prominent `role="alert"` banner while `engine_state` is `COMPILING_TRT` or `RUNNING_CANARY`
showing `engine_label`, `engine_model`, `engine_precision` and elapsed time computed from `engine_since` (it keeps
ticking between frames). Values the server has not measured render as "—", never as 0.

*"Muxing" is derived, not reported.* The engine has no mux telemetry. The pill shows Muxing when Step 3 is still
running but `stage_progress_pct` has reached 100 (every frame rendered, file not yet published). If the engine later
reports a real mux stage, `buildPillModel` is the one place to change.

The pre-existing hardware diagnostics pill (provider, VRAM) stays in the header; the run bar sits under it on every
route so a compilation is visible wherever you are.

## Canvas, HUD, timeline

* **Compare** (HUD button) opens a full-bleed canvas with Wipe (horizontal/vertical, draggable), **Crossfade**
  (persistent A↔B loop that keeps running while you scrub or edit; pausable; period adjustable; starts paused under
  `prefers-reduced-motion`), Alpha Blend, Diff Map, and a 3.5× Loupe that shows individual pixels. The Diff Map
  heat-maps exactly which pixels changed; differences under 3 levels (JPEG preview noise) count as unchanged. A
  readout shows the pointer's media coordinate to two decimals. Compare is unavailable during playback (the
  side-by-side view owns the `<video>` elements). The Paint/Erase mask brush is the existing *Edit Mask* mode.
* **3D pose & landmark HUD**: *Landmarks* and *Pose* toggles draw the five ArcFace points (coloured by anatomy) and a
  live `Y +12.0°  P -3.0°  R +1.0°` badge on **every** detected face. Pitch + = looking up. The server's detection
  response now carries `pose3d` (standard frame) and `pose_cell` beside the legacy `pose`.
* **Use Face From Frame**: right-click a face box, or press **Use Face From Frame** in the Target Faces bar, to open a
  menu with *Use as Active Source Face*, *Add to .fsz Faceset (Auto-Pose Cell [Yaw, Pitch] → cell)* (needs an active
  source FaceSet) and *Lock as Target Identity Cluster*. Each calls `POST /api/faces/extract_from_frame` with the face's
  left-to-right rank as `face_index`, updates the source gallery, shows a toast and re-renders the swapped preview.
  In Compare view the overlay is visual-only and the canvas forwards the right-click for hit-testing.
* **Timeline** (existing: filmstrip, SMPTE ruler, in/out trim, 0.25×–4× speed) gains **chapter markers**: add at the
  playhead, click to seek, double-click to rename, × to remove, previous/next buttons; stored per clip in
  `localStorage` (a convenience that degrades to in-memory if storage is unavailable).

## Pipeline node bar and buffer soloing

Six nodes, each mapped to a server `solo_node` buffer: Input (`detector`), 3D Pose & TPS Grid (`pose`), Raw Swap
(`swapper`), Occlusion & Seam Mask (`matte`, a black-and-white mask), Restorer (`restorer`), Final Output (`final`).
Solo sends `solo_node` to `POST /api/preview`. The old *Color & Grain* node is gone from the bar (its bypass was
reachable only from there); the colour match remains a Restorer-panel control.

## Self-documenting settings

`components/controls/settingsRegistry.ts` is the single source for every setting's default, bounds, recommended
range, plain-language effect and sweep values; `SettingControl` renders each as a drag-to-scrub label with inline
expression math (`0.5+0.25`), a slider, one-click ↺ Reset, and a hover/focus tooltip. Settings are grouped into
**3D Geometry** (2.5D→3D blend, pose frontalization, TPS expression weight), **Masking** (swap weight, erosion,
feather) and **Restorer** (colour match, restorer blend, CodeFormer fidelity). *CodeFormer fidelity is shown
disabled with an explanation*: the preview compositor has no CodeFormer adapter, so the control would do nothing.
Undo/Redo and the A/B/C/D snapshot slots already existed; snapshots now capture through `POST /api/preview`.

## 3×3 Sweep

Each panel has a **3×3 Sweep** button. Pick any two active settings; the sheet renders nine real previews of the
current frame across Low / Medium / High (sequentially, tiles appear as they finish; a 503 busy answer is retried
after `retry_after_ms`), and clicking a tile applies both values. This replaces the old four-parameter modal that
used a separate backend batch.

## Tooling gate

* `npm run build` = `tsc --noEmit` → `eslint .` → `vite build`.
* ESLint 9 with `react-hooks/rules-of-hooks` (error) and `exhaustive-deps` (warning). **`typescript-eslint` cannot be
  used here**: this project is on TypeScript 7, which ships no compiler JS API, and typescript-eslint requires
  `<6.1`. ESLint therefore parses `.ts/.tsx/.js/.jsx` with Babel (syntax-level); types are checked by `tsc`. In `.ts/.tsx`
  the `no-unused-vars` rule is off because Babel cannot see type-only usage. The build has 0 lint errors and ~74
  warnings, almost all pre-existing unused imports and hook-dependency notes.
* UI tests are Vitest + jsdom (Playwright is not installed): `routes`, `pipelinePill`, `faceContextMenu`,
  `faceOverlayHud`, `settingsAndSweep`, `chaptersAndCanvas`, plus the updated `pipelineSolo`.

## Not verified

No visual check in a real browser was possible in this session (the Chrome extension was not connected). The
built bundle was confirmed to be served by the real server and to contain the new features, but layout, tooltip
positioning, and the feel of scrubbing, crossfade and the loupe are untested by eye. Pose HUD, TPS and
frontalization behaviour on real footage also depend on the real models (see `POSE_STAGE5.md`).
