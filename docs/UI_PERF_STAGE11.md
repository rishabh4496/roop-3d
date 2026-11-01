# Stage 11 + Stage 26: UI performance, WebGL2 inspection canvas, workstation ergonomics

Frontend only. No backend, engine or API contract changed.

## Path mapping (the stage prompt names paths that do not exist)

| Prompt | Real location |
| --- | --- |
| `react-ui/` | `app/roop_3d_ui/` |
| `react-ui/src/store/` | `app/roop_3d_ui/src/stores/` (`atomicStore.ts`, `telemetryStore.ts`, `playheadStore.ts`) |
| `components/canvas/WebGLViewport.jsx` | `src/components/canvas/WebGLViewport.jsx` (+ `webglModel.js`) |
| `PipelinePill.jsx` | two exist: `components/faceswap/PipelinePill.jsx` (3-step run bar) and `components/hud/PipelinePill.jsx` (Diagnostics HUD) |
| Zustand | not a dependency. `stores/atomicStore.ts` is a ~60-line `useSyncExternalStore` store with the same two modes (transient `subscribeSelector`, and a selector hook that re-renders only on a changed slice) |

## 1. Telemetry / playhead isolation

Before: `App` held `telemetry` in `useState` and set it on every `/api/v1/ws/events` hardware tick, and held the playhead `time`
that every playback `timeupdate` wrote, so each tick re-rendered the whole workstation.

Now:

* `telemetryStore` is the single sink for hardware ticks (`App`'s events socket, the HUD socket) and `pipeline.status` frames (run
  bar). Readers subscribe to a slice: the HUD pill, the run bar, `GpuReadout` (Processing) and `GpuName` (Settings) are the only
  re-rendering leaves. `SettingsScreen`, `WorkstationScreens`, `VideoViewport` and `App` no longer receive or read telemetry.
* **Committed vs live playhead.** `App.time` is the *committed* time (seek, step, pause, load). While playing, `timeupdate` writes
  only `playheadStore`; on pause `VideoViewport` commits the final tick back to `App`. Leaves that follow playback
  (`LiveNeedle`, `LiveTimecode`, `LiveFrameCounter`, `LiveScrubber`, the curve editor frame, `FaceOverlayCanvas`, `FaceSelectionTray`,
  the floating caption) subscribe through `usePlayhead` / `<LiveTime>`. The timeline's auto-scroll and range-loop effects use a
  transient subscription (no render). Handlers that act on "the current frame" read `getPlayhead()`.
* Chapters re-render the timeline only when the *chapter id* changes (primitive selector).
* **Virtualization** (`components/common/VirtualList.tsx`, windowed rows, measured heights, follows the nearest scroll parent or its
  own box, `aria-setsize/posinset`): Face Manager identities and the five pose-bucket crop lists, FaceSet library, History cards,
  Rendered outputs gallery, Processing job list. Not virtualized: the 3x3 pose matrix (fixed 9 cells), the face tray (a handful of
  clusters), the queue drawer.

## 2. WebGL2 inspection canvas

`WorkstationCanvas` (full-bleed compare view) renders through `WebGLViewport` when WebGL2 exists and the shaders compile; otherwise it
falls back to the previous DOM + 2D path unchanged (jsdom, old GPUs, context loss).

* One fragment shader, all modes: split wipe (horizontal/vertical), alpha crossfade (runs its own rAF uniform: no React render per
  frame), alpha blend, **diff** (`|swapped - original|`, noise floor 3/255, `gain` mapped to **Inferno or Viridis** through a 256x1
  LUT; same constants as the 2D heat map), **3.5x loupe** with **bilinear / bicubic (Catmull-Rom, 9 bilinear taps)** toggle.
* Textures are uploaded at the image's native size (mipmapped; only downscaled if it exceeds `MAX_TEXTURE_SIZE`); the backing store is
  `CSS size x devicePixelRatio` (a 1920x1080 stage at DPR 2 is 3840x2160).
* **Pose gizmo**: a second tiny program draws yaw (green) / pitch (red) / roll (blue) arrows anchored at the nose bridge
  (midpoint of eye-centre and nose tip from the 5-point landmarks; the 106-point layout is deliberately not used, see
  `POSE_STAGE5.md`). Standard pose frame: yaw+ = nose to image right, pitch+ = up, roll+ = clockwise on screen (the roll sign is this
  document's reading of "in-plane tilt"; if the engine's roll is the other way, flip `sr` in `poseAxes`).
* Draws only when something changed, coalesced to one frame. `preserveDrawingBuffer` is opt-in (snapshots / tests).
* Colormaps are 10-stop piecewise-linear samples of matplotlib's Inferno / Viridis, not the full 256-entry tables.

## 3. Stage 26 ergonomics

* **Floating preview** (`FloatingPreview.tsx`): three presentations of the same live React content. `Shift+F` toggles a draggable,
  resizable in-app window portalled to `<body>` (visible on every route, position/size/open-state persisted, keyboard move/resize on the
  title bar); `Shift+G` / the PiP button opens the browser **Document Picture-in-Picture** window (always on top, Chromium 116+; falls
  back to the in-app window when unsupported or rejected); the title-bar ↗ keeps the old secondary-window popup. Keys pressed in a
  detached window are forwarded to the workstation. While a preview is pinned, `VideoViewport` keeps rendering live swaps even when
  another screen is showing.
* **Hotkey remapper** (`Hotkeys.tsx`): central action table with groups, conflict detection, Backspace to unassign, persistence in
  `roop3d-hotkeys-v1`, and cross-tab sync. A stored map from before this stage keeps its remaps (new actions get their default if
  free, else are left unassigned). The timeline's hardcoded `I/O/Shift+I/Shift+O/Alt+R` listener (which ignored remaps and double-fired
  with the remappable ones) was removed; all of them are bindable now.

  | Action | Default | | Action | Default |
  | --- | --- | --- | --- | --- |
  | Frame step | `Left` / `Right` | | Solo buffer 1-6 | `1`-`6` |
  | Mark in / out | `I` / `O` | | Extract face from frame | `E` (paused frame) |
  | Jump to in / out | `Shift+I` / `Shift+O` | | Snapshot A/B/C/D | `Shift+A` ... `Shift+D` |
  | Toggle split wipe | `W` | | Floating preview / PiP | `Shift+F` / `Shift+G` |
  | Toggle diff map | `D` | | Reset range | `Alt+R` |

  (`Alt+A`-`Alt+D` were avoided for snapshots: `Alt+D` focuses Chrome's address bar. Snapshot `A/B/C/D` use `Shift`.)
* **Expression math** (`NumericInput.tsx`): a leading `+`, `*` or `/` is relative to the field's current value (`*1.25`, `/2`,
  `+0.05`); `0.4 * 1.5` -> `0.6`; `×`, `÷`, `−` accepted; a live `= result` preview shows before Enter. A leading `-` stays a negative
  literal; type `+-0.1` to subtract. Applied to every scrub input, including the film-grain and batch-recipe fields that were raw
  `<input type="number">`. Evaluation is a bounded recursive descent; text is never executed.

## Verification gates

```
cd app/roop_3d_ui
npm run build        # tsc --noEmit -> eslint . (0 errors) -> vite build
npx vitest run       # jsdom: 224 pre-existing + Stage 11 tests
npm run test:e2e     # Playwright, system Chrome, Vite dev server
```

* `stage11Perf.test.tsx` mounts the **real `<App/>`**, streams 90 frames at "30 Hz" into the events socket, the run-bar socket and the
  HUD socket, and asserts **0 renders** of `App`, `VideoViewport`, `TimelineWorkstation`, `SettingControl` (also under a real React
  `<Profiler>`: 0 commits), `SettingsInspector` and `FaceManager`, while the run bar / HUD show the last frame. A control test shows the
  probes do move when `App` state changes. Mutation-checked: making `App` subscribe to the telemetry store makes it fail (90 renders).
* `e2e/telemetry.e2e.ts` is the same gate in Chrome: frames pushed from Node at 30 Hz through `page.routeWebSocket`, render counts read
  from `window.__R3D_RENDER_PROBE__` (`utils/renderProbe.ts`: a no-op unless that sink exists), on `#/faceswap`, `#/facemgr`,
  `#/settings`; plus frame pacing (median rAF interval < 40 ms). It also checks the floating window and a remapped hotkey survive route
  changes and a full reload. Mutation-checked the same way (App rendered 360 times).
* `e2e/webgl.e2e.ts` runs the **real GLSL** in Chrome and reads pixels back: wipe, blend, diff (Inferno and Viridis against the CPU
  reference), bilinear vs bicubic at an edge, gizmo on/off, and a native 3840x2160 texture into a 3840x2160 backing store.
* `stage11Canvas.test.tsx` / `stage11Model.test.tsx` / `stage11Math.test.tsx` cover shader maths, colormaps, the gizmo, the viewport
  against a recording fake context, the fallback, the stores, virtualization, the timeline isolation, hotkey migration, the floating
  window, Document PiP (fake `documentPictureInPicture`) and expression math across the scrub inputs.

## Known limits

* **Job progress still re-renders `App`.** `job.*` events on `/api/v1/ws/events` trigger a debounced `refreshJobs()` (120 ms) that
  `setJobs`, so a running render re-renders the workstation at up to ~8 Hz. That is the jobs list, not the telemetry stream this stage
  targets; moving job progress into the store is the obvious next step.
* The three telemetry consumers still hold their own sockets (events, run bar, HUD). They all write the same store; merging them is a
  cleanup, not a correctness issue.
* Document PiP was exercised against a fake `documentPictureInPicture` and its fallback in a real Chrome; headless Chrome cannot grant
  the user activation a real PiP window needs, so the real PiP window itself was not driven.
* The Playwright suite runs against the Vite **dev** server (React dev build, StrictMode), so the pacing numbers are pessimistic; the
  zero-render assertions hold in either build.
