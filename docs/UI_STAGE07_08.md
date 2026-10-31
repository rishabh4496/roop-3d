# Workstation UI stages 7 and 8

Status: core UI implemented with passing automated checks; neither stage has full
acceptance sign-off. Outstanding work includes visual/Pinokio-tab/multi-monitor
acceptance, real-model acceptance of the new preview/sweep path, aligned UV
projection, XSeg, grain processing/bypass, UltraMax/CodeFormer, worker flamegraphs,
distributed telemetry, unconfigured Extras tools and OpenVINO controls. The
existing golden render test does not establish acceptance of these new features.

The application shell supports `#/home`, `#/faceswap`, `#/batch`, `#/processing`,
`#/facemgr`, `#/extras`, `#/gallery`, `#/history`, and `#/settings`. The old `#/`
link resolves to the workstation. Unknown routes resolve to the dashboard.
Screens remain mounted when hidden: canvas controls, drafts, selections and
playhead position survive navigation. Playback pauses when leaving the workstation.
Projects retain the existing verified manifest/checkpoint workflow.

The workstation has four panels: Timeline, Face Tray, Inspector, and 3D Pose &
Geometry. Drag headings or use the arrow buttons to reorder them. The lower-right
handle resizes each panel; arrow keys on that handle adjust size. Collapse keeps
the panel contents mounted. Four presets and up to 30 named custom layouts are
stored in browser-local storage. The floating preview uses a same-origin React
portal into `window.open`, so updates share state without transferring media to
another service. A blocked popup falls back to an in-page preview.

Numeric labels support horizontal dragging and arrow keys. Shift multiplies the
step by 10; Alt multiplies it by 0.1. Numeric fields accept parentheses and basic
arithmetic, with a leading `+` meaning an increment relative to the current value.
Enter or blur commits; Escape restores the last committed value. Bounds and integer
constraints apply to both input methods. Invalid expressions leave values unchanged.
No JavaScript evaluator is used. The settings shortcut editor searches actions and
keys, detects conflicts, and persists remaps. Typing and modal dialogs suppress
workstation shortcuts.

## Pipeline previews

`POST /api/v1/workstation/preview` accepts a validated local job request, playhead
timestamp and workstation parameters. It reuses the target hash/inspection/FaceSet
checks, shares render admission, loads models only on demand and releases its
session and admission on success or failure. Tracking replays from frame zero to
preserve track identity. Work is bounded to 1200 frames, 120 seconds, and 4K input.
Returned JPEG previews preserve aspect ratio with a maximum edge of 1920 pixels.
Disconnects cancel work cooperatively between inference operations.

Solo selects the source reference, detector overlay, observed dense landmarks,
unrestored replacement, actual grayscale matte, restored replacement, color matched
replacement, or final composite. Optional capture hooks do not change reference
compositor output. The available matte is BiSeNet with its existing landmark-hull
fallback. Pose is estimated 2.5D; the UI does not claim an aligned UV projection.
The configured local restorer is GPEN. XSeg, UltraMax, CodeFormer, and grain adapters
are not supplied by this UI stage. Unsupported stages show explicit reasons.

Restorer and color bypasses affect the preview and render request parameters.
New requests and saved projects carry restoration blend, mask feather, color
strength, pose strength and bypass state. Older requests retain their reference
defaults and their prior identity digest when the optional parameters are absent.

Parameter Sweep sends one request with two distinct three-value axes. Supported
axes are restoration blend, mask feather, and color strength. Tracking and swap
inference run once; the compositor creates nine independent variants. Temporal
state resets per cell, so the contact sheet is a single-frame comparison rather
than a temporal video fidelity claim. Selecting a cell applies its full captured
parameter state. Changing the request, playhead, axes, or parameters aborts and
discards stale results. Swept stages are enabled to make each axis effective.

## Other screens and capability boundaries

Dashboard, history and gallery use persisted project/job data. The batch matrix
queues selected saved-target/source combinations with current execution settings.
Processing shows live queue progress and measured preview stage timings. There is
no distributed worker telemetry or nested worker flamegraph feed configured.
Execution-provider controls remain capability-gated. Media-tool cards identify
unconfigured operations; the extras screen provides the existing local stream probe.

## Validation

- Python: 409 tests and 18 subtests passed, including the existing unmocked golden
  clip render. Full-suite temporary files were isolated under
  `roop_3d/.runtime/pytest-ui78-full`.
- Frontend: 70 tests passed, including DOM integration for route retention,
  scrubbing/arithmetic, remapping, layouts, floating preview, real-buffer selection,
  stale-result cancellation and batched sweep selection.
- `npm run build` passed (TypeScript and production Vite bundle).
- Pinokio launched this checkout's `start.js`; API health, production UI and the
  preview OpenAPI route returned successfully at `127.0.0.1:8030`.
- Read-only legacy integrity audit passed at `2026-10-31T14:57:52.217134+00:00`:
  68,390 comprehensive immutable checks and 2,645 historical checks; zero
  violations. The existing policy classified 1,555 runtime files separately.
  Preserved baselines were not replaced.
- Visual browser inspection was unavailable: Computer Use could not determine
  the Windows browser URL confidently and stopped. DOM tests do not replace visual
  or multi-monitor hardware acceptance.

No speed, GPU-profile fidelity, optional-model inference, licensing or rights claim
is established by these tests. Neural inference is replaced with deterministic
fixtures in the new preview contract tests; the original golden test remains real.
