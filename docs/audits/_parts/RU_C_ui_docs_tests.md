# RU_C â€” roop-ultimate slice C: react-ui/, docs/, app/tests/

Reference: roop-ultimate @71ebb65 (read-only clone). Method: every file in the slice was opened programmatically; per-file purposes below are taken from each file's own header comment/docstring (or first test names), not guessed. Items marked [INFERENCE] are not directly verified.

## 0. Executive summary

- **UI = React 19-style SPA (Vite) with 9 hash routes** (`react-ui/src/App.jsx:98-107`: home, faceswap, batch, processing[transient], facemgr, extras, gallery, history, settings). Face Swap + Processing are eager; others lazy (`App.jsx:37-75`).
- **There is NO 3D volumetric viewport / mesh / point-cloud view.** The only "3D" UI items are (a) a `use_3d_recon` toggle that merely requests a pose-matched source crop for image-source swappers (`FaceSwap.jsx:4007`, `defaults.js:103-105` default false), (b) a WebGL2 equirectangular VR180/360 + stereo inspector (`StereoVRViewport.jsx:15-16,29,180`), (c) 3D `.cube` LUT grading (`cubeLut.js`, `webgl2GradingPipeline.js`). Face geometry is 2D warp + similarity transform (`trajectory_cache`, M/M_inv + yaw/pitch/roll).
- Present: split/diagonal wipe compare (`CompareSlider.jsx:20` angle 0..360), multi-track NLE timeline with minimap/ruler/scene-cut/presence/identity-keyframe/override lanes (`timeline/*`), head-pose HUD yaw/pitch/roll (`headPoseModel.js:10` BAR_RANGE yaw 90/pitch 60/roll 45), pose radar for face-bank coverage (`facemgr/PoseRadar.jsx`), mask brush (`maskBrush.js:50`) + Bezier roto layer (`RotoLayer.jsx`, feather 0-120 px at :321), ParserRegions (19 CelebAMask-HQ classes), terminal-mirror dashboard, TRT compile banner, pipeline pill, flamegraph.
- **docs/** (328 files): 123 markdown + 205 measured artefacts (JSON/TXT/logs, TRT engine reports) under `docs/perf/`. Several `docs/audits/STAGE*` reports are explicitly flagged *synthetic inputs / INVALID - DO NOT CITE* (marked 2026-10-09) â€” treat their numbers as non-evidence.
- **app/tests/** (717 files; 6453 `def test_` by regex count): 474 `test_*.py` pytest modules + 243 harness/probe/diag/support files (A/B scripts, probes, fixtures, one `.npz`).

## 1. react-ui/ (377 files)

### 1.1 Capability map (the questions asked)

| Capability | Present? | Where / evidence |
|---|---|---|
| 3D viewport (mesh / volumetric / point cloud) | **No** | No three.js/mesh code; `grep -il "volumetric\|mesh\|three"` finds only App/api/Extras/FaceSwap/Processing/Settings/constants text hits (none render 3D). `use_3d_recon` toggle only: `FaceSwap.jsx:4007`, `defaults.js:105`. |
| VR / stereo 3D inspector | Yes (WebGL2) | `faceswap/StereoVRViewport.jsx` modes vr180/vr360 (:15-16), yaw/pitch/fov state (:27-29), FOV clamp 30-120 (:180), stereoMode/stereoLayout `sbs` (:30-31). Checked by `.render-check/render-stage21-stereo-vr.mjs`. |
| Split-wipe compare canvas | Yes | `faceswap/CompareSlider.jsx` (angle 0..360, :20), `wipeTransform.js`, `PreviewCanvas.jsx` (uncontrolled canvas), `ABReferencePanel.jsx` (previous render wiped vs live), `player/VideoCompareStage.jsx` (original vs result in one WebGL canvas), `OutputCompare.jsx` (two finished renders), `CompareGrid.jsx` / `QuadInspectionGrid.jsx` (4-up) / `ParameterSweepModal.jsx` (9-tile sweep, :27,:94). |
| Timeline trim / scrubber | Yes (NLE) | `faceswap/Timeline.jsx` (68 KB): minimap, adaptive ruler, render-state strip, scene cuts, presence, audio wave, identity keyframes, override lanes; `timeline/viewMath.js:13-16` FRAME_CELL_MIN_PX 56, FRAME_VIEW_PX 140; J/K/L `shuttle.js`; segments `SegmentBar.jsx`/`SegmentMatrix.jsx`. |
| Pose / telemetry HUD | Yes | `HeadPoseHud.jsx` + `headPoseModel.js:10` BAR_RANGE {yaw 90, pitch 60, roll 45}; data = `GET /api/trajectory/pose?frame=` (HeadPoseHud.jsx:83) i.e. Stage-2 smoothed keypoints, flags 1 gap-filled / 2 flow-tracked / 4 predicted (headPoseModel.js:14-18). Telemetry: `shell/TelemetryPill.jsx`, `useTelemetrySocket.js`, `store/telemetryStore.js`, `FrameFlamegraph.jsx`, `LiveTelemetry.jsx`. |
| Mask brush / roto | Yes | `maskBrush.js:46-62` (brushSize = on-screen diameter, r>=0.5, feather inner radius), `maskStrokes.js`, `RotoLayer.jsx` (Bezier pen, feather slider 0-120 px :321, `rotoModel.js`), `ParserRegions.jsx` (face-parser class toggles). |
| Pose-aware face bank UI | Yes | `facemgr/PoseRadar.jsx` (5-axis radar: frontal, L/R profile, high/low), `PersonGroups.jsx` (coarse pose buckets, per-person source mapping), `FacesetLibrary.jsx` (.fsz), `facemgr/MorphLab.jsx` + `IdentityBlender.jsx` (2-5 identity slerp blend), `extras/VideoFramesetExtractor.jsx` (video â†’ .fsz). |
| Temporal controls | Yes | `TemporalModeControl.jsx` + `temporalModes.js:28-` four modes Off / Smart-Skip ROI / Balanced EMA / Cinema Cut-Aware (cinema = one_euro stabilize_face + landmarks + enhancer + mask + hf_texture + track_identities). |
| Queue / checkpoint UI | Yes | `QueuePanel.jsx`, `useQueue.js`, `store/jobStore.js`+`useJobRecovery.js` (reattach to in-flight render), `ProjectsPanel.jsx`, `home/ResumableCheckpoints.jsx`, `history/CheckpointDiff.jsx`, `BatchSwap.jsx` (Batch Matrix, 94 KB), `segments/*`. |
| Terminal / diagnostics | Yes | `ProcessingTerminal.jsx` (ANSI, circular log ring `logRing.js`: ERROR>WARN>PERF>INFO), `ConsoleMirror.jsx`, `DiagnosticsPanel.jsx`, `RunModelsPanel.jsx`, `PipelinePill.jsx` (1 Pre-Scan â†’ 2 Range Lock â†’ 3 Render & Mux), `PipelineReport.jsx`, `CompileBanner.jsx` (TRT compile/canary), `TrtCachePanel.jsx`, `EnvironmentHealth.jsx`. |
| Colour / scopes | Yes | `ColorGradingPanel.jsx`, `cubeLut.js` (.cube DOMAIN_MIN/MAX), `webgl2GradingPipeline.js` (linear float, lift/gamma/gain, kelvin/tint), `BroadcastScopesDock.jsx` (RGB parade, luma wave, vectorscope), `falseColorZebra.js` (>95 IRE zebra), `SkinToneScope.jsx`. |
| Fast frame path | Yes | `player/FastCanvasPlayer.jsx` + `fastPlayer.worker.js` (OffscreenCanvas), `glRenderer.js` (2 picture layers, 1 draw call), `transport/frameSocket.js` (`/ws/frames` binary), `frameDecoder.worker.ts`, `usePlaybackBuffer.js`, `useThrottledFrameRequest.js` + `retryPolicy.js:27` RETRY_DELAYS_MS [250,500,1000]. |
| Webcam / live | Partial | `useLiveCam.js` (backend `/api/livecam/*`), no dedicated live view in nav; [INFERENCE] lives inside Extras/FaceSwap. |
| Mock backend | Yes | `mock-server/server.ts` (130 KB; mock of app/api.py, SVG placeholders), `mockQueue.ts`, `identityMock.ts`, `mouthCustody.ts`, `pipelineTypes.ts` (stage progress split 0-10% S1, 10-25% S2, rest S3). |

### 1.2 Per-file table (all 377 files)

Purpose text is each file's own header comment (truncated). Bytes in parentheses are not repeated; see inventory.


#### `react-ui/` (11 files)

| file | purpose / key symbols (from header) |
|---|---|
| .gitignore | Logs |
| .oxlintrc.json | JSON data/config |
| README.md | Roop Ultimate â€” React UI |
| index.html | (no header comment) |
| package-lock.json | JSON data/config |
| package.json | JSON data/config |
| postcss-ink-floor.js | no text-white/N renders fainter than the ink at 86% |
| postcss.config.js | After Tailwind: it rewrites the `text-white/N` rules Tailwind generates. |
| tailwind.config.js | @type {import('tailwindcss').Config} |
| tsconfig.json | /*.ts", "src/* |
| vite.config.js | The proxy below is for `npm run dev` ONLY. In production there is no Vite server and no proxy at all: the launcher runs `vite build`, and the FastAPI backend serves react-ui/dist itself (see the SPA f |

#### `react-ui/.render-check/` (48 files)

| file | purpose / key symbols (from header) |
|---|---|
| batch-matrix-check.mjs | Behavioural checks for the Batch Matrix staging logic (per-file matrix, grouped, one-to-many, recipes). These import and RUN the module BatchSwap.jsx ships (src/components/faceswap/batchMatrix.js), so |
| brush-mapping-check.mjs | Mask Brush pointer -> canvas mapping (plain node). Run: node .render-check/brush-mapping-check.mjs Pins: sub-pixel positions are kept (never snapped), the brush radius follows the on-screen diameter,  |
| console-mirror.fixture.json | JSON data/config |
| error-boundary-check.mjs | Runner for render-error-boundary.mjs: same shape as run.mjs (Vite's own SSR pipeline, so the JSX and aliases are the ones the production build uses, and a result reported through a global because Vite |
| face-mapping-check.mjs | Behavioural checks for the target-person -> source-face mapping. These import and RUN the real module the UI ships, rather than asserting on source text, so a refactor that keeps the strings but break |
| face-roles-check.mjs | Behavioural checks for the target-pin / bystander-exclusion decision the Media Canvas badges and the Person Groups dropdown share. They import and RUN the real module (src/components/faceswap/faceRole |
| fixtures.json | JSON data/config |
| gen-console-mirror-fixture.py | Brings console-mirror.fixture.json up to the dashboard's CURRENT latency columns, using the real backend code. app/env/Scripts/python.exe react-ui/.render-check/gen-console-mirror-fixture.py The fixtu |
| gen-stage14-fixture.py | Regenerates stage14.fixture.json from the REAL backend code (run from the repo root): app/env/Scripts/python.exe react-ui/.render-check/gen-stage14-fixture.py Nothing in the fixture is typed by hand:  |
| mouth-custody.cases.json | JSON data/config |
| mutation-check.mjs | Mutation check: prove the render harness actually detects the regressions it claims to guard, by reintroducing each bug and confirming the check goes red. A green test suite is worth exactly as much a |
| pipeline-hud.fixture.json | JSON data/config |
| pipeline-report.fixture.json | JSON data/config |
| player-check.mjs | Frame transport + player pipeline checks (plain node, no browser). What these pin, because each is a way the fast path could "work" while showing the wrong thing: * the client reads the header at the  |
| provider-matrix.fixture.json | JSON data/config |
| recognition-sync-check.mjs | Behavioural check for the recognition-selection <-> settings-state sync. Runs the real module the UI ships. Run with: node .render-check/recognition-sync-check.mjs |
| render-api-client.mjs | The API client's contract, exercised for real (fetch is stubbed; nothing else is). * every request has a 15 s deadline unless the call passes `timeout: 0`; * thrown errors say WHICH request and HOW it |
| render-console-mirror.mjs | The terminal dashboard on #/processing, driven by a REAL snapshot. `console-mirror.fixture.json` is generated by roop/terminal_dashboard.py itself (a TerminalProgressManager fed the calls the pipeline |
| render-error-boundary.mjs | The one ErrorBoundary, exercised for real. There used to be two (a bare copy in main.jsx and components/ErrorBoundary.jsx) that had drifted apart. They are one component with two variants now, and wha |
| render-head-pose-hud.mjs | The head-pose HUD (live yaw / pitch / roll), driven by `GET /api/trajectory/pose`. * model: angles print with a sign and one decimal, a missing angle prints a dash (never 0.0 degrees); bar markers cla |
| render-history.mjs | Stage 23: undo/redo history engine, transactions, and the A/B/C/D snapshot slots (plain node). node .render-check/run-module.mjs render-history.mjs The properties that matter and that a screenshot can |
| render-identity-keyframes.mjs | Stage 20: the identity keyframe model and the Identity Blender recipe's normalisation (plain node, no browser). node .render-check/run-module.mjs render-identity-keyframes.mjs Three kinds of check: -  |
| render-mouth-custody.mjs | Mouth custody's pure logic (plain node): the legacy migration, the request fields and the slip arithmetic. node .render-check/run-module.mjs render-mouth-custody.mjs The cases in `mouth-custody.cases. |
| render-pipeline-controls.mjs | The 3-stage pipeline controls and their live telemetry, on #/faceswap, #/settings and #/processing. 1 the rules swapper list + groups, the three restorer modes and the settings each one writes (and th |
| render-pipeline-pill.mjs | The 3-Step Pipeline Pill (1 Pre-Scan -> 2 Range Lock -> 3 Render & Mux), driven by the frames /ws/telemetry carries. * model: each backend stage state maps to a step state; Stage 4 (not in `stages`) k |
| render-pipeline-report.mjs | The per-job pipeline report in the UI, driven by a REAL report. `pipeline-report.fixture.json` is the `report` of an actual render (docs/perf/stage9/verify_stage4_report.json, run `cfr_3stage_trim`: a |
| render-processing.mjs | Render the Processing tab for real, in Node, against payloads captured from the live backend. WHY THIS EXISTS, given the tab already had 162 passing tests. Those tests â€” and every other `test_ui_*.py` |
| render-provider-matrix.mjs | The "Benchmark Pipeline Providers" result in the UI, driven by a REAL `provider_matrix`. `provider-matrix.fixture.json` is the `result` of an actual `bench.run_benchmark('providers')` on the 4070 (doc |
| render-retired-controls.mjs | Phase 12: controls that were removed stay removed, and what replaced them behaves. 1 Identity Blender one attribute dial (Feature Dominance / jaw) - the only direction identity_render_validation.json  |
| render-stage11-logic.mjs | Stage 11's pure logic (plain node, no browser): the Pen tool's geometry and the identity pins / confidence strip. node .render-check/run-module.mjs render-stage11-logic.mjs The geometry is the part th |
| render-stage14.mjs | Stage 14 on the screen: the TensorRT compile banner + receipt, and the Decode / GPU / Encode queue row. `stage14.fixture.json` is generated by the BACKEND (gen-stage14-fixture.py): `predictor.warmup_s |
| render-stage18-lut.mjs | Stage 18: WebGL2 3D .cube LUT Color Pipeline, Skin-Protected Vibrance, and Scopes Tests Exercises: - .cube parser with LUT_3D_SIZE, DOMAIN_MIN, DOMAIN_MAX - Explicit DOMAIN_MIN < DOMAIN_MAX validation |
| render-stage19-monitor.mjs | Stage 19: BroadcastChannel Synchronization Bridge & Pop-Out Reference Monitor Tests Exercises: - createReferenceChannel with asynchronous message delivery - Zero-copy message broadcasting (FRAME, TRAN |
| render-stage21-stereo-vr.mjs | Stage 21: Stereoscopic 3D & WebGL2 Equirectangular VR Viewport Render Checks. |
| render-stage22-nle-vfx-provenance.mjs | Stage 22: NLE Timeline Interchange (CMX3600 EDL / OTIO), Standalone VFX Matte Pass Export, and Cryptographic Provenance Render Checks. |
| render-stage25-logic.mjs | Stage 25's pure logic (plain node, no browser): the parameter-curve model, how a curve reaches a render as pieces, and the face context menu's geometry and skin-tone maths. node .render-check/run-modu |
| render-stage26-logic.mjs | Stage 26's pure logic (plain node, no browser): the expression parser behind numeric fields, the scrub arithmetic, and the key-combo / keymap model behind the shortcut overlay and the remapper. node . |
| render-stage27-logic.mjs | Stage 27 Logic Render Check (plain node, no browser): 1. Resizable Multi-Panel Workspace Store: - Panel size defaults (leftWidth: 350, rightWidth: 330, bottomHeight: 280) - Strict dimension clamping ( |
| render-stage5-logic.mjs | Stage 5's pure logic (plain node, no browser): the Morph Lab's pad geometry and weight rules, the Segment Chunk Matrix's pill/summary rules, the A/B reference store, and the checkpoint-vs-current sett |
| render-stage8-logic.mjs | Stage 8's pure logic (plain node, no browser): the terminal's circular log buffer and severity filter, and the compare divider's transform geometry. node .render-check/run-module.mjs render-stage8-log |
| render-timeline-logic.mjs | The multi-track timeline's pure logic (plain node, no browser): the view window, the adaptive ruler, thumbnail slots, the render/cache strip, scene + presence geometry, per-segment overrides, the J/K/ |
| render-workstation-logic.mjs | The workstation UI's pure logic (plain node, no browser): the quick recipes, the temporal stability modes, the telemetry pill's reading, the runtime section rows, the Settings tab map and the workspac |
| retry-policy-check.mjs | Retry policy for frame requests (plain node, no browser). The defect this pins: a failed frame/chunk request was treated as "still wanted, so ask again" -- from `finally` through a 150 ms throttle in  |
| run-module.mjs | Runner for any check module that needs Vite's resolver (extensionless imports, JSX, the aliases the production build uses): `node .render-check/run-module.mjs render-api-client.mjs`. The module report |
| run.mjs | Runner: transform the JSX through Vite's own SSR pipeline, then execute. Using Vite rather than a standalone esbuild/babel step is deliberate â€” it resolves aliases, CSS imports and the exact plugin ch |
| stage14-async-check.mjs | Stage 14: stale asynchronous state â€” delayed-response races on the client. These import and RUN the real modules the UI ships (processingSelection.js, faceMapping.js). The race section replays FaceSwa |
| stage14.fixture.json | JSON data/config |
| video-io-hud-check.mjs | The HUD's Video I/O line (telemetryStore.videoIoTextOf) and the fields behind it. The defect this pins: a hardware rung (NVDEC, NVENC) that gave way to a slower one mid-run was recorded in a list the  |

#### `react-ui/e2e/` (40 files)

| file | purpose / key symbols (from header) |
|---|---|
| a11y.spec.js | axe on every tab, and on the Batch Matrix strategies (see helpers.STATES). A view's known violations are recorded in allowlist.json as `rule id -> node count`; the test fails when a rule appears that  |
| allowlist.json | JSON data/config |
| api-client.spec.js | The API client's three behaviours that only mean something in a browser (the Node checks in .render-check/render-api-client.mjs cover the client's logic itself): * a failure that used to be swallowed  |
| baseline.js | Re-record e2e/allowlist.json from what the UI does RIGHT NOW. Run it after fixing something (to ratchet the file down) or after deciding that a new failure is acceptable (to record it) -- and read the |
| canvas-tools.spec.js | Where an object-contain picture of w x h sits in the canvas's box (independent of the app code). |
| compile-banner.spec.js | The backend announces STAGE_COMPILING_TRT / STAGE_RUNNING_CANARY on /ws/v1/pipeline/status (app/roop/pipeline_status.py). The mock replays that channel and is driven through POST /__mock/pipeline. Two |
| console-mirror.spec.js | The terminal dashboard on #/processing in a real browser, fed a REAL snapshot: `console-mirror.fixture.json` is produced by roop/terminal_dashboard.py (see .render-check/render-console-mirror.mjs), an |
| contrast-themes.spec.js | Computed-background contrast under EVERY preset theme. Opt-in (about 5 minutes): npm run test:e2e:themes fail if a theme got WORSE than its baseline npm run test:e2e:themes:baseline re-record e2e/allo |
| contrast.js | WCAG contrast with COMPUTED backgrounds, for the text axe cannot decide. axe-core gives up ("background color could not be determined due to a background gradient") on anything painted over this app's |
| contrast.spec.js | WCAG AA (4.5:1, or 3:1 for large text) for every visible text node on every view, measured against the COMPUTED backdrop -- see contrast.js for how, and for why axe's own contrast rule is not enough h |
| error-boundary.spec.js | The live path of the one ErrorBoundary (the markup and state machine are covered in .render-check/render-error-boundary.mjs): a tab's lazy chunk fails to download -- the server restarted, or the conne |
| face-roles.spec.js | not JSON |
| faceswap-layout.spec.js | Face Swap's slider grid, type scale and run controls, measured in the real layout at 1440px (the width the slider labels used to truncate at). * no slider-tracker label is cut off (they wrap to two li |
| frame-budget.spec.js | Frame-time, long-task and React-commit measurements for the workstation, in a real (headless) Chromium against the production build, with the mock server standing in for app/api.py. What it measures,  |
| frame-busy.spec.js | While the pipeline's pre-pass (Stage 1 sparse pre-scan / Stage 2 active-range pre-pass) has the GPU flat out, the backend answers an UNCACHED frame with a fast 503 {"busy": true, "stage": "STAGE_2_ACT |
| frame-unavailable.spec.js | When the stage's frame cannot be fetched, the preview says so and offers a retry, instead of silently keeping the previous picture on screen (or an empty placeholder). |
| global-setup.js | The suite serves dist/ with `vite preview`. A missing or STALE build would test code that is no longer in src/, and a stale one passes quietly -- so refuse both here instead of letting a green run mea |
| head-pose-hud.spec.js | The head-pose HUD in a real browser: on the Processing tab's live panels (where the render is watched) and in Face Swap's collapsible diagnostics section. Its states, its numbers, its request budget,  |
| helpers.js | Open Settings on a given tab ('providers' / 'cpu' / 'paths' / 'diagnostics' / 'general') by seeding the value the app itself persists (store/workspaceStore.js, key roop_workspace), before the page loa |
| history.spec.js | not JSON |
| identity-mixer.spec.js | Stage 20 in a real Chromium, against the production build with the mock server standing in for app/api.py: the Latent Identity Mixer (Face Manager's Morph Lab, Face Swap's Identity Blender) and the Id |
| idle-cpu.spec.js | An idle Face Swap tab whose backend is FAILING must not cost much more CPU than one whose frames load: within 2x. Before the retry policy a 500 made the tab re-request, re-fail and re-render ~6.5 time |
| idle-requests.spec.js | An idle Face Swap tab must not keep hitting /api/target/preview. * preview answers 500 -> at most 6 STAGE-FRAME requests in the 30 s window * preview answers a PNG -> no preview request at all once th |
| keyboard.spec.js | Press Tab until the focused control's name matches; returns the presses taken. |
| mouth-custody.spec.js | __mock/* hooks are not proxied by `vite preview`; they are on the mock's own port (as compile-banner.spec.js does). |
| nav-visibility.spec.js | The workstation shell's contract at the widths the app is actually used at. * every route is a button in the rail, on screen and not covered by anything; * the page never scrolls sideways and nothing  |
| offline.spec.js | A local app must not need the internet to look right or to say so honestly. * the fonts are bundled (src/assets/fonts, @font-face in index.css): no tab makes a request to any origin but the app's own, |
| pipeline-controls.spec.js | Resolves with the next autosave body (the whole settings object) that satisfies `want`. |
| pipeline-pill.spec.js | Replace /ws/telemetry with a scripted socket: `frames` in order, `gap` ms apart, the last one repeated. The app also polls GET /api/progress (the socket's fallback) and the mock answers "idle", which  |
| pipeline-report.spec.js | Replace /ws/telemetry with a scripted socket: `frames` are sent in order, `gap` ms apart, the last repeated. |
| playwright.config.js | End-to-end checks for the React UI: axe, idle-request budget, nav visibility and tab-stop count. They run against a PRODUCTION build served by `vite preview`, with the mock server (react-ui/mock-serve |
| run-themes.js | Runs the opt-in theme-contrast sweep (contrast-themes.spec.js). node e2e/run-themes.js check against the recorded baselines node e2e/run-themes.js --update re-record them |
| stage11.spec.js | not JSON |
| stage25.spec.js | Page coordinates of (frame, value) on the graph, for the parameter range [min, max]. |
| stage26.spec.js | The slider's own min / max / step, so expectations follow the control rather than a number copied from it. |
| stage5.spec.js | Stage 5, in a real Chromium against the production build with the mock backend: facemgr Morph Lab (pad + sliders agree, 2-4 identities, save / preview) and the pose radar + Auto-Prune processing the S |
| stage8-virtualization.spec.js | Scroll a virtualized scroller to its true end (its height keeps being re-measured as rows mount). |
| tab-stops.spec.js | How many Tab presses it takes to walk the Face Swap tab once. Counted by pressing Tab for real (so display:none, disabled, inert and tabindex all count the way the browser counts them) until focus ret |
| timeline.spec.js | unsupported: the count stays 0 |
| workstation.spec.js | Stage 2 of the workstation overhaul, in a real browser against the mock backend: * the studio slider gestures (double-click reset, Shift+drag, typed values); * the Face Swap inspector (recipes, paddin |

#### `react-ui/mock-server/` (5 files)

| file | purpose / key symbols (from header) |
|---|---|
| identityMock.ts | BlendRecipe.from_payload. |
| mockQueue.ts | {group, index, count} when this job is one slice of a split render (Batch Matrix stamps it). |
| mouthCustody.ts | Why AUDIO_DRIVEN_SYNC runs as SOURCE_MOUTH_RESTORE for this clip, or ''. |
| pipelineTypes.ts | A running job's three stages, placed by overall progress (0..1): 0-10% Stage 1, 10-25% Stage 2, the rest Stage 3. |
| server.ts | This is a MOCK of app/api.py: simulated swaps, invented telemetry, SVG placeholders instead of frames. It exists so the React UI can be developed without a GPU or the Python backend. Nothing in produc |

#### `react-ui/public/` (2 files)

| file | purpose / key symbols (from header) |
|---|---|
| favicon.svg | SVG asset |
| icons.svg | SVG asset |

#### `react-ui/scripts/` (12 files)

| file | purpose / key symbols (from header) |
|---|---|
| api-callsites.mjs | API call-site scan: every request the UI makes states whether it may outrun the 15 s default deadline, and what it says matches src/longRunning.js. node scripts/api-callsites.mjs # exit 1 on any viola |
| audit-truncation.mjs | Layout leak audit for filename / metadata displays. WHAT ACTUALLY CAUSES A LEAK --------------------------- The common advice is "put min-w-0 on every flex child that truncates". That is too broad, an |
| probe-minwidth-rule.mjs | Establish the ACTUAL rule for when min-w-0 is required, empirically. The folklore is "always add min-w-0 to flex children that truncate". The CSS spec is narrower: a flex item's automatic minimum size |
| test-audit-truncation.mjs | Self-test for audit-truncation.mjs. An auditor that reports zero findings is worthless unless it is known to fire on a real leak. These fixtures pin both directions: the shapes that must be reported,  |
| test-verify-offline-bundle.mjs | ! tailwindcss v4 / MIT / https://tailwindcss.com |
| unreachable-modules.mjs | Unreachable-module scan: every source module under src/ must be reachable from the app's entry point, or the build is not telling the truth about what ships. node scripts/unreachable-modules.mjs # exi |
| verify-backend-served-ui.mjs | Acceptance check: does the shipped UI load and work when served by the BACKEND, with no Vite process anywhere? This is the cross-device regression this file exists for. The launcher used to run `vite  |
| verify-build.mjs | Build preconditions and postconditions for the launcher scripts. node scripts/verify-build.mjs vite # before `npm run build`: the vite executable is installed node scripts/verify-build.mjs dist # afte |
| verify-containment.mjs | Measure each of the four audited sites with its REAL markup and the REAL production CSS, before vs after, and report whether the fix changes anything. Method note: the container must NOT clip, or the  |
| verify-offline-bundle.mjs | A local app must look right and say so with the network cable pulled. This scans the PRODUCTION bundle (dist/) for anything that would reach another machine: 1. a FETCH position - <link href>, <script |
| verify-telemetry-e2e.mjs | End-to-end proof that telemetry reaches a REAL browser from the REAL backend. Everything before this is a unit test or a measurement. This is the only check that exercises the whole path the user actu |
| verify-trt-settings-ui.mjs | PLAYWRIGHT_REQUIRE_FROM: a directory whose node_modules holds playwright-core (e.g. an npx cache entry); defaults to this package. CHROME_PATH: the Chromium executable to drive. |

#### `react-ui/src/` (16 files)

| file | purpose / key symbols (from header) |
|---|---|
| App.css | stylesheet |
| App.jsx | ROUTE LOADING, in two tiers. EAGER: Face Swap and Processing are statically imported, so they ship in the entry chunk. Face Swap is the default route and Processing is where a run lands the instant St |
| api.js | `timeout:` for a call whose path is computed (a wrapper like `act(path)`): 0 if the registry lists it as long-running, else the default. |
| failureLog.js | ignore |
| icons.jsx | â”€â”€ Icon vocabulary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ One place that decides what every recurring concept in the app LOOKS like, built on lucide-react (already a dependency, and t |
| index.css | â”€â”€ Typography, vendored â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Plus Jakarta Sans used to be pulled from fonts.googleapis.com. This app is launched by Pinokio and runs entirely against 127.0 |
| longRunning.js | The endpoints that are ALLOWED to run past api.js's default 15 s deadline. api.js gives every request 15 s unless the call passes `timeout: 0`. That default exists because a backend that accepts the s |
| main.jsx | NOTICE.md's intended-use terms, once per install, before anything else. |
| motion.jsx | â”€â”€ Shared motion foundation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ One vocabulary of spring physics + entrance variants so every animated surface across the app feels like the same instrument. |
| mutationBus.js | Subscribe; returns the unsubscribe. |
| pipelineStatus.js | How long the banner stays up as a receipt after a build it watched finish. |
| themeVars.js | â”€â”€ Custom theme derivation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ The 38 built-in themes are hand-written blocks of CSS variables in index.css. That is fine for a curated set someone tuned by |
| themes.js | Single source of truth for UI themes. `className` matches the .theme-* blocks in index.css (empty string = the :root default). `accent` / `bg` drive the swatch preview in the theme gallery; `mode` tag |
| useApi.js | The lifetime of one mounted component as an AbortSignal. Pure (no React) so it can be tested; the hook below drives it from effects. React StrictMode mounts, unmounts and mounts again in development,  |
| usePipelineStatus.js | The backend's current compile / canary state, re-rendering when it changes. |
| useTelemetrySocket.js | Subscribe to live run telemetry. @param {(frame: object) => void} onFrame called with each telemetry frame. @param {boolean} enabled set false to tear the socket down entirely. @returns {{connected: b |

#### `react-ui/src/assets/` (5 files)

| file | purpose / key symbols (from header) |
|---|---|
| fonts/plus-jakarta-sans-latin-ext.woff2 | bundled font (offline, no Google Fonts) |
| fonts/plus-jakarta-sans-latin.woff2 | bundled font (offline, no Google Fonts) |
| hero.png | image asset |
| react.svg | SVG asset |
| vite.svg | SVG asset |

#### `react-ui/src/components/` (43 files)

| file | purpose / key symbols (from header) |
|---|---|
| AutoTunePanel.jsx | Auto-tune: replays the LAST render on a short stretch of its own target and measures provider x cross-frame swap batch, then NVENC presets. Screening (100 frames, each arm twice) only ranks; a setting |
| BatchSwap.jsx | Helper to convert index to target preview URL |
| BenchmarkPanel.jsx | (no header comment) |
| CommandPalette.jsx | Lightweight subsequence fuzzy match with a simple relevance score. |
| CompileBanner.jsx | The backend is building a TensorRT engine or running the fidelity canary that follows it. Both take minutes on a cold cache and have no frame counter, so without this the app looks hung and (before ap |
| EnhancerSettings.tsx | (no header comment) |
| EnvironmentHealth.jsx | â”€â”€ Environment health & update compatibility â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Two questions this app could not answer while idle: "is this machine set up correctly right now?" The structured runtime r |
| ErrorBoundary.jsx | The app's one error boundary. It used to be two: a bare copy in main.jsx (the whole app) and this one (a tab panel), which had drifted apart -- different logging, no retry at the top level, and a "cra |
| Extras.jsx | Human labels for the frame post-processor subtypes. |
| FaceManager.jsx | eslint-disable react-hooks/exhaustive-deps -- intentional: subscribe once; notify is stable |
| FaceSwap.jsx | (no header comment) |
| Gallery.jsx | no history yet |
| Home.jsx | â”€â”€ Home â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ A launch pad and a ledger. Top to bottom: the live run when there is one, the quick-launch dropzone (a source and a target, t |
| LiveTelemetry.jsx | Render prop over useThrottledSelector. |
| ModelSplash.jsx | Startup model check (app/roop/model_integrity.py via GET /api/models/integrity). While the backend is hashing or downloading the manifest models this covers the app with per-file progress, because not |
| NumericInput.jsx | Evaluate what is in the field. Returns false only when the field must stay open (Enter on a bad expression). |
| OutputCompare.jsx | Put two finished renders side by side. The comparison suite in the Face Swap tab compares before/after of the CURRENT preview, and Run History diffs two runs' SETTINGS. Neither answers the question yo |
| OutputVideoPlayer.jsx | OutputVideoPlayer â€” a finished render, and optionally the original beside it. RANGE REQUESTS. The <video> element seeks with HTTP Range requests and the backend answers them properly (routes_output._s |
| PersonGroups.jsx | Coarse pose buckets we consider "primary coverage" for a person. Anything the backend labels (e.g. "Left Profile + Up Tilt") is matched against these by substring so combos still count toward the base |
| PipMiniViewport.jsx | The floating mini-viewport (280x158 by default, resizable): Face Swap's current picture, kept in a corner of the screen while you work somewhere else. Where the picture comes from, and how it is renew |
| Processing.jsx | The Processing tab. All of this used to live inside the Face Swap tab, which meant a run took that tab over: the settings sidebar, the asset rail, the timeline and the preview controls were all hidden |
| QualityProfilesModal.jsx | (no header comment) |
| QualityReport.jsx | Grade ring |
| RecognitionPanel.jsx | Embedding backend for the identity API: which recogniser and which provider, what this machine is best at, and applying it. Data comes from /api/recognition/models and /current, applying is POST /api/ |
| RunHistory.jsx | Run History â€” a browsable record of every completed swap. The live DiagnosticsPanel answers "how is THIS run going" and then vanishes the moment the job ends. This view is the durable other half: each |
| RuntimeSections.jsx | â”€â”€ The runtime report: all fourteen sections â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ api/progress carries `runtime`: the backend's own snapshot of what it is doing, as fourteen named sections (roop/runtime_st |
| Settings.jsx | A Section that participates in the settings search and the "only changed" filter. With either active it keeps just the controls that match (or the whole section when its own title matches the query),  |
| StorageManager.jsx | (no header comment) |
| TermsGate.jsx | First-run screen: NOTICE.md's intended-use terms, served by the backend (`GET /api/terms`), must be accepted once per install (and again whenever the text changes -- the version is a hash of it). The  |
| ThemeGallery.jsx | Swatch preview |
| ThemeStudio.jsx | â”€â”€ Theme Studio â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Authors a custom theme from the six decisions that actually matter, and previews it live on the real page rather than in a sw |
| TrtCachePanel.jsx | TensorRT engine cache: what the running backend builds into, what is stale, and clearing it. Every number comes from /api/trt_cache (routes_trt_cache.py). |
| confirm.jsx | Promise-based, themed replacements for window.confirm / window.prompt so destructive actions don't pop a jarring native OS dialog in the premium UI. Any component can `await confirmDialog({...})` with |
| constants.js | Shared non-component constants. Kept out of ui.jsx so that file only exports React components (satisfies react/only-export-components / Fast Refresh). |
| hoverScrub.js | 0..1 across the card for a pointer x. A zero-width card is 0, not NaN. |
| lazyPanel.js | Idempotent: safe to call from a render-phase hook and from an event handler. |
| outputUrl.js | Absolute URL for an output record from /api/progress (`progress.output`). |
| recognitionSync.js | App.jsx autosaves the WHOLE settings object after any edit. If a successful recognition apply did not also land in that object, the next unrelated edit (a slider, a toggle) would post the old recognit |
| runtimeRows.js | A section's `values` as flat [label, text] rows: nested objects become `parent.child`. |
| settingsCatalog.js | GENERATED by tools/gen_settings.py from app/settings.py (UI_SETTINGS). Do not edit: change settings.py and run `python tools/gen_settings.py`; app/tests/test_settings_schema.py fails while this file i |
| settingsDiff.js | Shared vocabulary for describing and diffing a run's settings. Lived in RunHistory.jsx until the Outputs tab needed the same thing: when you put two rendered files side by side, the question that foll |
| settingsTabs.js | The tab holding setting `key`, given its catalog section name. |
| ui.jsx | Panels rise in on mount (cinematic reveal) and lift on hover. â”€â”€ Elevation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Tilt + cursor glare used to be the DEFAULT for every Card, whic |

#### `react-ui/src/components/extras/` (1 files)

| file | purpose / key symbols (from header) |
|---|---|
| VideoFramesetExtractor.jsx | â”€â”€ Video to faceset (.fsz) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Pull a faceset out of a video without leaving the Editor: load the clip, scrub to a frame where the person is well lit and  |

#### `react-ui/src/components/facemgr/` (3 files)

| file | purpose / key symbols (from header) |
|---|---|
| MorphLab.jsx | Embedding Mixer & Morph Lab. Pick 2-5 identities (Face Swap source faces, saved .fsz identities, or the Face Manager's own set), then blend them: by sliders (auto-normalised to 100%, or free-weighted) |
| PoseRadar.jsx | Face Bank Quality & Pose Radar. Left: a five-axis radar of how many faces the set holds in each pose bucket (frontal, left / right profile, high / low angle) - a lopsided polygon is a set the swap wil |
| morphPad.js | Vertex positions (0..1) for n identities, in the order the identities were picked. |

#### `react-ui/src/components/faceswap/` (118 files)

| file | purpose / key symbols (from header) |
|---|---|
| ABReferencePanel.jsx | A/B reference: a previous render (sent from Outputs) wiped against the LIVE preview, frame for frame. A (left of the divider) the reference - an image, or a video held on the frame matching the playhe |
| AIScannerOverlay.jsx | A single soft scanner sweep signals "working" â€” calm, no HUD clutter. |
| AmbilightGlow.jsx | AmbilightGlow Renders a soft dynamic blurred glow behind the media preview canvas. Mirrors the active previewSrc image to generate natural ambient illumination. |
| BroadcastScopesDock.jsx | Stage 18: Broadcast Scopes Dock (RGB Parade, Luma Waveform, Chroma Vectorscope). Driven by off-thread / downsampled OffscreenCanvas WebGL2 readbacks throttled via requestAnimationFrame to preserve Sta |
| ColorGradingPanel.jsx | Stage 18: WebGL2 3D .cube LUT Grading Panel. Controls: - .cube LUT file loading with strict DOMAIN_MIN / DOMAIN_MAX validation - 3D LUT Blend Opacity slider (0.0 .. 1.0) - Display-Only Preview vs Bake |
| CompareGrid.jsx | Generic side-by-side comparison grid with a single shared zoom/pan, used by both the Enhancer grid and the Mask-engine grid. Each cell shows one variant's rendered preview (or a live "Renderingâ€¦" spin |
| CompareSlider.jsx | Ref to a PreviewCanvas. When present the wipe is applied on the canvas. |
| ComparisonGridPanel.jsx | Shared comparison-grid chrome for enhancer, mask, swapper, and upscaler comparisons. The parent owns selection state and preview loading; this component owns only the repeated selector and result-grid |
| ConsoleMirror.jsx | The mirror for one snapshot. Exported on its own so it can be rendered from a fixture. |
| DiagnosticsPanel.jsx | Diagnostics shown inside the Preview box while a job runs. The processing panel used to be a status line, a bar and a short console floating in a mostly empty 16:9 box. A long run is exactly when you  |
| EnhancerPipelineStack.jsx | â”€â”€ The post-swap chain, as four stages â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Swapper Core -> Face Restorer -> Texture & Grain Closer -> LAB Color & Skin-Tone Match Each stage is one card: an enable sw |
| FaceContextMenu.jsx | The right-click menu on a detected face. Rendered in a portal on <body> and positioned `fixed`, so the stage's overflow clipping, its zoom / pan transform and its stacking context cannot cut it off, a |
| FacesetLibrary.jsx | Persistent, named .fsz facesets on disk. Save the selected source faceset here once and reload it any time without re-uploading. Point the library folder (Settings â†’ "Faceset library folder") at OneDr |
| FileDrop.jsx | Drop zone / file picker. `progress` is the live upload state from api.js's xhrUpload â€” see there for why uploads are XHR rather than fetch. Passing it in turns the indeterminate spinner into a real ba |
| FloatingActionDock.jsx | FloatingActionDock â€” the always-visible action bar at bottom centre: preview, workspace layout, ambilight, preset studio, pop-out, drawer toggles. (Not Start/Cancel: the run bar owns those. The Face S |
| FrameFlamegraph.jsx | Live Per-Frame Bottleneck Flamegraph & Telemetry Export (#/processing). Horizontal Stacked Frame-Time Flamegraph Bar (100% = total ms/frame): Decode (Cyan) / Detect/Align (Purple) / Swap (Emerald) / M |
| FrameUnavailable.jsx | "Frame unavailable - Retry": the stage could not fetch the frame for the playhead after every retry (useThrottledFrameRequest), so whatever picture is up -- a swap of an earlier frame, or nothing -- i |
| HeadPoseHud.jsx | The number carries the reading; the bar is a glance (centre = 0). |
| IdentityBlender.jsx | Identity Blender: latent blend of up to five source identities on the ArcFace hypersphere (Slerp / spherical mean, or the older normalised sum), attribute offsets along fitted directions, optional de- |
| InspectorGroup.jsx | storage unavailable: the state lasts for this session |
| InteractivePreview.jsx | (no header comment) |
| LinkedPaddingControl.jsx | â”€â”€ Mask padding: one slider, or four â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ mask_top / mask_bottom / mask_left / mask_right grow the face mask beyond the landmark hull on each side. Nearly everyone w |
| LiveProcessingPeek.jsx | LiveProcessingPeek The image shown inside the processing box while a job runs. Two different things can be on screen here, and the badge says which: - LIVE â€” the frame the pipeline most recently finis |
| MediaTabSessionBar.jsx | MediaTabSessionBar Allows working with multiple media targets in a tabbed session workspace. |
| MiniSwitch.jsx | A compact on/off switch for a card header (the enhancer stack's stages). The full-size labelled one is `Toggle` in ui.jsx; this carries no visible label of its own, so `label` is mandatory and becomes |
| MouthCustodyPanel.jsx | "Audio & Mouth Custody": who owns the swapped mouth, and which audio the delivered file carries. faceswap the full panel (mode + the lip-sync engine controls + the audio track + the slip). extras `sho |
| PanelResizeDivider.jsx | Resizable panel divider for the Face Swap workstation. Supports horizontal dragging for panel widths (Left Inspector, Right Media Rail) and vertical dragging for panel heights (Bottom Timeline Deck).  |
| ParameterSweepModal.jsx | Synchronized Zoom & Pan state across all 9 tiles |
| ParserRegions.jsx | Which parsed parts of the face the Face Parser mask counts as "swap this". The engine has always produced all 19 CelebAMask-HQ classes and always thrown 10 of them away against a hardcoded list â€” so h |
| PipelineControls.jsx | The mode is one control over three settings; the bypass one is also addressable by its own name, so the command palette entry for "Auto-Bypass Restorer" lands here. |
| PipelinePill.jsx | The silent-video fallback, said out loud. Not aria-hidden: a state change a screen-reader user needs, announced once. |
| PipelineReport.jsx | A finished report: headline numbers, then one block per stage. `report` is the backend's summary object. |
| PopoutPreviewManager.js | PopoutPreviewManager Broadcasts preview frames and media updates to a detached pop-out window. Uses BroadcastChannel API with localStorage fallback. |
| PreFlightLinter.jsx | Smart Pre-Flight Render Readiness Linter & 1-Click Auto-Fix. Runs real-time diagnostics before single-video render or Batch Matrix jobs: Check 1: Disk Space Custody (free space vs required output + te |
| PresetStudioModal.jsx | PresetStudioModal Preset Studio & Recipe Manager modal for Roop Ultimate. Includes curated 1-click quality recipes, visual parameter diffs, export/import, and recipe sharing. |
| PreviewCanvas.jsx | â”€â”€ The preview stage, as an UNCONTROLLED canvas â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ What this replaces: two <img> layers whose `src` came from React state holding a base64 data URL, cross-faded by animating an |
| ProcessingDock.jsx | ProcessingDock Interactive control dock positioned inside the processing stage box. Gives direct control over pause/resume, desktop alerts, and cancellation. There was a "Power Saver / Eco Mode" butto |
| ProcessingTerminal.jsx | Terminal-style live feed shown inside the Preview box while a job runs. Polished features: - PART TABS: Grouping by resume chunk, with visual finalized badges. - PINNED STATUS LINE: Real-time progress |
| ProjectsPanel.jsx | â”€â”€ Persistent projects â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ api.py creates a project record on EVERY /api/swap (see `_create_processing_project`, called from the swap route) and upda |
| ProviderBenchmark.jsx | "Benchmark Pipeline Providers": one click measures each part of the pipeline on the CPU, on the primary provider (TensorRT when active) and on native OpenVINO when an Intel accelerator exists, plus th |
| ProviderMatrixView.jsx | The provider-matrix result as markup. Pure (props in, markup out) and free of any API import, so the render check can drive it with a real backend result. |
| QuadInspectionGrid.jsx | 4-Up Quad Inspection Grid. Simultaneously renders 4 synchronized sub-viewports (zoomed & panned together in 100% lock-step): [1] Original Target / [2] Parsed Occlusion / Box Mask [3] Unrestored Raw Sw |
| QueuePanel.jsx | Live Active Job Render Peak & Progress Monitor |
| RadarNavigator.jsx | Mini Radar Navigator (PiP). Automatically appears in the preview corner when zoomed past 1.5x, showing the full frame with an interactive draggable viewport rectangle. |
| RecipePresetBar.jsx | â”€â”€ The quick recipe bar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Four one-click looks at the top of the inspector, plus the user's own presets and the preset management that used to be thr |
| RotoLayer.jsx | synthetic event |
| RunModelsPanel.jsx | RunModelsPanel â€” WHICH MODELS ARE ACTUALLY LOADED AND RUNNING. The Processing tab could say how fast a run was going and how far it had got, but not what was doing the work. That is the first question |
| SegmentBar.jsx | The segment strip under the timeline. Segments exist because a run has exactly one trim range (it lives on the target entry, not in the swap payload), so rendering three shots out of a long clip is th |
| SkinToneScope.jsx | 160x100px Studio-Grade Mini RGB Parade & Skin-Tone Vectorscope. Samples the active face ROI to help dial in LAB color transfer accurately. |
| SliderTrackerBar.jsx | Slider steps are 0.01/0.05, so a value that has been through parseFloat or a JSON round trip can sit a float ULP off the table it is compared against. Both users of that fact need the same tolerance:  |
| SnapshotSlots.jsx | bitmap closed under us: leave it blank |
| StereoVRViewport.jsx | Stage 21 â€” WebGL2 Equirectangular VR180/360 Viewport & Stereoscopic 3D Inspector. Implements: - Interactive Pan-Tilt-FOV drag and wheel zoom. - Equirectangular 180/360-degree spherical ray-traced proj |
| TemporalModeControl.jsx | â”€â”€ Temporal stability: one choice over a dozen switches â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Off / Smart-Skip ROI / Balanced EMA / Cinema Cut-Aware, as a segmented control. Picking one writes that mode's patch (face |
| Timeline.jsx | The clip timeline - a multi-track NLE surface for the target video. Top to bottom: the whole-clip MINIMAP (always visible, with zoom handles), an adaptive RULER (timecode -> seconds -> exact frame num |
| abReference.js | A reference is {url, name, kind: 'image'/'video', fps?, offset?}. Anything else is rejected. |
| batchMatrix.js | The Batch Matrix's staging logic, kept free of React so it can be run as code by `.render-check/batch-matrix-check.mjs` (and, through it, by app/tests/test_batch_matrix_queue.py) instead of being re-i |
| coalescedPreviewController.ts | Coalesced Preview Controller Guarantees: 1. Single-Flight Execution: at most 1 in-flight preview request on the GPU / network at any time. 2. Animation Frame Coalescing: multiple rapid changes (e.g. d |
| cubeLut.js | WebGL2 3D .cube LUT Parser, Domain Validator, and Premultiplied Alpha Engine. Supports standard Adobe / DaVinci Resolve .cube specifications: - LUT_3D_SIZE <size> - DOMAIN_MIN <r> <g> <b> - DOMAIN_MAX |
| dashboardModel.js | Same text as the console: `01m 14s`, `1h 02m 05s`. |
| decoder.worker.js | @type {Map<number, AbortController>} |
| defaults.js | Baked-in defaults for every setting editable inside the Face Swap tab. Snapshot of the user's preferred configuration (reconciled 2026-08-22 against app/config.yaml and app/settings.py). The "Reset de |
| faceContextModel.js | Where a menu of size w x h opens for a click at (x, y), kept inside a vw x vh viewport. All in ONE coordinate space (the caller passes visual pixels - what getBoundingClientRect and window.innerWidth  |
| faceMapping.js | Target-person -> source-face mapping, shared by FaceSwap.jsx (which sends it to the backend as `face_mapping`) and PersonGroups.jsx (which must SHOW the same decision in its per-person dropdown). It l |
| faceRoles.js | The role of ONE detected face. personId stable id of the captured person this box was matched to, or null/undefined for a face nobody captured Returns { role, excluded, canExclude, reason } |
| faceSample.js | The image behind a stage layer as an ImageBitmap. A bitmap the stage already holds (`frame`) is used as it is and is never closed here - the owner closes it. Otherwise the url is fetched and decoded;  |
| falseColorZebra.js | False-Color Luminance & Skin-Tone Zebra Overlay Engine. Professional broadcast-grade IRE luminance zones and exposure zebra overlay. Highlights: - Clipped highlights (>95 IRE) with 45Â° diagonal warnin |
| flamegraphModel.js | (no header comment) |
| frameDecoder.js | â”€â”€ Main-thread facade over the decoder worker â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ One shared worker for the whole app (frame decodes are already serialised by the backend's single video decoder, so a pool wo |
| headPoseModel.js | Half-range of each dial's bar, in degrees: a head at this angle pins the marker to the end of its bar. |
| heatmap.js | Computes /top - base/ * gain with false-color LUT mapping directly onto target canvas ctx. Runs at 60fps on modern canvases. |
| identityBlend.js | A copy of `recipe` carrying only the offered dials (each finite, clamped to Â±1); safe for null / partial input. |
| identityKeyframes.js | The backend's 0-based absolute frame index for a 1-based UI frame. |
| identityLock.js | The label the pin button and the pin list show for a person. |
| lastPreview.js | The last still the Face Swap tab had on screen. The processing view lives in its own tab now, and tabs are mutually exclusive â€” so while the run is being watched, the Face Swap component that owns `pr |
| logRing.js | One severity per line, most severe first: ERROR > WARN > PERF > INFO. A line is never in two filters, so the chip counts add up and "ERROR" never shows a warning. |
| maskBrush.js | Where an `object-fit: contain` image of w x h sits inside `rect` (centered). |
| maskStrokes.js | Canvas-pixel sample -> normalised (x,y by their own axis; radii by width, as the canvas is scaled uniformly). |
| mouthCustody.js | The slip nudge buttons, in ms. |
| objectUrls.js | Create an object URL owned by `owner`. Releasing the owner revokes it. @param {Blob} blob @param {object/string/symbol} owner any stable token (a ref object works) |
| pillModel.js | Backend stage state -> the pill's step state. |
| pipelineControlsInfo.js | The help text behind the 3-stage pipeline controls. Moved here verbatim from FaceSwap.jsx / Settings.jsx when the controls became one shared component (PipelineControls.jsx), so the two tabs cannot dr |
| pipelineRules.js | The restorer auto-bypass (roop/smart_restorer_bypass.py) only fires for these swappers. |
| playbackWindow.js | â”€â”€ Which frame the playback prefetcher should ask for next â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Pure, so .render-check/player-check.mjs can test it without a browser. cur the playhead start/end the In/Out points (inclusiv |
| processingSelection.js | Build the canonical selection from the current UI context. `selectionState` is the existing serializable target-selection contract (buildTargetSelectionState) â€” kept inside the selection so the backen |
| providerMatrix.js | Per-stage table rows: one per measured stage. A missing arm is a dash; an arm that errored says so. |
| recipes.js | One key of a recipe against the live settings. `default_det_size` is true when absent, like everywhere else. |
| reportFormat.js | done / running / failed / skipped / restored / pending, for a state chip. |
| retryPolicy.js | Consecutive failures that exhaust a key: the first attempt plus MAX_RETRIES retries. |
| rotoModel.js | Set a Bezier handle (an OFFSET from its anchor). With `mirror` the opposite handle takes the negated offset, which is a smooth point; a click-drag with the pen creates exactly that. |
| sweepParameters.js | (no header comment) |
| targetPayload.js | Apply a target-faces API payload (the shape every /api/target/* mutation returns, via api._target_faces_payload) to Face Swap's parallel state. Lifted out of PersonGroups so that anything else which c |
| temporalModes.js | The mode the settings are in: 'off' / 'smart_skip' / 'balanced' / 'cinema', or 'custom'. Smart-Skip is not part of balanced/cinema (a mode that leaves it alone), so it does not disqualify them; it IS  |
| trackerConfig.js | Single source of truth for the Slider Tracker bar. These numbers used to live in four places â€” TRACKER_SLIDERS' defaultVal, the 'Default' built-in preset, resetTrackerSliders() and the bypass object i |
| useClipAdvisor.js | @param targets the loaded target list (only its length is read here) @param selTarget index of the target to analyse @param settings the current Face Swap params, sent so the backend can return only a |
| useCompareGrid.js | @param {string} storageKey localStorage key holding the chosen list. @param {string[]} defaults used when nothing valid is stored. @param {(x:any)=>boolean} [isValid] per-item check; defaults to "a st |
| useEngineRun.js | The live engine state (is a canary running now, for which model) from the telemetry store. Subscribes with atomic shallow selector to avoid redundant re-renders. |
| useGridPreviewLoader.js | eslint-disable react-hooks/exhaustive-deps -- intentional: `load` is rebuilt every render and closes over current values on purpose; the effect is keyed on the things that should actually re-render th |
| useIdentityKeyframes.js | private mode / quota: keyframes are a convenience, not state we own |
| useJobRecovery.js | â”€â”€ Reattach to a render that is already in flight â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Runs once on mount, then keeps a slow heartbeat. See jobStore.js for why the order (ask the server FIRST, then reconcile the  |
| useLiveCam.js | backend gone |
| usePlaybackBuffer.js | @param frame current playhead (the loop both reads and writes this) @param setFrame playhead writer @param selTarget index of the target being played @param maxFrames total frames in that target @para |
| useProfiles.js | Manages named setting presets ("profiles"). Persists to localStorage (instant/offline) AND the backend (survives cache clears, shareable via profiles.json on disk). Extracted verbatim from FaceSwap.js |
| useQueue.js | backend not up yet â€” the next poll or action will catch up |
| useRenderLite.js | private mode |
| useRunCompleteAlert.js | private mode |
| useRuntimeEstimate.js | @param settings current Face Swap params @param estFrames frames the run would cover @param faceCount faces in the current preview frame â€” a density hint @param processing whether a run is in flight ( |
| useSegments.js | private mode / quota â€” segments are a convenience |
| useSnapshotSlots.js | Empty slot or `overwrite` captures; a filled slot recalls. |
| useTargetAudio.js | unknown stays unknown |
| useTelemetry.js | System telemetry (GPU/VRAM/CPU/RAM/threads) for the HUDs. ONE poll for the whole app, reference-counted by the components that want it, writing into the telemetry store. It used to be a private setInt |
| useThrottledFrameRequest.js | â”€â”€ One frame request at a time, throttled, and cancelled the moment it is superseded â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ This replaces useSequentialImage, which was right abou |
| useUserDefaults.js | storage blocked â€” non-fatal |
| useViewPersistence.js | @returns a ref holding `{ target, frame, image }` captured at mount, or null when there was nothing fresh to restore. A ref, not state, because the rehydrate callback reads it once and re-rendering on |
| useWorkspaceLayout.js | â”€â”€ What each workspace mode actually does â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ The dock offered four modes but only 'cinema' changed anything; 'dual' and 'timeline' were menu entries that did nothing at a |
| utils.js | audio not available |
| webgl2GradingPipeline.js | Hardware-Accelerated WebGL2 Float 3D .cube LUT & Color Grading Pipeline. Implements: - Linear float space execution (IEC 61966-2-1 piecewise transfer) - Three-way Lift/Gamma/Gain color wheels - Kelvin |
| wipeTransform.js | @param {number} pct 0..100, how much of the stage shows the processed side @param {number} angle 0 = linear curtain, otherwise the wipe direction in degrees @param {boolean} horizontal linear curtain  |
| zoomPan.js | Shared zoom/pan maths for the preview stage and the comparison grids. Both surfaces render their content as `scale(z) translate(pan.x/z, pan.y/z)` or direct affine 2D transformation matrices about the |

#### `react-ui/src/components/faceswap/curves/` (5 files)

| file | purpose / key symbols (from header) |
|---|---|
| CurveEditor.jsx | A segment-shape glyph for the interpolation buttons. |
| curveHistory.js | The undo-history label for a change to the parameter curves. The channel's value is { clipKey: { paramId: keyframes } } (only clips edited this session appear), so the label is found by comparing the  |
| curveModel.js | The curves on offer. `key` is the settings / swap-payload key a direct curve writes (the same name in both, which is why one patch serves the preview and the render). `quantum` is the render's step: a |
| curvePieces.js | Cut override pieces (overridePieces output) further, wherever a curve moves by a render step. @returns {{pieces: Array<{start, end, override, curves: object/null}>, widen: number, steps: object}} `wid |
| useParameterCurves.js | private mode / quota |

#### `react-ui/src/components/faceswap/timeline/` (15 files)

| file | purpose / key symbols (from header) |
|---|---|
| IdentityKeyframeInspector.jsx | Edit one identity keyframe: which curve the segment to the NEXT keyframe follows (with a draggable bezier when it is a Bezier), and the share of each blended source at this keyframe. The numbers under |
| IdentityKeyframeLane.jsx | The Identity Keyframe Track: one row beneath the film strip (where the chapter markers are carried down), with a diamond per identity keyframe and, between neighbours, a glyph of the curve that segmen |
| Lanes.jsx | The lane components. Each is one <canvas> that repaints from plain data when the view or the data changes; none of them re-renders on a playback tick (the playhead is a DOM overlay owned by Timeline), |
| Minimap.jsx | The always-visible overview bar above the ruler: the whole clip at a glance (render state, scene cuts, who is on screen) with the visible window drawn on it. Drag the lit window to pan, drag either ha |
| OverrideInspector.jsx | One column: the Face Swap stage is ~420 px wide in the default layout, and two columns left the range inputs ~28 px (collapsed to nothing - the control was there and could not be used). |
| hit.js | The cut within `tolFrames` of `frame` (nearest wins), or null. |
| laneModel.js | Sorted, merged, clipped to [lo, hi]. Intervals that touch (end + 1 === start) merge too. |
| lanes.js | Theme colours the canvas needs, read from the live CSS variables (canvas cannot use var()). |
| overrides.js | Bypass is expressed through the face-match threshold, which "swap every face" modes never read. |
| shuttle.js | Next shuttle state for a J / K / L key press. Unknown keys return the state unchanged. |
| useAudioTrackWave.js | The waveform of a BOUND external audio track (the file /api/lipsync/audio/add returned), for the timeline's audio row. Same envelope as the target's own (`GET /api/audio/waveform` answers from the ide |
| useLaneCanvas.js | A <canvas> that repaints itself when `deps` change and when it is resized, at most once per animation frame however many changes arrive. `draw(ctx, {w, h, theme})` paints in CSS pixels (the context is |
| useSceneOverrides.js | private mode / quota |
| useTimelineData.js | Scene cuts, per-person presence, and the last run's Stage 1/2 maps for one target. `poll` makes it re-ask every few seconds (a run is in flight, so the maps are filling in); otherwise it asks once per |
| viewMath.js | The smallest window the view may shrink to: the frame-by-frame view, FRAME_VIEW_PX per frame. |

#### `react-ui/src/components/history/` (1 files)

| file | purpose / key symbols (from header) |
|---|---|
| CheckpointDiff.jsx | Compare Checkpoint vs. Current Settings. A saved run on the left, what is set RIGHT NOW on the right, one row per parameter that differs ("CodeFormer fidelity 0.50 â†’ 0.80", "Mask feather 12px â†’ 24px") |

#### `react-ui/src/components/home/` (2 files)

| file | purpose / key symbols (from header) |
|---|---|
| QuickLaunch.jsx | â”€â”€ Quick launch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Drop a source face and a target here and go: each lands in the backend's own workspace through the same endpoints the Face  |
| ResumableCheckpoints.jsx | â”€â”€ Resumable checkpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ The latest project checkpoints a run can actually be picked up from. This is a SHORT VIEW over /api/projects, the list Face |

#### `react-ui/src/components/media/` (3 files)

| file | purpose / key symbols (from header) |
|---|---|
| HoverVideoThumb.jsx | A video's picture in a list row: a still of its first frame, and a real <video> only while the pointer is over it. (The Outputs tile does the same with scrubbing on top; see Gallery.jsx VideoHoverCard |
| useHoverVideo.js | A still for a video tile. `url` is the first frame once captured, null when it could not be read (the tile shows its placeholder), and undefined while it is still being fetched. |
| videoThumb.js | Free a media element's decoder NOW rather than whenever the garbage collector gets to it. |

#### `react-ui/src/components/monitor/` (2 files)

| file | purpose / key symbols (from header) |
|---|---|
| ReferenceMonitor.jsx | Stage 19: Pop-Out Clean Feed Reference Monitor (#/monitor). A dedicated chrome-free popup display that renders live swapped frames, split-wipe comparison coordinates, and Stage 12/18 LUT and Zebra ove |
| referenceChannel.js | Stage 19: BroadcastChannel('roop_ultimate_reference_out') Synchronization Bridge. Provides a 60fps zero-copy synchronization bus between the main workstation (InteractivePreview / Extras) and the pop- |

#### `react-ui/src/components/player/` (5 files)

| file | purpose / key symbols (from header) |
|---|---|
| FastCanvasPlayer.jsx | â”€â”€ <FastCanvasPlayer /> â€” encoded frames straight to the GPU â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ A canvas that shows a stream of JPEG frames (the live render view, timeline playback) without React, <img>, base64 or blob UR |
| VideoCompareStage.jsx | â”€â”€ <VideoCompareStage> â€” original vs. result, one WebGL canvas â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Two <video> elements, the OUTPUT (master: its clock, its audio, its controls' state) and the ORIGINAL target (follower, muted |
| fastPlayer.worker.js | â”€â”€ <FastCanvasPlayer>'s render thread â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Owns an OffscreenCanvas transferred from the page, and with it the whole frame path: decode (createImageBitmap), texture uplo |
| framePipeline.js | @param {object} o @param {import('./glRenderer').GlRenderer} o.renderer @param {(bytes: ArrayBuffer, mime: string) => Promise<ImageBitmap>} o.decode @param {(cb: () => void) => void} o.schedule reques |
| glRenderer.js | â”€â”€ GlRenderer: two picture layers, one draw call â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Shared by <FastCanvasPlayer> (pushed JPEG frames, usually inside a worker on an OffscreenCanvas) and <VideoCompareStage> (two |

#### `react-ui/src/components/segments/` (3 files)

| file | purpose / key symbols (from header) |
|---|---|
| SegmentMatrix.jsx | Segment Chunk Matrix. A video rendered in segments is N queue jobs whose finished files wait in `.segments/<split>/` until they are joined. This is the custody view of that set: one pill per chunk (Qu |
| segmentModel.js | 'queued' / 'rendering' / 'completed' / 'failed'. An unknown state reads as queued, never as done. |
| useSegmentGroups.js | The server's view of every split render (`GET /api/queue/segments`): per slice its state, its file on disk, and the join with its verification. Kept apart from useQueue on purpose - the queue answers  |

#### `react-ui/src/components/shell/` (7 files)

| file | purpose / key symbols (from header) |
|---|---|
| HistoryPopover.jsx | â”€â”€ The history stack, from the header â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Undo / redo buttons and a popover listing the last 50 actions. Clicking a past row time-travels to that state in one step: t |
| LayoutSwitcher.jsx | The workstation workspace layout switcher & custom layout manager. Allows switching between built-in presets (Standard, Max Canvas, Inspection Suite, My Roto Setup, Dual-Monitor Color, Minimal Laptop) |
| TelemetryPill.jsx | â”€â”€ Live hardware telemetry: one pill, one popover â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ This is the ONLY place the app shows GPU / VRAM / CPU. It replaces the "Hardware telemetry HUD" banner under the header, th |
| WorkstationHeader.jsx | â”€â”€ The top workstation header â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ One slim bar above every route: where you are, the run chip while a job runs, the workspace layout switcher, the hardware p |
| WorkstationRail.jsx | The accent bar is the selected state; colour on the icon alone would be the only cue otherwise. |
| telemetryReading.js | What the header's telemetry pill reads out of one /api/system/telemetry sample. Pure, so the render-check can feed it the shapes the backend really sends (with and without nvidia-smi) without mounting |
| useQueueSummary.js | How many queue jobs are still to run (running or waiting), for the rail's Processing badge. The queue itself belongs to Face Swap (faceswap/useQueue, mounted only on that tab), so the shell cannot rea |

#### `react-ui/src/components/shortcuts/` (3 files)

| file | purpose / key symbols (from header) |
|---|---|
| KeyChips.jsx | The keys an action is on, as <kbd> chips. "unbound" when the user cleared it. |
| KeyboardShortcutsTab.jsx | Settings -> Keyboard Shortcuts: click an action, press the new combination. * Change replaces all of an action's keys with the one pressed. * + adds a second key (J/K/L-style alternates, or Ctrl+= bes |
| ShortcutOverlay.jsx | the opener is gone |

#### `react-ui/src/components/virtual/` (2 files)

| file | purpose / key symbols (from header) |
|---|---|
| VirtualScroll.jsx | Windowed rows with measured (variable) heights. `renderRow(index)` returns the row's content; the wrapper owns positioning and measurement, so a row that grows (an expanded report, a wrapped log line) |
| useContentWidth.js | [attachRef, width]: the element's content width in layout px, kept current by a ResizeObserver. Layout px (clientWidth), not getBoundingClientRect, so it is in the same units as scrollTop and survives |

#### `react-ui/src/history/` (3 files)

| file | purpose / key symbols (from header) |
|---|---|
| historyChannels.js | Report `value` as the current state of channel `id`. `apply` is read through a ref so a fresh closure each render never re-registers the channel. `enabled` false keeps the channel out of the history u |
| historyDom.js | True when the browser's own text undo should win. A focused range slider is NOT text. |
| historyEngine.js | True when two state maps are equal channel by channel. `eq[channel]` overrides Object.is. |

#### `react-ui/src/numeric/` (3 files)

| file | purpose / key symbols (from header) |
|---|---|
| mathExpr.js | Typographic operators and decimal commas, folded to ASCII before anything reads the text. |
| scrubMath.js | Pointer travel before a press becomes a drag; below it the press is a click. |
| useScrub.js | capture already gone |

#### `react-ui/src/pip/` (1 files)

| file | purpose / key symbols (from header) |
|---|---|
| pipModel.js | Releasing within this distance of a corner snaps (docks) to it. |

#### `react-ui/src/shortcuts/` (3 files)

| file | purpose / key symbols (from header) |
|---|---|
| actions.js | Every rebindable keyboard action in the workstation, with the keys it ships on. This table is the ONE place a default key is written. The handlers (App, FaceSwap, InteractivePreview, Timeline, Extras, |
| combo.js | A single printable character that is neither a letter, a digit nor a space: its Shift is part of the character. |
| keymap.js | The combos `id` uses: its override if there is one, else its defaults. |

#### `react-ui/src/store/` (10 files)

| file | purpose / key symbols (from header) |
|---|---|
| historyStore.js | Stage 23: the undo/redo history, as a store. WHAT IS TRACKED, AND WHY NOT recipeStore. The brief asked for middleware around the Zustand recipe & mask slices. `recipeStore.ts` has no consumer anywhere |
| index.ts | (no header comment) |
| jobStore.js | Backend job id, when the backend gives us one. |
| liveSubscribe.js | Subscribe `onFlush` to the store, called at most `hz` times a second. Returns an unsubscribe. Used by every component below. |
| pipStore.js | the layout lasts for this session |
| recipeStore.ts | â”€â”€ Types for Normalized Swap Recipe Slices â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Canonically imported from react-ui/src/types/api-contract.ts |
| shortcutStore.js | the binding lasts for this session |
| slotStore.js | already closed |
| telemetryStore.js | The most any telemetry-driven piece of UI updates per second. |
| workspaceStore.js | Built-in workspace presets including specialized workstation profiles. |

#### `react-ui/src/transport/` (3 files)

| file | purpose / key symbols (from header) |
|---|---|
| frameProtocol.js | Split one binary message into its header fields and payload. The payload is returned as a VIEW-free ArrayBuffer slice, because it is about to be transferred to a worker, and a transfer detaches the WH |
| frameSocket.js | onclose follows and owns the retry |
| useFrameSocket.js | Whether /ws/frames is currently open; re-renders only on a transition. |

#### `react-ui/src/types/` (1 files)

| file | purpose / key symbols (from header) |
|---|---|
| api-contract.ts | Canonical API Contract Module (Stage 15). Shared TypeScript request/response contracts and data schemas. Imported by both the React UI (store/components) and the mock-server (server.ts / mockQueue.ts) |

#### `react-ui/src/workers/` (1 files)

| file | purpose / key symbols (from header) |
|---|---|
| frameDecoder.worker.ts | â”€â”€ Dedicated Frame Decoder Web Worker â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Decodes image frames off the main thread using createImageBitmap with { premultiplyAlpha: 'none', colorSpaceConversion:  |

#### 1.3 What the UI test layer certifies (react-ui/.render-check + e2e)
- Logic checks run in plain Node/Vite SSR: batch matrix staging (`batch-matrix-check.mjs`), brushâ†’canvas mapping sub-pixel (`brush-mapping-check.mjs`), face mapping/roles, retry policy (3 retries, 250/500/1000 ms), frame protocol + player pipeline (`player-check.mjs`), head-pose HUD model (`render-head-pose-hud.mjs`: signed 1-decimal angles, dash for missing), history/undo engine + A/B/C/D slots (`render-history.mjs`), identity keyframes (stage 20), LUT/scopes (stage 18), BroadcastChannel monitor (stage 19), stereo VR (stage 21), NLE EDL/OTIO + VFX matte + provenance (stage 22), parameter curves + context menu (stage 25), expression parser + shortcuts (stage 26), resizable workspace store defaults leftWidth 350/rightWidth 330/bottomHeight 280 (stage 27), timeline logic, pipeline pill/report/controls, TRT compile banner (stage 14), console mirror from a REAL backend-generated snapshot, mutation-check (re-introduces bugs, asserts red).
- Playwright e2e (production build + mock server): axe a11y with allowlist ratchet, WCAG contrast per theme, frame-budget/long-task, idle CPU/request budget (â‰¤6 stage-frame requests / 30 s when preview 500s), nav visibility, tab-stop count, offline (no external requests), frame-busy 503, frame-unavailable retry, head-pose HUD, timeline, stage5/11/25/26 etc.

## 2. docs/ (328 files)

### 2.1 Markdown docs â€” one line each (key claim)

- `BENCHMARKS.md` â€” **Benchmarks and validation**: How performance and quality are measured on the two validation targets (RTX 4070 12 GB desktop and RTX 3060 Laptop 6 GB). Moved here from the README on 2026-09-22; the text is unchanged. The maintained result tables are
- `CHANGELOG.md` â€” **Changelog and incident notes**: Dated notes on behaviour changes and the incidents behind them, in reverse order. The full session record is [`SESSION_LOGS.md`](SESSION_LOGS.md); the running engineering state lives outside the repository (`RECODE_STATUS.md` in the operator's `roop-keep`
- `CPU_GATE_D.md` â€” **Gate D â€” Intel hybrid CPU optimization**: Gate D adds capability-driven CPU profiling and controlled CPU-distribution selection. It does not make i9-14900K behavior mandatory and does not claim a CPU policy is globally optimal without end-to-end evidence on both required
- `DETECTION_INTERACTION_TRACEABILITY_2026-09-23.md` â€” **Detection and interacting-face traceability matrix**: Date: 2026-09-23 This matrix maps every explicit scenario in the request and every changed public behavior to a concrete check and its observed result. It deliberately separates detector evidence from proof that a selected source was actually applied in a complete video render. - Exact required gate
- `ENV_FLAGS.md` â€” **Environment Flags (`ROOP_*`)**: All runtime tunables read from environment variables. Every flag is **opt-in or safe-by-default** â€” with none set, the app runs its validated default behavior. Values are read at process start (some at module import), so set them **before**
- `FACESET_V2.md` â€” **FaceSet V2 format**: `.fsz` V2 = ZIP of legacy PNGs + deterministic `metadata.json` (schema roop.fsz v2, sorted keys, fixed ZIP timestamps, SHA-256 per member); per-source identity embedding, bbox, 68-pt + 3D 68 landmarks, yaw/pitch/roll, quality, appearance, expression, detail masks. Invalid CRC/unsafe names rejected.
- `FINAL_ARCHITECTURE.md` â€” **Roop Ultimate â€” final optimization architecture**: Phase 16 flow: decodeâ†’detect/qualityâ†’tracks+identity+pose+expressionâ†’FaceSet lookup+pose-aware sourceâ†’alignâ†’illumination adaptâ†’identity-detail restoreâ†’enhancerâ†’maskâ†’temporal compositingâ†’temporal QCâ†’encode/mux; TRT cache namespaced by precision/SM/CUDA/driver/TRT/ORT; CPU always last-resort provider.
- `FINAL_VALIDATION_MATRIX.md` â€” **Final validation matrix**: Master record for the final end-to-end validation, repair and regression campaign. One row per test. A row is PASS only when a run on this repository's current code produced the evidence named in its own cell.
- `GATE_ABE_4070.md` â€” **Gates A, B and E â€” RTX 4070, 2026-08-30**: All measurements are end-to-end renders of the locked controlled fixture (`d4.mp4`, two facesets, realswap / GPEN 256 Pro / RealityUX, TensorRT mixed) at **600 frames â€” production length**. Short-window (120-frame) numbers appear
- `GATE_A_ADVERSARIAL_REVIEW.md` â€” **Gate A â€” Independent Adversarial Performance Review**: Date: 2026-08-29 Scope: current repository implementation and performance harnesses, before the Gate A fixes in this review.
- `HARDWARE_VALIDATION.md` â€” **Hardware Validation Ledger**: This file is the per-change validation ledger required by the dual-GPU policy. The RTX 3060 and RTX 4070 are never combined into one generic benchmark. The detailed historical matrix is in [`HARDWARE_VALIDATION_MATRIX.md`](HARDWARE_VALIDATION_MATRIX.md).
- `HARDWARE_VALIDATION_MATRIX.md` â€” **Dual-GPU hardware validation matrix**: This is the acceptance record for the two first-class NVIDIA targets. Results are never averaged across GPUs. Runtime identity is detected from the active stack and persisted with `hardware_profile_key`; free VRAM is telemetry and is
- `HDR_PIPELINE.md` â€” **HDR / high-bit-depth colour path**: `app/roop/hdr_color.py` holds the colour maths and `app/roop/hdr_pipeline.py` the I/O. Settings live in the **Output** section: `hdr_pipeline`, `hdr_source_transfer`, `hdr_source_primaries` and `hdr_output_codec` (env `ROOP_HDR`, `ROOP_HDR_TRANSFER`,
- `LIVE_MODE.md` â€” **Real-Time / Live Webcam Mode**: Live webcam: buffer_size=1 latest-frame mailbox, detector every **6th** frame + Lucas-Kanade flow between, histogram scene-cut detector flushes temporal state, **45 ms** target budget (unverified), enhancers disabled, OBS virtual cam via pyvirtualcam, endpoints `/api/livecam/*`.
- `OPTIMIZATION_PROGRESS.md` â€” **Optimization Progress**: Full record: `docs/FINAL_VALIDATION_MATRIX.md`, section `# RTX 4070 campaign (2026-09-01)`. Handoff and next steps: `docs/PHASE_HANDOFF.md`, top section.
- `PERFORMANCE_OPTIMIZATION_HANDOFF.md` â€” **Performance Optimization Handoff**: **PERFORMANCE FOUNDATION COMPLETE â€” READY FOR REALISM/QUALITY OPTIMIZATION** This document is the authoritative handoff for the performance-foundation work completed through the stable implementation point below. It describes what is
- `PHASE11_ENHANCER_INVENTORY.md` â€” **Phase 11 enhancer inventory**: Status: source-tree inventory completed 2026-09-01. This document is the authoritative inventory for Phase 11; it is based on the implementations under `app/roop/processors`, the registered processors in `app/roop/ProcessMgr.py`,
- `PHASE11_ENHANCER_MATRIX.md` â€” **Phase 11 enhancer performance matrix**: This is the complete matrix schema for the 30 source-discovered enhancement, restoration, colorization, and super-resolution paths. The benchmark stores one copy per hardware profile; rows are never averaged across GPUs. `pending`
- `PHASE6_POSE_QUALITY.md` â€” **Phase 6 Pose and Source-Bank Quality Evaluation**: Pose eval harness; selector match tolerance **15Â° yaw & pitch**; V1 `.fsz` archives had five real yaw plates incl. profile, no pitch/inversion plates (coverage gap flagged).
- `PHASE6_TEMPORAL_IDENTITY.md` â€” **Phase 6 Temporal Identity Consistency Engine**: `TemporalIdentityStabilizer` (opt-in `ROOP_TEMPORAL_IDENTITY=1`): source-bank switch after **3** persistent requests, bounded immediate re-estimation on major yaw/pitch/roll change; low-frequency-only output blending; crop blending bypassed on parallel stab path.
- `PHASE7_OCCLUSION.md` â€” **Phase 7 - Temporal Occlusion and Interacting-Face Engine**: Temporal occlusion engine (opt-in `ROOP_TEMPORAL_OCCLUSION`): event threshold **0.12**, interaction **0.08**, stable frames 3, refresh 5, appearance 0.16, leave alpha 0.35, enter alpha 0.90; mask-only transitions.
- `PHASE8_EXPRESSION.md` â€” **Phase 8 â€” Target Expression Preservation**: Model-free expression layer on 106-pt landmarks (opt-in): alpha 0.28, motion alpha 0.82, closed ratio 0.48, open ratio 0.70, event strength 0.86; independent per-eye blink/wink states; real-video validation still OPEN.
- `PHASE_HANDOFF.md` â€” **Handoff - shape-profile A/B: CLOSED, no measurable effect (2026-09-03, final)**: Device: RTX 4070, driver 616.56, TRT 10.9.0.34, ORT 1.23.2, i9-14900K, 31.7 GB. Commits `d542608`, `c264e86`, `4ec1f0f`, `66d754e`, `91c9d48`. Suite **2002 green**, 1 skipped. Supersedes the VOID run described below; the guard that
- `POSE_AWARE_SOURCE_SELECTION.md` â€” **Phase 5: Pose-aware source selection**: Pose record `_pose_v5` (yaw/pitch/roll/scale/off-axis/perspective risk); source scoring on yaw/pitch/circular-roll + expression + lighting; switch hysteresis margin default **0.035** (`ROOP_POSE_SOURCE_SWITCH_MARGIN`); requests 3D crop correction when pose missing, between sparse bank poses, inverted, high perspective risk; 3D plan bounds yaw shear, tighter pitch shear.
- `SECOND_GPU_VALIDATION.md` â€” **Validating the second GPU**: Everything here is capability-detected at runtime. Nothing needs editing to move between cards: models, thread count, pool sizes and the safety policy all come from the machine's own `config.yaml` and its own detected VRAM tier.
- `SESSION_LOGS.md` â€” **Session Log Archive â€” Roop Ultimate**: Verbatim session logs moved out of `CLAUDE.md` on 2026-09-14 to keep the always-loaded file small. Nothing here was edited. The distilled, still-load-bearing rules live in `CLAUDE.md`; this file is the evidence and the history behind them.
- `TROUBLESHOOTING.md` â€” **Troubleshooting**: Start with the logs: Pinokio writes every script's terminal output under `logs/` in the project root -- `logs/api/<script>/` for `install.js`, `start_react.js` and friends, with a `latest` file per script -- and the backend's own `[Backend]` lines are in the
- `VERIFICATION_ROOP_KEEP.md` â€” **Verification sweep over the roop-keep corpus (2026-09-03, RTX 4070)**: Harness `tests/verify_roop_keep.py`. Report: `<roop-keep>/output/verification_report.{html,json}`, 18 diagnostic strips in `output/diagnostic_frames/`, 8 rendered outputs.
- `archive/facegemini.md` â€” **Roop Ultimate - Comprehensive Memory & Engineering Reference (facegemini.md)**: *Complete System Architecture, Technical Memory, Hardware Profiling, and Inverted Face Swap Reference* *Date: 2026-08-23 | Status: Production Live | Test Suite: 1023/1023 Passing (100%)* > ### âš  SECTIONS 1-7 BELOW ARE PARTLY SUPERSEDED â€” READ THE FINAL SESSION LOG FIRST
- `archive/roopv2.md` â€” **Roop Ultimate â€” v2 Audit Memory**: **Started:** 2026-08-27 (Asia/Kolkata) **Purpose:** A preservation-first, evidence-led audit of Roop Ultimate. This file is the authoritative continuation record for the audit. It must be updated after every material inspection, test, decision, code change, and verification result. - Audit code line
- `audits/STAGE0_BENCHMARK_REPORT.md` â€” **STAGE 0 â€” PERFORMANCE + QUALITY REPRODUCIBLE BASELINE REPORT**: **Timestamp:** `2026-09-30T15:58:49.468921+00:00` **Classification:** **GPU-BOUND (High GPU compute/bandwidth saturation; GPU stage optimization directly scales throughput)** - **GPU:** NVIDIA GeForce RTX 4070 (12282 MiB VRAM, Compute 8.9)
- `audits/STAGE10_TO_15_WORKSTATION_AUDIT_REPORT.md` â€” **Stages 10â€“15 â€” Workstation Hardening Audit**: **Range:** commits `53cf2c7` (Stage 10) to `df332de` (Stage 15), all 2026-10-28. **Written:** 2026-10-28, Stage 16. **Sources:** `docs/CHANGELOG.md`, the six commit messages, the code and tests named below, and one test run made while writing this report (see Â§8). Nothing here was re-measured on a G
- `audits/STAGE1_BOTTLENECK_REPORT.md` â€” **STAGE 1 â€” PIPELINE BOTTLENECK ANALYSIS REPORT**: **Hardware Platform:** NVIDIA GeForce RTX 4070 Desktop (12,282 MiB VRAM) | Intel Core i9-14900K (24 physical cores / 32 logical threads) | 31.7 GB System RAM **Software Environment:** CUDA 12.8 | TensorRT 10.9.0.34 | ONNX Runtime 1.19+ | Python 3.10 **Test Fixture:** `scenario_01_frontal_face.mp4` (
- `audits/STAGE2_MODEL_LIFECYCLE_REPORT.md` â€” **STAGE 2 â€” Model Lifecycle and Runtime Architecture Audit Report**: **Repository:** `rishabh4496/roop-ultimate` **Execution Profile:** Main Workstation (NVIDIA GeForce RTX 4070 12GB Desktop, 32GB RAM, Compute 8.9) **Secondary Profile Verified:** Laptop Workstation (RTX 3060 Laptop 6GB, sub-7GB CUDA safety rules, 0/0 pool, strictly < 2.5GB system RSS)
- `audits/STAGE3_DETECTOR_OPTIMIZATION_REPORT.md` â€” **STAGE 3 â€” SCRFD Face Detector Optimization & Comprehensive Audit Report**: > **Synthetic inputs (marked 2026-10-09).** The figures in this report come from `tools/benchmark_adaptive_detector.py`, whose inputs are > generated: one still photograph replicated into a fake video with simulated cuts, pans and profile warps. They describe the generated scene, not the application
- `audits/STAGE4_ALIGNMENT_GEOMETRIC_STABILITY_REPORT.md` â€” **STAGE 4 â€” Face Alignment and Geometric Stability Report**: SYNTHETIC banner. Claims yaw>35Â° far-side landmark collapse, >45Â° 3-point stable profile solver, |roll|>45Â° canonical roll rotate.
- `audits/STAGE5_HYPERSWAP_AUDIT_REPORT.md` â€” **STAGE 5 â€” HyperSwap Quality and Performance Audit Report**: **INVALID - DO NOT CITE** (identity similarities were typed in, per perf/quality_harness_method.md).
- `audits/STAGE6_RESTORE_ULTRA_AUDIT_REPORT.md` â€” **STAGE 6 â€” RESTORE ULTRA QUALITY & THROUGHPUT AUDIT REPORT**: > **Synthetic inputs (marked 2026-10-09).** The figures in this report come from `tools/benchmark_restore_ultra.py`, whose inputs are > generated: a drawn face, a model "simulation" (output = input) whenever no session loads, and an FP16 row that is the FP32 time times a typed-in 0.65. They describe
- `audits/STAGE7_XSEG3_AUDIT_REPORT.md` â€” **STAGE 7 â€” XSEG 3 MASK QUALITY & PERFORMANCE AUDIT REPORT**: > **Synthetic inputs (marked 2026-10-09).** The figures in this report come from `tools/benchmark_xseg3.py`, whose inputs are > generated: drawn canvases and np.random.uniform stand-ins for the mask network output in the quality tests. They describe the generated scene, not the application on real f
- `audits/STAGE8_COMPOSITING_AUDIT_REPORT.md` â€” **Stage 8 â€” Compositing Quality Engine Audit & Technical Report**: > **Synthetic inputs (marked 2026-10-09).** The figures in this report come from `tools/benchmark_compositing.py`, whose inputs are > generated: flat-colour panels and seeded-RNG patterns as target and paste. They describe the generated scene, not the application on real footage, and the tool now sa
- `audits/STAGE9_TEMPORAL_TRACKING_AUDIT_REPORT.md` â€” **Stage 9 â€” Temporal Face Tracking and Flicker Control Audit Report**: SYNTHETIC INPUTS banner (2026-10-09). Claims Kalman + ArcFace identity lock state machine, cubic Hermite interpolator, IoU>0.25 interaction; profile handling table: 20-45Â° semi-profile far-eye weight â‰ˆ0.3, 45-90Â° 3-point anchor solver. Not real-footage evidence.
- `development/ARCHITECTURE.md` â€” **Roop Ultimate Architecture Baseline**: React UI (react-ui/src) â”‚ loopback HTTP JSON/multipart â–¼
- `development/BATCH_MATRIX_DATA_FLOW.md` â€” **Batch Matrix Data Flow**: Traced 2026-09-22 from the code, not from the UI copy. Line numbers are as of commit `d355803`; the function names are the stable handles. Since `4a38666` the staging logic lives in `react-ui/src/components/faceswap/batchMatrix.js`
- `development/CONTRIBUTING.md` â€” **Contributing: layout, tests, CI**: Moved here from the README on 2026-09-22. The agent rule file for AI-assisted work is [`../../AGENTS.md`](../../AGENTS.md). roop-ultimate/
- `development/CURRENT_STATE.md` â€” **Current Repository State**: **React UI 1.0 (`react-ui/`) is the sole production client.** `start.js` re-exports `start_react.js`, the Pinokio menu has one Start action, and `react-ui-v2/`, `start_react_v2.js` and the five `test_ui2_*.py` files are
- `development/DECISIONS.md` â€” **Architectural Decisions Recorded in the Repository**: This is an audit of decisions evidenced by source or tracked history. It does not approve new behavior. - Keep ownership boundaries explicit when adding UI 2.0 features. - Add versioned API contracts before changing payloads or status semantics.
- `development/ENVIRONMENT_CONTRACT.md` â€” **Environment and Launcher Contract**: **React UI 2.0 is the default client.** `start.js` re-exports `start_react_v2.js`; the Pinokio menu default starts V2; React UI 1.0 keeps its own action in every launcher branch and is covered by `install.js` and
- `development/JOB_CONTRACT.md` â€” **Job and Lifecycle Contract**: Canonical job states incl. `PAUSE_REQUESTED` transient hold; legacy map pending/running/finished; interrupted run â†’ `RECOVERABLE`; pause ack `{requested, acknowledged, active_work:0, pending_output:0}`.
- `development/KNOWN_ISSUES.md` â€” **Known Issues and Open Questions**: CPython 3.10's proactor loop closes the LISTENING socket when a peer aborts mid-accept (`WinError 64` in `IocpProactor.accept`, "Accept failed on a socket"). Chromium's speculative pre-connects on navigation trigger it, and a
- `development/MASTER_PLAN.md` â€” **Roop Ultimate Development Master Plan**: **Reported as a screenshot of the V2 client failing:** `[vite] http proxy error` / `connect ECONNREFUSED 127.0.0.1:42003` repeating against every endpoint. **The backend was not broken. It had started perfectly and then been killed by
- `development/NETWORK_CONTRACT.md` â€” **Online / Offline Operation Contract**: Stage 12 defines the network boundary for the existing application. Internet access is not part of the local processing contract when the selected models, dependencies, project files, and output tools are already installed locally.
- `development/PERFORMANCE_REGRESSION.md` â€” **Performance and fidelity regression suite**: `tests/test_performance_regression.py` â€” real production renders (the Start button's path, `core.batch_process_with_options` through `roop.benchmark.regression`), never `BenchmarkRunner`.
- `development/PROCESSING_CONTRACT.md` â€” **Processing Contract**: Audit-first stage; four defects were found and fixed, each pinned in `app/tests/test_stage15_identity_audit.py` (mutation-checked: the tests go red with the fix reverted):
- `development/RECOGNIZER_CALIBRATION.md` â€” **Recognition model calibration: every registered model vs w600k (2026-10-03)**: Question: should any registered recognition model other than w600k be wired into live identity matching, and what distance threshold would it need? (`antelopev2` is the same file as `glintr100` and is not run separately.) Tool: `app/tools/calibrate_recognition.py`; raw summaries in [`recognizer_cali
- `development/RECOGNIZER_VIDEO_COMPARISON.md` â€” **All six recognition models on one video, with a face swap (2026-10-03)**: Tool: `app/tools/compare_recognition_swap.py` (phases `collect`, `analyze`, `swap`, `render`). Clip: `D:\Monica Bellucci .mp4` (7,647 frames, 1280x720, 30 fps, 4.2 min; the file name has a space before `.mp4`). Source: `facesets/harjot.fsz`. Output: a 3x2 grid video with each model's decisions, live
- `development/REGRESSION_BENCHMARK.md` â€” **Regression benchmark**: python run.py --benchmark --benchmark-mode regression # from app/ portable\run.bat --benchmark --benchmark-mode regression # portable runtime Exit codes: **0** pass (or baseline recorded), **1** regression, **2** the suite could
- `development/SESSION_HANDOFF.md` â€” **Stage 22 Session Handoff - React UI 2.0 removed, its real capabilities migrated**: **React UI 1.0 is the sole production client.** `react-ui-v2/`, `start_react_v2.js` and five `test_ui2_*.py` files are gone; `start.js` re-exports `start_react.js`; the Pinokio menu has one Start action.
- `development/STAGE10_PIPELINE_CONCURRENCY.md` â€” **STAGE 10 â€” CPU/GPU PIPELINE CONCURRENCY ARCHITECTURE & BENCHMARK REPORT**: Stage 10 delivers a formal, evidence-based concurrency audit of all threads, executors, queues, and locks across Roop Ultimate, establishes the architectural design for a bounded 8-stage video processing pipeline, and evaluates performance across 5 distinct concurrency configurations on the RTX 4070
- `development/STAGE11_VRAM_RAM_MANAGEMENT.md` â€” **STAGE 11 â€” VRAM/RAM MANAGEMENT ARCHITECTURE & RUNTIME LIMITS REPORT**: Stage 11 establishes a formal, evidence-based memory management infrastructure across Roop Ultimate, auditing and profiling every large allocation category, identifying memory duplication and accumulation hotspots, and implementing bounded, reusable memory primitives. The operational principle gover
- `development/STAGE12_VIDEO_IO_PIPELINE.md` â€” **STAGE 12 â€” VIDEO I/O PIPELINE OPTIMIZATION & BOTTLENECK ANALYSIS**: Stage 12 delivers an exhaustive audit and empirical benchmark of the FFmpeg video I/O pipeline across Roop Ultimate, measuring decode throughput, neural processing throughput, encode throughput, disk bandwidth, pixel-format conversion overhead, and CPU/GPU bus transfer latencies. The fundamental ope
- `development/STAGE13_INFERENCE_RUNTIME.md` â€” **Stage 13: Inference Runtime Optimization**: Stage 13 establishes a comprehensive model inference optimization, validation, and benchmarking harness across the Roop Ultimate pipeline. It audits models across all functional pipeline families, evaluates five distinct runtime execution modes, evaluates advanced execution features, enforces the em
- `development/STAGE14_ADVANCED_SETTINGS.md` â€” **Stage 14: Advanced Settings Redesign & Presets Engine**: Stage 14 establishes a comprehensive settings classification, auditing, conflict resolution, and dynamic presets engine in [`app/roop/advanced_settings_manager.py`](file:///G:/pinokio/api/roop-ultimate/app/roop/advanced_settings_manager.py). The subsystem addresses four historical failure modes in u
- `development/STORAGE_CONTRACT.md` â€” **Storage Manager Contract**: This document defines the Stage 10 storage boundary. The storage manager is a review tool and an explicit single-item cleanup boundary; it is not a drive-wide cleaner and it never deletes on startup or during an inventory request.
- `development/TERMINAL_CONTRACT.md` â€” **Terminal and Runtime Reporting Contract**: Stage 11 adds a structured runtime report to the existing terminal surface. The report is observational: it reads backend-owned progress, runtime profile, queue, project/checkpoint, resource, and log state. It does not select a
- `development/UI2_BEHAVIOR_CONTRACT.md` â€” **React UI 2.0 Behavior & Compatibility Contract**: > **SUPERSEDED â€” 2026-09-02.** React UI 2.0 (`react-ui-v2/`) was **removed** > after an endpoint-level audit found its unique surface to be seven items, > all of which were migrated into React UI 1.0 and verified against a live
- `development/UI2_CONTRACT.md` â€” **React UI 2.0 Boundary Contract and React V1 Forensic Audit**: > **SUPERSEDED â€” 2026-09-02.** React UI 2.0 (`react-ui-v2/`) was **removed** > after an endpoint-level audit found its unique surface to be seven items, > all of which were migrated into React UI 1.0 and verified against a live
- `development/UI2_DESIGN_SPECIFICATION.md` â€” **Roop Ultimate â€” React UI 2.0 Design Specification**: > **SUPERSEDED â€” 2026-09-02.** React UI 2.0 (`react-ui-v2/`) was **removed** > after an endpoint-level audit found its unique surface to be seven items, > all of which were migrated into React UI 1.0 and verified against a live
- `development/UI2_FINAL_ACCEPTANCE.md` â€” **React UI 2.0 Final Acceptance Review**: > **SUPERSEDED â€” 2026-09-02.** React UI 2.0 (`react-ui-v2/`) was **removed** > after an endpoint-level audit found its unique surface to be seven items, > all of which were migrated into React UI 1.0 and verified against a live
- `development/UI2_MIGRATION_PLAN.md` â€” **React UI 2.0 migration review and plan**: > **SUPERSEDED â€” 2026-09-02.** React UI 2.0 (`react-ui-v2/`) was **removed** > after an endpoint-level audit found its unique surface to be seven items, > all of which were migrated into React UI 1.0 and verified against a live
- `development/UI2_PERFORMANCE_AUDIT.md` â€” **React UI 2.0 Performance & Concurrency Optimization Audit**: > **SUPERSEDED â€” 2026-09-02.** React UI 2.0 (`react-ui-v2/`) was **removed** > after an endpoint-level audit found its unique surface to be seven items, > all of which were migrated into React UI 1.0 and verified against a live
- `development/UI2_PREVIEW_VALIDATION.md` â€” **React UI 2.0 Preview Subsystem Validation & Verification Guide**: > **SUPERSEDED â€” 2026-09-02.** React UI 2.0 (`react-ui-v2/`) was **removed** > after an endpoint-level audit found its unique surface to be seven items, > all of which were migrated into React UI 1.0 and verified against a live
- `development/UI2_REGRESSION_REPORT.md` â€” **React UI 1.0 vs React UI 2.0 Functional Regression Report**: > **SUPERSEDED â€” 2026-09-02.** React UI 2.0 (`react-ui-v2/`) was **removed** > after an endpoint-level audit found its unique surface to be seven items, > all of which were migrated into React UI 1.0 and verified against a live
- `development/UI2_UX_UI_REVIEW.md` â€” **React UI 2.0 UX/UI Forensic Review & Enhancement Plan**: > **SUPERSEDED â€” 2026-09-02.** React UI 2.0 (`react-ui-v2/`) was **removed** > after an endpoint-level audit found its unique surface to be seven items, > all of which were migrated into React UI 1.0 and verified against a live
- `development/UI_V1_V2_MIGRATION_AUDIT.md` â€” **React UI 1.0 / 2.0 Migration Audit**: **Method.** The comparison below is derived from the repository, not from the UI2 contract documents. Three independent passes: 1. **Endpoint diff** â€” every `/api/...` literal in `react-ui/src` and
- `development/UPDATE_AUDIT.md` â€” **Stage 9A â€” Update System Audit**: Audit date: 2026-09-02 Gate: Stage 9A â€” Update System Audit Status: audit and design only; no update mechanism was changed
- `development/UPDATE_CONTRACT.md` â€” **Compatibility-Gated Update Contract**: This document defines the Stage 9B update boundary. It describes the implemented compatibility decision, not a promise that every future release is installable.
- `development/VALIDATION_MATRIX.md` â€” **Validation Matrix**: Device: RTX 4070 12GB, driver 616.56, CUDA 12.8, TensorRT 10.9.0.34, ORT 1.23.2, i9-14900K 24P/32L, 31.7GB RAM. All rows measured against a live backend on 127.0.0.1:8001 with the client on Vite.
- `development/VISUAL_CONTRACT.md` â€” **Visual-Quality Contract â€” Stage 2A Audit**: Stage 2A audit mapping every blend/mask/sharpen control to code lines (e.g. paste_upscale ellipse âˆ© landmark hull, feather, model mask, temporal matte, ownership `procmgr_masking.py:380-480`; merger chain sharpen/clarity/hist/motion-blur/grain `ProcessMgr.py:5412-5427`).
- `perf/aux_batch_2026-10-05.md` â€” **Batched aux models (recognition, 106- and 68-point landmarks) in the tracking pre-pass (20**: **CODE REMOVED (the owner chose removal after this report).** The implementation, its 23 tests and the harness were deleted; they are recoverable from commit `f11c8c2` (`app/roop/aux_batch.py`, the `_apply_aux` / `aux_batch_scope` seam in `face_util.py`, `app/tests/ab_aux_batch.py`, `app/tests/test_
- `perf/aux_canary_calibration.md` â€” **Aux canary floor calibration**: Generated by `app/tests/aux_canary_calibration.py` (2026-10-10 20:09). Metric = `aux_canary.compare`, reference = CUDA FP32 with TF32 off. `good` = healthy engines on ~500 real faces; `canary` = the same engines on the two synthetic canary inputs; `wrong` = an engine that answers for a different fac
- `perf/aux_fp16_onnx_2026-10-11.md` â€” **Auxiliary models in FP16 (xseg, 2d106det, 1k3d68) - 2026-10-11**: **Result: NOT shipped. All three stay FP32 (`precision_policy.PRECISION_OVERRIDES`). No strategy of FP32 op exclusions cleared the gates, and the FP16 ONNX is not faster on the CUDA EP for two of the three models.** Brief: convert the three to FP16 with `onnxconverter_common`, exclude sensitive ops,
- `perf/baseline_2026-10-04.md` â€” **Performance baseline 2026-10-04**: Reference every later stage must match. Measurement only: produced by `app/tests/baseline_snapshot.py` at `f9aaf96` (uncommitted python files at run time: none). Measurement only: nothing in the render path was changed, and no setting was tuned. The numbers below are the reference a later stage has
- `perf/color_ema/README.md` â€” **Temporal colour-transfer EMA â€” measured, NOT wired (2026-10-22)**: Probe: `app/tests/probe_color_ema.py` (same rig and clips as `probe_skin_tone_transfer.py`: a real faceset swapped by the production pipeline with the colour transfer OFF; every arm is then applied offline to the SAME swapped crop). Code under test: `app/roop/temporal_color.py` (`lct_affine` is bit-
- `perf/flow_vs_neural/README.md` â€” **Flow-placed landmarks vs the landmark net on the flow box (2026-10-22)**: `app/tests/probe_flow_vs_neural_landmarks.py`, K=4 simulated on real frames. Truth = production detector + `landmark_2d_106` on every frame. Flow = `FlowCoaster` (a SIMILARITY fit of the tracked cloud applied to the KEY FRAME's landmarks: it carries the key frame's expression and cannot follow a bli
- `perf/fused_postprocess/README.md` â€” **Fused Stage 3 post-processing kernel + L2 persistence (2026-10-22) â€” built and measured, N**: Code: `app/core/fused_postprocess.py` (Triton), `app/core/l2_persistence.py` (cuda-python). Tests: `tests/test_fused_postprocess.py` (23: parity with the unfused maths, the maths against OpenCV, grain statistics, strided in-place composite, refusals), `tests/test_l2_persistence.py` (7: the API calls
- `perf/gpu_engine_2026-10-04.md` â€” **detector_engine `retinaface_r50_gpu` â€” wiring and A/B (2026-10-04)**: Code: `app/roop/retinaface_gpu_engine.py` (commit `49187bf`), registered in `face_util._HYBRID_ENGINES`, `api.py`, `bench.py` and the UI fallback list. **Not the default.** RTX 4070, `config.yaml` live (scrfd, 10 threads), `ROOP_PROFILE=1`, `ROOP_STAB_CHUNK_MB` pinned per clip, one render at a time.
- `perf/gpu_roi_paste/README.md` â€” **`CudaAffineBatch.paste_faces_roi` (2026-10-22) â€” built and priced, NOT wired into the rend**: There is no `GpuFaceAligner`; `CudaAffineBatch` (`optimized_processor.py`) is the existing GPU aligner: bicubic `grid_sample` warp, inverse-warp paste, feathered box matte, occlusion multiply. Its `paste_faces` builds the sampling grid and runs `grid_sample` over the WHOLE frame per face and clones
- `perf/grain_match/README.md` â€” **Sensor-noise match after the paste (2026-10-22)**: Code `app/roop/grain_match.py`, wired in `paste_upscale` (after the rim match, before the composite); counters printed in the swap audit as `grain match: ...`. `ROOP_GRAIN_MATCH=0` disables it; default 1 (self-gating). Tests `test_grain_match.py` (21; the wiring tests have the feature off as the con
- `perf/matte_roi/README.md` â€” **Face-neighbourhood matte in `paste_upscale` (2026-10-22)**: Code: `procmgr_masking.MaskingMixin._matte_roi_eligible` / `_matte_roi`, `create_landmark_mask(origin=)`. Off switch `ROOP_BLEND_ROI_MATTE=0`. Tests `app/tests/test_matte_roi.py` (42). Probes `app/tests/probe_matte_roi_real_faces.py`, `ab_env_e2e.py`.
- `perf/nvenc_quality/README.md` â€” **NVENC quality options (multipass / spatial AQ / temporal AQ) â€” measured, shipped OPT-IN (2**: Code: `app/roop/nvenc_quality.py` (the only place the option spellings live), wired into `ffmpeg_writer.FFMPEG_VideoWriter` (`_build_cmd`), `video_stream.NVHardwareVideoWriter` (`_command`, `_encoder_preset`) and `video_io_hw.NvZeroCopyWriter` (`CreateEncoder`). Tests: `app/tests/test_nvenc_quality.
- `perf/phase14/README.md` â€” **Phase 14 â€” pre-pass detection interval, 1-euro landmark filter, trajectory cache (2026-10-**: Pre-pass detection interval K: K=2/4/8 cut pre-pass 11-14%/21-26%/~30% but find 5/10/20% fewer faces (Love), 8%/31% fewer source-bound frames at K=4/8; flow-placed landmark error median ~1.2% eye distance vs 3.9-12.7% linear. Default stays K=1. Zero-phase 1-euro filter indistinguishable end to end; default stays `adaptive`. Trajectory cache (M, M_inv, pose per face per frame) feeds head-pose HUD.
- `perf/phase14/tables.md` â€” ****: **Love** (900 frames, reference = the second dense run; the two dense runs are bit-identical) **MonicaBellucci** (900 frames, reference = the second dense run; the two dense runs are bit-identical) **Weeds** (900 frames, reference = the second dense run; the two dense runs are bit-identical)
- `perf/pyramid_2026-10-04.md` â€” **Multi-scale detector: reuse the single pass, stop spawning executors (2026-10-04)**: Change: `face_detector.MultiScaleFaceDetector.detect` (commit `292faa0`). Reference code: `c4bfff9`. RTX 4070, `config.yaml` live, 10 threads. **Scope first.** The pyramid is reached only by the `retinaface` / `retinaface_r50` engines. The user's config is
- `perf/quality_harness_method.md` â€” **tools/quality_harness.py - method, what it replaced, and what it found about the old audit**: `tools/benchmark_hyperswap_audit.py` did not measure what its table said. It printed identity similarity typed in per model (0.88 / 0.84 / 0.81 / 0.865 / 0.86 plus Gaussian noise), "eye / mouth alignment error" computed as a keypoint set compared with itself plus noise, profile / occlusion / tempora
- `perf/r50_static_profile_2026-10-09.md` â€” **retinaface_r50: static TensorRT profile and letterbox A/B (2026-10-09)**: RTX 4070, TensorRT 10.9, ORT 1.23.2, `config.yaml` live (`trt_precision` mixed, detector pool 2). **Defaults are unchanged:** the static profile is opt-in (`ROOP_TRT_STATIC_PROFILE=1`), and the letterbox flag (`ROOP_R50_GPU_PREPROCESS`) already existed and stays on `squash`. Evidence: `docs/perf/r50
- `perf/rescue_aux_2026-10-04.md` â€” **Rotated-rescue aux work, and the r50 upscale rescue (2026-10-04)**: Two proposed changes to `face_util._detect_faces`, each gated on a measurement. Reference: `baseline_2026-10-04.md` (code `f9aaf96`). Change under test: `de63f34`. RTX 4070, `config.yaml` live, `ROOP_PROFILE=1`, 10 threads. The gate was: skip it only if it never gains a face. `tests/probe_r50_upscal
- `perf/restorer_fp16_islands_2026-10-10.md` â€” **RestoreFormer++ FP16: where the SSIM goes, whether FP32 islands buy it back, and an enhanc**: Brief: today the shipped FP16 "mixed" restorer scores SSIM ~0.996 mean / 0.995 min against FP32; target >= 0.998 at >= 90% of FP16 speed. (1) rank layers by FP16-vs-FP32 error on 32 real crops; (2) build a variant with the worst layers (and likely attention softmax / norms) as FP32 islands; (3) meas
- `perf/stab_warmup.md` â€” **Parallel-stabilization warm-up: what is state-independent (2026-10-09)**: Target: `ProcessMgr._run_stab_parallel` (`_process_block`, ProcessMgr.py ~2652) and `procmgr_stabilization._stab_parallel_geometry`. Config analysed is the live `app/config.yaml`: hyperswap, DFL XSeg, Restore Ultra, `temporal_detection: true`, `track_identities: true`, `stabilize_face/mask/enhancer/
- `perf/stage2_canary_cosine_2026-10-18.md` â€” **Stage 2: cosine of the production TensorRT engines against FP32, and the cost of RF++ at F**: Brief: canary cosine floors of >= 0.998 (xseg, 2d106det, 1k3d68) and >= 0.995 (RF++, UltraMax, GPEN Realistic), FP32 fallback below them; RF++ FP32 by default. Raw numbers: `stage2_canary_cosine_2026-10-18.json`. Harnesses (run from the repo root with `app/env/Scripts/python.exe`): `tools/measure_ca
- `perf/stage3_zero_copy_ring_2026-10-18.md` â€” **Stage 3 â€” zero-copy enhancer ring (2026-10-18, RTX 4070, TensorRT mixed, ORT 1.23.2)**: Policy: `CUDA_EXECUTION_POLICY.md` "Stage 3". Code: `ZeroCopyBufferPool` / `ZeroCopySlot` in `app/roop/optimized_processor.py`; wired in `Enhance_GPENRealistic`, `Enhance_RestoreFormerPPlus`, `Enhance_UltraMax`. Tests: `test_zero_copy_buffer_pool.py`, `test_enhance_common_gpu.py`,
- `perf/stage4_segments_governor_2026-10-18.md` â€” **Stage 4 â€” pipeline governor, segmented render, namespace, status channel (2026-10-18)**: Code: `app/roop/segment_output.py`, `app/roop/pipeline_status.py`, `app/routes_pipeline_status.py`, `app/roop/core.py` (`final_video_destination`), `app/api.py` (`_clear_output_folder`), `app/routes_queue.py`, `app/roop/runtime_scheduler.py`, `app/roop/distributed_render.py`,
- `perf/stage_precision_verify_lighting_2026-10-05.md` â€” **XSeg / RestoreFormer++ precision, verify routing, and the `lighting` stage (2026-10-05)**: Brief, from the Stage 0 provider logs: (1) if XSeg or RestoreFormer++ is not on TensorrtExecutionProvider with fp16 in a TRT render, find out why (dynamic batch axis, `trt_shape_profile`) and fix it so they run TRT mixed per `precision_policy`; validate masks IoU >= 0.995 and enhancer PSNR >= 45 dB
- `perf/sustained_throughput_2026-10-11/BRIEF.md` â€” **Sustained-throughput investigation: shared brief (2026-10-11)**: Branch `perf/sustained-throughput`, base commit `b3189690fcd1fb3049d6f7ee283fe6a1fcb76327` (clean tree at start). Repo root: `G:\pinokio\api\roop-ultimate`. Python: `app\env\Scripts\python.exe` (3.10.20). Clips and the running status log: `G:\pinokio\roop-keep\` (`RECODE_STATUS.md` newest section fi
- `perf/sustained_throughput_2026-10-11/ISSUES.md` â€” **Issue register (lead-owned)**: Base: `b3189690`, branch `perf/sustained-throughput`. Status values: open, investigating, confirmed, rejected, fixed, wontfix, unverified.
- `perf/sustained_throughput_2026-10-11/SWARM_ASSIGNMENTS.md` â€” **Swarm model assignments as actually spawned, 2026-10-11**: The request asked for Claude Sonnet 5.5 High as lead and Gemini 3.8 High as reviewer. This Jcode install did not expose those exact model names. The coordinator used `claude-sonnet-5-5` via Anthropic OAuth. Swarm workers used the closest available Gemini route, `gemini-3.8-flash-tiered` via Antigrav
- `perf/sustained_throughput_2026-10-11/acceptance_matrix.md` â€” **Acceptance Matrix and Requirement Traceability**: Date: 2026-10-11 This matrix maps the master request to a concrete check and its observed result. Custom unit tests and static analysis are identified as supporting evidence only. The production `run.py --benchmark --benchmark-mode regression` path was also exercised and is the direct end-user accep
- `perf/sustained_throughput_2026-10-11/architecture_map.md` â€” **Architecture Map and Execution Flow Analysis**: > Coordinator evidence correction: architectural findings are source-trace conclusions. They are not runtime confirmation of progressive degradation unless separately supported by the baseline or validation artifacts. Investigation: Progressive FPS degradation during video face swap Date: 2026-10-11
- `perf/sustained_throughput_2026-10-11/baseline_report.md` â€” **Baseline Report: Sustained Throughput and Null Control**: **Date:** 2026-10-11 **Branch:** `perf/sustained-throughput` from `b3189690fcd1fb3049d6f7ee283fe6a1fcb76327` **Hardware:** RTX 4070 Desktop, 12,282 MiB VRAM, driver 616.56, CUDA 12.8, 31.7 GB RAM
- `perf/sustained_throughput_2026-10-11/experiments_report.md` â€” **Stage 4 Experiments Report**: **Date:** 2026-10-11 **Status:** Implemented-fix checks and short A/B validation completed. Long-run acceptance experiments remain open. Baseline and final arms were separate 600-frame processes using the same fixture and pinned stabilization budget. Output FPS was 8.84 and 9.72 for baseline, and 9.
- `perf/sustained_throughput_2026-10-11/independent_review.md` â€” **Independent Validation and Adversarial Review**: **Date:** 2026-10-11 **Part 1 investigator:** Agent E, Gemini 3.8 Flash Tiered via Antigravity, high effort. **Part 2 final reviewer:** Attempted with Gemini 3.8 Flash Tiered via Antigravity, high effort, but the worker failed before producing a report.
- `perf/sustained_throughput_2026-10-11/inference_optimization_report.md` â€” **GPU Inference and Performance Engineering Report (Agent D)**: > Coordinator evidence correction: this report is static analysis. Its `confirmed` labels mean the code mechanism exists, not that it caused the measured 600-frame run. Runtime status is reconciled in `root_cause_report.md` and `validation_report.md`. **Author:** Agent D (GPU Inference and Performan
- `perf/sustained_throughput_2026-10-11/profiling_report.md` â€” **Profiling Report: Sustained Throughput & Progressive FPS Degradation Investigation**: > Coordinator evidence correction: this report contains the harness design and profiling plan plus the measurements explicitly marked as run. The recorded sidecar process-RSS fields were invalid because the original sampler did not aggregate the benchmark process tree. Runtime conclusions are reconc
- `perf/sustained_throughput_2026-10-11/resource_lifecycle_report.md` â€” **Resource Lifecycle, Queue Dynamics, and Memory Analysis (Agent C)**: > Coordinator evidence correction: this report is static analysis. Its `confirmed` labels mean the code mechanism exists, not that it caused the measured 600-frame run. Runtime status is reconciled in `root_cause_report.md` and `validation_report.md`. - **Date**: 2026-10-11 - **Investigator**: Agent
- `perf/sustained_throughput_2026-10-11/root_cause_report.md` â€” **Root Cause Report: Progressive FPS Degradation Investigation**: **Date:** 2026-10-11 **Hardware:** RTX 4070 Desktop, 12 GB VRAM, 31.7 GB RAM **Worker configuration:** Gemini 3.8 Flash Tiered via Antigravity, high effort.
- `perf/sustained_throughput_2026-10-11/validation_report.md` â€” **Validation Report: Before vs After Sustained-Throughput Fixes**: **Date:** 2026-10-11 **Coordinator:** Jcode coordinator **Worker configuration:** Gemini 3.8 Flash Tiered via Antigravity, high effort. Direct Gemini route attempts failed.
- `perf/swap_batch_equivalence_2026-10-10.md` â€” **Is the batched hyperswap the same picture as B=1? (2026-10-10)**: Test: `app/tests/test_swap_batch_equivalence.py` (GPU-gated; `python tests/test_swap_batch_equivalence.py --out f.json` prints the table). Data: `docs/perf/swap_batch_equivalence_2026-10-10/`. RTX 4070, live `config.yaml` (realswap, `trt_precision` mixed, TRT 10.9, ORT 1.23.2). 16 REAL aligned crops
- `perf/swap_batch_render_2026-10-11.md` â€” **Cross-frame swap batching, end to end: ON vs OFF (2026-10-11)**: **Result: batching the swap stage does not speed the render up. On two-person footage it is 4-9% SLOWER on the 4070; on single-person footage it is inside the noise.** No code or default was changed. Question (Stage 5 brief, option 2): does batching the `realswap` pair (hyperswap + the hififace eyel
- `perf/trt_bound_sessions_2026-10-09.md` â€” **TensorRT build heuristics, bound/graph static sessions, RF++ batching (2026-10-09)**: Brief (A/B only, all opt-in, defaults unchanged). RTX 4070 12 GB, TensorRT **10.9.0.34**, ORT **1.23.2**, live `app/config.yaml`: hyperswap, DFL XSeg, Restore Ultra, scrfd @ 512, `trt_precision: mixed`, threads 10, pools 2/2/2. Continues the paused 10-06 session.
- `perf/trt_per_model_precision_2026-10-10.md` â€” **Per-model TensorRT precision for XSeg / w600k_r50 / 2d106det / 1k3d68 (2026-10-10)**: Brief, four items: (1) what XSeg really runs under in a TensorRT render, and whether its NHWC input adds transposes; (2) mixed vs FP32 on 500 real faces for xseg, w600k_r50, 2d106det, 1k3d68; (3) swap_canary-style startup checks cached by engine hash, and precision per model instead of the one globa
- `perf/trt_precision_fidelity.md` â€” **TensorRT mixed vs FP32 reference on real faces: XSeg, w600k_r50, 2d106det, 1k3d68**: Generated by `app/tests/trt_precision_fidelity.py report` from `app/output/trt_precision_fidelity/results.json` (2026-10-10 19:39). Method, arms and gates: the module docstring. Samples: {'xseg_aligned': 500, 'xseg_box': 500, 'w600k_r50': 500, '2d106det': 500, '1k3d68': 500}. TRT-FP32 vs CUDA-FP32 f
- `perf/trt_precision_shipped_check.md` â€” **Shipped per-model precision: live sessions vs the CUDA FP32 reference**: Generated by `app/tests/aux_canary_calibration.py shipped` (2026-10-10 20:08), global `trt_precision` = `mixed`. Each session was built by the app's own loader (`Mask_XSeg.Initialize`, the buffalo_l bundle), its `trt_fp16_enable` was read back and compared with `precision_policy`'s table, and it was
- `perf/ui_poll_contention_2026-10-11.md` â€” **UI traffic vs the render clock: poll-contention A/B (2026-10-11)**: Question (Stage 7 brief): does the React UI's polling and HTTP 206 reading, served from a uvicorn thread inside the rendering process, cost the render frame rate? Raw data: `ui_poll_contention_2026-10-11.json`. Harness: `app/tests/ab_ui_poll_contention.py` +
- `perf/vram_budget_2026-10-05.md` â€” **Keeping peak VRAM under ~90% of the card (2026-10-05)**: Brief: using the Stage 0 VRAM logs, keep peak VRAM below ~90% of total with **no model or precision change**. (1) After `_precompute_temporal` + `_release_replayed_analysis`, also shrink the detector and FaceAnalysis pools to width 1. (2) Do not leave the analyser, detector and processor pools all a

### 2.2 Measured artefacts (non-markdown, 205 files) by directory


**`docs/development`** (2 files) â€” recognizer calibration/video comparison raw JSON (distance thresholds per recognition model vs w600k)
`recognizer_calibration_2026-10.json`, `recognizer_video_comparison_2026-10.json`

**`docs/perf (root)`** (49 files) â€” 2026-10-04..11 baselines & A/Bs: baseline snapshots (pinned clips Love/d1/d4/d6), GPU retinaface r50 engine recall/renders/speed, pyramid detector A/B, r50 upscale gain, rescue-aux A/B, aux batch equivalence, aux canary calibration, stab blocks-vs-sequential/dedup, stage2 canary cosine; plus sheet jpg and logs zip
`aux_batch_equiv_2026-10-05.json`, `aux_batch_off_identity_2026-10-05.json`, `aux_batch_prepass_2026-10-05-d4.json`, `aux_batch_prepass_2026-10-05-d4cpu.json`, `aux_canary_calibration.json`, `aux_canary_enhancer_calibration.json`, `baseline_2026-10-04.json`, `baseline_2026-10-04_logs.zip`, `baseline_2026-10-04_pinned_Love.json`, `baseline_2026-10-04_pinned_d1.json`, `baseline_2026-10-04_pinned_d4.json`, `baseline_2026-10-04_pinned_d6.json`, `gpu_engine_recall_2026-10-04.json`, `gpu_engine_recall_2026-10-04_squash.json`, `gpu_engine_recall_2026-10-09_letterbox.json`, `gpu_engine_recall_2026-10-09_squash.json`, `gpu_engine_renders_2026-10-04.json`, `gpu_engine_renders_2026-10-04_control.json`, `gpu_engine_speed_2026-10-04.json`, `pyramid_ab_2026-10-04.json`, `pyramid_renders_2026-10-04.json`, `pyramid_speed_2026-10-04.json`, `r50_upscale_gain_2026-10-04.json`, `r50_upscale_gain_2026-10-04_sheet.jpg`, `rescue_aux_ab_2026-10-04.json`, `rescue_aux_renders_2026-10-04.json`, `restorer_batch_2026-10-09.json`, `stab_blocks_vs_sequential_2026-10-09.json`, `stab_blocks_vs_sequential_2026-10-09_lossless.json`, `stab_blocks_vs_sequential_2026-10-09_seqcontrol.json`, `stab_dedup_2026-10-09.json`, `stab_dedup_2026-10-09_final.json`, `stab_dedup_2026-10-09_final2.json`, `stab_dedup_2026-10-09_mult.json`, `stage2_canary_cosine_2026-10-18.json`, `swap_batch_render_hyperswap_2026-10-11.json`, `swap_batch_render_realswap_2026-10-11.json`, `trt_bound_render_2026-10-09.json`, `trt_bound_sessions_2026-10-09-det-rfpp.json`, `trt_bound_sessions_2026-10-09-rfpp-xseg-v2.json`, `trt_build_heuristics_2026-10-06.json`, `trt_mixed_fidelity_2026-10-05.json`, `trt_precision_fidelity.json`, `trt_precision_probe.json`, `trt_precision_probe_shipped.json`, `trt_precision_shipped_check.json`, `ui_poll_contention_2026-10-11.json`, `vram_budget_2026-10-05.json`, `vram_budget_ballast.py`

**`docs/perf/color_ema`** (2 files) â€” measurement raw data / probe output behind the same-directory README
`probe_frames260.full.txt`, `probe_step.full.txt`

**`docs/perf/flow_vs_neural`** (3 files) â€” measurement raw data / probe output behind the same-directory README
`Love_k4.json`, `Weeds_k4.json`, `prepass_k4_weeds.json`

**`docs/perf/fused_postprocess`** (3 files) â€” measurement raw data / probe output behind the same-directory README
`benchmark.json`, `l2_persistence.json`, `manual_mask_probe.json`

**`docs/perf/gpu_roi_paste`** (1 files) â€” measurement raw data / probe output behind the same-directory README
`report.json`

**`docs/perf/matte_roi`** (3 files) â€” measurement raw data / probe output behind the same-directory README
`e2e_d1_1080p.json`, `e2e_d1_1080p.log`, `real_faces_parity.txt`

**`docs/perf/nvenc_quality`** (2 files) â€” measurement raw data / probe output behind the same-directory README
`rd_Love_hevc_cq19-31.json`, `rd_Weeds_hevc_cq19-31.json`

**`docs/perf/openvino_int8`** (2 files) â€” measurement raw data / probe output behind the same-directory README
`arcface_4c06341c33c2_int8.gate.json`, `xseg_0b57328efcb8_int8.gate.json`

**`docs/perf/openvino_seams`** (1 files) â€” measurement raw data / probe output behind the same-directory README
`seams_e2e_cpu_device.json`

**`docs/perf/phase11`** (1 files) â€” measurement raw data / probe output behind the same-directory README
`probe_phase11_interconnect.json`

**`docs/perf/phase14`** (18 files) â€” raw JSON behind phase14/README.md (interval K, 1-euro filter, flow backends, e2e Weeds; recheck_2026-10-25 at HEAD 500dfca)
`e2e_Weeds.json`, `filter_Weeds.json`, `filter_final_Love.json`, `filter_final_MonicaBellucci.json`, `filter_final_Weeds.json`, `flow_backends.json`, `interval2_Love.json`, `interval2_MonicaBellucci.json`, `interval2_Weeds.json`, `interval_Weeds.json`, `interval_Weeds_explain.json`, `interval_final_Love.json`, `interval_final_MonicaBellucci.json`, `interval_final_Weeds.json`, `recheck_2026-10-25/e2e_Weeds_HEAD500dfca.json`, `recheck_2026-10-25/interval_Weeds_HEAD500dfca.json`, `recheck_2026-10-25/interval_Weeds_HEAD500dfca_landmark_refresh.json`, `recheck_2026-10-25/swapflags.py`

**`docs/perf/prepass_session_build`** (2 files) â€” measurement raw data / probe output behind the same-directory README
`weeds_150f_2jobs_callsites.json`, `weeds_150f_4jobs.json`

**`docs/perf/providers`** (2 files) â€” measurement raw data / probe output behind the same-directory README
`providers_rtx4070_2026-10-22.json`, `providers_rtx4070_2026-10-22_cold_cache_run1.json`

**`docs/perf/qsv_encode`** (2 files) â€” measurement raw data / probe output behind the same-directory README
`mixed_encoder_concat.json`, `segment_encoder_failure.txt`

**`docs/perf/r50_static_profile_2026-10-09`** (6 files) â€” measurement raw data / probe output behind the same-directory README
`compare.txt`, `letterbox_explain.json`, `letterbox_explain.log`, `probe_baseline_512.json`, `probe_baseline_640.json`, `probe_static_640.json`

**`docs/perf/restorer_fp16_islands_2026-10-10`** (41 files) â€” RestoreFormer++ FP16 islands experiment: arms JSON, per-layer error ranking, 27 TRT engine reports
`arms_anchors1.json`, `arms_draws.json`, `arms_explore1.json`, `arms_explore2.json`, `arms_explore3.json`, `arms_explore4.json`, `arms_explore5.json`, `arms_explore6.json`, `arms_frontier.json`, `arms_outputs.json`, `composite_control_native_fp32.json`, `composite_report.json`, `engines/anc_add_all.json`, `engines/anc_add_dec.json`, `engines/anc_add_enc.json`, `engines/anc_ort14.json`, `engines/draw_l2.json`, `engines/draw_l3a.json`, `engines/draw_l3b.json`, `engines/draw_l3c.json`, `engines/draw_l4.json`, `engines/draw_l5.json`, `engines/isl_allconv.json`, `engines/isl_attn.json`, `engines/isl_dec.json`, `engines/isl_dec_hi.json`, `engines/isl_dec_lo.json`, `engines/isl_enc.json`, `engines/isl_enc_quant.json`, `engines/isl_enc_quant_declo.json`, `engines/isl_nonconv.json`, `engines/isl_norm.json`, `engines/isl_norm_attn.json`, `engines/isl_q_enchigh.json`, `engines/isl_q_enclow.json`, `engines/isl_quant.json`, `engines/isl_softmax.json`, `engines/native_fp16.json`, `engines/native_fp32.json`, `rank.json`, `validate.json`

**`docs/perf/sparse_scan_stage1`** (6 files) â€” Stage-1 sparse pre-scan results (Love, Monica, Weeds, angle, angle_inverted, angle_rot90)
`Love.json`, `Monica.json`, `Weeds.json`, `angle.json`, `angle_inverted.json`, `angle_rot90.json`

**`docs/perf/stage2_seek`** (3 files) â€” measurement raw data / probe output behind the same-directory README
`seek_vs_grab.json`, `seek_vs_grab_synthetic.json`, `vfr_landing_offsets.txt`

**`docs/perf/stage3_gpu_blend`** (1 files) â€” measurement raw data / probe output behind the same-directory README
`probe_stage3_blend_gpu_vs_cpu.json`

**`docs/perf/stage3_plan`** (17 files) â€” Stage 3 plan coverage per clip (love240, monica300, weeds300, angle, angle_inverted; nfa0/nfa1 = no-face-action variants; geom_*)
`after/angle_inverted_nfa1.json`, `after/love240_nfa1.json`, `after/monica300_nfa1.json`, `after/weeds300_nfa1.json`, `angle.json`, `angle_inverted_nfa0.json`, `angle_inverted_nfa1.json`, `angle_nfa1.json`, `geom_love240.json`, `geom_love240_916.json`, `love240.json`, `love240_nfa0.json`, `love240_nfa1.json`, `monica300.json`, `monica300_nfa1.json`, `weeds300.json`, `weeds300_nfa1.json`

**`docs/perf/stage3_resident_graph`** (1 files) â€” measurement raw data / probe output behind the same-directory README
`probe_stage3_graph_and_warp.json`

**`docs/perf/stage4_gop`** (6 files) â€” GOP/seek plan + render timing JSON per clip (love240, monica300, weeds300)
`love240.json`, `monica300.json`, `plan_love240.json`, `plan_monica300.json`, `plan_weeds300.json`, `weeds300.json`

**`docs/perf/stage4_telemetry`** (1 files) â€” measurement raw data / probe output behind the same-directory README
`love240_sparse.json`

**`docs/perf/stage5_verify`** (3 files) â€” 3-stage pipeline verify JSON + yaw warp detection text
`first_run_before_range_fix.json`, `verify_three_stage_pipeline.json`, `yaw_warp_detection.txt`

**`docs/perf/stage6`** (5 files) â€” pre-pass round-trip, regression benchmark, source roll invariance, identity/resume verifies
`real_prepass_roundtrip.txt`, `regression_benchmark.txt`, `source_roll_invariance.json`, `verify_stage0_identity.json`, `verify_stage_resume.json`

**`docs/perf/stage7`** (1 files) â€” measurement raw data / probe output behind the same-directory README
`regression_benchmark.txt`

**`docs/perf/stage8`** (2 files) â€” measurement raw data / probe output behind the same-directory README
`pcie_hops.json`, `verify_gpu_recovery.json`

**`docs/perf/stage9`** (7 files) â€” stage-4 report verify JSONs + UI screenshots (history, processing live/report, settings report)
`first_run_before_count_fix.json`, `second_run_vfr_ground_truth_wrong.json`, `ui_history_report.png`, `ui_processing_live.png`, `ui_processing_report.png`, `ui_settings_report.png`, `verify_stage4_report.json`

**`docs/perf/sustained_throughput_2026-10-11`** (10 files) â€” (non-md files) sustained throughput harness data
`baseline_null1.log`, `baseline_null2.log`, `baseline_telemetry_null1.csv`, `baseline_telemetry_null2.csv`, `benchmark_results.csv`, `final_arm1.log`, `final_arm2.log`, `final_telemetry_arm1.csv`, `final_telemetry_arm2.csv`, `telemetry_test.csv`

**`docs/perf/swap_batch_equivalence_2026-10-10`** (2 files) â€” measurement raw data / probe output behind the same-directory README
`equiv_with_cuda_control.json`, `equiv_with_cuda_control.log`

## 3. app/tests/ (717 files)

Counts: `#tests` = regex count of `def test_` (pytest items; parametrization not expanded). Purpose = module docstring/header (first ~190 chars) or first test names.

### 3.1 Subsystem index

| group | files |
|---|---:|
| B. Pipeline stages: pre-pass, plan, sparse scan, stage cache, stage profiling | 29 |
| C. Detection / alignment / landmarks / pose / angle / orientation | 53 |
| D. Tracking / temporal / smoothing / scene-cut / optical flow | 50 |
| E. Faceset (.fsz) / identity / embedding / recognition / 3D / face roles | 35 |
| F. Swap / enhance / restore / mask / blend / colour / quality | 88 |
| G. Compute: ORT / TensorRT / CUDA / VRAM / providers / OpenVINO / precision / runtime | 95 |
| H. Queue / jobs / checkpoint / projects / lifecycle | 24 |
| I. Video IO / FFmpeg / encode / audio / output / preview | 25 |
| J. API / routes / server / terminal / telemetry / diagnostics / safety | 27 |
| K. UI contract / React / launcher / settings / install / repo hygiene | 38 |
| L1. A/B harness scripts (ab_*), not pytest tests | 30 |
| L2. Probe scripts (probe_*), not pytest tests | 58 |
| L3. Diagnostic scripts (diag_*) | 21 |
| L4. Other harness / data / support files (non-test_) | 134 |
| M. Other test_ files | 10 |

### 3.2 Behaviours certified (sampled from test names; evidence = test files)

- **Angle / pose**: `test_pose_source_selector.py` (15): prefers matching lateral pose over frontal; yaw 30Â° between bank 0/60 â†’ conservative 3D bridge; yaw 58 w/o profile source â†’ 3D; inverted orientation never hidden by good yaw match; hysteresis stops near-tie flips; yaw grid frontalâ†’profile w/o frontal bias. `test_pose_canonicalization.py`: roll 60Â° uprighted with exact inverse; profile anchors cannot displace model template. `test_head_pose.py` (30): canonical (yaw,pitch,roll) order never guessed from magnitudes; EPnP poses rank profile as least frontal. `test_angle_portfolio.py` (14): lattice bins follow physical head direction, nearest-neighbour fill, pitch relative to person's neutral (near-frontal only), fused embedding frontal-only. `test_gpu_filmstrip_and_scene_cuts.py`: bitstream cut detector spikes + chapter markers, sparse pre-scan HW path, trim-range keyframe mid-points. `test_stage8_border_and_roll_geometry.py`, `test_visibility_alignment.py`, `test_face_frontalize_orientation.py` cover alignment under roll/border.
- **Temporal**: `test_temporal_smoother.py` (37): 1-euro beta continuous across threshold, monotonic in velocity, velocity normalised by face scale, frame gap declines instead of blending, per-track state isolated. `test_flow_coast.py` (34): similarity fit recovers translation to sub-pixel, rotation+scale, chains over 10-frame dropout, refuses jumps beyond LK range, featureless/two-equal-motions not believed. `test_trajectory_cache.py` (34): known similarity recovered exactly, non-square pixels fitted in display space, unusable rows NaN, flags say why a face is not a detection, pose in head_pose convention. `test_face_tracker.py`: crossing faces keep track ids. Also tracks: assignment, crossshot, gapfill, reid, stitch, coasting; temporal identity/occlusion/expression/quality/color/compositing/smart_skip/parallel_blocks.
- **Queue / lifecycle**: `test_queue.py` (33): jobs survive restart, interrupted job â†’ runnable again, restart does not auto-resume batch, stale reorder cannot drop a job, running job cannot be edited/removed, clear keeps the mid-render job, target resolved by name not stored index, missing target fails that job only. Plus `test_queue_segments.py`, `test_project_checkpoint.py`, `test_batch_matrix_queue.py`, `test_progress_chunks.py`.
- **Compute**: provider fallback is loud, provider discovery safe, TRT build guard/override/context manager, per-model precision, canary cosine floors, FP16 islands, VRAM/OOM guards, small-card admission, zero-copy ring, CUDA graph/gridsample5d.
- **Video IO**: ffmpeg writer colour/lifecycle, NVENC/QSV encode, range seek, NVDEC reader, finalize web video, early close, frame cache/ws.


### B. Pipeline stages: pre-pass, plan, sparse scan, stage cache, stage profiling (29 files)

| file | #tests | purpose / what it certifies |
|---|---:|---|
| test_audit_sync_three_stages.py | 6 | Audit verification test suite: 1. Monotonic PTS-to-Frame-Index Canonical Mapping & Scene-Cut Propagation. 2. Timeline [trim_in, trim_out] Offset Contract in TrajectoryCache. 3. Persistent (s |
| test_prepass_interval.py | 17 | `prepass_detect_interval` (K): one detector call per K frames, flow-carried faces between. What is pinned here, and why each one is a test of its own: * the POLICY (`roop/prepass_policy.py`) |
| test_prepass_interval_pipeline.py | 17 | `prepass_detect_interval` through the REAL pre-pass (`_precompute_temporal`), on real image frames. The scheduler tests (test_prepass_interval.py) drive `TemporalFaceTracker` alone. These dr |
| test_prepass_report.py | 6 | What the pre-pass says about itself in the per-job report (Stage 2 of `roop/stage_telemetry.py`). A render that ran with `prepass_detect_interval` > 1 found fewer faces and swapped fewer fra |
| test_prepass_stage11_policy.py | 5 | Pins the ladder-stride replay in `probe_prepass_stage11.py` (Stage 11, 2026-10-19). The probe's verdict ("striding the rescue ladder through faceless runs is lossless only WITH rewind, and t |
| test_single_clip.py |  | Dedicated single clip tester for UltraMax + RealityUX + RealSwap. Runs pure swapped output without side-by-side or AI upscaling. |
| test_small_card_admission.py | 4 | tests: small_model_is_admitted_from_live_headroom, model_is_rejected_when_live_headroom_is_insufficient, unknown_model_size_stays_on_cpu |
| test_small_card_prepass.py | 14 | The sub-7GB pre-pass stops paying for rescues that cannot succeed. Sampling the live temporal pre-pass on the RTX 3060 Laptop 6GB (2026-09-29, py-spy, 40 s) put 72% of its wall clock in resc |
| test_sparse_range_scan.py | 51 | Unit tests for the Stage 1 Sparse Auto-Range Pre-Scan (roop/sparse_range_scan.py). Nothing here needs a GPU. The demuxer is driven by a FakeCap that records every call, the detector by a pro |
| test_stage0_preflight.py | 34 | Stage 0 (Phase 6): the frozen source-identity table. The table's one promise is that it answers EXACTLY what the live per-face selection would have, for every head pose. So the central tests |
| test_stage11_backend.py | 20 | Stage 11 plumbing: request -> roop.globals, the Face Bank's per-sample scores, the confidence lane route, and the UI-frame <-> pipeline-frame conversion at the real ProcessMgr sites. The pur |
| test_stage13_noface_passthrough.py | 2 | Stage 13 asked for a "no-face passthrough": frames the pre-pass marked face-free should skip every GPU stage. That path already exists in `ProcessMgr.swap_faces` (the `_temporal_faces` cache |
| test_stage14_async_state.py | 29 | Stage 14: stale asynchronous state across preview, queue, render, session. Every test here drives the REAL boundaries (api.preview, api.trigger_swap, api._run_swap's request derivation, rout |
| test_stage14_hybrid_masker_sam2_unused.py | 3 | Stage 14 asked for a SAM 2 sliding-window memory bank in `face_engine.pipeline.hybrid_masker.HybridVideoMasker`. `step()` never calls SAM 2. The "SAM 2 face hull mask" is a landmark ellipse  |
| test_stage14_trt_vram_queues.py | 32 | Stage 14: TensorRT compile receipts + terminal spinner, the runtime VRAM rungs, Decode/GPU/Encode queue levels. Every test drives the REAL call site (`predictor.warmup_session`, `UnifiedRunt |
| test_stage15_features_regression.py | 11 | Stage 15: Unit Tests for Features from Stages 9â€“14. Covers: 1. Terminal Non-TTY Logger Safety (Stage 9) 2. Runtime Anomaly Watchdog Triggers & Crash Autopsy (Stage 10) 3. Segment Custody Ver |
| test_stage15_identity_audit.py | 17 | Stage 15 audit regressions: faceset pipeline and identity routing. Three defects found by the Stage 15 audit, each pinned here: B-1 In single-person "Selected face" mode the source slot hand |
| test_stage19_desktop_capture.py | 11 | Stage 19: Desktop Window & Region Capture and Livecam Subsystem Unit Tests. Covers: 1. Native Desktop and Window Capture Provider (Win32 GDI / OpenInputDesktop / Synthetic fallback) 2. Inter |
| test_stage2_plan.py | 14 | stage2_frame_plan: a read-only index of the frames that hold a face matched to a source. It is a view over `_temporal_faces` + `_track_source_map` (the real Stage 2 -> Stage 3 hand-off), so  |
| test_stage2_precision_profiles.py | 8 | Stage 2: the FP32 locks, and the TensorRT optimisation profiles each model can actually take. Two things the Stage 2 brief asserted that the shipped files and the measurements say otherwise  |
| test_stage2_range_tracking.py | 9 | Stage 2 pre-pass over Stage 1 active ranges: what a track may carry across a gap. The brief for this stage feared that a locked track from the end of range i could use its position to claim  |
| test_stage3_cache_precision.py | 3 | The faces Stage 2 hands to Stage 3 keep float32 and image-space coordinates. `_build_temporal_faces` gap-fills (interpolates) and smooths landmarks. Doing that arithmetic in Python floats or |
| test_stage3_retry_skip.py | 8 | Stage 3 does not re-detect a frame the temporal pre-pass already scanned. With "Retry rotated" chosen, every frame that swaps nothing used to run `retry_rotated`: up to two more full detecti |
| test_stage8_border_and_roll_geometry.py | 10 | Stage 8 geometry: faces cut by the frame border, and heads rolled +-60 degrees. No models. These pin the arithmetic that every detector result then flows through: the ROI window, the alignme |
| test_stage8_roi_derotation.py | 14 | Stage 8: a track last seen sideways has its ROI stood upright BEFORE the detector sees it. Measured 2026-10-19 on real Love / Weeds / Monica Bellucci / b1 frames rolled by +90 / -90 / 180 de |
| test_stage_cache.py | 68 | Stage 1 / Stage 2 persistence (Phase 6): the codec, the project record, and the adapter. The cache has one job: a resumed project must get back EXACTLY what the interrupted run's pre-passes  |
| test_stage_profiler.py | 3 | Regression tests for the opt-in Phase 14 stage profiler. |
| test_stage_telemetry_and_busy.py | 27 | Phase 4 (REST + telemetry): the 3-stage progress payload, the busy preview, the 200 ms start. Three things the React client depends on, each driven through the real boundary: * `/ws/telemetr |
| test_sustained_throughput_fixes.py | 4 | Targeted regression tests for sustained-throughput fixes. These tests stay CPU-only. They verify the contracts that prevent repeated disk work, stale temporal-state copying, excessive VRAM a |

### C. Detection / alignment / landmarks / pose / angle / orientation (53 files)

| file | #tests | purpose / what it certifies |
|---|---:|---|
| test_adaptive_enhancer_visibility.py | 9 | An adaptive run that restored nothing must say so, and say why. MEASURED, not hypothesised. On the RTX 4070, the shipped BALANCED profile on this project's own locked fixture (`double/d4.mp4 |
| test_alignment.py | 38 | estimate_norm â€” the 5-point crop alignment, and the opt-in profile variant. The headline guarantee here is that the alignment fit is a pure similarity. It is opt-in precisely because it chan |
| test_all_7_inverted.py |  | Test and process all 7 video clips from <MEDIA_DIR>/inverted/ using RealSwap + RealityUX + UltraMax (natural saturation, dark spots, hififace eyelashes). Extracts inspection preview frames a |
| test_angle_flicker_fix.py | 6 | tests: action_for_roll_hysteresis_lower_bound, action_for_roll_hysteresis_upper_bound, resolve_track_rolls_eliminates_boundary_chatter |
| test_angle_portfolio.py | 14 | Contract for roop/angle_portfolio.py: the 9-bin lattice, selection, fusion, export. |
| test_angle_scan_api_hooks.py | 5 | api.py's /ws/angle-scan hooks against the REAL target-context state: _angle_scan_resolve (which clip, which person, their angle bank) and _angle_scan_apply (the portfolio reaching the person |
| test_auto_angles_intake.py | 20 | What /api/target/auto_angles is allowed to put in a person's angle bank. The bank is the reference every identity gate in the swap measures against, and each of them takes the MINIMUM over a |
| test_bystander_bank.py | 23 | Stage 16: the differential-margin lock gate and the bystander bank. Pure numpy; no GPU, no detector. Every refusal test is paired with an acceptance test on the SAME inputs one notch the oth |
| test_bystander_veto.py | 7 | Stage 16: a track closer to a registered bystander than to the target is not the target. The pre-existing gates bind a track to a person on ABSOLUTE distance to the captured stills, plus a m |
| test_compositing_fallback_visibility.py | 7 | A quality layer that fell back must say so, once per cause. Three sites in `procmgr_masking.paste_upscale` / `process_mask` catch any exception and continue with the legacy path: temporal ma |
| test_detect_cost.py | 26 | Guards on the two per-face costs that turned a GPU-bound pipeline CPU-bound. Both were correctness fixes that shipped without a cost gate: the outcome guard re-ran a FULL face analysis on ev |
| test_detect_overrides.py | 6 | A per-call det_size/det_thresh override must reach the engine actually in use. `_rescue_downscaled` is the close-up rescue, and it is the only rescue in the ladder that works by changing det |
| test_detection_interaction_improvements.py | 16 | Targeted tests for face detection and swapped face interaction improvements. Covers: 1. Multi-scale detector explicit single scale / pyramid off fix (Fix D7). 2. Scale pyramid coordinate inv |
| test_detector_failure_visibility.py | 6 | A swallowed detector exception must not be silent. `face_util.get_all_faces` catches every exception and returns `[]`. That return value is correct and must not change -- callers all over th |
| test_detector_fixed_size.py | 8 | A fixed-shape detector export must ignore face_detector_size, not break. `face_detector_size` is a single global shared by every detector engine, but the engines are not alike: retinaface's  |
| test_detector_pools.py | 9 | The hybrid detectors must not go back to one shared instance behind a mutex. Every hybrid engine (retinaface / yoloface / yunet) brings its OWN detector and only borrows buffalo_l's aux mode |
| test_detector_resolution_options.py | 8 | The detection-resolution dropdown must offer the size that measures best. `detect` is the largest stage in a render â€” 42.4% of one measured 60,460-frame job. Measured on an RTX 4070, retinaf |
| test_enhancer_align.py | 12 | Enhancer alignment, and the paste-scale guard next to it. The restorers here learned their priors from FFHQ-aligned 512 faces, but the crop they are handed comes from the SWAPPER's template, |
| test_enhancer_dmdnet_landmarks.py | 7 | DMDNet must not die when a face carries no 106-point landmarks. `insightface`'s `Face.__getattr__` returns None for a key the analyser never wrote, so `face.landmark_2d_106` is None -- not a |
| test_exception_visibility.py | 3 | Regression guards for the repository-wide fallback observability contract. The production tree contains compatibility fallbacks that must keep rendering, but none should disappear errors wit |
| test_face_contact.py | 25 | face_contact â€” the detector and the recogniser when two faces are touching. Two rules, both narrow, both easy to get wrong in the direction that costs a real face rather than a phantom. Ever |
| test_face_frontalize_orientation.py | 4 | face_frontalize must never hand the swapper a mirrored face. Until 2026-09-25 the frontal reference was re-projected at rvec = 0. _REF3D_68 is y-up with the nose toward +z and an OpenCV came |
| test_face_overlap.py | 43 | face_overlap â€” who owns a pixel when two swapped faces meet. Each face is pasted under a matte built from its own geometry alone: a hull that has been dilated, given a forehead extension rea |
| test_face_pose_source.py | 13 | Where the pipeline gets a head's yaw and pitch from, and what breaks when it lies. Two consumers act on the target face's angles, and both are invisible failures when the number is wrong â€” n |
| test_face_sanity.py | 17 | face_sanity: every Face the analyser returns is either usable or visibly dropped. Uses InsightFace's own Face, because its `__getattr__` (None for a missing key instead of AttributeError) is |
| test_head_pose.py | 30 | Head pose: one type, one adapter, one convention across the source bank. Four pose representations are live (see roop/head_pose.py). Two defects came from mixing them, both on the multi-face |
| test_identity_detail_visibility.py | 6 | A requested identity-detail restoration that cannot run must say so. `identity_detail_strength` is a strength dial, so a non-zero value is a statement of intent. `FaceSet.identity_detail_for |
| test_landmark_mask.py | 13 | create_landmark_mask / _mask_crop_box â€” the paste-time face hull. The 106-point landmark model stops at the eyebrows, so the mask extends a forehead on top. That extension used to be project |
| test_landmark_refresh.py | 12 | roop/landmark_refresh.py and its use by the pre-pass gap bridge. The helper is tested with a fake landmark model (anything with ``get(img, face)``); the bridge by calling ``TrackingMixin._fl |
| test_landmark_symmetry.py | 18 | Occlusion-aware landmark symmetry inpainting, and every case it refuses. The refusals matter more than the repairs. Inpainting replaces a MEASUREMENT with an ESTIMATE, so a repair applied wh |
| test_nms.py | 21 | Non-max suppression: one face detected twice vs two faces touching. Plain NMS answers both questions with one IoU number, and at the 0.40 default the second one loses â€” a head partly behind  |
| test_no_face_action.py | 3 | The "no face" policy must be the policy that was chosen. `no_face_action` is a five-way choice about what to do with a frame the swap did not land on. Four of the five are non-retry policies |
| test_open_mouth_pose.py | 15 | An open mouth reads as head pitch â€” unless the solve knows about jaws. Two of the five arcface keypoints are the MOUTH CORNERS, and the jaw is a moving part. Anything that solves head pose f |
| test_oral_cavity_landmarks.py | 6 | The oral-cavity restore must read the INNER lip ring of InsightFace's 106 landmarks, never the outer upper edge. 2026-09-27, Love.mp4: the landmark fallback took indices 66..71, three of whi |
| test_orientation.py | 18 | The roll resolver: does continuity recover the roll the detector loses? The failure being modelled is measured, not imagined â€” on a profile plate rolled past ~90 degrees the eye->mouth midli |
| test_phase6_pose_quality.py | 4 | Dependency-light contracts for the Phase 6 real-photo evaluation tool. |
| test_pose_canonicalization.py | 5 | Synthetic invariants for roll canonicalisation and profile alignment. Run from ``app`` with: env/Scripts/python.exe -m unittest tests.test_pose_canonicalization |
| test_pose_quality.py | 17 | Contract for roop/pose_quality.py: perspective pose, quality factors, batch API. |
| test_pose_ratios.py | 11 | kps_pose_ratios â€” the 5-keypoint yaw / pitch proxies. These exist because the heuristic they replaced (nose asymmetry between the two eyes) is NOT monotonic in yaw: it peaks around 30-40 deg |
| test_pose_shape.py | 5 | How wrong the 5-point pose solve is on a head that is not the reference head. This is the measurement that decides whether the three angle layers (`yaw_align='pose'`, the hidden-surface trim |
| test_pose_source_selector.py | 15 | Phase 5 tests for pose-aware source selection and safe geometry hints. |
| test_pyramid_reuse.py | 16 | MultiScaleFaceDetector.detect: reuse the single pass, and where the extra scales run. 1. When the adaptive single pass triggers the pyramid, its result is the pyramid's scale-1.0 level and i |
| test_retinaface_dynamic.py | 3 | Regression tests for the dynamic-shape RetinaFace r50 export. |
| test_retinaface_gpu_engine.py | 23 | detector_engine 'retinaface_r50_gpu' (roop/retinaface_gpu_engine.py). Everything here pins the LOGIC around the network with a fake `_infer`: when the pyramid triggers, that the single pass  |
| test_rotated_face_match.py | 3 | Autorotate must re-detect THIS face, not whichever face is leftmost in the cut. The rotated cut is padded 45% of the face size each side, so two faces in contact are both in it. `process_fac |
| test_rotated_pass.py | 10 | The rotated rescues run the aux models only on faces they keep. `_rescue_rotated` and `_detect_faces`'s partial-miss rescue used to detect on a rotated frame WITH aux (recognition, 106- and  |
| test_routes_angle_scan.py | 12 | Contract for routes_angle_scan.py: the /ws/angle-scan pipeline and the session endpoints the React Biometric Angle HUD drives. Only the router is mounted (no api.py, no GPU): the detector, l |
| test_swap_faces_detected_list.py | 2 | `faces` inside ProcessMgr.swap_faces is THIS FRAME'S detected faces. 4bd577d (2026-09-24) reused the name for a source faceset's faces while building `allowed_source_indices`: faces = getatt |
| test_temporal_detection_optimization.py | 5 | tests: histogram_scene_cut_detection, scale_face_coordinates, dynamic_detection_interval_lifecycle |
| test_trt_settings_visibility.py | 4 | Tests for TensorRT settings visibility, provider distinction, and admission state. |
| test_ui_perf_settings_exposed.py | 7 | Every performance tunable the app persists must be reachable in the UI. WHY THIS IS A SEPARATE GUARD FROM test_ui_settings_catalog.py That one checks the Settings panel and the command-palet |
| test_verify_detector_routing.py | 5 | `swap_moved_the_face` (the post-swap verification) detects through whichever engine is active. With `detector_engine == 'retinaface_r50_gpu'` (the face_engine on-device RetinaFace) the verif |
| test_visibility_alignment.py | 18 | Stage 17: occlusion-visibility-weighted alignment (pure numpy). Every correction test has its control: the same inputs through the UNWEIGHTED solve must be visibly wrong, or the test proves  |

### D. Tracking / temporal / smoothing / scene-cut / optical flow (50 files)

| file | #tests | purpose / what it certifies |
|---|---:|---|
| test_api_schema_parity.py | 9 | Stage 15: Fast Schema Parity and API Contract Tests. Verifies that FastAPI route response models in `app/api.py` and its mounted routers match the expected JSON keys consumed by the React UI |
| test_blur_and_blink.py | 11 | Unit tests for motion-blur harmonization and Eye Aspect Ratio (EAR) blink preservation. |
| test_dispatch_tracker_ordering.py | 15 | The dispatch tracker is order-dependent, and it was shared by every worker. `FaceTracker.update` predicts every track forward by `dt`, increments `missed` on all of them, and clears it only  |
| test_execution_batch_size_flag.py | 8 | --execution-batch-size must reach ProcessMgr's batch ceiling and stay there. |
| test_execution_providers.py | 40 | Execution-provider upgrades: TensorRT shape profiles, NPU offload, fallback. Three of these assertions exist because of a defect this repo has already paid for once, and they are the reason  |
| test_expression_gaze_blink.py | 17 | Expression Transfer Strength / Eye-Gaze Follow Ratio / Blink Sync. The split must be free for every config written before it: with no gaze ratio the keypoint maths has to be BIT-identical to |
| test_face_tracker.py | 1 | Synthetic collision coverage for persistent Kalman/Hungarian face tracks. |
| test_flow_backend.py | 21 | The point-tracking ladder under the flow chain (`roop/flow_backend.py`). Ground truth is a known similarity applied to a blurred-noise texture (the same fixture family as test_flow_coast.py) |
| test_flow_coast.py | 34 | Stage 16: Lucas-Kanade landmark propagation through a detector dropout. Synthetic, deterministic images: a blurred-noise "head" texture that can be moved by a known similarity transform, wit |
| test_flow_gapfill.py | 12 | Stage 16: the flow-bridged gap fill in `_build_temporal_faces`. The scan loop records, per dropped frame, where optical flow says the face went (`track['flow'][frame] = FlowResult`) and, at  |
| test_gpu_filmstrip_and_scene_cuts.py | 8 | Tests for GPU Filmstrip thumbnail generation, scene cut endpoints, and Hardware Stage 1 Pre-Scan integration. |
| test_identity_lock.py | 16 | Identity crossover lock (Stage 11): hysteresis, trajectory, pins and the confidence strip. The crossing is SIMULATED with known truth: two bodies on straight paths cross at frame 40, and for |
| test_kalman_kps_stabilizer.py | 13 | KalmanKpsStabilizer (`stabilize_method: kalman`): a drop-in, opt-in keypoint smoother. Measured 2026-10-03 on a static 266 px face rendered with sensor noise through x264 (detector sigma 0.5 |
| test_mask_flow_warp.py | 7 | MaskStabilizer's flow-warp (setting `mask_flow_warp`) and the XSeg guided filter (`mask_guided_filter`). Pure CPU; no model is loaded. |
| test_multi_face_reid_cut.py | 15 | Comprehensive tests for Scene Cut Detection, Buffer Flushing, Face Identity Clustering (DBSCAN/Agglomerative), Hungarian Matching Re-ID, Face Bank, Multi-Target Mapping, and Identity Confide |
| test_one_euro_vec.py | 23 | The vectorized zero-phase 1-euro landmark filter (`roop/one_euro_vec.py`). The measured claims are asserted against the stored detections in tests/data/synthetic_face_jitter.npz (a still and |
| test_scene_cut_keyframe_stats.py | 14 | The hardware cut detector must not call a keyframe a scene cut because it IS a keyframe. The keyframe pre-scan samples only I-frames. An I-frame decodes as ~100% intra coding units and at a  |
| test_selected_track_hold.py | 17 | The swapped face must not blink out because ONE frame read badly. Measured on the clip this was reported from (Selected-person mode, one person, one faceset, 800 frames at t=33 min): 996 det |
| test_settings_schema.py | 10 | app/settings.py is the single source for settings; the generated files must match it, and the shared env mapping must behave exactly as the old hand-written block in run.py did. Three genera |
| test_shot_boundaries.py | 12 | Shot boundaries must be detected, and must confine tracking. WHY THESE EXIST. The pipeline already had hard-cut detection and already had two consumers for it, but the gate was a fixed `mean |
| test_stab_block_dispatch.py | 12 | Stabilized-render block geometry: more blocks per chunk, same block grid. THE PROBLEM. `_run_stab_parallel` decodes the clip in chunks and splits each chunk into contiguous blocks handed to  |
| test_stab_block_mult.py | 10 | ROOP_STAB_BLOCK_MULT: how many times its warm-up a stabilization block is. A block discards `warm_up` frames of work, so its redundant priming is 1/mult of its work (25% at the default 4x).  |
| test_stab_chunk_overlap.py | 5 | The stabilization pool is persistent and chunks overlap; output order must not move. `_run_stab_parallel` used to spawn a thread per block-slot for every chunk and join them all before the n |
| test_stab_dedup.py | 32 | StabRawCache: the warm-up dedup's bookkeeping, and the eligibility rules that keep it off when it could be wrong. The cache's job is to make two neighbouring stabilization blocks compute an  |
| test_stab_live_chunk_accounting.py | 8 | The stabilizer's RAM budget must count the buffers the code actually holds. `_run_stab_parallel` keeps several chunk-sized frame buffers alive at once, and `_default_stab_chunk_mb` divides t |
| test_stab_parallel_lifecycle.py | 2 | Regression tests for bounded stabilized-reader/writer shutdown. |
| test_stab_warmup.py | 18 | Parallel stabilization must derive its warm-up, not guess it. Splitting a clip into contiguous blocks and giving each its own smoothing filter is only output-equivalent to running the clip i |
| test_temporal_color.py | 11 | roop/temporal_color.py: the split transform must equal the shipped one, and the recurrence must be safe. |
| test_temporal_compositing.py | 6 | Phase 12 compositor implementation and regression tests. |
| test_temporal_expression.py | 9 | Phase 8 unit tests: target expression state, asymmetry and integration hooks. |
| test_temporal_faces_replay.py | 7 | The track builder's per-frame handoff to the swap phase. `_build_temporal_faces` is the ONLY thing that turns whole-clip tracks into the `{frame_idx: [Face, ...]}` map the swap phase consume |
| test_temporal_identity.py | 12 | Regression contracts for the opt-in Phase 6 temporal identity layer. |
| test_temporal_occlusion.py | 9 | Regression contracts for Phase 7 temporal occlusion state. |
| test_temporal_parallel_blocks.py | 16 | The opt-in temporal engines on the parallel-block path. Until 2026-08-31 `ROOP_TEMPORAL_IDENTITY` or `ROOP_TEMPORAL_OCCLUSION` pinned the whole render to one worker. Measured on the locked 6 |
| test_temporal_quality.py | 10 | tests: normal_frame_is_pass_through_without_correction, detects_identity_drift_and_reselects_source, detects_brightness_and_skin_color_jumps |
| test_temporal_smart_skip.py | 12 | Unit and integration tests for Motion-Compensated Temporal Frame Reuse (Smart-Skip). |
| test_temporal_smoother.py | 37 | Contracts for roop/temporal_smoother.py. These assert PROPERTIES rather than thresholds wherever possible, per the project's test convention: a number pinned to today's tuning fails on the n |
| test_temporal_smoother_wiring.py | 32 | Proof that ProcessMgr actually REACHES roop/temporal_smoother.py. This file exists because of `memory/adaptive-controller-was-unreachable.md`: the safe adaptive controller was constructed, c |
| test_temporal_tracker.py | 16 | Behavioral contract for Phase 3 temporal tracking and detection policy. |
| test_timeline_tracks.py | 33 | What the React timeline's lanes read from the backend (``roop/timeline_tracks.py`` + the two routes). The lanes draw whatever these return, so every wrong answer here is a lane that looks ri |
| test_track_assignment.py | 19 | Which tracklets are allowed to inherit a selected person's source. The reported failure: with two people on screen and ONE of them selected, the swap is correct while both are visible, but a |
| test_track_crossshot.py | 24 | Carrying identity across a CUT, and the shape-change half of the outcome guard. Both come from the same report â€” "one faceset swaps and the other does not, at different angles, even when the |
| test_track_gapfill.py | 10 | Gap-fill continuity and claim order â€” the two rules that stop a swap being painted onto the background while the real face goes unswapped. The temporal pre-pass fills a track's detection mis |
| test_track_reid.py | 11 | The appearance-only fallback in the tracking scan, driven end to end. The scan associates a detection to a track on two kinds of evidence: spatial continuity (IoU against the predicted box)  |
| test_track_stitch.py | 27 | Chaining tracklets that are one person interrupted. The reported symptom is "the swap flickers enormously when the face touches an object, at lateral poses, whatever the mask engine". The au |
| test_tracker_coasting.py | 16 | Does a face survive an occlusion, and does it stop surviving one in time? Every test here drives `roop.tracker` directly with synthetic detections, so a pass means the mechanism works -- NOT |
| test_trajectory_cache.py | 34 | The Stage 2 smoothed-trajectory cache (`roop/trajectory_cache.py`): matrices, pose, file, checkpoint, route. The matrices are solved in closed form for every face at once. A cache that held  |
| test_ui_shortcut_keys.py | 6 | No two keyboard actions that can be live together may share a key. Every keydown handler in the React UI is registered on `window`, and `preventDefault()` does not stop a sibling listener on |
| test_ui_slider_tracker.py | 18 | Slider Tracker bar â€” that its sliders agree with the panel they mirror. The tracker is a SECOND view of settings the Face Swap panel already exposes. Two views of one number is a drift risk  |
| test_zero_phase_smoother.py | 14 | ZeroPhaseTrackSmoother (Phase 7): the pre-pass's landmark smoother, run forward AND backward. The claims, each pinned: * a causal filter trails fast motion and a zero-phase one does not (aga |

### E. Faceset (.fsz) / identity / embedding / recognition / 3D / face roles (35 files)

| file | #tests | purpose / what it certifies |
|---|---:|---|
| test_build_faceset_v2.py | 4 | A V2 archive is not automatically a working identity-detail source. `format_version == 2` is the check the handoff named, and on its own it is not sufficient: `faceset_v2.migrate_legacy_fsz` |
| test_calibrate_recognition.py | 22 | tools/calibrate_recognition.py: the analysis maths on cases with known answers (no GPU, no footage). |
| test_compare_recognition_swap.py | 13 | tools/compare_recognition_swap.py: the decision metrics, track stitching and compositing maths (no GPU, no video). |
| test_face_analyser_recognition.py | 30 | face_analyser: pluggable recognition backend, extract_face_embedding, IdentityBank. The backend tests use a FAKE engine class so the lifecycle (hot-swap, release, failure keeps the old engin |
| test_face_reference.py | 4 | Unit coverage for multi-shot ArcFace reference clustering. No model is loaded here: detector outputs are represented as dicts, which is also the supported runtime shape for this module. |
| test_face_roles_js.py | 1 | The Media Canvas role badges / "Exclude (never swap)" decision, run through node. react-ui/.render-check/face-roles-check.mjs imports the REAL module (src/components/faceswap/faceRoles.js) a |
| test_facemgr_v2_banking.py | 3 | Unit and integration tests for Face Manager (#/facemgr) .fsz v2 banking, pose-binned multi-angle embeddings, backward-compatible mean_embedding, and get_pose_matched_embedding GPU SLERP help |
| test_faceset_manager.py | 30 | faceset_manager: read-only embeddings of a faceset in a chosen recognition model's space. Archives are REAL .fsz files (legacy ZIPs of PNGs, V2 via the repo's own migrate_legacy_fsz); the de |
| test_faceset_mean_format_neutral.py | 4 | The bench's reference identity vector must not depend on FaceSet format. Matrix defect D.9, recorded on the RTX 3060 campaign as FOUND, NOT FIXED. `FaceSet.AverageEmbeddings()` overwrites `f |
| test_faceset_v2.py | 48 | Phase 4 tests for FaceSet V2 serialization, migration, and lookup. |
| test_identity_algebra.py | 29 | roop.identity_algebra: hypersphere blending, tangent offsets, the identity guard, direction fitting, recipes, and the wiring into ProcessMgr / the API. numpy only; the render-path proof (doe |
| test_identity_detail.py | 7 | Phase 9 identity-detail tests and measurable synthetic quality guards. |
| test_identity_lab.py | 29 | Face Manager Pose Radar, Auto-Prune and the Morph Lab (roop/identity_lab.py + routes_identity_lab.py). Pure rules are checked against hand-built sets where the right answer is known by const |
| test_identity_pins.py | 7 | User identity pins in the pre-pass binding (Stage 11). A pin is the user saying "this box, on this frame, is that person". The automatic gates in `_assign_track_sources` exist to ESTIMATE th |
| test_identity_settings_wiring.py | 5 | The Identity & tracking settings must reach the flags they claim to drive. A setting can look completely wired â€” a control in the panel, an entry in the catalogue, a key in config.yaml â€” and |
| test_identity_slerp_keyframes.py | 62 | Stage 20: Slerp / spherical-mean blending, orthogonal de-identification, keyframed identity. Every check here has a ground truth known by construction, not a number read off a run: * Slerp k |
| test_identity_stage20_routes.py | 15 | Stage 20 over the real ASGI app: the recipe's new fields, the keyframe readout, the .fsz export and the Morph Lab's 5-source / mode / de-identification paths. Exports are read back the way t |
| test_multi_identity_assignment.py | 11 | Unit tests for Hungarian Bipartite Identity Matching & Hysteresis State Machine. Simulates multi-identity trajectory crossings and identity retention for multi-person videos (e.g., 'double'  |
| test_project_provider_identity.py | 11 | The provider identity a project checkpoint is validated against. A project is written on every /api/swap and revalidated before it may be resumed. `runtime_identity` used to unwrap a provide |
| test_project_target_faces_serialization.py | 6 | A project must be able to record the target faces it exists to restore. `_project_target_faces` hands the checkpoint the detector's own facts â€” bbox, kps, embedding, landmark_2d_106, landmar |
| test_recognition_crop_keypoints.py | 15 | AdaFace reads a crop built from the DETECTOR's keypoints, not the 68-landmark-refined ones. Measured on 16 real clips (docs/development/RECOGNIZER_CALIBRATION.md): aligning from the refined  |
| test_recognition_engine.py | 30 | Recognition engine: fallback chain, post-warm-up verification, preprocessing. The fallback tests drive a FAKE onnxruntime session, because the behaviour under test is exactly the one a real  |
| test_recognition_registry.py | 11 | Recognition model registry: integrity of the table and of the resolver. Offline -- the network is replaced by a fake `requests.get`. The one thing this file cannot prove is that the live URL |
| test_recognizer_adaface.py | 15 | AdaFace as a matching-only recogniser. Two properties matter more than anything the model does: 1. OFF BY DEFAULT IS INERT. AdaFace is opt-in; with ROOP_ADAFACE unset every helper must behav |
| test_selected_face_integration.py | 3 | Discoverable wrapper for the real-frame Selected-Face harness. The heavy part -- real detector + recogniser + swapper on the composed frame -- is gated behind ``ROOP_RUN_SELECTED_INTEGRATION |
| test_selected_face_regression.py | 18 | Deterministic unit-level regression harness for the "Selected Face" bug. The bug: selecting one target person could still swap the wrong person, swap a bystander, or (its worst form) silentl |
| test_selected_face_safety.py | 6 | Regression coverage for the Selected Face no-target safety contract. |
| test_selected_target_mapping.py | 6 | Selected-face target mapping: UI payload contract + backend behaviour. Two halves, because the bug spanned both sides of the wire: 1. The UI decision itself is JavaScript, so it is exercised |
| test_source_faceset_mapping.py | 10 | Source faceset mapping stays separate from target-person identity state. These tests exercise the pure mapping contract used by both the React payload builder and the API boundary. They deli |
| test_swap_identity_isolation.py | 19 | The swapper's identity input is isolated from the tracking recogniser. Tracking may run any registered model (face_analyser.set_recognition_model); the swappers must only ever see the 512-d  |
| test_target_appearance.py | 8 | Phase 10 target-conditioned lighting/color tests and regressions. |
| test_target_exclusion_mapping.py | 3 | Stage 19: "Exclude (never swap)" survives the backend. The React UI writes an exclusion as an EXPLICIT `null` for the person (`react-ui/src/components/faceswap/faceRoles.js`). The backend ha |
| test_target_face_capture.py | 11 | Stage 12 regression tests for explicit target-face capture. The tests replace detector output with deterministic fake faces. This keeps the contract under test independent of GPU providers w |
| test_target_person_identity.py | 12 | Stage 13 regressions for stable target-person identity and mapping. |
| test_ui_recognition.py | 28 | Recognition settings: hardware tiers, catalogue, apply signals, and the routes. The tier tests feed classify_hardware() hand-made facts, so every tier is covered without that hardware; the r |

### F. Swap / enhance / restore / mask / blend / colour / quality (88 files)

| file | #tests | purpose / what it certifies |
|---|---:|---|
| test_adaptive_enhancer.py | 19 | Phase 11 adaptive enhancer contracts. These tests use a fake processor for orchestration behavior. Existing enhancer implementation tests continue to exercise each real processor independent |
| test_batch_swap_fallback.py | 14 | The sequential fallback inside RunBatch / RunBatchMulti. When a batched swap inference fails (a static-batch ONNX export, or a TensorRT engine built for batch 1), the swapper re-runs the cro |
| test_batcher_mask_exclusion.py | 11 | Cross-frame swap batching vs the swap model's own face mask. These two features are mutually exclusive by construction. The batcher coalesces crops from several worker threads into one infer |
| test_benchmark_clip_resolution.py | 11 | `benchmark_pipeline.py` must find its clip, not name one machine's drive. It defaulted to ``r"D:\\k1.mp4"`` in three places (docstring, constructor, ``--video``). The existing guard (`test_f |
| test_color_transfer_lct_exact.py | 11 | `_color_transfer_lct` and the grayscale guard are bit-identical to the implementation they replaced. The 'lighting' stage ran 6.7-6.9 ms per call (two calls per swapped face: the 256 px swap |
| test_dermal_and_tone.py | 2 | CPU regression tests for low-light dermal restoration and tone matching. |
| test_enhance_blend.py | 6 | roop.enhance_blend: Restore Ultra's low-from-swap / high-from-restorer recombination and the inner-feature region. Pure CPU. |
| test_enhance_common_gpu.py | 5 | CUDA ports of the enhancers' host post-processing must be the host's bytes, not close to them. `Enhance_GPENRealistic` / `Enhance_RestoreFormerPPlus` run their finish on the GPU through thes |
| test_enhance_gate.py | 18 | The small-face restorer gate: right decision, no flicker at the edge, truly wired. |
| test_enhancer_fp16_collapse.py | 11 | FP16 precision failures that the NaN guard cannot see. `is_usable` rejects non-finite output, which catches the LOUD failure: GPEN at 1024/2048 overflows under TensorRT FP16, `np.clip` does  |
| test_enhancer_gpen256_pro.py | 26 | Unit tests for GPEN 256 Pro: Upgraded sharper, high-texture, photo-realistic face restorer. |
| test_enhancer_gpen_realistic.py | 18 | GPEN Realistic: GPEN-512's luminance, the swapper's chrominance. Two findings these tests encode, the second correcting the first build. 1. GPEN's problem is COLOUR, not detail. It pushes th |
| test_enhancer_guards.py | 23 | The three lean-host-path enhancers: guards that cannot fire, and grain that cannot flicker. GPEN 256 Pro, GPEN Realistic and UltraMax share a post-processing lineage â€” a LUT gather in, one ` |
| test_enhancer_names.py | 3 | Every enhancer name a bench hands to `init_pipeline` must be one core matches. `roop.core.get_processing_plugins` selects the enhancer by comparing `selected_enhancer` against a chain of exa |
| test_enhancer_output_guard.py | 7 | Every restorer must refuse to hand back a non-finite frame. The shared ending â€” clip to [-1, 1], rescale, cast to uint8 â€” silently turns a NaN into a BLACK PIXEL, because `np.clip` propagate |
| test_enhancer_pool.py | 13 | The enhancer must be able to run on more than one thread at a time. `_gpu_guard` exempts a processor from the global TensorRT lock only if that processor owns a SessionPool. An enhancer with |
| test_enhancer_ultimate_profiles.py | 33 | GPEN Ultimate and Restore Ultra: the two ported "ultimate quality" profiles. WHAT THESE TWO PROFILES ACTUALLY ARE, because the name suggests more than the change is. Neither adds a network.  |
| test_enhancer_ultramax.py | 35 | Unit tests for the UltraMax enhancer. UltraMax runs codeformer.fp16.onnx directly (same weights as `Codeformer (fp16)`) on a leaner host path. Its texture restore and dual-stream engine were |
| test_enhancer_zero_copy.py | 13 | GPEN Realistic and RestoreFormer++ on the zero-copy ring: the real Run(), both paths, same bytes. The processors are built around a tiny synthetic ONNX graph on the CUDA execution provider ( |
| test_expression_fidelity.py | 14 | Expression & gaze fidelity (Stage 11): iris retention, micro-expression tension, wide-mouth teeth. The faces here are synthetic on purpose: the property under test is "the output's iris is w |
| test_expression_restore.py | 36 | The expression-restore geometry, tested without the models. Expression_LivePortrait keeps its maths in module-level functions precisely so this suite can run: no 537 MB download, no GPU, no  |
| test_eye_mouth_passthrough.py | 9 | Tests for EAR blink detection, multi-scale eyelid blending, and teeth/inner-mouth passthrough. |
| test_eye_restore.py | 9 | apply_eyes_area â€” bringing the target's own eyes back over the swap. Properties, not pixel values. What must hold: * OFF IS FREE. At strength 0, or with no keypoints, the frame comes back bi |
| test_face_parsing.py | 20 | Semantic-parsing protection for hair/bangs and eyeglass frames. The measurements these assertions encode were taken with `tests/probe_parser_protection.py` and `tests/probe_glasses_rim.py` o |
| test_faceswap_component_layout.py | 2 | Architecture guard for the FaceSwap comparison-grid boundary. |
| test_ffmpeg_resolution.py | 6 | The render path must resolve ffmpeg, not assume it is on PATH. WHY THIS EXISTS. `ffmpeg_writer.FFMPEG_BINARY` was the literal string `"ffmpeg"`, and `util_ffmpeg` built its command lines wit |
| test_ffmpeg_writer_colour.py | 10 | The output writer must convert AND tag its colour space on every path. The defect these cover: the scale-for-odd-dimensions branch and the colour conversion were an if/elif, so an odd-sized  |
| test_frame_upscale_batch.py | 5 | Frame_Upscale tile-batch correctness and bounded fallback tests. |
| test_grain_match.py | 18 | roop/grain_match.py: the estimator, the variance arithmetic, and the wiring into the real ``paste_upscale``. The estimator and the arithmetic are pure NumPy/OpenCV. The wiring tests drive `` |
| test_gridsample5d.py | 12 | The 5-D GridSample rewrite, checked against onnxruntime's own kernel. No model download and no GPU: the rewrite is pure graph surgery, so tiny synthetic graphs exercise exactly the same code |
| test_hdr_color.py | 18 | roop.hdr_color / roop.hdr_pipeline: the managed HDR / high-bit-depth path. Anchors are published values (and were cross-checked against colour-science 0.4.6 and OpenColorIO 2.5.2 when this w |
| test_light_harmonizer.py | 6 | Tests for neural environment re-lighting and normal harmonization. |
| test_lighting_specular_directional.py | 5 | Tests for Directional Relighting, Specular Skin Highlight Recovery, and Sub-Dermal Warmth. Ensures: 1. Neutral 0.0 values produce exact bit-identical no-op. 2. Specular highlight recovery ex |
| test_lipsync_acoustics.py | 8 | Stage 17: the viseme fallback opens the mouth FROM THE AUDIO, at the slipped time. Found by the real-render proof (tests/verify_stage17_render.py): the slip changed which audio second every  |
| test_lipsync_audio.py | 49 | Lip-sync â€” the parts that are provable without the MuseTalk weights. roop.lipsync_audio's frame/time mapping and Lipsync_MuseTalk.face_bbox_crop are plain functions kept out of the model cla |
| test_manual_mask.py | 9 | Manual brush mask â€” that what the preview box paints survives to the render. The brush was end-to-end dead before this: the canvas was exported nowhere, and both /api/preview and /api/swap p |
| test_mask_engine_pair.py | 17 | Two occlusion engines instead of one. When something comes in front of a face and the swap gets painted over it anyway, the usual cause is not that masking is off â€” it is that the single eng |
| test_mask_fusion.py | 14 | Stage 18: flow-advected mask fusion, hysteresis and the guided matte. Masks: 1 = keep the original (occluded), 0 = swap. Each behavioural claim has its control: the same input through a plai |
| test_mask_occlusion.py | 4 | Contract tests for the opt-in asymmetric temporal occlusion response. |
| test_mask_reuse.py | 6 | The mask must be computed once per face, not once per target. With an enhancer active, `swap_faces` masks TWICE per face â€” the swapped crop and the enhanced crop. Everything `process_mask` d |
| test_mask_roi.py | 9 | roop.mask_roi must equal the full-frame cv2 call BIT FOR BIT. The matte is non-zero only around a face, so the blur / erode / dilate run on its padded bounding box. That is only a legitimate |
| test_masking_pipeline.py | 38 | Stage 2: composite mask, Reinhard LAB colour match, temporal matrix filter, and their integration in HiFiFaceSwapper.post_process. Layers: * light: TemporalMatrixStabilizer (numpy only) -- r |
| test_matte_roi.py | 7 | The face-neighbourhood matte in ``paste_upscale`` must give the SAME frame as the full-frame matte, to within the one thing it cannot reproduce. The canvas is only allowed to exist because e |
| test_merger_postops.py | 35 | DeepFaceLab merger post-ops â€” that each one is inert until asked for. The whole premise of this family is "costs nothing when off". That is only true if every op returns its input UNTOUCHED  |
| test_mouth_custody.py | 31 | Stage 17: Viseme Mouth Custody - the mode switch, the audio slip, the silent-video fallback, timestamp alignment. Three layers, each proving something the next one relies on: * the pure rule |
| test_mouth_custody_accounting.py | 9 | Stage 17: the lip-sync model is accounted for like every other model - lifecycle table, TensorRT receipt, VRAM plan. Each claim names its site and drives it: * `Lipsync_Wav2Lip._register_run |
| test_mouth_custody_api.py | 9 | Stage 17 through the real ASGI app: the request refusals, the upload-confined waveform route, and the media editor's audio half (a real silent mp4, a real uploaded wav, the delivered file de |
| test_mouth_custody_js.py | 4 | Stage 17: the React model (`mouthCustody.js`) and the backend (`roop/mouth_custody.py`) agree on the same cases. `react-ui/.render-check/mouth-custody.cases.json` is the shared reference: a  |
| test_new_swap_models.py | 17 | InStyleSwapper256 A/B/C and CSCS â€” the wiring, and the trap CSCS exposed. Both models come from VisoMaster (GPL-3.0). Two things about them are not guessable from the spec table and were mea |
| test_nonfrontal_mask_path.py | 14 | The unwarped-crop masking path, and the premise it was built on. `process_mask` could route a face down one of two very different derivations. The normal one runs the mask model on the align |
| test_nvenc_quality.py | 13 | roop/nvenc_quality.py: the opt-in NVENC multipass / spatial AQ / temporal AQ options. Pure tests run everywhere. ``HardwareBitstream`` drives a real NVENC and is marked ``gpu``; it skips (wi |
| test_nvenc_writer_lut_kwargs.py | 2 | SegmentedVideoWriter forwards lut_path/lut_bake to whichever writer it opens; the NVENC one must accept them. |
| test_occluder_edge.py | 9 | How wide the boundary between a foreign object and the swap ends up on screen. The occluder family (`Face Occluder`, `Face Occluder v3`) thresholds its output to a hard 0/1 and then softens  |
| test_occlusion_color_pipeline.py | 4 | tests: signed_radius_erodes_and_dilates_landmark_contour, neutral_photometric_controls_are_bit_exact, warmth_and_saturation_change_only_photometric_output |
| test_occlusion_gate_population.py | 5 | The occlusion admission gate is judged on the face, not the jaw contour. Real-file finding (<MEDIA_DIR>/target_clip.mp4, frame 1): a fully visible face was marked `partial` at hidden fractio |
| test_occlusion_mask.py | 7 | Tests for foreground occlusion masking and temporal mask smoothing in face_swapper. |
| test_occlusion_wiring.py | 18 | Is the occlusion work actually REACHED on the shipped default configuration? This file exists because of the failure mode this project keeps rediscovering: code that is complete, correct, co |
| test_parser_regions.py | 10 | Mask_FaceParser region selection. The Face Parser produced all 19 CelebAMask-HQ classes and threw 10 of them away against a hardcoded array, so hair, glasses, ears and neck were permanently  |
| test_paste_interp.py | 3 | ROOP_PASTE_INTERP selects the filter of the face's inverse warp, and the default is exactly the shipped bicubic. |
| test_phase16_final_quality_gate.py | 6 | Contracts for the Phase 16 standardized final-quality gate. |
| test_quality_harness.py | 47 | tools/quality_harness.py: the metrics are real, the guard fires, and nothing in the file can fabricate a number. The harness exists because tools/benchmark_hyperswap_audit.py printed identit |
| test_quality_regression.py | 5 | Deterministic quality-regression guards for the render contract. These tests complement (rather than replace) the model-specific identity, landmark, mask, eye/mouth, stabilization, and enhan |
| test_realityux_nonface_set.py | 7 | The BiSeNet class set RealityUX subtracts, and its documentation, must agree. RealityUX lets the Face Parser subtract pixels from XSeg's swap region, but only for classes it is unambiguously |
| test_realswap.py | 48 | RealSwap â€” two swap nets behind one model entry, composited by region. hyperswap is the base for the whole face; hififace contributes only the eyelid and eyelash band. That split is by FEATU |
| test_realswap_lip_colour.py | 19 | RealSwap's lip-colour transfer: properties that must hold without a GPU. The rendered proof lives in `tests/verify_realswap_lip_colour.py`, which runs both nets on real frames and grades aga |
| test_release_signoff_vfr_color_cancel.py | 38 | Release sign-off: variable-frame-rate sources, colour description, cooperative cancel, installer manifests. What the brief asked for and where it is ground truth in this repo (a claim needs  |
| test_render_guard.py | 5 | tests: large_gpu_with_headroom_is_admitted, low_free_memory_fails_before_session_creation, small_gpu_is_not_rejected_by_total_vram |
| test_restore_audio_frame_count.py | 2 | An untrimmed render must keep EVERY video frame when the audio is a hair short. Found 2026-10-03 by tests/test_performance_regression.py: a 90-frame, 29.97 fps render came back with 89 frame |
| test_restore_audio_metadata.py | 4 | The audio mux must carry the ORIGINAL container's global metadata. WHY THIS EXISTS. `restore_audio` is the final pass of every video render: it takes the freshly encoded video (input 0) and  |
| test_restore_audio_trim_offset.py | 4 | A trimmed render's video must not start late against its audio. `restore_audio` cut the source audio with an INPUT-side `-ss` and `-c:a copy`. For stream copy that is not a cut: the demuxer  |
| test_restorer_fp16_tools.py | 11 | tools/restorer_fp16_ranking.py and app/roop/trt_native_runner.py - the parts that need no GPU. The ranking tool simulates FP16 on a CPU graph by routing a module's weights, inputs and output |
| test_restorers.py | 20 | Restorer comparison and audit: PSNR, SSIM and perceptual edge gradient. Two things in one file, because they answer the same question from opposite ends: * `python -m unittest tests.test_res |
| test_roto_polygons.py | 20 | Vector roto zones (Stage 11): outline, 2-point pin, matte and the two composites. Every assertion is on pixels or coordinates that a wrong implementation would get wrong; none is on a call c |
| test_sam2_runs.py | 17 | SAM 2 gated to the frames the earlier stages kept (Phase 7). Two layers, both without the SAM 2 checkpoint (it is not downloaded here): * `roop/sam2_runs.py`: which frames, which seeds, wher |
| test_smart_restorer_bypass.py | 9 | Tests for Smart Restorer Bypass and Multi-Resolution Dynamic TRT Swapper Registry. |
| test_stage18_lut_and_scopes.py | 6 | Unit tests for Stage 18: WebGL2 3D .cube LUT Color Pipeline, Skin-Protected Vibrance, and Real-Time Broadcast Scopes. Validates: 1. .cube domain validation invariants: strict DOMAIN_MIN < DO |
| test_startup_preflight_import_order.py | 7 | Startup must not break the `onnxruntime` module it then inspects. A previous change added `from settings import _enable_tensorrt_runtime` ABOVE `import onnxruntime` in run.py. `settings` imp |
| test_swap_audit.py | 16 | Swap audit: the breakdown that says WHY a detected face was not swapped. The audit exists to answer one report â€” "the swap flickers on and off" â€” which every identity gate in swap_faces can  |
| test_swap_audit_distributions.py | 19 | The audit must say what a gate change would BUY, not just how often it fired. Two lines in the SWAP AUDIT are the ones users act on: fallback missed (over match threshold) <- "loosen the mat |
| test_swap_batch_equivalence.py | 2 | Is the batched hyperswap the same picture as the B=1 one? (GPU-gated; `-m gpu`) env/Scripts/python.exe -m pytest tests/test_swap_batch_equivalence.py -s env/Scripts/python.exe tests/test_swa |
| test_swap_batcher.py | 1 | tests: short_result_releases_every_waiter_with_an_error |
| test_swap_canary.py | 22 | Startup canary for TensorRT swapper engines (roop/swap_canary.py). The failure it exists for: an inswapper TensorRT FP16/"mixed" engine that builds, warms up, reports the TensorRT provider a |
| test_swap_model_single_parse.py | 3 | The swap net's protobuf is parsed ONCE per Initialize. `FaceSwapInsightFace.Initialize` called `onnx.load(model_path)` twice for every emap model (inswapper, reswapper, instyleswapper): once |
| test_swapper_abstraction.py | 32 | BaseFaceSwapper contract, ModelRegistry, and swap_model flag routing. Three layers: * light (numpy only): registry resolution, tensor validation, provider names, the abstract contract; * the |
| test_target_path_resolution.py | 4 | Where a RELATIVE target path is resolved from. The backend's working directory is `app/` â€” start_react.js runs `python run.py` with `path: "app"` â€” but everything a person is likely to name  |
| test_texture_warmup.py | 9 | Block warm-up for the ordered stabilisation path: texture/mask filters take a looser seed bound than geometric ones. `_stab_warmup_frames` used to ask every active filter for the frames need |
| test_tile_diffusion.py | 18 | Tests for the high-fidelity diffusion tile restoration pass in roop-ultimate. Validates: 1. Dynamic Tiling & VRAM Control: - Overlapping tiled inference (512x512 tiles with 64px Gaussian fea |
| test_ui_light_themes.py | 10 | Light themes must repoint `--color-white`, and none may be able to miss it. Tailwind v4 compiles the whole `*-white` utility family â€” text, background, border, ring, at every alpha, under ev |

### G. Compute: ORT / TensorRT / CUDA / VRAM / providers / OpenVINO / precision / runtime (95 files)

| file | #tests | purpose / what it certifies |
|---|---:|---|
| test_autotune.py | 27 | Auto-tune decisions (roop/benchmark/autotune.py), driven by a fake GPU. The executor that runs real renders is exercised live; everything that DECIDES is here: which arms exist, the order th |
| test_autotuner_noise_floor.py | 6 | The autotuner must not promote a candidate that noise alone could produce. Gate A finding 12 (2026-08-30) recorded that `MIN_IMPROVEMENT = 1%` sits below this project's measured run-to-run s |
| test_aux_canary.py | 67 | Startup canary for the small TensorRT models (roop/aux_canary.py), cached by engine hash. What makes the guard trustworthy, and what these pin: * it FAILS a wrong picture (an engine answerin |
| test_backend_manager.py | 7 | tests: provider_name_matching_accepts_short_and_encoded_names, tensorrt_falls_back_to_cuda_then_cpu, missing_gpu_provider_is_cpu |
| test_baseline_controlled.py | 16 | The Phase 2 baseline parser must not read the wrong number as the FPS. This guards a defect that nearly reached `PERFORMANCE_BASELINE.md`, which is the locked reference every later phase and |
| test_baseline_probe.py | 31 | The baseline probes (roop/baseline_probe.py): measurement only, never behaviour. Two things are pinned and they are different kinds of thing: * the PROBE's own contract -- gated off by defau |
| test_bench.py | 68 | Guards on the hardware benchmark's decision rules and on its wiring. The benchmark this replaced measured a batched `torch.matmul` in ONE thread, called the batch dimension "threads", report |
| test_bench_config_parity.py | 6 | The benchmark harness must render the stack config.yaml describes. This guards the defect class that has now invalidated measurements three times, each time in a different harness and each t |
| test_bench_perf_env.py | 3 | The comparison bench must apply the same ROOP_* env the app does. A bench that sets a different set of performance flags than `run.py` is measuring a different machine. That is not hypotheti |
| test_bench_source_revision.py | 9 | The benchmark harness must record, and check, the tree each arm ran on. On 2026-09-03 two feature commits landed in the working tree WHILE a counterbalanced A/B was rendering, both adding ~2 |
| test_benchmark_optimizer.py | 67 | Tests for roop.benchmark.optimizer. These assert PROPERTIES of the search, not thresholds: a test that pins a constant only records what the code currently says. The properties chosen are th |
| test_benchmark_ui_dashboard.py | 44 | Tests for the benchmark's presentation layer and its API surface. The dashboard is a persuasion surface, so most of these assert what it must REFUSE to say: an estimate presented as a measur |
| test_buffer_pool_acquire.py | 4 | Regression tests for PinnedBufferPool.acquire's non-blocking default. A buffer only returns to the pool through `release()`. `roop/nvdec_reader.py` is the pool's only caller and never releas |
| test_canonical_provider_decision.py | 4 | Tests for authoritative canonical provider selection pipeline. |
| test_capability_probes.py | 5 | Guards against capability probes that can never succeed. Two probes in this project asked for attributes that do not exist: ``torch._C._cuda_getDriverVersion`` and ``builder.platform_has_fas |
| test_convert_dynamic_shapes.py | 13 | tools/convert_dynamic_shapes.py: a model is "dynamic" only if a batch of N == N single runs. Synthetic graphs with the pitfalls found on the real models: * a Reshape that bakes batch 1 (fixa |
| test_cpu_topology.py | 4 | Unit tests for hybrid CPU topology detection and thread pinning. |
| test_cuda_affine_batch.py | 39 | CudaAffineBatch: device-side alignment, inversion, masks and compositing. What is pinned here, and why each number is what it is ------------------------------------------------------ * MATR |
| test_cuda_affine_roi.py | 6 | CudaAffineBatch.paste_faces_roi must equal paste_faces and touch nothing outside the face's box. Device-agnostic torch math (the class says so), so this runs on CPU tensors everywhere the fu |
| test_cudnn_algo.py | 15 | Guards for the per-model cuDNN conv-algorithm policy. The policy's failure modes are all SILENT, which is what these cover: * a suspect key that no processor actually passes never probes, so |
| test_cudnn_determinism.py | 2 | `roop.core` must not turn on `cudnn.benchmark`. It used to (`_configure_torch_cuda_acceleration`), and with it on: * a render whose ROI size changes every frame re-autotunes cuDNN on every n |
| test_default_client_capability.py | 3 | The default client must be able to START A JOB, not merely render. WHY THIS EXISTS. React UI 2.0 was promoted to the production default and passed every acceptance row that was run against i |
| test_diagnose_runtime.py | 5 | Tests for standalone diagnostic mode (--diagnose-runtime). |
| test_distributed_render.py | 13 | tests: plan_gop_boundaries_and_stream_copy, non_keyframe_trim_rejected, vfr_rejected |
| test_engine_telemetry.py | 12 | Engine compile / canary state on `/ws/telemetry` (Phase 6, part 3). The compile signal already existed on its own channel (`/ws/v1/pipeline/status`, which feeds CompileBanner). The telemetry |
| test_export_presets.py | 7 | crop_to_fill â€” the centered-crop math behind one-click export presets. No ffmpeg execution here, same convention as test_enhancer_align.py testing affine math without running a model: this i |
| test_frame_cache.py | 12 | Preview-frame decoding: the walk-forward seek and the decoded-frame cache. Both exist to make the timeline usable â€” a cv2 seek costs a flat 40-200 ms on long-GOP video whatever the distance, |
| test_frame_player_js.py | 2 | The React frame transport + player pipeline, run through node. react-ui/.render-check/player-check.mjs exercises the REAL client modules (transport/frameProtocol.js, components/player/frameP |
| test_frame_rate_and_early_close.py | 10 | Frame-rate selection, the writers' exact ``-r``, and the reader's early close. 1. ``detect_fps`` read ``r_frame_rate`` after 52acd20. On a VFR clip (4 s @30 + 4 s @15: r=30/1, avg=2700/119,  |
| test_frames_ws.py | 11 | /ws/frames: the binary frame transport, against the REAL app. What has to hold for the UI to trust it: * the header packs and unpacks to the documented 20-byte layout, so the client's DataVi |
| test_fresh_install_tensorrt.py | 6 | Verify that clean installs on new user devices receive TensorRT and precision mode. Validates that: 1. Fresh launches without an existing config.yaml seed from default_config.yaml with provi |
| test_full_benchmark_e2e.py | 34 | End-to-end verification of the full benchmark user journey. select workload -> run -> report -> DECLINE (settings preserved) -> load from history -> APPLY (roop.globals.CFG updated) WHAT IS  |
| test_gpu_event_timer.py | 14 | Non-blocking CUDA event timing (roop/gpu_event_timer.py). `torch.cuda.Event.elapsed_time` raises "device not ready" while the end event is still queued, and the old `StageProfiler.end()` cal |
| test_gpu_preflight.py | 8 | Tests for app/roop/gpu_preflight.py authoritative preflight logic. |
| test_gpu_recovery.py | 19 | A Stage 3 frame that dies of a device out-of-memory error is rerun alone, once (Phase 8, roop/gpu_recovery.py). The claims, each with a negative control where one is possible: * which errors |
| test_gpu_stage_locks.py | 16 | The GPU lock is split per stage. These are the invariants that keeps. WHY THIS FILE EXISTS. TensorRT's only rule is that one execution context must not be entered by two threads at once. A s |
| test_hardware_fallback_surfacing.py | 18 | A hardware path that gives way to a slower one must say so where the report and the HUD read it. The failure class (AGENTS.md, "Something reports success while not running"): a broad ``excep |
| test_hardware_portability.py | 25 | One config.yaml, two machines: an RTX 4070 12GB and an RTX 3060 6GB. Everything in this project's perf surface is derived from the GPU â€” pool sizes (session_pool's VRAM tiers), thread count  |
| test_hardware_signature_migration.py | 14 | A widened signature FORMAT is not a new GPU. `hardware_signature` gained fields over time. A config written before that carries the short form, so a byte comparison reports "different hardwa |
| test_hardware_streamer.py | 4 | tests: shared_memory_frame_ring_write_and_read, shared_memory_frame_ring_attach, shared_memory_frame_ring_close_without_unlink_detaches_finalizer |
| test_hardware_validation.py | 5 | tests: profile_key_ignores_transient_available_vram, dual_target_report_keeps_missing_target_pending, mismatched_runtime_identity_is_not_accepted |
| test_in_memory_frame_paths.py | 19 | The render and its helper caches must not dump frames to disk. Three places used to: the scrub-preview fallback (one JPEG per probed frame in a temp folder), the SAM2 pre-pass (every frame o |
| test_inference_engine.py | 23 | Stage 3: OptimizedInferenceSession, IO binding, FP16 conversion. Each test here pins something that went wrong while building it: * ORT REJECTS '1'/'0' for TRT boolean options and has_user_c |
| test_launcher_provider_override.py | 3 | Regression coverage for the launcher's explicit provider override. The public ``run.py --execution-provider`` switch is parsed once by run.py and again by ``roop.core``. Keep both halves cov |
| test_model_integrity.py | 17 | model_integrity: resumable, SHA256-verified model downloads and the startup check. |
| test_model_registry_precision.py | 6 | The two model registries must agree, and a declared precision must be the real one. The app has two places that describe a model: `app/model_manifest.json` (what the startup integrity check  |
| test_no_shadowed_imports.py | 4 | A late import inside a function makes the name local for the WHOLE function. `ProcessMgr.process_face` used the module-level `estimate_norm` at the top of the enhancer re-align, and 200 line |
| test_no_unmanaged_gpu_sessions.py | 3 | Every ONNX Runtime session built in the app goes through roop.strict_session, or is on this list with the reason it must not. A raw ``onnxruntime.InferenceSession(path, providers=[CUDA, CPU] |
| test_nvdec_reader.py | 4 | Regression tests for the safe NVDEC/host-frame boundary. The application intentionally hands ordinary mutable BGR NumPy arrays to its existing consumers. These tests make the bounded prefetc |
| test_nvsmi_timeout_backoff.py | 4 | nvidia-smi HUD probe: a timeout backs off, it does not disable the probe. Under a full render nvidia-smi can take longer than its 3 s timeout. That used to count toward the 3-strike permanen |
| test_oom_guards.py | 19 | Three guards against the memory failure that killed a 40934-frame render. Reported from a 16 GB / RTX 3060 (6 GB) machine: the run died at frame 5001, twice, having burned 33 minutes each ti |
| test_openvino_runner.py | 19 | roop/openvino_runner.py: device policy, ORT-surface parity on the REAL SCRFD / ArcFace / XSeg graphs, concurrent-caller isolation, error recovery and the INT8 fail-closed rule. The parity te |
| test_optimizer.py | 55 | End-to-end simulation of the optimizer across four hardware profiles. ``test_benchmark_optimizer.py`` unit-tests the pieces. This drives the WHOLE search -- noise floor, Phase A/B/C, bottlen |
| test_ort_thread_overrides.py | 12 | ORT thread overrides run EXACTLY as set -- they are advised, never clamped. `get_onnx_session_options` used to fold an explicit override through `min(upper, value)`, so `ROOP_ORT_INTRA_THREA |
| test_performance_optimizations.py | 14 | Comprehensive test suite for the three high-performance systems optimizations: 1. Garbage Collection (GC) Lifecycle Management 2. Non-Target & Empty Frame Early Fast-Path 3. Frame Buffer Pre |
| test_phase14_autotuner.py | 8 | tests: candidate_controls_reach_the_actual_child_environment, software_encoder_uses_software_preset_variable, result_metrics_carries_work_counts_for_comparability |
| test_pool_overrides.py | 12 | An explicit pool size must be honoured exactly. These knobs used to be CLAMPED. `_resolve` silently reduced any ROOP_TRT_POOL / ROOP_DETMASK_POOL / ROOP_DETECTOR_POOL above `auto * 2` to tha |
| test_precision_policy.py | 10 | Unit tests for the model-specific precision gate. These tests deliberately use provider descriptions and tiny temporary model files; they do not claim to be GPU benchmarks. The hardware gate |
| test_predictor_provider_assert.py | 26 | The provider assertion must fire on a silent EP drop and nowhere else. The defect it exists for is invisible by construction: onnxruntime returns a working session when an execution provider |
| test_preview_keeps_analyser_pool.py | 5 | A preview must not shrink the FaceAnalysis pool a render built. `session_pool.detmask_pooling_enabled()` returns False whenever `roop.globals.is_preview` is set, so `_ensure_face_analyser` u |
| test_preview_provider.py | 24 | InteractiveFramePreviewProvider: the bound on how long an interactive frame request can wait. The defect it exists for: `/api/target/preview` & co. decoded through one shared `cv2.VideoCaptu |
| test_provider_benchmark_api.py | 4 | The provider-matrix benchmark profile ('providers') through the API worker that runs it. `bench.run_benchmark` is stubbed (this file checks the transport, not the GPU): what must hold is tha |
| test_provider_discovery_is_safe.py | 4 | No production code may call `get_available_providers` unguarded. A broken ONNX Runtime install is not hypothetical here. When the package is absent but a directory of that name is importable |
| test_provider_fallback_is_loud.py | 8 | Two silent downgrades that together produce "I set 7 threads and it runs one". Neither is new behaviour. What was new is that both used to happen without a word, so the only observable was a |
| test_provider_initialization.py | 18 | Automated tests for provider initialization across the 16-arm test matrix. Validates BOTH: 1. Canonical provider decision (requested, admitted, active, degradation) 2. Actual constructed ONN |
| test_provider_matrix.py | 10 | `bench.measure_provider_matrix`: the per-stage CPU / primary / OpenVINO table and the four ARMS built from it. Real onnxruntime on a tiny generated ONNX model drives the CPU arm and `make_fe |
| test_provision_runtime.py | 3 | Tests for fail-closed runtime provisioning decisions. |
| test_regression_benchmark.py | 18 | roop.benchmark.regression: the pure gates and metrics, plus the CLI wiring. The render itself needs the GPU stack and is exercised by running `python run.py --benchmark --benchmark-mode regr |
| test_retired_frame_telemetry.py | 18 | Every fps in the report divides frames the OUTPUT WRITER accepted, never planned or merely-processed frames. Before: `processing_fps` and `effective_fps` divided the trim's planned `total_fr |
| test_runtime_hardening.py | 7 | Regression coverage for startup and diagnostic fallbacks seen in live logs. |
| test_runtime_optimizer.py | 40 | tests: mask_and_enhancer_smoothing_receive_stabilization_workers, no_smoothing_does_not_allocate_stabilization_workers, explicit_4070_twelve_thread_profile_reaches_stabilization |
| test_runtime_scheduler.py | 6 | Contract tests for Gate E's unified scheduler. These tests exercise scheduling and admission with synthetic callbacks. They do not present as RTX measurements; physical GPU validation remain |
| test_runtime_selection_invariant.py | 10 | The runtime layer never changes which face is selected. A provider failure is allowed to change the RUNTIME (TensorRT -> CUDA -> CPU, batch -> sequential, or a controlled render error). It i |
| test_runtime_state.py | 11 | Contract tests for the backend-owned structured runtime state. |
| test_runtime_state_canary_latest.py | 7 | PRECISION.canary_verdicts reports the LATEST verdict per model, and `nan_inf_free` is established, not asserted. The canary store keeps one verdict per engine hash ever verified (xseg 21, re |
| test_scheduler_queue_vocabulary.py | 13 | The bottleneck classifier must read the queue names its callers actually send. There are two callers and they use two vocabularies: * `UnifiedRuntimeScheduler.run_stream` reports its own que |
| test_source_portfolio.py | 16 | Contract for roop/source_portfolio.py and its render hooks in ProcessMgr: the SOURCE angle portfolio, the target frame LUT, routing, far-eye damping. |
| test_stage_report.py | 31 | Phase 9: the per-job report (stage wall clocks, seek counts, passthrough / GPU frames, peak VRAM, Stage 4, speedup estimate) and the three places it must reach: `/ws/telemetry`, the history  |
| test_startup_optional_tensorrt.py | 4 | Exercise startup code paths when TensorRT is not installed or optional. Validates that missing TensorRT packages do not emit [Fallback] or ModuleNotFoundError diagnostic messages that Pinoki |
| test_strict_session.py | 23 | roop.strict_session: a GPU session is on the GPU or it does not exist. The factory is exercised against a fake ``onnxruntime`` injected into ``sys.modules`` so the whole contract -- no CPU r |
| test_strict_trt_and_sweep.py | 6 | The strict TensorRT assertion, and verify_trt_fps.py's batch sweep. assert_strict_tensorrt_session used to require get_providers() == ["TensorrtExecutionProvider"]. ONNX Runtime appends CPUE |
| test_sub_7gb_tensorrt_policy.py | 6 | Tests for sub-7GB TensorRT admission policy across hardware tiers (6GB, 8GB, 12GB, CPU). |
| test_trt_bound_session.py | 11 | Opt-in TensorRT bound/graph sessions and the build-heuristics switch: both must be INERT unless asked. The A/B levers added 2026-10-06/09 are env flags (ROOP_TRT_BOUND, ROOP_TRT_BUILD_HEURIS |
| test_trt_build_guard.py | 8 | roop.trt_build_guard: torch's idle VRAM is released immediately before and after a TensorRT build or canary run. Each test drives the real call site (the strict session factory, the canary e |
| test_trt_build_override.py | 17 | Per-model TensorRT build overrides (roop/precision_policy.apply_build_override). core.py sets one heuristics flag and one builder level for EVERY engine. The override layer lets one model di |
| test_trt_context_manager.py | 14 | Phase 3 contracts for TensorRT context budgeting and safe pool lifecycle. |
| test_trt_model_precision.py | 22 | Per-model TensorRT precision (roop/precision_policy.PRECISION_OVERRIDES / ROOP_TRT_MODEL_PRECISION). `trt_precision` is one global setting. The measured table (docs/perf/trt_precision_fideli |
| test_trt_probe.py | 2 | Tests for the deterministic, non-Git-LFS TensorRT probe graph. |
| test_ui_zoom_pan.py | 19 | Guards on the shared zoom/pan maths behind the preview stage and the grids. Every bug this file pins was silent â€” the zoom still "worked", it just did the wrong thing, and nothing but a care |
| test_verify_ort.py | 21 | Tests for app/verify_ort.py verification logic. |
| test_vram_governor.py | 20 | The per-render VRAM governor (roop/vram_governor.py). What must hold: * the planner steps the cheap thing down first (the swap batch) and the look-changing thing (GPEN resolution) only once  |
| test_vram_pool_budget.py | 41 | Keeping peak VRAM under ~90% of the card without changing a model or a precision. Three mechanisms, pinned here: 1. After a replayed render's pre-pass the FaceAnalysis pool and every hybrid  |
| test_windows_runtime_compat.py | 6 | Tests for the Windows NVIDIA binary/runtime compatibility verifier. |
| test_worker_pool.py | 10 | roop.worker_pool: each spawned worker owns an independent, verified recognition engine. These spawn REAL processes (that is the point: pickling, spawn, per-process sessions) on the `default` |
| test_zero_copy_buffer_pool.py | 23 | ZeroCopyBufferPool: the ring, its events, its locks, its VRAM accounting, and ORT binding. What is pinned, and the failure each test stands for ---------------------------------------------- |

### H. Queue / jobs / checkpoint / projects / lifecycle (24 files)

| file | #tests | purpose / what it certifies |
|---|---:|---|
| test_batch_isolation.py | 10 | Isolated video batch: one spawn child per video, and a queue that survives a bad video. Found 2026-10-03 by running `pinokio_batch_runner.py` for real over a 50-clip queue: it died on the fi |
| test_batch_matrix_queue.py | 27 | A Batch Matrix job list, pushed through the real queue. docs/development/BATCH_MATRIX_DATA_FLOW.md traces how the four strategies become N single-target jobs in POST /api/queue/add_batch and |
| test_checkpoint_resilience.py | 8 | A checkpoint write must survive a transient rename, and never wedge a render. WHY THIS EXISTS. On the physical RTX 3060 host a project's PROCESSING state update raised `PermissionError: [Win |
| test_checkpoint_segment_commit.py | 4 | `_checkpoint_segment` must actually reach `update_checkpoint`. It is wrapped in `except Exception` -- deliberately, because a failed checkpoint must not kill a render -- and it called `_proj |
| test_cleanup_script.py | 12 | scripts/cleanup.py (the sidebar's Clean button): what it may and may not delete. Every case builds its own tree in a temp dir; nothing here touches a real install. |
| test_ffmpeg_writer_lifecycle.py | 6 | The encoder must not turn a late ffmpeg exit into a truncated success. |
| test_jobs_active.py | 6 | `GET /api/jobs/active` â€” the endpoint a reconnecting client asks first. Pinokio reloads this webview on every tab switch and a render routinely runs for forty minutes, so a fresh client over |
| test_pause_resume.py | 2 | Stage 8A contract tests for the real processing pause boundary. |
| test_progress_chunks.py | 26 | Terminal progress reporting: fixed wall-clock updates, not chunk commits. A progress bar is a thing you rewrite in place, which needs a terminal to move the cursor on AND nothing else printi |
| test_project_checkpoint.py | 4 | Stage 8B persistence and recoverability tests. |
| test_project_io_nle.py | 7 | tests: round_trip_preserves_portable_relative_reference, journal_recovers_newer_snapshot, autosave_controller_writes_on_demand |
| test_queue.py | 33 | The batch queue must not lose work. That is the whole reason it moved out of the browser, so these test the ways it could still lose a job rather than the happy path: * an app restart mid-jo |
| test_queue_segments.py | 17 | The queue as a split-render runner: segments are named, retained and joined; state is on disk. The runner loop is real; only `_run_swap` is faked, by a renderer that does what core does for  |
| test_resume_progress_base.py | 7 | What a resumed project reports as its progress. `ApiProgress` maps a run's own 0..1 onto `[base, 1]`, where `base` is the share of the frame range the run will NOT redo. `_start_existing_pro |
| test_run_eta.py | 18 | The "time left" the web UI shows must be the one the terminal is showing. The UI used to derive it as elapsed * (1 - fraction) / fraction, which silently assumes every second so far was spen |
| test_segment_concat_disk_full.py | 4 | A merge the output drive cannot hold must fail loudly and keep the parts. 2026-09-27: G: filled during the final stream-copy concat. `_concat()` returned False, `close()` returned normally,  |
| test_segment_custody.py | 15 | Segment custody: the chunk matrix, re-rendering ONE slice, and a join that checks what it made. Built on the same harness as test_queue_segments.py: the runner loop is real, only `_run_swap` |
| test_segment_output.py | 17 | Split-render segments: unique names, retained files, a verified stream-copy join. The defect (queue "Split into N segments"): every slice named its video from the output template alone, so s |
| test_segment_parts.py | 13 | The output-parts registry behind the console's part tabs. A resumable render writes numbered part files into the output folder and merges them at the end. Those parts are the run's chapters  |
| test_startup_state_machine.py | 15 | Comprehensive tests for the transactional startup state machine. |
| test_storage_manager.py | 5 | Stage 10 storage inventory and deletion safety tests. |
| test_ui_job_recovery.py | 17 | Persistent job recovery: what may be remembered, and in what order. The failure this design exists to avoid has two symmetric halves, and getting the ORDER wrong picks one of them: * restore |
| test_upright_recovery.py | 11 | An inverted face must not be mistaken for an upright one. The measured failure these guard (tests/frontal_roll_video.py builds the clip): between roll ~140 and ~220 the detector does not los |
| test_watchdog_and_autopsy.py | 26 | Stage 10: Tests for Runtime Anomaly Watchdog & Crash Autopsy Engine. |

### I. Video IO / FFmpeg / encode / audio / output / preview (25 files)

| file | #tests | purpose / what it certifies |
|---|---:|---|
| test_audit_source_and_preview_fixes.py | 14 | Regression tests for three bugs found in the extended post-fix audit. BUG 1 â€“ Deferred preview flag lost during scrub / playback The useEffect that fires preview after a run finished cleared |
| test_backend_outlives_gradio.py | 9 | A Gradio failure must not take the FastAPI backend down with it. WHAT HAPPENED. A second launcher instance collided on the Gradio port -- both React launchers derive it as ROOP_API_PORT + 2  |
| test_finalize_output.py | 12 | Stage 4 finalization: mux, verify, and delete the intermediate ONLY after the verification passed. Real ffmpeg, real files. The defect these guard is not an argv: `core.batch_process_regular |
| test_finalize_web_video_nvenc.py | 7 | finalize_web_video encodes on NVENC first and falls back to libx264. The subprocess is faked, so this pins the command lines and the retry order and does not depend on the host having NVENC. |
| test_live_acceptance_suite.py |  | (no header) |
| test_live_mode.py | 5 | Focused tests for the dedicated bounded live path. |
| test_live_perf_settings.py | 11 | The per-render performance settings reach their flags without a restart, and each flag is actually READ by the code it names (AGENTS.md: a control bound to a value nothing consumes looks com |
| test_live_preview.py | 14 | The processing box's live frame â€” throttled, downscaled, encoded once. This exists on the hot swap path: `_publish_live` is called for every frame the pipeline finishes. The previous version |
| test_output_range_serving.py | 17 | HTTP 206 serving of output media, and the compare view's source endpoint. The output player seeks by Range request, and three of the shapes a media element sends were answered wrongly before |
| test_output_route_layout.py | 2 | Keep API output routes behind one explicit router boundary. |
| test_phase12_settings_streamlining.py | 15 | Phase 12: settings that were removed stay removed, and what the pipeline now decides for itself is decided. * Sparse pre-scan, disk-frame extraction, Keep frames, Wait before creating video, |
| test_pipeline_3stage_sync_and_fps.py | 15 | End-to-end proof that the 3-stage pipeline is frame-synchronized, reports honest numbers and leaves no artifacts. Real renders through the production path (`core.batch_process_with_options`, |
| test_preview_admission.py | 7 | The still/preview path must actually run its frame operation. WHY THIS EXISTS. `pause_aware` gates every decorated frame operation on `roop.globals.processing`, and `PauseController.begin` R |
| test_preview_reinit_cost.py | 7 | A repeat preview must not re-derive what cannot have changed. `live_swap` calls `ProcessMgr.initialize` on EVERY `/api/preview`, so anything that path re-does per call is paid on every scrub |
| test_preview_render_lockout.py | 4 | A swap preview must not run beside an active render. `/api/preview` sets the process-global `roop_globals.is_preview` (which narrows `session_pool.pool_size()` to 1 for EVERY thread, the ren |
| test_preview_render_selection_contract.py | 2 | Preview/render parity tests for the canonical processing request. |
| test_qsv_encode.py | 18 | roop/qsv_encode.py: Intel QuickSync as a whole-render video encoder (NOT a per-segment split). No QuickSync frame is encoded here: the bundled ffmpeg has no ``*_qsv`` encoder and no Intel GP |
| test_range_seek.py | 26 | Frame-exact positioning of the Stage 2 pre-pass capture (roop/range_seek.py). The fixtures carry their own frame number in the picture (three base-16 digits in flat blocks, readable after lo |
| test_reader_shutdown.py | 7 | The frame reader must always be able to exit. `read_frames_thread` ends by handing each worker a `None` sentinel. That was a bare `put(None)` â€” no timeout â€” while the data put directly above |
| test_scrub_preview.py | 12 | Scrub-time previews: `scrubbing=1` proxy frames and `frame_seq_id` supersession on /api/target/preview. Two layers are exercised, each at the real call site: * `capturer.get_video_frame(canc |
| test_target_media_isolation.py | 9 | Stage 11 regression tests for target-media context isolation. These tests use fake ProcessEntry-like objects and the real target selection, removal, preview, and queue-request boundaries. No |
| test_ui_preview_pipeline.py | 32 | Guards for the canvas preview pipeline, the scrub throttle, and object-URL lifetime. Three properties, and none of them is visible to the build, to oxlint, or to any behavioural test in this |
| test_video_io_hw.py | 51 | roop/video_io_hw.py: zero-copy NVDEC/NVENC, true-IDR keyframe decode, the QSV rung, the fallbacks, the telemetry. Two classes. ``PureLogic`` needs no GPU and runs in the light CI profile (it |
| test_video_stream_drain.py | 5 | Tests for StderrDrainer and subprocess pipe buffer deadlock prevention. When FFmpeg processes videos (especially with hardware decode/encode and timestamps/seeking), it frequently writes war |
| test_video_stream_release.py | 3 | A pipe released from another thread is END OF STREAM, not a decode failure. `ProcessMgr._run_stab_parallel`'s cleanup calls `cap.release()` on purpose, to interrupt a blocking pipe read befo |

### J. API / routes / server / terminal / telemetry / diagnostics / safety (27 files)

| file | #tests | purpose / what it certifies |
|---|---:|---|
| test_api_access.py | 22 | Network exposure of the backend: loopback by default, token in share mode, foreign Origins refused. Before api_access existed the API bound to 127.0.0.1 but answered ANY browser page: CORS w |
| test_api_routes.py | 5 | Route-registration guards for `api.app`. Written after `GET /api/settings` spent a release answering 422. The helper `_public_settings(cfg)` was extracted and inserted directly beneath the e |
| test_check_api_mock_parity.py | 21 | Stage 16: `scripts/verification/check_api_mock_parity.py` holds the repo to its contract, and can fail. The script compares route signatures and the /ws/telemetry frame keys between `app/api |
| test_file_endpoint_roots.py | 7 | What /api/file is allowed to hand out. The endpoint takes an absolute path straight off the query string and streams it back, so its root list is the entire access-control boundary. Anything |
| test_hud_fixture_current.py | 1 | The React render check feeds `react-ui/.render-check/pipeline-hud.fixture.json` to the HUD. That file is only evidence about the backend while its SHAPE equals what the real telemetry writer |
| test_interactive_console_and_journal.py | 12 | Stage 13: Interactive Command Console and Journaled Crash Recovery Tests. Tests: 1. Atomic write-ahead journal persistence (journal.json.tmp -> journal.json) 2. Frame-accurate resume & ffpro |
| test_monitor_telemetry.py | 13 | Phase 15 monitor: absence of a measurement must not read as a measurement. Every defect covered here was measured on BOTH validation GPUs: the summary reported `end_to_end_fps` 0.0, `cpu_uti |
| test_nonfrontal_router.py | 30 | The non-frontal mask-routing score and its temporal latch. Two guarantees are load-bearing here: 1. `nonfrontal_score(...) > 1.0` is EXACTLY the verdict the previous OR-of-four-booleans gave |
| test_offline_operation.py | 8 | Offline/connected behavior for network-backed optional resources. |
| test_phase15_monitor.py | 5 | tests: disabled_monitor_is_a_noop, rolling_summary_exposes_stage_and_resource_metrics, summary_exposes_memory_pressure_dimensions |
| test_pipeline_monitor.py | 28 | The live pipeline monitor: exact FPS, honest starvation verdicts, never fatal. |
| test_pipeline_status.py | 21 | The pipeline status channel and the governor's absolute VRAM floor. A render's first minutes (TensorRT engine build, canary) have no frame counter, so the React client's 15 s deadline fired  |
| test_safe_paths.py | 23 | Every endpoint that takes a path, a filename or an upload stays inside the app's own folders. Reviewed 2026-09-22: /api/file and /outputs/ used abspath, so a symlink inside the output folder |
| test_scanner.py | 9 | Contract for roop/scanner.py (TemporalPrePassScanner) on a synthetic clip. The detector is a stand-in that reads two coloured squares out of each frame, so these tests exercise the real deco |
| test_stage_hud_telemetry.py | 17 | Per-stage HUD telemetry: Stage 1 keyframes/s + scene cuts, Stage 2 flow-tracked vs neural-detected frames, Stage 3 ms per operation with the backend each ran on, and the OpenVINO `aux_offloa |
| test_telemetry_rate_window.py | 4 | RateWindow: the current frame rate the HUD's ms/frame is derived from. |
| test_telemetry_ws.py | 12 | Telemetry transport tests, against the REAL app. These cover the properties the transport has to hold for a render to be safe: * the socket serves, greets with current state, and answers a l |
| test_terminal_dashboard.py | 67 | Stage 9: the terminal processing dashboard (`roop/terminal_dashboard.py`). Every behaviour is driven through the writer production uses and read back through the consumer it exists for: * th |
| test_ui_api_surface.py | 4 | The React UI's API surface: what it calls, what exists, what the README says. `react-ui/README.md` carried an API table of about a dozen rows while the UI calls 126 distinct endpoints, and i |
| test_ui_api_timeouts.py | 4 | Every long-running API call in the React UI opts out of the 15 s deadline, explicitly. react-ui/src/api.js gives each getJSON / postJSON / postFile(s) call 15 seconds unless it passes `timeo |
| test_ui_settings_catalog.py | 3 | The palette's settings catalogue and the Settings panel must list the same keys. App.jsx offers every setting by name in the command palette, and it cannot read them out of Settings.jsx: tha |
| test_update_check_route.py | 8 | `GET /api/update/check` is read-only, and must stay that way. `update_manager.py` carries two very different halves: `check()` collects compatibility evidence and changes nothing, while `app |
| test_update_health.py | 6 | tests: smoke_feeds_follow_actual_session_shapes, dynamic_dimensions_are_bounded_to_a_single_smoke_batch, dependency_check_fails_closed_for_missing_distribution |
| test_update_manager.py | 22 | tests: explicit_compatible_manifest_is_safe, unknown_provider_is_not_safe, critical_runtime_change_requires_review |
| test_update_manifest_head.py | 5 | HEAD must accept itself as an update candidate. app/update_manager.py admits a fetched commit only when that commit's update_manifest.json passes evaluate_manifest(). Between the manifest's  |
| test_verify_three_stage_logic.py | 14 | The logic of `verify_three_stage_pipeline.py` that needs no GPU: scene truth, the five gates on synthetic run records, and the sanitiser step. A gate that cannot fail proves nothing, so each |
| test_win_asyncio_compat.py | 3 | An aborted accept must not take the API listener down (Windows, 3.10). Reproduces the 2026-09-21 outage in miniature: a proactor-loop server whose first pending accept completes with WinErro |

### K. UI contract / React / launcher / settings / install / repo hygiene (38 files)

| file | #tests | purpose / what it certifies |
|---|---:|---|
| test_backend_serves_ui.py | 12 | The backend serves the built UI without shadowing the API. `app/api.py` serves `react-ui/dist` so the app needs no Vite process at runtime (see test_react_launcher_serves_build for why that  |
| test_calib_version.py | 6 | Learned runtime estimates must not outlive the pipeline they measured. `runtime_calib` stores wall-clock ms/frame per settings signature and uses it for the pre-run "estimated time". The sig |
| test_capture_seed.py | 20 | capture_seed â€” choosing WHICH frame the target people get captured from. The bank the user captures IS the identity model: every later gate (the manual capture gates in /api/target/add_angle |
| test_env_flags.py | 19 | Behaviour of the shared ROOP_* environment reader. These pin the semantics the old per-module copies disagreed about: what a malformed value does, whether a typo can switch a feature on, and |
| test_fixture_paths.py | 6 | Benchmark harnesses must resolve fixtures, not name one machine's drive. TWO defects are pinned here, and the second was introduced while fixing the first. 1. **Hardcoded `<MEDIA_DIR>/...`.* |
| test_golden_clean_install.py | 3 | Golden Clean Install Verification Suite. Target: Windows 10/11 + NVIDIA. Starts from a completely pristine environment: - no app/env - no config.yaml - no TensorRT cache - no frontend node_m |
| test_install_state.py | 7 | Transactional installation-state tests. |
| test_launcher_activation.py | 14 | The launcher must open the React client, and start.js must resolve onto it. WHY THIS EXISTS. React UI 1.0 is the sole production client: `start.js` re-exports `start_react.js` and the Pinoki |
| test_launcher_conditions.py | 3 | Launcher `when` expressions must never throw. Pinokio evaluates a step's `when` as a JavaScript expression. If the expression raises, the step is skipped -- silently. That is not hypothetica |
| test_launcher_e2e_harness.py | 10 | End-to-End Pinokio Launcher Simulation Test Harness. Simulates Pinokio launcher execution across 10 distinct system and environment states: 1. Fresh directory (fresh checkout: no app/env, no |
| test_no_undefined_names.py | 2 | Every module must define, import or receive every name it uses. This exists because the decomposition broke three call sites and none of the other checks noticed. Splitting a module verifies |
| test_phase11_analysis_release.py | 4 | Phase 11 item 3: the Stage 1/2 analysis models are ALREADY unloaded when Stage 2 seals its replay. The brief asked for the detector and the recognition session to be unloaded from VRAM after |
| test_phase_harness_contracts.py | 30 | Contracts every Phase harness must keep, so a phase cannot pass by doing nothing. Each test here is anchored to a defect that was live in this tree and that every other check reported clean  |
| test_precommit_scratch_guard.py | 12 | Repo hygiene: scratch output at the repo root cannot be committed again. 86ace34 ("commit remaining working-tree files") committed scratch_capture.log, scratch_eval.log, scratch_probe.log an |
| test_process_mgr_layout.py | 3 | Architecture guard for ProcessMgr's batch orchestration boundary. |
| test_react_launcher_serves_build.py | 13 | The backend serves the built React client, and Node is build-time only. This guards the cross-device failure the previous topology caused. The launcher used to run `vite preview` as a SECOND |
| test_settings_tabs.py | 7 | The Settings tabs, the runtime report and the workspace store agree with the backend. Three lists in the React UI are copies of something the backend owns, and each can drift without anythin |
| test_settings_wiring.py | 15 | Settings wiring â€” that a control reaching the UI actually reaches the render. The recurring failure here is a HALF-WIRED setting: a toggle that appears in the UI, saves fine, and silently ch |
| test_standalone_install.py | 5 | Roop Ultimate must be physically self-contained. Until 2026-08-23 this was not true, and nothing said so. `app/env`, `app/models` and `app/facesets` were NTFS junctions into a different work |
| test_startup_optional_config.py | 7 | Exercise the actual early startup function without importing GPU/server code. All config reads happen in a temporary working directory. No live settings, models, dependencies, frontend build |
| test_synthetic_label.py | 14 | Output labelling and the intended-use gate (NOTICE.md's asks, made real). Audited 2026-09-22: app/roop had no content or consent safeguard of any kind. These pin the three that now exist --  |
| test_target_selection_contract.py | 18 | Regression tests for the explicit target-person selection contract. |
| test_ui_accessibility.py | 4 | Two accessibility rules that fail silently, so they are checked mechanically. Neither shows up in a diff, in oxlint, or in the build. Both were live: * `Toggle` rendered its real <input type |
| test_ui_compare_grids.py | 3 | The four comparison grids must not share a localStorage key. Enhancers, mask engines, swapper models and AI upscalers each remember which variants you last put side by side. They now share o |
| test_ui_hook_arguments.py | 5 | A hook's ARGUMENTS must not name something declared further down. `test_ui_hook_order` guards the other direction â€” a hook's RESULT read before the hook runs. This is the mirror image, and i |
| test_ui_hook_order.py | 3 | A hook's results must be declared before anything reads them. This guards the specific way extracting a hook from a large component goes wrong, which happened while doing exactly that to Fac |
| test_ui_preset_recipes.py | 7 | Every preset must set REAL settings to values the backend accepts. A preset is a block of `key: value` pairs applied straight into the live params. Nothing validates them at runtime: an inve |
| test_ui_primitive_props.py | 3 | Every `size`/`variant` a call site asks a primitive for must actually exist. The shared primitives pick their classes out of a lookup: className={`â€¦ ${variants[variant]} ${sizes[size]} â€¦`} w |
| test_ui_processing_tab.py | 7 | The run lives on its own tab, and every half of that has to stay wired. A running job used to take the Face Swap tab over: the settings sidebar, the asset rail, the timeline, the preview con |
| test_ui_render_lite_modes.py | 7 | Lite UI must be reachable, and must survive a tab switch. `data-render-lite` strips the 59 backdrop-filter panels and the three keyframe animations that repaint rather than composite. index. |
| test_ui_settings_defaults.py | 2 | Every setting the Settings panel binds must be a real key in settings.py. The panel's controls are wired through `bind('key')` / `bindToggle('key')`, which hands a control four things at onc |
| test_ui_slider_commit.py | 6 | A slider drag must not re-render the whole app sixty times a second. `<Slider>` is used at 53 call sites and its `onChange` is, at nearly all of them, `set(key, value)` -- which writes App's |
| test_ui_type_scale.py | 4 | The type scale is named, and nothing may quietly step outside it. This UI is dense, and most of its text sits BELOW Tailwind's `text-xs` (12px) where Tailwind offers no steps. With nothing f |
| test_ui_unreachable_modules.py | 2 | No module under react-ui/src may be unreachable from the app's entry point. Six component folders (studio, preview, timeline, facebank, queue, telemetry: 6,771 lines in 23 modules) sat in sr |
| test_ui_upload_and_alerts.py | 11 | Uploads must report progress, and the finish alert must tell you the truth. Three regressions this pins, none of which a build or a lint can see: * `fetch()` CANNOT report request-body progr |
| test_verify_away_tier_and_lock_floor.py | 10 | Two ways a frontal face got painted onto a head that shows none. Measured on D:\\out.mp4 (2026-10-14, harjot + ashna, two women), both while the OTHER woman in the same frames swapped cleanl |
| test_verify_harness_cli.py | 6 | The verify harness's entry point must actually start. verify_roop_keep.py shipped once with --base-dir defaulting to None and no fallback applied, so every invocation died on the first line  |
| test_verify_tol.py | 8 | Per-model tolerance for the swap outcome guard. face_util.swap_moved_the_face throws away a swap that put the face somewhere the plate's face was not. Its threshold used to be one shared con |

### L1. A/B harness scripts (ab_*), not pytest tests (30 files)

| file | #tests | purpose / what it certifies |
|---|---:|---|
| ab_cpu_pcore_ort.py |  | Counterbalanced A/B: P-core pinning + ORT thread widening, end to end. Arm A is the shipped default (`cpu_distribution=auto`, ORT serial per session). Arm B is the requested i9-14900K policy |
| ab_enhancer_gate_stage14.py |  | End-to-end A/B of the small-face restorer gate (`enhance_min_face_px`): off vs 128 px, counterbalanced, with the output compared. env/Scripts/python.exe tests/ab_enhancer_gate_stage14.py [-- |
| ab_enhancers_render.py |  | Counterbalanced end-to-end render A/B across ENHANCERS: fps and swap rate per arm. env/Scripts/python.exe tests/ab_enhancers_render.py --source <faceset> env/Scripts/python.exe tests/ab_enha |
| ab_env_e2e.py |  | End-to-end A/B of one environment switch on REAL renders, counterbalanced, with a null control and the swap rate as the guard. app\\env\\Scripts\\python.exe app\\tests\\ab_env_e2e.py --clip  |
| ab_face_count.py |  | Does swapping TWO faces cost more than twice swapping ONE? The report: on an RTX 3060 6GB the render collapses to 1-2 fps *only when two people are being swapped*; a single-face clip on the  |
| ab_gpu_engine.py |  | retinaface_r50_gpu vs the baseline detector: is any face lost, and what does it cost? env/Scripts/python.exe tests/ab_gpu_engine.py --phase recall env/Scripts/python.exe tests/ab_gpu_engine. |
| ab_gpu_renders.py |  | Whole-render acceptance for detector_engine 'retinaface_r50_gpu' against the baseline engine. env/Scripts/python.exe tests/ab_gpu_renders.py Per clip, real renders through `baseline_snapshot |
| ab_phase14_e2e.py |  | Phase 14 end to end: the pre-pass filter and the detection interval on REAL renders, with the swap rate as the guard. `probe_phase14_prepass.py` measures the pre-pass alone. This renders the |
| ab_pool_grid.py |  | Pool x swap-batch grid: which smaller configs free VRAM at no fps cost? env/Scripts/python.exe tests/ab_pool_grid.py plan [--detector-values 1,2] env/Scripts/python.exe tests/ab_pool_grid.py |
| ab_pose_routing.py |  | A/B: does pose-adaptive source routing change the swapped identity, by yaw band? Arms OFF / ON / OFF / ON on one clip with one multi-angle source faceset, real renders through the regression |
| ab_pyramid.py |  | Old vs new MultiScaleFaceDetector.detect: same boxes, fewer detector inferences? env/Scripts/python.exe tests/ab_pyramid.py --base HEAD --phase equiv env/Scripts/python.exe tests/ab_pyramid. |
| ab_pyramid_renders.py |  | Whole-render check of the pyramid change on retinaface_r50: no hang, same output. env/Scripts/python.exe tests/ab_pyramid_renders.py --base c4bfff9 The function-level A/B (ab_pyramid.py) can |
| ab_r50_static_profile.py |  | retinaface_r50 TensorRT profile: band (320-1280, opt 512, batch 1-8) vs static 1x3x640x640. # one arm per process (the provider list, the engine and the pool are rebuilt per arm): ROOP_TRT_S |
| ab_rescue_aux.py |  | Old vs new rotated rescues: is every face identical, and how many aux calls went away? env/Scripts/python.exe tests/ab_rescue_aux.py --base f9aaf96 `_rescue_rotated` and `_detect_faces`'s pa |
| ab_rescue_renders.py |  | Whole-render A/B for the rotated-rescue aux change: same output, fewer aux calls, faster pre-pass? env/Scripts/python.exe tests/ab_rescue_renders.py --base f9aaf96 OLD = `app/roop/face_util. |
| ab_shape_profile.py |  | A/B the TensorRT shape profile end to end, counterbalanced. WHAT IS BEING MEASURED. `roop/trt_shape_profile.py` gives TensorRT an explicit min/opt/max optimization profile for the models who |
| ab_small_card_pools.py |  | What the small-card pool guard costs, with GPU utilisation sampled. `session_pool._auto_pool_defaults` returns 0/0 below 7GB, so on an RTX 3060 6GB BOTH pools are off -- and `_gpu_guard(pool |
| ab_stab_blocks_vs_sequential.py |  | Does a LARGER stabilization block land closer to a SEQUENTIAL render? (ground truth for block sizing) env/Scripts/python.exe tests/ab_stab_blocks_vs_sequential.py --clip d4 --window 0 240 -- |
| ab_stab_chunk_mb.py |  | A/B the stabilized render's chunk size: ROOP_STAB_CHUNK_MB. env/Scripts/python.exe tests/ab_stab_chunk_mb.py \ --clip "<MEDIA_DIR>/inverted/Cervical....mp4" --source person_a WHAT IS BEING T |
| ab_stab_dedup.py |  | ABBA: parallel-stabilization warm-up dedup (ROOP_STAB_DEDUP) off vs on, end to end, on the live config. env/Scripts/python.exe tests/ab_stab_dedup.py --clips d4,s7 [--window 0 600] [--date 2 |
| ab_stab_overlap.py |  | A/B the stabilization pool's chunk overlap (and anything else) end to end, counterbalanced. `ab_temporal_detection.py` varies `roop.globals` keys. The overlap switch is an environment variab |
| ab_stab_owner.py |  | Counterbalanced A/B: one inference owner (streaming) vs parallel blocks. WHY THIS EXISTS. A terminal profile of a stabilized render reads like a threading problem -- hundreds of OS threads,  |
| ab_stabilization_combo.py |  | Counterbalanced A/B of the three stabilization switches, end to end. WHY. `stabilize_face` / `stabilize_mask` / `stabilize_enhancer` do not compose into three independent costs -- they selec |
| ab_swap_batch_render.py |  | ABBA end-to-end render: cross-frame swap batching ON (shipped default) vs OFF, for a chosen swap model. env/Scripts/python.exe tests/ab_swap_batch_render.py --swap-model realswap --clips d4, |
| ab_temporal_detection.py |  | A/B `temporal_detection` end to end, with the swap rate as a guard. WHY IT MIGHT HELP. In the swap loop the detect stage runs inside `_gpu_guard`, competing with swap, mask and enhance for t |
| ab_trt_bound_render.py |  | ABBA end-to-end render: shipped call path vs `ROOP_TRT_BOUND` (opt-in bound / CUDA-graph XSeg sessions). env/Scripts/python.exe tests/ab_trt_bound_render.py --clips d4,s7 --b "xseg=graph" [- |
| ab_trt_bound_sessions.py |  | Static-shape TensorRT sessions: production call path vs persistent device I/O + user stream vs + CUDA graph. env/Scripts/python.exe tests/ab_trt_bound_sessions.py [--calls 300] [--stability  |
| ab_trt_build_heuristics.py |  | ROOP_TRT_BUILD_HEURISTICS: engines built with heuristics ON (shipped, mixed) vs OFF. env/Scripts/python.exe tests/ab_trt_build_heuristics.py [--calls 300] [--faces 40] Prints the TensorRT /  |
| ab_trt_mixed_fidelity.py |  | XSeg and RestoreFormer++ on the production TensorRT session vs FP32, on real aligned crops. env/Scripts/python.exe tests/ab_trt_mixed_fidelity.py [--faces 120] The precision policy ships bot |
| ab_ui_poll_contention.py |  | Does UI traffic cost the render anything? idle vs realistic-UI vs 10x-stress, end to end. env/Scripts/python.exe tests/ab_ui_poll_contention.py --clips d4,s7 Question (RECODE_STATUS 2026-10- |

### L2. Probe scripts (probe_*), not pytest tests (58 files)

| file | #tests | purpose / what it certifies |
|---|---:|---|
| probe_align_e2e_stage17.py |  | Stage 17: does the REAL pre-pass keep the swap crop steady behind an occluder? Builds two clips from one source window -- a reference with nothing pasted, and a copy with a skin-coloured occ |
| probe_alignment_chain_stage17.py |  | Stage 17: alignment through a MULTI-FRAME, MOVING occluder, and without an oracle. `probe_alignment_stage17.py` occludes one frame and gives flow the previous frame clean plus the exact occl |
| probe_alignment_stage17.py |  | Stage 17: how far does a partly covered face pull the swap crop, and does visibility weighting put it back? The ground truth is free: take a real frame, run the real detector on it (clean),  |
| probe_assoc_stage16.py |  | Stage 16: replay the pre-pass association on recorded detections. `probe_bystander_stage16.py collect` records, per detection, the box, keypoints, pose, crop contamination and the unit w600k |
| probe_bystander_stage16.py |  | Stage 16 census: can a DIFFERENTIAL margin tell the target from a bystander when the two share a recognition crop? The brief's gate binds a face to the target only if S_target(i) >= tau_lock |
| probe_bystander_veto_stage16.py |  | Stage 16: does the bystander veto ever fire on real footage, and what does it cost? `test_bystander_veto.py` proves the rule on synthetic embeddings. This asks the only question that matters |
| probe_changed_area_stage18.py |  | Stage 18: does the swapped region of the OUTPUT hold still when something crosses it? For each render, per frame, inside the target's plate face hull (the hull is dilated so a swap that spil |
| probe_color_ema.py |  | Does a per-track EMA of the LAB colour-transfer transform remove lighting flicker, and what does it cost in lag? app\\env\\Scripts\\python.exe app\\tests\\probe_color_ema.py --frames 70 app\ |
| probe_edge_roll_recall.py |  | Does the detector keep a face that the frame edge cuts, or that is rolled? Measured on real frames. env/Scripts/python.exe tests/probe_edge_roll_recall.py --section edges env/Scripts/python. |
| probe_enhancer_face_sizes_stage14.py |  | How many of the faces the restorer sees are small enough for a "small-face bypass" to matter? Measured on the roster clips. env/Scripts/python.exe tests/probe_enhancer_face_sizes_stage14.py  |
| probe_eye_geometry.py |  | Do the eyes land where `swap_template_points` says, in the mask crop? A glasses-frame protection has to keep the LENS-OVER-EYE swappable while protecting rim/bridge/arms. BiSeNet cannot help |
| probe_flow_e2e_stage16.py |  | Stage 16: does the REAL pre-pass place a dropped face better with flow on? `probe_flow_stage16.py` measures the flow module in isolation. This drives roop's own `_precompute_temporal` (scan  |
| probe_flow_stage16.py |  | Stage 16: does optical flow beat geometry-only filling of a detector dropout? Ground truth is the detector itself on the UNOCCLUDED frames. A run of frames is "dropped": the filler is not al |
| probe_flow_vs_neural_landmarks.py |  | Between detector key frames, are flow-placed landmarks worse than re-running the landmark net on the flow box? app\\env\\Scripts\\python.exe app\\tests\\probe_flow_vs_neural_landmarks.py --c |
| probe_frame_space_texture.py |  | Is the skin-texture gap closable at all, and is it closable in TEMPLATE space? Context. `tests/sweep_detail_transfer.py` swept detail transfer over its whole 0-1 range on s1.mp4 and moved re |
| probe_glasses_rim.py |  | Can the eyeglass FRAME be separated from the LENS by shape alone? `Mask_RealityUX` deliberately excludes BiSeNet's glasses class (6) from the protected set, and the reason recorded there is  |
| probe_gop_bypass_stage4.py |  | Phase 4 probe: can a GOP slice with no verified target frame be STREAM-COPIED instead of rendered? Two measurements, both needed before such a bypass is written: 1. COVERAGE. Does "this GOP  |
| probe_gop_seek_stage2.py |  | Stage 2 probe: does a per-range ``cap.set(POS_FRAMES)`` land on the right frame, and is sequential ``grab()`` across the inter-range gap cheaper than the seek? Two readers are measured becau |
| probe_gpu_roi_paste.py |  | What does the ROI sub-tensor GPU paste cost next to the production CPU paste, at 720p / 1080p / 4K? app\\env\\Scripts\\python.exe app\\tests\\probe_gpu_roi_paste.py [--faces 1,2,4] [--out re |
| probe_isolation_stage17.py |  | Stage 17: do swapped pixels land on the bystander, and is the target over-protected? Two faces touch, ONE person is the swap target, the other is a bystander who must keep their own face. Th |
| probe_landmark_refresh_prepass.py |  | The REAL pre-pass at K = 4, with and without ROOP_PREPASS_LANDMARK_REFRESH, scored against its own K = 1 run. app\\env\\Scripts\\python.exe app\\tests\\probe_landmark_refresh_prepass.py --cl |
| probe_mask_flicker_stage18.py |  | Stage 18: how much does the swapped face flicker when something crosses it? Three symptoms, each asked of the OUTPUT against the PLATE (the untouched source footage), inside the target's fac |
| probe_mask_fusion_stage18.py |  | Stage 18: offline A/B of mask temporal filters on CAPTURED production masks. `capture_masks_stage18.py` records, per swapped frame and in order, the occluder model's raw mask (256, what `_ms |
| probe_matte_roi_real_faces.py |  | Canvas matte vs full-frame matte on REAL detected faces (production detector + landmarks), compared before any encode. app\\env\\Scripts\\python.exe app\\tests\\probe_matte_roi_real_faces.py |
| probe_media_io_stage13.py |  | Does the video decode / encode side wait on the GPU, and would pinned host memory change it? Measured on the app's own reader/writer. env/Scripts/python.exe tests/probe_media_io_stage13.py - |
| probe_mixed_encoder_concat.py |  | Can segments from TWO different encoders be joined without loss? (the premise of "split-GOP NVENC + QSV") env/Scripts/python.exe tests/probe_mixed_encoder_concat.py [out.json] Three segments |
| probe_nvenc_kwargs_effect.py |  | Does each PyNvVideoCodec kwarg CHANGE the bitstream? Acceptance is meaningless (unknown kwargs are ignored). app\\env\\Scripts\\python.exe app\\tests\\probe_nvenc_kwargs_effect.py Measured 2 |
| probe_nvenc_rd.py |  | Rate-distortion of NVENC option sets through the production ffmpeg path (raw BGR pipe -> hevc_nvenc). app\\env\\Scripts\\python.exe app\\tests\\probe_nvenc_rd.py <clip.mp4> [frames=150] [cod |
| probe_openvino_seams_e2e.py |  | End-to-end check of the OpenVINO seams on the two models the unit tests only reach through the runner: SCRFD (Stage 1 detector, ``retinaface._build_one``) and XSeg (Stage 3 mask, ``Mask_XSeg |
| probe_parser_protection.py |  | Measure what the RealityUX accessory gate actually does to hair and glasses. Written because `Mask_RealityUX.Run` gates BiSeNet's non-face subtraction on `accessory_allowed = clip((xseg_mask |
| probe_phase11_interconnect.py |  | Phase 11 baseline: what the five "interconnectivity" items would actually save, measured on a real render. The brief asks for (1) one probe/demuxer context across the stages, (2) precomputed |
| probe_phase11_progress_cost.py |  | Phase 11 item 5: cost of ProcessMgr.update_progress in isolation (600 calls, 1 and 20 threads, chunk and bar styles). Measured 2026-10-21 (4070 box): 5-71 us per call, 3.6-82 ms for 600 fram |
| probe_phase14_prepass.py |  | Phase 14 probe: `prepass_detect_interval` (detect every K frames, optical flow between) and the zero-phase 1-euro landmark filter, measured on a REAL pre-pass of a real clip. Four sections.  |
| probe_pose_gate_and_source_embedding.py |  | Two Stage 9 questions, answered on real heads instead of argued. env/Scripts/python.exe tests/probe_pose_gate_and_source_embedding.py --section gates env/Scripts/python.exe tests/probe_pose_ |
| probe_prepass_session_build.py |  | Where do the ~5 s before the first face frame of every pre-pass go, and does job 2 pay them again? Phase 14's re-check (docs/perf/phase14/recheck_2026-10-25/) saw ~5 s before frame 4 in EVER |
| probe_prepass_stage11.py |  | Where does the PRODUCTION pre-pass spend its time, and which Stage 11 asks can move it? ROOP_PROFILE=1 env/Scripts/python.exe tests/probe_prepass_stage11.py --section baseline --clip double/ |
| probe_progress_cost.py |  | Measure the REAL /api/progress payload, not a mock of it. Imports the actual app and drives it through TestClient, so the numbers come from the endpoint the React UI really polls once a seco |
| probe_r50_upscale_gain.py |  | Does `_rescue_upscaled` EVER gain a face on retinaface_r50? env/Scripts/python.exe tests/probe_r50_upscale_gain.py --date 2026-10-04 The question behind it: `_detect_faces` runs a 2x-upscale |
| probe_sam2_scaling_stage14.py |  | Does the production SAM 2 pre-pass slow down, or grow VRAM, as the clip gets longer? What grows with it? env/Scripts/python.exe tests/probe_sam2_scaling_stage14.py [--frames 600] [--clip dou |
| probe_segment_encoder_failure.py |  | What happens when a hardware encoder fails to OPEN at segment 2 or later of a segmented render? env/Scripts/python.exe tests/probe_segment_encoder_failure.py Real ffmpeg throughout. Segment  |
| probe_skin_tone_transfer.py |  | Does the swapped face's skin tone need a mask-isolated, temporally smoothed LAB transfer? env/Scripts/python.exe tests/probe_skin_tone_transfer.py --section static env/Scripts/python.exe tes |
| probe_source_roll_invariance.py |  | Does a SOURCE identity vector depend on in-plane (roll) rotation of the source photo? Stage 0 keys its table on (yaw, pitch, roll) but every roll bin addresses the same vector, because nothi |
| probe_sparse_scan_stage1.py |  | Stage 1 sparse pre-scan on real footage: what does it MISS, and what does it cost? Ground truth is the production detector (`get_all_faces`, full rescue ladder, aux models) run on EVERY fram |
| probe_stage25_capture.py |  | Capture, from a REAL temporal pre-pass, every track's landmarks twice: RAW (what the detector said, taken at the entry of `AdaptiveLandmarkSmoother.smooth`) and as CACHED after the shipped s |
| probe_stage25_e2e.py |  | End to end: does zero-phase landmark smoothing change the RENDERED swap? Each clip is rendered twice through the production path (user's live config.yaml models, `stabilize_face` ON as confi |
| probe_stage25_filters.py |  | Offline comparison of landmark filters on CAPTURED real tracks (`probe_stage25_capture.py`). Arms, all on the same tracks: RAW what the detector said SHIPPED what the cache holds after `Adap |
| probe_stage3_blend_gpu_vs_cpu.py |  | Stage 3 blend step in isolation: the production CPU paste vs the repo's existing GPU batched paste. The "VRAM-resident Stage 3" brief moves warp / feather / blend onto the GPU. `AGENTS.md`:  |
| probe_stage3_geometry.py |  | Stage 3 probe: what precision and coordinates does the cached Stage 2 face data carry? Runs the real pre-pass (`_precompute_temporal`) on a clip, then for the cached faces reports * the dtyp |
| probe_stage3_graph_and_warp.py |  | Two claims of the "VRAM-resident Stage 3" brief, priced on the real engine: CUDA-graph replay of a TensorRT swapper engine, and the batched GPU bicubic warp against the CPU `cv2.warpAffine`  |
| probe_stage3_plan_coverage.py |  | Stage 3 probe: if the swap pass skipped every frame absent from `stage2_frame_plan`, how many REAL swaps would it lose? Hooks ProcessMgr: records the frame index of every `process_face` call |
| probe_stage8_pcie.py |  | What do the host<->device hops of ONE face cost, against what the face costs? Phase 8 asks for the aligned crop to stay in VRAM from the swapper through XSeg and the restorer to the paste. T |
| probe_stage_cache_scale.py |  | Cost of the Stage 2 cache at scale: write time, restore time, peak RSS and file size for N frames x 2 faces. The faces carry the REAL field mix of a temporal pre-pass (float32 bbox / kps / b |
| probe_stage_telemetry_e2e.py |  | Phase 4 probe: does the 3-stage telemetry EXECUTE in a real render (not just in the unit tests)? Renders one clip with `sparse_auto_range` ON through the production harness and samples `roop |
| probe_stream_overlap_stage12.py |  | Can XSeg and the swap overlap on the 4070, and is there anything left to overlap? Measured with the real models. env/Scripts/python.exe tests/probe_stream_overlap_stage12.py [--seconds 6] [- |
| probe_swap_noise_floor.py |  | Is a swapped face cleaner than the footage it sits in, and does grain matching close the gap? Renders a window of a clip through the whole pipeline (the live config.yaml: enhancer, swapper,  |
| probe_telemetry_gap.py |  | Prove the log/parts freeze if the telemetry socket replaces the poll outright. The telemetry frame is a deliberate SUBSET of /api/progress: it carries the numbers that change per frame, not  |
| probe_telemetry_routes.py |  | Confirm the telemetry routes are actually reachable on the real app. Route-list introspection is unreliable here: this FastAPI version wraps included routers in `_IncludedRouter` objects tha |
| probe_ws_feasibility.py |  | Probe the three facts that decide the Stage 5 telemetry design. Run: app/env/Scripts/python.exe app/tests/probe_ws_feasibility.py 1. Does uvicorn in THIS env actually serve a WebSocket? (nee |

### L3. Diagnostic scripts (diag_*) (21 files)

| file | #tests | purpose / what it certifies |
|---|---:|---|
| diag_align_scale.py |  | Diagnostic: does the align_crop-fitted similarity-transform scale disagree with the face bbox's own size, specifically at extreme pitch? If a normal frontal frame's (fitted_scale / bbox_diag |
| diag_auto_angles.py |  | Does the app's own /api/target/auto_angles feature (multi-pose-angle capture) fix the s6 flicker, by giving the track-assignment gate several reference poses to match against instead of one  |
| diag_auto_angles2.py |  | Two-person variant of diag_auto_angles.py: seed both targets the way the test harness normally does (capture_targets on a separated frame), then run the real /api/target/auto_angles feature  |
| diag_auto_angles_one.py |  | Two-person variant of diag_auto_angles2.py that enriches only ONE of the two people's reference angles (leaving the other as a single best-frontal capture), for isolating whether a richer re |
| diag_best_frontal_capture.py |  | Two-person target capture that picks each person's MOST FRONTAL frame independently, rather than one shared frame (which may not have both people looking reasonably frontal at the same time  |
| diag_contact_identity.py |  | Did the swap land on the RIGHT face? Asked of the OUTPUT, per face. A pixel diff between two renders cannot answer this. The stabilizers carry state across frames, so an arm that swaps a fac |
| diag_contact_panel.py |  | Look at the frames the contact tier decided, side by side with the plate. The one check that is NOT circular. Whether a face in contact was swapped as the right person cannot be settled by a |
| diag_contact_scan.py |  | Where in this clip do two faces actually SHARE a recognition crop? "The swap flickers when another face is near it" is a claim about particular stretches of footage, and picking windows by e |
| diag_d11_bisect.py |  | Bisect d11's confirmed t~24-26s swap-dropout: trim the SOURCE clip to N frames and run it through the exact same capture+swap pipeline `run_all_samples.py` uses for the real `baseline_double |
| diag_device.py |  | Why is this machine slow? One command, one verdict. Written for the report "I set Max Threads to 7 and it runs ONE thread at 1-2 fps". That symptom has a specific cause and the app used to r |
| diag_dump_capture.py |  | Dump the actual frame images capture_targets_best_frontal() would pick for each person, for visual sanity-checking a reference capture (does the off-axis-degrees NUMBER match what the pixels |
| diag_flicker_gate.py |  | Why a face that WAS swapped goes un-swapped on the next frame. The reported symptom is the swapped face blinking on and off. The SWAP AUDIT names the population it comes from -- "frames wher |
| diag_landmark_jitter.py |  | How much does the swapped face wobble against the head underneath it? Non-circular by construction: it never reads the pipeline's own landmarks. An INDEPENDENT detector (insightface SCRFD de |
| diag_mask.py |  | Render the actual XSeg mask (and the aligned crop it runs on) for one exact frame, to check whether the mask is cutting off the forehead at an extreme backward-pitch pose -- the leading theo |
| diag_mask_both.py |  | Like diag_mask.py but dumps the aligned crop + XSeg mask + overlay for EVERY detected face in the frame (not just the widest), for investigating mask boundary behaviour when two faces are cl |
| diag_realswap_lip_colour.py |  | How far apart are hyperswap's and hififace's LIPS, and is it only colour? RealSwap's base is hyperswap; the user reports hififace's lip colour as the better one and wants that colour alone,  |
| diag_trt_fallback.py |  | Diagnostic: does hyperswap's batch>1 TRT failure permanently disable TensorRT for single-frame (batch=1) Run() calls too? Theory being tested: FaceSwapInsightFace._infer() catches ANY except |
| diag_ultramax_cost_and_colour.py |  | Where UltraMax's per-face time goes, and how far its COLOUR drifts. Two questions in one process, because both need the pipeline's real init (TensorRT DLLs on PATH; without it a 4 ms model r |
| diag_ultramax_pool_scaling.py |  | Enhancer throughput against the number of TensorRT contexts. The live ROOP_PROFILE reads enhance = 61.94 ms/call while UltraMax's own Run() measures 33.5 ms on an idle GPU. The gap is queuei |
| diag_verify_panel.py |  | Standing visual-verification format: original / swapped / amplified diff heatmap, side by side, labeled with the frame's yaw/pitch/roll. Built per the user's 2026-08-15 instruction after thi |
| diag_verify_video.py |  | Standing visual-verification format: a single VIDEO with three synced panels (original / newly-generated swap / amplified diff heatmap), each frame headed with yaw/pitch/roll and diff mean/m |

### L4. Other harness / data / support files (non-test_) (134 files)

| file | #tests | purpose / what it certifies |
|---|---:|---|
| README.md |  | Geometry tests |
| __init__.py |  | (no header) |
| _cancel_scenario.py |  | One cancelled render, run in its OWN process so the exit code is observable (Windows 0xC0000005 is a process death). Run by tests/test_release_signoff_vfr_color_cancel.py; prints one JSON li |
| acceptance_selected_face_bugcheck.py |  | Does the acceptance measurement actually catch the OLD bug? The pre-fix UI, with two captured people and ONE source face, emitted [0, 1] for every highlighted person: the highlighted person  |
| acceptance_selected_face_swap.py |  | End-to-end acceptance for "Selected face": only the highlighted person swaps. This drives the SHIPPING backend over HTTP exactly as the React UI does, with the payload the fixed UI now produ |
| angle_bench.py |  | Angle bench â€” measure what the swap does to a face as the head angle changes. Not a unit test. It needs the GPU, the model files and a real swap, so it lives beside surface_snapshot.py as a  |
| angle_video.py |  | Render the angle sweep as VIDEO, swap it through the real pipeline, grade it. The still bench (angle_bench.py) cannot exercise the orientation fix at all: the roll ambiguity is resolved alon |
| autotune_live_smoke.py |  | Live check of the VRAM governor and the auto-tune, on a real backend. app/env/Scripts/python.exe tests/autotune_live_smoke.py [--frames 600] [--no-tune] 1. Starts the React backend exactly a |
| aux_canary_calibration.py |  | Calibrate the aux canary's floors (roop/aux_canary.py SPECS) against real faces and against a wrong-picture control. env/Scripts/python.exe tests/aux_canary_calibration.py # needs output/trt |
| baseline_controlled.py |  | Phase 2: the controlled, reproducible baseline the plan requires. WHY THIS EXISTS. `PERFORMANCE_BASELINE.md`'s "OFFICIAL BASELINE -- TO BE FILLED IN PHASE 2" block is still blank in every fi |
| baseline_snapshot.py |  | The performance baseline snapshot: one run per clip, everything a later stage must match. env/Scripts/python.exe tests/baseline_snapshot.py --date 2026-10-04 For each clip it drives `two_fac |
| batch_stub_workers.py |  | Module-level workers for tests/test_batch_isolation.py. A `multiprocessing.spawn` child unpickles its target by module name, so these cannot live inside a test function or a test module that |
| bench_3d_comparisons.py |  | NOTE 2026-08-23: this arm used to name the enhancer in a spelling `roop.core.get_processing_plugins` does not match, so it rendered with NO ENHANCER AT ALL and every "x faster than CodeForme |
| bench_adaptive_enhancer_video.py |  | Phase 11 video-level enhancer matrix. This extends ``compare_enhancers_video`` rather than creating another render pipeline. Each arm renders the same locked clip with one manual enhancer or |
| bench_faceset_v2.py |  | Measure FaceSet V2 creation and lookup overhead without model claims. |
| bench_final_folder.py |  | Benchmark & Side-by-Side Comparison Generator for all video clips in roop-keep/final. Runs both CodeFormer (Baseline) and UltraMax (Photorealistic Fusion) using RealSwap + RealityUX, generat |
| bench_identity_detail.py |  | Microbenchmark and quality metrics for Phase 9 identity-detail restoration. |
| bench_mask_occlusion.py |  | Small real-runtime benchmark for the opt-in temporal mask policy. This deliberately measures only the CPU mask-filter operation. It is not a video quality claim and does not substitute for t |
| bench_phase11_enhancers.py |  | Phase 11: per-face cost of EVERY face-restoration path, on the production provider. WHY THIS FILE EXISTS. The RTX 4070 table in docs/PHASE11_ENHANCER_MATRIX.md was produced by an ad-hoc pass |
| bench_phase11_frames.py |  | Phase 11: the FRAME paths -- super-resolution, colorizers, classical resamplers. The face paths live in `bench_phase11_enhancers.py`. This covers the other 15 rows of the matrix, which had n |
| bench_phase8_transfer.py |  | Phase 8 CPU/GPU transfer and full-frame copy benchmark. This is deliberately a small, dependency-light harness. It measures the operations that the production path actually uses at 1080p and |
| bench_phase9_nvdec.py |  | Repeatable Phase 9 decode benchmark. This measures the decode boundary independently from model inference, then can run the same two-face render under CPU decode and adaptive NVDEC. The two  |
| bench_pose_source_selector.py |  | Microbenchmark for the Phase 5 classical pose/source selector. This measures selection overhead and pose-plan throughput only. It is not a video FPS, detector, GPU, or quality benchmark. |
| bench_target_appearance.py |  | Phase 10 component benchmark: lighting retention and temporal stability. |
| bench_temporal_compositing.py |  | Synthetic Phase 12 quality/cost benchmark for final paste-back blending. This isolates the compositor so a missing video/GPU cannot be mistaken for a completed quality result. The matrix cov |
| bench_temporal_identity_cost.py |  | Microbenchmark the Phase 11 temporal-identity hot paths. This is component evidence only; it is not an end-to-end video FPS claim. Run from the repository root, for example:: app/env/Scripts |
| bench_temporal_quality.py |  | Synthetic Phase 13 event/cost benchmark. This measures the controller itself, not model inference. Use it to compare the normal pass-through cost with event correction bookkeeping on the sam |
| bench_temporal_smoother.py |  | Per-call cost of the three roop/temporal_smoother.py mechanisms. WHAT THIS DOES AND DOES NOT ESTABLISH. This is an ISOLATED microbenchmark: it prices each operator at the sizes the pipeline  |
| bench_temporal_tracker.py |  | Benchmark the Phase 3 scheduler/tracker without model or GPU claims. This measures the policy overhead and the full-frame-call reduction for a deterministic two-face motion script. Use basel |
| bench_ultramax_vs_codeformer.py |  | Per-face cost of UltraMax against the `Codeformer (fp16)` it runs inside. INTERLEAVED, because end-to-end render times on this machine vary by ~18% run to run â€” larger than the effect being  |
| benchmark_perf_matrix.py |  | Benchmark matrix for RealSwap + RealityUX + UltraMax. Evaluates pool sizing, thread scaling, temporal pre-pass stepping, and enhancer cadence to find the maximum bottleneck-free throughput. |
| browser_driver.py |  | Drive a real Chromium over the DevTools Protocol, with no new dependencies. WHY THIS EXISTS. Every acceptance gate from Stage 13 onward records the same sentence -- "browser interaction coul |
| build_faceset_v2.py |  | Build a FaceSet V2 archive from a legacy V1 `.fsz`, WITH real detection. The locked `person_a`/`person_c` archives are bare PNG bundles -- five root-level `N.png` members and no metadata --  |
| capture_masks_stage18.py |  | Stage 18 helper: run `two_face_video.py` and record the production masks. For every swapped face this keeps, in call order (run it with --threads 1 so the order is frame order): raw the mask |
| check_pose_routing_render.py |  | Prove pose-adaptive source routing runs inside a REAL render, and measure what it changes. Uses the regression benchmark's own setup (config.yaml via config_sync, the synthetic 1080p clip, t |
| compare_2faces_codeformer_vs_ultramax.py |  | Comparative benchmark between CodeFormer and UltraMax on 8509564-uhd_3840_2160_25fps.mp4 swapping BOTH faces: - Left person (Target 0) -> person_a - Right person (Target 1) -> person_b |
| compare_arms.py |  | Compare two angle_video arms frame-by-frame on (pair, yaw, roll). The sweep's summary table averages within an arm, which hides the thing an A/B needs: whether a change helped the SAME frame |
| compare_codeformer_vs_ultramax_8509564.py |  | Direct comparative benchmark between CodeFormer and UltraMax on 8509564-uhd_3840_2160_25fps.mp4. Produces: 1. Swapped video with CodeFormer (FP16). 2. Swapped video with UltraMax (Photoreal  |
| compare_colour_fixes.py |  | Side-by-side of the two 2026-08-24 colour fixes, through the REAL pipeline. Renders the same frames twice with `live_swap` -- once with both fixes turned off via their env knobs (the old out |
| compare_enhancers_video.py |  | Side-by-side A/B of two enhancers on one clip, with the fps each one ran at. env/Scripts/python.exe tests/compare_enhancers_video.py \ --clip "<MEDIA_DIR>/inverted/s1.mp4" --source person_a  |
| compare_two_face.py |  | Compare two two_face_video arms on the identity columns, frame by frame. The per-arm summary two_face_video prints answers "did the pipeline swap this face, and did it pick the right faceset |
| compat_one.py |  | Swap a short clip under ONE (precision, mask engine, enhancer) combination and report whether the result is a VALID face -- not merely whether it threw. Requirement 18. The two precision fai |
| config_sync.py |  | The one implementation of "make this bench render what the user runs". WHY THIS IS SHARED. The rule in CLAUDE.md -- read `app/config.yaml` live and bench the models the user actually runs -- |
| convert_faceset_library_v2.py |  | Convert every archive in the faceset library to the current FaceSet V2. Why this exists rather than `faceset_v2.migrate_legacy_fsz`: that function is lossless for the reference PNGs, but cal |
| data/synthetic_face_jitter.npz |  | (no header) |
| enhancer_regression_sweep.py |  | Every enhancer, end to end, with proof that it actually ran. WHY THIS EXISTS, AND WHY IT IS NOT bench_phase11_enhancers.py. That file times each restorer in isolation, which is the right ins |
| exposure_ladder.py |  | Does the swapped face stay as dark as the scene it is in? WHY SYNTHETIC. The campaign's lighting phase asks whether a swap in a dark scene comes out realistically dark, and specifically whet |
| expression_bench.py |  | Swap the expression/ clips and measure whether the EXPRESSION survived. These clips exist to exercise what identity metrics cannot see. A swap can score perfectly on identity while flattenin |
| expression_eye_bench.py |  | Expression / gaze / blink bench â€” does each control move the OUTPUT toward the target, and what does it cost? Not a unit test: needs the GPU, the models and a real swap. Every arm renders th |
| expression_tracking.py |  | Do the target's eyes and mouth still drive the swapped face? The campaign asks for two things the source must NOT control: eye state (blinking, squinting, winking) and mouth/expression. Both |
| extract_frames.py |  | Dump specific timestamps from a plate + swapped video pair as PNGs for visual inspection, side by side. Time-based (not frame-index) so it works correctly across mismatched fps between plate |
| facegeom.py |  | Synthetic face geometry for the alignment / pose tests. Projects a known 3-D head at a chosen yaw / pitch / roll so tests can assert on pose behaviour without needing a video, a detector, or |
| find_profile_angles.py |  | Find frames worth banking as extra PROFILE angles for a two-person clip. Why this is not just "pick some frames where a face looks turned": * `two_face_video.capture_targets` banks extra ang |
| first_inference_probe.py |  | Run tests/two_face_video.py (same argv) and record, for EVERY ONNX Runtime session the render builds, the real provider, whether TensorRT FP16 is on, and the device memory just BEFORE and ju |
| fixtures.py |  | Locate the shared benchmark clips on whichever validation target is running. WHY THIS EXISTS. Every harness here baked in a `<MEDIA_DIR>/...` fixture path. That is one machine's drive layout |
| frame_transport_ab.py |  | Real-browser A/B of timeline playback: the pre-2026-09-24 client vs. the binary-socket / FastCanvasPlayer / 10 Hz-playhead client. WHY A HARNESS AND NOT A NUMBER IN A COMMIT MESSAGE This pro |
| frame_transport_live_smoke.py |  | Real render, real browser: the LIVE frame socket and the output compare view. frame_transport_ab.py covers timeline playback. This covers the other two paths, which only exist while (and aft |
| frontal_roll_video.py |  | Roll a FRONTAL face through a full turn as video, swap it, grade every angle. Why this exists alongside angle_video.py: that bench sweeps YAW (profile plates from a studio faceset) and rolls |
| frontalize_yaw_bench.py |  | Frontalize-by-yaw bench â€” is the existing affine frontalizer (`use_frontalization`, roop/face_frontalize.py) better or worse than no frontalization, per yaw band, on real footage? The questi |
| gate_d_cpu_benchmark.py |  | Gate D: end-to-end CPU distribution benchmark. This harness compares CPU scheduling policies on one physical validation GPU at a time. It delegates execution and telemetry to ``baseline_cont |
| gradeability_survey.py |  | How many frames of a clip can two_face_video actually GRADE for identity? two_face_video blanks its `own`/`other` identity columns whenever a face's recognition crop is shared with its neigh |
| gradio_outage_repro.py |  | Reproduce the live outage: Gradio fails, does the API survive? Occupies the Gradio port so ui/main.py's launch fails, starts run.py, and then asks the ONE question that matters: is /api/meta |
| hardware_probe.py |  | Target-bound NVIDIA discovery shared by the performance harnesses. |
| identity_algebra_bench.py |  | Does the Identity Blender change the SWAPPED face, or only the vector? Real frames go through ``roop.core.live_swap`` with the user's live config (``init_pipeline(sync_config=True)``); only  |
| image_swap_smoke.py |  | Single-IMAGE face swap, end to end, graded on identity. WHY THIS EXISTS. "Image face swap works" is an acceptance criterion of the final validation campaign and neither GPU campaign had a ru |
| integration_selected_face_regression.py |  | Integration-level regression harness for the "Selected Face" bug. Runs the SHIPPING pipeline (real detector, real recogniser, real swapper, real compositor) on a real, deterministic frame wi |
| local_only_probe.py |  | Measure whether the running backend talks to anything but loopback. WHY THIS EXISTS. Offline operation is an acceptance criterion, and every previous gate recorded it as simulated: "determin |
| measure_color_roundtrip.py |  | Measure the BGR -> encoder -> decoder -> BGR colour round-trip error. The pipeline processes frames as bgr24 in host memory and hands them to an ffmpeg rawvideo pipe. Whatever matrix the enc |
| measure_detect_tradeoffs.py |  | Does dropping det_size and stepping the temporal scan cost angle recall? The two proposed savings on the detect stage (42.4% of a render) are `face_detector_size` 640 -> 512 and `ROOP_TEMPOR |
| measure_output_noise_floor.py |  | Render one unchanged configuration N times and report how much it moves. WHY. Every pixel-level A/B this project runs -- "is this refactor equivalent", "did this feature change anything", "i |
| occlusion_gates.py |  | Thresholds for the Stage 19 interaction scenes: the brief's, and the ratchets. BRIEF what the Stage 19 brief asked for, verbatim. Some are not met; each unmet one is a strict xfail in tests/ |
| occlusion_ground_truth.py |  | Does a foreign object in front of the face survive the swap? WHY A SYNTHETIC OCCLUDER. "Foreign-object occlusion works" is the campaign's highest-priority behaviour and had no run behind it  |
| occlusion_interaction.py |  | Measure what the pipeline does when a second face or an object crosses the target (Stage 19). `synthetic_scenes.py` draws the scenes; this module renders them through the real `two_face_vide |
| phase12_benchmark.py |  | Reproducible Phase 12 end-to-end benchmark matrix. Each arm launches the real two-face video pipeline, so decode, inference, masking, enhancement, compositing, and encode remain in the measu |
| phase13_benchmark.py |  | Phase 13 end-to-end encoder and segment-rotation benchmark. Each row runs the real controlled decode -> inference -> postprocess -> encode pipeline. Codec selection is explicit for every chi |
| phase14_autotune.py |  | Run the bounded Phase 14 autotuner against the real controlled pipeline. This is an explicit/manual retune command. Normal application launches load a matching cached profile and do not repe |
| phase14_bottleneck_report.py |  | Compare Phase 14 detailed profiles without changing runtime settings. The report is intentionally a separate analysis tool: it never promotes a faster candidate, changes a provider, or treat |
| phase15_regression_audit.py |  | Generate the Phase 15 cross-hardware regression-audit report. This command is intentionally observational. It does not build TensorRT engines, delete caches, or claim a backend passed merely |
| phase16_final_quality_gate.py |  | Generate the standardized Phase 16 final production-quality report. Clip paths are supplied explicitly as ``--clip id=path``. This command never pretends that a missing category was covered  |
| phase16_integrity.py |  | Phase 16 output-integrity sweep: is the file we shipped actually intact? WHY THIS EXISTS. Every other Phase 16 harness grades the RENDER -- fps, identity, swap rate, wrong-faceset. None of t |
| phase5_quality_matrix.py |  | Phase 5: the model-quality precision matrix that previously produced no result. WHY THE EARLIER ATTEMPT FAILED, AND IT WAS NOT THE WORKLOAD. The 2026-08-28 run bounded each arm at 180 s and  |
| phase6_cuda_graph_ab.py |  | Phase 6: the provider-level CUDA-graph A/B that never reached steady state. WHAT WAS OPEN. Phase 6 closed its stream question (auxiliary=1 rejected: 29.57 against 32.58 FPS enhanced) and rej |
| phase6_pose_quality.py |  | Phase 6 real-photo pose/source-bank evaluation. This is an evaluation tool, not a production-path change. Existing local V1 FaceSets are loaded with the same detector path as ``angle_bench`` |
| phase6_temporal_bench.py |  | Real-input benchmark for Phase 6 temporal identity stabilization. This tool intentionally requires a user-provided video. It measures the low-frequency aligned-face stabilizer in isolation f |
| phase7_occlusion_bench.py |  | Real-input benchmark entry point for Phase 7 occlusion cases. The production renderer can export one restore-mask image/NPY per face crop; this harness consumes those real masks and measures |
| phase8_expression_bench.py |  | Real-video Phase 8 expression accuracy benchmark. This harness never fabricates frames or scores. Give it the original target clip and the corresponding rendered output clip. It compares tar |
| precision_matrix.py |  | Run the real compatibility guard once per backend/precision combination. Each arm is a fresh process because ORT/TensorRT builds and caches sessions at initialization. Results are written as |
| preview_pipeline_smoke.py |  | Real-browser smoke for the canvas preview pipeline and job recovery. The unit tests in test_ui_preview_pipeline.py read the SOURCE. They can prove that the code says the right thing; they ca |
| process_duo_folder.py |  | Grade the duo/ clips with TWO facesets bound to two people. Replaces a version that measured nothing about the thing it was named for. That one called `sb.run_swap(...)` as `out_file, elapse |
| process_expression_folder.py |  | Process all clips in <MEDIA_DIR>/expression with the updated RealSwap (85/15 + 100% hififace eyelashes) + RealityUX + UltraMax (natural saturation + dark spots) stack. |
| process_final_folder.py |  | Pure Video Swapper for all clips in roop-keep/final. Runs UltraMax + RealityUX + RealSwap ONLY (no CodeFormer baseline, no side-by-side). Outputs direct full-quality swapped videos to app/ou |
| process_inverted_folder.py |  | Process and test all video clips in <MEDIA_DIR>/inverted/ using RealSwap + RealityUX + UltraMax, extracting preview frames for verification. |
| provider_selection_parity.py |  | Does the selected target stay the same across CPU, CUDA and TensorRT? The runtime layer (execution provider, precision, batch path) must never change WHICH face is swapped. This harness rend |
| quant_quality_bench.py |  | Grade the swap network's INT8 / FP8 engines against FP16 on held-out faces. Model-level, on the HOLDOUT set that tools/build_calibration_set.py writes from clips that were never calibrated o |
| r50_letterbox_explain.py |  | Why does roop/retinaface.py say letterboxing r50 "suppresses scores under TensorRT on 16:9"? env/Scripts/python.exe tests/r50_letterbox_explain.py [--per-clip 40] [--out f.json] The claim ha |
| r50_profile_probe.py |  | retinaface_r50 (ONNX Runtime + TensorRT) shape-profile probe: what profile is it built under, and what does ONE pooled instance cost in device memory around its first inference? env/Scripts/ |
| react_ui_controls_acceptance.py |  | Live Chromium audit for React tabs and processing controls. This intentionally drives the UI against a real, already-running backend. It is not a source-only check: tab clicks, pause/resume/ |
| restore_ultra_bench.py |  | Restore Ultra: linear vs frequency-split recombination, graded on real swaps. Every Restore Ultra call made while swapping real frames (roop.core.live_swap, live config) is intercepted, and  |
| review_sheet.py |  | Side-by-side PLATE vs OUTPUT face crops, tiled into PNG sheets to LOOK at. Requirements 3, 5, 10, 16 and 17 of the phase-4 checklist are appearance questions -- is a hand painted over, is a  |
| run_all_samples.py |  | Drive sample_bench's swap logic over every clip in roop-keep/single and roop-keep/double in one process (keeps detector/swap/enhancer model pools warm across clips instead of paying cold-sta |
| runtime_lifecycle.py |  | End-to-end runtime acceptance against the REAL backend on the real GPU. WHY THIS EXISTS. Batch queue, true pause/resume, persistent projects and interruption recovery are React UI 2.0's head |
| sample_bench.py |  | Run one real sample clip (from <MEDIA_DIR>/single or /double) through the real pipeline with real library facesets, using the app's OWN live config.yaml settings (detector, tracking, mask en |
| scan_inverted_clips.py |  | Scan inverted folder clips to analyze roll angles, detected faces, and autorotation actions across frames. |
| size_pumping.py |  | Does the swapped face change SIZE from frame to frame while the jaw moves? Requirement 16. "Pumping" is a TEMPORAL artifact -- the face breathing bigger and smaller as the mouth opens and cl |
| surface_snapshot.py |  | Public-surface snapshot â€” proves a refactor was a PURE MOVE. Decomposing a large module is only safe if nothing observable changes. This records everything a caller could depend on: * every  |
| survey_fixtures.py |  | What is actually IN each clip, so a phase gets the right fixture. WHY. Fixtures here have been chosen by name and by memory, and it has gone wrong twice in ways that invalidated whole result |
| sweep_detail_transfer.py |  | Sweep `detail_transfer_strength` against the skin-texture gap it exists to close. env/Scripts/python.exe tests/sweep_detail_transfer.py \ --clip "<MEDIA_DIR>/inverted/s1.mp4" --source person |
| synthetic_scenes.py |  | Deterministic multi-frame scenes for the interaction stress tests (Stage 19). Drawn faces are not detected, so the "faces" are the app's own real 256x256 faceset photos (`app/facesets/*.png` |
| telemetry.py |  | Host and device telemetry sampling for the phase benchmarks. Split out of `bench_phase9_nvdec.py` so Phase 2's controlled baseline, the Phase 11 enhancer matrix and any later phase all repor |
| trt_matrix.py |  | TensorRT build matrix: heuristics {on, off} x builder level {3, 5} per small model. env/Scripts/python.exe tests/trt_matrix.py prepare # real inputs + CUDA-FP32 references, once env/Scripts/ |
| trt_precision_fidelity.py |  | Shipped TensorRT 'mixed' vs a genuine FP32 reference, on ~500 REAL faces, for the four small models that ride the global trt_precision: XSeg, w600k_r50, 2d106det, 1k3d68. env/Scripts/python. |
| trt_precision_probe.py |  | What precision do XSeg / w600k_r50 / 2d106det / 1k3d68 REALLY run under in the shipped TensorRT config, and does the XSeg NHWC input cost layers? env/Scripts/python.exe tests/trt_precision_p |
| two_face_video.py |  | Two people, two facesets, faces that touch â€” the repro bench. The reported failure ("both faces flicker, go unswapped, or swap into each other, but ONLY while the two people are close or tou |
| ui_browser_acceptance.py |  | Real-browser acceptance for React UI 2.0 and a React UI 1.0 regression smoke. WHY THIS EXISTS. Stages 13 through 17A all recorded "no browser runtime was available" and marked V2 launch, rou |
| ui_poll_launcher.py |  | Run tests/two_face_video.py (same argv) with the REAL API server alive on a thread of the render process. ROOP_BENCH_LAUNCHER=tests/ui_poll_launcher.py ROOP_UIPOLL_PORT=<port> (see tests/ab_ |
| unittest_shim.py |  | Make module-level ``test_*`` functions visible to ``unittest discover``. WHY THIS EXISTS. This project's standing verification command -- the one in CLAUDE.md, and the one every session log  |
| verify_double_router.py | 3 | tests: sliding_window_centroid_and_similarity, dual_threshold_hysteresis, person_k_and_person_e_crossing_simulation |
| verify_glasses_protection.py |  | Before/after on real footage, through the SHIPPED engine. `probe_parser_protection.py` answers "what does the accessory gate read". This answers a different question -- "did the change actua |
| verify_gpu_recovery.py |  | A real render with a one-shot CUDA out-of-memory injected into chosen frames: recovered or written untouched? Arms (same clip, user's live models, `swap_faces` raising the exact ORT message  |
| verify_realswap_lip_colour.py |  | Does the lip-colour transfer do what it claims, and ONLY that? Four checks on real frames rather than synthetic arrays, because three of them are about what the operator must NOT do: 1. the  |
| verify_roop_keep.py |  | End-to-end quality and regression verification over the roop-keep corpus. Renders every clip in `<base>/single` against one faceset and every clip in `<base>/double` against two, then grades |
| verify_s5_profile.py | 1 | tests: s5_frames_alignment |
| verify_stage0_identity.py |  | Phase 6, Stage 0 end to end: the frozen source-identity table hands the swapper the SAME vector the live per-face selection would have, and the swap pass stops re-deriving it. Same harness a |
| verify_stage11_render.py |  | Prove the Stage 11 hooks EXECUTE inside a real render, on real footage, and measure what they change. AGENTS.md: "before believing no effect, prove the code path executes ... count the stage |
| verify_stage17_render.py |  | Prove Viseme Mouth Custody EXECUTES inside a real render, on real footage, through the real request chain. AGENTS.md: "before believing no effect, prove the code path executes ... count the  |
| verify_stage20_render.py |  | Prove identity KEYFRAMES execute inside a real, TRIMMED render and land on the right frames. AGENTS.md: "before believing no effect, prove the code path executes ... count the stage, not the |
| verify_stage4_report.py |  | Phase 9 end-to-end: Stage 4 finalization and the per-job report, on real renders of the Phase 5 clip. Reuses `verify_three_stage_pipeline.py`'s ground-truth clip (600 frames, known empty / d |
| verify_stage_resume.py |  | Phase 6 end-to-end: an interrupted job resumes at Stage 3 without re-running Stage 1 or Stage 2. Built on the Phase 5 harness (`verify_three_stage_pipeline.py`: the same synthesized 600-fram |
| verify_swap_visual.py |  | Ground-truth check: did an output video actually get a face swapped in, or does it just look like the plate while the pipeline's own SWAP AUDIT claims success? Session 3 of the roop-recode p |
| verify_swap_visual2.py |  | Ground-truth check for 2-person clips: does the SAME screen position keep the SAME source identity throughout, or does it interchange? Independent of the pipeline's own track/source bookkeep |
| verify_temporal_optimization_corpus.py | 2 | tests: s5_single_face_stability_and_speedup, double_d1_multi_face_stability |
| verify_thread_binding.py |  | Dry-run verification that CPU affinity and ORT thread settings ACTUALLY apply. Run this BEFORE a render, not after, to answer three questions that this project has repeatedly got wrong by re |
| verify_three_stage_pipeline.py |  | End-to-end verification of the 3-stage pipeline on a synthesized 600-frame ground-truth clip. Stage 1 sparse pre-scan -> Stage 2 active-range pre-pass -> Stage 3 main swap, rendered through  |
| xseg_refine_bench.py |  | Does the XSeg edge refinement / flow-warped mask stabilizer actually help? Two features, one question each: mask_guided_filter -- does XSeg's boundary land closer to the real face edge? mask |

### M. Other test_ files (10 files)

| file | #tests | purpose / what it certifies |
|---|---:|---|
| test_7_boost_fixes.py | 7 | Comprehensive verification tests for all 7 boost fixes and edge cases. |
| test_association_cost.py | 16 | Stage 16: the pre-pass association cost. `legacy_cost` is the cost `_consume` has always used; these pin its four branches so lifting it out of the closure cannot have changed it. `composite |
| test_audit_regressions.py | 17 | Regression guards for correctness fixes found during the five-stage audit. These source-level checks deliberately avoid loading the model stack. They pin small integration seams whose failur |
| test_face_regressions.py | 7 | Focused regressions for motion continuity, object occlusion, and identity routing. |
| test_phase11_inventory.py | 3 | Phase 11 inventory and hardware-isolated matrix contracts. |
| test_phase12_pipeline.py | 10 | Dependency-light regression checks for the Phase 12 scheduling contract. |
| test_phase13_pipeline.py | 8 | Dependency-light contracts for the Phase 13 output pipeline. |
| test_phase14_profile.py | 3 | Regression tests for Phase 14 profile parsing and bottleneck reporting. WHY THIS IS unittest AND NOT pytest. This module used to `import pytest` at the top. `pytest` is not installed in `app |
| test_phase15_regression_audit.py | 6 | Unit contracts for the Phase 15 cross-hardware audit. |
| test_single_worker_warning.py | 7 | One worker on a GPU provider must not be silent. `max_threads: 1` in config.yaml with a working CUDA/TensorRT provider is the one remaining way to lose all of the pipeline's parallelism with |

## 4. Axis findings (slice)

1. **Detection/landmarks/angle/failure thresholds** â€” UI shows pose bounds only as HUD scale (yaw Â±90, pitch Â±60, roll Â±45: `headPoseModel.js:10`); no UI-side failure threshold. Docs: selector tolerance 15Â° yaw/pitch (`docs/PHASE6_POSE_QUALITY.md`), switch hysteresis 0.035 (`POSE_AWARE_SOURCE_SELECTION.md`), Stage4 audit (synthetic) cites yaw>35Â° far-eye collapse, >45Â° 3-point solver. Tests pin canonicalization at roll 60Â° (`test_pose_canonicalization.py:39`). Face bank pose coverage UI: `facemgr/PoseRadar.jsx`.
2. **2D warping vs 3D/faceset/.fsz** â€” Entire reference is 2D similarity warp (trajectory cache M/M_inv, `perf/phase14/README.md`). "3D" = optional pose-matched source crop (`use_3d_recon`, off by default, no effect on inswapper-family: `FaceSwap.jsx:4007`). `.fsz` V2 (`FACESET_V2.md`) stores 3D-68 landmarks + yaw/pitch/roll per source; tests `test_faceset_v2.py`, `test_gpu_filmstrip_and_scene_cuts.py::test_fsz_v2_pose_matched_embedding_and_facemgr`. No volumetric UI.
3. **Visual quality/blending/masks/colour/restorers** â€” Restorer pipeline stack UI (`EnhancerPipelineStack.jsx`: Swapper â†’ Restorer â†’ Texture&Grain â†’ LAB colour), swapper presets inswapper_128 / ReSwapper 256 / Alpha 256 / HyperSwap 256 / HyperSwap 512 (`pipelineRules.js:SWAPPER_PRESETS`), restorer auto-bypass for alpha/hyperswap (`pipelineRules.js` BYPASS_SWAPPERS), 4/9-up comparison grids, parser regions (19 classes), LUT/scopes. Docs: matte ROI, grain match, fused Triton post-process, FP16 islands, PRECISION policies (`docs/perf/*`).
4. **Temporal tracking/smoothing/scene-cut** â€” UI four modes (`temporalModes.js:28`), cut-aware is engine property not a setting; docs Phase 6/7/8 opt-ins with listed thresholds; Phase14 K-interval verdict (default 1) and one_euro option (default adaptive); live mode histogram cut detector. Tests: smoother/flow/trajectory suites above.
5. **CLI/terminal/telemetry/diagnostics** â€” Terminal dashboard mirrored in UI from real `roop/terminal_dashboard.py` snapshot (`ConsoleMirror.jsx`, fixture generated by backend), HUD pipeline pill + report, TelemetryPill (GPU/VRAM/CPU), flamegraph stage bars (decode/detect/swap/mask/restore), failure log (`failureLog.js`), `docs/development/TERMINAL_CONTRACT.md`.
6. **Compute** â€” UI: provider benchmark/matrix (`ProviderBenchmark.jsx`, `AutoTunePanel.jsx` screens provider Ã— swap batch Ã— NVENC), TRT cache panel, compile banner. Docs: dual-GPU matrix (RTX 4070 12GB / 3060 Laptop 6GB, never averaged), TRT 10.9.0.34 / ORT 1.23.2, CUDA_EXECUTION_POLICY, aux FP16 NOT shipped (stays FP32), cross-frame swap batching 4-9% slower on 2-person footage.
7. **Job queue/checkpoint/video IO** â€” UI jobStore reattach, ResumableCheckpoints, SegmentMatrix (queued/rendering/completed/failed; unknown â†’ queued), Batch Matrix; docs JOB_CONTRACT, STAGE12 video IO, NVENC quality options, QSV mixed-encoder concat failure notes; tests listed in Â§3.
8. **Workstation UI** â€” Rail + Header + LayoutSwitcher presets (Standard, Max Canvas, Inspection Suite, My Roto Setup, Dual-Monitor Color, Minimal Laptop: `LayoutSwitcher.jsx`), resizable panels (350/330/280 defaults), undo/redo with 50-action popover (`HistoryPopover.jsx`, `historyEngine.js`), rebindable shortcuts (`shortcuts/actions.js`), command palette, PiP mini-viewport, pop-out reference monitor (`#/monitor`, BroadcastChannel `roop_ultimate_reference_out`), 38 themes + Theme Studio, offline-bundle verification.

## 5. Gaps / stubs / defects found

- **No 3D/volumetric viewer**: nothing in react-ui renders mesh/volume/depth; `use_3d_recon` documented as NO-OP for the inswapper/ghost/hyperswap/simswap family (`FaceSwap.jsx:4007`).
- **UI 2.0 was removed (2026-09-02)** but 11 `docs/development/UI2_*` docs remain as SUPERSEDED; ENVIRONMENT_CONTRACT.md still says "React UI 2.0 is the default client" while CURRENT_STATE/SESSION_HANDOFF say UI 1.0 is sole client â€” **doc drift**.
- **Invalid/synthetic audit reports**: `docs/audits/STAGE3-STAGE9_*` carry synthetic-input banners; STAGE5 is INVALID - DO NOT CITE. Do not import their numbers.
- **Flow/neural trade-off**: K>1 pre-pass interval loses source-bound frames (K=8: -31% on Love) â†’ opt-in only (`phase14/README.md`).
- **Aux batching, FP16 aux models, GPU ROI paste, temporal colour EMA, cross-frame swap batching**: all measured and NOT shipped/wired (`perf/aux_batch_2026-10-05.md`, `aux_fp16_onnx_2026-10-11.md`, `gpu_roi_paste/README.md`, `color_ema/README.md`, `swap_batch_render_2026-10-11.md`) â€” baseline-only knowledge, useful as negative results.
- **Phase 7/8 real-video validation OPEN**, `.fsz` archives lacked pitch/inversion plates (`PHASE6_POSE_QUALITY.md`).
- **Live-cam 45 ms target unverified** (`LIVE_MODE.md`).
- **Test-suite shape**: 243 of 717 files are harness/probe/diag scripts (not collected as pytest tests); `test_live_acceptance_suite.py` has no docstring and drives a live server over HTTP (needs running backend) â€” [INFERENCE] not run in CI. Tests by regex: 6453 `def test_`.
- **UI**: `store/recipeStore.ts` has no consumer (`historyStore.js` header says so) â€” dead typed store; `src/App.css` empty header; `react-ui/package-lock.json` 72 KB.


## Coverage

Slice files in inventory: 1422 | files accounted for: 1422 | omitted: none
