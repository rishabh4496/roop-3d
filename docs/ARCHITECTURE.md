# Developer architecture

For people changing Roop 3D. Users never need this: see the [user manual](USER_MANUAL.md). This page maps the real layout, so older stage documents
that name `app/api.py`, `app/roop/*.py` or `react-ui/` should be read as the paths below.

## 1. Processes and ports

```
Pinokio ── start.js ──► python -B app/serve.py ──► roop_3d.server.__main__ (FastAPI + Uvicorn, 127.0.0.1:8030)
                                                     ├── /api/v1/…       REST (see §3)
                                                     ├── /api/preview     live swap preview (alias of /api/v1/preview)
                                                     ├── /ws/…            telemetry and job events (WebSocket)
                                                     └── /                the built web interface (app/roop_3d_ui/dist)
Browser ◄──────────────────────────────────────────── the same origin: no proxy, no CORS
```

* One process, one origin. Requests whose `Origin` is not this service get HTTP 403 `ORIGIN_DENIED`. Any `/api/<path>` also resolves to `/api/v1/<path>`.
* The server's **data root is `app/`** (`roop_3d/server/app.py`: `Path(__file__).absolute().parents[2]`). Everything user-owned lives under `app/roop_3d/.runtime`;
  model files under `app/roop_3d/models/cache`. `tools/project_admin.py` runs from the **project** root; for anything that touches data it must use
  `APP_ROOT = ROOT / "app"` (a Stage 19 defect fixed exactly there: model import used the project root and the server never found the files).
* Independence: the app never imports from, launches or modifies Roop Ultimate (`docs/STANDALONE_REPORT.md`).

## 2. Repository layout

```
install.js start.js update.js reset.js torch.js pinokio.js pinokio.json   Pinokio launcher
tools/project_admin.py ci_runner.py release_packager.py …                 project-level CLI (preflight, model-import, reset check, CI gates)
app/serve.py                                                              entrypoint
app/roop_3d/                      Python package
  server/       routers: routes, workstation, workstation_preview, live_preview, face_extract, face_bank, identity_lock, identity_assets,
                orchestrator, professional_media, finishing, queue_routes, installer_routes, reliability_routes, ws, middleware, security
  core/         settings (Pydantic, single source of truth), hardware probing and profile resolution, errors (DomainError)
  jobs/         JobStore, ProjectStore (.r3dp), worker, checkpoints, schema (JobRequest, JobFailure{code,message,recoverable})
  engine/       pipeline_runner (frame loop, mux), vram_governor + scheduler (admission by profile), stage runners
  media/        probe, inspect, preview, export_profiles, video_writer, finishing, vfx, sequence, capabilities
  faceset/ reconstruction/ identity/   FaceSet reader and quality analysis, 3D identity assets, identity lock and guard
  models/       registry, governance (licence/safety gate), local_provider, downloader, hub, verifier, catalog
  orchestrator/ capability-based selection of pipeline components per tier
  installer/    preflight, model_manager (offline import), offline_cache, update_manager, verification
  reliability/  soak harness, startup reconciliation, repair, resource audit
  compositor/ swappers/ enhancers/ …   the neural stages
  tests/        pytest (offline; no weights required)
app/roop_3d_ui/                   React 19 + TypeScript + Vite interface
  src/App.tsx   the workstation shell: state, API calls, routing, dialogs
  src/components/{studio,facemgr,faceswap,hud,queue,export,timeline,canvas,workstation,diagnostics,modelhub,three,controls,guide,tour,…}
  src/stores/   small external stores (playhead, telemetry, engine selection, history, curves)
  src/utils/    letterbox/zoom maths, motionPrefs, themeContrast, resource ledger, exporters
  e2e/          Playwright (system Chrome, mocked service) · mock-server/ Express mock for `npm run dev:mock`
docs/           this documentation + stage reports
```

## 3. API surface (selected)

| Area | Endpoints |
|---|---|
| Health, hardware | `GET /api/v1/health`, `/system/capabilities`, `/system/hardware`, `/system/telemetry`, `POST /system/preflight-lint` |
| Config | `GET/PUT /config` (every value validated by `core.settings.Settings`) |
| Media | `POST /media/probe`, `/media/inspect-faces`, `GET /media/preview-frame`, `/media/keyframes`, `GET /media/export-profiles` |
| FaceSets | `GET /facesets`, `…/{id}/views`, `…/coverage`, `…/identity-asset` (GET never builds; POST builds) |
| Projects | `GET/POST /projects`, `PUT /projects/{id}`, `POST /projects/{id}/jobs`, `/projects/{id}/resume`, `/projects/import`, `GET …/download` |
| Jobs and queue | `GET/POST /jobs`, `…/{id}/pause|resume|cancel`, `GET /queue`, `POST /queue/{id}/retry|duplicate`, `/batch/*` |
| Models | `GET /models/hub`, `POST /models/hub/verify`, downloads (explicit, hash-verified) |
| Installer | `GET /installer/preflight`, `/installer/offline-health`, `POST /installer/models/scan|import|verify`, `/installer/recover`, update/rollback |
| Orchestrator | `GET /orchestrator/engines?tier=`, `POST /orchestrator/plan` |
| Reliability | `GET /reliability/report|resources|audit`, `POST /reliability/repair` |

`GET /openapi.json` is the authoritative schema. Errors are always `{"error": {"code", "message", "request_id"}}`; the interface keeps the code
(`ApiError`) and shows the request ID.

## 4. Jobs, checkpoints and recovery

A job is a `JobRecord` (state machine: queued → preparing → running → pausing/paused → succeeded/failed/cancelled, plus `interrupted` after an
unclean stop). Progress is persisted every frame and a checkpoint of finished **output frames** about every 30 frames. A checkpoint does not hold tracker or model
state, so **resume replays the analysis from frame 0** and skips only re-saving finished frames (`docs/RELIABILITY_STAGE18.md`). Startup
reconciliation quarantines manifest-less outputs, releases leaked locks and never touches job ids known to `queue.db`. `JobFailure.code` is what the
interface turns into recovery guidance.

## 5. The interface

* **State.** `App.tsx` owns project/target/inspection/assignments. The committed playhead is React state; per-tick playback time lives in `playheadStore`,
  telemetry in `telemetryStore` (neither re-renders the workstation). Store style: `useSyncExternalStore` (`stores/atomicStore.ts`); Zustand is not used.
* **Startup.** `connect()` loads capabilities, config, FaceSets and jobs together; `/projects` loads **separately** because the server re-verifies every saved
  project's input hashes and can take many seconds (it must never hold the rest of the app hostage); the UI shows *Checking saved projects…* meanwhile.
* **Layouts.** `studio` (four panels + timeline) or `classic`, per browser. Panels are error-isolated (`PanelBoundary`). At ≤1279 px the preview frame is ordered first
  and its height is clamped to the window, never grown by the neighbouring column, and it never asks the server for a smaller image.

### Stage 19 modules (`src/components/guide`, `src/utils`, `src/components/controls`)

| Module | Role |
|---|---|
| `helpContent.ts` | **Single source of truth** for every explanation: workflow steps, photo rules, engine/hardware/preset/job-state text, recovery steps. Numbers here mirror backend constants and are pinned by `stage19.test.ts` |
| `errorGuide.ts` | Backend error code → *title, cause, steps, doc anchor*. `explainError({code,message})` falls back by code family, a code named in the message, then browser failure text, and returns `null` rather than inventing |
| `workflowState.ts` | Pure `workflowProgress(inputs)` → the seven steps with `done/current/todo`, the next step and its position; onboarding flag |
| `coverageGuide.ts` | Plain-language reading of the pose-coverage grid; parses `[required]/[recommended]/[optional]` request prefixes |
| `GuideDialog`, `WelcomeDialog`, `NextStepBar`, `WorkflowChecklist`, `RecoveryNotice`, `EngineExplainer`, `HardwareExplainer`, `AccessibilityControls` | Presentation only; they receive state, they do not fetch it (except preset lists) |
| `ExportPresetBar` + `export/exportOptions.ts` | The Export preset control, `useExportProfiles()` and `normalizeExport()` (drops server-default fields so the UNSAVED badge stays honest) |
| `controls/useDialogA11y.ts`, `guide/DialogFrame`, `guide/DialogPanel` | **One keyboard contract for every dialog**: focus in, Tab trap, Esc closes, focus restored |
| `utils/motionPrefs.ts` | `system | reduce | full`; one effective boolean drives `html[data-motion]` (CSS) and the 3D loops (`reducedMotion()`) |
| `utils/themeContrast.ts` | After any theme change, measures the token pairs the UI uses and overrides only failing ones on `<body>` (4.5:1); pure core `enforce()` is unit-tested |
| `stage19.css` | Focus ring, reduced motion, guide styles, minimum target sizes, light-theme fixes, tablet layout. Loaded after `styles.css` |

### Adding or changing guidance

1. Add or edit the entry in `helpContent.ts` / `errorGuide.ts`. Give an error a `doc` anchor that exists in `docs/COMMON_PROBLEMS.md`.
2. Update the matching docs page. `app/roop_3d/tests/test_stage19_docs.py` fails if a documented code, preset, profile, number, path or command no longer
   matches the code.
3. Add the new code to `explainError` tests if it introduces a new fallback rule.

## 6. Accessibility rules the code relies on

* Every dialog uses `DialogFrame`/`DialogPanel` (or `Modal`). Do not hand-roll focus handling.
* Controls need an accessible name; icon-only buttons need `aria-label`. `aria-disabled` (not `disabled`) is used where pressing must explain why it cannot run.
* Never convey state by colour alone (coverage cells show counts and have labels).
* Minimum target 24 × 24 CSS px (WCAG 2.5.8), checked by axe's `target-size` rule in `e2e/stage19.e2e.ts`.
* Use theme tokens (`--text --muted --accent --ink --danger --bg --surface --raised --soft --line`). `.mint`/`.amber` are *background* classes.

## 7. Testing

| Suite | Command (from `app/roop_3d_ui`) | Notes |
|---|---|---|
| Typecheck + lint + build | `npm run build` | `typescript-eslint` cannot run on TypeScript 7; zero lint **errors** is the bar |
| Unit (Vitest + jsdom) | `npx vitest run` | `stage19.test.ts` covers the Stage 19 logic |
| Browser (Playwright, system Chrome) | `npx playwright test` · `npx playwright test e2e/stage19.e2e.ts` | The service is mocked per test (`e2e/helpers/mockApi.ts`); includes axe-core WCAG 2.2 AA, the all-themes contrast sweep, five viewports × nine screens with screenshots in `test-results/stage19/` |
| Python | from `app/`: `roop_3d/.venv/Scripts/python.exe -m pytest roop_3d/tests` | Never run two pytest sessions at once (`--basetemp` is shared); the full suite is long, run it in chunks |

The Stage 19 browser spec needs a desktop-class window for studio drags (1680 × 1050).

## 8. Where decisions are recorded

`docs/UX_STAGE19.md` (this stage), `docs/RELIABILITY_STAGE18.md`, `docs/HARDWARE_COMPATIBILITY_MATRIX.md`, `docs/POST_PRODUCTION.md`, `docs/UI_PERF_STAGE11.md`,
`docs/WORKSTATION_STAGE6.md`, `docs/STANDALONE_REPORT.md`.
