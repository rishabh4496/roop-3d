# Stage 8: Complete offline video workflow

## Technical implementation plan

This stage integrates the accepted workstation and local renderer; it does not add another detector,
tracker, compositor, queue, FaceSet library, or inference adapter.

### Existing production call path

`App.tsx` imports via chunked `/assets/uploads`, inspects with `/media/inspect-faces`, assigns tracks
through `useFaceSelectionStore`, previews through `/api/preview`, and posts `JobRequest` to `/jobs`.
`JobWorker` dispatches the reference engine through `make_stream_runner`; workstation requests use
`MasterPipelineRunner` because they contain geometry, timeline, and finishing options. That runner
reuses `ReferencePipeline`, its identity rigs / motion transfer, compositor and restoration, saves
completed frames and checkpoints, and calls `encode_video`. The writer re-reads the source timeline,
substitutes processed frames by index, muxes all audio tracks, verifies decoded output PTS/count,
and publishes the video and manifest before the worker records success.

### Proven gaps and minimal changes

1. Checkpoints bind target/request/settings/FaceSets but omit model digests and inspection bytes.
   Extend the existing checkpoint with verified render dependency identities and reject drift before
   inference. Preserve legacy checkpoint decoding; refuse continuation of completed legacy frames
   whose model identity cannot be established. Reuse the project's model snapshot logic.
2. Project saves always create another ID, omit preview state and asset references, and have no job
   association or output history. Extend `.r3dp` additively with typed preview state, revisioned updates,
   asset build references, and linked jobs. Queue project jobs through the existing JobStore/worker;
   expose history derived from those durable records, so browser closure cannot lose a render.
3. The queue drawer revives localStorage records after an empty server response and offers visual-only
   reordering. Make server records authoritative, retain truthful controls, and add inspection, video
   export and manifest links only for successful published jobs.
4. Project restore loses source/target selection and identity lock; selecting a different job for the
   same target leaves the old controls. Restore all saved controls consistently. Persist viewport
   preview preferences without coupling 30 Hz playback ticks to the root component.
5. Verify the existing rational VFR, orientation, colour metadata, all-track audio, cancellation,
   pause/restart, bounded frame transport, and atomic publication contracts with integration tests.
   Fix defects only when demonstrated.

### Acceptance campaign

* New API/regression tests: two independent projects, optimistic update conflicts, preview/asset state,
  project-job history, input/model/checkpoint mismatch, failed/cancelled output gating and publication faults.
* Existing worker/media/identity tests plus new VFR and metadata integration coverage.
* Frontend unit tests, production typecheck/lint/build, and browser regressions.
* Real production UI/API journey on loopback using local FaceSets and actual verified model inference:
  import, inspect, assign, coverage, build/reuse asset, select engine, preview, edit, project save/reopen,
  queue, pause/resume, completion, inspect and export video/manifest. Record executed evidence and limits.
* Full Roop 3D Python suite and read-only legacy audit. No model downloads or legacy writes.

## Integrated behaviour

* Import a local `.fsz` in **Identity**, import a video in **Target**, inspect, select a target track,
  and assign the FaceSet. Import waits for the service's upload capability rather than silently
  ignoring a file during startup.
* The workstation requests `full_video: true` from the existing media inspector, with a 900-second
  time budget. It traverses to EOF instead of stopping at the legacy keyframe/decoded-frame sampling
  caps. Legacy callers retain their old bounded defaults. Time/identity limits still produce an
  explicit truncated report; the UI never treats that result as a complete inventory.
* The existing coverage panel and explicit 3D asset build/view controls remain authoritative.
  The inspector selects the existing reference engine and standard / pose / reenactment modes.
  The frame preview now carries the identity lock and motion strength as well as the existing
  geometry, masks, curves, finishing, restoration and lighting options.
* Workstation previews select **source timestamps**. Identity-aware previews replay the existing
  tracker and compositor from frame zero; direct frame selectors use exact decoded ordinals.
  VFR timestamps cannot collide in the preview cache by rounding to one nominal frame number.
  Once a frame is decoded, timestamp and ordinal aliases share its cached result, including during
  GPU contention; the public configuration hash remains independent of the frame selector.
  The legacy timestamp/keyframe fast path remains available for older, non-identity CFR inspections.
* **Save project** updates the current `.r3dp`; **Save copy** creates another independent project.
  Revisions reject conflicting browser saves. Projects retain source/inspection/FaceSet hashes,
  engine and render options, curves/masks/ranges, verified asset build keys, selected source/target,
  playhead, comparison/overlay/grid/zoom preferences, checkpoint link and render history.
  Reopening restores the identity lock too. Selecting another job restores its controls even when
  its target video is already open.
* **Queue render** saves the current project, then creates a normal durable job associated with
  that project and its effective settings. Changes to global settings cannot silently change a
  queued project's render. The worker, scheduler and reference inference remain the existing ones.
* The existing queue is authoritative even when empty. Pause/resume/cancel use the existing legal
  transitions. The old visual-only queue reorder buttons were removed because they never changed
  the worker's order. Failed jobs can still be retried as fresh jobs through the existing control.
  Transitional states are labelled `PAUSING` / `CANCELLING` until the worker acknowledges them.
* A successful queue card offers **Inspect output**, **Export video**, and **Render manifest**.
  The inspection dialog uses a real HTML5 video player, includes the synthetic-media notice, and
  downloads the same published files. Failed, paused, cancelled or incomplete jobs cannot use
  the new artifact endpoints. Export rechecks the video hash and requires its manifest to exist.

## Additive contracts and recovery

| Contract | Behaviour |
| --- | --- |
| `POST /api/v1/media/inspect-faces` | Optional `full_video`; skips legacy frame/keyframe caps, retains time/identity/storage bounds |
| `POST /api/v1/projects` | Existing create contract; accepts optional typed `preview` state |
| `PUT /api/v1/projects/{id}` | Save current project; requires its last `revision`; conflict is HTTP 409 |
| `POST /api/v1/projects/{id}/jobs` | Queue the saved request with `Idempotency-Key`, project ID and effective settings |
| `GET /api/v1/projects/{id}` and `/download` | Include the project's durable render history, including completed output references |
| `GET /api/v1/jobs/{id}/artifacts/video` | Completed baked video only, verified against the durable output SHA-256 |
| `GET /api/v1/jobs/{id}/artifacts/manifest` | Manifest of a completed published render |
| Existing `POST /jobs`, preview and job control routes | Preserved; new project association/settings fields are optional on `JobRecord` |

Checkpoint identity now includes verified available model digests, inspection bytes, and the
3D asset build keys / explicit unavailability used by the session, alongside the existing
target/request/settings/FaceSet identities. Installing an optional model between attempts or
changing source geometry cannot silently mix two configurations. Model bytes are verified;
registered model names alone are insufficient. Missing optional weights are omitted from the
available dependency snapshot; corruption at a registered weight remains fatal.

Existing `.r3dp` files and checkpoint schemas still decode. **Completed older checkpoints without
model identity must be rendered again as a new job**; their bytes are preserved, and continuation
returns `JOB_CHECKPOINT_MODELS_UNKNOWN`. This explicit safety migration is regression-tested.
One unreadable project no longer makes every other project disappear from the library. A project
that exceeds the existing 512 KiB manifest contract fails before publishing an unusable save.

The reference path continues to process frames by decoded index, store lossless completed frames,
and encode from the source's rational timeline. VFR remains VFR under `preserve`; `cfr` is an explicit
existing policy, with a frame map. Orientation is baked into upright pixels; pixel aspect ratio is
preserved (inverted for a quarter-turn), as are the existing colour fields and all-track audio policy.
The writer verifies decoded frame count, PTS, audio decodability and output pixel aspect before
publishing; mux progress is reported separately. The worker only records success after publication.
The existing synthetic-media comment, provenance recipe and optional signing paths remain intact.
The active GPEN/CodeFormer model hash is captured before teardown clears the inference session;
the manifest also records the restorer's separate weights, inference and provenance evidence.

## Executed evidence

These are results of this stage's execution, not numbers copied from historical reports.

| Gate | Result |
| --- | --- |
| Focused backend integration / preview / media regressions | 48 passed, including 12 Stage 8 tests; full traversal vs. legacy sampling, project/checkpoint safety, VFR, audio/metadata and preview compatibility |
| Frontend typecheck, ESLint and production build | Passed; existing non-fatal lint warnings remain |
| Vitest | 40 files / 333 tests passed |
| Existing real-browser regression suite | 24 passed, including same-video project/job switching, telemetry/render isolation and WebGL |
| Production UI / real local models | Passed: import → inspect → assign → coverage → build/view 3D asset → reference engine → reenactment preview → pose preview/edit → enable GPEN → save/reopen → queue → pause/resume → inspect/play → download video + manifest |
| Independent real output verification | 12/12 frames; equal rational PTS; exact 12/25 s video duration; one decodable audio track; matching output SHA-256; identity and GPEN inference `VERIFIED_LOCAL`; source 3D asset reused; active restorer hash retained |
| Real output pixel checks | Maximum per-frame background mean difference 0.3333 LSB; maximum face-region mean difference 7.2284 LSB (compression included; smoke evidence, not a fidelity benchmark) |
| Local model file audit | 95/95 registered files match their recorded sizes/hashes; this does not claim inference or licence validation for all 95 models |
| Read-only legacy audit | Passed: 2,645 historical + 68,390 comprehensive immutable files, zero violations; 1,568 runtime files classified separately |
| Full Python suite | 940 passed, 10 skipped, 18 subtests passed; one Starlette/httpx deprecation warning. The skips require unavailable optional PyTorch/CUDA backends |

The real browser campaign denies every request outside loopback. This host injects requests to
`gc.kis.v2.scr.kaspersky-labs.com` independently of the application bundle. Those requests were
blocked and recorded separately; application endpoints and media stayed local. No security setting
was disabled. The completed browser campaign recorded zero page errors.

Local screenshots, exported media, manifests and machine evidence are ignored under
`app/roop_3d/.runtime/stage08_acceptance/`. They are not repository source or distributable fixtures.

### Re-run the real production journey

From `app/`, using the dedicated interpreter and two existing **owned local** FaceSets:

```powershell
$env:PYTHONPATH = (Get-Location).Path
& ./roop_3d/.venv/Scripts/python.exe -B roop_3d/tests/manual_stage08_acceptance.py --source roop_3d/outputs/facesets/<source>.fsz --target-faceset roop_3d/outputs/facesets/<target>.fsz
& ./roop_3d/.venv/Scripts/python.exe -B serve.py
```

While that production server is running, from `app/roop_3d_ui/`:

```powershell
npm run build
npm run test:e2e:offline
```

Then, from `app/`, verify the actual output pixels, timestamps, audio and model evidence:

```powershell
& ./roop_3d/.venv/Scripts/python.exe -B roop_3d/tests/manual_stage08_acceptance.py --verify-output
```

The campaign uses installed system Chrome and existing local weights. It does not download models,
mock API responses or simulate inference. Its generated clip is a short test container containing
an actual local face image; it is not evidence of long-video or multi-person visual quality.

## Known limits

* Only the accepted local reference engine is selectable for this workflow; experimental engines
  remain capability-gated. No online provider was added. Dependencies/weights and per-frame 3D
  fallback limits remain those of the accepted adapters.
* Identity-aware previews are bounded to 1,200 replayed frames / 120 seconds of processing. A later
  playhead can fail with `PREVIEW_FRAME_BUDGET` or `PREVIEW_BUDGET`; the full render is not subject
  to the preview budget. The timeline's nominal frame labels are not a VFR timestamp index.
* Full inspection retains a 900-second time limit and the existing 64-identity capacity. An inspection
  that reaches a bound is marked truncated and may omit later people. Identity-report segments and
  events retain the existing bounded storage policy; no unlimited per-frame biometric log is created.
* Workstation geometry/curves/finishing requests use the reference path. Its in-memory transport is
  pull-based, with no unbounded decoded-frame queue; lossless checkpoint images consume disk in
  proportion to video length. Resume replays earlier inference to restore tracking/temporal state.
* The 12-frame real journey exercises pose projection and a limited multiview identity asset.
  It does not establish accurate unseen geometry, long-range identity continuity, reenactment fidelity,
  performance, or acceptance on the unavailable RTX 3060. Existing 6 GB / 12 GB policies are unchanged.
  Its actual admission was the 6 GB profile with `CUDAExecutionProvider` on this host; that is not
  RTX 3060 hardware acceptance.
* GPEN ran in the real render, but its existing capability still labels preprocessing `CANDIDATE`
  and provenance/licence `UNKNOWN`. Running a model is not evidence of rights or restoration fidelity.
* The existing audio contract preserves every track: an incompatible codec or an unaligned trim can
  explicitly transcode even under `copy`. `drop` is rejected for footage containing audio. The output
  contract remains 8-bit YUV420; HDR fidelity is not established by this campaign. Odd upright dimensions
  and arbitrary-angle orientation are explicitly rejected by the existing writer.
* Media, FaceSets and asset arrays remain local referenced inputs. `.r3dp` is not a portable media
  bundle, and the existing 512 KiB manifest limit applies to masks/curves/history as well.
  The writer also retains its existing 64 KiB render-metadata budget.
* Atomic publication means success requires both video and manifest. A process death between their
  file renames can leave an unadvertised artifact on disk; it is never a successful job. A fresh retry
  can render under a new job ID. No claim of a cross-file filesystem transaction is made.
* The browser campaign performs real local imports and retains test projects/outputs for review.
  It restores the prior service configuration and does not modify legacy source or model files.
* Windows asyncio logged connection-lost callback errors after closing browser connections. The
  browser campaign itself had zero page errors and its exported output passed independent validation.

## Modified source inventory

Paths are relative to the repository root. Ignored local evidence and generated production bundles
are not staged or committed.

| File | Change |
| --- | --- |
| `README.md` | Replace the stale scaffold claim and link this workflow |
| `app/roop_3d/docs/STAGE08_OFFLINE_WORKFLOW.md` | Plan, API contracts, executed evidence and limitations |
| `app/roop_3d/engine/pipeline_runner.py` | Bind checkpoint dependencies/inspection/assets; mux progress; render manifest evidence |
| `app/roop_3d/jobs/project_store.py` | Revisioned saves, preview state, asset references, render association/history, corrupt-project recovery |
| `app/roop_3d/jobs/schema.py` | Optional project association and effective settings on durable records |
| `app/roop_3d/jobs/store.py` | Persist those associations; scope idempotency to the project |
| `app/roop_3d/jobs/worker.py` | Execute a project job with its captured settings |
| `app/roop_3d/media/video_writer.py` | Preserve/verify upright pixel aspect and record orientation policy |
| `app/roop_3d/media/schema.py` | Optional full-video inspection scope, preserving legacy defaults |
| `app/roop_3d/media/service.py` | Reuse the tracker/decode loop through EOF while retaining time/identity bounds |
| `app/roop_3d/models/catalog.py` | Shared actual-byte dependency snapshot; optional-weight absence handling |
| `app/roop_3d/server/live_preview.py` | Lock/motion pass-through, exact frame selection, VFR-safe cache selectors |
| `app/roop_3d/server/routes.py` | Completed-artifact export gate |
| `app/roop_3d/server/workstation.py` | Project update/queue routes and history-bearing project download |
| `app/roop_3d/server/workstation_preview.py` | Exact ordinals and identity/VFR tracker/compositor replay |
| `app/roop_3d/tests/test_identity_lock.py` | Explicit dependencies for the existing scripted-session contract fixture |
| `app/roop_3d/tests/test_stage05_media.py` | Non-square-pixel fixtures; test invalid runtime roots independently of the owned TEMP policy |
| `app/roop_3d/tests/test_stage08_offline_workflow.py` | New project/checkpoint/codec/cache/artifact integration regressions |
| `app/roop_3d/tests/manual_stage08_acceptance.py` | Prepare local browser media and independently verify the real output |
| `app/roop_3d_ui/src/App.tsx` | Save/update/copy/queue projects; restore complete saved editing/preview context |
| `app/roop_3d_ui/src/VideoViewport.tsx` | Persist viewport preferences and select live previews by timestamp |
| `app/roop_3d_ui/src/previewPreferences.ts` | Typed, shared viewport defaults |
| `app/roop_3d_ui/src/contracts.ts` | Additive project/preview/history types |
| `app/roop_3d_ui/src/livePreview.ts` | Timestamp selector and identity-lock/motion parameters |
| `app/roop_3d_ui/src/components/facemgr/FacesetLibrary.tsx` | Disable import until capabilities are ready |
| `app/roop_3d_ui/src/components/queue/QueueDrawer.jsx` | Authoritative server queue, legal controls, completed output inspection/export |
| `app/roop_3d_ui/src/stage8Workflow.test.tsx` | Queue/export/preview regressions |
| `app/roop_3d_ui/src/liveSwapPreview.test.tsx` | Assert the timestamp selector used by the viewport |
| `app/roop_3d_ui/src/pipelineSolo.test.tsx` | Assert timestamp selection for intermediate buffers |
| `app/roop_3d_ui/src/queueAndHud.test.tsx` | Assert explicit video/manifest export controls |
| `app/roop_3d_ui/e2e/identityLock.e2e.ts` | Project-bound render route in existing browser contract tests |
| `app/roop_3d_ui/e2e/studio.e2e.ts` | Revisioned project-save route in existing studio tests |
| `app/roop_3d_ui/e2e/swapPreviewAudit.e2e.ts` | Assert playhead timestamp selection |
| `app/roop_3d_ui/acceptance/stage8.real.ts` | Complete production-browser journey, actual APIs/models, denied external requests |
| `app/roop_3d_ui/browserPaths.ts` | Keep test profiles/temp/artifacts local and reject linked paths |
| `app/roop_3d_ui/playwright.config.ts` | Use the owned browser temp directory |
| `app/roop_3d_ui/playwright.stage8.config.ts` | Separate real production acceptance campaign |
| `app/roop_3d_ui/package.json` | `test:e2e:offline` command |
| `app/roop_3d_ui/tsconfig.json` | Typecheck acceptance/config helpers as well as the app |
