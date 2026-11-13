# UI-04 — Final Workstation and Render Benchmark

**Status: PARTIAL; not a full certification.** The six-zone shell and simultaneous rendering of supported settings domains are implemented. The requested 100% cross-repository feature parity, dual-canvas GPU worker engine, and sub-16-ms preview path are not implemented or certified. Unsupported engine controls were not fabricated.

## Delivered UI changes

- `App.tsx` groups the source library, target inspection, and face mapping into Zone 2; exposes telemetry, preview, inspector, timeline, and queue as the other five zones.
- `styles.css` defines the desktop six-area grid and two-column/stacked responsive layouts.
- `SettingsInspector.tsx` renders all six API-schema setting domains together rather than hiding five behind category tabs. Supported groups and the model list default open. Studio inspector groups and the face tray also default open.
- Settings remain schema/capability-driven. No sliders were added for settings the backend does not implement.

At a 1920×1080 mock-app viewport, browser inspection found all six named zones mounted and not hidden. The full-page capture was 1922×1182. Inspector and dock contents scroll internally; this proves simultaneous zone presence, not that every individual control fits above the fold. Existing dialogs and route-specific surfaces were not all removed.

## Verification results

| Check | Result |
|---|---|
| Focused workstation tests | Pass: 25/25 tests across `settingsVisibility`, `appShell`, and `stage7Studio`. |
| React telemetry / binary stream tests | Pass: 16/16 across `stage11Perf` and `liveStream`. |
| Complete frontend unit suite | Pass: 86 files, 880 tests. |
| TypeScript typecheck | Pass. |
| ESLint on changed frontend files | Pass. |
| Mock browser visual inspection | Six zones visible at 1920×1080. |
| Mock browser animation cadence | 59 RAF callbacks/s over a 1.016 s sample (61 frames); this is not preview or sustained application FPS. |
| Telemetry E2E pacing | During the 30 Hz mock stream: `#/faceswap` 243 RAF frames, median 16.7 ms / p95 33.3 ms; `#/facemgr` 257 frames, median/p95 16.7 ms; `#/settings` 261 frames, median/p95 16.7 ms. Presentation cadence only, not inference FPS. |
| React rerender test | Render-delta assertions passed with 0 measured commits for the covered consumers at 30 Hz on all three routes; the E2E test later failed because its mock WebSocket routes were already closing. The Vitest test also passed 90 ticks. This does not establish behavior for every component or at 60–120 Hz. |
| Local model verification | Pass: 95/95 expected records checked; no violations or unavailable scopes. The verifier also found the pinned FLAME artifact and MakeHuman CC0 asset locally. |
| Full backend pytest suite | Completed: 2,872 passed, 47 skipped, 13 subtests passed, 1 failed. The CPU-job failure was isolated to the HUD sampler querying GPU telemetry unconditionally; the regression test now passes after per-job GPU sampling was gated. The complete suite was not rerun. |
| Focused backend regression and audit tests | Pass: 106 tests across reliability harness, terminal telemetry, GPU preview/profile fixes, and W2 isolation audit tests. |
| Isolation audit | Pass: all five checks; 661 files scanned, with no sibling imports/writes, interpreter, origin, or tracked-artifact violations. |
| Mock-backed browser E2E suite | Full run: 80/102 passed; 22 failed before correcting E2E expectations for open-by-default drawers, named panel regions, and the Mask-feather slider role. A targeted post-correction rerun passed 4/4 affected studio/morph flows. Other full-run failures remain unresolved; the full suite was not rerun after these test updates. |

The browser cadence sample was collected on the mock app, without target media or a real inference pipeline. No p50/p95/p99 end-to-end preview latency, slider-to-visible-frame latency, VRAM, GPU, or 4K benchmark was measured.

## Existing preview path and performance limits

The source audit in [UI-03](UI_03_ALL_VISIBLE_LAYOUT_AND_FAST_RENDER_SPEC.md) records the verified existing contracts:

- `POST /api/preview` returns binary PNG by default; optional JSON/Base64 is not the default.
- `/ws/v1/preview/stream` sends binary JPEG frames with a 16-byte little-endian header and latest-wins delivery. `/ws/v1/pipeline/status` remains JSON.
- The live single-frame request path waits 150 ms after slider activity before issuing a request. That debounce alone precludes a <16 ms slider-to-preview claim for that path.
- The existing backend cache stores bounded, complete CPU-side preview frames under exact frame/configuration keys. It is not a reusable GPU intermediate-stage cache.
- Existing telemetry stores and imperative `ImageBitmap` canvas presentation avoid React commits in the covered high-frequency path.

No code was added for `OffscreenCanvas` worker-owned split/mesh canvases, a typed-array telemetry ring, binary frames on the pipeline-status socket, pinned-memory/VRAM Stage 1–3 caching, or stage invalidation. No safe reusable stage interface or key contract exists in the current backend. The current spatial inspector was not replaced by a second always-live mesh canvas. No 60–120 FPS or <16 ms claim is justified.

## Audit provenance and coverage

- [UI-01](UI_01_ROOP_ULTIMATE_COMPLETE_CONTROL_INVENTORY.md) records the reference repository at commit `71ebb65c5fa5f11a72e343e8fc4f285bf591b555`, the generated UI settings key catalog, routes, major control groups, and render paths. The mounted `rea` GitHub-source tool was unavailable; the public GitHub raw-file endpoint and commit API were used. The inventory explicitly does not reconstruct every default/range or claim every upstream JSX line was reopened in this run.
- [UI-02](UI_02_ROOP_3D_COMPLETE_CONTROL_INVENTORY.md) records the local settings/API contract, existing UI surfaces, and preview bottlenecks.
- [UI-03](UI_03_ALL_VISIBLE_LAYOUT_AND_FAST_RENDER_SPEC.md) defines the all-visible acceptance meaning and preserves the distinction between supported controls and requested-but-nonexistent backend features.
- No reference repository files were modified. The requested named five-agent/model/effort configuration was not available through the exposed agent tooling.

## Unmet acceptance criteria

- Every reference-repository feature is not present in this app, and unsupported controls were not represented by inert UI.
- All six zones are present, but not every individual setting, route, diagnostics row, or panel item fits in the visible viewport simultaneously; inner panels scroll.
- No new 14-section diagnostics HUD, virtualized 10,000-item gallery/log, 10-state queue, or source-to-target batch matrix was added.
- No real 3D identity-mesh viewport, WebGL2/OffscreenCanvas worker rendering, typed-array ring buffer, GPU stage cache, or sub-16-ms slider preview was delivered or benchmarked.
- Verification status: the full backend run had one failure that is fixed and verified by a focused test, but the full suite was not rerun. The isolation audit passes. The full browser E2E run was not clean; this remains a partial implementation record, not proof that all requested checks passed.
