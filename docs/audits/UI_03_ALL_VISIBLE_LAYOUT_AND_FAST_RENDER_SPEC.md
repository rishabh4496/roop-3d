# UI-03 — All-Visible Workstation Layout and Fast-Render Specification

## Scope and acceptance meaning

This spec uses controls and contracts verified in [UI-01](UI_01_ROOP_ULTIMATE_COMPLETE_CONTROL_INVENTORY.md) and [UI-02](UI_02_ROOP_3D_COMPLETE_CONTROL_INVENTORY.md). It is a **roop-3d** implementation plan; the reference repository is read-only and remains independent.

“All visible” means the six primary workstation zones and all supported setting groups are present in the default Face Swap workspace, with setting groups expanded and no category tab required to discover a group. It does **not** mean every individual control, all routes, or every diagnostic report row can fit above the fold on every monitor: that is geometrically impossible on finite displays. Panels may scroll internally, and the compact responsive layout must preserve access at narrow sizes. Controls unsupported by the backend are explicitly absent, rather than represented by inert sliders.

## Six-zone default workstation

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ Z1  persistent command / project / connection / provider / telemetry ribbon   │
├──────────────────┬──────────────────────────────────────┬────────────────────┤
│ Z2  source,       │ Z3  interactive video preview        │ Z4  inspector      │
│ FaceSet, target,  │     wipe / overlays / mask tools      │     all supported │
│ track mapping     │     3D spatial viewer remains an      │     groups open   │
│                   │     inspector surface                 │                    │
├──────────────────┼──────────────────────────────────────┼────────────────────┤
│                  │ Z5  timeline / trim / face and scene lanes │ Z6 queue / logs │
└──────────────────┴──────────────────────────────────────┴────────────────────┘
```

| Zone | Existing source and planned default composition | Visibility rule |
|---|---|---|
| Z1 — Telemetry / command | Persistent `TopBar`, `TelemetryHeader`, connection, hardware/provider and pipeline badges. Existing telemetry store remains the source; do not poll/re-render the whole application on every frame. | Stays outside route panels and visible in the workstation shell. |
| Z2 — Source / target dock | Existing FaceSet library and pose coverage; target media/inspection; selected target face, identity lock and per-track mappings, grouped in one left dock rather than three independent grid tracks. | Source, target and mapping sections expanded; contents may scroll within the dock. |
| Z3 — Preview / spatial view | Existing WebGL2 2D preview canvas, wipe/compare, overlays and mask tools. Keep existing spatial 3D inspection functional; do not claim it is a second always-live mesh canvas unless one is actually mounted. | Main preview occupies the largest cell. 3D inspector stays reachable without a modal; rendering fallback remains supported. |
| Z4 — Pipeline inspector | Existing swap, pose, occlusion, restorer/color/relight and configuration controls. Replace six mutually exclusive schema tabs with six simultaneously rendered schema groups. Keep the supported parameter domain authoritative to the API schema. | Every section defaults expanded; narrow panel scroll is allowed. |
| Z5 — Timeline | Existing track lanes, frame navigation, timeline dock, trim and keyframe/scene data. | Persistent lower-center zone; its nested face tray defaults expanded. |
| Z6 — Queue / terminal | Existing persistent queue and job actions, with live log/diagnostics surfaces where currently supported. Do not fabricate 10 queue lifecycle states if the backend reports a different contract. | Persistent lower-right zone; independent scrolling. |

The full 14-section runtime report, Model Hub, and installer remain distinct route/drawer experiences. Adding all their rows into the settings inspector would duplicate existing screens and overload the narrow dock; the default shell must keep diagnostics one action away and report controls/sections truthfully.

## Current layout deltas

- The present studio grid uses three side-by-side left areas (source, target, mapping), preview, and inspector; timeline and queue are combined below it. This is not a six-area grid.
- Settings domain tabs render only the selected section; an `advanced-diagnostics` `<details>` drawer and several nested inspector groups also start closed.
- A 3D spatial inspector already exists, but it is nested with the inspector rather than rendered as an equal-sized second center canvas. Do not duplicate or detach its mesh/resources without preserving the existing viewer lifecycle and state.
- Breakpoints currently collapse into two columns and then one. Small screens cannot simultaneously show six panes; responsive overflow/scroll remains an explicit exception to desktop visibility.

## Fast preview: reuse existing contracts first

### Existing fast path

1. Frame requests are coalesced and stale requests aborted in `livePreview.ts`; the client currently waits 150 ms after slider activity.
2. `/api/preview` sends a binary PNG by default. JSON/Base64 is optional; keep binary as the default and do not introduce an extra encode/decode conversion.
3. `/ws/v1/preview/stream` already sends binary JPEG with a 16-byte little-endian frame/latency header and latest-wins delivery. `/ws/v1/pipeline/status` is JSON and remains a status channel.
4. `LiveStreamCanvas` decodes with `createImageBitmap` and paints imperatively, avoiding React commits per frame; the existing zero-rerender test is the baseline contract.
5. Backend `PreviewCache` is bounded and stores complete CPU-side preview frames under the exact configuration/frame key. It is not a valid cache for partially changed settings.

### Safe incremental behavior

- Keep high-frequency telemetry in existing external stores and selectors. Introduce typed-array/ring storage only when a measured allocation/notification profile shows a benefit and consumers can define overwrite, snapshot, and memory ownership semantics.
- Preserve latest-wins, cancellation, backpressure, object-URL/bitmap disposal and authenticated same-origin local transport. Do not render Base64 frame payloads in a component tree.
- Retain exact full-frame cache keys. Slider changes that affect final pixels must change the cache key; no stale cross-configuration frame may be returned.
- Only add worker-owned OffscreenCanvas after browser support/fallback, resize/context-loss, worker disposal, WebGL context budgeting, pixel correctness and integration tests are specified. Keep the main-thread renderer as fallback.
- Stage cache keys must include source media digest, frame timestamp/selector, detection/tracker model identity, FaceSet/mesh identity, every upstream parameter affecting the stage, device/provider/precision, and code/model version. Entries must be bounded and own explicit CPU/GPU resources. GPU eviction/disposal and concurrent preview/render admission must be tested before shipping.

### Explicit non-goals until measured and implemented

The current evidence does **not** justify claims of WebGL2 worker rendering, zero-copy GPU frame transfer, typed-array telemetry rings, sub-16-ms full inference, or the proposed 3-tier stage cache. The requested stages are meaningful only if measured pipeline boundaries actually permit reuse. Current live previews can run detection/pose, mesh/swap, restoration and composition; no production API provides reusable Stage 1/2/3 GPU handles. A safe patch must not invent these claims or add UI toggles that the engine ignores.

## Verification contract

- DOM tests: all six schema domains and supported controls exist in the default inspector without switching a tab; no required group is `hidden` or collapsed by default. A separate shell/workspace test checks six named zones in desktop mode and preserves responsive behavior.
- Existing preview contracts: binary HTTP remains binary by default; binary socket header and latest-wins behavior remain unchanged; telemetry stream causes zero React commits in the already-covered consumers.
- Backend: existing exact-frame/full-config cache behavior and preview admission tests remain green. No stage-cache acceptance claim without a dedicated correctness/eviction test and a real measured benchmark.
- Visual proof: launch the application or production UI at a desktop viewport; inspect the six visible zone labels and ensure no overlap/overflow hides their controls. Record the viewport and browser. This does not establish 60–120 FPS.
- Performance report: separate canvas presentation FPS, UI interaction latency, preview request latency, and end-to-end inference latency. Include hardware, media/frame, warm/cold status, sample count and p50/p95/p99. Do not substitute an animation loop's FPS for preview latency.
