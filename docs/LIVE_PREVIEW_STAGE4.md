# Stage 4: live swap preview, cache-first serving, model-lifecycle mutex

The prompt's `app/api.py` and `react-ui/` map to `app/roop_3d/server/` and `app/roop_3d_ui/`.

## What was wrong

* The preview box only ever displayed **un-swapped** frames (`/media/preview-frame`) unless the user
  entered pipeline *solo* mode, which called the heavy `/workstation/preview` sweep endpoint.
* There was **no SVG placeholder fallback** in the server or UI (the audit found none: the only `<svg>`
  elements are the framing-guide overlay and the curve editor). The defect was the missing swap request,
  not a fake image, and the fix does not introduce any stand-in imagery: if a swap cannot be rendered the
  pane shows the previous real swap (dimmed) plus a status pill, or the plain source frame.
* Nothing coordinated preview inference with a job's model teardown/re-bind.

## Concurrency contract

| Situation | Result |
|---|---|
| Frame + configuration in the LRU cache | `200` immediately, even while Stage 1/2, a render job or a TensorRT compile owns the GPU |
| Uncached, engine owned elsewhere | `503 {"busy": true, "stage": "<stage>", "retry_after_ms": 1000}` + `Retry-After: 1`, no model load, no thread-pool slot |
| Uncached, engine free | Inference inside `EngineGate.preview_section()`, result cached |
| Start pressed during an in-flight preview | The job's worker waits for that preview to finish (and its session to be released) before it prepares/re-binds models; new previews get `503 stage=MODEL_REBIND` meanwhile so scrubbing cannot starve it |

`stage` is one of `STAGE_1_SPARSE_PRESCAN`, `STAGE_2_ACTIVE_PREPASS`, `STAGE_3_MAIN_SWAP`,
`TENSORRT_COMPILATION`, `TENSORRT_CANARY`, `MODEL_REBIND`, `RENDER_JOB` (a job is preparing),
`GPU_CAPACITY` or `PREVIEW_IN_FLIGHT`.

`EngineGate` (`server/engine_gate.py`) has two kinds of participant. *Previews* never wait. *Owners*
(render jobs, anything that tears down or re-binds models) wait for in-flight previews to drain, and
several owners may coexist (the job scheduler already bounds them). It offers sync (`owner()`) and async
(`aowner()`, which waits on a thread so the event loop stays free) entry points. `JobWorker` takes the
gate as an owner around each job; the legacy `POST /api/v1/workstation/preview` is gated the same way.

The cache is keyed by `(media_id, frame_index, swap_config_hash)`. `media_id` is the inspection id (it is
bound to the target file's SHA-256). `swap_config_hash` covers every pixel-affecting field of the payload
plus a stamp of the settings and the referenced FaceSet files, so editing a FaceSet or changing settings
cannot serve stale pixels. LRU bounds: 128 entries / 192 MiB.

## Endpoints

`/api/<x>` is served from `/api/v1/<x>` by the existing alias middleware.

* `POST /api/preview`: render one frame. Body below. Returns `image/png`; metadata is in the
  `X-Preview-Meta` JSON header (`faces` with `box`, `landmarks5`, `yaw`, `pitch` (+ = up since Stage 5), `roll`; `applied`;
  `unsupported`; `latency`), plus `X-Preview-Cache: HIT|MISS`, `X-Preview-Config-Hash`,
  `X-Preview-Frame-Index`. `?response=json` returns the metadata plus a base64 PNG instead.
* `GET /api/target/preview?config=<hash>&frame_index=N` (or `&timestamp=S`): same, for a configuration
  already POSTed (the hash comes from `X-Preview-Config-Hash`). Cache-addressable, so usable as an `<img src>`.
* `GET /api/target/preview_grid?config=<hash>&rows=3&cols=3`: contact sheet of swapped frames spread
  evenly over the clip. Uncached cells are rendered inside `budget_ms`; cells still missing are blank and
  listed in `X-Preview-Pending` / `X-Preview-Cells`. `503` only when no cell could be served.
* `GET /api/target/preview_seq?config=<hash>&start=0&count=8&step=1`: JSON manifest of consecutive frames
  (`ready` / `busy` / `pending` / `error`, each ready frame with a `url` for `/target/preview`).

### Payload (`POST /api/preview`)

Exactly one of `frame_index` / `timestamp`, and one way of choosing faces:

* `face_mappings: [{track_id, faceset_id}]` (what the UI sends), **or**
* `source_face_ids` and/or `fsz_faceset_id` (the FaceSet id returned when a `.fsz` archive is uploaded
  through `/assets/uploads`) plus `target_face_id` (`track_0000`) **or** `all_faces: true` (optionally with
  `cluster_index`, an index into the inspection's tracks).

Other fields: `identity_blend {method: slerp|linear, source_face_ids, weights, tangent_dials}`,
`frontalization_weight`, `tps_expression_weight`, `occlusion {erosion_px, feather_px}`, `mask_base64`,
`mask_strokes [{mode: paint|erase, radius, points}]` (coordinates normalised 0..1, radius as a fraction of
frame height), `restorer` (`off|gpen`), `restoration_blend`, `fidelity_weight`, `color_strength`,
`swap_weight`, `bypass`, `solo_node` (`final|detector|pose|swapper|matte|restorer|color|source`), and the
job pass-throughs `face_segments`, `parameter_curves`, `export`.

### What is applied vs. reported-only

Only controls the engine really has are applied; the rest are accepted, hashed, and **listed in
`unsupported`** in the response rather than silently ignored.

| Field | Status |
|---|---|
| `identity_blend` (2-8 FaceSets) | Applied. Weighted Slerp (Fréchet mean on the unit sphere) or Linear mean of the canonical identity; per-source tangent dials slide the result along the tangent plane. Observed pose views come from the dominant source. Several `source_face_ids` without a recipe default to equal-weight Slerp. |
| `frontalization_weight` | Applied as `pose_strength` in 2.5D→3D mesh-blend mode. |
| `occlusion`, `mask_*`, `swap_weight`, `color_strength`, `bypass`, `solo_node` | Applied. |
| `restorer: gpen` + `restoration_blend` | Applied (requires GPEN weights). |
| `restorer: codeformer/gfpgan/ultramax` | `422 RESTORER_UNAVAILABLE`: no verified adapter in the preview compositor. |
| `tps_expression_weight` | Applied since Stage 5 (TPS expression warp, see [POSE_STAGE5.md](POSE_STAGE5.md)); `pose_frontalization` (pre-swap 3D frontalization) was added there too. |
| `fidelity_weight` | Reported as `unsupported`: only CodeFormer exposes a fidelity input and it is not wired into the preview compositor. |

## Client (`app/roop_3d_ui`)

`livePreview.ts` builds the payload from the current job request + parameters, debounces 150 ms, aborts
superseded requests, and treats `503 busy` as a status: the last real image stays (dimmed), a pill reads
"GPU busy · stage 1 · sparse prescan · retrying", and the request is re-sent after `retry_after_ms`.
Queueing a render dispatches `roop3d:job-start`, which drops the in-flight preview so the worker does not
wait on it. The HUD has a **Live swap** toggle and a **PNG** button that downloads the exact PNG bytes the
server rendered through a blob URL created in the click handler, named
`roop3d_preview_<ISO-8601 UTC, colons dashed>.png`.

## Examples

```bash
curl -s -D- -o frame.png -X POST http://127.0.0.1:8030/api/preview -H 'Content-Type: application/json' -d '{
  "target_path": "roop_3d/.runtime/media/target.mp4", "inspection_id": "<32 hex>", "frame_index": 120,
  "source_face_ids": ["alice"], "target_face_id": "track_0000", "frontalization_weight": 0.6,
  "restorer": "gpen", "restoration_blend": 0.7 }'          # 503 body: {"busy":true,"stage":"...","retry_after_ms":1000}
```

```python
import json, httpx
r = httpx.post("http://127.0.0.1:8030/api/preview", json=payload)
if r.status_code == 503:
    wait = r.json()["retry_after_ms"] / 1000   # GPU busy: retry later
else:
    open("frame.png", "wb").write(r.content); meta = json.loads(r.headers["x-preview-meta"])
```

```javascript
const r = await fetch('/api/preview', { method: 'POST', body: JSON.stringify(payload), headers: { 'Content-Type': 'application/json' } });
if (r.status === 503) { const { stage, retry_after_ms } = await r.json(); /* retry after retry_after_ms */ }
else { const blob = await r.blob(); const meta = JSON.parse(r.headers.get('X-Preview-Meta')); }
```

## Verification

`app/roop_3d/tests/test_remediation_stage4.py` (Python) and `app/roop_3d_ui/src/liveSwapPreview.test.tsx`
(vitest). Inference is the deterministic fixture used by `test_workstation_preview.py`: these tests verify
the HTTP, caching and concurrency contracts, not neural output quality or GPU timing, which need a run on
the target hardware with real weights.
