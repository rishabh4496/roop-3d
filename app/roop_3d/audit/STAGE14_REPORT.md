# Stage 14: Video Workstation, Face Picker, Queue, and Projects

## Result

Required Stage 14 automated gates pass. The workstation is built in
`app/roop_3d_ui/` and served by FastAPI from the API origin after `npm run build`.
No legacy UI, source, launcher, setting, environment, model artifact, or preserved
audit baseline was modified. No directories were created outside the owned
Roop 3D package/UI trees.

Visual browser review is `BLOCKED`: the computer-use native pipe remained unavailable
after retry/reset, and the browser access host was unavailable. No screenshot or
interactive browser rendering is claimed. Actual HTTP assets, capability/config
responses, range serving, and WebSocket connectivity were verified separately.

## Workstation

- Responsive dark/light layout, target/source asset library, linked source/output
  playback, dual/wipe comparison, decoded-frame previews, timeline scrubbing,
  bounded sampled keyframes, and export trim bounds. Selecting another job opens
  its matching target; unrelated targets/outputs are never silently compared.
- Bounded chunk uploads use the advertised request limit, sequential offsets, an
  8 GiB video limit, and a 256 MiB FaceSet archive limit. Failed/interrupted uploads
  can discard their own partial files. The existing Stage 04 archive validator and
  reader validate FaceSets before registration; raw embeddings are not returned by
  the coverage endpoint.
- Target inspection shows actual track thumbnails, frame/time spans, sampling
  warnings, and estimated continuity. Each track can keep its original identity or
  map to a loaded FaceSet. Invalid/stale track/source IDs are excluded from payloads.
- Settings are generated from the Stage 02 JSON schema and current config. Runtime
  capabilities provide unavailable-option reasons, infrastructure read-only fields,
  feature gates, and legal job actions. Optional diffusion, unsupported restorers,
  Wav2Lip, unsupported precision/provider paths, and remote processing are disabled
  with reasons. The audio inspector respects the existing source-audio preservation
  policy instead of promising unsupported track removal.
- Queue and telemetry use the real event stream, bounded reconnect backoff, sequence
  gap detection, and HTTP queue refresh after ready/gap/reconnect. All ten job states
  are covered by Vitest. FPS, frame counts, GPU free/total memory, and CPU core counts
  are shown only from backend data; CPU utilization is not invented.
- Frame preview requests are debounced and aborted when superseded. Every temporary
  preview ObjectURL is revoked on replacement/unmount. Videos stream via range-aware
  FileResponse with metadata-only preload. Three.js controls, geometry, materials,
  textures, renderer/context, resize observer, and animation callbacks are disposed.

## Honest geometry inspection

The default Three.js scene contains orientation axes and a reference coverage
hemisphere with actual FaceSet yaw/pitch observations. It contains no placeholder
head. Estimated track pose and 2.5D data are labeled as estimates.

Actual Stage 08 artifacts can be inspected through bounded, source-bound receipts
under `roop_3d/.runtime/reconstructions/<32 lowercase hex characters>.json`:

```json
{
  "method": "tps_2_5d",
  "coordinate_system": "opencv",
  "source_sha256": "<actual target SHA-256>",
  "mesh_path": "roop_3d/.runtime/reconstructions/actual.obj",
  "mesh_sha256": "<actual OBJ SHA-256>",
  "uv_path": "roop_3d/.runtime/reconstructions/actual.png",
  "uv_sha256": "<actual texture SHA-256>"
}
```

`method` accepts `flame` or `tps_2_5d`; coordinates accept `opencv` or `opengl`.
Texture fields are optional. The endpoint verifies the target/artifact hashes,
owned paths, byte limits, geometry, and image dimensions before returning data.
The frontend converts OpenCV coordinates explicitly, honors face UV indices, and
labels imported FLAME versus approximate geometry. Imported provenance is
`USER_SUPPLIED_ARTIFACT`, not independently verified FLAME fitting or licensing.
No reconstruction is generated merely to populate the viewport.

Three.js disposal follows its [official resource guide](https://threejs.org/manual/pages/how-to-dispose-of-objects.html).
Build configuration follows the [Vite guide](https://vite.dev/guide/). Package
versions and integrity records are pinned in the dedicated `package-lock.json`;
no root lockfile or Python dependency changes were required.

## Projects and checkpoint continuation

Version 1 `.r3dp` manifests contain owned target/inspection/FaceSet paths and SHA-256
digests, trim bounds, track mappings, selected reference-chain model identities,
effective typed settings, an optional checkpoint job reference, and the zero-based
last completed frame. They contain no raw media, thumbnails, embeddings, credentials,
or provider tokens. Import requires the referenced inputs to exist locally at their
saved paths; the project is not a media archive.

Save/open/resume verify input identities; model artifacts are rehashed at save and
open. Resume also checks effective settings, checkpoint job/request/input identity,
completed-frame continuity, and the existence of completed checkpoint frames before
queueing the paused/interrupted job. Changed inputs/models/settings block continuation
and preserve the manifest. An uncheckpointed project can be opened and queued anew.

Pipeline checkpoints now retain settings and FaceSet digests. Trim bounds reach the
existing rational-timestamp writer. Unset trim fields are omitted from canonical
request hashing, preserving earlier untrimmed job/checkpoint/idempotency identities.
The Stage 06 reference inference path remains intact.

## Production serving and runtime evidence

`server/api.py` mounts the built UI at `/` after API routes, supports known
workstation/project SPA paths, and keeps arbitrary unknown/API/asset paths at 404.
No Vite process is needed for production. Missing builds show local build instructions.
Vite development remains bound to 127.0.0.1:5174 and requires the existing explicit
development-origin configuration on the backend.

Live origin check at `http://127.0.0.1:8030`: `VERIFIED_LOCAL`. The actual backend
served the HTML and both initial hashed assets, advertised its worker, returned
config/capabilities, preserved API 404 behavior, and emitted `stream.ready` over
WebSocket. The verification server was then stopped through its own execution
session; no other port owner was stopped. No real-media face replacement, FLAME
fit, inference quality, or throughput benchmark was performed by this stage.

## Verification commands and final results

Frontend commands ran from `G:\pinokio\api\roop-3d\app\roop_3d_ui` with Node 24.18.0
and npm 11.16.0. npm cache writes were restricted to `.cache/npm` inside that directory.
Python commands ran from `G:\pinokio\api\roop-3d\app` using the dedicated venv.

| Command | Exit | Final result |
| --- | ---: | --- |
| `npm ci --cache .cache/npm --no-audit --no-fund` | 0 | Clean install from dedicated lockfile |
| `npm run typecheck` | 0 | Strict TypeScript check |
| `npm test` | 0 | 18 Vitest tests passed |
| `npm run build` | 0 | Production HTML/CSS/JS and lazy Three.js chunks built without warnings |
| `roop_3d\.venv\Scripts\python.exe -B -m pytest -c roop_3d\pyproject.toml --confcutdir=roop_3d roop_3d\tests\test_stage14_workstation.py -q` | 0 | 15 Stage 14 tests passed |
| `roop_3d\.venv\Scripts\python.exe -B -m pytest -c roop_3d\pyproject.toml --confcutdir=roop_3d roop_3d\tests\test_stage01_scaffold.py::test_error_envelopes_and_body_limit roop_3d\tests\test_stage14_workstation.py -q` | 0 | 16 compatibility/integration tests passed |
| `roop_3d\.venv\Scripts\python.exe -B -m pytest -c roop_3d\pyproject.toml --confcutdir=roop_3d roop_3d\tests -q` | 0 | 390 passed, 18 subtests passed; one existing Starlette TestClient deprecation warning |
| `roop_3d\.venv\Scripts\python.exe -B roop_3d\audit\verify_isolation.py` | 0 | Historical 2,645/2,645 and immutable comprehensive 68,390/68,390; zero violations; runtime activity separately classified by the unchanged audit policy |

Earlier attempts found a Vite chunk-option type mismatch, an unnecessary CSS import,
a Windows static-path separator issue, an incorrect FaceSet pose fixture, and an
overly broad SPA fallback. These were corrected. The first full Python run had
389 passes and one 404-contract failure; the final full repeat passed all 390.
No assertion was weakened or core unit mocked to suppress a failure. Project tests
use small real hashed fixture weights; trim/upload/compositor-format checks use
real media/archive handling. They do not claim actual model inference.

## Files changed

- `jobs/project_store.py`, `jobs/schema.py`, `jobs/store.py`
- `engine/pipeline_runner.py`
- `server/api.py`, `server/app.py`, `server/contracts.py`, `server/routes.py`, `server/workstation.py`
- `tests/test_stage14_workstation.py`, `audit/STAGE14_REPORT.md`
- `../roop_3d_ui/package.json`, `package-lock.json`, `tsconfig.json`, `vite.config.ts`, `index.html`
- `../roop_3d_ui/src/App.tsx`, `SettingsInspector.tsx`, `VideoViewport.tsx`, `ThreeViewport.tsx`
- `../roop_3d_ui/src/api.ts`, `contracts.ts`, `events.ts`, `state.ts`, `main.tsx`, `styles.css`, `workstation.test.tsx`

Required acceptance: pass, zero final failures, zero blocked required checks.
Optional visual browser review: blocked by unavailable automation helpers. Manual
visual review and a real-media/hardware rendering campaign remain separate evidence
gates; no visual fidelity or performance claim is made here.
