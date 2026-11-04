# Roop 3D

**New to Roop 3D?** Start with the [installation guide](docs/INSTALLATION_GUIDE.md) and the [user manual](docs/USER_MANUAL.md); in the app, press **Guide**. All guides are indexed in [docs/README.md](docs/README.md); problems: [common problems and solutions](docs/COMMON_PROBLEMS.md).

Post-production controls are available in **Extras** (`#/extras`): compositing
passes, FCPXML/EDL/OTIO interchange, LUT/grain finishing, and optional provenance.
See [Post-production and acceptance limits](docs/POST_PRODUCTION.md) for formats,
requirements, local endpoints, and the distinction between estimated geometry,
integrity hashes, and cryptographic signatures.

An independent Pinokio application with its own source checkout, Git repository,
virtual environment, launchers, cache, logs and process. It does not launch or
import Roop Ultimate. The workstation supports local FaceSet import, target inspection,
3D identity assets, frame preview, project save/reopen, resumable queued rendering,
and video/manifest export. See the [Stage 8 offline workflow](app/roop_3d/docs/STAGE08_OFFLINE_WORKFLOW.md)
for the real execution path, acceptance evidence, dependency gates and limitations.

## Open the app

In Pinokio Home / Installed, choose **Roop 3D** (cube icon), then **Start Roop 3D**.
The app's own sidebar changes to **Open Roop 3D** when health polling succeeds.
Use the app's Files / Ask AI workspace for subsequent stages: this checkout is
`<PINOKIO_HOME>/api/roop-3d`, not the Roop Ultimate directory.

The API and local page default to **http://127.0.0.1:8030/**. Development Vite uses
port **5174** only. A busy API port produces an actionable diagnostic and never
kills its owner. Set `ROOP3D_PORT` in this app's environment if needed.

## Independence contract

- Code lives in `app/roop_3d/`; workstation UI code lives in `app/roop_3d_ui/`.
- Python lives in `app/roop_3d/.venv/`; dependency distributions were freshly
  installed from the hash-pinned lock, with copy mode and this app's own cache.
- This is a separate Git repository: `rishabh4496/roop-3d`. Its initial commit does
  not inherit Roop Ultimate's repository history or share Git objects/worktrees.
- Git hooks reject a push to any other repository, including Roop Ultimate.
- Install/update/reset/start target only this checkout. Reset stops this app's
  `start.js` and removes only its venv/readiness marker, preserving source and data.
- Roop Ultimate's existing weights may be read directly through the hashed model
  registry. They are read-only inputs; no model weights were copied or downloaded.
- Existing files under Roop Ultimate are historical source material only. Later
  development must happen in this app. Its source, settings, launchers, environment
  and Git state must never be changed by Roop 3D tooling.

## Install, update and reset

Pinokio provides Install/Reinstall, Start, Update and Reset Environment entries.
Install creates this app's dedicated environment and syncs the pinned lock.
Update pulls this repository with `--ff-only` and installs its dependencies.
Reset verifies all deletion targets stay under this checkout before removing the
venv. No application-wide kill or sibling-app reset is performed.

Core dependencies include FastAPI, Uvicorn, Pydantic v2, PyAV/FFmpeg, NumPy, OpenCV,
ONNX Runtime, PyYAML and Httpx. Pytest/ONNX inspection tools are included in the
development lock. Heavy diffusion/TRT/FLAME extras are separate, not base imports.

## API

- `GET /api/v1/health`: version, uptime, standalone app ID and the last audit status.
- `GET /api/v1/system/hardware`: live OS/CPU/NVIDIA/provider probes or explicit unknowns.
- `GET /openapi.json`: local endpoint schema.

JavaScript:

```javascript
const state = await fetch('http://127.0.0.1:8030/api/v1/health').then(r => r.json());
console.log(state);
```

Python:

```python
import httpx
print(httpx.get('http://127.0.0.1:8030/api/v1/system/hardware').json())
```

Curl:

```shell
curl http://127.0.0.1:8030/api/v1/health
```

The host is restricted to 127.0.0.1. Request bodies are bounded (default 2 MiB);
origins/Host headers are checked, and errors use JSON envelopes with request IDs.
`ROOP3D_DEV_CORS=1` enables only the development origin `http://127.0.0.1:5174`.
There is no remote-provider adapter. FaceSet archives use validated local upload
endpoints. Secrets remain in environment/OS secret stores, never project manifests or logs.

### Identity panel API (Stage 7)

Local-only additions used by the workstation's identity panel (see
`app/roop_3d/docs/STAGE07_WORKSTATION_UI.md`):

- `GET /api/v1/facesets/{id}/views/{index}/image?size=160`: bounded JPEG of one stored view (no embeddings).
- `GET /api/v1/facesets/{id}/identity-asset`: pose-coverage readiness plus the cached 3D identity asset
  manifest, or `state: "not_built"`. It never builds.
- `POST /api/v1/facesets/{id}/identity-asset`: explicitly build (or reuse) the asset on the CPU; the same
  asset is reused by `pose_3d` / `reenactment_3d` renders. 409 while a build of that FaceSet is running.
- `GET /api/v1/facesets/{id}/identity-asset/mesh`: the built asset as a viewer mesh (glTF frame, observed
  atlas with unseen texels grey); 404 until built.
- `POST /api/preview` additionally accepts `render_mode` (`standard|pose_3d|reenactment_3d`) and
  `occlusion_mode` (`standard|robust`); omitted values keep the previous behaviour.

```shell
curl -X POST http://127.0.0.1:8030/api/v1/facesets/<faceset_id>/identity-asset
```

```python
import httpx
status = httpx.get('http://127.0.0.1:8030/api/v1/facesets/<faceset_id>/identity-asset').json()
print(status['state'], status['coverage']['readiness'])
```

```javascript
const mesh = await fetch('/api/v1/facesets/<faceset_id>/identity-asset/mesh').then(r => r.json());
console.log(mesh.vertices.length, mesh.provenance);
```

### Single origin, aliases and telemetry

The built workstation (`app/roop_3d_ui/dist`) is served by the same process, port and origin as
the API, so the production UI needs no proxy or CORS. Any `/api/<path>` also resolves to
`/api/v1/<path>`. On loopback, a request whose `Origin` is not this service (`127.0.0.1` or
`localhost` on the service port) gets HTTP 403 `ORIGIN_DENIED`; WebSockets are closed with 1008.

- `GET /api/meta`: `{"mock": false, "version", "api_versions", "share_mode"}`. The dev mock server answers `{"mock": true}`.
- `GET /api/system/hardware` and `GET /api/system/telemetry` (the latter adds `pipeline`): besides the
  full probe, the active GPU as `gpu_name`, `compute_capability`, `vram_total_gb`, `vram_available_gb`,
  `cuda_version`, `tensorrt_version`, `onnxruntime_providers`, `tensor_core_precisions`,
  `nvdec_available`, `nvenc_available`. Unknown stays `null`. Tensor Core precisions are inferred from
  the compute capability, and NVDEC/NVENC mean FFmpeg registered the codecs; neither proves a model or
  codec actually runs.
- `WS /ws/telemetry`: `job.state`, `job.progress`, `system.telemetry` and `pipeline.status` events.
  `WS /ws/v1/pipeline/status`: only `pipeline.status` frames, starting with a snapshot.

A `pipeline.status` frame: `stage` (`STAGE_1_SPARSE_PRESCAN` | `STAGE_2_ACTIVE_PREPASS` |
`STAGE_3_MAIN_SWAP` | null), `stages` (`PENDING` | `RUNNING` | `DONE` | `SKIPPED` | `FAILED` | `RESTORED`),
`stage_progress_pct`, `active_ranges_count`, `verified_target_frames`, `skipped_frames_pct`,
`eta_seconds`, `fps`, and `engine_state` (`IDLE` | `COMPILING_TRT` | `RUNNING_CANARY`) with
`engine_label`, `engine_model`, `engine_precision`, `engine_since`. Streaming jobs report all three
stages and their measured policy counts; sparse absence estimates are not proof of no faces.
Reference jobs retain stage-1/2 `SKIPPED` and unmeasured quantities remain `null`.
Engine fields are driven by the TensorRT builder observers.

```javascript
const ws = new WebSocket(`ws://${location.host}/ws/v1/pipeline/status`);
ws.onmessage = e => { const f = JSON.parse(e.data); if (f.type === 'pipeline.status') console.log(f.stage, f.stage_progress_pct, f.eta_seconds); };
```

```python
import httpx
hw = httpx.get('http://127.0.0.1:8030/api/system/hardware').json()
print(hw['gpu_name'], hw['vram_available_gb'], hw['tensor_core_precisions'])
```

```shell
curl http://127.0.0.1:8030/api/system/telemetry
```

### Sharing on the network (`--server_share`)

`python -B app/serve.py --server_share` (or `ROOP3D_SERVER_SHARE=1`) listens on all interfaces and
prints, on this terminal only, a per-launch random token with ready-to-open URLs. Remote peers must
send it as `Authorization: Bearer <token>`, `?token=<token>`, or the HTTP-only `roop3d_token`
cookie that opening `/?token=<token>` sets; their `Origin`, when present, must equal their `Host`.
Peers on this machine (loopback) need no token and keep the strict loopback rules. Traffic is
plain HTTP, so share only on a network you trust. Leave the flag off for normal use.

### UI development without a GPU

`npm run dev:mock` in `app/roop_3d_ui` starts an Express mock backend (`mock-server/server.ts`,
every response has `X-Mock-Server: true`) and Vite on 5174 proxied to it. Node `^20.19 || >=22.12`;
`npm ci` is the supported install, and the Inter font is bundled so the UI is fully offline.

## Validate and continue

From this app's `app/` directory on Windows:

```powershell
& ./roop_3d/.venv/Scripts/python.exe -B -m pytest -c roop_3d/pyproject.toml --confcutdir=roop_3d roop_3d/tests -q
& ./roop_3d/.venv/Scripts/python.exe -B roop_3d/audit/local_models.py --verify
& ./roop_3d/.venv/Scripts/python.exe -B roop_3d/audit/verify_isolation.py --report
```

On Unix use `.venv/bin/python`. The tests run offline without weights. Model
verification reads the 95-file local registry without executing inference. The
legacy integrity gate preserves the original source/model hashes and records
explicit runtime activity separately; its detailed machine-local report is ignored.
The base server can boot even when a new machine has no completed audit report.

Read [AGENTS.md](AGENTS.md) and [CONTRIBUTING.md](CONTRIBUTING.md) for future stages.
Historical Stage 00/01 records under the Python audit package predate this split;
current separation evidence is in [docs/STANDALONE_REPORT.md](docs/STANDALONE_REPORT.md).

Stage 2 TensorRT provisioning, verified dynamic ONNX conversion, strict sessions,
compilation telemetry and prebuild/benchmark commands are documented in
[docs/TRT_STAGE2.md](docs/TRT_STAGE2.md). These tools preserve the existing
reference inference paths; GPU performance and production-model parity require
separate hardware runs.

Stage 3 streaming, numeric prepass checkpoints, the zero-pickle frame ring,
CUDA I/O binding, NVDEC/NVENC, restorer contracts and thread benchmark routes are
documented in [docs/STREAM_STAGE3.md](docs/STREAM_STAGE3.md).

Stage 4 live swap preview (`POST /api/preview`, `GET /api/target/preview{,_grid,_seq}`), the cache-first /
HTTP 503 busy contract and the preview-vs-Start model mutex are documented in
[docs/LIVE_PREVIEW_STAGE4.md](docs/LIVE_PREVIEW_STAGE4.md).

Stage 5 (unified `Pose3D` and 5-point estimator, the Stage 0 3x3 pose matrix and identity guard, 3D pose
frontalization, TPS expression warping and `POST /api/faces/extract_from_frame`) is documented in
[docs/POSE_STAGE5.md](docs/POSE_STAGE5.md).

Stage 6 (nine deep-linked routes, the 3-step run-bar pill, full-bleed comparison canvas, pose HUD, on-canvas
"Use Face From Frame" menu, six-node buffer soloing, self-documenting settings and the 3×3 sweep) is documented in
[docs/WORKSTATION_STAGE6.md](docs/WORKSTATION_STAGE6.md).

Stage 8 (Stage 25 parameter curves, timestamped One-Euro stabilization, CodeFormer fidelity, animated Slerp, durable person banks and the 3x3 archive studio) is documented in [docs/REMEDIATION_STAGE8.md](docs/REMEDIATION_STAGE8.md).

Stage 9 (Unified Adaptive CPU/GPU Runtime, RTX 4070 Desktop 12 GB vs RTX 3060 Laptop 6 GB profiles, INSwapper FP32 precision contract, warm session residency, sequential VRAM eviction, and 10-category benchmarks) is documented in [app/roop_3d/docs/STAGE09_OPTIMIZATION.md](app/roop_3d/docs/STAGE09_OPTIMIZATION.md). Benchmark metrics vs Stage 2 baseline are recorded in [app/roop_3d/audit/STAGE09_BENCHMARKS.json](app/roop_3d/audit/STAGE09_BENCHMARKS.json).

Stage 14 optional NVIDIA codec/CV-CUDA adapters, temporal flow propagation and
explicit FP8 admission are documented in [docs/REMEDIATION_STAGE14.md](docs/REMEDIATION_STAGE14.md).
Native CV-CUDA/TensorRT hardware acceptance remains pending.
