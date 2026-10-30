# Roop 3D

An independent Pinokio application with its own source checkout, Git repository,
virtual environment, launchers, cache, logs and process. It does not launch or
import Roop Ultimate. The current release is the Stage 01 local service scaffold;
face replacement and volumetric inference are later implementation stages.

## Open the app

In Pinokio Home / Installed, choose **Roop 3D** (cube icon), then **Start Roop 3D**.
The app's own sidebar changes to **Open Roop 3D** when health polling succeeds.
Use the app's Files / Ask AI workspace for subsequent stages: this checkout is
`<PINOKIO_HOME>/api/roop-3d`, not the Roop Ultimate directory.

The API and local page default to **http://127.0.0.1:8030/**. Development Vite uses
port **5174** only. A busy API port produces an actionable diagnostic and never
kills its owner. Set `ROOP3D_PORT` in this app's environment if needed.

## Independence contract

- Code lives in `app/roop_3d/`; future UI code lives in `app/roop_3d_ui/`.
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
There are no remote-provider or archive-import endpoints in this stage. Secrets
remain in the environment/OS secret stores, never project manifests or logs.

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
