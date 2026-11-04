# Installation guide

Roop 3D is a Pinokio app. Everything it needs (its own Python environment, its own web interface build, its own caches) is created
inside its own folder, `<Pinokio home>/api/roop-3d`. It never touches other Pinokio apps.

> This guide describes what the launcher scripts (`install.js`, `start.js`, `update.js`, `reset.js`) and `tools/project_admin.py`
> really do. If something differs on your machine, the Settings → *Environment & Offline Manager* screen is the authority: it
> runs the same checks live.

## 1. What your computer needs

| Item | Requirement | Notes |
|---|---|---|
| Operating system | Windows 10 or 11, 64-bit (fully tested) · Linux x86-64, Ubuntu 22.04 / Debian 12 (compatible) | macOS on Apple Silicon is experimental and runs on the CPU only. AMD and Intel GPUs are not supported. 32-bit systems are not supported. |
| Pinokio | Installed and running | Pinokio provides `uv`, Node.js, Git and conda to the launcher (`requires: { bundle: "ai" }`). |
| Python | 3.10, in the app's own environment `app/roop_3d/.venv` | Created for you. Never use the system Python. |
| Node.js | 20.19 or newer (or 22.12 or newer) | Used once to build the web interface. Pinokio's bundled Node is used. |
| Graphics card (recommended) | NVIDIA, driver 550 or newer (CUDA 12) | Without a usable NVIDIA GPU, Roop 3D uses the CPU: it works, much more slowly. See [GPU troubleshooting](GPU_TROUBLESHOOTING.md). |
| Video memory | 12 GB class (10 GB or more), 6 GB class (5 GB or more), or none | Roop 3D picks a *hardware profile* from this. See [Performance guide](PERFORMANCE_GUIDE.md). |
| Disk space | Room for the environment, the model files (several GB) and your outputs | The preflight reports free space. Renders refuse to start when space is low (`DISK_SPACE_LOW`). |
| Network | Only while installing | After the environment and model files are on disk, Roop 3D runs entirely offline. See [Offline model installation](OFFLINE_MODELS.md). |

## 2. Install

1. Open Pinokio, find **Roop 3D** (cube icon), and open it.
2. Press **Install Roop 3D**. This runs `install.js`, which, in order:
   1. runs the project preflight (`tools/project_admin.py preflight`): checks the folder identity (`.roop3d-project.json`) and the repository guard, so the tools only ever act on this app;
   2. creates `app/roop_3d/.venv` and installs the pinned, hash-checked packages (`uv pip sync --require-hashes app/roop_3d/requirements.lock`);
   3. runs `torch.js` (the AI libraries Pinokio provides);
   4. in `app/roop_3d_ui`, runs `npm ci` and `npm run build` (the interface is built once and served by the app itself);
   5. writes `cache/install-ready.json` (`mark-ready`) only when every step above succeeded.
3. Wait until the sidebar offers **Start Roop 3D**. If any step prints an error, the install stops there; fix the cause and press *Reinstall environment*.

## 3. Start

1. Press **Start Roop 3D**. `start.js` runs the preflight, rebuilds the interface (`npm run build`), makes sure the Python packages match the
   lock (an NVIDIA GPU on Windows or Linux uses `requirements.txt`, everything else `requirements.lock`), then starts the server
   (`python -B app/serve.py`).
2. When the server reports its address (`http://127.0.0.1:<port>`) and its health check answers, the sidebar changes to **Open Roop 3D**.
3. Press **Open Roop 3D**. The first time, a *Welcome* dialog explains the workflow; **Start a guided project** takes you to the workstation.

The server only listens on `127.0.0.1` (your own computer). The default port is **8030**; set `ROOP3D_PORT` in the app's environment to
change it. If the port is busy, the start fails with `PORT_IN_USE` and an explanation; Roop 3D never closes another program's process.

## 4. After installing: two checks

Open **Settings → Environment & Offline Manager**:

1. **Environment Preflight & Hardware Profile**: press *Re-Run Preflight*. The 12 checks are *Windows Version, Python Version, Node.js, CPU
   Architecture & Cores, System RAM, NVIDIA GPU, NVIDIA Driver, CUDA Compatibility, ONNX Runtime, TensorRT, FFmpeg Codecs, Available Disk
   Space*. The badge shows the hardware profile Roop 3D chose (`12GB`, `6GB` or `CPU`).
2. **Offline Inference & Core Models**: press *Verify Offline Health*. It must show **100% OFFLINE READY** before you can process a video. If it
   says *MODELS NEEDED*, follow [Offline model installation](OFFLINE_MODELS.md).

## 5. Update, reinstall, reset

| Sidebar entry | What it does | What it keeps |
|---|---|---|
| **Update Roop 3D** (`update.js`) | Stops the app, runs the preflight, `git pull --ff-only`, then repeats the install steps | Your projects, FaceSets, models and outputs |
| **Reinstall environment** | Runs `install.js` again | Everything under `app/roop_3d/.runtime` |
| **Reset environment** (`reset.js`) | Deletes **only** `app/roop_3d/.venv` and `cache/install-ready.json` | Source, models, FaceSets, projects, outputs. It refuses to run while the app's port is still in use. |

## 6. Where things live

All paths are relative to `<Pinokio home>/api/roop-3d`.

| What | Where |
|---|---|
| Python environment | `app/roop_3d/.venv` |
| Imported videos | `app/roop_3d/.runtime/media` (each import is a copy with a generated name) |
| Finished videos and manifests | `app/roop_3d/.runtime/outputs` |
| Projects (`.r3dp`) | `app/roop_3d/.runtime/projects` |
| Render jobs and checkpoints | `app/roop_3d/.runtime/jobs` |
| FaceSets and 3D identity assets | `app/roop_3d/.runtime/facesets`, `app/roop_3d/.runtime/identity_assets` |
| Model files | `app/roop_3d/models/cache` |
| Settings | `app/roop_3d/.runtime/config.json` |
| Logs | `logs/` (Pinokio's launcher logs) |

> The imported-video folder grows with every import (each import is a full copy). Delete copies you no longer need after you have saved your
> outputs; projects keep a *reference* to a video, so deleting a copy makes the project report `MEDIA_NOT_FOUND` until you import it again.

## 7. Command line (optional)

From the app's own environment (on Windows: `app\roop_3d\.venv\Scripts\python.exe -B tools\project_admin.py <command>`, run in the project folder):

| Command | Purpose |
|---|---|
| `preflight [-v]` | Verify the folder identity and environment |
| `check-env`, `verify-install` | Environment and installation checks |
| `offline-health` | The same report as *Verify Offline Health* |
| `model-import --bundle <zip-or-folder>` | Import model files ([details](OFFLINE_MODELS.md)) |
| `model-verify` | Re-hash every installed model file |
| `check-update`, `backup-config`, `rollback [--backup-id ID]`, `recover` | Update and recovery helpers |
| `check-reset` | What *Reset environment* verifies before deleting |

## 8. If something goes wrong

* The app will not open: [Common problems → service offline](COMMON_PROBLEMS.md#service-offline).
* Install stops with a package error: the lock files are hash-pinned; do not edit them. Press *Reinstall environment* with a working network.
* GPU not used: [GPU troubleshooting](GPU_TROUBLESHOOTING.md).
