# Roop 3D: Release Checklist & Operational Runbook

**Release Version:** 0.1.0
**Stage:** Stage 15 (E2E Golden-Clip Validation, Hardware Benchmarks & Release Gate)
**Status:** **READY FOR RELEASE**
**Audit Baseline:** 100% Byte-for-Byte Protected Legacy Repository Isolation Verified

---

## 1. Executive Release Verdict

The Roop 3D clean-room application has completed all 15 stages of specification, development, and validation. The release gates require 100% passing results across Tier 1 (CPU & Contracts), Tier 2 (Golden-Clip End-to-End Video Swap), Tier 4 (Resilience & Fault Injection), and the Protected-Path Isolation Audit. All gates have passed without exception or mock substitutions.

| Gate / Validation Tier | Status | Result / Metrics |
|---|:---:|---|
| **Tier 1 â€” CPU & Contract Suite** | **PASS** | 390 passed, 18 subtests passed; UI TypeScript & Vitest (18 tests) & Vite build passed |
| **Tier 2 â€” Mandatory Golden-Clip E2E Smoke Test** | **PASS** | Real video swap (`test_stage15_e2e_golden.py`), audio preserved, background pixels unchanged, face swapped, manifest written |
| **Tier 3 â€” Hardware Benchmark Protocol** | **PASS** | Benchmarked on host NVIDIA GeForce RTX 4070 (12 GB VRAM). RTX 3060 Laptop marked `UNAVAILABLE_LOCAL` without fabrication |
| **Tier 4 â€” Resilience & Fault Injection** | **PASS** | 5/5 tests passed: cancellation, pause/resume checkpoint, port collision, corrupt weight quarantine, simulated CUDA OOM fallback |
| **Integrity Gate â€” Isolation & Immutability** | **PASS** | 2,645 historical hashes and 68,390 comprehensive hashes 100% byte-identical; zero repository violations; clean Git porcelain |

**Final Recommendation:** **READY FOR RELEASE**

---

## 2. 4-Tier Validation Matrix

### Tier 1: CPU & Contract Suite
- **Python Unit & Contract Suite (Stages 01â€“14):**
  - Command: `roop_3d\.venv\Scripts\python.exe -B -m pytest -c roop_3d/pyproject.toml --confcutdir=roop_3d roop_3d/tests -q`
  - Exit Code: `0`
  - Output: `390 passed, 18 subtests passed`
  - Covers: Config migrations, job state machine transitions, model downloader digest validation, hostile `.fsz` path traversal and zip bomb rejection, rational PTS/VFR timestamp mapping, ComfyUI mock bridge, and full-frame compositor math.
- **Frontend Workstation Suite (`app/roop_3d_ui/`):**
  - `npm run typecheck`: Exit `0` (strict TypeScript validation)
  - `npm test`: Exit `0` (18/18 Vitest unit and integration tests passed)
  - `npm run build`: Exit `0` (Production HTML5/CSS3/ES2022 bundle and lazy Three.js chunks compiled cleanly)

### Tier 2: Mandatory Golden-Clip End-to-End Smoke Test
- **Test File:** `app/roop_3d/tests/test_stage15_e2e_golden.py`
- **Execution Command:** `roop_3d\.venv\Scripts\python.exe -B -m pytest -c roop_3d/pyproject.toml --confcutdir=roop_3d roop_3d/tests/test_stage15_e2e_golden.py -v`
- **Exit Code:** `0` (1 passed in 49.40s)
- **Verified Contracts:**
  1. Real unmocked inference through `worker.py` and `pipeline_runner.py` with local verified models (`scrfd_2.5g_kps.onnx`, `arcface_w600k_r50.onnx`, `inswapper_128.onnx`).
  2. Input video clip with AAC audio decoded and processed cleanly.
  3. Job submitted via `POST /api/v1/jobs` transitioned to `succeeded`.
  4. Published video `roop_3d/outputs/roop3d_<job_id>.mp4` verified via PyAV: decodes cleanly, exact source frame count, exact rational duration.
  5. Audio stream preserved and decodable in output container.
  6. Background pixels outside face crop verified unchanged (mean difference < 3.0 LSB under H.264 compression).
  7. Face region pixels verified changed (mean difference > 5.0, peak > 20.0).
  8. Output manifest written with SHA-256 model digests for all active inference models.

### Tier 3: Hardware Benchmark Protocol
- **Benchmark Tool:** `app/roop_3d/run_benchmarks.py`
- **Execution Command:** `roop_3d\.venv\Scripts\python.exe -B roop_3d/run_benchmarks.py --frames 10`
- **Exit Code:** `0`
- **Hardware Evidence:**
  - Host OS: Windows 10 (AMD64)
  - CPU: 32 logical cores / 24 physical cores
  - System RAM: 32,451 MiB total (~32 GB)
  - Host GPU: NVIDIA GeForce RTX 4070 Desktop (12,282 MiB VRAM, driver 621.01)
  - Target Status:
    - **RTX 4070 Desktop:** `VERIFIED_LOCAL` (Available and benchmarked)
    - **RTX 3060 Laptop:** `UNAVAILABLE_LOCAL` (Hardware not physically present; unexecuted per anti-fabrication mandate)
- **Measured Latency Profile (512x512 video):**
  - Warm-up Latency (Frame 0): 14,137.46 ms (includes JIT engine init and weight transfers)
  - Steady-State Per-Stage Latencies (Frames 1+):
    - `decode`: 1.50 ms (p50: 1.50 ms, p95: 1.50 ms)
    - `detect_track`: 625.00 ms (p50: 622.05 ms, p95: 651.13 ms)
    - `swap`: 207.60 ms (p50: 220.00 ms, p95: 230.66 ms)
    - `retarget`: 7,144.26 ms (p50: 7,139.21 ms, p95: 7,196.75 ms)
    - `restore`: 1,380.59 ms (p50: 1,380.73 ms, p95: 1,389.65 ms)
    - `composite`: 1,656.26 ms (p50: 1,651.02 ms, p95: 1,677.54 ms)
    - `encode`: 0.57 ms (p50: 0.58 ms, p95: 0.64 ms)
  - Peak GPU VRAM: 3,970 MiB (Initial: 3,681 MiB, Delta: 289 MiB)
  - Peak System RAM: 23,562 MiB (Initial: 20,029 MiB, Delta: 3,533 MiB)

### Tier 4: Resilience & Fault Injection Suite
- **Test File:** `app/roop_3d/tests/test_stage15_resilience.py`
- **Execution Command:** `roop_3d\.venv\Scripts\python.exe -B -m pytest -c roop_3d/pyproject.toml --confcutdir=roop_3d roop_3d/tests/test_stage15_resilience.py -v`
- **Exit Code:** `0` (5/5 tests passed in 3.82s)
- **Scenarios Verified:**
  1. `test_tier4_mid_job_cancellation`: Mid-job cancellation stops pipeline cleanly, aborts encoding, prevents partial output publication, and removes job workspace.
  2. `test_tier4_pause_resume_checkpoint`: Pause mid-flight captures atomic `checkpoint.json` with completed frame index; resume continues from checkpoint to 100% completion.
  3. `test_tier4_port_collision_fail_closed`: Port collision is detected without attempting to terminate the existing port owner or crashing system resources.
  4. `test_tier4_corrupt_model_file_quarantine`: Tampered or corrupt weight payload is rejected by `ModelCatalog.verified_bytes` with `DomainError("MODEL_DRIFT")`.
  5. `test_tier4_simulated_oom_fallback`: Simulated CUDA out-of-memory triggers checkpoint preservation, resets admission profile, and succeeds on CPU profile.

---

## 3. Final Protected-Path Integrity Gate

- **Isolation Verifier:** `app/roop_3d/audit/verify_isolation.py`
- **Exit Code:** `0`
- **Verified Baselines:**
  - Historical Baseline (`protected_paths.json`): 2,645 / 2,645 entries intact.
  - Comprehensive Machine Snapshot (`protected_all_paths.json`): 68,390 / 68,390 files verified byte-identical.
  - Violations: `0`
  - Reparse Points / Symlinks / Junctions into legacy writable roots: `0`
- **Git State:** `git status --porcelain` confirms all modifications and new test/benchmark files are restricted strictly to `app/roop_3d/**`.

---

## 4. Operational Runbook: Startup & Deployment

### 1-Click Launch (Pinokio Managed Runtime)
1. In Pinokio, select the **Roop 3D** application tile.
2. Click **Start** (triggers `start.js`).
3. The launcher executes preflight environment validation, starts `app/serve.py` in the dedicated virtual environment `app/roop_3d/.venv`, captures the loopback URL `http://127.0.0.1:8030`, and verifies readiness via `/api/v1/health`.
4. Pinokio opens the Workstation Web UI automatically.

### Manual / Headless CLI Startup
From the checkout root:
```powershell
app\roop_3d\.venv\Scripts\python.exe -B app\serve.py
```
Or with custom binding:
```powershell
$env:ROOP3D_PORT = "8030"
$env:ROOP3D_HOST = "127.0.0.1"
app\roop_3d\.venv\Scripts\python.exe -B app\serve.py
```
The server serves both the REST API `/api/v1/*` and the compiled Workstation SPA `/` from `app/roop_3d_ui/dist`.

---

## 5. Rollback & Maintenance Instructions

### Resetting Dependencies & Cache
If the runtime environment becomes corrupted:
1. In Pinokio, click **Reset** (or run `reset.js`).
2. `reset.js` removes temporary caches and the dedicated `.venv` inside `app/roop_3d/`.
3. Legacy model files and repository weights are READ-ONLY and remain untouched.
4. Run `install.js` to perform a clean hash-locked sync via UV (`requirements.lock`).

### Rebuilding Frontend Assets
If changes are made to the Workstation UI source:
```powershell
cd app/roop_3d_ui
npm run build
```
Vite will compile production bundles to `app/roop_3d_ui/dist`, which FastAPI serves automatically.

---

## 6. Known Hardware Limitations & Profile Guidelines

1. **GPU Profiles:**
   - **12 GB Profile:** Recommended for multi-face tracking and concurrent LivePortrait retargeting with GPEN-512 restoration.
   - **6 GB Profile:** Uses half-precision weights where available; caps memory reserve to avoid CUDA out-of-memory events.
   - **CPU Profile:** Fallback execution provider for environments without CUDA capability. All models execute deterministically on CPU.
2. **ONNX Runtime Windows GridSample 4-D Restriction:**
   - Under ONNX Runtime's `CUDAExecutionProvider` on Windows, the `GridSample` operator in `dense_motion_network` requires 4-D tensors. The pipeline automatically catches this known operator boundary and falls back to CPU for that node while retaining GPU acceleration for all other models.
3. **Optional Adapters:**
   - Heavy diffusion models (FLUX, SDXL) and restricted parametric mesh models (FLAME) remain isolated optional modules. If restricted weights or licenses are absent, the primary volumetric/swap pipeline operates at 100% functionality with the license-free 2.5D Landmark TPS geometry fallback.
