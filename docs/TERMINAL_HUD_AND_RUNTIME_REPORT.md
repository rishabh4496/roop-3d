# Terminal HUD, runtime report and TensorRT build heartbeat

What Roop 3D prints in the Pinokio terminal, what the Settings drawer shows, how both are produced, and where the numbers stop
being real. Code: `app/roop_3d/telemetry/` (`pinokio_terminal.py`, `runtime_report.py`, `trt_heartbeat.py`,
`trt_heartbeat_sidecar.py`), the hooks in `jobs/worker.py`, `engine/pipeline_runner.py`, `server/pipeline_status.py`, and on the
interface `components/diagnostics/RuntimeReportPanel.tsx` and `components/hud/EngineStatusBar.tsx`.
Tests: `app/roop_3d/tests/test_pinokio_terminal_telemetry.py`, `app/roop_3d_ui/src/runtimeReport.test.tsx`.

The design rule throughout: **the HUD only presents values that something measured.** A value nobody measured prints `--` or
`not measured`. Nothing is estimated to fill a gap, and no section is hard-coded to say "PASS".

## 1. Live, UTF-8 terminal

`PYTHONUNBUFFERED=1`, `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` are set in three places so no step can disagree:

| Where | How |
|---|---|
| `start.js`, `install.js`, `torch.js` | the `env` of every Python `shell.run` |
| `app/serve.py` and `roop_3d.server.__main__.main()` | `enforce_terminal_environment()` sets them in `os.environ` (every child inherits it) and reconfigures this process's stdout/stderr to UTF-8 |
| Child processes with an explicit `env=` | `child_environment()` returns `os.environ` plus the contract |

`update.js` and `reset.js` only run short admin checks and are unchanged.

## 2. The 14-section runtime report

Printed once after the server's `ready` line (boot) and again at the start of every job, drawn with box characters and ANSI colour
in the terminal. The same collectors answer `GET /api/v1/system/runtime-report` (JSON plus a plain-text `text` field) and fill the
Settings drawer, so the three cannot disagree.

| # | Section | Reads |
|---|---|---|
| 1 | Host / OS / Venv Isolation | OS build, interpreter, venv, the terminal contract above |
| 2 | CPU Cores & Worker Pool Allocation | hardware probe, `Settings` thread/pool values, the last pool benchmark |
| 3 | GPU Architecture & Compute Capability | driver probe; architecture is **inferred** from the compute capability and says so |
| 4 | Execution Provider Chain & Mixed Precision | ORT providers, resolved accelerator, precision policy, models actually loaded |
| 5 | TensorRT Engine Cache | `app/models/.trt_cache/*.engine`, `app/roop_3d/cache/engines/*.bundle`, last build, compile state |
| 6 | Zero-Copy OrtValue Ring Buffer | ring settings; is the torch-backed GPU ring importable; ORT boundary copies counted so far |
| 7 | VRAM Budget & OOM-Guard Thresholds | effective budget, reserve, watchdog thresholds (read from the code), live device usage |
| 8 | Model Governance & SHA-256 Verification | the live model catalogue: installed, hashed, verified, and installed-without-a-hash |
| 9 | 3D Head Topology & Anatomical Sub-Mesh | topology asset check, dense-face backbone, anatomical pack |
| 10 | Full-Craniofacial Swap Lock | swap scope, skull blend and the silhouette stages (this job's values at job start) |
| 11 | Physical Camera & SH Relighting | SH model, relight defaults, camera model constants; calibration is per footage |
| 12 | GOP Video Slicer & Audio Remux | FFmpeg, NVENC, checkpoint cadence, audio and timing policy |
| 13 | Offline Isolation Audit | the last completed isolation audit, remote-processing flag, proxy variables |
| 14 | Loopback Server & Deep-Link URLs | bind address, every screen's deep link, health, status stream, this report |

Each section is `OK`, `INFO` (configuration, nothing to judge), `WARN`, `UNAVAILABLE` (the capability is absent on this machine) or
`UNKNOWN` (the collector itself failed; the others are unaffected and only the exception *type* is reported). Home-directory paths
print as `~`; the share-mode token is never read. Building the report takes about two seconds (it probes hardware), so the server
builds it on a background thread.

**It must not trip the launcher.** `start.js` ends a start on `PORT_IN_USE|INVALID_SETTINGS|ROOP3D_GUARD_ERROR|ModuleNotFoundError|error:`
and captures the first `http://<numeric host>`. Everything the HUD prints goes through `defuse()`, which rewrites those tokens, and
the boot report is printed only after the `ready` line, so the first URL Pinokio sees is the real loopback address.

## 3. TensorRT build heartbeat

```
[TRT-BUILDER] ⚙ Compiling inswapper_128.onnx (FP16+FP32 Mix | opt=4x3x512x512) ... Elapsed: 14.0s | VRAM: 3.2 GB
```

`FP16+FP32 Mix` is shown for `FP16` builds because the builder always keeps an FP32 fallback for layer-norm and unsafe nodes.
`opt=` is the optimisation-profile shape and reads `n/a` on build paths that do not know it (only the strict builder passes it).
`VRAM:` is this process's own memory where the driver reports it, otherwise the whole device, marked `(dev)` (Windows WDDM reports
only the device). There is no percentage, because TensorRT exposes no build progress.

### The constraint that shaped the design

**ONNX Runtime holds Python's GIL for the whole of a TensorRT `InferenceSession(...)`.** Measured on this machine with a unique
synthetic model: a 7 s cold build let a 20 Hz Python thread wake twice (largest gap 6.99 s). While the build runs, *no* thread of the
server process runs, including a heartbeat thread and the asyncio loop that serves the WebSocket. A heartbeat that only lives inside
the process would be silent for exactly the 15-60 s it exists for. So:

* **Terminal:** `trt_heartbeat_sidecar` is a separate process (its own GIL) that prints the line once a second, in place on a TTY,
  and measures VRAM itself. It prints nothing for its first second, so an engine-cache hit leaves no trace; it exits when its stdin
  closes and erases its own line. If it cannot start, the line is printed in-process instead (correct for any build that does
  release the GIL). Verified on a real cold build: a line every ~1.1 s with VRAM rising 2.8 to 3.2 GB while the process was frozen.
* **WebSocket and UI:** the server publishes a `trt_compilation_progress` *start* frame (elapsed 0, `started_at`) **before** the
  native call and sleeps 0.15 s so the event loop delivers it; a `finished` frame follows the build. In between the server is
  silent and cannot be anything else. The React strip counts elapsed time itself from the last frame, shows VRAM only while a frame
  that carried it is under 2.5 s old (otherwise `n/a`, with a note that the terminal has the live figure), and clears on the finish
  frame, a dropped socket, or a 15 minute ceiling. Server-measured frames (elapsed and VRAM once a second) arrive whenever the
  process is able to run.

Because the whole process is blocked, HTTP requests are expected to wait out a cold build too (the GIL hold was measured; the
request stall was inferred, not separately timed). A real fix would compile engines in a child process and load the cached engine
in the server; that touches all three build paths and was not attempted here.

## 4. Live frame and stage-latency line

During a render one line is rewritten in place (`\r\x1b[K`); permanent lines (milestones, log records) scroll above it:

```
▶ 3fa9c2e1 │ FULL_CRANIOFACIAL_LOCK │ ████████░░░░░░  45.0% 540/1200 │ 23.4/21.9 fps │ ETA 00:30 │ VRAM 3.1/3.9/8.0GB │ CPU 62% │ Det 4.1 3D-LBS 6.2 Inpaint+Cage -- 3DSwap+SH 9.8 Enhancer 15.1 LapBlend -- Comp 8.2 ms
```

* `fps` is *current* (last two progress reports) / *rolling* (5 s window); ETA is remaining frames over the rolling rate.
* `VRAM` is used / peak since job start / budget. `VRAM(dev)` means device-wide (see above). CPU is system-wide, from the OS
  (kernel32 `GetSystemTimes`, `/proc/stat`); on macOS it prints `--`.
* **Stage columns map to what the pipeline really times** (`core/stage_timing.STAGES`): `Det`=detection, `3D-LBS`=pose/TPS,
  `3DSwap+SH`=inswapper, `Enhancer`=restorer. The compositor times `Inpaint+Cage` and `LapBlend` together as `masking`, so
  those two columns read `--` and the combined figure is shown as `Comp`. They are not split because no split is measured.
* The line is truncated to the terminal width; segments are dropped from the right (stages first), never the job, progress or rate.
* Not a TTY (a log file, CI): no cursor codes; a plain line at most every 5 s.
* Sampling (`nvidia-smi`, CPU counters) runs on its own thread once a second, so a render never waits for it.

Milestones are single `[TAG] event key=value` lines: `[JOB] started|succeeded|failed`, `[GOP] segment complete` and
`[CHECKPOINT] committed` every `CHECKPOINT_EVERY` frames (120), `[MUX] encode and audio remux started`, and
`[REMUX] output written and verified ... audio=copy - lossless stream copy, no re-encode`. A "GOP segment" here is a checkpoint
interval: the job runner encodes once at the end and does not use the GOP chunk stitcher in `execution/worker_pools.py`.

## 5. The Settings drawer

*Settings → UX & Diagnostics → Structured runtime report* (also in the studio's *Advanced Diagnostics*). It loads when opened (the
report probes hardware), opens the sections that need attention, and **Copy Structured Runtime Report** puts the plain-text
box form on the clipboard (a textarea fallback covers browsers that refuse the clipboard API). Below it is the thread/pool
benchmark runner: *Quick run* / *Full run* call `POST /api/v1/settings/benchmark_threads`, poll `GET .../benchmark_status`, mark the
fastest configuration and show the server's own caveat. It times a synthetic affine and host-transfer workload, **not** model or
video FPS, refuses to start while a job is rendering, and never runs by itself.

## 6. Switches

| Variable | Effect |
|---|---|
| `ROOP3D_HUD=0` | The HUD stays silent (CI, log capture); nothing else changes |
| `ROOP3D_TRT_SIDECAR=0` | No sidecar process: the heartbeat line is printed in-process (and so stalls during a TensorRT build) |
| `NO_COLOR` / `FORCE_COLOR` | Disable / force ANSI colour |

## 7. Known limits

* A cold TensorRT build freezes the whole server process (see section 3); only the terminal shows live progress during it.
* The existing `GET /system/diagnostics-report` (a different 14 sections) is untouched. Several of its fields are constants
  (`per_node_latency_breakdown`, `zero_external_calls`, an unconditional `overall_status: "PASS"`), it counts engines in a folder that
  does not exist, and `POST /system/benchmark` returns fixed numbers. Prefer `runtime-report` and `benchmark_threads`.
* Section 11 cannot show a calibrated camera: calibration is estimated per footage at render time and is not stored.
* The frame line and milestones were verified by driving a real job through the real worker and runner with a stand-in inference
  session, and the line's layout as pure rendering. No real face-swap render was watched in a live Pinokio terminal, so the in-place
  redraw on a real pty is covered by the byte-level tests, not by eye.
