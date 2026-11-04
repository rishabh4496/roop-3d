# Stage 18 — Long-duration reliability engineering

**Status: complete except two acceptance runs that could not be executed on this machine.** The 30-minute and 60-minute
video runs were started twice and terminated both times by Kaspersky behavioural detection (details below). Everything
else in the brief was exercised against the real code, and the work found and fixed **eight real defects**, two of
which would have lost or slowed long jobs in normal use.

Path mapping, as in earlier stages: the prompt's generic names map to `app/roop_3d/` (backend), `app/roop_3d_ui/`
(frontend) and `docs/`. New package: `app/roop_3d/reliability/`.

## Acceptance

| Criterion | Evidence | Result |
| --- | --- | --- |
| No memory growth beyond a bounded profile | `docs/reliability/bounded_profile.json`; soak runs below, all judged by the profile; a deliberate 4 MiB/job leak **is** flagged (`soak_detector_selftest.json`) | Met at the scales measured |
| No corrupted output after cancellation | cancel while running / paused / queued; hard kill; disk full; OOM; corrupt input — outputs either do not exist or are complete, hashed and frame-identical to an uninterrupted run | Met |
| No job corruption after restart | clean stop, `TerminateProcess` at early / late / during-mux points, then restart + resume → frame-identical output | Met (after fixing a defect that made every crashed job unresumable, see D1) |
| No orphan processes | Windows Job Object (kill-on-close) proven by terminating a parent; persistent child registry with pid-reuse-safe startup sweep; exact `ffmpeg_process` gauge | Met |
| No silent processing failure | every stop carries a specific code and an actionable message; unclassified errors are named by type in the log buffer and the report | Met |
| 30-minute and 60-minute video | **Not completed** — see "What was not achieved" | Open |

## What was not achieved

`python -m roop_3d.reliability.soak long_video --minutes 30` (45,000 frames) and `--minutes 60` (90,000 frames) were
launched four times. Each process was stopped by the antivirus, at the instants Kaspersky (`avp`) logged behavioural
events naming the exact command line (Application log, source `avp`, 01:04:18, 01:06:41, 01:14:28, 01:14:48, and again
01:51:07 for a 60-job run). A 118-second solo run and every pytest-hosted run (up to 5½ minutes) were not interrupted;
sustained, continuous checkpoint writing (hundreds of PNG creates + `fsync` + `os.replace` per second, for minutes) is
what triggers it. I did not add antivirus exclusions or reshape the workload to avoid the detection: that is a decision
about your security software, and a real user running a long render under Kaspersky would hit the same thing.

Two consequences worth knowing:

* **A render killed by antivirus is exactly the "unexpected termination" case**, and the recovery work below is what
  makes it survivable: the job comes back `interrupted` and resumes from its checkpoint.
* A *frozen* (suspended, not dead) render used to be invisible. The audit now reports `JOB_STALLED` for a job that is
  `running` but silent for more than 180 s.

To complete the two runs, exclude `G:\pinokio\api\roop-3d\app\roop_3d\.runtime\soak18` (and the miniforge `python.exe`
for that folder) in Kaspersky, then from `app/` run the commands in "Running the soaks". The scaled equivalent that
did run is 2,500 frames (100 s of video) through the identical path, below.

## Defects found and fixed

| # | Defect | How it was found | Fix |
| --- | --- | --- | --- |
| D1 | **A job that is killed (crash, kill, power loss) can never be resumed.** Progress is persisted every frame, checkpoints every 30; the resumed attempt reports a lower frame than the stale stored one and `progress()` rejected it as `PROGRESS_REGRESSION`, failing the job **terminally**. A clean pause hid it because it checkpoints at the stop. | hard-kill test | `JobStore.transition(→preparing)` rewinds progress; the runner reports the checkpoint position on resume |
| D2 | **One OS handle leaked per rendered job.** `core.storage.exclusive` cached a `threading.RLock` per lock-file path forever; each render has its own `.roop3d_<job>.mp4.lock`, and on Windows/CPython 3.10 an `RLock` owns a kernel semaphore. 500 paths → +500 handles. | successive-jobs soak flagged `handles` (+0.8/job) | locks held weakly; +0 handles for 500 paths, job series flat; mutual exclusion regression-tested across 6 threads |
| D3 | **Job history made every frame slower.** `state.json` held every job ever and every `get()`/`progress()` re-parsed and re-validated all of it, several times per frame: at 800 jobs `get()` 25 ms, `progress()` 71 ms — a hard ~6 fps cap from bookkeeping alone. | store scaling measurement | stat-fingerprint read cache (cross-process safe, failed writes leave no phantom state) + compact C-encoder JSON for the store only: `get()` **0.7 ms**, `progress()` **15.8 ms** at 800 jobs. Still O(history) on write: see "Limits" |
| D4 | **Output published without its manifest blocks the job forever.** Between `os.replace(output)` and writing the manifest a kill leaves a file the writer then refuses (`OUTPUT_EXISTS`). | analysis; reproduced by a test that fails without the fix | startup reconciliation **quarantines** (never deletes) outputs no succeeded job owns, so the resume succeeds |
| D5 | **Disk exhaustion was an unnamed terminal failure** (`WORKER_FAILED`). The writer also turned ENOSPC into "encode failed". | fault injection at frame store, job, and encoder level | up-front space check (`DISK_SPACE_LOW`); `DISK_FULL` anywhere → job becomes `interrupted` + resumable with the reason; Windows codes 112/39 handled |
| D6 | **Failed jobs kept all their checkpoint frames forever** (tens of GB for a long video; a failed job cannot resume). | code audit, test | frames released on failure; the small forensic `checkpoint.json` is kept |
| D7 | **Real host allocation failures were not recognised as out-of-memory.** ONNX Runtime raises `RuntimeException: bad allocation` (not `MemoryError`) → terminal `WORKER_FAILED`. Real CUDA OOM on the RTX 4070 *was* recognised. | triggered real 64 GiB allocations on CPU and CUDA | marker added; exhaustion of every memory profile (`PIPELINE_OOM`) now leaves the job resumable instead of failed |
| D8 | The worker re-ran cleanup for every cancelled job ever recorded on every 150 ms poll, and logged a cleanup error after a *successful* job as a job failure. | code audit; test counting calls | cleaned-set; post-success cleanup errors are tolerated and left to the audit |

One defect was **introduced by this stage and caught by the existing suite before it shipped**: the first version of
startup reconciliation treated the persistent queue's paused-job checkpoints (same `.runtime/jobs/<id>/` layout) as
orphaned work directories and deleted one (`test_stage09_remediation::test_fastapi_server_restart_restores_queued_and_paused_jobs_intact`).
Anything the queue database knows is now never audited or repaired, and an unreadable queue database makes the audit
fail closed. Regression test: `test_jobs_owned_by_the_persistent_queue_are_never_audited_or_repaired`.

Also hardened (measured, not leaks): `QueueManager` closed its SQLite connections only by refcount — handles stayed
flat on the normal path but rose by 8 while an exception was retained; connections now close deterministically.
FFmpeg children are now tracked, terminated `terminate → wait → kill`, and bound to the server's lifetime.

## What was built

```
app/roop_3d/reliability/
  probe.py       process RSS/private/peak/handles/threads/children (ctypes + /proc, no psutil), nvidia-smi VRAM,
                 torch allocator fragmentation (when torch is loaded), live ONNX Runtime session count
  ledger.py      counted tokens for decoders/encoders/ffmpeg/…; "abandoned" = collected without close (a definite leak)
  growth.py      bounded-profile analysis (Theil-Sen slope + high-water + net growth), DEFAULT and IDLE profiles
  disk.py        space estimate/refusal, ENOSPC/winerror recognition
  classify.py    exception -> stable code, message, resumable?  (never contains exception text)
  processes.py   Job Object kill-on-close, PDEATHSIG, ProcessRegistry + startup sweep, terminate/terminate_tree
  recovery.py    ReliabilityAuditor: audit (read-only) / repair (safe) / reconcile_on_startup
  monitor.py     always-on bounded sampler: peaks, per-job peaks, stage timings, sanitized warning buffer
  report.py      redacted local diagnostic report
  harness.py, soak.py, crash_child.py     stub-inference rig, scenarios + CLI, hard-kill child
server/reliability_routes.py     /api/v1/reliability/{report,resources,audit,repair}
roop_3d_ui/src/utils/resourceLedger.ts   object URLs, WebSockets, WebGL contexts, heap (opt-in: ?r3d-diag=1)
```

Startup (`server/app.py` lifespan, before the worker starts): job/queue recovery as before → orphan-child sweep →
`reconcile_on_startup` (stale `.part` files, encoder temp, stranded outputs, garbage work directories, queue mirror) →
resource monitor → worker. `server/__main__.py` binds the process to the kill-on-close Job Object.

Repair rules: never touch a job that can still run, never delete a published output (quarantine, purged after 7 days),
never touch FaceSet archives, the only destructive action (`reset_job_store`) must be requested explicitly.

## Diagnostic report

`GET /api/v1/reliability/report[?series=1&save=1]` — local only (loopback gate; no network I/O is performed; a test
blocks non-loopback sockets while building it). Sections: application version, hardware, drivers and library versions,
model versions/availability and whether render dependencies are satisfied, configured/advertised/verified execution
providers and per-job providers and fallbacks, per-stage timings (mean/max, per job), errors and warnings (counts,
last 25 sanitized, failure codes), peak RAM/VRAM (process, per job), growth verdict, job status, disk usage, resource
ledger, orphan-guard state, startup reconciliation and the live audit. `save=1` writes it to
`roop_3d/.runtime/diagnostics/` (20 kept).

Redaction (tested with sentinel media/person names): no pixels, masks, landmarks, embeddings or FaceSet contents (keys
dropped, numeric vectors replaced); media and FaceSet names → stable 8-hex hashes; application root → `<APP>`, home →
`~`, other absolute paths → `<PATH>`, user name removed; secrets via the existing redactor/log sanitizer.

## Measured results

All judged by `docs/reliability/bounded_profile.json`. The stub-inference rig is real for decode, per-frame PNG
checkpointing, state machine, locks, worker threads, mux/encode and publication; only the neural step is a stub.

| Scenario | Scale | Verdict | Notes |
| --- | --- | --- | --- |
| Long video through full pipeline | 2,500 frames (100 s), 2 tracks; 37.6 fps render | bounded | RSS peak 307 MiB → 220 at end; handles 810 → 797; threads 45 → 35; no children/encoders/decoders live after |
| Successive mixed jobs | 30 jobs (26 ok / 4 cancelled), pauses included | bounded | handles 709 → 708, threads 34 → 34, RSS +16 MiB (allowance 48); 26 outputs = 26 successes; 0 stranded work dirs |
| Repeated model switching (real ORT) | 300 loads/evictions | bounded | RSS 219.7 → 219.7; 0 live sessions; `evict_for_build` never refused |
| Real CUDA sessions on the RTX 4070 | 40 create/run/destroy | bounded | device VRAM growth asserted < 160 MiB; 0 sessions |
| High resolution | 1920×1080 noisy, 100 frames | bounded | peak RSS ≈ 630 MiB and released (216 MiB after); stable over 6 repeats |
| Many identities | 12 tracks / 12 FaceSets × 3 jobs | passes | all identities in the manifest |
| Server WebSockets | 150 connect/disconnect | no leak | subscribers 0, handles +≤8 |
| Browser (real Chrome) | 15 laps × 6 screens + PiP; 60 WebGL mount/unmount | bounded | nodes 3769 → 3775, listeners 1123 → 1159, 3 sockets live (events + 2 telemetry), 0 object URLs, WebGL live 2 (the workstation canvases), 0 detached |
| Detector validity | 4 MiB retained per job | **flagged** | `rss_mib`, `private_mib`; also a leaked WebGL context is reported as `detached: 1` |

Throughput of the stub rig (for planning the long runs): ~40 fps render, ~60 fps mux ⇒ about 28 minutes for a
30-minute clip and about an hour for a 60-minute clip.

## Recovery behaviour verified

| Event | Result |
| --- | --- |
| Cancel (running / paused / queued), ×3 jobs | `cancelled`, no output, work dir released, cleanup runs ≤ 2× per job |
| 4 pause/resume cycles | frame-identical to uninterrupted |
| Application restart (clean stop) | `interrupted` → resume → frame-identical |
| `TerminateProcess` at ~frame 12, ~frame 60, and during mux | recover → resume → frame-identical, outputs exactly the expected ones, audit clean |
| Disk full mid-job (injected ENOSPC) | `interrupted`, code `DISK_FULL`, checkpoint kept; resume after "freeing" → identical |
| Not enough space at start | `DISK_SPACE_LOW` before any frame is written |
| CUDA OOM at frame 30 | CPU retry from checkpoint → identical; the OOM'd session is released |
| Host memory / locked file / dead encoder pipe | `interrupted` with `PIPELINE_OOM` / `FILE_ACCESS_DENIED` / `ENCODER_TERMINATED`; resume succeeds |
| Truncated video | `failed` with a media code, nothing published |
| Garbage in mid-stream | never a silent damaged output (fails with a named code, or completes complete and hash-verified) |
| Corrupt model (real ORT protobuf error) | `failed`, `MODEL_CORRUPT`, nothing published, GPU lease released |
| Crash leftovers (`.part`, encoder temp, stranded output, ghost work dir, queue mirror drift) | found by audit, repaired; live jobs, FaceSets untouched |
| Damaged checkpoint (frame missing) | rolled back to last contiguous frame; resume succeeds |
| Unreadable job store | reported; only reset on explicit request (quarantined, not deleted) |

## Limits and design properties you should know

* **Resume cost.** A checkpoint stores finished *output frames*, not model/tracker state, so a resumed job **replays
  inference from frame 0** (documented in `MasterPipelineRunner`) and skips only re-saving finished frames. A crash at
  minute 55 of a 60-minute render costs ~55 minutes of inference to get back. Making resume O(remaining) needs
  serialisable tracker/temporal state — a separate project.
* **Job history is still O(n) on write** (15.8 ms/progress update at 800 jobs; extrapolates to ≈ 50 ms at 3,000). A retention policy
  (e.g. keep the newest 200 terminal jobs) is a product decision about History visibility, so it is not applied here.
* **Not testable here:** real power loss (writes are `fsync`'d and replaced atomically; the NTFS directory entry is not
  separately fsync'd on Windows), an actually full disk (injected at three layers instead), and a *real* GPU OOM inside
  the full pipeline (the real ORT CUDA and CPU allocation failures were reproduced and are recognised; the pipeline-level
  retry uses a stub raising the same message).
* **GPU accounting.** On Windows/WDDM `nvidia-smi` reports no per-process VRAM, so GPU growth is judged on the device
  total against a baseline. `torch` is not installed, so allocator fragmentation is reported as not loaded.
* **Stub inference.** Model-session memory is measured separately (real ORT sessions on CPU and CUDA) rather than inside
  the soak jobs.
* `pyproject.toml`'s package list was already stale for several earlier stages (installer, security, orchestrator …);
  only `roop_3d.reliability` was added.

## Verification status

Python: the suite was run in short sequential sessions (all files, in groups) with every file passing, including the
Stage 18 files (69 recovery/report/unit tests, 12 live-resource/soak tests). **A single full-suite pytest process
cannot finish on the development machine**: Kaspersky stopped both attempts at ~90% (02:24:20, 02:33:35), the same
behavioural detection as the long soaks, and a third attempt was reaped by Claude Code for low system memory. Use
chunked runs (≈12 files each, under ~3 minutes) there. UI: 41 files / 341 unit tests, `npm run build` (0 lint errors),
and the two Chrome specs pass.

The ONNX Runtime allocation-failure test uses a 16 PiB request: a 64 GiB request is silently satisfied once Windows
grows the page file under memory pressure.

## Running things

```
# from app/   (PowerShell: $env:PYTHONPATH = "."; add Node to PATH for the UI commands)
roop_3d/.venv/Scripts/python.exe -m pytest roop_3d/tests/test_stage18_*.py          # ~9 min, never run two pytest sessions at once
ROOP3D_RECORD_SOAK=1 ... pytest roop_3d/tests/test_stage18_soak_bounded.py            # re-records docs/reliability/*.json

# the long runs (need an antivirus exclusion on this machine, see above)
roop_3d/.venv/Scripts/python.exe -m roop_3d.reliability.soak long_video --minutes 30 --out ../docs/reliability/soak_long_30min.json
roop_3d/.venv/Scripts/python.exe -m roop_3d.reliability.soak long_video --minutes 60 --interval 10 --out ../docs/reliability/soak_long_60min.json
roop_3d/.venv/Scripts/python.exe -m roop_3d.reliability.soak hires --width 3840 --height 2160 --frames 60

# UI (from roop_3d_ui/)
npx vitest run src/stage18Resources.test.tsx
npx playwright test e2e/resourceLeaks.e2e.ts
# open the app with ?r3d-diag=1 and read window.__R3D_RESOURCES__.snapshot() in the console
```

Tests added: `test_stage18_growth_ledger.py` (unit), `test_stage18_recovery.py` (integration, incl. real hard kills),
`test_stage18_report_api.py`, `test_stage18_soak_bounded.py`, `test_stage18_resources_live.py`; UI
`stage18Resources.test.tsx` and `e2e/resourceLeaks.e2e.ts`.
