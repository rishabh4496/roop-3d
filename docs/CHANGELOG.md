# Roop 3D Project Changelog

All notable changes to the Roop 3D project are documented in this file.

---

## [Unreleased]

### Stage 5: Timeline scrubbing, filmstrip and on-canvas face selection
- **New:** `GET /media/info`, `/media/frame` (JPEG + `X-Frame-Meta`, optional faces) and `/media/filmstrip`, served by a persistent LRU `VideoFrameService`: ~88 frames/s consecutive, ~30 ms random seek, one container for a whole session.
- **UI:** latest-frame-wins scrubbing, a one-request filmstrip, per-frame face boxes (dashed cyan `#n` + pose pill, hover glow, emerald `TARGET LOCKED`), a Detected Faces in Frame strip above the timeline, and a one-click Live Frame Swap Preview (split wipe). Locking a face creates its track, maps it to the active source and drives the 3D viewer's Live Pose.
- **Fixed:** frame indices >= 10 000 were refused (Use as source failed past ~5.5 min at 30 fps); the filmstrip decoded a whole GOP per thumbnail; a continuous drag never produced a frame.
- **Not provided, by design:** `embedding_norm` (always 1.0 after L2 normalisation), per-frame `track_id`, a duplicate `/media/preview-swap`. See STAGE05_TIMELINE_SCRUBBING.md.

### Stage 19: Professional UX, Accessibility and Documentation
- **Guidance:** first-run Welcome dialog, guided project start, a live seven-step checklist (Home, Guide, Next-step bar), target-face tutorial, FaceSet photo checklist with the real quality thresholds, plain-language 3D-coverage reading with prioritised requests, engine recommendation and hardware-profile explanations, queue/checkpoint guidance (resume replays from frame 0), and recovery instructions for 49 backend error codes (codes and request IDs are now kept and shown).
- **New control:** *Export preset* (seven presets plus HDR, subtitle and image-format policies). The server already supported them; the interface had no way to choose one.
- **Accessibility:** skip link, a visible focus ring everywhere, one keyboard contract for all dialogs (six had none or partial), repaired ARIA structure, minimum 24 px targets, contrast enforcement across all 37 themes plus light/AMOLED component fixes (axe: 0 violations), *Reduce motion* (system / always / never, including the 3D viewers).
- **Responsive:** tablet layouts at 1024 and 820 px (preview first and on screen), Face Manager no longer overflows, preview requests are identical at every window size.
- **Fixed:** a real 12 GB GPU (reports 12282 MiB) was demoted to the 6 GB profile at render time; offline `model-import` wrote to a folder the server never reads; the app was unusable for ~11 s while `/projects` re-verified saved projects; a malformed linter response blanked the page; the unavailable "foreground" pass could be ticked; shortcut labels did not match the pipeline stages (keys 7 and 8 added).
- **Documentation:** installation, offline models, user manual, FaceSet guide, engine comparison, GPU troubleshooting, performance, export, common problems, developer architecture, and the Stage 19 report (`docs/UX_STAGE19.md`), checked against the code by `test_stage19_docs.py`.
- **Known limit:** on this checkout the model-governance gate refuses the unreviewed core models (`LICENSE_UNVERIFIED`), so a full real-video render could not be demonstrated; documented in `docs/OFFLINE_MODELS.md`.

---

## [0.2.1] - 2026-11-05

### Stage 18: Long-Duration Reliability Engineering
- **Crash recovery fix:** a job killed mid-render could never be resumed (`PROGRESS_REGRESSION` failed it terminally). Progress now restarts from the checkpoint on every attempt.
- **Resource leaks fixed:** one OS handle leaked per rendered job (per-path lock cache); `QueueManager` SQLite connections now close deterministically; FFmpeg children are tracked, terminated and bound to the server's lifetime (Windows Job Object, POSIX `PDEATHSIG`) with a pid-reuse-safe startup orphan sweep.
- **Job store scaling:** read cache and compact JSON take `get()` from 25 ms to 0.7 ms and `progress()` from 71 ms to 16 ms at 800 jobs of history.
- **Failure handling:** disk exhaustion (`DISK_FULL`, `DISK_SPACE_LOW` preflight), host/GPU memory exhaustion (real ONNX Runtime `bad allocation` is now recognised), locked files and dead encoder pipes leave the job `interrupted` and resumable with a named reason; corrupt model/media get specific codes; failed jobs release their checkpoint frames.
- **Startup reconciliation and audit:** stale atomic-write files, encoder temp, stranded (manifest-less) outputs, garbage work directories, damaged checkpoints (rolled back), queue mirror drift, stalled jobs. Outputs are quarantined, never deleted.
- **Diagnostics:** local, redacted `GET /api/v1/reliability/report` (versions, hardware, drivers, models, providers, stage timings, warnings, peak RAM/VRAM, jobs, disk), plus `/audit`, `/repair`, `/resources`.
- **Leak instrumentation:** server-side lifecycle ledger and bounded-growth analysis; browser object-URL / WebSocket / WebGL-context ledger (`?r3d-diag=1`); soak harness `python -m roop_3d.reliability.soak`.
- **Not completed:** the 30- and 60-minute video runs were terminated by antivirus behavioural detection on the development machine (see `docs/RELIABILITY_STAGE18.md`).

---

## [0.2.0] - 2026-11-05

### Stage 17: Automated CI/CD and Hardware Compatibility
- **15-Gate CI/CD Runner:** Implemented `tools/ci_runner.py` automating 15 verification gates across Python unit tests, integration tests, frontend Vitest, TypeScript, ESLint, production build, API, Playwright browser tests, FaceSet archive validation, model registry, video timing, golden video inference, model hash verification, security boundaries, and legacy isolation.
- **Hardware Compatibility Matrix:** Audited 14 core dependencies (Python, NumPy, OpenCV, ONNX Runtime, PyTorch, CUDA, TensorRT, PyAV, FFmpeg, Node.js, React, TypeScript, Vite, Three.js) and established `docs/HARDWARE_COMPATIBILITY_MATRIX.md`.
- **Hardware Profiles:** Documented and verified `12gb` (RTX 4070 Desktop resident profile), `6gb` (RTX 3060 Laptop clamped profile), and `cpu` (CPU fallback).
- **Release System:** Built `tools/release_packager.py` producing CycloneDX SBOM (`sbom_roop_3d.json`), Model Compatibility Register (`model_compatibility_register.json`), Pinned Lockfiles (`requirements.lock`), SHA-256 Checksums (`checksums.sha256`), and HMAC-signed Release Manifest (`release_manifest.json`).
- **CI Workflows:** Created GitHub Actions workflows for general CI testing (`.github/workflows/ci.yml`) and self-hosted GPU runner validation (`.github/workflows/gpu-validation.yml`).

---

## [0.1.16] - 2026-11-04

### Stage 16: Roop 3D Offline Installer and Environment Manager
- **Environment Preflight:** Implemented 12-point environment preflight verifying Windows version, Python version, Node.js, CPU architecture, system RAM, NVIDIA GPU, driver version, CUDA compatibility, ONNX Runtime, TensorRT, FFmpeg codecs, and available disk space.
- **Offline Manager:** Created local model directory scanner, ZIP/folder bundle importer with zip-slip protection, SHA-256 verification, and clear missing-model messages.
- **Update & Rollback:** Added repository update status checker, atomic configuration backup system, and instant rollback.
- **Reset Safety:** Hardened reset script to strictly delete Roop 3D virtual environment without deleting FaceSets, projects, or outputs, and never touching sibling applications.

---

## [0.1.15] - 2026-11-03

### Stage 15: Professional Media and Color-Management Engine
- **Linear-Light Blending:** Implemented ITU-R BT.709 linear light compositing and transfer functions preventing dark halos on swapped boundaries.
- **PyAV Frame Engine:** Added high-precision video reader and writer with millisecond-accurate PTS/DTS synchronization and constant frame rate (CFR) enforcement.
- **Resilience:** Implemented non-finite NaN/Inf sanitization, dynamic packet flush, and golden video end-to-end regression tests.

---

## [0.1.14] - 2026-11-02

### Stage 14: Capability-Based Multi-Engine Orchestrator
- **Heterogeneous Scheduler:** Capability-based device scheduler balancing tasks across TensorRT, CUDA, and CPU execution providers.
- **Workstation Preview:** Solo preview modes for isolated node inspection (`normal_map`, `sh_lighting`, `face_swap`).

---

## [0.1.13] - 2026-11-01

### Stage 13: Security Assessment, Privacy Enforcement, and Synthetic Media Protection
- **C2PA / Provenance:** Added cryptographic provenance metadata and digital signatures to exported media.
- **Biometric Protection:** Per-job consent enforcement and biometric data protection in local storage.

---

## [0.1.12] - 2026-10-31

### Stage 12: Model Governance and Secure Management
- **Model Governor:** Policy enforcement preventing untrusted model execution and verifying licenses.
- **Telemetry & VRAM HUD:** Real-time VRAM pressure gauge and engine memory monitor in UI.

---

## [0.1.11] - 2026-10-30

### Stage 11: Diffusion Refinement & Evaluation Framework
- **SDXL-Turbo / FLUX.1 Support:** Optional diffusion-based texture enhancement and high-frequency blending.
- **Quality Metrics:** SSIM, PSNR, and LPIPS objective evaluation framework.

---

## [0.1.10] - 2026-10-29

### Stage 10: Audio Synchronization & Compositor
- **Lip Synchronization:** Wav2Lip alignment and audio waveform preservation.
- **GPU Compositor:** Fast alpha-blending shader pipeline for 30 FPS playback.

---

## [0.1.9] - 2026-10-28

### Stage 9: Adaptive Runtime & Occlusion Engine
- **Occlusion Handling:** BiSeNet segmentation integration for hair, glasses, and hands masking.

---

## [0.1.8] - 2026-10-27

### Stage 8: 3D FLAME Mesh & Geometry Pipeline
- **FLAME 2020 Model:** 3D parametric head model fitting and landmark tracking.
- **TPS Warping:** Thin-plate spline geometric rectification.

---

## [0.1.7] - 2026-10-26

### Stage 7: Workstation UI & Identity Panel
- **Studio Shell:** 9-screen React 19 UI with Three.js 3D viewport and theme gallery.

---

## [0.1.1] to [0.1.6] - 2026-10-20 to 2026-10-25

### Stages 1 to 6: Scaffold, Storage, Models, FaceSets, Pipeline, and Engine Cache
- Established standalone repository identity, Git write boundaries, FaceSet `.fsz` specification, ONNX Runtime and TensorRT engine cache, and REST API foundation.
