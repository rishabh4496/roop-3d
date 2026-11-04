# Roop 3D Project Changelog

All notable changes to the Roop 3D project are documented in this file.

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
