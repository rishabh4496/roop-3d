# Roop 3D Hardware & Dependency Compatibility Matrix

## 1. Executive Summary

Roop 3D is a standalone, high-performance 3D facial reenactment, relighting, and compositing workstation.
This document establishes the verified software dependencies, hardware execution profiles, and supported platform combinations.

> [!IMPORTANT]
> Roop 3D maintains strict platform integrity: unsupported and untested hardware configurations are never advertised as supported. Cloud connectivity is never required for local inference.

---

## 2. Dependency Audit & Compatibility Matrix

All 14 core project dependencies have been verified against runtime execution tests.

| # | Dependency | Version Tested | Supported Range | Status | Architecture Notes & Restrictions |
|---|---|---|---|---|---|
| 1 | **Python** | `3.10.20` (64-bit) | `3.10.x` – `3.11.x` | **VERIFIED** | Dedicated `.venv` only. 64-bit required. Python $\le$ 3.9 unsupported. |
| 2 | **NumPy** | `1.26.4` | `^1.26.0` | **VERIFIED** | Pinned to 1.26.x. NumPy 2.x is currently **UNSUPPORTED** due to C-API binary ABI incompatibility with certain ONNX/TensorRT bindings. |
| 3 | **OpenCV** | `4.11.0.86` | `>=4.8.0, <5.0.0` | **VERIFIED** | `opencv-python-headless` required. GUI-enabled OpenCV is forbidden to prevent Qt/Win32 message loop conflicts in server processes. |
| 4 | **ONNX Runtime** | `1.23.2` | `1.18.x` – `1.23.x` | **VERIFIED** | `onnxruntime-gpu` with `TensorrtExecutionProvider`, `CUDAExecutionProvider`, and `CPUExecutionProvider`. |
| 5 | **PyTorch** | `2.6.0+cu124` | `2.1.x` – `2.6.x` | **OPTIONAL / DEV** | **Inference runs directly via ONNX Runtime & TensorRT.** PyTorch is not loaded to boot the API or run production swap pipelines (`torch.js`). |
| 6 | **CUDA** | `12.8` (Driver 621.01) | `11.8` – `12.8` | **VERIFIED** | NVIDIA Driver $\ge$ 550.00 required for CUDA 12. CUDA $\le$ 11.7 unsupported. |
| 7 | **TensorRT** | `10.9.0.34` | `10.0.x` – `10.9.x` | **VERIFIED** | `tensorrt-cu12` runtime with dynamic shape profiles and engine serialization. |
| 8 | **PyAV** | `17.1.0` | `>=10.0.0` | **VERIFIED** | Provides direct C-level FFmpeg bindings for millisecond-accurate PTS/DTS timestamp tracking and CFR remuxing. |
| 9 | **FFmpeg** | `8.1.2` | `>=6.0.0` | **VERIFIED** | Hardware accelerated NVENC (`av1_nvenc`, `h264_nvenc`, `hevc_nvenc`) and NVDEC (`cuvid`). Software fallback via `libx264`/`aac`. |
| 10 | **Node.js** | `v24.22.1` | `^20.19.0 \|\| >=22.12.0` | **VERIFIED** | Required for UI build and Vite workstation runtime. Node < 20 unsupported. |
| 11 | **React** | `19.3.1` | `^19.0.0` | **VERIFIED** | Modern concurrent rendering, fast single-page workstation shell. |
| 12 | **TypeScript** | `7.0.2` | `>=5.4.0` | **VERIFIED** | Strict type safety across all 3D canvas, state stores, and IPC protocols. |
| 13 | **Vite** | `8.3.5` | `>=5.0.0` | **VERIFIED** | Fast HMR dev server (port 5174) and production roll-up packager. |
| 14 | **Three.js** | `0.187.0` | `r160` – `r187` | **VERIFIED** | WebGL2 3D viewport, FLAME mesh topology inspector, and 9-band Spherical Harmonics light probe sphere. |

---

## 3. Hardware Profiles

Roop 3D classifies target execution hardware into three explicit profiles:

### 3.1. Profile `12gb`: RTX 4070 Desktop (High-Capacity Resident Profile)

- **Target GPUs:** NVIDIA GeForce RTX 4070, RTX 4070 Ti, RTX 4080, RTX 3080 12GB, RTX 3090, RTX 4090.
- **Minimum VRAM:** 10,240 MiB (Recommended: $\ge$ 12,288 MiB).
- **Execution Policy:**
  - Unthrottled resident TensorRT execution engines (`inswapper_128`, `facelib_landmark`, `face_parser`).
  - Concurrent multi-face batching (batch size up to 8).
  - High-throughput NVENC encoding (`av1_nvenc` or `hevc_nvenc` with zero B-frames).
  - Concurrent 3D FLAME mesh fitting and 9-band Spherical Harmonics relighting in real-time preview.
- **Memory Pressure Budget:** Up to 9,500 MiB resident GPU memory before triggering cache compaction.

### 3.2. Profile `6gb`: RTX 3060 Laptop (Low-VRAM Clamped Profile)

- **Target GPUs:** NVIDIA GeForce RTX 3060 Laptop/Desktop, RTX 4050 Laptop, RTX 4060 Laptop, RTX 2060 6GB.
- **Minimum VRAM:** 5,120 MiB (Recommended: 6,144 MiB).
- **Execution Policy:**
  - Clamped resident VRAM budget (maximum 4,600 MiB active memory).
  - Sequential pipeline execution (batch size constrained to 1 or 2).
  - Dynamic TensorRT engine swapping: models evicted from VRAM when dormant.
  - Aggressive garbage collection (`cuda.empty_cache()` equivalent) between job intervals.
  - Hardware encoding clamped to standard NVENC `h264_nvenc` to minimize encoder VRAM footprint.

### 3.3. Profile `cpu`: CPU Fallback

- **Target Systems:** Systems lacking an NVIDIA GPU or with incompatible compute capability (< 7.0).
- **Execution Policy:**
  - Pure `CPUExecutionProvider` for all ONNX model inference.
  - Software encoding via `libx264` and `aac`.
  - Sequential single-frame pipeline execution.
  - Resolution clamped to 512x512 preview resolution for interactive sanity checking.
  - No TensorRT compilation attempted.

---

## 4. Platform & Host Operating System Support

| OS / Platform | Architecture | Support Tier | Notes |
|---|---|---|---|
| **Windows 11 (64-bit)** | x86_64 (AMD64) | **TIER 1 (OFFICIAL / TESTED)** | Primary development and validation platform. Build 26300+ tested. |
| **Windows 10 (64-bit)** | x86_64 (AMD64) | **TIER 1 (OFFICIAL / TESTED)** | Fully supported. Driver $\ge$ 550 required for CUDA 12. |
| **Linux (Ubuntu 22.04 / Debian 12)** | x86_64 (AMD64) | **TIER 2 (COMPATIBLE)** | Supported with NVIDIA proprietary driver $\ge$ 550 and CUDA 12 toolkit. |
| **macOS (Apple Silicon M1/M2/M3/M4)** | ARM64 | **UNTESTED / EXPERIMENTAL** | CPU execution provider only. Metal/MPS is not integrated into production pipeline. |
| **Linux (ARM64 / Jetson)** | aarch64 | **UNTESTED** | Untested in current release. Requires custom TensorRT wheels. |
| **AMD GPUs (ROCm)** | Any | **UNSUPPORTED** | ROCm is not supported on Windows; untested on Linux. |
| **Intel Arc / OneAPI** | Any | **UNSUPPORTED** | OpenVINO / OneAPI execution providers are not supported in this release. |
| **32-bit Windows / Linux** | x86 (32-bit) | **STRICTLY UNSUPPORTED** | Modern machine learning runtimes require 64-bit pointers and instruction sets. |

---

## 5. Offline & Privacy Contract

1. **Zero External Network Dependencies:** Once model weights and pip wheels are cached locally, all video inspection, face extraction, 3D mesh landmarking, relighting, and swapping execute strictly on `127.0.0.1` without internet connectivity.
2. **Model Integrity:** All model weights are cryptographically verified with SHA-256 before loading into GPU memory.
3. **No Sibling Interference:** Roop 3D maintains total process and filesystem isolation from sibling applications such as `roop-ultimate`.
