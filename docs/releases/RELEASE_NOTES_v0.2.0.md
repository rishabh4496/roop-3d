# Roop 3D v0.2.0 Release Notes

- **Release Date:** November 5, 2026
- **Build:** v0.2.0-stable
- **Release Architecture:** Standalone Pinokio Workstation
- **Integrity Digest:** Verified via HMAC-SHA256 (`release_manifest.json`)

---

## 1. Overview

Roop 3D v0.2.0 is a milestone release delivering a fully automated 15-gate continuous integration and testing pipeline, comprehensive hardware compatibility profiles (RTX 4070 12GB resident, RTX 3060 6GB clamped, and CPU fallback), an audited Software Bill of Materials (SBOM), and an air-gapped cryptographic release verification system.

Roop 3D operates completely independently of legacy environments (such as `roop-ultimate`), requiring zero external cloud connections for production face reenactment, 3D mesh alignment, and Spherical Harmonics relighting.

---

## 2. Key Highlights & New Features

### 2.1. 15-Gate Automated Continuous Integration Pipeline (`tools/ci_runner.py`)
A unified test runner enforcing 15 automated quality, fidelity, and security gates:
1. **Python Unit Tests:** Verification of schemas, coordinate transforms, 3D mesh geometry, and linear-light color science.
2. **Python Integration Tests:** End-to-end compositor execution, multi-engine capability orchestration, and live preview generation.
3. **Frontend Unit Tests:** 40 test suites (333 unit tests) running under Vitest for reactive UI components and state machines.
4. **TypeScript Type Checking:** Strict `tsc --noEmit` validation across 3D canvas and IPC boundaries.
5. **Linting & Syntax:** ESLint conformance and Python bytecode AST compilation verification.
6. **Production Build:** Vite rollup optimization verifying single-bundle asset integrity.
7. **API Tests:** Full lifecycle verification for REST endpoints, jobs, identity management, and installer preflight.
8. **Playwright Browser Tests:** Real headless Chrome browser testing with SwiftShader WebGL2 and zero external CDN leaks.
9. **FaceSet Archive Validation:** Cryptographic verification, schema conformance, and ZIP safety checks on `.fsz` archives.
10. **Model Registry Validation:** Manifest role evidence, license attribution, and tensor input/output contracts.
11. **Video Timing Tests:** Millisecond PTS/DTS synchronization and constant frame rate (CFR) remuxing via PyAV.
12. **Golden Video Inference:** Visual similarity regression testing against reference swaps.
13. **Model Hash Verification:** Cryptographic SHA-256 verification of all local model weights with drift detection.
14. **Security Tests:** Hard write boundary protection, reset target isolation, and synthetic media provenance.
15. **Legacy Isolation Audit:** Read-only integrity verification confirming zero modifications across 2,645 legacy sibling files.

### 2.2. Hardware Profiles & Resident Memory Optimization
- **Profile `12gb` (RTX 4070 Desktop):** High-capacity resident profile keeping primary TensorRT execution engines resident in GPU VRAM, allowing concurrent multi-face batching (up to 8) and hardware AV1/HEVC encoding.
- **Profile `6gb` (RTX 3060 Laptop):** Clamped memory profile enforcing a 4,600 MiB resident GPU memory budget, sequential pipeline execution, aggressive dormant engine eviction, and memory pressure recovery.
- **Profile `cpu` (CPU Fallback):** Pure CPUExecutionProvider execution with libx264 software encoding, allowing operational verification on systems without NVIDIA GPUs.

### 2.3. Cryptographic Release System & Artifacts
- **Software Bill of Materials (SBOM):** CycloneDX v1.5 compliant SBOM (`docs/releases/sbom_roop_3d.json`) cataloging all runtime Python packages, Node.js packages, and host binaries.
- **Model Compatibility Register:** Registry (`docs/releases/model_compatibility_register.json`) detailing execution provider and hardware profile compatibility for all 121 registered models.
- **Reproducible Lockfiles:** Pinned dependencies recorded in `docs/releases/requirements.lock` and `app/roop_3d_ui/package-lock.json`.
- **Integrity Manifest & Signatures:** Full SHA-256 manifest (`docs/releases/checksums.sha256`) and HMAC-signed release declaration (`docs/releases/release_manifest.json`).

---

## 3. Compatibility Matrix Summary

| Component | Tested Version | Supported Range |
|---|---|---|
| OS | Windows 10/11 64-bit | Windows 10/11, Ubuntu 22.04 LTS |
| Python | 3.10.20 | 3.10.x – 3.11.x |
| CUDA | 12.8 (Driver 621.01) | 11.8 – 12.8 |
| TensorRT | 10.9.0.34 | 10.0.x – 10.9.x |
| ONNX Runtime | 1.23.2 | 1.18.x – 1.23.x |
| Node.js | 24.22.1 | ^20.19.0 \|\| >=22.12.0 |
| React | 19.3.1 | ^19.0.0 |
| Three.js | 0.187.0 | r160 – r187 |

---

## 4. Upgrading

Refer to [MIGRATION_v0.2.0.md](file:///G:/pinokio/api/roop-3d/docs/releases/MIGRATION_v0.2.0.md) for step-by-step instructions on updating from previous versions without affecting existing FaceSets, projects, or outputs.
