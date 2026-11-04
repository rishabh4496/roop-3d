# Stage 13 — Roop 3D Security, Privacy and Synthetic Media Threat Model

## Executive Summary

Roop 3D is an independent, local-first application designed for 3D face reenactment,
volumetric performance transfer, and identity replacement. Because synthetic media
technologies present significant security, privacy, and ethical risks, Roop 3D implements
a comprehensive, defense-in-depth security architecture.

Processing services are strictly bound to loopback by default. No media, embeddings,
or telemetry are transmitted externally without per-job cryptographic consent.
All inputs (videos, archives, models, project manifests) are subject to content-level
validation, strict size caps, and traversal defenses. Every output preserves immutable
provenance, visible synthetic-media disclosures, and optional cryptographic authentication.

---

## Architecture & Trust Boundaries

```
       +-------------------------------------------------------------+
       |               Untrusted External World / Network             |
       +-------------------------------------------------------------+
                                     |  [Blocked by Default: No Egress]
                                     x  [Cloud Providers Disabled]
                                     |
+------------------------------------+-------------------------------------------+
| Roop 3D Trust Boundary (Loopback 127.0.0.1 / Local Machine)                     |
|                                                                                |
|  +------------------------+           +-------------------------------------+  |
|  | Frontend Workstation   | --------> | FastAPI ASGI Server (Port 8030)     |  |
|  | (Vite Port 5174,       |           | - IntakeMiddleware (Host/Origin)    |  |
|  |  Loopback CORS Only)   | <-------- | - WebSockets (Receive-Only Streams) |  |
|  +------------------------+           +-------------------------------------+  |
|                                                          |                     |
|              +-------------------------------------------+                     |
|              |                    |                      |                     |
|              v                    v                      v                     |
|     +------------------+ +-----------------+ +-----------------------+         |
|     | File Ingestion   | | Model Hub       | | Job Worker & Engine   |         |
|     | - Magic validation| - Pinned hashes   | - Concurrency limits  |         |
|     | - Zip-bomb filter| - Pickle rejected | - VRAM governor       |         |
|     | - Safe unlinking | - Safe renames    | - Cancellation/timeout|         |
|     +------------------+ +-----------------+ +-----------------------+         |
|              |                    |                      |                     |
|              +--------------------+----------------------+                     |
|                                   |                                            |
|                                   v                                            |
|                    +------------------------------+                            |
|                    | Owned Filesystem Boundary    |                            |
|                    | - roop_3d/.runtime/          |                            |
|                    | - roop_3d/outputs/           |                            |
|                    | - Symlink rejection          |                            |
|                    +------------------------------+                            |
+--------------------------------------------------------------------------------+
```

---

## Detailed Coverage of Security Requirements

### 1. Loopback Binding by Default
- The ASGI application binds exclusively to `127.0.0.1` (or `localhost`) on port `8030`.
- `IntakeMiddleware` inspects the HTTP `Host` header and client IP, rejecting any non-loopback
  peer unless the opt-in remote sharing mode (`--server_share`) is explicitly engaged with a
  32-byte cryptographic bearer token.
- `ComfyUIBridge` and local helpers enforce that destination endpoints are loopback IPs.

### 2. External Media Transmission Prevention
- No media files, decoded video frames, or biometric embeddings are transmitted externally.
- `CloudStub` returns `NOT_CONFIGURED` for all cloud adapter requests.
- `ConsentVerifier.verify_egress_policy()` fails closed if an external destination is specified
  without explicit user configuration and an HMAC-signed per-job consent token from `ConsentAuthority`.

### 3. File Type and Actual Content Validation
- File extensions are never trusted on their own.
- **Videos**: `validate_video_container()` verifies container signatures:
  - MP4/MOV: ISO Base Media File Format box headers (`ftyp`, `moov`, `mdat`, `skip`, `free`).
  - MKV/WebM: EBML header signature (`\x1a\x45\xdf\xa3`).
  - AVI: RIFF header with `AVI ` four-character code.
  - MPEG-TS: Packet synchronization byte `0x47`.
  - Disguised binaries (Windows PE `MZ`, Linux ELF `\x7fELF`, shell scripts `#!`, Python pickling)
    are immediately rejected.
- **FaceSets**: `validate_faceset_archive_bytes()` checks zip magic bytes (`PK\x03\x04`). Inside
  the archive, `ArchiveValidator` inspects the actual header bytes of every image entry:
  - PNG: `\x89PNG\r\n\x1a\n`
  - JPEG: `\xff\xd8\xff`
  - WebP: `RIFF....WEBP`
  - JSON: Valid UTF-8 parseable JSON.

### 4. Upload and Decompression Limits
- Upload requests are capped at `MAX_ASSET_BYTES` (8 GiB) for videos, and 256 MiB for FaceSet archives.
- Streaming uploads are processed in sequential chunks (`CHUNK_BYTES = 1 MiB`) with strict offset tracking.
- `ArchiveValidator` enforces zip-bomb defenses:
  - Maximum entry count: 512 entries.
  - Maximum single entry uncompressed size: 32 MiB.
  - Maximum total uncompressed archive size: 256 MiB.
  - Maximum compression expansion ratio: 20.0x (for entries > 64 KiB).

### 5. Directory Traversal and Symlink Attacks
- `validate_safe_filename()` rejects path separators (`/`, `\`), null bytes, Windows reserved
  device names (`CON`, `PRN`, `AUX`, `NUL`, `COM1-9`, `LPT1-9`), and trailing dots/spaces.
- `validate_safe_path()` resolves target paths and verifies they remain strictly within their
  designated parent directory.
- `_check_symlink()` inspects ZipInfo attributes (`S_IFLNK` and Windows reparse point attribute `0x400`)
  to reject symlinks and junctions before reading any archive entry.
- File unlinking and directory deletion reject symlinks at all levels.

### 6. Protected Project and Output Directories
- Manifests and runtime state are strictly confined to `roop_3d/.runtime/` and `roop_3d/outputs/`.
- Project files (`.r3dp`) require exact 32-character hexadecimal identifiers.
- Writes are guarded by exclusive file locks (`exclusive()`) and atomic file replacement (`atomic_json()`).

### 7. Unsafe Archive Extraction Prevention
- Archive extraction to the filesystem is intentionally **not supported**.
- `FaceSetReader` parses FaceSet archives in-memory and via binary streaming.
- Model downloads reject archive formats (`.zip`, `.tar`, `.gz`, `.7z`) prior to network access.
  No model code or installer is unpacked or executed.

### 8. Protection Against Malicious Model Files
- PyTorch pickle checkpoints are blocked from inference (`ModelCompatibilityScanner`).
- Tensor loading uses safe serialization formats: SAFETENSORS and ONNX.
- When PyTorch loading is necessary, `load_restricted_torch_weights()` strictly enforces
  `weights_only=True` and fails closed on older runtime versions.
- Every model download is verified against registered SHA-256 hashes and exact byte sizes.
  Corrupt or modified cache files are quarantined.

### 9. Sanitization of Errors and Application Logs
- Global exception handlers in FastAPI catch unhandled exceptions and return generic
  `INTERNAL_ERROR` envelopes with a correlation `request_id`, suppressing tracebacks and system paths.
- `SecurityRedactionFilter` is installed on application loggers to scrub:
  - Bearer tokens, access tokens, and `roop3d_token` cookies.
  - API keys, consent tokens, and cryptographic signatures.
  - PEM private key blocks.
  - 512-dimensional biometric embedding arrays.
  - Sensitive query string parameters.

### 10. Credentials Outside Project Manifests
- `ProjectManifest` schemas forbid credentials, tokens, or private keys.
- `ProjectStore.save()` and `ProjectStore.load()` run regex audits over the raw JSON
  to ensure keys matching `api_key`, `secret`, `password`, `consent_token`, `token`, or `private_key`
  fail closed with `PROJECT_CREDENTIALS_REJECTED`.

### 11. API Origins and WebSocket Connection Validation
- `IntakeMiddleware` checks the `Origin` header against permitted loopback origins
  (`http://127.0.0.1:8030`, `http://localhost:8030`, and dev port `5174` if `dev_cors` is enabled).
- Repeated security-sensitive headers (`Host`, `Origin`, `Content-Length`) trigger immediate rejection (`AMBIGUOUS_HEADERS`).
- WebSocket endpoints `/ws/telemetry`, `/ws/events`, and `/ws/queue` enforce origin validation.
  The event stream is server-to-client only; incoming client payloads trigger close code 1008.

### 12. Unsafe Local-Provider URL Redirects
- `ComfyUIBridge` uses `LoopbackConnect` to reject redirects during WebSocket handshake.
- HTTP requests via `httpx.AsyncClient` configure `follow_redirects=False` and `trust_env=False`.
- Endpoints are validated as loopback origins; credentials and query parameters are forbidden.
- The model downloader blocks all redirects, except the strictly reviewed Hugging Face LFS storage host (`us.aws.cdn.hf.co`).

### 13. Resource Exhaustion Protection
- Workstation frame extraction enforces a strict 15-second decode budget (`DECODE_BUDGET_EXCEEDED`).
- `JobWorker` caps concurrency with `max_concurrent_jobs` and thread pool limits.
- `VRAMGovernor` dynamically manages tensor execution, falling back to CPU execution upon OOM
  via checkpointed state.
- Pipelines support cooperative frame-by-frame cancellation checks (`check_cancel()`).

### 14. Safe Deletion of Temporary Media and Biometric Artifacts
- `SafeArtifactCleaner` provides verified deletion:
  - Partial upload sessions (`.part`, `.json`, `.lock`) are purged safely.
  - Job workspaces (`roop_3d/.runtime/jobs/{job_id}`) are removed bottom-up after symlink validation.
  - FaceSet deletion purges all manifests, coverage data, `.fsz` archives, and reconstructed 3D
    identity assets (`roop_3d/.runtime/identity_assets/{faceset_id}`).
  - Stale temporary uploads and caches are purgeable on demand via `/api/v1/security/purge-artifacts`.

### 15. Checkpoint and Project Restoration Verification
- `ProjectStore.load()` verifies file sizes (<= 512 KiB) and confirms that all input target,
  inspection, FaceSet, and model hashes match the registered state on disk.
- Resuming jobs verifies `PipelineCheckpoint`:
  - Matches `job_id`, `input_sha256`, `request_sha256`, and `settings_sha256`.
  - Verifies that all completed frame images exist and are non-empty.
  - Uses `read_checkpoint_npz()` with `allow_pickle=False`.

---

## Privacy Architecture

1. **Source Identity Likeness Permission**:
   - `ConsentVerifier.verify_source_consent()` ensures that likeness permission (`consent_granted=True`)
     is explicitly present.
   - Operations are checked against `permitted_operations` (`swap`, `reenactment`, `3d_reconstruction`).
2. **Target Footage Lawful Use**:
   - `ConsentVerifier.verify_target_permission()` requires lawful use attestation and binds permissions
     to the target file's SHA-256 digest.
3. **Zero Hidden Network Egress**:
   - All cloud providers are disabled by default (`CloudStub`).
   - Remote execution requires per-job opt-in with a short-lived (1-900s), HMAC-SHA256 signed token
     from `ConsentAuthority` detailing the exact file manifests and destination host.

---

## Synthetic Media Governance & Deceptive Impersonation Safeguards

### Visible Synthetic-Media Disclosure
Every rendered output manifest includes a mandatory provenance section:
```json
"provenance": {
  "synthetic_media": true,
  "processor": "Roop 3D",
  "processing": "local",
  "transformed": true,
  "notice": "Synthetic media processed locally by Roop 3D"
}
```
The synthetic-media notice parameter cannot be disabled in standard configurations (`disabled_parameters`).

### Truthful Identity & Cryptographic Manifests
- **The Identity Honesty Principle**:
  > Ordinary SHA-256 integrity hashes establish artifact byte integrity only; they must **never** be
  > described as cryptographic proof of a person's legal or biometric identity.
- Provenance verification explicitly reports:
  ```json
  "identity_proof": false,
  "identity_notice": "A SHA-256 hash establishes artifact byte integrity only; it does not prove a person's legal or biometric identity."
  ```
- `IdentityHonestyPolicy.audit_claim()` rejects deceptive assertions claiming hash digests prove person identity.
- Where configured, cryptographic manifests use HMAC-SHA256 (`ROOP3D_PROVENANCE_KEY`) or ES256 C2PA
  signing (`ROOP3D_C2PA_CERT_PEM` and `ROOP3D_C2PA_KEY_PEM`) without remote ingredients.

### Safeguards Against NCII and Deceptive Impersonation
- `SafetyGuard.inspect_request()` monitors input parameters, notes, and FaceSet labels.
- Requests containing terms or intent associated with Non-Consensual Intimate Imagery (NCII)
  (e.g., `deepnude`, `undress`, `revenge_porn`, `intimate_leak`) or deceptive impersonation
  fail closed with `SAFETY_POLICY_VIOLATION` (HTTP 403).

---

## STRIDE Threat Modeling Matrix

| Threat Category | Specific Threat / Attack Vector | Severity | Mitigation Implemented | Verification Test |
| :--- | :--- | :--- | :--- | :--- |
| **Spoofing** | Foreign Origin / CSRF browser hijacking | High | `IntakeMiddleware` host & origin check; cookie `SameSite=strict` | `test_api_origin_validation` |
| **Spoofing** | Disguised executable payload in uploads | Critical | `validate_video_container`, `validate_image_header` magic-byte checks | `test_validate_video_container_magic_bytes` |
| **Spoofing** | Misrepresenting SHA-256 as person identity proof | Medium | `IdentityHonestyPolicy` reject deceptive identity claims | `test_identity_honesty_rejects_sha256_as_identity_proof` |
| **Tampering** | Zip-bomb decompression attack via FaceSet | High | `ArchiveValidator` ratio cap (20x), size cap (256 MB), entry cap (512) | `test_archive_validator_zip_bomb_defense` |
| **Tampering** | Symlink / junction path escape | High | `validate_safe_path`, `_check_symlink` attribute check | `test_symlink_rejected`, `test_validate_safe_path_traversal` |
| **Tampering** | Checkpoint state tampering or format alteration | High | `PipelineCheckpoint` hash binding, `allow_pickle=False` | `test_checkpoint_restoration_verification` |
| **Repudiation** | Unattributed synthetic video generation | Medium | Compulsory `.manifest.json` with input/output SHA-256 & models | `test_synthetic_media_disclosure_and_manifest` |
| **Information Disclosure** | Secret / token leakage in application logs | High | `SecurityRedactionFilter` scrubbing tokens, keys & embeddings | `test_log_sanitizer_redacts_tokens_and_biometrics` |
| **Information Disclosure** | Credentials stored in project manifests | High | `ProjectStore` regex audit rejecting `api_key`, `secret`, `token` | `test_project_manifest_rejects_credential_injection` |
| **Information Disclosure** | Unintended cloud egress of biometric data | Critical | `ConsentVerifier` egress policy; `CloudStub` disabled adapter | `test_external_egress_policy_blocks_non_loopback` |
| **Denial of Service** | Oversized video decoding / infinite loop | Medium | 15-second decode budget in frame inspection | `test_stage02_media.py` |
| **Denial of Service** | GPU VRAM exhaustion during batch jobs | High | `VRAMGovernor` dynamic tracking and automatic CPU fallback | `test_stage12_governor.py` |
| **Elevation of Privilege**| Arbitrary code execution via pickle weights | Critical | PyTorch pickle blocked; ONNX/safetensors data formats; `weights_only=True` | `test_pickle_serialization_rejected_for_inference` |

---

## Acceptance Verification

1. **Python Test Suite**:
   - `test_stage13_security.py`: **26/26 passed**.
   - Full test suite: **967 passed, 38 skipped, 18 subtests passed**.
2. **Frontend Build**:
   - `npm run build`: **passed** (typecheck, lint, and Vite build succeeded).
3. **Model Registry Audit**:
   - `audit/local_models.py --verify`: **95/95** verified, 0 violations.
4. **Legacy Isolation Audit**:
   - `audit/verify_isolation.py --report`: **2,645/2,645** historical entries and **68,390/68,390** comprehensive protected entries verified, 0 violations.
