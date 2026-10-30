# Stage 04: FaceSet reader, pose selector and V2 schema

Implemented in the independent `roop-3d` checkout, under `app/roop_3d/`.
All source changes stay inside `app/roop_3d/faceset/`, `app/roop_3d/models/catalog.py`,
`app/roop_3d/tests/`, and `app/roop_3d/audit/`. No launchers, sibling application files,
environments, protected baselines, or root policy files were modified.

## Delivered behavior

### 1. Archive security validator (`faceset/validator.py`)
- Central directory headers are inspected before opening or extracting any entry.
- Hard limits:
  - Max entries: 512.
  - Max single entry uncompressed size: 32 MiB.
  - Max total uncompressed memory budget: 256 MiB.
  - Compression ratio limit (zip-bomb defense): 20.0 max expansion ratio for entries larger than 64 KiB.
- Security defenses:
  - Path traversal rejection (`..`, `../`, `..\\`).
  - Absolute path rejection (POSIX `/`, Windows drive `C:\`, UNC `\\`).
  - Null-byte filename rejection (`\x00`).
  - Windows reserved device rejection (`CON`, `PRN`, `AUX`, `NUL`, `COM1..9`, `LPT1..9`).
  - Symbolic links and Windows reparse points / junctions rejected fail-closed via `stat.S_IFLNK` and attribute `0x400`.
  - Disallowed payload extensions (`.pkl`, `.pickle`, `.py`, `.exe`, `.dll`, `.so`, `.sh`, `.bat`, etc.) rejected fail-closed.
- Directory validation (`validate_directory`): validates directories of face images against symlinks, traversal, budgets, and extension allowlists.

### 2. Legacy format normalization (`faceset/faceset_reader.py`)
- Archives are opened strictly in read-only binary mode (`"rb"`).
- SHA-256 digests and file metadata are checked before and after ingestion to guarantee zero byte mutation on input fixtures.
- Legacy V2 normalization:
  - Parses `metadata.json` containing `sources`, precomputed 512-D ArcFace embeddings, 2D landmarks, 3D head poses (`[yaw, pitch, roll]`), expression openness, and sharpness metrics into an immutable `FaceSetDocument`.
- Legacy V1 normalization:
  - Ingests archives containing only sequential raw face crops (`0.png`, `1.png`, ...).
  - Lazily infers face landmarks and bounding boxes using Stage 03's verified local SCRFD detector.
  - Estimates geometric yaw, pitch, eye-line roll, and Laplacian variance sharpness.
  - Computes 512-D L2-normalized ArcFace embeddings using Stage 03's verified local `w600k_r50.onnx` recognizer running on `CPUExecutionProvider`.
  - Caches inferred metadata atomically under `roop_3d/.runtime/cache/facesets/<archive_sha256>/metadata.json`. Subsequent reads hit cache with verified digests.
- Image directory ingestion:
  - Scans and normalizes directories of face image files into `FaceSetDocument`.

### 3. Identity aggregation & canonical embeddings
- Computes both:
  1. **Canonical Mean Embedding**: L2-normalized mean 512-D embedding across high-confidence views (`quality.score >= 0.5` and `detector_confidence >= 0.5`).
  2. **Medoid Embedding**: The view in the high-confidence set maximizing total pairwise cosine similarity to all other views in the set, L2-normalized.
- Retains per-view 512-D embeddings on every `FaceView`.

### 4. Dynamic pose-bin coverage & nearest-view selector (`faceset/pose_selector.py`)
- Classifies head poses into discrete bins across yaw (`left`, `center`, `right`) and pitch (`up`, `center`, `down`) forming compound bins (e.g. `center_center`, `left_center`, `right_up`).
- `PoseSelector.select_best_reference(target_yaw, target_pitch, target_roll, target_expression)`:
  - Fast query method evaluating angular delta, circular roll wrap-around, expression openness delta, and quality score weighting.
  - Returns `ReferenceMatch` with observed crop bytes, landmarks, confidence score in `[0.0, 1.0]`, pose metadata, and `is_synthetic: False`.

### 5. Honest provenance & V2 schema export (`faceset/schemas.py`)
- Pydantic v2 immutable contracts (`Contract` with `frozen=True`, `extra="forbid"`, `allow_inf_nan=False`).
- Honest provenance: all views are explicitly enforced as `is_synthetic: Literal[False] = False`. No 3D meshes or synthetic views are fabricated or labeled as observed.
- Versioned V2 manifest schema (`FaceSetV2Manifest`) supporting:
  - `schema_version = 2`, `schema = "roop.fsz.v2"`.
  - Explicit pose bins with representative views and support counts.
  - Mandatory `ConsentMetadata` (`consent_granted`, `consent_timestamp`, `commercial_use_allowed`, `notes`).
  - Optional Stage 08 UV map field (`uv_map_path`).
- `export_v2()` writes exports exclusively to new files inside `roop_3d/outputs/facesets/` using atomic writes (`.tmp` replace) and strictly refuses to overwrite existing files.

---

## Verification Evidence

All validation commands executed from this checkout's `app/` directory:

| Command | Exit Code | Result |
| :--- | :---: | :--- |
| `roop_3d/.venv/Scripts/python.exe -B -m pytest -c roop_3d/pyproject.toml --confcutdir=roop_3d roop_3d/tests -q` | `0` | **207 passed**, 18 subtests passed, 1 warning (existing upstream TestClient deprecation) |
| `roop_3d/.venv/Scripts/python.exe -B -m roop_3d.tests.manual_stage04_acceptance` | `0` | Real discovered legacy V2 (`output/faceset.fsz`) and V1 (`akansha.fsz`) verified; zero mutations; V2 export roundtrip passed |
| `roop_3d/.venv/Scripts/python.exe -B roop_3d/audit/verify_isolation.py --report` | `0` | **68,392** comprehensive + **2,647** historical immutable files matched byte-for-byte; **0 violations** |
| `roop_3d/.venv/Scripts/python.exe -B roop_3d/audit/local_models.py --verify` | `0` | **95/95** model hashes verified intact; **0 copies, 0 downloads** |

The product is now ready for Stage 05/06 (face swapping and LivePortrait conditioning pipelines).
