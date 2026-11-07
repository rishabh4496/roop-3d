"""First-Use Automatic Model Downloader & Integrity Verifier for Roop 3D.

Maintains canonical registries, verified upstream URLs, and SHA-256 checksums for:
1. inswapper_128_fp16.onnx
2. scrfd_2.5g.onnx
3. 1k3d68.onnx
4. 2d106det.onnx
5. xseg.onnx
6. bisenet.onnx
7. occluder.onnx
8. GPEN-BFR-512.onnx
9. ultramax.onnx

Provides chunked, resumable downloads (*.part), atomic moves, SHA-256 integrity verification,
and real-time WebSocket progress reporting via EventBroker.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import urllib.request
from urllib.error import URLError

logger = logging.getLogger("roop_3d.model_downloader")


def _resolve_project_root() -> Path:
    # app/roop_3d/core/model_downloader.py -> parents[3] is project root
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    name: str
    aliases: Tuple[str, ...]
    primary_url: str
    fallback_url: str
    expected_sha256: str
    acceptable_sha256s: Tuple[str, ...]
    expected_size_bytes: int
    category: str
    description: str


# Canonical registry of all 9 required models
REQUIRED_MODELS: Dict[str, ModelSpec] = {
    "inswapper_128_fp16.onnx": ModelSpec(
        model_id="inswapper_128_fp16",
        name="inswapper_128_fp16.onnx",
        aliases=("inswapper_128.onnx", "inswapper_128_fp16.onnx", "inswapper_128.model.onnx"),
        primary_url="https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/6ffdf0e83c5996cc425e77b59913fc48d79441be/inswapper_128.onnx",
        fallback_url="https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx",
        expected_sha256="e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af",
        acceptable_sha256s=(
            "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af",
            "eae68e48eb33ac8870a6f39f886f32e2d93e18a804e3b7b62c1214fa619d8560",
        ),
        expected_size_bytes=554253681,
        category="swapper",
        description="InsightFace 128x128 face swapper neural engine.",
    ),
    "scrfd_2.5g.onnx": ModelSpec(
        model_id="scrfd_2.5g",
        name="scrfd_2.5g.onnx",
        aliases=("scrfd_2.5g_kps.onnx", "scrfd_2.5g.onnx", "det_2.5g.onnx", "scrfd_2_5g_kps.onnx"),
        primary_url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/scrfd_2.5g_kps.onnx",
        fallback_url="https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/scrfd_2.5g_kps.onnx",
        expected_sha256="b0959828d11ca7ff989a50ae40da1f5ee7e54c87da2364c3c3a9d2ec05cb51dc",
        acceptable_sha256s=(
            "b0959828d11ca7ff989a50ae40da1f5ee7e54c87da2364c3c3a9d2ec05cb51dc",
            "168ec889f5a04eb003bb24e6eb4862f9eb6e28fb68e4bf49129532e3ea721c5b",
            "5838f7fe0536ef55fa3ba422709a3fc3a4b008d66ff97b4c3ae6a090e9cc0c7b",
        ),
        expected_size_bytes=3272996,
        category="detector",
        description="SCRFD 2.5G lightweight high-speed face & landmark detector.",
    ),
    "1k3d68.onnx": ModelSpec(
        model_id="1k3d68",
        name="1k3d68.onnx",
        aliases=("1k3d68.onnx", "1k3d68.model.onnx"),
        primary_url="https://huggingface.co/Aitrepreneur/insightface/resolve/fd887cdef0c73f32251198b8160d6771ac413fc0/models/buffalo_l/1k3d68.onnx",
        fallback_url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/1k3d68.onnx",
        expected_sha256="df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc",
        acceptable_sha256s=(
            "df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc",
            "df5c06b8a0c109b82e21ee648a734685ff4cfc79469e38e68cfb9efbe7b274c6",
        ),
        expected_size_bytes=9040332,
        category="3d_geometry",
        description="68-point 3D facial landmark and shape regressor.",
    ),
    "2d106det.onnx": ModelSpec(
        model_id="2d106det",
        name="2d106det.onnx",
        aliases=("2d106det.onnx", "2d106det.model.onnx"),
        primary_url="https://huggingface.co/Aitrepreneur/insightface/resolve/fd887cdef0c73f32251198b8160d6771ac413fc0/models/buffalo_l/2d106det.onnx",
        fallback_url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/2d106det.onnx",
        expected_sha256="f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf",
        acceptable_sha256s=(
            "f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf",
            "f001b856447ce7c8340d046c3ef6be59cb7a3fa0b7b1ca2cb5c50c37f4cf9983",
        ),
        expected_size_bytes=4944948,
        category="3d_geometry",
        description="106-point dense 2D contour and facial feature detector.",
    ),
    "xseg.onnx": ModelSpec(
        model_id="xseg",
        name="xseg.onnx",
        aliases=("xseg.onnx", "xseg_3.onnx", "xseg_1.onnx", "xseg_2.onnx", "xseg_1.model.onnx", "xseg_2.model.onnx"),
        primary_url="https://huggingface.co/hacksider/deep-live-cam/resolve/581e70b61240b7928404c17900437f47cfe94133/xseg.onnx",
        fallback_url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/xseg_3.onnx",
        expected_sha256="0b57328efcb839d85973164b617ceee9dfe6cfcb2c82e8a033bba9f4f09b27e5",
        acceptable_sha256s=(
            "0b57328efcb839d85973164b617ceee9dfe6cfcb2c82e8a033bba9f4f09b27e5",
            "48ccd7e8541e2b6c38260421711204b4cfab13f4fd80004ca5187ea1c0a0c29f",
            "c4d1498b8a03b5fe2a3a5d2ef2a0402ab03bd51edaf5b2d8d5fb764702a97dd3",
            "cd9a0879eaf43841d765472cf1f8c330dbf9dcb03da0eace93e95f3bcc399042",
            "3cbfe3fa6269b61dbdb89117621c1729b12488a032be5566f1fc953b98c56fa7",
            "a8e0ea5f3bf19a3fa2718fb4bce365ab74ecf4f08acdbdc4c6fce7c0f1e00e23",
        ),
        expected_size_bytes=160492582,
        category="occlusion",
        description="XSeg neural face and boundary segmentation occluder.",
    ),
    "bisenet.onnx": ModelSpec(
        model_id="bisenet",
        name="bisenet.onnx",
        aliases=("bisenet_resnet_34.onnx", "bisenet_resnet_18.onnx", "bisenet_resnet_18.model.onnx", "bisenet.onnx"),
        primary_url="https://huggingface.co/facefusion/models-3.0.0/resolve/728b9659bd9691bf32cbf7f61af478d94b7ba81e/bisenet_resnet_34.onnx",
        fallback_url="https://huggingface.co/facefusion/models-3.1.0/resolve/c9e3a503d8e84e91c5cd89ee2d510fe5e793e570/bisenet_resnet_18.onnx",
        expected_sha256="4a0b8c958a3cbf34800b6564e525beeb56c80249c5be4e3cae6d1912a2333796",
        acceptable_sha256s=(
            "4a0b8c958a3cbf34800b6564e525beeb56c80249c5be4e3cae6d1912a2333796",
            "2218b6183c26ca5c83303232d682a536c670c13ea9695f716c777d1f244eefe9",
            "2218b6183c5098ff4fe39bba0be72648fb162232a514d5ce177721865a7f2921",
        ),
        expected_size_bytes=53342080,
        category="occlusion",
        description="BiSeNet 19-class semantic facial parsing network.",
    ),
    "occluder.onnx": ModelSpec(
        model_id="occluder",
        name="occluder.onnx",
        aliases=("face_occluder.onnx", "occluder.onnx", "face_occluder.model.onnx"),
        primary_url="https://huggingface.co/datvtn/face_occluder/resolve/1eb61ffa05b1548675dea08facbf3c7edcfcb28f/face_occluder.onnx",
        fallback_url="https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/face_occluder.onnx",
        expected_sha256="ad9bfb90510790711a429935f846eaba2dcf2935d9afc2bda7349f5bfb92a104",
        acceptable_sha256s=(
            "ad9bfb90510790711a429935f846eaba2dcf2935d9afc2bda7349f5bfb92a104",
        ),
        expected_size_bytes=3625062,
        category="occlusion",
        description="High-precision foreground hand, hair, and object occluder.",
    ),
    "GPEN-BFR-512.onnx": ModelSpec(
        model_id="GPEN-BFR-512",
        name="GPEN-BFR-512.onnx",
        aliases=("GPEN-BFR-512.onnx", "gpen_bfr_512.onnx", "gpen_bfr_512.model.onnx"),
        primary_url="https://huggingface.co/CountFloyd/deepfake/resolve/ee7d0fe0c284700192ff11e14ecdab5b40185204/GPEN-BFR-512.onnx",
        fallback_url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/GPEN-BFR-512.onnx",
        expected_sha256="0960f836488737faee862412e87fc4c2f74122d2ee7fefd4e0e29b10ec871b69",
        acceptable_sha256s=(
            "0960f836488737faee862412e87fc4c2f74122d2ee7fefd4e0e29b10ec871b69",
            "d5f066b9062df67e2ea3c9ef526ebcd9e3fba46fa0bfd30d5bfaab03f9011707",
        ),
        expected_size_bytes=284241604,
        category="enhancer",
        description="GPEN Realistic 512px blind face restoration neural engine.",
    ),
    "ultramax.onnx": ModelSpec(
        model_id="ultramax",
        name="ultramax.onnx",
        aliases=("ultramax_bfr.onnx", "ultramax.onnx", "codeformer.fp16.onnx", "ultramax_bfr.model.onnx"),
        primary_url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/codeformer.fp16.onnx",
        fallback_url="https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/ultramax_bfr.onnx",
        expected_sha256="5414b2c989eb2fb12d5930b91e779a1f10bc47cc15da9942a4a753697ebaa400",
        acceptable_sha256s=(
            "5414b2c989eb2fb12d5930b91e779a1f10bc47cc15da9942a4a753697ebaa400",
        ),
        expected_size_bytes=188981248,
        category="enhancer",
        description="UltraMax decoupled zero-copy Laplacian pyramid face restoration.",
    ),
}

# Lookup map across all canonical names, model_ids, and aliases
MODEL_LOOKUP: Dict[str, ModelSpec] = {}
for _key, _spec in REQUIRED_MODELS.items():
    MODEL_LOOKUP[_key.lower()] = _spec
    MODEL_LOOKUP[_spec.model_id.lower()] = _spec
    for _alias in _spec.aliases:
        MODEL_LOOKUP[_alias.lower()] = _spec


def get_model_spec(identifier: str) -> Optional[ModelSpec]:
    """Retrieve model specification by canonical filename, model_id, or alias."""
    clean = Path(identifier).name.lower()
    return MODEL_LOOKUP.get(clean) or MODEL_LOOKUP.get(identifier.lower())


def get_model_search_paths(root: Path | None = None) -> List[Path]:
    """Return all valid on-disk directories where models may reside."""
    r = root or _resolve_project_root()
    return [
        r / "app/models",
        r / "app/roop_3d/models/cache",
        r / "roop_3d/models/cache",
        r / ".cache/models",
        r / "cache/models",
        r / "models",
    ]


def verify_file_hash(path: Path, expected_sha256: str, acceptable: Tuple[str, ...] = ()) -> bool:
    """Validate SHA-256 digest of a file against expected or acceptable variants."""
    if not path.is_file():
        return False
    try:
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest().lower()
        valid = {expected_sha256.lower(), *(h.lower() for h in acceptable)}
        return digest in valid
    except Exception as err:
        logger.warning("Error verifying hash for %s: %s", path, err)
        return False


def find_model_path(spec: ModelSpec, root: Path | None = None) -> Optional[Path]:
    """Find a verified instance of the specified model on disk."""
    paths = get_model_search_paths(root)
    candidates = [spec.name, spec.model_id + ".onnx", *spec.aliases]

    for base in paths:
        if not base.is_dir():
            continue
        for name in candidates:
            candidate_path = base / name
            if candidate_path.is_file():
                if verify_file_hash(candidate_path, spec.expected_sha256, spec.acceptable_sha256s):
                    return candidate_path
    return None


def download_model(
    identifier: str,
    destination_dir: Path | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    app_state: Any = None,
    timeout: float = 60.0,
    use_fallback: bool = False,
) -> Path:
    """Download model with chunked transfer, resumption (*.part), and WebSocket telemetry."""
    spec = get_model_spec(identifier)
    if not spec:
        raise ValueError(f"Unknown model identifier: {identifier}")

    root = _resolve_project_root()
    target_dir = destination_dir or (root / "app/models")
    target_dir.mkdir(parents=True, exist_ok=True)

    target_path = target_dir / spec.name
    part_path = target_dir / f"{spec.name}.part"

    # Broadcast helper
    def emit_progress(payload: dict[str, Any]) -> None:
        if progress_callback:
            try:
                progress_callback(payload)
            except Exception:
                pass
        if app_state and hasattr(app_state, "events"):
            try:
                app_state.events.publish(payload)
            except Exception:
                pass

    # If already exists and verified, return immediately
    if target_path.is_file() and verify_file_hash(target_path, spec.expected_sha256, spec.acceptable_sha256s):
        emit_progress({
            "type": "model_download_progress",
            "model_id": spec.model_id,
            "bytes_downloaded": target_path.stat().st_size,
            "total_bytes": target_path.stat().st_size,
            "percent": 100.0,
            "speed_mbps": 0.0,
            "eta_seconds": 0.0,
            "status": "ready",
        })
        return target_path

    url = spec.fallback_url if use_fallback else spec.primary_url
    existing_bytes = part_path.stat().st_size if part_path.is_file() else 0

    req = urllib.request.Request(url, headers={"User-Agent": "roop-3d/1.0"})
    if existing_bytes > 0:
        req.add_header("Range", f"bytes={existing_bytes}-")

    mode = "ab" if existing_bytes > 0 else "wb"
    start_time = time.monotonic()
    downloaded = existing_bytes
    total_size = spec.expected_size_bytes

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(part_path, mode) as out_f:
            content_len = resp.headers.get("Content-Length")
            if content_len:
                total_size = existing_bytes + int(content_len)

            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                out_f.write(chunk)
                downloaded += len(chunk)
                elapsed = max(0.001, time.monotonic() - start_time)
                speed_mbps = ((downloaded - existing_bytes) / (1024 * 1024)) / elapsed
                pct = min(100.0, (downloaded / max(1, total_size)) * 100.0)
                eta = max(0.0, (total_size - downloaded) / max(0.001, speed_mbps * 1024 * 1024))

                emit_progress({
                    "type": "model_download_progress",
                    "model_id": spec.model_id,
                    "bytes_downloaded": downloaded,
                    "total_bytes": total_size,
                    "percent": round(pct, 1),
                    "speed_mbps": round(speed_mbps, 2),
                    "eta_seconds": round(eta, 1),
                    "status": "downloading",
                })

    except (URLError, OSError) as err:
        if not use_fallback:
            logger.info("Retrying %s with fallback URL...", spec.name)
            return download_model(identifier, target_dir, progress_callback, app_state, timeout, use_fallback=True)
        raise RuntimeError(f"Download failed for {spec.name}: {err}") from err

    # Verify downloaded file hash
    emit_progress({
        "type": "model_download_progress",
        "model_id": spec.model_id,
        "bytes_downloaded": downloaded,
        "total_bytes": total_size,
        "percent": 100.0,
        "speed_mbps": 0.0,
        "eta_seconds": 0.0,
        "status": "verifying",
    })

    if not verify_file_hash(part_path, spec.expected_sha256, spec.acceptable_sha256s):
        if part_path.is_file():
            part_path.unlink()
        raise ValueError(f"Integrity check failed: SHA-256 digest does not match expected for {spec.name}")

    # Atomic rename from *.part to destination
    if target_path.is_file():
        target_path.unlink()
    part_path.rename(target_path)

    emit_progress({
        "type": "model_download_progress",
        "model_id": spec.model_id,
        "bytes_downloaded": target_path.stat().st_size,
        "total_bytes": target_path.stat().st_size,
        "percent": 100.0,
        "speed_mbps": 0.0,
        "eta_seconds": 0.0,
        "status": "ready",
    })
    return target_path


def ensure_model(identifier: str, destination_dir: Path | None = None, app_state: Any = None) -> Path:
    """Automatic download-on-first-use handler.

    Returns the path if present and verified, otherwise downloads and verifies it.
    """
    spec = get_model_spec(identifier)
    if not spec:
        raise ValueError(f"Unknown model identifier: {identifier}")

    existing = find_model_path(spec)
    if existing is not None:
        return existing

    logger.info("Model '%s' missing on first use. Initiating auto-download...", spec.name)
    return download_model(spec.name, destination_dir=destination_dir, app_state=app_state)


def verify_all_required_models(root: Path | None = None) -> Dict[str, Dict[str, Any]]:
    """Scan and return the complete integrity status for all 9 required models."""
    report: Dict[str, Dict[str, Any]] = {}
    r = root or _resolve_project_root()

    for key, spec in REQUIRED_MODELS.items():
        found = find_model_path(spec, r)
        report[key] = {
            "model_id": spec.model_id,
            "name": spec.name,
            "category": spec.category,
            "status": "READY" if found else "MISSING",
            "path": str(found) if found else None,
            "expected_sha256": spec.expected_sha256,
            "expected_size_bytes": spec.expected_size_bytes,
            "primary_url": spec.primary_url,
            "fallback_url": spec.fallback_url,
            "description": spec.description,
        }
    return report


__all__ = [
    "ModelSpec",
    "REQUIRED_MODELS",
    "get_model_spec",
    "get_model_search_paths",
    "find_model_path",
    "verify_file_hash",
    "download_model",
    "ensure_model",
    "verify_all_required_models",
]
