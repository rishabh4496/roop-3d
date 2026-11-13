"""Hash and register local weight artifacts without loading legacy Python code.

VERIFIED_LOCAL means bytes were read locally and hashed. It is deliberately
independent of licensing, upstream provenance, and successful inference.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import io
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import unquote, urljoin, urlsplit
import uuid

# Bind to the copy of ``isolation`` that matches how THIS module was loaded. Loaded as ``roop_3d.audit.local_models`` it must use
# ``roop_3d.audit.isolation`` even when another module has put audit/ on sys.path; loaded as a script or a bare ``local_models``
# it uses the sibling ``isolation``. Mixing them gives two AuditError classes and ``except AuditError`` silently misses.
if __package__:
    from roop_3d.audit.isolation import AuditError, is_link, relative_path, safe_path, sha256_file, write_json, write_text
else:
    from isolation import AuditError, is_link, relative_path, safe_path, sha256_file, write_json, write_text
try:
    from project_paths import legacy_root
except ImportError:  # imported as roop_3d.audit.local_models (the audit directory is not on sys.path)
    from roop_3d.audit.project_paths import legacy_root


class EvidenceState(str, Enum):
    VERIFIED_LOCAL = "VERIFIED_LOCAL"
    VERIFIED_REMOTE = "VERIFIED_REMOTE"
    CANDIDATE = "CANDIDATE"
    UNKNOWN = "UNKNOWN"
    BLOCKED = "BLOCKED"
    UNSUPPORTED = "UNSUPPORTED"


WEIGHT_EXTENSIONS = {".onnx", ".pth", ".pt", ".ckpt", ".safetensors", ".bin"}
PRUNE = {".git", "env", "venv", ".venv", "node_modules", "__pycache__",
         "roop_3d", "roop_3d_ui", "output", "outputs", "temp", "logs", "site-packages"}


def family_hint(name: str) -> str:
    name = name.casefold()
    groups = (
        ("lipsync", ("wav2lip", "musetalk")),
        ("animation", ("liveportrait",)),
        ("swapper", ("inswapper", "hyperswap", "hififace", "simswap")),
        ("detector", ("scrfd", "retinaface", "det_10g", "yoloface", "yunet")),
        ("recognizer", ("arcface", "w600k", "glintr", "adaface", "sface")),
        ("parser", ("bisenet", "xseg", "resnet18", "face_occluder")),
        ("restorer", ("gfpgan", "codeformer", "gpen", "restoreformer", "dmdnet")),
    )
    return next((label for label, hints in groups if any(h in name for h in hints)), "other")


def graph_metadata(path: Path) -> dict[str, Any]:
    if path.suffix.casefold() != ".onnx":
        return {"state": "UNKNOWN", "reason": "Not deserialized; pickle execution is forbidden during discovery"}
    try:
        import onnx
        # External tensor paths are reported as graph metadata, never followed.
        model = onnx.load_model(str(path), load_external_data=False)
        initializers = {tensor.name for tensor in model.graph.initializer}

        def tensor_info(value: Any) -> dict[str, Any]:
            tensor = value.type.tensor_type
            return {"name": value.name, "dtype": onnx.TensorProto.DataType.Name(tensor.elem_type),
                    "shape": [d.dim_value if d.HasField("dim_value") else
                              d.dim_param if d.HasField("dim_param") else None
                              for d in tensor.shape.dim]}

        return {"state": "VERIFIED_LOCAL", "evidence": "ONNX graph metadata parsed locally; no inference",
                "onnx_version": onnx.__version__,
                "inputs": [tensor_info(v) for v in model.graph.input if v.name not in initializers],
                "outputs": [tensor_info(v) for v in model.graph.output],
                "opsets": [{"domain": op.domain, "version": op.version} for op in model.opset_import],
                "external_data": [{"tensor": t.name, "entries": {e.key: e.value for e in t.external_data}}
                                  for t in model.graph.initializer if t.external_data]}
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        return {"state": "UNKNOWN", "reason": f"{type(error).__name__}: {error}"}
    except Exception as error:
        # Corrupt protobufs are evidence of failed inspection, never valid tensor specs.
        return {"state": "UNKNOWN", "reason": f"Graph inspection failed: {type(error).__name__}: {error}"}


def discover(scopes: Mapping[str, Path], expected_hashes: Mapping[str, str] | None = None) -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    seen: set[str] = set()
    scope_evidence = []
    expected_hashes = expected_hashes or {}
    for scope, base in scopes.items():
        if not base.exists():
            scope_evidence.append({"scope": scope, "state": "UNKNOWN", "reason": "Directory absent"})
            continue
        safe_path(base, "probe")
        scope_evidence.append({"scope": scope, "state": "VERIFIED_LOCAL", "evidence": "Directory scanned read-only"})

        def onerror(error: OSError) -> None:
            raise AuditError("MODEL_SCAN_FAILED", str(error))

        for parent, directories, filenames in os.walk(base, followlinks=False, onerror=onerror):
            folder = Path(parent)
            directories[:] = sorted(d for d in directories
                                    if d.casefold() not in PRUNE and not is_link(folder / d))
            for filename in sorted(filenames):
                path = folder / filename
                if path.suffix.casefold() not in WEIGHT_EXTENSIONS or is_link(path):
                    continue
                if path.suffix.casefold() == ".bin" and not (
                    filename.casefold() == "model.bin" or filename.casefold().endswith("_model.bin")
                ):
                    # TensorRT timing caches and executables are not weight files.
                    continue
                identity = os.path.normcase(str(path.absolute()))
                if identity in seen:
                    continue
                seen.add(identity)
                size, digest = sha256_file(path)
                if identity in expected_hashes and expected_hashes[identity] != digest:
                    raise AuditError("MODEL_DRIFT", str(path))
                models.append({"name": filename, "locator": {"scope": scope, "path": path.relative_to(base).as_posix()},
                               "state": "VERIFIED_LOCAL", "size_bytes": size, "sha256": digest,
                               "evidence": "Local file bytes read and SHA-256 computed; not a download checksum",
                               "family_hint": family_hint(filename), "family_evidence": "Filename heuristic only",
                               "provenance_state": "UNKNOWN", "license_state": "UNKNOWN",
                               "commercial_permission": "UNKNOWN", "inference_state": "UNKNOWN",
                               "graph": graph_metadata(path)})
    return {"schema_version": 1, "timestamp": datetime.now(timezone.utc).isoformat(),
            "verification_contract": "VERIFIED_LOCAL verifies local bytes, not inference or licensing",
            "scope_evidence": scope_evidence, "models": models,
            "family_counts": dict(sorted(Counter(m["family_hint"] for m in models).items()))}


def verify_registry(registry: dict[str, Any], scopes: Mapping[str, Path],
                    unavailable: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Re-hash every registered model, read-only. A scope whose directory is absent (e.g. the sibling checkout) is reported under
    ``unavailable_scopes`` with its reason and its models are NOT counted as checked; it is not a crash and not a pass for them.
    ``unavailable`` supplies reasons for scopes that could not even be resolved to a path."""
    unavailable = dict(unavailable or {})
    models = registry.get("models")
    if not isinstance(models, list) or not models:
        raise AuditError("INVALID_REGISTRY", "A nonempty models array is required")
    checked = 0
    violations = []
    skipped: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    for model in models:
        locator = model["locator"]
        key = (locator["scope"], locator["path"])
        if key in seen:
            raise AuditError("INVALID_REGISTRY", f"Duplicate model: {key}")
        seen.add(key)
        scope_base = scopes.get(key[0])
        if scope_base is None or not scope_base.exists():
            unavailable.setdefault(key[0], "scope directory is absent" + (f": {scope_base}" if scope_base is not None else ""))
            skipped[key[0]] = skipped.get(key[0], 0) + 1
            continue
        try:
            if model["state"] != EvidenceState.VERIFIED_LOCAL.value:
                raise AuditError("INVALID_REGISTRY", f"Unverified model: {key}")
            path = safe_path(scopes[key[0]], key[1])
            size, digest = sha256_file(path)
            if size != model["size_bytes"] or digest != model["sha256"]:
                raise AuditError("MODEL_DRIFT", str(key))
            checked += 1
        except (OSError, AuditError, KeyError) as error:
            violations.append({"locator": locator, "detail": str(error)})
    return {"ok": not violations, "checked": checked, "expected": len(models), "violations": violations,
            "unavailable_scopes": [{"scope": scope, "reason": unavailable[scope], "models": count}
                                   for scope, count in sorted(skipped.items())]}


def default_scopes(repo: Path) -> tuple[dict[str, Path], dict[str, str]]:
    """``(scopes, unavailable)``: the model scope directories and, for any that cannot be resolved (the sibling legacy checkout
    is configured through ``model_sources.json`` / Pinokio's home and may not exist on this machine), the reason."""
    home = Path.home()
    scopes = {"user_cache": home / ".cache", "insightface_cache": home / ".insightface", "torch_cache": home / ".torch"}
    unavailable: dict[str, str] = {}
    try:
        scopes = {"repository": legacy_root(repo), **scopes}
    except (OSError, ValueError, KeyError) as error:
        unavailable["repository"] = f"legacy model source cannot be resolved ({type(error).__name__}: {error})"
    return scopes, unavailable


# --------------------------------------------------------------------------------------------------------------------
# Pinned open-licence artifacts: model files that are NOT discovered by scanning but are loaded by exact name, only when
# their SHA-256 matches. Each entry carries the licence and the attribution the licence requires. Only artifacts whose
# licence permits commercial use belong here: FLAME 2020 and every other academic-only MPG model (DECA, EMOCA, MICA, VOCA)
# are deliberately absent and are never loaded.
# --------------------------------------------------------------------------------------------------------------------
OPEN_ARTIFACTS: dict[str, dict[str, Any]] = {
    "flame2023_open": {
        "name": "FLAME 2023 Open",
        "filename": "flame2023_Open.pkl",
        "sha256": "e75a0990728ba038c7da2a420ae4396f7ddd7781366c13026066c5b24f127623",
        "size_bytes": 53023668,
        "license_id": "CC-BY-4.0",
        "license_url": "https://flame.is.tue.mpg.de/modellicense.html",
        "commercial_use": "allow",
        "attribution": ("FLAME 2023 Open, Max Planck Institute for Intelligent Systems, CC BY 4.0. "
                        "T. Li, T. Bolkart, M. J. Black, H. Li and J. Romero, Learning a model of facial shape and expression "
                        "from 4D scans, ACM Transactions on Graphics (Proc. SIGGRAPH Asia), 2017."),
        "source": "https://flame.is.tue.mpg.de/",
        "format": "pickle holding numpy arrays and a scipy.sparse matrix only; read with an allow-listed unpickler",
        "provenance": "Official FLAME2023_Open download; the file is user-supplied, git-ignored and never bundled",
        # Checkout-relative locations searched in order (the file is never fetched automatically).
        "search_paths": ("app/roop_3d/.runtime/models/flame2023_open/flame2023_Open.pkl", "flame/flame2023/flame2023_Open.pkl"),
    },
}

# Local, procedurally synthesized CC0 assets are registered separately from
# downloadable model weights. Their bytes are validated by their own embedded
# digest; no external model source or network request is involved.
LOCAL_CC0_ASSETS: dict[str, dict[str, Any]] = {
    "makehuman_cc0_anatomical_pack": {
        "name": "MakeHuman CC0 Anatomical Full Head Pack",
        "filename": "makehuman_cc0_anatomical_pack.npz",
        "relative_path": "roop_3d/.runtime/models/anatomical/makehuman_cc0_anatomical_pack.npz",
        "license_id": "CC0-1.0",
        "commercial_use": "allow",
        "source": "Pinned MakeHuman CC0 head data plus repository-local NumPy synthesis",
        "fallback": "Analytic CC0 head synthesis when pinned MakeHuman data is not installed",
    },
}


def verify_local_cc0_assets(repo: Path, *, build_assets: bool = False) -> dict[str, Any]:
    """Validate registered CC0 assets against their embedded content digest (``.npz`` vs its ``.npz.json`` sidecar).

    Strictly read-only unless ``build_assets`` is set (``--build-assets``): an absent asset is then a violation naming the flag,
    instead of being silently synthesized."""
    rows: list[dict[str, Any]] = []
    violations: list[dict[str, str]] = []
    for artifact_id, record in LOCAL_CC0_ASSETS.items():
        try:
            # The audit CLI can be run as a file with only audit/ on sys.path;
            # expose the application package for its local asset reader.
            repo_text = str(repo.absolute())
            if repo_text not in sys.path:
                sys.path.insert(0, repo_text)
            from roop_3d.reconstruction.anatomical_pack import AnatomicalFullHeadAsset, build_anatomical_pack

            path = safe_path(repo, record["relative_path"])
            if not path.exists():
                if not build_assets:
                    raise AuditError("CC0_ASSET_MISSING", f"{record['relative_path']} is not present; verification never builds it "
                                     "(run with --build-assets to synthesize it)")
                build_anatomical_pack(path)
            asset = AnatomicalFullHeadAsset.load(path, auto_build=False, verify=True)
            size, digest = sha256_file(path)
            rows.append({
                "artifact_id": artifact_id,
                "name": record["name"],
                "state": "VERIFIED_LOCAL",
                "path": record["relative_path"],
                "size_bytes": size,
                "sha256": digest,
                "license_id": asset.license_id,
                "commercial_use": record["commercial_use"],
                "template_source": asset.template_source,
                "evidence": "Local CC0 synthesis loaded with allow_pickle=False and matched its recorded content digest",
            })
        except Exception as error:
            violations.append({"artifact_id": artifact_id, "detail": str(error)})
    return {"ok": not violations, "assets": rows, "violations": violations}


def open_artifact_record(artifact_id: str) -> dict[str, Any]:
    try:
        return OPEN_ARTIFACTS[artifact_id]
    except KeyError:
        raise AuditError("OPEN_ARTIFACT_UNKNOWN", artifact_id) from None


def verify_open_artifact(artifact_id: str, path: Path) -> dict[str, Any]:
    """Hash ``path`` and require it to be exactly the pinned artifact. Raises ``MODEL_DRIFT`` on any difference."""
    record = open_artifact_record(artifact_id)
    size, digest = sha256_file(path)
    if digest != record["sha256"] or size != record["size_bytes"]:
        raise AuditError("MODEL_DRIFT", f"{artifact_id}: {path} is not the pinned file (sha256 {digest[:16]}..., {size} bytes)")
    return {"artifact_id": artifact_id, "state": EvidenceState.VERIFIED_LOCAL.value, "sha256": digest, "size_bytes": size,
            "license_id": record["license_id"], "commercial_use": record["commercial_use"], "attribution": record["attribution"],
            "evidence": "Local file bytes read and SHA-256 compared with the pinned value"}


def locate_open_artifact(artifact_id: str, repo: Path) -> Path | None:
    """First existing pinned location inside the checkout (``repo`` is the checkout root, i.e. the app's parent)."""
    for relative in open_artifact_record(artifact_id)["search_paths"]:
        candidate = safe_path(repo, relative)
        if candidate.is_file():
            return candidate
    return None


def verify_open_artifacts(repo: Path) -> dict[str, Any]:
    """Verify every pinned open artifact that is present. An absent optional artifact is reported, not a failure; a present
    file that does not match its pinned hash is a violation."""
    rows, violations = [], []
    for artifact_id in OPEN_ARTIFACTS:
        located = locate_open_artifact(artifact_id, repo)
        if located is None:
            rows.append({"artifact_id": artifact_id, "state": "NOT_PRESENT", "evidence": "Optional artifact is not installed"})
            continue
        try:
            rows.append(verify_open_artifact(artifact_id, located))
        except AuditError as error:
            violations.append({"artifact_id": artifact_id, "detail": str(error)})
    return {"ok": not violations, "artifacts": rows, "violations": violations}


def license_register(registry: dict[str, Any]) -> str:
    """Keep artifact integrity separate from permissions we have not verified."""
    output = io.StringIO(newline="")
    fields = ["artifact_name", "family_hint", "verification_state", "locator", "sha256",
              "graph_metadata_state", "inference_state", "license_state", "commercial_permission",
              "download_url_state", "evidence"]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for model in registry["models"]:
        locator = model["locator"]
        writer.writerow({"artifact_name": model["name"], "family_hint": model["family_hint"],
                         "verification_state": model["state"], "locator": f"{locator['scope']}:{locator['path']}",
                         "sha256": model["sha256"], "graph_metadata_state": model["graph"]["state"],
                         "inference_state": model["inference_state"], "license_state": model["license_state"],
                         "commercial_permission": model["commercial_permission"], "download_url_state": "UNKNOWN",
                         "evidence": model["evidence"]})
    for record in LOCAL_CC0_ASSETS.values():
        writer.writerow({"artifact_name": record["name"], "family_hint": "anatomical_asset",
                         "verification_state": "REGISTERED_LOCAL_SYNTHESIS",
                         "locator": f"application_runtime:{record['relative_path']}",
                         "graph_metadata_state": "NOT_APPLICABLE", "inference_state": "NOT_APPLICABLE",
                         "license_state": record["license_id"],
                         "commercial_permission": record["commercial_use"], "download_url_state": "NOT_APPLICABLE",
                         "evidence": f"{record['source']}; {record['fallback']}."})
    for name in ("FLAME", "Wav2Lip", "InstantID", "PuLID", "SDXL-Turbo", "FLUX.1-Schnell", "remote_provider"):
        writer.writerow({"artifact_name": name, "verification_state": "CANDIDATE", "graph_metadata_state": "UNKNOWN",
                         "inference_state": "UNKNOWN", "license_state": "UNKNOWN", "commercial_permission": "UNKNOWN",
                         "download_url_state": "UNKNOWN",
                         "evidence": "Optional capability; upstream artifacts, permission and execution not verified by this audit"})
    return output.getvalue()


# ====================================================================================================================
# Universal model download manager
#
# Every model and 3D asset roop-3d needs is described once in ``universal_model_manifest.json`` (generated by
# ``tools/build_universal_manifest.py``, committed, read offline). The manager downloads from zero-auth mirrors, resumes
# with HTTP Range into ``<file>.part``, verifies SHA-256 before the atomic rename into place, and on a bad hash drops the
# ``.part`` and tries the next mirror. Assets that can be built locally fall back to ``procedural_assets`` with no network.
#
# Licensing is NOT bypassed: an entry whose licence is not CC0/MIT/Apache/CC-BY (or is unverified) is only fetched when the
# caller passes ``accepted_license=True``, exactly like ``roop_3d.models.registry.authorize``.
# ====================================================================================================================
MANIFEST_PATH = Path(__file__).with_name("universal_model_manifest.json")
MANIFEST_SCHEMA = "roop3d.universal_model_manifest.v1"
OPEN_LICENSES = frozenset({"CC0-1.0", "MIT", "Apache-2.0", "CC-BY-4.0", "BSD-3-Clause"})
# Exact hosts a manifest URL may start on. Redirects may additionally land on these storage domains (bytes are verified by
# SHA-256 regardless; the hosts only keep a compromised redirect from pointing somewhere arbitrary).
DOWNLOAD_HOSTS = frozenset({"huggingface.co", "github.com", "raw.githubusercontent.com", "cdn.jsdelivr.net",
                            "dl.polyhaven.org", "drive.usercontent.google.com"})
REDIRECT_SUFFIXES = (".hf.co", ".huggingface.co", ".githubusercontent.com", ".googleusercontent.com")
MAX_REDIRECTS = 5
STREAM_CHUNK = 1 << 20
SET_WORKERS = 4            # members of an asset_set fetched at once (the 130 small MakeHuman files are latency bound)
DISK_MARGIN_BYTES = 256 << 20
USER_AGENT = "roop-3d-model-manager/1"
ENTRY_KINDS = frozenset({"weight", "asset_set", "compiled", "bundled"})

# Lifecycle states of an installed entry (``state``) and of a transfer (``status``).
READY, MISSING, PARTIAL, CORRUPT = "READY", "MISSING", "PARTIAL", "CORRUPT"
NOT_APPLICABLE, PROCEDURAL_FALLBACK = "NOT_APPLICABLE", "PROCEDURAL_FALLBACK"
INSTALLED_STATES = frozenset({READY, PROCEDURAL_FALLBACK})


class _Transient(Exception):
    """Network trouble that resuming or retrying the same mirror may fix."""


class _Unusable(Exception):
    """This mirror cannot serve the file (404, auth wall, bad range, oversize); try the next one."""


class _Cancelled(Exception):
    pass


class _AnyEvent:
    """Reads as set when any of several events is (a batch cancel, or a sibling download that already failed)."""

    def __init__(self, *events: threading.Event):
        self._events = events

    def is_set(self) -> bool:
        return any(event.is_set() for event in self._events)


def _url_allowed(url: str, *, redirect: bool) -> bool:
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        decoded = unquote(unquote(parts.path))
        if (parts.scheme != "https" or parts.port not in (None, 443) or parts.username or parts.password or parts.fragment
                or any(c in url for c in "\r\n\t\\") or any(p in {".", ".."} for p in decoded.split("/"))):
            return False
        return host in DOWNLOAD_HOSTS or (redirect and any(host.endswith(s) for s in REDIRECT_SUFFIXES))
    except ValueError:
        return False


def _sha256_stream(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * STREAM_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class ManifestEntry:
    id: str
    name: str
    kind: str
    category: str
    install_root: str
    install_path: str
    license: str
    commercial_use: str
    requires_acceptance: bool
    mirrors: tuple[str, ...] = ()
    sha256: str | None = None
    size_bytes: int | None = None
    files: tuple[Mapping[str, Any], ...] = ()
    license_url: str | None = None
    license_note: str = ""
    required_for_stages: tuple[str, ...] = ()
    optional: bool = False
    acquisition: str = ""
    depends_on: tuple[str, ...] = ()
    after: tuple[str, ...] = ()
    build_inputs: Mapping[str, str] = field(default_factory=dict)
    procedural_fallback: str | None = None
    content_digest: str | None = None
    attribution: str | None = None
    format: str | None = None
    filename: str | None = None
    note: str = ""
    registry_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ManifestEntry":
        def fail(detail: str) -> AuditError:
            return AuditError("MANIFEST_INVALID", f"{raw.get('id', '?')}: {detail}")

        entry_id = raw.get("id", "")
        if not isinstance(entry_id, str) or not entry_id or not all(c.islower() or c.isdigit() or c == "_" for c in entry_id):
            raise fail("id must be lowercase letters, digits and underscores")
        if raw.get("kind") not in ENTRY_KINDS:
            raise fail(f"kind must be one of {sorted(ENTRY_KINDS)}")
        if raw.get("install_root") not in {"app", "repo"}:
            raise fail("install_root must be 'app' or 'repo'")
        install_path = relative_path(str(raw.get("install_path", "")))    # rejects absolute paths and '..'
        license_id = raw.get("license")
        if not license_id or raw.get("commercial_use") not in {"allow", "deny", "unknown"}:
            raise fail("license and commercial_use are required")
        accept = bool(raw.get("requires_acceptance"))
        if license_id not in OPEN_LICENSES and not accept:
            raise fail(f"licence {license_id!r} is not an open licence, so requires_acceptance must be true")
        mirrors = tuple(raw.get("mirrors", ()))
        for url in mirrors:
            if not _url_allowed(url, redirect=False):
                raise fail(f"mirror {url!r} is not https on an approved host")
        kind = raw["kind"]
        files = tuple(raw.get("files", ()))
        if kind == "weight":
            if not (_hex64(raw.get("sha256")) and isinstance(raw.get("size_bytes"), int) and raw["size_bytes"] > 0):
                raise fail("a weight needs a 64-hex sha256 and a positive size_bytes")
        if kind == "asset_set":
            if not files or any(not (_hex64(f.get("sha256")) and f.get("size_bytes", 0) > 0) for f in files):
                raise fail("an asset_set needs files with sha256 and size_bytes")
            for f in files:
                relative_path(f["path"])
            if any(not url.endswith("/") for url in mirrors):
                raise fail("asset_set mirrors are base URLs ending in '/'")
        if kind in {"bundled", "compiled"} and raw.get("content_digest") is not None and not _hex64(raw["content_digest"]):
            raise fail("content_digest must be 64 hex characters")
        if kind == "bundled" and not raw.get("content_digest"):
            raise fail("a bundled asset needs its pinned content_digest")
        if kind in {"compiled", "bundled"} and not raw.get("procedural_fallback"):
            raise fail("compiled and bundled assets name their procedural_fallback handler")
        return cls(
            id=entry_id, name=str(raw.get("name", entry_id)), kind=kind, category=str(raw.get("category", "other")),
            install_root=raw["install_root"], install_path=install_path.as_posix(), license=license_id,
            commercial_use=raw["commercial_use"], requires_acceptance=accept, mirrors=mirrors, sha256=raw.get("sha256"),
            size_bytes=raw.get("size_bytes"), files=files, license_url=raw.get("license_url"),
            license_note=str(raw.get("license_note", "")), required_for_stages=tuple(raw.get("required_for_stages", ())),
            optional=bool(raw.get("optional")), acquisition=str(raw.get("acquisition", "")),
            depends_on=tuple(raw.get("depends_on", ())), after=tuple(raw.get("after", ())),
            build_inputs=dict(raw.get("build_inputs", {})), procedural_fallback=raw.get("procedural_fallback"),
            content_digest=raw.get("content_digest"), attribution=raw.get("attribution"), format=raw.get("format"),
            filename=raw.get("filename"), note=str(raw.get("note", "")), registry_ids=tuple(raw.get("registry_ids", ())))

    @property
    def total_bytes(self) -> int:
        if self.kind == "asset_set":
            return sum(int(f["size_bytes"]) for f in self.files)
        return int(self.size_bytes or 0)

    @property
    def license_badge(self) -> str:
        return "open" if self.license in OPEN_LICENSES else "unverified" if self.license == "UNVERIFIED" else "restricted"


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, ManifestEntry]:
    document = json.loads(Path(path).read_text("utf-8"))
    if document.get("schema") != MANIFEST_SCHEMA:
        raise AuditError("MANIFEST_INVALID", f"unsupported manifest schema {document.get('schema')!r}")
    entries: dict[str, ManifestEntry] = {}
    locations: set[tuple[str, str]] = set()
    for raw in document["models"]:
        entry = ManifestEntry.from_dict(raw)
        if entry.id in entries:
            raise AuditError("MANIFEST_INVALID", f"duplicate id {entry.id}")
        location = (entry.install_root, entry.install_path.casefold())
        if location in locations:
            raise AuditError("MANIFEST_INVALID", f"{entry.id}: two entries install to {entry.install_path}")
        locations.add(location)
        entries[entry.id] = entry
    for entry in entries.values():
        for other in (*entry.depends_on, *entry.after, *entry.build_inputs.values()):
            if other not in entries:
                raise AuditError("MANIFEST_INVALID", f"{entry.id}: unknown reference {other}")
    return entries


class _HashCache:
    """SHA-256 of local files keyed by (size, mtime): a multi-GB model is hashed once, then only when it changes."""

    def __init__(self, path: Path | None = None):
        self._path, self._lock, self._rows = path, threading.Lock(), {}
        if path is not None and path.is_file():
            try:
                self._rows = {k: tuple(v) for k, v in json.loads(path.read_text("utf-8")).items()}
            except (OSError, ValueError, TypeError):
                self._rows = {}

    def digest(self, path: Path, *, force: bool = False) -> str:
        info = path.stat()
        key = str(path)
        with self._lock:
            row = self._rows.get(key)
        if not force and row and row[0] == info.st_size and row[1] == info.st_mtime_ns:
            return row[2]
        value = _sha256_stream(path)
        with self._lock:
            self._rows[key] = (info.st_size, info.st_mtime_ns, value)
        return value

    def forget(self, path: Path) -> None:
        with self._lock:
            self._rows.pop(str(path), None)

    def save(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._path.with_name(self._path.name + ".tmp")
            with self._lock:
                temporary.write_text(json.dumps({k: list(v) for k, v in self._rows.items()}), "utf-8")
            os.replace(temporary, self._path)
        except OSError:
            pass


class _Progress:
    """Per-entry transfer meter. Emits ``model_download_progress`` events (rate limited, always on a status change)."""

    def __init__(self, entry: ManifestEntry, sink: Callable[[dict], None] | None, clock: Callable[[], float],
                 batch_id: str | None, interval: float = 0.25):
        self.entry_id, self.total, self.sink, self.clock, self.batch_id, self.interval = entry.id, entry.total_bytes, sink, clock, batch_id, interval
        self.downloaded, self.status, self.mirror, self.detail = 0, "queued", None, None
        self._window: list[tuple[float, int]] = []
        self._last_emit = -1e9
        self._lock = threading.RLock()
        self.files: dict[str, int] = {}          # bytes held per member file; the entry total is their sum

    def track(self, key: str, received: int, **fields: Any) -> None:
        """Record how many bytes of member ``key`` are on disk; files of a set may be fetched concurrently."""
        with self._lock:
            self.files[key] = received
            self.update(sum(self.files.values()), **fields)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = self.clock()
            self._window = [(t, b) for t, b in self._window if now - t <= 5.0]
            speed = 0.0
            if len(self._window) >= 2 and self._window[-1][0] - self._window[0][0] >= 0.25:
                speed = max(0.0, (self._window[-1][1] - self._window[0][1]) / (self._window[-1][0] - self._window[0][0]))
            remaining = max(0, self.total - self.downloaded)
            eta = round(remaining / speed, 1) if speed > 0 and self.status == "downloading" else None
            return {"type": "model_download_progress", "model_id": self.entry_id, "downloaded_bytes": self.downloaded,
                    "bytes_downloaded": self.downloaded, "total_bytes": self.total,
                    "percent": 100.0 if self.status == "ready" else round(100.0 * self.downloaded / self.total, 1) if self.total else 0.0,
                    "speed_mbps": round(speed * 8 / 1e6, 2), "eta_seconds": eta, "mirror_used": self.mirror,
                    "status": self.status, "batch_id": self.batch_id, "detail": self.detail}

    def update(self, downloaded: int | None = None, *, status: str | None = None, mirror: str | None = None,
               detail: str | None = None) -> None:
        with self._lock:
            changed = status is not None and status != self.status
            if downloaded is not None:
                self.downloaded = min(downloaded, self.total) if self.total else downloaded
                self._window.append((self.clock(), self.downloaded))
            if status is not None:
                self.status = status
            if mirror is not None:
                self.mirror = mirror
            if detail is not None or changed:
                self.detail = detail
            now = self.clock()
            if not (changed or now - self._last_emit >= self.interval):
                return
            self._last_emit = now
        self.emit()

    def emit(self) -> None:
        if self.sink is None:
            return
        try:
            self.sink(self.snapshot())
        except Exception:           # a broken subscriber must never fail a download
            pass


@dataclass
class _Batch:
    batch_id: str
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    skipped: list[dict[str, str]] = field(default_factory=list)
    futures: dict[str, Future] = field(default_factory=dict)
    cancel: threading.Event = field(default_factory=threading.Event)
    started_at: float = 0.0
    finished_at: float | None = None

    def public(self) -> dict[str, Any]:
        statuses = {row["status"] for row in self.entries.values()}
        running = bool(statuses & {"queued", "downloading", "verifying", "compiling"})
        failed = bool(statuses & {"failed"})
        state = "running" if running else "cancelled" if self.cancel.is_set() else "failed" if failed else "completed"
        return {"batch_id": self.batch_id, "state": state, "entries": {k: dict(v) for k, v in self.entries.items()},
                "skipped": list(self.skipped)}


class UniversalModelDownloadManager:
    """Resumable, mirror-failover, checksum-verified downloads for the whole manifest. See the block comment above."""

    def __init__(self, app_root: Path, *, manifest: Mapping[str, ManifestEntry] | None = None, manifest_path: Path = MANIFEST_PATH,
                 transport: Any = None, sink: Callable[[dict], None] | None = None, max_workers: int = 3,
                 handlers: Mapping[str, Any] | None = None, on_installed: Callable[[], None] | None = None,
                 hash_cache_path: Path | None = None, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep, mirror_retries: int = 1):
        self.app_root = Path(app_root)
        self.repo_root = self.app_root.parent if self.app_root.name == "app" else self.app_root
        self.entries: dict[str, ManifestEntry] = dict(manifest) if manifest is not None else load_manifest(manifest_path)
        self.transport, self.sink, self.on_installed = transport, sink, on_installed
        self.clock, self._sleep, self.mirror_retries = clock, sleep, mirror_retries
        self.cache = _HashCache(hash_cache_path)
        if handlers is None:
            if __package__:
                from roop_3d.audit.procedural_assets import HANDLERS as handlers
            else:
                from procedural_assets import HANDLERS as handlers      # script / bare import: audit/ is on sys.path
        self.handlers = dict(handlers)
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="roop3d-universal-download")
        self._lock = threading.RLock()
        self._batches: dict[str, _Batch] = {}
        self._progress: dict[str, _Progress] = {}
        self._verified: dict[str, tuple[tuple, dict]] = {}      # compiled/bundled verification results, keyed by file stamp
        self._active: dict[str, str] = {}            # entry id -> batch id, so a second request joins instead of racing
        self._closed = False

    # ------------------------------------------------------------------------------------------------ paths
    def _root(self, entry: ManifestEntry) -> Path:
        return self.app_root if entry.install_root == "app" else self.repo_root

    def target(self, entry: ManifestEntry) -> Path:
        return safe_path(self._root(entry), entry.install_path)

    def quarantine(self, path: Path) -> Path:
        """Move a file that failed verification aside (never delete a user's file); returns the new location."""
        folder = path.parent / ".quarantine"
        folder.mkdir(parents=True, exist_ok=True)
        destination = folder / f"{path.name}.{time.strftime('%Y%m%dT%H%M%S')}.{uuid.uuid4().hex[:6]}.mismatch"
        os.replace(path, destination)
        self.cache.forget(path)
        return destination

    # ------------------------------------------------------------------------------------------------ inspection
    def _file_state(self, path: Path, size: int, sha256: str, *, force: bool) -> str:
        if not path.is_file():
            return MISSING
        try:
            if path.stat().st_size != size:
                return CORRUPT
            return READY if self.cache.digest(path, force=force) == sha256 else CORRUPT
        except OSError:
            return CORRUPT

    def _find_alias(self, entry: ManifestEntry, *, force: bool) -> Path | None:
        """The same bytes under another name in ``app/models`` (e.g. a model the user dropped in by hand)."""
        if entry.kind != "weight" or entry.install_root != "app" or not entry.install_path.startswith("models/"):
            return None
        folder = self.app_root / "models"
        try:
            candidates = [p for p in folder.iterdir() if p.is_file() and not p.is_symlink() and not p.name.endswith(".part")
                          and p.stat().st_size == entry.size_bytes]       # an unfinished download is never "installed"
        except OSError:
            return None
        for candidate in sorted(candidates):
            try:
                if self.cache.digest(candidate, force=force) == entry.sha256:
                    return candidate
            except OSError:
                continue
        return None

    def inspect(self, entry: ManifestEntry | str, *, force: bool = False, _seen: frozenset[str] = frozenset()) -> dict[str, Any]:
        entry = self.entries[entry] if isinstance(entry, str) else entry
        target = self.target(entry)
        state, detail, location, kind_hint = MISSING, "", target, "pinned"
        installed_bytes = partial_bytes = 0
        files_ready = files_total = None
        if entry.kind == "weight":
            state = self._file_state(target, int(entry.size_bytes), entry.sha256, force=force)
            if state == MISSING:
                alias = self._find_alias(entry, force=force)
                if alias is not None:
                    state, location, detail = READY, alias, f"installed as {alias.name}"
                else:
                    part = target.with_name(target.name + ".part")
                    if part.is_file() and 0 < part.stat().st_size < int(entry.size_bytes):
                        state, partial_bytes = PARTIAL, part.stat().st_size
            elif state == CORRUPT:
                detail = "size or SHA-256 differs from the pinned value"
            installed_bytes = int(entry.size_bytes) if state == READY else 0
        elif entry.kind == "asset_set":
            files_total = len(entry.files)
            states = [self._file_state(target / f["path"], int(f["size_bytes"]), f["sha256"], force=force) for f in entry.files]
            files_ready = sum(s == READY for s in states)
            installed_bytes = sum(int(f["size_bytes"]) for f, s in zip(entry.files, states) if s == READY)
            state = (READY if files_ready == files_total else CORRUPT if CORRUPT in states
                     else PARTIAL if files_ready else MISSING)
            if state == CORRUPT:
                detail = f"{states.count(CORRUPT)} file(s) differ from their pinned SHA-256"
        else:                                                   # compiled / bundled: verified by its handler
            handler = self.handlers.get(entry.procedural_fallback)
            unmet = [d for d in entry.depends_on if self.inspect(d, _seen=_seen | {entry.id})["state"] not in INSTALLED_STATES] if entry.id not in _seen else []
            if not target.is_file():
                state, detail = (NOT_APPLICABLE, f"needs {', '.join(unmet)} first") if unmet else (MISSING, "")
            elif handler is None:
                state, detail = CORRUPT, f"no handler {entry.procedural_fallback!r}"
            else:
                info = target.stat()
                stamp = (info.st_size, info.st_mtime_ns, entry.content_digest)
                cached = self._verified.get(entry.id)
                if cached is not None and cached[0] == stamp and not force:      # a polling UI must not reload assets every call
                    result = cached[1]
                else:
                    try:
                        result = handler.verify(target, entry.content_digest)
                    except Exception as error:       # noqa: BLE001 - a damaged asset is a CORRUPT row, never a failed request
                        result = {"ok": False, "detail": f"{type(error).__name__}: {error}"}
                    self._verified[entry.id] = (stamp, result)
                if result["ok"]:
                    kind_hint = result.get("kind", "pinned")
                    state = PROCEDURAL_FALLBACK if kind_hint == "procedural" else READY
                    detail = result.get("detail", "")
                    installed_bytes = target.stat().st_size
                else:
                    state, detail = CORRUPT, result.get("detail", "verification failed")
        progress = self._progress.get(entry.id)
        live = progress.snapshot() if progress and progress.status in {"queued", "downloading", "verifying", "compiling"} else None
        buildable = entry.procedural_fallback in self.handlers if entry.kind in {"compiled", "bundled"} else bool(entry.procedural_fallback)
        hash_status = {READY: "verified", PROCEDURAL_FALLBACK: "verified", CORRUPT: "mismatch"}.get(state, "not_installed")
        return {
            "id": entry.id, "name": entry.name, "kind": entry.kind, "category": entry.category, "filename": entry.filename,
            "install_path": entry.install_path, "state": state, "installed": state in INSTALLED_STATES,
            "verified": state in INSTALLED_STATES, "hash_status": hash_status, "detail": detail,
            "location": (location.relative_to(self._root(entry)).as_posix() if location.is_relative_to(self._root(entry)) else location.name),
            "installed_bytes": installed_bytes, "partial_bytes": partial_bytes, "size_bytes": entry.total_bytes, "sha256": entry.sha256,
            "files_ready": files_ready, "files_total": files_total, "mirrors": list(entry.mirrors), "mirror_count": len(entry.mirrors),
            "acquisition": entry.acquisition, "license": entry.license, "license_badge": entry.license_badge,
            "license_url": entry.license_url, "license_note": entry.license_note, "commercial_use": entry.commercial_use,
            "requires_acceptance": entry.requires_acceptance, "attribution": entry.attribution,
            "required_for_stages": list(entry.required_for_stages), "optional": entry.optional, "format": entry.format,
            "registry_ids": list(entry.registry_ids), "procedural_fallback_available": bool(buildable), "note": entry.note,
            "download": live,
        }

    def manifest(self, *, force: bool = False) -> dict[str, Any]:
        """Everything the UI needs: each model with licence badge, install + hash state, live transfer, and disk usage."""
        rows = [self.inspect(entry, force=force) for entry in self.entries.values()]
        self.cache.save()
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["state"]] = counts.get(row["state"], 0) + 1
        needed = sum(r["size_bytes"] - r["installed_bytes"] for r in rows if r["state"] in {MISSING, PARTIAL, CORRUPT} and r["mirror_count"])
        try:
            models_dir = self.app_root / "models"
            free = shutil.disk_usage(models_dir if models_dir.exists() else self.app_root).free
        except OSError:
            free = None
        return {"schema": MANIFEST_SCHEMA, "models": rows,
                "summary": {"total": len(rows), "by_state": counts, "licenses": {b: sum(r["license_badge"] == b for r in rows) for b in ("open", "restricted", "unverified")},
                            "required_missing": [r["id"] for r in rows if not r["optional"] and r["state"] not in INSTALLED_STATES]},
                "disk": {"installed_bytes": sum(r["installed_bytes"] for r in rows), "download_needed_bytes": needed,
                         "free_bytes": free, "enough_space": None if free is None else free >= needed + DISK_MARGIN_BYTES}}

    # ------------------------------------------------------------------------------------------------ downloading
    def _client(self):
        import httpx
        try:
            from roop_3d.models.downloader import tls_context
            verify: Any = tls_context()
        except ImportError:
            import ssl
            verify = ssl.create_default_context()
        return httpx.Client(transport=self.transport, timeout=httpx.Timeout(30.0, read=60.0), follow_redirects=False,
                            trust_env=True, verify=verify)

    def _download_file(self, client: Any, mirrors: Sequence[str], destination: Path, sha256: str, size: int,
                       progress: _Progress, key: str, cancel: threading.Event, announce: bool = True) -> str:
        """Fetch one file into ``destination`` via ``.part``; returns the mirror that supplied it.

        ``announce=False`` keeps the entry's status steady while verifying one member of a larger set."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        part = destination.with_name(destination.name + ".part")
        meta = destination.with_name(destination.name + ".part.json")

        def finish(source: str) -> str | None:
            progress.track(key, size, status="verifying" if announce else None, mirror=source)
            if _sha256_stream(part) != sha256:
                part.unlink(missing_ok=True)
                meta.unlink(missing_ok=True)
                return None
            with part.open("r+b") as stream:
                os.fsync(stream.fileno())
            os.replace(part, destination)            # the only moment the final name appears: complete and verified
            meta.unlink(missing_ok=True)
            self.cache.forget(destination)
            return source

        if part.is_file() and part.stat().st_size == size:       # a complete .part left by a crash before the rename
            if finish("local .part") is not None:
                return "local .part"
        errors: list[str] = []
        for mirror in mirrors:
            attempts = 1 + self.mirror_retries
            while attempts > 0:
                attempts -= 1
                if cancel.is_set():
                    raise _Cancelled()
                try:
                    offset = self._stream_to_part(client, mirror, part, meta, size, sha256, progress, key, cancel)
                except _Transient as error:
                    errors.append(f"{mirror}: {error}")
                    self._sleep(min(2.0, 0.25 * (2 ** (self.mirror_retries - attempts))))
                    continue
                except _Unusable as error:
                    errors.append(f"{mirror}: {error}")
                    break
                if finish(mirror) is not None:
                    return mirror
                errors.append(f"{mirror}: SHA256_MISMATCH")
                if offset > 0:           # bytes resumed from an earlier (possibly different) source may be the bad ones
                    attempts = max(attempts, 1)
                    continue
                break                    # a clean transfer from this mirror was wrong: the mirror serves different bytes
        raise AuditError("ALL_MIRRORS_FAILED", "; ".join(errors) or "no mirror is available")

    def _stream_to_part(self, client: Any, mirror: str, part: Path, meta: Path, size: int, sha256: str,
                        progress: _Progress, key: str, cancel: threading.Event) -> int:
        import httpx
        offset = part.stat().st_size if part.is_file() else 0
        if offset >= size or offset < 0:
            part.unlink(missing_ok=True)
            offset = 0
        validator = None
        if offset and meta.is_file():
            try:
                saved = json.loads(meta.read_text("utf-8"))
                if saved.get("mirror") == mirror and saved.get("size") == size:
                    validator = saved.get("validator")
            except (OSError, ValueError):
                validator = None
        headers = {"Accept-Encoding": "identity", "User-Agent": USER_AGENT}
        if offset:
            headers["Range"] = f"bytes={offset}-"
            if validator:
                headers["If-Range"] = validator
        url = mirror
        progress.track(key, offset, status="downloading", mirror=mirror)
        try:
            for hop in range(MAX_REDIRECTS + 1):
                if not _url_allowed(url, redirect=hop > 0):
                    raise _Unusable(f"URL_NOT_ALLOWED {urlsplit(url).hostname}")
                with client.stream("GET", url, headers=headers) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise _Unusable("REDIRECT_WITHOUT_LOCATION")
                        url = urljoin(url, location)
                        continue
                    return self._consume(response, mirror, part, meta, size, sha256, offset, progress, key, cancel)
            raise _Unusable("TOO_MANY_REDIRECTS")
        except httpx.TransportError as error:                     # connect/read timeouts, resets, TLS, truncated body
            raise _Transient(f"NETWORK {type(error).__name__}") from None

    def _consume(self, response: Any, mirror: str, part: Path, meta: Path, size: int, sha256: str, offset: int,
                 progress: _Progress, key: str, cancel: threading.Event) -> int:
        code = response.status_code
        if code == 416:
            part.unlink(missing_ok=True)                           # our partial is not usable against this file: start over
            meta.unlink(missing_ok=True)
            raise _Transient("RANGE_NOT_SATISFIABLE (restarting from 0)")
        if code in {401, 403}:
            raise _Unusable(f"HTTP_{code} (login or access wall)")
        if code in {404, 410, 451}:
            raise _Unusable(f"HTTP_{code}")
        if code == 429 or code >= 500:
            raise _Transient(f"HTTP_{code}")
        if code not in {200, 206}:
            raise _Unusable(f"HTTP_{code}")
        if response.headers.get("content-encoding", "identity").lower() != "identity":
            raise _Unusable("COMPRESSED_RESPONSE")
        if code == 206:
            if response.headers.get("content-range") != f"bytes {offset}-{size - 1}/{size}":
                raise _Unusable("CONTENT_RANGE_MISMATCH")
            mode = "ab" if offset else "wb"
        else:                                                      # 200: the server ignored Range, so the whole file follows
            offset, mode = 0, "wb"
            progress.track(key, 0)
        length = response.headers.get("content-length")
        if length is not None and (not length.isdigit() or int(length) != size - offset):
            raise _Unusable("CONTENT_LENGTH_MISMATCH")
        etag, modified = response.headers.get("etag", ""), response.headers.get("last-modified", "")
        validator = etag if etag and not etag.startswith("W/") else modified or None
        meta.write_text(json.dumps({"mirror": mirror, "size": size, "sha256": sha256, "validator": validator}), "utf-8")
        total = offset
        oversize = False
        with part.open(mode) as stream:
            if mode == "ab" and stream.tell() != offset:
                raise _Transient("PART_CHANGED")
            try:
                # iter_raw yields bytes as they arrive; iter_bytes(n) would buffer up to n and lose them if the link drops.
                for block in response.iter_raw():
                    if cancel.is_set():
                        raise _Cancelled()
                    if total + len(block) > size:
                        oversize = True
                        break
                    stream.write(block)
                    total += len(block)
                    progress.track(key, total)
            finally:
                stream.flush()
                os.fsync(stream.fileno())
        if oversize:                       # a source that overruns the pinned size is not trusted for any of its bytes
            part.unlink(missing_ok=True)
            meta.unlink(missing_ok=True)
            raise _Unusable("RESPONSE_LARGER_THAN_PINNED_SIZE")
        if total != size:
            raise _Transient("TRUNCATED_RESPONSE")                 # keep the .part: the next attempt resumes from it
        return offset

    def _install_weight(self, entry: ManifestEntry, progress: _Progress, cancel: threading.Event) -> None:
        target = self.target(entry)
        if self.inspect(entry)["state"] == READY:
            return
        self._check_space(entry.total_bytes - (target.with_name(target.name + ".part").stat().st_size
                                              if target.with_name(target.name + ".part").is_file() else 0))
        if target.is_file():                                       # same name, wrong bytes: set it aside, never overwrite blindly
            self.quarantine(target)
        with self._client() as client:
            progress.mirror = self._download_file(client, entry.mirrors, target, entry.sha256, int(entry.size_bytes), progress, "", cancel)

    def _install_asset_set(self, entry: ManifestEntry, progress: _Progress, cancel: threading.Event) -> None:
        """Fetch the missing members of a set a few at a time; the entry reports one steady status and a summed byte count."""
        root = self.target(entry)
        pending = [f for f in entry.files if self._file_state(root / f["path"], int(f["size_bytes"]), f["sha256"], force=False) != READY]
        self._check_space(sum(int(f["size_bytes"]) for f in pending))
        pending_paths = {f["path"] for f in pending}
        for item in entry.files:
            if item["path"] not in pending_paths:
                progress.files[item["path"]] = int(item["size_bytes"])
        progress.update(sum(progress.files.values()), status="downloading")
        failed = threading.Event()
        stop = _AnyEvent(cancel, failed)          # one member failing stops its siblings

        def fetch(client: Any, item: Mapping[str, Any]) -> str:
            destination = root / item["path"]
            if destination.is_file():
                self.quarantine(destination)
            return self._download_file(client, [m + item["path"] for m in entry.mirrors], destination, item["sha256"],
                                       int(item["size_bytes"]), progress, item["path"], stop, announce=False)

        errors: list[BaseException] = []
        with self._client() as client, ThreadPoolExecutor(max_workers=SET_WORKERS, thread_name_prefix="roop3d-set-member") as pool:
            for future in [pool.submit(fetch, client, item) for item in pending]:
                try:
                    progress.mirror = future.result()
                except BaseException as error:         # noqa: BLE001 - collected, the first real one is re-raised below
                    failed.set()
                    errors.append(error)
        if cancel.is_set():
            raise _Cancelled()
        real = [e for e in errors if not isinstance(e, _Cancelled)]
        if real:
            raise real[0]

    def _check_space(self, needed: int) -> None:
        try:
            free = shutil.disk_usage(self.app_root).free
        except OSError:
            return
        if free < needed + DISK_MARGIN_BYTES:
            raise AuditError("INSUFFICIENT_DISK_SPACE", f"{needed} bytes needed plus a {DISK_MARGIN_BYTES >> 20} MiB margin, {free} free")

    def _compile(self, entry: ManifestEntry, progress: _Progress) -> None:
        handler = self.handlers.get(entry.procedural_fallback)
        if handler is None:
            raise AuditError("PROCEDURAL_HANDLER_MISSING", str(entry.procedural_fallback))
        progress.update(status="compiling")
        target = self.target(entry)
        if target.is_file():
            self.quarantine(target)
        inputs: dict[str, Path] = {}
        for argument, source_id in entry.build_inputs.items():
            source = self.entries[source_id]
            if self.inspect(source)["state"] in INSTALLED_STATES:
                inputs[argument] = self.target(source)
        built = handler.build(target, **inputs)
        result = handler.verify(Path(built), entry.content_digest)
        if not result["ok"] and result.get("kind") != "procedural":
            Path(built).unlink(missing_ok=True)
            raise AuditError("PROCEDURAL_VERIFY_FAILED", result["detail"])

    # ------------------------------------------------------------------------------------------------ batches
    def _order(self, ids: Sequence[str]) -> list[str]:
        ordered: list[str] = []

        def visit(entry_id: str) -> None:
            if entry_id in ordered or entry_id not in ids:
                return
            entry = self.entries[entry_id]
            for before in (*entry.after, *entry.depends_on, *entry.build_inputs.values()):
                visit(before)
            ordered.append(entry_id)

        for entry_id in ids:
            visit(entry_id)
        return ordered

    def start_batch(self, ids: Sequence[str] | None = None, *, accepted_license: bool = False, commercial: bool = False,
                    force: bool = False) -> dict[str, Any]:
        """Queue every missing/partial/corrupt entry (or only ``ids``). Returns at once; progress arrives as events."""
        unknown = [i for i in (ids or ()) if i not in self.entries]
        if unknown:
            raise AuditError("MODEL_UNKNOWN", f"not in the manifest: {', '.join(unknown)}")
        selected = list(ids) if ids else list(self.entries)
        # Hashing a multi-GB model can take seconds: do it before taking the lock that live progress readers need.
        rows = {entry_id: self.inspect(entry_id) for entry_id in self._order(selected)}
        with self._lock:
            if self._closed:
                raise AuditError("DOWNLOADS_CLOSED", "the model download manager has been shut down")
            batch = _Batch(uuid.uuid4().hex[:12], started_at=self.clock())
            installed: list[str] = []
            plan: list[ManifestEntry] = []
            for entry_id, row in rows.items():
                entry, state = self.entries[entry_id], row["state"]

                def skip(reason: str, detail: str) -> None:
                    batch.skipped.append({"id": entry_id, "reason": reason, "detail": detail})

                if state in INSTALLED_STATES and not force:
                    installed.append(entry_id)
                elif entry_id in self._active:
                    skip("ALREADY_DOWNLOADING", f"batch {self._active[entry_id]} is already fetching this")
                elif entry.requires_acceptance and not accepted_license:
                    skip("LICENSE_ACCEPTANCE_REQUIRED", f"{entry.license}: {entry.license_note or 'review the licence, then pass accepted_license=true'}")
                elif commercial and entry.commercial_use != "allow":
                    skip("COMMERCIAL_USE_NOT_PERMITTED", f"commercial use is {entry.commercial_use} for {entry.license}")
                elif state == NOT_APPLICABLE:
                    skip("DEPENDENCY_MISSING", row["detail"])
                elif entry.kind in {"weight", "asset_set"} and not entry.mirrors:
                    skip("NO_PUBLIC_SOURCE", entry.note or entry.acquisition)
                else:
                    plan.append(entry)
            for entry in plan:
                progress = _Progress(entry, self.sink, self.clock, batch.batch_id)
                self._progress[entry.id] = progress
                self._active[entry.id] = batch.batch_id
                batch.entries[entry.id] = {"status": "queued", "error": None, "mirror_used": None}
                progress.update(0, status="queued")
            for entry in plan:
                batch.futures[entry.id] = self._executor.submit(self._run_entry, batch, entry)
            self._batches[batch.batch_id] = batch
            if not plan:
                batch.finished_at = self.clock()
            return {**batch.public(), "already_installed": installed, "queued": [e.id for e in plan]}

    def _run_entry(self, batch: _Batch, entry: ManifestEntry) -> None:
        progress = self._progress[entry.id]
        row = batch.entries[entry.id]
        status, error = "ready", None
        try:
            for before in (*entry.after, *entry.depends_on, *entry.build_inputs.values()):    # finish prerequisites queued in this batch first
                future = batch.futures.get(before)
                if future is not None:
                    future.result()
            if batch.cancel.is_set():
                raise _Cancelled()
            if entry.kind == "weight":
                self._install_weight(entry, progress, batch.cancel)
            elif entry.kind == "asset_set":
                self._install_asset_set(entry, progress, batch.cancel)
            else:
                self._compile(entry, progress)
            state = self.inspect(entry, force=True)["state"]
            if state not in INSTALLED_STATES:
                raise AuditError("INSTALL_NOT_VERIFIED", f"{entry.id} is {state} after install")
        except _Cancelled:
            status = "cancelled"
        except AuditError as exc:
            status, error = "failed", exc.code
            progress.detail = exc.detail
        except Exception as exc:         # noqa: BLE001 - one entry's failure must not take down the batch
            status, error = "failed", type(exc).__name__
            progress.detail = str(exc)[:300]
        row.update(status=status, error=error, mirror_used=progress.mirror)
        if status == "ready":
            progress.update(entry.total_bytes, status="ready")
        else:
            progress.update(status=status, detail=progress.detail)
        with self._lock:
            self._active.pop(entry.id, None)
            if all(r["status"] not in {"queued", "downloading", "verifying", "compiling"} for r in batch.entries.values()):
                batch.finished_at = batch.finished_at or self.clock()
        self.cache.save()
        if status == "ready" and self.on_installed is not None:
            try:
                self.on_installed()
            except Exception:            # noqa: BLE001
                pass

    def batch_status(self, batch_id: str) -> dict[str, Any]:
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                raise AuditError("BATCH_UNKNOWN", batch_id)
            # statuses live on the progress meters; copy them in so a poller sees them
            for entry_id, row in batch.entries.items():
                meter = self._progress.get(entry_id)
                if meter is not None and row["status"] in {"queued", "downloading", "verifying", "compiling"}:
                    row["status"] = meter.status
            return batch.public()

    def wait(self, batch_id: str, timeout: float | None = None) -> dict[str, Any]:
        with self._lock:
            futures = list(self._batches[batch_id].futures.values())
        for future in futures:
            future.result(timeout=timeout)
        return self.batch_status(batch_id)

    def cancel(self, batch_id: str) -> dict[str, Any]:
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                raise AuditError("BATCH_UNKNOWN", batch_id)
            batch.cancel.set()
        return self.batch_status(batch_id)

    def active_progress(self) -> list[dict[str, Any]]:
        """Latest frame of every in-flight entry (sent to a WebSocket client that connects mid-download)."""
        with self._lock:
            return [m.snapshot() for m in self._progress.values() if m.status in {"queued", "downloading", "verifying", "compiling"}]

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for batch in self._batches.values():
                batch.cancel.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------------------------------------ verify & repair
    def verify_and_repair(self, ids: Sequence[str] | None = None, *, repair: bool = True, accepted_license: bool = False,
                          commercial: bool = False) -> dict[str, Any]:
        """Re-hash everything (cache bypassed), set corrupt files aside, then re-fetch whatever is missing or bad."""
        selected = [self.entries[i] for i in (ids or self.entries)]
        report, need, quarantined = [], [], []
        for entry in selected:
            row = self.inspect(entry, force=True)
            if row["state"] == CORRUPT:
                target = self.target(entry)
                if entry.kind == "asset_set":
                    for item in entry.files:
                        path = target / item["path"]
                        if self._file_state(path, int(item["size_bytes"]), item["sha256"], force=True) == CORRUPT:
                            quarantined.append({"id": entry.id, "moved_to": self.quarantine(path).relative_to(self._root(entry)).as_posix()})
                elif target.is_file():
                    quarantined.append({"id": entry.id, "moved_to": self.quarantine(target).relative_to(self._root(entry)).as_posix()})
                row = self.inspect(entry, force=True)
                row["previous_state"] = CORRUPT
            if row["state"] in {MISSING, PARTIAL}:
                need.append(entry.id)
            report.append({k: row[k] for k in ("id", "state", "hash_status", "detail", "installed_bytes", "size_bytes", "optional")}
                          | {"previous_state": row.get("previous_state")})
        self.cache.save()
        batch = self.start_batch(need, accepted_license=accepted_license, commercial=commercial) if repair and need else None
        corrupt = [r["id"] for r in report if r.get("previous_state") == CORRUPT]
        return {"ok": not any(r["state"] == CORRUPT for r in report), "checked": len(report),
                "verified": sum(r["state"] in INSTALLED_STATES for r in report), "missing": need, "corrupt": corrupt,
                "quarantined": quarantined, "models": report, "repair": batch}


def universal_manifest_report(app_root: Path, *, require_installed: bool = False) -> dict[str, Any]:
    """Read-only check used by ``--verify``: the manifest is valid, no installed entry is corrupt, optionally nothing is missing."""
    manager = UniversalModelDownloadManager(app_root, hash_cache_path=None)
    try:
        listing = manager.manifest(force=True)
    finally:
        manager.close()
    rows = listing["models"]
    corrupt = [r["id"] for r in rows if r["state"] == CORRUPT]
    missing = [r["id"] for r in rows if r["state"] in {MISSING, PARTIAL, NOT_APPLICABLE} and not r["optional"]]
    return {"ok": not corrupt and not (require_installed and missing), "entries": len(rows), "by_state": listing["summary"]["by_state"],
            "corrupt": corrupt, "required_not_installed": missing, "licenses": listing["summary"]["licenses"]}


def _manager_cli(repo: Path, arguments: argparse.Namespace) -> int:
    def print_progress(frame: dict) -> None:
        if frame["status"] in {"ready", "failed", "cancelled", "verifying", "compiling"}:
            print(f"  {frame['model_id']}: {frame['status']}" + (f" via {frame['mirror_used']}" if frame.get("mirror_used") else ""), file=sys.stderr)

    manager = UniversalModelDownloadManager(repo, sink=print_progress if arguments.download is not None or arguments.repair else None,
                                            hash_cache_path=repo / "roop_3d/.runtime/models/verify_cache.json")
    try:
        if arguments.list:
            listing = manager.manifest()
            rows = [{k: r[k] for k in ("id", "state", "license", "license_badge", "requires_acceptance", "mirror_count", "size_bytes")}
                    for r in listing["models"]]
            print(json.dumps({"summary": listing["summary"], "disk": listing["disk"], "models": rows}, indent=2))
            return 0
        if arguments.repair:
            report = manager.verify_and_repair(accepted_license=arguments.accept_license)
            batch = report["repair"]
            if batch and batch["queued"]:
                report["repair"] = manager.wait(batch["batch_id"]) | {"skipped": batch["skipped"]}
            print(json.dumps(report, indent=2))
            failed = [k for k, v in (report["repair"] or {}).get("entries", {}).items() if v["status"] != "ready"]
            return 0 if report["ok"] and not failed else 1
        ids = arguments.download or None
        batch = manager.start_batch(ids, accepted_license=arguments.accept_license)
        outcome = manager.wait(batch["batch_id"]) if batch["queued"] else batch
        print(json.dumps({**outcome, "already_installed": batch["already_installed"], "skipped": batch["skipped"]}, indent=2))
        return 0 if all(v["status"] == "ready" for v in outcome["entries"].values()) else 1
    finally:
        manager.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Read and verify the generated registry and the universal manifest")
    parser.add_argument("--require-installed", action="store_true",
                        help="with --verify: also fail when a non-optional manifest entry is not installed")
    parser.add_argument("--list", action="store_true", help="Print every manifest entry with its install and hash state")
    parser.add_argument("--download", nargs="*", metavar="ID", default=None,
                        help="Download missing manifest entries (all of them when no ID is given) and wait for completion")
    parser.add_argument("--repair", action="store_true", help="Re-hash everything, set corrupt files aside, re-download what is bad")
    parser.add_argument("--build-assets", action="store_true",
                        help="with --verify: allow synthesizing a missing registered CC0 asset (verification is read-only without it)")
    parser.add_argument("--accept-license", action="store_true",
                        help="Acknowledge the licences of entries that are not CC0/MIT/Apache/CC-BY (needed to fetch them)")
    arguments = parser.parse_args()
    repo = Path(__file__).absolute().parents[2]
    registry_path = "roop_3d/audit/verified_local_models.json"
    try:
        if arguments.list or arguments.download is not None or arguments.repair:
            return _manager_cli(repo, arguments)
        scopes, unavailable = default_scopes(repo)
        if arguments.verify:
            registry = json.loads(safe_path(repo, registry_path).read_text("utf-8"))
            result = verify_registry(registry, scopes, unavailable)
            if result["unavailable_scopes"] and arguments.require_installed:
                result["ok"] = False
                result["violations"] = [*result["violations"], {"unavailable_scopes": result["unavailable_scopes"]}]
            open_result = verify_open_artifacts(repo.parent)   # checkout root; optional pinned open-licence files
            result["open_artifacts"] = open_result["artifacts"]
            local_result = verify_local_cc0_assets(repo, build_assets=arguments.build_assets)
            result["local_assets"] = local_result["assets"]
            if not local_result["ok"]:
                result["ok"] = False
                result["violations"] = [*result["violations"], *local_result["violations"]]
            if not open_result["ok"]:
                result["ok"] = False
                result["violations"] = [*result["violations"], *open_result["violations"]]
            manifest_result = universal_manifest_report(repo, require_installed=arguments.require_installed)
            result["universal_manifest"] = manifest_result
            if not manifest_result["ok"]:
                result["ok"] = False
                result["violations"] = [*result["violations"], {"universal_manifest": {
                    "corrupt": manifest_result["corrupt"], "required_not_installed": manifest_result["required_not_installed"]
                    if arguments.require_installed else []}}]
        else:
            old = json.loads(safe_path(repo, "roop_3d/audit/local_models_discovered.json").read_text("utf-8"))
            expected = {os.path.normcase(str(Path(m["path"]).absolute())): m["sha256"] for m in old["models"]}
            registry = discover(scopes, expected)
            result = {"ok": True, "models": len(registry["models"]), "family_counts": registry["family_counts"]}
            # Refuse to silently lose artifacts from the original evidence set.
            identities = {os.path.normcase(str((scopes[m["locator"]["scope"]] / m["locator"]["path"]).absolute()))
                          for m in registry["models"]}
            missing = sorted(set(expected) - identities)
            if missing:
                raise AuditError("MODEL_DISCOVERY_INCOMPLETE", f"Prior models not rediscovered: {missing}")
            write_json(repo, registry_path, registry)
            write_text(repo, "roop_3d/audit/model_and_license_register.csv", license_register(registry))
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1
    except (OSError, ValueError, KeyError, AuditError) as error:
        envelope = error.envelope() if isinstance(error, AuditError) else {
            "ok": False, "error": {"code": "DISCOVERY_FAILED", "detail": str(error)}}
        print(json.dumps(envelope, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
