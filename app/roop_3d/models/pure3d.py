"""Offline component resolution. READY means resolvable inputs, never successful inference.

All weights cross ModelCatalog's integrity/use-policy gate. This module does not
create sessions, download, build assets, or waive the production-renderer guard.
"""
from __future__ import annotations

from functools import lru_cache
import importlib.util
import json
from pathlib import Path
from typing import Any

from roop_3d.audit.isolation import AuditError, safe_path
from roop_3d.core.errors import DomainError
from roop_3d.models.catalog import ModelCatalog, observed_digest
from roop_3d.models.registry import authorize
from roop_3d.models.verifier import verify_file

_WEIGHT_COMPONENTS = {
    "detector": ("Buffalo face detector", ("det_10g.onnx", "scrfd_10g_bnkps.onnx")),
    "recognizer": ("ArcFace identity encoder", ("w600k_r50.onnx", "arcface_w600k_r50.onnx")),
    "dense_landmarks_3d": ("Measured 3D landmarks", ("1k3d68.onnx",)),
    "dense_landmarks_2d": ("Dense contour landmarks", ("2d106det.onnx",)),
    "face_parser": ("Hair-excluding semantic parser", ("bisenet_resnet_34.onnx",)),
    "occlusion_mask": ("XSeg occlusion mask", ("xseg.onnx", "xseg_3.onnx", "face_occluder.onnx")),
    "uv_enhancer": ("UltraMax canonical UV detail", ("ultramax.onnx",)),
}
COMPONENT_IDS = ("detector", "recognizer", "dense_landmarks_3d", "dense_landmarks_2d",
                 "head_reconstructor", "uv_synthesis", "face_parser", "occlusion_mask", "uv_enhancer")


def _error(error: DomainError) -> dict[str, Any]:
    return {"code": error.code, "message": error.message, "status": error.status}


def _state(error: DomainError) -> str:
    if error.code in {"MODEL_NOT_INSTALLED", "MAKEHUMAN_ASSET_MISSING", "ANATOMICAL_PACK_MISSING", "DENSE_FACE_BUNDLE_MISSING"}:
        return "MISSING"
    if error.code in {"MODEL_DRIFT", "MODEL_CHANGED", "MODEL_QUARANTINED", "MAKEHUMAN_ASSET_CORRUPT",
                      "ANATOMICAL_PACK_CORRUPT", "DENSE_FACE_BUNDLE_INVALID"}:
        return "CORRUPT"
    if error.status == 403 or error.code in {"LICENSE_UNVERIFIED", "MODEL_BLOCKED"}:
        return "BLOCKED"
    return "UNAVAILABLE"


def _row(component: str, label: str, kind: str) -> dict[str, Any]:
    return {"component": component, "label": label, "kind": kind, "required": True,
            "state": "UNAVAILABLE", "model_id": None, "filename": None, "path": None,
            "sha256": None, "size_bytes": None, "trust_basis": None, "integrity_verified": False,
            "license": {"license_id": "UNKNOWN", "reviewed": False, "commercial_use": "unknown", "notice": ""},
            "error": None, "acquisition": None, "alternatives": [], "assets": []}


def _acquisition(catalog: ModelCatalog, names: tuple[str, ...]) -> str:
    """Use only reviewed registry URLs/digests, or state the explicit local-license route."""
    records = [r for r in catalog.registry.records() if any(n in r.filenames for n in names)]
    downloads = []
    for record in records:
        try:
            authorize(record, action="download", accepted_license=True)
        except DomainError:
            continue
        downloads.append(f"{record.upstream_url} (model_id={record.model_id}; SHA-256={record.expected_sha256}; "
                         f"bytes={record.byte_size}; license acknowledgement required={record.manual_acceptance_required})")
    instructions = (f"Obtain {names[0]} under your own license and place it at {catalog.root / 'models' / names[0]}. "
                    "Placement enables the existing local-use policy, not license approval. Managed imports require "
                    "the registered SHA-256 and size.")
    return instructions + (" Reviewed pinned download(s): " + "; ".join(dict.fromkeys(downloads)) if downloads else
                           " No verified public download is registered; use a legally obtained local artifact.")


def _weight(catalog: ModelCatalog, component: str) -> dict[str, Any]:
    label, names = _WEIGHT_COMPONENTS[component]
    row = _row(component, label, "weight")
    failures = []
    for name in names:
        try:
            record, path = catalog.resolve_named_artifact(name)
            identity = catalog.artifact_identity(path)
            matched = catalog.registry.get(identity["registry_model_id"]) if identity["registry_model_id"] else None
            reviewed = bool(matched and matched.license_id not in {"UNKNOWN", "LOCAL-USE-ACKNOWLEDGED-UNVERIFIED"})
            row.update(state="READY", model_id=record.model_id, filename=path.name, path=str(path),
                       sha256=identity["sha256"], size_bytes=identity["size_bytes"],
                       trust_basis=identity["trust_basis"], integrity_verified=matched is not None,
                       license={"license_id": matched.license_id if matched else "UNKNOWN",
                                "reviewed": reviewed, "commercial_use": matched.commercial_use if matched else "unknown",
                                "notice": matched.license_notice if matched else "Explicit local-use bytes; upstream rights unreviewed"})
            row["alternatives"] = failures
            return row
        except DomainError as error:
            failures.append({"filename": name, "error": _error(error)})
    chosen = next((f["error"] for f in failures if f["error"]["code"] != "MODEL_NOT_INSTALLED"), failures[0]["error"])
    error = DomainError(chosen["code"], chosen["message"], chosen["status"])
    row.update(state=_state(error), filename=names[0], error=chosen,
               alternatives=failures, acquisition=_acquisition(catalog, names))
    return row


@lru_cache(maxsize=16)
def _check_asset(path: str, identity: tuple, kind: str) -> None:
    # identity includes both the NPZ and sidecar stats, so rebuilt inputs invalidate the cache.
    if kind == "makehuman":
        from roop_3d.reconstruction.makehuman_head import MakeHumanHeadModel
        MakeHumanHeadModel.load(path, auto_build=False, verify=True)
    else:
        from roop_3d.reconstruction.anatomical_pack import AnatomicalFullHeadAsset
        AnatomicalFullHeadAsset.load(path, auto_build=False, verify=True)


def _asset(root: Path, kind: str, relative: str) -> dict[str, Any]:
    row = _row(kind, kind, "asset")
    path = root / relative
    row.update(model_id=kind, filename=path.name, path=str(path))
    try:
        path = safe_path(root, relative)
        if not path.is_file():
            code = "MAKEHUMAN_ASSET_MISSING" if kind == "makehuman" else "ANATOMICAL_PACK_MISSING"
            raise DomainError(code, f"Prepared head asset is missing: {path}", 404)
        sidecar = safe_path(root, relative + ".json")
        if not sidecar.is_file():
            code = "MAKEHUMAN_ASSET_CORRUPT" if kind == "makehuman" else "ANATOMICAL_PACK_CORRUPT"
            raise DomainError(code, f"Asset digest sidecar is missing: {sidecar}", 422)
        signature = tuple((p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns) for p in (path, sidecar))
        _check_asset(str(path), signature, kind)
        digest, size = observed_digest(path)
        row.update(state="READY", sha256=digest, size_bytes=size, trust_basis="ASSET_DIGEST", integrity_verified=True,
                   license={"license_id": "CC0-1.0", "reviewed": True, "commercial_use": "allow",
                            "notice": "MakeHuman-derived CC0 asset; local geometry/sidecar digest checked, not inference."})
    except (OSError, ValueError, KeyError, TypeError, EOFError, AuditError) as error:
        error = DomainError("MAKEHUMAN_ASSET_CORRUPT" if kind == "makehuman" else "ANATOMICAL_PACK_CORRUPT", str(error), 422)
        row.update(state=_state(error), error=_error(error))
    except DomainError as error:
        row.update(state=_state(error), error=_error(error))
    if row["error"]:
        command = ("python -B -m roop_3d.reconstruction.makehuman_build" if kind == "makehuman"
                   else "python -B -m roop_3d.reconstruction.anatomical_pack")
        row["acquisition"] = command + " (build from pinned CC0 sources; no restricted FLAME license is waived)"
    return row


def _dense_bundle(root: Path) -> dict[str, Any]:
    row = _row("dense_face_bundle", "Optional dense-face reconstructor", "asset")
    path = root / "roop_3d/.runtime/models/dense_face/bundle.json"
    row.update(required=False, model_id="dense_face_bundle", filename=path.name, path=str(path))
    try:
        path = safe_path(root, "roop_3d/.runtime/models/dense_face/bundle.json")
        if not path.is_file():
            raise DomainError("DENSE_FACE_BUNDLE_MISSING", f"No prepared dense-face bundle at {path}", 404)
        spec = json.loads(path.read_text("utf-8"))
        if not isinstance(spec, dict) or spec.get("schema") != "roop3d.native_dense_face.v1" or not spec.get("assets"):
            raise DomainError("DENSE_FACE_BUNDLE_INVALID", "Dense-face bundle schema/assets are invalid", 422)
        for role, asset in spec["assets"].items():
            child = safe_path(path.parent, asset["file"])
            if not child.is_file():
                raise DomainError("DENSE_FACE_BUNDLE_MISSING", f"Bundle asset {role} is missing: {child}", 404)
            verify_file(child, asset["sha256"], child.stat().st_size)
        digest, size = observed_digest(path)
        row.update(state="READY", sha256=digest, size_bytes=size, trust_basis="ASSET_DIGEST", integrity_verified=True,
                   license={"license_id": "UNKNOWN", "reviewed": False, "commercial_use": "unknown",
                            "notice": "Bundle hashes do not establish BFM/FLAME/SMIRK/HRN licenses. Review each original license before use."})
    except DomainError as error:
        row.update(state=_state(error), error=_error(error))
    except (OSError, ValueError, KeyError, TypeError, AuditError) as error:
        error = DomainError("DENSE_FACE_BUNDLE_INVALID", f"Invalid dense-face bundle: {error}", 422)
        row.update(state=_state(error), error=_error(error))
    if row["error"]:
        row["acquisition"] = ("Obtain BFM/FLAME and checkpoint licenses from their official sources, then prepare "
                              "the bundle using roop_3d/tools/prepare_dense_face.py; no automatic license acceptance or fabricated mirrors.")
    return row


def resolve_component(catalog: ModelCatalog, component_id: str, *, root: Path | None = None) -> dict[str, Any]:
    """Resolve component inputs only. Loaded provider/forward evidence belongs to the session layer."""
    root = root or catalog.root
    if component_id in _WEIGHT_COMPONENTS:
        return _weight(catalog, component_id)
    if component_id == "head_reconstructor":
        row = _row(component_id, "MakeHuman full-head reconstructor", "asset")
        assets = [_asset(root, "makehuman", "roop_3d/.runtime/models/makehuman/makehuman_head_v1.npz"),
                  _asset(root, "anatomical", "roop_3d/.runtime/models/anatomical/makehuman_cc0_anatomical_pack.npz"),
                  _dense_bundle(root)]
        failure = next((a for a in assets if a["required"] and a["state"] != "READY"), None)
        row.update({key: assets[0][key] for key in ("model_id", "filename", "path", "sha256", "size_bytes", "license")})
        row.update(state=failure["state"] if failure else "READY", assets=assets,
                   integrity_verified=failure is None, trust_basis="ASSET_DIGEST" if failure is None else None,
                   error=failure["error"] if failure else None, acquisition=failure["acquisition"] if failure else None)
        return row
    if component_id == "uv_synthesis":
        row = _row(component_id, "Visibility-weighted mathematical UV synthesis", "implementation")
        module = "roop_3d.reconstruction.uv_synthesizer"
        spec = importlib.util.find_spec(module)
        row.update(model_id=module + ".UVSynthesizer", path=spec.origin if spec else None,
                   filename=Path(spec.origin).name if spec and spec.origin else None,
                   state="READY" if spec else "MISSING", trust_basis="IMPLEMENTATION" if spec else None)
        if not spec:
            row["error"] = _error(DomainError("UV_IMPLEMENTATION_MISSING", f"Required module {module} is unavailable", 503))
        return row
    raise DomainError("MODEL_COMPONENT_UNKNOWN", f"Unknown pure-3D component: {component_id}", 404)


def resolve_components(catalog: ModelCatalog, *, root: Path | None = None) -> list[dict[str, Any]]:
    return [resolve_component(catalog, component, root=root) for component in COMPONENT_IDS]
