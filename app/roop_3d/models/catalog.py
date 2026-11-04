"""Compatibility summaries backed by the authoritative Stage 03 registry."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from roop_3d.audit.project_paths import application_root
from roop_3d.core.errors import DomainError
from roop_3d.models.downloader import Downloader, DownloadResult
from roop_3d.models.governance import ModelCompatibilityScanner
from roop_3d.models.local_provider import LocalWeightProvider
from roop_3d.models.registry import ModelRecord, ModelRegistry, authorize, license_info
from roop_3d.core.storage import atomic_json


def render_dependencies(catalog) -> dict[str, str]:
    """Snapshot reference/stream model candidates, including optional local adapters.

    These are dependency identities, not a claim that every candidate ran. Actual activation
    still verifies bytes and reports providers through the existing inference adapters.
    """
    names = {"inswapper_128.onnx", "inswapper_128.batch.onnx", "2d106det.onnx",
             "liveportrait_landmark.onnx", "liveportrait_appearance_feature_extractor.onnx",
             "liveportrait_motion_extractor.onnx", "liveportrait_warping_spade.onnx",
             "liveportrait_stitching.onnx", "liveportrait_stitching_eye.onnx",
             "liveportrait_stitching_lip.onnx", "bisenet_resnet_34.onnx",
             "gpen_bfr_512.onnx", "CodeFormerv0.1.onnx"}
    selected = [catalog.detector(), catalog.recognizer()]
    required = {record["model_id"] for record in selected}
    seen = set()
    for record in catalog.records():
        if record["name"] in names and record["name"] not in seen:
            selected.append(record)
            seen.add(record["name"])
            if record["name"] == "inswapper_128.onnx":
                required.add(record["model_id"])
    records = {record["model_id"]: record for record in selected}
    available = {}
    for key, record in records.items():
        try:
            catalog.verify(record)
        except DomainError as error:
            if key not in required and error.code == "MODEL_NOT_INSTALLED":
                continue
            raise
        available[key] = record["sha256"]
    return available


class ModelCatalog:
    def __init__(self, root: Path | None = None, *, cache_root: Path | None = None,
                 search_paths: Callable[[], tuple[str, ...]] | None = None,
                 registry: ModelRegistry | None = None):
        self.root = root or application_root()
        self.registry = registry or ModelRegistry(self.root)
        self.provider = LocalWeightProvider(self.registry, root=cache_root or self.root, search_paths=search_paths)
        self.downloader = Downloader(cache_root or self.root)
        self.governance = ModelCompatibilityScanner()
        self.pin_root = cache_root or self.root

    @staticmethod
    def _legacy(record: ModelRecord) -> dict[str, Any]:
        return {"model_id": record.model_id, "name": record.name, "sha256": record.expected_sha256,
                "size_bytes": record.byte_size, "state": record.status.upper(), "family_hint": record.family_hint,
                "verified_at": record.verified_at, "license_state": "UNKNOWN" if record.license_id == "UNKNOWN" else "VERIFIED_REMOTE",
                "provenance_state": record.provenance_state, "inference_state": "UNKNOWN",
                "commercial_permission": "UNKNOWN", "graph": record.input_output_contract,
                "locator": record.locator.model_dump() if record.locator else None,
                "role": record.role, "license_id": record.license_id, "commercial_use": record.commercial_use,
                "license_notice": record.license_notice, "manual_acceptance_required": record.manual_acceptance_required}

    def records(self) -> list[dict[str, Any]]:
        return [self._legacy(record) for record in self.registry.records() if record.status == "verified_local"]

    def public(self) -> list[dict[str, Any]]:
        return [{key: value for key, value in record.items() if key not in {"graph", "locator"}}
                for record in self.records()]

    def get(self, model_id: str) -> dict[str, Any]:
        return self._legacy(self.registry.get(model_id))

    def _path(self, record: dict[str, Any]) -> Path:
        return self.provider.resolve(record["model_id"])

    def verify(self, record: dict[str, Any]) -> None:
        self.provider.resolve(record["model_id"])

    def verified_bytes(self, record: dict[str, Any]) -> bytes:
        model = self.registry.get(record["model_id"])
        if model.input_output_contract.get("external_data"):
            raise DomainError("MODEL_UNSUPPORTED", "Detector cannot load external tensor files")
        self.resolve_for_inference(model.model_id)
        with self.provider.open(model) as stream:
            data = stream.read()
        if len(data) != model.byte_size or hashlib.sha256(data).hexdigest() != model.expected_sha256:
            raise DomainError("MODEL_DRIFT", "Model integrity differs from the preserved registry")
        return data

    def ensure_available(self, model_id: str, *, accepted_license: bool = False,
                         commercial: bool = False) -> DownloadResult:
        model = self.registry.get(model_id)
        authorize(model, action="activate", accepted_license=accepted_license, commercial=commercial)
        if model.expected_sha256 and model.byte_size:
            try:
                return DownloadResult(self.provider.resolve(model), False, False)
            except DomainError as error:
                if error.code != "MODEL_NOT_INSTALLED":
                    raise
            cached = self.downloader.cached(model)
            if cached:
                return DownloadResult(cached, False, False)
        return self.downloader.download(model, accepted_license=accepted_license, commercial=commercial)

    def lifecycle(self, model_id: str, *, hardware: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return governance state; resolving verifies bytes and never loads the model."""
        model = self.registry.get(model_id)
        try:
            self.provider.resolve(model)
        except DomainError as error:
            if error.code == "MODEL_NOT_INSTALLED":
                return self.governance.scan(model, installed=False, integrity_verified=None, hardware=hardware)
            if error.code == "MODEL_QUARANTINED":
                result = self.governance.scan(model, installed=True, integrity_verified=False, hardware=hardware)
                result["lifecycle_state"] = "QUARANTINED"
                return result
            raise
        return self.governance.scan(model, installed=True, integrity_verified=True, hardware=hardware)

    def resolve_for_inference(self, model_id: str, *, hardware: dict[str, Any] | None = None,
                              precision: str | None = None) -> Path:
        """Resolve only a hash-verified artifact that passes the governance gate."""
        model = self.registry.get(model_id)
        path = self.provider.resolve(model)
        result = self.governance.scan(model, installed=True, integrity_verified=True,
                                      hardware=hardware, precision=precision)
        if result["lifecycle_state"] != "READY":
            code = "LICENSE_UNVERIFIED" if result["lifecycle_state"] == "LICENSE_RESTRICTED" else "MODEL_INCOMPATIBLE"
            raise DomainError(code, "Model is not eligible for inference: " + "; ".join(result["issues"]), 409)
        authorize(model, action="activate")
        return path

    def pin_version(self, family: str, model_id: str, *, hardware: dict[str, Any] | None = None) -> dict[str, Any]:
        """Atomically pin an already-installed, fully governed immutable version."""
        if not family or len(family) > 96 or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in family):
            raise DomainError("MODEL_VERSION_FAMILY_INVALID", "Version family must be a portable lowercase identifier", 422)
        record = self.registry.get(model_id)
        if record.version_family != family:
            raise DomainError("MODEL_VERSION_FAMILY_MISMATCH", "Model is not registered in this version family", 409)
        state = self.lifecycle(model_id, hardware=hardware)
        if state["lifecycle_state"] != "READY":
            raise DomainError("MODEL_VERSION_NOT_READY", "Only installed, verified, compatible, licensed versions can be pinned", 409)
        relative = "roop_3d/.runtime/model-pins.json"
        path = self.pin_root / relative
        try:
            data = json.loads(path.read_text("utf-8")) if path.exists() else {"schema_version": 1, "families": {}}
            families = data["families"]
            history = list(families.get(family, []))
        except (OSError, ValueError, KeyError, TypeError):
            raise DomainError("MODEL_PINS_INVALID", "Model version pin state is invalid", 503) from None
        if not history or history[-1] != model_id:
            history.append(model_id)
        atomic_json(self.pin_root, relative, {"schema_version": 1, "families": {**families, family: history}})
        return {"family": family, "active_model_id": model_id, "previous_model_id": history[-2] if len(history) > 1 else None}

    def update_version(self, family: str, model_id: str, *, accepted_license: bool = False,
                       commercial: bool = False, hardware: dict[str, Any] | None = None) -> dict[str, Any]:
        """Download an explicitly selected registered revision, then pin it after readiness checks."""
        record = self.registry.get(model_id)
        if record.version_family != family:
            raise DomainError("MODEL_VERSION_FAMILY_MISMATCH", "Model is not registered in this version family", 409)
        result = self.ensure_available(model_id, accepted_license=accepted_license, commercial=commercial)
        pin = self.pin_version(family, model_id, hardware=hardware)
        return {**pin, "downloaded": result.downloaded, "resumed": result.resumed}

    def rollback_version(self, family: str) -> dict[str, Any]:
        """Restore the previous registered pin without replacing either artifact."""
        relative = "roop_3d/.runtime/model-pins.json"
        path = self.pin_root / relative
        try:
            data = json.loads(path.read_text("utf-8"))
            history = list(data["families"][family])
        except (OSError, ValueError, KeyError, TypeError):
            raise DomainError("MODEL_VERSION_ROLLBACK_UNAVAILABLE", "No previous model version pin is available", 409) from None
        if len(history) < 2:
            raise DomainError("MODEL_VERSION_ROLLBACK_UNAVAILABLE", "No previous model version pin is available", 409)
        previous = history[-2]
        state = self.lifecycle(previous)
        if state["lifecycle_state"] != "READY":
            raise DomainError("MODEL_VERSION_ROLLBACK_UNAVAILABLE", "Previous model version is no longer verified and compatible", 409)
        history.append(previous)
        atomic_json(self.pin_root, relative, {"schema_version": 1,
                    "families": {**data["families"], family: history}})
        return {"family": family, "active_model_id": previous, "rolled_back": True}

    def pinned_version(self, family: str) -> ModelRecord:
        relative = "roop_3d/.runtime/model-pins.json"
        try:
            data = json.loads((self.pin_root / relative).read_text("utf-8"))
            model_id = data["families"][family][-1]
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            raise DomainError("MODEL_VERSION_PIN_NOT_FOUND", "No model version is pinned for this family", 404) from None
        return self.registry.get(model_id)

    def select_version(self, record: ModelRecord) -> ModelRecord:
        """Use an explicit family pin when one exists; otherwise preserve registry order."""
        if record.version_family == "UNKNOWN":
            return record
        try:
            pinned = self.pinned_version(record.version_family)
        except DomainError as error:
            if error.code == "MODEL_VERSION_PIN_NOT_FOUND":
                return record
            raise
        return pinned if pinned.role == record.role and pinned.version_family == record.version_family else record

    def license(self, model_id: str) -> dict[str, Any]:
        return license_info(self.registry.get(model_id))

    def detector(self) -> dict[str, Any]:
        for registered in self.registry.records():
            record = self.select_version(registered)
            record = self._legacy(record)
            if record["name"] in {"scrfd_10g_bnkps.onnx", "det_10g.onnx"} and len(record["graph"].get("outputs", [])) == 9:
                return record
        raise DomainError("DETECTOR_UNAVAILABLE", "No supported local SCRFD model is registered", 503)

    def recognizer(self) -> dict[str, Any]:
        for preferred in ("w600k_r50.onnx", "arcface_w600k_r50.onnx", "w600k_mbf.onnx"):
            for registered in self.registry.records():
                record = self._legacy(self.select_version(registered))
                if record["name"] == preferred and record.get("role") == "recognizer":
                    return record
        for registered in self.registry.records():
            record = self._legacy(self.select_version(registered))
            if record.get("role") == "recognizer":
                return record
        raise DomainError("RECOGNIZER_UNAVAILABLE", "No supported local recognizer model is registered", 503)
