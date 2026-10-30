"""Compatibility summaries backed by the authoritative Stage 03 registry."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

from roop_3d.audit.project_paths import application_root
from roop_3d.core.errors import DomainError
from roop_3d.models.downloader import Downloader, DownloadResult
from roop_3d.models.local_provider import LocalWeightProvider
from roop_3d.models.registry import ModelRecord, ModelRegistry, authorize, license_info


class ModelCatalog:
    def __init__(self, root: Path | None = None, *, cache_root: Path | None = None,
                 search_paths: Callable[[], tuple[str, ...]] | None = None,
                 registry: ModelRegistry | None = None):
        self.root = root or application_root()
        self.registry = registry or ModelRegistry(self.root)
        self.provider = LocalWeightProvider(self.registry, root=cache_root or self.root, search_paths=search_paths)
        self.downloader = Downloader(cache_root or self.root)

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

    def license(self, model_id: str) -> dict[str, Any]:
        return license_info(self.registry.get(model_id))

    def detector(self) -> dict[str, Any]:
        for record in self.records():
            if record["name"] in {"scrfd_10g_bnkps.onnx", "det_10g.onnx"} and len(record["graph"].get("outputs", [])) == 9:
                return record
        raise DomainError("DETECTOR_UNAVAILABLE", "No supported local SCRFD model is registered", 503)
