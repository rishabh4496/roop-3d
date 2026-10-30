"""Keep artifact identity, recorded integrity, inference and legal evidence separate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from roop_3d.audit.isolation import safe_path, sha256_file
from roop_3d.audit.project_paths import application_root, legacy_root
from roop_3d.core.errors import DomainError


class ModelCatalog:
    def __init__(self, root: Path | None = None):
        self.root = root or application_root()

    def records(self) -> list[dict[str, Any]]:
        path = safe_path(self.root, "roop_3d/audit/verified_local_models.json")
        if not path.exists():
            return []
        registry = json.loads(path.read_text("utf-8"))
        return [{**item, "model_id": hashlib.sha256(
            f"{item['locator']['scope']}:{item['locator']['path']}".encode()).hexdigest()[:24],
                 "verified_at": registry["timestamp"]} for item in registry["models"]]

    def public(self) -> list[dict[str, Any]]:
        keys = {"model_id", "name", "sha256", "size_bytes", "state", "family_hint", "verified_at",
                "license_state", "provenance_state", "inference_state", "commercial_permission"}
        return [{key: value for key, value in record.items() if key in keys} for record in self.records()]

    def get(self, model_id: str) -> dict[str, Any]:
        for record in self.records():
            if record["model_id"] == model_id:
                return record
        raise DomainError("MODEL_NOT_FOUND", "Model is not in the local registry", 404)

    def _path(self, record: dict[str, Any]) -> Path:
        if record["state"] != "VERIFIED_LOCAL":
            raise DomainError("MODEL_UNSUPPORTED", "Model has no verified local artifact")
        locator = record["locator"]
        home = Path.home()
        scopes = {"repository": legacy_root(self.root), "user_cache": home / ".cache",
                  "insightface_cache": home / ".insightface", "torch_cache": home / ".torch"}
        return safe_path(scopes[locator["scope"]], locator["path"])

    def verify(self, record: dict[str, Any]) -> None:
        size, digest = sha256_file(self._path(record))
        if size != record["size_bytes"] or digest != record["sha256"]:
            raise DomainError("MODEL_DRIFT", "Model integrity differs from the preserved registry")

    def verified_bytes(self, record: dict[str, Any]) -> bytes:
        if record.get("graph", {}).get("external_data"):
            raise DomainError("MODEL_UNSUPPORTED", "Detector cannot load external tensor files")
        path = self._path(record)
        with path.open("rb") as stream:
            data = stream.read()
        if len(data) != record["size_bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise DomainError("MODEL_DRIFT", "Model integrity differs from the preserved registry")
        return data

    def detector(self) -> dict[str, Any]:
        for record in self.records():
            if record["name"] in {"scrfd_10g_bnkps.onnx", "det_10g.onnx"} and len(record["graph"].get("outputs", [])) == 9:
                return record
        raise DomainError("DETECTOR_UNAVAILABLE", "No supported local SCRFD model is registered", 503)
