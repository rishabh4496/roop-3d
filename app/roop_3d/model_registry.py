"""Unified model inspection backed by the authoritative artifact registry.

Identity and download evidence are never duplicated as hand-written hashes or
guessed tensor/VRAM declarations. Byte availability does not establish inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from roop_3d.audit.project_paths import application_root
from roop_3d.core.errors import DomainError
from roop_3d.models.catalog import ModelCatalog
from roop_3d.models.registry import ModelRegistry, authorize

ModelCategory = Literal["enhancer", "3d_geometry", "detector_embedder", "occlusion_masking"]
ReadinessState = Literal["DOWNLOADED (NEEDS TRT)", "MISSING", "CORRUPT HASH", "BLOCKED"]
PrecisionPolicy = Literal["FP32", "FP16"]


@dataclass(frozen=True)
class ModelManifestEntry:
    model_id: str
    name: str
    category: ModelCategory
    primary_url: str | None
    fallback_url: str | None
    expected_size_bytes: int
    expected_sha256: str
    input_shapes: dict[str, list[int | str]]
    output_shapes: dict[str, list[int | str]]
    precision_policy: PrecisionPolicy
    supports_dynamic_batch: bool
    vram_footprint_mb: int | None
    description: str
    companion_files: tuple[str, ...] = ()
    license_id: str = "UNKNOWN"
    registry_model_id: str = ""


def _entries() -> tuple[ModelManifestEntry, ...]:
    registry = ModelRegistry()
    selections = (
        ("1k3d68", "f4d0609acdc3491ba09003bc", "3d_geometry"),
        ("2d106det", "7515daa40c84ce8be15c8a3c", "3d_geometry"),
        ("det_10g", "caa047119ea73fb50b225d60", "detector_embedder"),
        ("w600k_r50", "d9235365b06d994596a8ef35", "detector_embedder"),
        ("yoloface_8n", "d1d47cf8fa4f92704da30528", "detector_embedder"),
        ("xseg", "58e30ce4736df56fb21d47e2", "occlusion_masking"),
        ("xseg_3", "48cc67a78ac3f81a9c5c1499", "occlusion_masking"),
        ("bisenet_resnet_34", "9aed4e37a414df474326e07a", "occlusion_masking"),
        ("ultramax_uv", "ultramax_uv", "enhancer"),
    )
    result = []
    for model_id, registry_id, category in selections:
        record = registry.get(registry_id)
        graph = record.input_output_contract
        shapes = lambda key: {node["name"]: node["shape"] for node in graph.get(key, ())
                              if "name" in node and "shape" in node}
        result.append(ModelManifestEntry(
            model_id=model_id, name=record.name, category=category,
            primary_url=record.upstream_url, fallback_url=None,
            expected_size_bytes=record.byte_size, expected_sha256=record.expected_sha256,
            input_shapes=shapes("inputs"), output_shapes=shapes("outputs"),
            precision_policy="FP32", supports_dynamic_batch=False, vram_footprint_mb=None,
            description=record.role_evidence, license_id=record.license_id, registry_model_id=registry_id))
    return tuple(result)


UNIFIED_MODEL_ENTRIES = _entries()
UNIFIED_MANIFEST_MAP = {entry.model_id: entry for entry in UNIFIED_MODEL_ENTRIES}
UNIFIED_NAME_MAP = {entry.name: entry for entry in UNIFIED_MODEL_ENTRIES}


class UnifiedModelRegistry:
    """Read-only byte inspection; session execution remains a separate receipt."""

    def __init__(self, root: Path | None = None):
        self.root = root or application_root()
        self.entries = UNIFIED_MANIFEST_MAP
        self.name_map = UNIFIED_NAME_MAP
        self.catalog = ModelCatalog(self.root)

    def get_search_paths(self) -> list[Path]:
        return [self.root / "models", self.root / "models/buffalo_l",
                self.root / "roop_3d/models/cache"]

    def resolve_model_file(self, entry: ModelManifestEntry) -> Path | None:
        try:
            return self.catalog.resolve_named_artifact(entry.name)[1]
        except DomainError as error:
            if error.code == "MODEL_NOT_INSTALLED":
                return None
            raise

    def inspect_model_readiness(self, entry: ModelManifestEntry) -> dict[str, Any]:
        path = None
        identity = None
        error = None
        try:
            _, path = self.catalog.resolve_named_artifact(entry.name)
            identity = self.catalog.artifact_identity(path)
            readiness = "DOWNLOADED (NEEDS TRT)"
            reason = "Input bytes resolved; provider initialization and forward execution have not been verified."
        except DomainError as failure:
            error = {"code": failure.code, "message": failure.message, "status": failure.status}
            readiness = ("MISSING" if failure.code == "MODEL_NOT_INSTALLED" else
                         "CORRUPT HASH" if failure.code in {"MODEL_DRIFT", "MODEL_QUARANTINED"} else "BLOCKED")
            reason = failure.message
        record = self.catalog.registry.get(entry.registry_model_id)
        try:
            authorize(record, action="download")
            download_error = None
        except DomainError as failure:
            download_error = failure.code
        return {
            "model_id": entry.model_id, "registry_model_id": entry.registry_model_id,
            "name": entry.name, "category": entry.category, "readiness_state": readiness,
            "vram_footprint_mb": None, "precision_policy": None, "supports_dynamic_batch": None,
            "expected_size_bytes": entry.expected_size_bytes, "expected_sha256": entry.expected_sha256,
            "actual_sha256": identity["sha256"] if identity else None,
            "hash_valid": bool(identity and identity["sha256"] == entry.expected_sha256
                               and identity["size_bytes"] == entry.expected_size_bytes),
            "trust_basis": identity["trust_basis"] if identity else None,
            "reason": reason, "error": error, "model_path": str(path) if path else None,
            "trt_engine_path": None, "engine_presence_verified": False, "actual_inference_available": False,
            "description": entry.description, "primary_url": record.upstream_url, "fallback_url": None,
            "download_gate": download_error, "license_id": record.license_id,
            "license_notice": record.license_notice, "commercial_use": record.commercial_use,
        }

    def get_full_readiness_report(self) -> list[dict[str, Any]]:
        from roop_3d.models.stage16 import readiness_report
        return [self.inspect_model_readiness(entry) for entry in UNIFIED_MODEL_ENTRIES] + readiness_report(self.root)
