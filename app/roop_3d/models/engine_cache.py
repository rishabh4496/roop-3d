"""Hardware-keyed metadata only; this module does not compile or load engines."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from roop_3d.audit.isolation import sha256_file
from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract
from roop_3d.core.storage import atomic_json, exclusive, owned
from roop_3d.models.verifier import verify_file


class ShapeRange(Contract):
    minimum: tuple[int, ...] = Field(min_length=1, max_length=16)
    optimum: tuple[int, ...] = Field(min_length=1, max_length=16)
    maximum: tuple[int, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def ordered(self) -> "ShapeRange":
        if (len({len(self.minimum), len(self.optimum), len(self.maximum)}) != 1
                or any(not 0 < a <= b <= c for a, b, c in zip(self.minimum, self.optimum, self.maximum))):
            raise ValueError("Shape profile dimensions must be positive and min <= opt <= max")
        return self


class EngineKey(Contract):
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    precision: Literal["fp32", "fp16", "int8"]
    gpu_compute_capability: str = Field(pattern=r"^[0-9]+\.[0-9]+$")
    driver: str = Field(min_length=1, max_length=128)
    onnxruntime_version: str = Field(min_length=1, max_length=128)
    tensorrt_version: str | None = Field(max_length=128)
    shape_profile: dict[str, ShapeRange] = Field(min_length=1, max_length=64)

    def digest(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


class EngineMetadata(Contract):
    schema_version: Literal[1] = 1
    key: EngineKey
    kind: Literal["engine", "onnx"]
    compiled_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiled_byte_size: int = Field(gt=0, strict=True)


class EngineCache:
    def __init__(self, root: Path):
        self.root = root

    def paths(self, key: EngineKey, kind: Literal["engine", "onnx"] = "engine") -> tuple[Path, Path]:
        if kind not in {"engine", "onnx"}:
            raise ValueError("Unsupported compiled artifact type")
        if kind == "engine" and not key.tensorrt_version:
            raise ValueError("TensorRT engines require an observed TensorRT version")
        base = f"roop_3d/cache/engines/{key.digest()}.{kind}"
        return owned(self.root, base), owned(self.root, base + ".json")

    def record(self, key: EngineKey, kind: Literal["engine", "onnx"] = "engine") -> EngineMetadata:
        artifact, metadata = self.paths(key, kind)
        with exclusive(self.root, metadata.relative_to(self.root).as_posix() + ".lock"):
            artifact, metadata = self.paths(key, kind)
            if artifact.stat().st_nlink != 1:
                raise DomainError("ENGINE_CACHE_UNSAFE", "Compiled artifacts cannot be hardlinked")
            size, digest = sha256_file(artifact)
            value = EngineMetadata(key=key, kind=kind, compiled_sha256=digest, compiled_byte_size=size)
            atomic_json(self.root, metadata.relative_to(self.root).as_posix(), value.model_dump(mode="json"))
            return value

    def lookup(self, key: EngineKey, kind: Literal["engine", "onnx"] = "engine") -> Path | None:
        artifact, metadata = self.paths(key, kind)
        if not artifact.exists() or not metadata.exists():
            return None
        try:
            value = EngineMetadata.model_validate_json(metadata.read_text("utf-8"))
            if value.key != key or value.kind != kind:
                raise ValueError("Cache identity differs")
            return verify_file(artifact, value.compiled_sha256, value.compiled_byte_size)
        except (OSError, ValueError, DomainError):
            raise DomainError("ENGINE_CACHE_INVALID", "Compiled artifact or cache metadata differs from its identity") from None
