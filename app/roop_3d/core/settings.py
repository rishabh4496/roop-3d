"""Authoritative v2 catalog. Secrets belong in environment/OS stores only."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from roop_3d.audit.isolation import AuditError, safe_path
from roop_3d.audit.project_paths import application_root
from roop_3d.core.storage import atomic_bytes, atomic_json, exclusive, owned


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True, validate_default=True,
                              hide_input_in_errors=True)


def runtime_path(value: str, root: Path | None = None) -> Path:
    root = root or application_root()
    candidate = Path(value)
    if candidate.is_absolute():
        try:
            value = candidate.relative_to(root).as_posix()
        except ValueError:
            raise ValueError("Path must stay inside the owned runtime") from None
    try:
        path = safe_path(root, value, write=True)
    except AuditError:
        raise ValueError("Path escapes the owned runtime or traverses a link") from None
    if not path.is_relative_to(root / "roop_3d/.runtime"):
        raise ValueError("Path must stay inside roop_3d/.runtime")
    return path


class Settings(Contract):
    temporal_min_cutoff: float = Field(default=.15, gt=0, le=120)
    temporal_beta: float = Field(default=.04, ge=0, le=10)
    temporal_d_cutoff: float = Field(default=1, gt=0, le=120)
    pose_hysteresis_degrees: float = Field(default=2.5, ge=0, lt=10)
    # Stage 15: Heterogeneous Intel + NVIDIA hybrid execution scheduler
    heterogeneous_split_mode: bool = False
    openvino_device: str = "AUTO:GPU,NPU,CPU"
    qsv_offload_enabled: bool = False
    auxiliary_offload_enabled: bool = True

    schema_version: Literal[2] = 2
    execution_mode: Literal["swap", "volumetric", "hybrid", "diffusion_vfx"] = "swap"
    hardware_profile: Literal["auto", "cpu", "6gb", "12gb"] = "auto"
    preferred_provider: Literal["auto", "TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"] = "auto"
    precision_policy: Literal["auto", "fp32", "fp16"] = "auto"
    swapper_model_id: Literal["inswapper_128", "inswapper_128_fp16", "hyperswap_1a_256", "hyperswap_1b_256",
        "hyperswap_1c_256", "realswap_256", "realswap_512", "simswap_256", "simswap_512_unofficial",
        "ghost_256_unet_1", "ghost_256_unet_2", "ghost_256_unet_3", "blendswap_256", "uniface_256"] = "inswapper_128"
    vram_limit_mb: int = Field(default=0, ge=0, le=131072, strict=True, description="0 uses available VRAM")
    memory_safety_reserve_mb: int = Field(default=768, ge=0, le=131072, strict=True)
    max_concurrent_jobs: int = Field(default=1, ge=1, le=8, strict=True)
    face_restoration_policy: Literal["off", "auto", "gpen", "gfpgan", "codeformer", "ultramax"] = "off"
    video_timing_policy: Literal["preserve", "cfr"] = "preserve"
    audio_policy: Literal["copy", "transcode", "drop"] = "copy"
    lip_sync_mode: Literal["off", "wav2lip"] = "off"
    model_search_paths: tuple[str, ...] = ()
    output_root: str = "roop_3d/.runtime/outputs"
    cache_root: str = "roop_3d/.runtime/cache"
    remote_processing_enabled: bool = False
    synthetic_media_notice: bool = True
    diagnostics_level: Literal["error", "warning", "info", "debug"] = "info"
    render_backend: Literal["auto", "reference", "stream"] = "auto"
    execution_threads: int = Field(default=1, ge=1, le=8, strict=True)
    stream_batch_size: int = Field(default=4, ge=1, le=8, strict=True)
    stream_ring_slots: int = Field(default=4, ge=2, le=32, strict=True)
    stream_segment_frames: int = Field(default=250, ge=1, le=10000, strict=True)
    sparse_interval: int = Field(default=25, ge=1, le=1000, strict=True)
    restorer_threads: int = Field(default=1, ge=1, le=8, strict=True)
    restorer_pool_size: int = Field(default=2, ge=1, le=32, strict=True)
    ultramax_native_bypass: bool = True
    ultramax_min_pixels: int = Field(default=320, ge=1, le=4096, strict=True)
    host: Literal["127.0.0.1"] = "127.0.0.1"
    port: int = Field(default=8030, ge=1, le=65535)
    max_body_bytes: int = Field(default=2 * 1024 * 1024, ge=1, le=64 * 1024 * 1024)
    dev_cors: bool = False

    @field_validator("port")
    @classmethod
    def reserve_vite_port(cls, value: int) -> int:
        if value == 5174:
            raise ValueError("Port 5174 is reserved for development Vite HMR; choose another API port")
        return value

    @field_validator("output_root", "cache_root")
    @classmethod
    def writable_path(cls, value: str, info: ValidationInfo) -> str:
        runtime_path(value, (info.context or {}).get("root"))
        return value

    @field_validator("model_search_paths")
    @classmethod
    def read_only_paths(cls, values: tuple[str, ...], info: ValidationInfo) -> tuple[str, ...]:
        # External weights are opened only by exact hashed registry locators.
        if len(values) > 32:
            raise ValueError("At most 32 model search roots are permitted")
        for value in values:
            runtime_path(value, (info.context or {}).get("root"))
        return values

    @classmethod
    def from_env(cls, base: "Settings | None" = None) -> "Settings":
        base = base or ConfigStore(application_root()).load() or cls()
        values = base.model_dump()
        values.update({name: os.environ[variable] for name, variable in {
            "host": "ROOP3D_HOST", "port": "ROOP3D_PORT",
            "max_body_bytes": "ROOP3D_MAX_BODY_BYTES", "dev_cors": "ROOP3D_DEV_CORS",
        }.items() if variable in os.environ})
        return cls.model_validate(values)


def migrate_config(data: dict[str, Any]) -> dict[str, Any]:
    """Only the recorded Stage 01 versionless transport catalog migrates to v2."""
    if not isinstance(data, dict):
        raise ValueError("Configuration must be an object")
    if data.get("schema_version") == 2:
        return dict(data)
    if "schema_version" not in data and set(data) <= {"host", "port", "max_body_bytes", "dev_cors"}:
        return {**data, "schema_version": 2}
    raise ValueError("Unsupported configuration schema; no implicit migration is permitted")


class ConfigStore:
    relative = "roop_3d/.runtime/config.json"

    def __init__(self, root: Path):
        self.root = root

    def load(self, *, migrate: bool = False) -> Settings | None:
        path = owned(self.root, self.relative)
        if not path.exists():
            return None
        with exclusive(self.root, self.relative + ".lock"):
            data = json.loads(path.read_text("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Configuration must be an object")
            if migrate:
                data = migrate_config(data)
            elif data.get("schema_version") != 2:
                raise ValueError("Explicit schema migration is required")
            return Settings.model_validate(data, context={"root": self.root})

    def save(self, value: Settings) -> None:
        data = value.model_dump(mode="json")
        Settings.model_validate(data, context={"root": self.root})
        with exclusive(self.root, self.relative + ".lock"):
            path = owned(self.root, self.relative)
            if path.exists():
                atomic_bytes(self.root, self.relative + ".bak", path.read_bytes())
            atomic_json(self.root, self.relative, data)
