"""Typed optional diffusion contract with evidence and memory gates.

Importing this module never imports PyTorch, Diffusers, or model weights. A
backend only resolves exact VERIFIED_LOCAL artifacts after an explicit load.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from importlib import import_module, util
from pathlib import Path
from typing import Any, Callable, Literal

from roop_3d.core.errors import DomainError
from roop_3d.models.catalog import ModelCatalog
from roop_3d.models.registry import ModelRecord

from .identity_adapter import ModelFamily, validate_adapter


class BackendState(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    UNSUPPORTED_VRAM = "UNSUPPORTED_VRAM"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class MemoryPlan:
    profile: Literal["cpu", "6gb", "12gb", "auto"]
    available_vram_bytes: int | None
    required_vram_bytes: int | None
    state: BackendState
    model_cpu_offload: bool = False
    vae_slicing: bool = False
    attention_slicing: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class BackendCapability:
    family: ModelFamily
    state: BackendState
    required_vram_tier: str
    supported_resolutions: tuple[tuple[int, int], ...]
    required_base_model_ids: tuple[str, ...]
    required_adapter_ids: tuple[str, ...]
    artifact_states: tuple[tuple[str, str], ...]
    license_state: str
    diffusers_state: str
    memory_plan: MemoryPlan
    reason: str | None = None


def check_memory_budget(available_vram_bytes: int | None,
                        required_vram_bytes: int | None) -> tuple[BackendState, str | None]:
    if available_vram_bytes is not None and available_vram_bytes < 0:
        raise ValueError("available_vram_bytes cannot be negative")
    if required_vram_bytes is not None and required_vram_bytes <= 0:
        raise ValueError("required_vram_bytes must be positive")
    if available_vram_bytes is None or required_vram_bytes is None:
        return BackendState.UNKNOWN, "A verified available and required VRAM budget is needed"
    if available_vram_bytes < required_vram_bytes:
        return BackendState.UNSUPPORTED_VRAM, "The verified free VRAM budget is below the registered model requirement"
    return BackendState.AVAILABLE, None


def configure_sdxl_memory(pipeline: Any, profile: str) -> MemoryPlan:
    """Apply the 6 GB policy before any pipeline is moved to a CUDA device."""
    if profile not in {"cpu", "6gb", "12gb", "auto"}:
        raise ValueError("Unknown hardware profile")
    low_vram = profile == "6gb"
    if low_vram:
        for method_name in ("enable_model_cpu_offload", "enable_vae_slicing", "enable_attention_slicing"):
            method = getattr(pipeline, method_name, None)
            if not callable(method):
                return MemoryPlan(profile, None, None, BackendState.UNAVAILABLE,
                                  reason=f"Pipeline does not support required 6 GB memory method: {method_name}")
            method()
    return MemoryPlan(profile, None, None, BackendState.UNKNOWN,
                      model_cpu_offload=low_vram, vae_slicing=low_vram, attention_slicing=low_vram,
                      reason="No verified per-model VRAM requirement is registered")


PipelineLoader = Callable[[Any, tuple[Path, ...]], Any]


class DiffusionBackend(ABC):
    """Family-specific policy shell around an explicitly supplied local loader."""

    family: ModelFamily
    base_model_ids: frozenset[str]
    adapter_model_ids: frozenset[str]

    def __init__(self, *, catalog: ModelCatalog | None = None, profile: str = "auto",
                 base_model_id: str, adapter_id: str, available_vram_bytes: int | None = None,
                 quantization: str | None = None):
        if profile not in {"cpu", "6gb", "12gb", "auto"}:
            raise ValueError("Unknown hardware profile")
        if base_model_id not in self.base_model_ids:
            raise DomainError("DIFFUSION_FAMILY_MISMATCH",
                              f"Base model {base_model_id} is not a {self.family} model", 422)
        # Run family validation before catalog inspection, imports, or file access.
        validate_adapter(self.family, adapter_id)
        if adapter_id not in self.adapter_model_ids:
            raise DomainError("DIFFUSION_FAMILY_MISMATCH",
                              f"Adapter {adapter_id} is not supported by the {self.family} backend", 422)
        if available_vram_bytes is not None and available_vram_bytes < 0:
            raise ValueError("available_vram_bytes cannot be negative")
        self.catalog = catalog or ModelCatalog()
        self.profile = profile
        self.base_model_id = base_model_id
        self.adapter_id = adapter_id
        self.available_vram_bytes = available_vram_bytes
        self.quantization = quantization
        self._pipeline: Any | None = None

    def _record(self, model_id: str) -> ModelRecord | None:
        try:
            return self.catalog.registry.get(model_id)
        except DomainError as error:
            if error.code == "MODEL_NOT_FOUND":
                return None
            raise

    @staticmethod
    def _artifact_state(record: ModelRecord | None) -> str:
        if record is None:
            return "UNKNOWN"
        if record.status == "candidate":
            return "CANDIDATE"
        if record.status == "blocked":
            return "BLOCKED"
        if record.status == "verified_local" and record.provenance_state == "VERIFIED_LOCAL":
            return "VERIFIED_LOCAL"
        if record.status == "verified_remote" and record.provenance_state == "VERIFIED_REMOTE":
            return "VERIFIED_REMOTE"
        return "UNKNOWN"

    def _family_artifact_state(self, record: ModelRecord | None, *, role: str) -> str:
        state = self._artifact_state(record)
        if state != "VERIFIED_LOCAL" or record is None:
            return state
        declared_family = record.input_output_contract.get("family")
        if (record.role != role or (record.family_hint.casefold() != self.family
                                    and declared_family != self.family)):
            return "UNKNOWN"
        return state

    def _registered_required_vram(self, records: tuple[ModelRecord | None, ...]) -> int | None:
        values: list[int] = []
        for record in records:
            if record is None:
                return None
            raw = record.input_output_contract.get("required_vram_bytes")
            if type(raw) is not int or raw <= 0:
                return None
            values.append(raw)
        return max(values, default=None)

    @abstractmethod
    def _family_memory_plan(self, required_vram_bytes: int | None) -> MemoryPlan:
        """Apply family-specific VRAM policy without loading model dependencies."""

    def capability(self) -> BackendCapability:
        records = (self._record(self.base_model_id), self._record(self.adapter_id))
        states = (self._family_artifact_state(records[0], role="diffusion_base"),
                  self._family_artifact_state(records[1], role="diffusion_adapter"))
        states = tuple(zip((self.base_model_id, self.adapter_id), states))
        required_bytes = self._registered_required_vram(records)
        memory = self._family_memory_plan(required_bytes)
        diffusers_state = "VERIFIED_LOCAL" if util.find_spec("diffusers") else "UNKNOWN"
        if memory.state == BackendState.UNSUPPORTED_VRAM:
            state, reason = memory.state, memory.reason
        elif any(value != "VERIFIED_LOCAL" for _, value in states):
            state, reason = BackendState.UNAVAILABLE, "Required base or identity adapter is not VERIFIED_LOCAL"
        elif diffusers_state != "VERIFIED_LOCAL":
            state, reason = BackendState.DEPENDENCY_UNAVAILABLE, "The optional diffusers package is not installed"
        elif memory.state == BackendState.UNAVAILABLE:
            state, reason = memory.state, memory.reason
        elif memory.state == BackendState.UNSUPPORTED_VRAM:
            state, reason = memory.state, memory.reason
        elif memory.state == BackendState.UNKNOWN:
            state, reason = BackendState.UNAVAILABLE, memory.reason
        else:
            state, reason = BackendState.AVAILABLE, None
        return BackendCapability(
            family=self.family, state=state, required_vram_tier="UNKNOWN",
            supported_resolutions=(), required_base_model_ids=(self.base_model_id,),
            required_adapter_ids=(self.adapter_id,), artifact_states=states,
            # A model checksum does not establish license permission.
            license_state="UNKNOWN", diffusers_state=diffusers_state,
            memory_plan=memory, reason=reason,
        )

    def _verified_local_paths(self) -> tuple[Path, ...]:
        capability = self.capability()
        if capability.state != BackendState.AVAILABLE:
            code = ("DIFFUSION_VRAM_UNSUPPORTED" if capability.state == BackendState.UNSUPPORTED_VRAM
                    else "DIFFUSION_UNAVAILABLE")
            raise DomainError(code, capability.reason or "Diffusion backend is unavailable", 503)
        result = []
        for model_id in (self.base_model_id, self.adapter_id):
            record = self.catalog.registry.get(model_id)
            result.append(self.catalog.resolve_for_inference(record.model_id))
        return tuple(result)

    def load(self, loader: PipelineLoader) -> Any:
        """Load only verified local files; ``loader`` owns the model-specific API contract."""
        if not callable(loader):
            raise DomainError("DIFFUSION_LOADER_INVALID", "A family-specific local pipeline loader is required", 422)
        paths = self._verified_local_paths()
        try:
            diffusers = import_module("diffusers")
        except ImportError:
            raise DomainError("DIFFUSION_DEPENDENCY_UNAVAILABLE", "Install the optional diffusion extra to load this backend", 503) from None
        try:
            pipeline = loader(diffusers, paths)
        except Exception:
            raise DomainError("DIFFUSION_PIPELINE_LOAD_FAILED", "The registered local diffusion pipeline could not be loaded", 503) from None
        if pipeline is None:
            raise DomainError("DIFFUSION_LOADER_INVALID", "Pipeline loader returned no pipeline", 503)
        self.validate_pipeline(pipeline)
        plan = self._family_memory_plan(self._registered_required_vram(
            (self._record(self.base_model_id), self._record(self.adapter_id))))
        if plan.model_cpu_offload:
            plan = self.configure_pipeline_memory(pipeline)
            if plan.state != BackendState.AVAILABLE:
                raise DomainError("DIFFUSION_MEMORY_POLICY_UNAVAILABLE", plan.reason or "Memory policy unavailable", 503)
        self._pipeline = pipeline
        return pipeline

    @abstractmethod
    def configure_pipeline_memory(self, pipeline: Any) -> MemoryPlan:
        """Apply family-appropriate memory options to an explicitly loaded pipeline."""

    @abstractmethod
    def validate_pipeline(self, pipeline: Any) -> None:
        """Reject a loader result whose declared pipeline class belongs to another family."""

    @abstractmethod
    def generate(self, **kwargs: Any) -> Any:
        """Run family-specific generation after the caller has loaded a verified pipeline."""

    def require_loaded_pipeline(self) -> Any:
        if self._pipeline is None:
            raise DomainError("DIFFUSION_PIPELINE_NOT_LOADED", "Load a verified local pipeline before generation", 409)
        return self._pipeline

    @staticmethod
    def deterministic_generation_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Convert an explicit job seed to a lazily created CPU torch generator."""
        values = dict(kwargs)
        seed = values.pop("seed", None)
        if type(seed) is not int or not 0 <= seed <= (2**63 - 1):
            raise DomainError("DIFFUSION_SEED_REQUIRED", "Each diffusion job requires an unsigned 63-bit seed", 422)
        if values.get("generator") is None:
            try:
                torch = import_module("torch")
            except ImportError:
                raise DomainError("DIFFUSION_DEPENDENCY_UNAVAILABLE", "PyTorch is required for diffusion generation", 503) from None
            values["generator"] = torch.Generator(device="cpu").manual_seed(seed)
        return values
