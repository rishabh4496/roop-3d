"""Lazy, read-only ONNX sessions with observed graph contracts and bounded threads."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.models.catalog import ModelCatalog
from roop_3d.models.registry import authorize


@dataclass(frozen=True)
class RuntimeConfig:
    provider: str = "CPUExecutionProvider"
    hardware_profile: str = "cpu"
    reserve_mib: int = 768
    cpu_threads: int = 2

    def __post_init__(self) -> None:
        if self.provider not in ("CPUExecutionProvider", "CUDAExecutionProvider"):
            raise ValueError("Reference inference supports CPU or CUDA; TensorRT belongs to Stage 07")
        if self.hardware_profile not in ("cpu", "6gb", "12gb") or self.reserve_mib < 0:
            raise ValueError("Invalid hardware profile")
        if self.provider == "CUDAExecutionProvider" and self.hardware_profile == "cpu":
            raise ValueError("CUDA requires an explicit 6gb or 12gb profile")
        if not 1 <= self.cpu_threads <= 8:
            raise ValueError("CPU thread budget must be within 1..8")

    @cached_property
    def gpu_budget_bytes(self) -> int:
        from roop_3d.core.hardware import probe_hardware
        from roop_3d.core.settings import Settings
        hardware = probe_hardware(Settings(hardware_profile=self.hardware_profile,
            preferred_provider="CUDAExecutionProvider", memory_safety_reserve_mb=self.reserve_mib))
        if not hardware.gpus or hardware.gpus[0].effective_vram_budget_mib <= 0:
            raise DomainError("VRAM_BUDGET_UNAVAILABLE", "No verified free CUDA budget is available", 503)
        return hardware.gpus[0].effective_vram_budget_mib * 1048576


def tensor(value: np.ndarray, shape: tuple[int, ...], *, dtype: np.dtype = np.dtype("float32")) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
        raise DomainError("INFERENCE_TENSOR_INVALID", "Tensor shape, dtype or finite-value contract failed", 422)
    return np.ascontiguousarray(value)


def image_bgr(value: np.ndarray) -> np.ndarray:
    if (not isinstance(value, np.ndarray) or value.dtype != np.uint8 or value.ndim != 3
            or value.shape[2] != 3 or not 0 < value.shape[0] * value.shape[1] <= 16777216):
        raise DomainError("INFERENCE_IMAGE_INVALID", "Expected a bounded uint8 HWC BGR frame", 422)
    return value


def record_named(catalog: ModelCatalog, name: str) -> dict[str, Any]:
    for record in catalog.records():
        if record["name"] == name:
            return record
    raise DomainError("MODEL_NOT_INSTALLED", "Required verified local artifact is unavailable", 503)


def cuda_fallback_reason(message: str) -> str | None:
    if "Only 4-D tensor is supported" in message:
        return "CUDA GridSample cannot execute the verified 5-D reference graph; CPU reference selected"
    if any(fragment in message for fragment in (
            "BFCArena::AllocateRawInternal", "cudaErrorMemoryAllocation", "CUDA failure 2:",
            "CUDNN_STATUS_ALLOC_FAILED", "CUDA out of memory")):
        return "CUDA allocation exceeded the available model budget; CPU reference selected"
    return None


class OrtModel:
    """Hash at load, stat-guard every call, and rehash at explicit verification.

    Sessions contain no per-frame history. No optimization files are written.
    CUDA arena budgets share one profile/free-memory snapshot. The allocation
    fractions for this reference chain sum to at most one, including optional lip.
    """

    def __init__(self, catalog: ModelCatalog, name: str, config: RuntimeConfig = RuntimeConfig()):
        import onnxruntime as ort

        self.catalog, self.record = catalog, record_named(catalog, name)
        authorize(catalog.registry.get(self.record["model_id"]), action="activate")
        if self.record["graph"].get("external_data"):
            raise DomainError("MODEL_GRAPH_UNSUPPORTED", "External tensor references are not supported by this adapter", 422)
        self.path = catalog._path(self.record)
        self._identity = self._stat()
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = config.cpu_threads, 1
        options.enable_mem_pattern = False
        options.log_severity_level = 3
        self.executions = 0
        self.provider = self.requested_provider = config.provider
        self.fallback_reason: str | None = None
        self._options = options
        self.session = None
        providers: list = ["CPUExecutionProvider"]
        if config.provider == "CUDAExecutionProvider":
            if config.provider not in ort.get_available_providers():
                raise DomainError("PROVIDER_UNAVAILABLE", "Requested CUDA provider is unavailable", 503)
            ort.preload_dlls()
            divisor = {"inswapper_128.onnx": 4, "liveportrait_warping_spade.onnx": 2,
                       "liveportrait_motion_extractor.onnx": 8, "liveportrait_landmark.onnx": 16,
                       "2d106det.onnx": 32, "liveportrait_appearance_feature_extractor.onnx": 64}.get(name, 192)
            providers = [(config.provider, {"gpu_mem_limit": config.gpu_budget_bytes // divisor,
                                           "arena_extend_strategy": "kSameAsRequested"}), "CPUExecutionProvider"]
        retry_reason = None
        try:
            self.session = ort.InferenceSession(str(self.path), sess_options=options, providers=providers)
        except Exception as error:
            retry_reason = cuda_fallback_reason(str(error)) if config.provider == "CUDAExecutionProvider" else None
            if retry_reason is None:
                raise DomainError("MODEL_SESSION_FAILED", "Verified model could not initialize on the requested provider", 503) from None
        if retry_reason is not None:
            self._switch_to_cpu(retry_reason)
        if self.session.get_providers()[0] != self.provider:
            self.session = None
            raise DomainError("PROVIDER_UNAVAILABLE", "Runtime did not activate the requested provider", 503)
        catalog.verify(self.record)
        self._check_graph()

    def _switch_to_cpu(self, reason: str) -> None:
        import onnxruntime as ort
        self.session = None
        self.catalog.verify(self.record)
        try:
            self.session = ort.InferenceSession(str(self.path), sess_options=self._options, providers=["CPUExecutionProvider"])
        except Exception:
            raise DomainError("MODEL_SESSION_FAILED", "CPU reference session could not initialize", 503) from None
        self.provider = "CPUExecutionProvider"
        self.fallback_reason = reason
        self._check_graph()

    def _stat(self) -> tuple[int, int, int]:
        stat = self.path.stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ino

    def _check_graph(self) -> None:
        for kind, actual in (("inputs", self.session.get_inputs()), ("outputs", self.session.get_outputs())):
            expected = self.record["graph"][kind]
            if len(actual) != len(expected):
                raise DomainError("MODEL_GRAPH_DRIFT", "Runtime graph differs from verified graph", 503)
            for node, declared in zip(actual, expected):
                if (node.name != declared["name"] or len(node.shape) != len(declared["shape"])
                        or any(isinstance(a, int) != isinstance(b, int) or isinstance(a, int) and a != b
                               for a, b in zip(node.shape, declared["shape"]))):
                    raise DomainError("MODEL_GRAPH_DRIFT", "Runtime tensor layout differs from verified graph", 503)
                expected_type = {"FLOAT": "tensor(float)", "FLOAT16": "tensor(float16)"}.get(declared["dtype"])
                if node.type != expected_type:
                    raise DomainError("MODEL_GRAPH_DRIFT", "Runtime tensor dtype differs from verified graph", 503)

    def run(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.session is None:
            raise DomainError("SESSION_RELEASED", "Inference session has been released", 409)
        try:
            current_identity = self._stat()
        except OSError:
            raise DomainError("MODEL_CHANGED", "Model file is no longer available", 409) from None
        if current_identity != self._identity:
            self.catalog.verify(self.record)
            raise DomainError("MODEL_CHANGED", "Model file changed after session preparation", 409)
        nodes = self.session.get_inputs()
        if set(values) != {node.name for node in nodes}:
            raise DomainError("INFERENCE_INPUT_NAMES", "Input names do not match the verified graph", 422)
        for node in nodes:
            value = values[node.name]
            shape = tuple(d if isinstance(d, int) else 1 for d in node.shape)
            tensor(value, shape, dtype=np.dtype("float16" if node.type == "tensor(float16)" else "float32"))
        retry_reason = None
        try:
            arrays = self.session.run(None, values)
        except Exception as error:
            retry_reason = cuda_fallback_reason(str(error)) if self.provider == "CUDAExecutionProvider" else None
            if retry_reason is None:
                raise DomainError("MODEL_INFERENCE_FAILED", "Model execution failed", 503) from None
        # Exit the exception handler first so its traceback cannot retain the
        # failed CUDA session/arena while the reference session is constructed.
        if retry_reason is not None:
            self._switch_to_cpu(retry_reason)
            try:
                arrays = self.session.run(None, values)
            except Exception:
                raise DomainError("MODEL_INFERENCE_FAILED", "CPU reference execution failed", 503) from None
        outputs = {}
        for node, value in zip(self.session.get_outputs(), arrays):
            shape = tuple(d if isinstance(d, int) else 1 for d in node.shape)
            outputs[node.name] = tensor(value, shape, dtype=np.dtype("float16" if node.type == "tensor(float16)" else "float32"))
        self.executions += 1
        return outputs

    def release(self) -> None:
        self.session = None

    def capability(self) -> dict:
        return {"model_id": self.record["model_id"], "sha256": self.record["sha256"],
                "artifact_state": "VERIFIED_LOCAL", "inference_state": "VERIFIED_LOCAL" if self.executions else "UNKNOWN",
                "provider": self.provider, "executions": self.executions,
                "requested_provider": self.requested_provider, "fallback_reason": self.fallback_reason,
                "license_state": self.record["license_state"], "commercial_permission": self.record["commercial_permission"]}
