"""Lazy, read-only ONNX sessions with observed graph contracts and bounded threads."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.models.catalog import ModelCatalog
from roop_3d.models.registry import authorize
from roop_3d.engine.resources import guarded, register

# Shared low-level session construction is also available to GPU graph callers.
# This additive adapter keeps the verified catalog as the only model path source.
from roop_3d.core.ort_session import SessionManager, SessionInfo


@dataclass(frozen=True)
class RuntimeConfig:
    provider: str = "CPUExecutionProvider"
    hardware_profile: str = "cpu"
    reserve_mib: int = 768
    cpu_threads: int = 2
    model_factory: Any = field(default=None, compare=False, repr=False)
    graph_optimization: str = "all"

    def __post_init__(self) -> None:
        if self.provider not in ("CPUExecutionProvider", "CUDAExecutionProvider"):
            raise ValueError("Reference inference supports CPU or CUDA; TensorRT belongs to Stage 07")
        if self.hardware_profile not in ("cpu", "6gb", "12gb") or self.reserve_mib < 0:
            raise ValueError("Invalid hardware profile")
        if self.provider == "CUDAExecutionProvider" and self.hardware_profile == "cpu":
            raise ValueError("CUDA requires an explicit 6gb or 12gb profile")
        if not 1 <= self.cpu_threads <= 8:
            raise ValueError("CPU thread budget must be within 1..8")
        if self.graph_optimization not in ("all", "extended"):
            raise ValueError("Unsupported graph optimization policy")

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
    return np.asarray(value) if value.ndim == 0 else np.ascontiguousarray(value)


def contract_array(value: np.ndarray, declared, *, dtype=np.dtype("float32"), batch_one: bool = False) -> np.ndarray:
    if (not isinstance(value, np.ndarray) or value.ndim != len(declared) or value.nbytes > 64 * 1048576
            or any(d <= 0 for d in value.shape) or batch_one and value.ndim > 0 and value.shape[0] != 1):
        raise DomainError("INFERENCE_TENSOR_INVALID", "Tensor rank, batch or memory budget failed", 422)
    shape = tuple(d if isinstance(d, int) else value.shape[i] for i, d in enumerate(declared))
    return tensor(value, shape, dtype=dtype)


def image_bgr(value: np.ndarray) -> np.ndarray:
    if (not isinstance(value, np.ndarray) or value.dtype != np.uint8 or value.ndim != 3
            or value.shape[2] != 3 or not 0 < value.shape[0] * value.shape[1] <= 16777216):
        raise DomainError("INFERENCE_IMAGE_INVALID", "Expected a bounded uint8 HWC BGR frame", 422)
    return value


def record_named(catalog: ModelCatalog, name: str) -> dict[str, Any]:
    # Explicit family pins take precedence over registry order after an update.
    for registered in catalog.registry.records():
        selected = catalog.select_version(registered)
        if selected.model_id != registered.model_id and selected.name == name:
            catalog.resolve_for_inference(selected.model_id)
            return catalog._legacy(selected)
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

    def __new__(cls, catalog: ModelCatalog, name: str, config: RuntimeConfig = RuntimeConfig()):
        if config.model_factory is not None and config.model_factory is not cls:
            return config.model_factory(catalog, name, config)
        return super().__new__(cls)

    @guarded
    def __init__(self, catalog: ModelCatalog, name: str, config: RuntimeConfig = RuntimeConfig()):
        import onnxruntime as ort

        self.catalog, self.record = catalog, record_named(catalog, name)
        authorize(catalog.registry.get(self.record["model_id"]), action="activate")
        if self.record["graph"].get("external_data"):
            raise DomainError("MODEL_GRAPH_UNSUPPORTED", "External tensor references are not supported by this adapter", 422)
        self.path = catalog.resolve_for_inference(self.record["model_id"])
        if config.provider == "CUDAExecutionProvider" and name == "liveportrait_warping_spade.onnx":
            try:
                from roop_3d.models.stage16 import prepare_generator
                self.path = prepare_generator(catalog)[0]
            except Exception:
                pass
        self._identity = self._stat()
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = config.cpu_threads, 1
        options.enable_mem_pattern = False
        options.log_severity_level = 3
        options.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            if config.graph_optimization == "extended" else ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
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
            divisor = {
                "inswapper_128.onnx": 4, "inswapper_128.batch.onnx": 4,
                "inswapper_128_dynamic.onnx": 4, "gpen_bfr_512.onnx": 4,
                "GPEN-BFR-512.onnx": 4, "CodeFormerv0.1.onnx": 3, "codeformer.fp16.onnx": 8,
                "scrfd_2.5g_kps.onnx": 32, "scrfd_10g_bnkps.onnx": 16, "det_10g.onnx": 16,
                "w600k_r50.onnx": 16, "bisenet_resnet_34.onnx": 16, "face_parser.onnx": 16,
                "liveportrait_warping_spade.onnx": 2,
                "liveportrait_motion_extractor.onnx": 8, "liveportrait_motion_extractor.batch.onnx": 8,
                "liveportrait_landmark.onnx": 16, "1k3d68.onnx": 16, "2d106det.onnx": 32, "landmark_203.onnx": 32,
                "liveportrait_appearance_feature_extractor.onnx": 32,
                "liveportrait_appearance_feature_extractor.batch.onnx": 32,
                "liveportrait_stitching.onnx": 32, "liveportrait_stitching.batch.onnx": 32,
                "liveportrait_stitching_eye.onnx": 32, "liveportrait_stitching_eye.batch.onnx": 32,
                "liveportrait_stitching_lip.onnx": 32,
            }.get(name, 32)
            arena = "kNextPowerOfTwo" if config.hardware_profile == "12gb" else "kSameAsRequested"
            cuda_opts = {
                "gpu_mem_limit": config.gpu_budget_bytes // divisor,
                "arena_extend_strategy": arena,
                "do_copy_in_default_stream": "1",
            }
            if config.hardware_profile == "12gb":
                cuda_opts["cudnn_conv_algo_search"] = "DEFAULT"
            else:
                cuda_opts["cudnn_conv_algo_search"] = "HEURISTIC"
            providers = [(config.provider, cuda_opts), "CPUExecutionProvider"]
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
        self._io_binder = None
        if self.provider == "CUDAExecutionProvider":
            try:
                from roop_3d.engine.adaptive_runtime import CudaIoBinder
                self._io_binder = CudaIoBinder(self.session)
            except Exception:
                self._io_binder = None
        register(self, "release")

    def _switch_to_cpu(self, reason: str) -> None:
        import onnxruntime as ort
        self.session = None
        self._io_binder = None
        self.path = self.catalog._path(self.record)
        self._identity = self._stat()
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
                        or any(isinstance(b, int) and (not isinstance(a, int) or a != b)
                               or isinstance(a, int) and a <= 0
                               for a, b in zip(node.shape, declared["shape"]))):
                    raise DomainError("MODEL_GRAPH_DRIFT", "Runtime tensor layout differs from verified graph", 503)
                expected_type = {"FLOAT": "tensor(float)", "FLOAT16": "tensor(float16)", "DOUBLE": "tensor(double)"}.get(declared["dtype"])
                if node.type != expected_type:
                    raise DomainError("MODEL_GRAPH_DRIFT", "Runtime tensor dtype differs from verified graph", 503)

    @guarded
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
            contract_array(value, node.shape, batch_one=True,
                           dtype=np.dtype({"tensor(float16)": "float16", "tensor(double)": "float64"}.get(node.type, "float32")))
        retry_reason = None
        try:
            if getattr(self, "_io_binder", None) is not None:
                out_map = self._io_binder.bind_and_run(values)
                arrays = [out_map[node.name] for node in self.session.get_outputs()]
            else:
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
            outputs[node.name] = contract_array(value, node.shape,
                dtype=np.dtype({"tensor(float16)": "float16", "tensor(double)": "float64"}.get(node.type, "float32")))
        self.executions += 1
        return outputs

    @guarded
    def release(self) -> None:
        self.session = None
        self._io_binder = None

    def capability(self) -> dict:
        return {"model_id": self.record["model_id"], "sha256": self.record["sha256"],
                "artifact_state": "VERIFIED_LOCAL", "inference_state": "VERIFIED_LOCAL" if self.executions else "UNKNOWN",
                "provider": self.provider, "executions": self.executions,
                "requested_provider": self.requested_provider, "fallback_reason": self.fallback_reason,
                "license_state": self.record["license_state"], "commercial_permission": self.record["commercial_permission"]}


class SessionFacade:
    """Keep legacy detector/recognizer preprocessing unchanged with a factory model."""
    def __init__(self, model):
        self.model = model

    def get_inputs(self):
        nodes = {node.name: node for node in self.model.session.get_inputs()}
        return [nodes[node["name"]] for node in self.model.record["graph"]["inputs"]]

    def get_outputs(self):
        nodes = {node.name: node for node in self.model.session.get_outputs()}
        return [nodes[node["name"]] for node in self.model.record["graph"]["outputs"]]

    def get_providers(self):
        return self.model.session.get_providers()

    def run(self, output_names, values):
        output = self.model.run(values)
        names = output_names if output_names is not None else [node.name for node in self.get_outputs()]
        return [output[name] for name in names]


def session_factory(catalog: ModelCatalog, config: RuntimeConfig):
    factory = config.model_factory
    if factory is None and config.provider == "CUDAExecutionProvider":
        factory = OrtModel
    if factory is None:
        return None
    return lambda record: SessionFacade(factory(catalog, record["name"], config))


def adaptive_session_factory(catalog: ModelCatalog, config: RuntimeConfig):
    """Session factory powered by AdaptiveModelPool for warm GPU residency."""
    from roop_3d.engine.adaptive_runtime import AdaptiveModelPool
    pool = AdaptiveModelPool.get_instance()
    return lambda record: SessionFacade(pool.acquire(catalog, record["name"], config))


def managed_session_factory(catalog: ModelCatalog, *, manager: SessionManager | None = None,
                            providers: tuple[str, ...] = ("TensorrtExecutionProvider",
                                                           "CUDAExecutionProvider",
                                                           "CPUExecutionProvider")):
    """Build TensorRT/CUDA/CPU sessions for hash-verified catalog entries.

    Unlike the strict canary builder this is a provider fallback adapter; it still
    resolves paths through ModelCatalog and checks the runtime graph against its
    registered input/output contract before returning the session.
    """
    manager = manager or SessionManager()

    def create(record: dict[str, Any]) -> SessionInfo:
        path = catalog.resolve_for_inference(record["model_id"])
        info = manager.create(path, model_name=record["name"], providers=providers)
        for kind, actual in (("inputs", info.session.get_inputs()),
                             ("outputs", info.session.get_outputs())):
            expected = record["graph"][kind]
            if len(actual) != len(expected) or any(
                node.name != contract["name"] or len(node.shape) != len(contract["shape"])
                or node.type != {"FLOAT": "tensor(float)", "FLOAT16": "tensor(float16)",
                                 "DOUBLE": "tensor(double)"}.get(contract["dtype"])
                for node, contract in zip(actual, expected)
            ):
                raise DomainError("MODEL_GRAPH_DRIFT", "Runtime graph differs from verified graph", 503)
        return info

    return create
