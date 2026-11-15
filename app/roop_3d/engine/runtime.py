"""Lazy, read-only ONNX sessions with observed graph contracts and bounded threads."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
import logging
import time
from typing import Any

import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.models.catalog import ModelCatalog
from roop_3d.models.registry import authorize
from roop_3d.engine.resources import guarded, inference, register

# Shared low-level session construction is also available to GPU graph callers.
# This additive adapter keeps the verified catalog as the only model path source.
from roop_3d.core.ort_session import (
    CPU, CUDA, DML, TRT, DEFAULT_PROVIDERS, GpuIoBinding, SessionManager, SessionInfo, provider_chain,
    record_session_failure, record_session_execution, record_session_release,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeConfig:
    provider: str = "CPUExecutionProvider"
    hardware_profile: str = "cpu"
    reserve_mib: int = 768
    cpu_threads: int = 2
    model_factory: Any = field(default=None, compare=False, repr=False)
    graph_optimization: str = "all"
    precision_policy: str = 'auto'
    detector_model_id: str = 'scrfd'

    def __post_init__(self) -> None:
        if self.provider not in ("auto", CPU, CUDA, DML, TRT):
            raise ValueError("Unsupported ONNX Runtime provider")
        if self.hardware_profile not in ("cpu", "6gb", "12gb") or self.reserve_mib < 0:
            raise ValueError("Invalid hardware profile")
        if self.provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider") and self.hardware_profile == "cpu":
            raise ValueError("GPU providers require an explicit 6gb or 12gb profile")
        if not 1 <= self.cpu_threads <= 8:
            raise ValueError("CPU thread budget must be within 1..8")
        if self.graph_optimization not in ("all", "extended"):
            raise ValueError("Unsupported graph optimization policy")
        if self.precision_policy not in ('auto', 'fp16', 'fp32') or self.detector_model_id not in ('scrfd', 'retinaface', 'yoloface'):
            raise ValueError('Unsupported precision or detector selection')

    @cached_property
    def gpu_budget_bytes(self) -> int:
        # Every preview builds a fresh RuntimeConfig, and the probe shells out to nvidia-smi three times (~3 s on
        # Windows).  The value only sets a session memory limit, so a 20 s old reading is as good as a new one.
        key = (self.hardware_profile, self.reserve_mib)
        cached = _BUDGET_CACHE.get(key)
        if cached is not None and time.monotonic() - cached[0] < 20.0:
            return cached[1]
        from roop_3d.core.hardware import probe_hardware
        from roop_3d.core.settings import Settings
        hardware = probe_hardware(Settings(hardware_profile=self.hardware_profile,
            preferred_provider="CUDAExecutionProvider", memory_safety_reserve_mb=self.reserve_mib))
        if not hardware.gpus or hardware.gpus[0].effective_vram_budget_mib <= 0:
            raise DomainError("VRAM_BUDGET_UNAVAILABLE", "No verified free CUDA budget is available", 503)
        value = hardware.gpus[0].effective_vram_budget_mib * 1048576
        _BUDGET_CACHE[key] = (time.monotonic(), value)
        return value


_BUDGET_CACHE: dict[tuple, tuple[float, int]] = {}


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
    record, _ = catalog.resolve_named_artifact(name)
    return catalog._legacy(record)




class OrtModel:
    """Hash at load, stat-guard every call, and rehash at explicit verification.

    Sessions contain no per-frame history. Derived precision/TRT graphs are written
    only to the owned cache; source artifacts remain read-only.
    """

    def __new__(cls, catalog: ModelCatalog, name: str, config: RuntimeConfig = RuntimeConfig()):
        if config.model_factory is not None and config.model_factory is not cls:
            return config.model_factory(catalog, name, config)
        return super().__new__(cls)

    @guarded
    def __init__(self, catalog: ModelCatalog, name: str, config: RuntimeConfig = RuntimeConfig()):
        try:
            self._construct(catalog, name, config)
        except Exception as error:
            record_session_failure(name, getattr(self, "path", None), error.__cause__ or error,
                                   requested_providers=provider_chain(config.provider))
            raise

    def _construct(self, catalog: ModelCatalog, name: str, config: RuntimeConfig):
        self._last_error = None
        self.catalog, self.record = catalog, record_named(catalog, name)
        self.config = config
        registered = catalog.registry.get(self.record["model_id"])
        try:
            authorize(registered, action="activate")
        except DomainError:
            # A weight the user placed in app/models is never switched off by a registry license/status gate.
            if catalog.provider.trusted_path(registered) is None:
                raise
        if self.record["graph"].get("external_data"):
            raise DomainError("MODEL_GRAPH_UNSUPPORTED", "External tensor references are not supported by this adapter", 422)
        self.path = catalog.resolve_for_inference(self.record["model_id"])
        trusted = catalog.provider.trusted_identity(self.path)
        # Local-use authorization is independent of graph identity. Exact pins keep strict tensor validation.
        self.local_trust = trusted is not None and trusted != (self.record["sha256"], self.record["size_bytes"])
        if trusted is not None:
            self.record = {**self.record, "sha256": trusted[0], "size_bytes": trusted[1]}
        self._identity = self._stat()
        self.executions = 0
        self.provider = self.requested_provider = config.provider
        self.fallback_reason: str | None = None
        self.session = None
        divisor = {
            "scrfd_2.5g_kps.onnx": 32, "scrfd_10g_bnkps.onnx": 16, "det_10g.onnx": 16,
            "w600k_r50.onnx": 16, "bisenet_resnet_34.onnx": 12, "face_parser.onnx": 12,
            "1k3d68.onnx": 12, "2d106det.onnx": 32, "landmark_203.onnx": 32,
            "xseg.onnx": 12, "xseg_3.onnx": 12, "face_occluder.onnx": 12, "occluder.onnx": 12,
        }.get(name, 4 if self.record.get("role") == "uv_enhancer" else 32)
        # The canonical 512px CodeFormer forward needs more than the old 1/32
        # arena (which could not even hold its weights). Keep a bounded share of
        # the admitted profile, including the parser/occluder working tensors.
        providers = provider_chain(config.provider)
        gpu_budget = config.gpu_budget_bytes if any(p in (CUDA, TRT) for p in providers) else 0
        self._gpu_memory_limit = gpu_budget // divisor if gpu_budget > 0 else None
        self._open_session(providers)
        catalog.verify(self.record)
        self._check_graph()
        self._io_binder = GpuIoBinding(self.session)
        # Upscalers report themselves to the active-model read-back (the detector reports from its own holder).
        self._active_key = None
        if self.record.get("role") in ("frame_upscaler", "uv_enhancer"):
            from roop_3d.engine import active_models
            entry = active_models.activate("enhancer", self.record["model_id"], self.path, self.provider,
                                           config.precision_policy, self.record.get("sha256"))
            self._active_key = entry.key
        register(self, "release")

    def _open_session(self, providers: tuple[str, ...]) -> None:
        """Every loader uses the same real-session probe and exact fallback diagnostics."""
        try:
            info = SessionManager(
                hardware_profile="12gb" if self.config.hardware_profile == "12gb" else "6gb",
                fp16=self.config.precision_policy != "fp32",
                cuda_fp16=self.config.precision_policy == "fp16",
                gpu_memory_limit=self._gpu_memory_limit,
                graph_optimization=self.config.graph_optimization,
            ).create(self.path, model_name=self.record["name"], providers=providers,
                     intra_op_threads=self.config.cpu_threads)
        except Exception as error:
            raise DomainError("MODEL_SESSION_FAILED", f"{self.path}: {type(error).__name__}: {error}", 503) from error
        # Diagnostics must not retain the native session after release/provider recovery.
        self._session_diagnostics = info.capability()
        self.session, self.provider = info.session, info.provider
        self.fallback_reason = "; ".join(info.fallback_errors) or None

    @guarded
    def _recover(self, reason: str) -> None:
        """Retry the same graph on the next supported provider, never skip its computation."""
        order = provider_chain(self.requested_provider)
        remaining = order[order.index(self.provider) + 1:]
        if not remaining:
            raise DomainError("MODEL_INFERENCE_FAILED", reason, 503)
        previous_reason = self.fallback_reason
        self.session = self._io_binder = None
        self._open_session(remaining)
        self.fallback_reason = "; ".join(item for item in (previous_reason, reason, self.fallback_reason) if item)
        from roop_3d.core.ort_session import record_session_fallback
        record_session_fallback(self.record["name"], self.path, self.fallback_reason)
        self._identity = self._stat()
        self._check_graph()
        self._io_binder = GpuIoBinding(self.session)

    def _stat(self) -> tuple[int, int, int]:
        stat = self.path.stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ino

    def _check_graph(self) -> None:
        if getattr(self, "local_trust", False):
            # A user-supplied export (for instance FP16, or dynamic instead of static axes) legitimately differs
            # from the pinned graph in dtype and extents. Names and ranks are what the adapters feed by, so a
            # mismatch there is logged and surfaces as an explicit input-name error on the first run, not as a refusal here.
            dtypes = {"tensor(float)": "FLOAT", "tensor(float16)": "FLOAT16", "tensor(double)": "DOUBLE",
                      "tensor(int64)": "INT64", "tensor(int32)": "INT32", "tensor(uint8)": "UINT8"}
            observed = dict(self.record["graph"])
            for kind, actual in (("inputs", self.session.get_inputs()), ("outputs", self.session.get_outputs())):
                expected = self.record["graph"][kind]
                if len(actual) != len(expected) or any(node.name != declared["name"] or len(node.shape) != len(declared["shape"])
                                                       for node, declared in zip(actual, expected)):
                    log.warning("Local weight %s differs from the registered %s layout; running it as supplied",
                                self.path.name, kind)
                # Adapters build their feeds from record["graph"], so it must describe the file that actually runs.
                observed[kind] = [{"name": node.name, "shape": list(node.shape), "dtype": dtypes.get(node.type, "FLOAT")}
                                  for node in actual]
            self.record = {**self.record, "graph": observed}
            return
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

    @inference
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
        # Adapters feed float32; an FP16 export is fed half precision so it runs instead of being refused.
        values = {node.name: (values[node.name].astype(np.float16) if node.type == "tensor(float16)"
                              and getattr(values[node.name], "dtype", None) == np.float32 else values[node.name])
                  for node in nodes}
        for node in nodes:
            value = values[node.name]
            contract_array(value, node.shape, batch_one=True,
                           dtype=np.dtype({"tensor(float16)": "float16", "tensor(double)": "float64"}.get(node.type, "float32")))
        while True:
            retry_reason = None
            try:
                out_map = self._io_binder.run(values)
                arrays = [out_map[node.name].numpy() for node in self.session.get_outputs()]
            except Exception as error:
                self._last_error = f"{self.provider}: {type(error).__name__}: {error}"
                record_session_execution(self.record["name"], self.path, error=error)
                if self.provider == CPU:
                    raise DomainError("MODEL_INFERENCE_FAILED", self._last_error, 503) from error
                retry_reason = self._last_error
            # Leave the traceback scope before allocating the next provider's arena.
            if retry_reason is None:
                break
            log.warning("%s failed on %s; retrying the same model: %s", self.path, self.provider, retry_reason)
            self._recover(retry_reason)
        outputs = {}
        try:
            for node, value in zip(self.session.get_outputs(), arrays):
                checked = contract_array(value, node.shape,
                    dtype=np.dtype({"tensor(float16)": "float16", "tensor(double)": "float64"}.get(node.type, "float32")))
                # Callers consume float32 regardless of the graph's storage precision.
                outputs[node.name] = checked.astype(np.float32) if checked.dtype == np.float16 else checked
        except DomainError as error:
            self._last_error = f"{self.provider}: {error.code}: {error}"
            record_session_execution(self.record["name"], self.path, error=error)
            raise
        self.executions += 1
        self._last_error = None
        record_session_execution(self.record["name"], self.path)
        return outputs

    @guarded
    def release(self) -> None:
        record_session_release(self.record["name"], self.path)
        self.session = None
        self._io_binder = None
        if getattr(self, "_active_key", None) is not None:
            from roop_3d.engine import active_models
            active_models.release("enhancer", self._active_key)
            self._active_key = None

    def capability(self) -> dict:
        diagnostics = self._session_diagnostics
        return {**diagnostics, "model_id": self.record["model_id"], "sha256": self.record["sha256"],
                "artifact_state": "VERIFIED_LOCAL", "inference_state": "VERIFIED_LOCAL" if self.executions else "UNKNOWN",
                "provider": self.provider, "executions": self.executions, "execution_verified": bool(self.executions),
                "path": str(self.path),
                "active_provider": self.provider if self.session is not None else None,
                "load_status": "loaded" if self.session is not None else "released", "error": self._last_error,
                "execution_providers": self.session.get_providers() if self.session is not None else [],
                "cpu_graph_partitions_allowed": diagnostics.get("cpu_partitions", False),
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
    if factory is None:
        factory = OrtModel
    return lambda record: SessionFacade(factory(catalog, record["name"], config))


def inspection_runtime(settings=None) -> RuntimeConfig | None:
    """Admit CUDA/DirectML inspection sessions, or reserve GPU memory and select CPU.

    Small NVIDIA cards retain the inspection memory policy; advertised providers
    are only candidates. SessionManager must initialize the actual model before
    diagnostics report a loaded provider. ``ROOP3D_INSPECTION_PROVIDER`` accepts cpu/cuda/dml.
    """
    import os
    override = os.environ.get("ROOP3D_INSPECTION_PROVIDER", "").strip().casefold()
    if override in ("cpu", "off", "0", "false"):
        return None
    from roop_3d.core.settings import Settings
    settings = settings or Settings()
    if settings.hardware_profile == "cpu" or settings.preferred_provider == "CPUExecutionProvider":
        return None
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        if override == "dml" or settings.preferred_provider == DML or CUDA not in available:
            return (RuntimeConfig(provider=DML, reserve_mib=settings.memory_safety_reserve_mb)
                    if DML in available else None)
        from roop_3d.core.hardware import MIN_12GB_PROFILE_VRAM_MIB, probe_hardware, resolve_hardware_profile
        hardware = probe_hardware(settings)
    except Exception as error:
        log.warning("Inspection hardware admission probe failed; CPU policy selected: %s: %s", type(error).__name__, error)
        return None
    if not hardware.gpus:
        return None
    gpu = hardware.gpus[0]
    if override != "cuda" and (gpu.total_vram_mib < MIN_12GB_PROFILE_VRAM_MIB or gpu.free_vram_mib < 2048):
        return None
    profile = resolve_hardware_profile(gpu.total_vram_mib, gpu.free_vram_mib, settings.hardware_profile)
    if profile == "cpu":
        return None
    return RuntimeConfig(provider="CUDAExecutionProvider", hardware_profile=profile,
                         reserve_mib=settings.memory_safety_reserve_mb)


def inspection_session_factory(catalog: ModelCatalog, settings_provider=None, config: RuntimeConfig | None = None):
    """Build observed model sessions on the admitted GPU, or CPU when inspection policy reserves GPU memory.

    CPU uses the same hash, graph and diagnostic path. Model/graph failures are not hidden by a second loader.
    """
    def factory(record):
        resolved = (config if config is not None
                    else inspection_runtime(settings_provider() if settings_provider is not None else None))
        if resolved is None:
            resolved = RuntimeConfig()
        return SessionFacade(OrtModel(catalog, record["name"], resolved))
    return factory


def build_with_cpu_fallback(factory, record, name: str):
    """Use the selected factory; its provider chain owns fallback and preserves graph failures."""
    del name
    return factory(record) if factory is not None else None


def adaptive_session_factory(catalog: ModelCatalog, config: RuntimeConfig):
    """Session factory powered by AdaptiveModelPool for warm GPU residency."""
    from roop_3d.engine.adaptive_runtime import AdaptiveModelPool
    pool = AdaptiveModelPool.get_instance()
    return lambda record: SessionFacade(pool.acquire(catalog, record["name"], config))


def managed_session_factory(catalog: ModelCatalog, *, manager: SessionManager | None = None,
                            providers: tuple[str, ...] = DEFAULT_PROVIDERS):
    """Build CUDA/DirectML/CPU sessions for hash-verified catalog entries.

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
