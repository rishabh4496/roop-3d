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
from roop_3d.core.ort_session import GpuIoBinding, SessionManager, SessionInfo, _cuda_fp16_model_path

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
        if self.provider not in ("CPUExecutionProvider", "CUDAExecutionProvider", "TensorrtExecutionProvider"):
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
        try:
            self._construct(catalog, name, config)
        except Exception as error:
            # An enhancer that cannot be built is reported with the backend's exact reason, so the UI can say why the
            # old model is still in use instead of silently keeping it.
            record = getattr(self, "record", None)
            if isinstance(record, dict) and record.get("role") == "face_restorer":
                from roop_3d.engine import active_models
                active_models.report_failure("enhancer", record.get("model_id") or name, error)
            raise

    def _construct(self, catalog: ModelCatalog, name: str, config: RuntimeConfig):
        import onnxruntime as ort

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
        # Auto-trusted local weights (FP16 / custom exports) carry their own digest and relaxed tensor checks.
        self.local_trust = trusted is not None
        if trusted is not None:
            self.record = {**self.record, "sha256": trusted[0], "size_bytes": trusted[1]}
        if config.provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider") and name == "liveportrait_warping_spade.onnx":
            try:
                from roop_3d.models.stage16 import prepare_generator
                prepared = (prepare_generator(catalog, batch_one=True) if config.provider == 'TensorrtExecutionProvider'
                            else prepare_generator(catalog))
                self.path = prepared[0]
            except Exception:
                pass
        if config.provider == "CUDAExecutionProvider" and config.precision_policy == "fp16":
            fp16_cache = catalog.root / "roop_3d/.runtime/cache/cuda_fp16"
            self.path, _ = _cuda_fp16_model_path(self.path, fp16_cache)
        self._identity = self._stat()
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = config.cpu_threads, 1
        options.enable_mem_pattern = False
        options.log_severity_level = 3
        options.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            if config.graph_optimization == "extended" else ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
        if config.provider == "TensorrtExecutionProvider":
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        self.executions = 0
        self.provider = self.requested_provider = config.provider
        self.fallback_reason: str | None = None
        self._options = options
        self.session = None
        providers: list = ["CPUExecutionProvider"]
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
        self._gpu_memory_limit = (config.gpu_budget_bytes // divisor
                                  if config.provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider") else None)
        if config.provider == "TensorrtExecutionProvider":
            if "TensorrtExecutionProvider" not in ort.get_available_providers():
                raise DomainError("PROVIDER_UNAVAILABLE", "Requested TensorRT provider is unavailable", 503)
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise DomainError("PROVIDER_UNAVAILABLE", "CUDAExecutionProvider is required for TensorRT graph partitions", 503)
            try:
                trt_provider = ("tensorrt_mixed" if config.precision_policy == "fp16"
                                else "TensorrtExecutionProvider")
                info = SessionManager(fp16=config.precision_policy != 'fp32', hardware_profile=config.hardware_profile,
                    gpu_memory_limit=self._gpu_memory_limit).create(self.path, model_name=name,
                        providers=(trt_provider, "CUDAExecutionProvider"))
                self.session = info.session
                self.provider = self.session.get_providers()[0]
                if self.provider != "TensorrtExecutionProvider":
                    self.fallback_reason = "; ".join(info.fallback_errors) or "TensorRT initialization failed; session remained on CUDAExecutionProvider"
            except Exception:
                raise DomainError("MODEL_SESSION_FAILED", "Verified model could not initialize on TensorRT or CUDA", 503) from None
        elif config.provider == "CUDAExecutionProvider":
            if config.provider not in ort.get_available_providers():
                raise DomainError("PROVIDER_UNAVAILABLE", "Requested CUDA provider is unavailable", 503)
            ort.preload_dlls()
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
            # Reference ONNX graphs include host shape/control operations. Allow
            # those graph partitions while keeping CUDA first and disabling
            # whole-session fallback below. Strict TRT builders retain their
            # separate GPU-only graph/canary policy.
            providers = [(config.provider, cuda_opts), "CPUExecutionProvider"]
        if config.provider != "TensorrtExecutionProvider":
            retry_reason = None
            try:
                self.session = ort.InferenceSession(str(self.path), sess_options=options, providers=providers)
                if config.provider == "CUDAExecutionProvider":
                    self.session.disable_fallback()
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
        self._io_binder = GpuIoBinding(self.session)
        # Restorers report themselves to the active-model read-back (swapper/detector report from their own holders).
        self._active_key = None
        if self.record.get("role") == "face_restorer":
            from roop_3d.engine import active_models
            entry = active_models.activate("enhancer", self.record["model_id"], self.path, self.provider,
                                           config.precision_policy, self.record.get("sha256"))
            self._active_key = entry.key
        register(self, "release")

    @guarded
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
        self._io_binder = GpuIoBinding(self.session)

    @guarded
    def _switch_to_cuda(self, reason: str) -> None:
        """Recover a failed TensorRT engine with a CUDA-only session."""
        self.session = None
        self._io_binder = None
        try:
            info = SessionManager(fp16=False, hardware_profile=self.config.hardware_profile,
                gpu_memory_limit=self._gpu_memory_limit).create(self.path, model_name=self.record["name"],
                    providers=("CUDAExecutionProvider",))
        except Exception:
            raise DomainError("MODEL_SESSION_FAILED", "CUDA-only recovery session could not initialize", 503) from None
        self.session, self.provider = info.session, info.provider
        self.fallback_reason = reason
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
        retry_reason = None
        try:
            if getattr(self, "_io_binder", None) is not None:
                out_map = self._io_binder.run(values)
                arrays = [out_map[node.name].numpy() for node in self.session.get_outputs()]
            else:
                arrays = self.session.run(None, values)
        except Exception as error:
            if self.provider == "TensorrtExecutionProvider":
                retry_reason = f"TensorRT inference failed; CUDA-only session selected: {type(error).__name__}"
            elif self.requested_provider == "TensorrtExecutionProvider":
                raise DomainError("MODEL_INFERENCE_FAILED", "CUDA-only inference failed after TensorRT recovery", 503) from None
            else:
                retry_reason = cuda_fallback_reason(str(error)) if self.provider == "CUDAExecutionProvider" else None
            if retry_reason is None:
                raise DomainError("MODEL_INFERENCE_FAILED", "Model execution failed", 503) from None
        # Exit the exception handler first so its traceback cannot retain the
        # failed CUDA session/arena while the reference session is constructed.
        if retry_reason is not None:
            if self.requested_provider == "TensorrtExecutionProvider":
                self._switch_to_cuda(retry_reason)
            else:
                self._switch_to_cpu(retry_reason)
            try:
                if self._io_binder is not None:
                    out_map = self._io_binder.run(values)
                    arrays = [out_map[node.name].numpy() for node in self.session.get_outputs()]
                else:
                    arrays = self.session.run(None, values)
            except Exception:
                message = ("CUDA-only inference failed after TensorRT recovery"
                           if self.requested_provider == "TensorrtExecutionProvider"
                           else "CPU reference execution failed")
                raise DomainError("MODEL_INFERENCE_FAILED", message, 503) from None
        outputs = {}
        for node, value in zip(self.session.get_outputs(), arrays):
            checked = contract_array(value, node.shape,
                dtype=np.dtype({"tensor(float16)": "float16", "tensor(double)": "float64"}.get(node.type, "float32")))
            # Callers consume float32 regardless of the graph's storage precision.
            outputs[node.name] = checked.astype(np.float32) if checked.dtype == np.float16 else checked
        self.executions += 1
        return outputs

    @guarded
    def release(self) -> None:
        self.session = None
        self._io_binder = None
        if getattr(self, "_active_key", None) is not None:
            from roop_3d.engine import active_models
            active_models.release("enhancer", self._active_key)
            self._active_key = None

    def capability(self) -> dict:
        return {"model_id": self.record["model_id"], "sha256": self.record["sha256"],
                "artifact_state": "VERIFIED_LOCAL", "inference_state": "VERIFIED_LOCAL" if self.executions else "UNKNOWN",
                "provider": self.provider, "executions": self.executions,
                "execution_providers": self.session.get_providers() if self.session is not None else [],
                "cpu_graph_partitions_allowed": self.requested_provider == "CUDAExecutionProvider",
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
    if factory is None and config.provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
        factory = OrtModel
    if factory is None:
        return None
    return lambda record: SessionFacade(factory(catalog, record["name"], config))


class InspectionOnCpu(RuntimeError):
    """The inspection factory chose (or fell back to) the CPU; the caller builds its own CPU session."""


def inspection_runtime(settings=None) -> RuntimeConfig | None:
    """CUDA runtime for the inspection / target-lock models (detector, ArcFace, 68-point landmarks), or None for CPU.

    Inspection and the target-lock keyframe scan analyse every frame of the video, usually while a render job is running.
    On the CPU that costs ~200 ms (RetinaFace) + ~200 ms (ArcFace) + ~45 ms (1k3d68) per frame and pegs every core; on
    CUDA the same graphs run in 26 / 17 / 10 ms with identical output.  Cards under 10 GiB stay on the CPU: the resident
    inspection models would shrink the free VRAM the job admission needs and could push the whole render onto the CPU
    profile.  ``ROOP3D_INSPECTION_PROVIDER`` = ``cpu`` | ``cuda`` overrides the decision.
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
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            return None
        from roop_3d.core.hardware import MIN_12GB_PROFILE_VRAM_MIB, probe_hardware, resolve_hardware_profile
        hardware = probe_hardware(settings)
    except Exception:
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
    """``factory(record_or_name_dict) -> session`` on CUDA, raising :class:`InspectionOnCpu` when the CPU should be used.

    The decision is re-evaluated on every call (sessions are built rarely) so it follows the current free VRAM.  A fixed
    ``config`` (a render job's own runtime) skips the decision; a CPU config always means CPU.
    """
    def factory(record):
        resolved = ((None if config.provider == "CPUExecutionProvider" else config) if config is not None
                    else inspection_runtime(settings_provider() if settings_provider is not None else None))
        if resolved is None:
            raise InspectionOnCpu("inspection models stay on the CPU")
        return SessionFacade(OrtModel(catalog, record["name"], resolved))
    factory.cpu_fallback = True      # callers build their own CPU session when this factory raises
    return factory


def build_with_cpu_fallback(factory, record, name: str):
    """``factory(record)``, or None when an inspection factory declines / fails (the caller then builds its CPU session).

    A factory without ``cpu_fallback`` (the render trackers') keeps its errors: a failed GPU session must stay visible there."""
    if factory is None:
        return None
    try:
        return factory(record)
    except Exception as error:
        if not getattr(factory, "cpu_fallback", False):
            raise
        if not isinstance(error, InspectionOnCpu):
            log.warning("%s could not start on the GPU (%s); using the CPU", name, error)
        return None


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
