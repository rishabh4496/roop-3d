"""Offline TensorRT EP compilation with eviction, workspace caps and atomic cache.

Online inference consumes an embedded EPContext model, never an original ONNX
graph that could unexpectedly trigger another engine compilation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import contextmanager
import ctypes
import ctypes.util
import importlib.util
import json
import os
from pathlib import Path
from typing import Callable
import uuid

import numpy as np

from roop_3d.audit.project_paths import application_root
from roop_3d.core.errors import DomainError
from roop_3d.core.hardware import probe_hardware
from roop_3d.core.settings import Settings
from roop_3d.core.storage import owned
from roop_3d.engine.resources import build_fence, evict_for_build, guarded
from roop_3d.models.catalog import ModelCatalog
from roop_3d.models.engine_cache import EngineCache, EngineKey, ShapeRange
from roop_3d.models.export_validation import ParityReport
from roop_3d.models.precision_policy import PrecisionPolicy


@dataclass(frozen=True)
class TrtAvailability:
    available: bool
    version: str | None
    reason: str | None


@guarded
def configure_native_paths(root: Path) -> None:
    """Keep driver JIT cache and native compiler temporaries in this checkout."""
    cache = owned(root, "roop_3d/cache/cuda")
    temporary = owned(root, "roop_3d/.runtime/native-tmp")
    cache.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_CACHE_PATH"] = str(cache)
    os.environ["CUDA_CACHE_MAXSIZE"] = str(256 * 1048576)
    # Myelin engines require this setting to match at build and execution.
    # Keep it stable for the dedicated application process once acceleration
    # is configured; a temporary build-only override cannot satisfy that rule.
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    for name in ("TEMP", "TMP", "TMPDIR"):
        os.environ[name] = str(temporary)


def trt_availability() -> TrtAvailability:
    import onnxruntime as ort
    if "TensorrtExecutionProvider" not in ort.get_available_providers():
        return TrtAvailability(False, None, "ORT does not advertise TensorRT EP")
    if importlib.util.find_spec("tensorrt") is not None:
        try:
            import tensorrt
            return TrtAvailability(True, str(tensorrt.__version__), None)
        except (ImportError, OSError) as error:
            return TrtAvailability(False, None, f"TensorRT import failed: {error}")
    # C++ EP can be installed without Python bindings. Native availability and an
    # observed version are both required; never fabricate a cache version.
    library = "nvinfer_10.dll" if os.name == "nt" else ctypes.util.find_library("nvinfer")
    try:
        if not library:
            raise OSError("TensorRT native library was not found")
        ctypes.CDLL(library)
    except OSError as error:
        return TrtAvailability(False, None, f"TensorRT native library unavailable: {error}")
    return TrtAvailability(False, None, "TensorRT native library loaded, but its version is unverified; install the pinned optional bindings")


def workspace_budget(free_bytes: int, profile: str, reserve_bytes: int, requested_bytes: int) -> int:
    if profile not in ("6gb", "12gb") or min(free_bytes, reserve_bytes, requested_bytes) < 0:
        raise ValueError("Invalid build budget inputs")
    ceiling = {"6gb": 6144, "12gb": 12288}[profile] * 1048576
    budget = min(free_bytes - reserve_bytes, ceiling - reserve_bytes, requested_bytes)
    if budget < 16 * 1048576:
        raise DomainError("TRT_WORKSPACE_UNAVAILABLE", "Free VRAM after reserve is below the minimum build budget", 503)
    return budget


@contextmanager
def full_precision_build():
    """Call only under the inference/build fence, after owned session eviction."""
    previous = os.environ.get("NVIDIA_TF32_OVERRIDE")
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("NVIDIA_TF32_OVERRIDE", None)
        else:
            os.environ["NVIDIA_TF32_OVERRIDE"] = previous


def session_options(root: Path, cpu_threads: int, *, profiling: bool = True):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = cpu_threads, 1
    options.log_severity_level = 3
    options.enable_mem_pattern = False
    if profiling:
        prefix = owned(root, f"roop_3d/.runtime/profiles/stage07-{uuid.uuid4().hex}")
        prefix.parent.mkdir(parents=True, exist_ok=True)
        options.enable_profiling = True
        options.profile_file_prefix = str(prefix)
    return options


def finish_profile(session, root: Path) -> dict[str, int]:
    path = session.end_profiling()
    if not path:
        return {}
    profile = owned(root, Path(path).absolute().relative_to(root.absolute()).as_posix())
    events = json.loads(profile.read_text("utf-8"))
    providers: dict[str, int] = {}
    for event in events:
        provider = event.get("args", {}).get("provider")
        if event.get("cat") == "Node" and provider:
            providers[provider] = providers.get(provider, 0) + 1
    return providers


class TensorRTBuilder:
    def __init__(self, catalog: ModelCatalog, *, root: Path | None = None, profile: str = "6gb",
                 reserve_mib: int = 768, workspace_mib: int = 1024, cpu_threads: int = 2):
        if profile not in ("6gb", "12gb") or reserve_mib < 0 or workspace_mib < 16 or not 1 <= cpu_threads <= 8:
            raise ValueError("Invalid build policy")
        self.catalog, self.root = catalog, root or application_root()
        configure_native_paths(self.root)
        self.profile, self.reserve_mib, self.workspace_mib, self.cpu_threads = profile, reserve_mib, workspace_mib, cpu_threads
        self.cache = EngineCache(self.root)
        self.last_diagnostics: dict = {}

    def build(self, record: dict, policy: PrecisionPolicy, precision: str, golden_digest: str,
              cases: tuple[dict[str, np.ndarray], ...], validate: Callable[[object], tuple[ParityReport, ...]]) -> EngineKey:
        import onnxruntime as ort
        self.last_diagnostics = {"state": "CANDIDATE", "model_sha256": record["sha256"], "precision": precision}
        policy.tolerance(precision)
        if policy.excluded_trt_ops:
            self.last_diagnostics["state"] = "UNSUPPORTED"
            raise DomainError("TRT_CONTEXT_UNSUPPORTED",
                "FP32 policy excludes " + ", ".join(policy.excluded_trt_ops)
                + "; a complete builder-free TRT context is unavailable for this graph", 503)
        available = trt_availability()
        if not available.available:
            self.last_diagnostics = {"state": "UNSUPPORTED", "reason": available.reason}
            raise DomainError("TRT_UNAVAILABLE", available.reason or "TensorRT is unavailable", 503)
        path = self.catalog._path(record)
        with build_fence(), self.cache.lock("gpu-0-build"):
            eviction = evict_for_build()
            hardware = probe_hardware(Settings(hardware_profile=self.profile, preferred_provider="CUDAExecutionProvider",
                                                memory_safety_reserve_mb=self.reserve_mib))
            if not hardware.gpus or not hardware.gpus[0].compute_capability:
                raise DomainError("TRT_HARDWARE_UNKNOWN", "GPU architecture/free VRAM must be observed before building", 503)
            gpu = hardware.gpus[0]
            workspace = workspace_budget(gpu.free_vram_mib * 1048576, self.profile,
                                         self.reserve_mib * 1048576, self.workspace_mib * 1048576)
            shapes = {name: ShapeRange(minimum=value.shape, optimum=value.shape, maximum=value.shape)
                      for name, value in cases[0].items()}
            if any(set(case) != set(shapes) or any(tuple(case[n].shape) != shapes[n].optimum for n in shapes) for case in cases):
                raise DomainError("TRT_PROFILE_MISMATCH", "Fixed golden cases must match every declared shape profile", 422)
            key = EngineKey(artifact_sha256=record["sha256"], precision=precision,
                gpu_compute_capability=gpu.compute_capability, driver=gpu.driver_version,
                onnxruntime_version=ort.__version__, tensorrt_version=available.version, shape_profile=shapes,
                build_signature=f"stage07-v2-tf32off-ws{workspace}-{policy.digest()}-{golden_digest}")
            self.last_diagnostics = {"state": "CANDIDATE", "model_sha256": record["sha256"],
                "eviction": eviction, "free_vram_bytes": gpu.free_vram_mib * 1048576,
                "workspace_bytes": workspace, "safety_reserve_bytes": self.reserve_mib * 1048576,
                "tf32_enabled": False,
                "hardware_profile": self.profile, "key": key.model_dump(mode="json")}
            with self.cache.transaction(key) as (directory, existing), full_precision_build():
                if existing is not None:
                    self.last_diagnostics["cache_hit"] = True
                    self.last_diagnostics["state"] = "VERIFIED_LOCAL"
                    return key
                context = directory / "model_ctx.onnx"
                trt_options = {"device_id": 0, "trt_max_workspace_size": workspace,
                    "trt_fp16_enable": precision == "fp16", "trt_layer_norm_fp32_fallback": True,
                    # ORT resolves this relative to the context model directory.
                    # Absolute cache paths are rejected by its context security policy.
                    "trt_engine_cache_enable": True, "trt_engine_cache_path": ".",
                    "trt_engine_cache_prefix": key.digest(), "trt_force_sequential_engine_build": True,
                    "trt_auxiliary_streams": 0, "trt_dump_ep_context_model": True,
                    "trt_ep_context_file_path": str(context), "trt_ep_context_embed_mode": 1}
                if policy.excluded_trt_ops:
                    trt_options["trt_op_types_to_exclude"] = ",".join(policy.excluded_trt_ops)
                for field in ("minimum", "optimum", "maximum"):
                    suffix = {"minimum": "min", "optimum": "opt", "maximum": "max"}[field]
                    trt_options[f"trt_profile_{suffix}_shapes"] = ",".join(
                        name + ":" + "x".join(str(v) for v in getattr(shape, field)) for name, shape in sorted(shapes.items()))
                session = None
                try:
                    session = ort.InferenceSession(str(path), sess_options=session_options(self.root, self.cpu_threads),
                        providers=[("TensorrtExecutionProvider", trt_options),
                            ("CUDAExecutionProvider", {"gpu_mem_limit": max(16 * 1048576, gpu.free_vram_mib * 1048576 - self.reserve_mib * 1048576),
                                "use_tf32": False, "arena_extend_strategy": "kSameAsRequested",
                                "cudnn_conv_use_max_workspace": False}), "CPUExecutionProvider"])
                    if session.get_providers()[0] != "TensorrtExecutionProvider":
                        raise DomainError("TRT_ACTIVATION_FAILED", f"ORT selected {session.get_providers()} instead of TensorRT", 503)
                    reports = validate(session)
                    policy.admit(precision, reports)
                    providers = finish_profile(session, self.root)
                    session = None
                    if not providers.get("TensorrtExecutionProvider") or not context.is_file():
                        raise DomainError("TRT_CONTEXT_UNAVAILABLE", "No builder-free complete TensorRT EPContext was emitted; CUDA fallback required", 503)
                    # Embedded mode is self-contained. Discard duplicated external
                    # engine blobs; only the validated context model is published.
                    for file in directory.iterdir():
                        if file != context:
                            owned(self.root, file.relative_to(self.root).as_posix()).unlink()
                    session = ort.InferenceSession(str(context), sess_options=session_options(self.root, self.cpu_threads),
                        providers=["TensorrtExecutionProvider", "CPUExecutionProvider"])
                    context_reports = validate(session)
                    policy.admit(precision, context_reports)
                    context_providers = finish_profile(session, self.root)
                    session = None
                    if not context_providers.get("TensorrtExecutionProvider"):
                        raise DomainError("TRT_CONTEXT_NOT_EXECUTED", "Embedded context did not execute TensorRT kernels", 503)
                    self.last_diagnostics["state"] = "VERIFIED_LOCAL"
                    receipt = {**self.last_diagnostics, "state": "VERIFIED_LOCAL", "golden_digest": golden_digest,
                        "policy_digest": policy.digest(), "precision": precision, "context_model": "model_ctx.onnx",
                        "node_providers": context_providers, "parity": [asdict(r) for r in context_reports]}
                    self.cache.publish_bundle(key, directory, receipt)
                    return key
                except DomainError:
                    raise
                except Exception as error:
                    raise DomainError("TRT_BUILD_FAILED", str(error)[:4096], 503) from None
                finally:
                    session = None
