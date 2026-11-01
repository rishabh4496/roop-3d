"""Strict, lazy TensorRT sessions with explicit profiles and executed canaries."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import time
import threading
from typing import Callable, Mapping

import numpy as np

from roop_3d.audit.isolation import sha256_file
from roop_3d.models.dynamic_shapes import assert_finite, input_shapes, runtime_inputs, sample_inputs, static_batch_reason
from roop_3d.models.native_libraries import register_native_libraries
from roop_3d.models.owned_files import cache_directory, owned_path, model_source
from roop_3d.core.storage import exclusive
from roop_3d.audit.project_paths import application_root
from roop_3d.engine.resources import guarded

LOG = logging.getLogger(__name__)
TRT = "TensorrtExecutionProvider"


class StaticBatchError(RuntimeError):
    pass


class TensorRTBuildError(RuntimeError):
    pass


def precision_for(model: str | Path, environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    if env.get("ROOP_TRT_FP16") == "0":
        return "FP32"
    name = Path(model).stem.casefold().removesuffix("_dynamic")
    if name == "inswapper_128" and env.get("ROOP_SWAP_FP16") != "1":
        return "FP32"
    return "FP16"


@dataclass(frozen=True)
class BatchProfile:
    batches: tuple[int, ...] = (1, 4, 8)
    optimum: int = 4

    def __post_init__(self):
        if (not self.batches or tuple(sorted(set(self.batches))) != self.batches
                or any(type(b) is not int or b < 1 for b in self.batches) or self.optimum not in self.batches):
            raise ValueError("Batches must be positive, unique and sorted; optimum must be declared")


def validate_profile(model, profile: BatchProfile) -> None:
    for value in runtime_inputs(model):
        dims = value.type.tensor_type.shape.dim
        if not dims or value.name in ("w", "fidelity_weight"):
            continue
        fixed = dims[0].dim_value
        if fixed and any(b != fixed for b in profile.batches):
            raise StaticBatchError(f"{value.name} has static batch {fixed}; convert to <name>_dynamic.onnx before using {profile.batches}")


def cuda_memory_budget(hardware_profile: str, workspace_bytes: int) -> int:
    from roop_3d.core.hardware import probe_hardware
    from roop_3d.core.settings import Settings
    hardware = probe_hardware(Settings(hardware_profile=hardware_profile, preferred_provider="CUDAExecutionProvider"))
    if not hardware.gpus:
        raise TensorRTBuildError("No verified local NVIDIA GPU memory budget is available")
    gpu = hardware.gpus[0]
    usable = min(gpu.free_vram_mib, gpu.total_vram_mib, {"6gb": 6144, "12gb": 12288}[hardware_profile]) * 1048576
    remaining = usable - 768 * 1048576 - workspace_bytes
    if remaining < 16 * 1048576:
        raise TensorRTBuildError("Insufficient free VRAM after workspace and safety reserve")
    return remaining


class TensorRTSessionBuilder:
    def __init__(self, *, telemetry: Callable[[dict], None] | None = None,
                 hardware_profile: str = "6gb", workspace_mib: int | None = None):
        if hardware_profile not in ("6gb", "12gb"):
            raise ValueError("Hardware profile must be 6gb or 12gb")
        ceiling = {"6gb": 1024, "12gb": 2048}[hardware_profile]
        self.workspace_mib = ceiling if workspace_mib is None else workspace_mib
        if not 16 <= self.workspace_mib <= ceiling:
            raise ValueError(f"Workspace must be between 16 and {ceiling} MiB for this profile")
        self.telemetry = telemetry
        self.hardware_profile = hardware_profile
        self.last_build: dict = {}
        self._lock = threading.RLock()

    def _emit(self, state: str, name: str, precision: str) -> None:
        payload = {"engine_state": state, "engine_label": "Compiling TensorRT Engine..." if state != "IDLE" else None,
                   "engine_model": name if state != "IDLE" else None,
                   "engine_precision": precision if state != "IDLE" else None,
                   "engine_since": time.time() if state != "IDLE" else None}
        if self.telemetry:
            try:
                self.telemetry(payload)
            except Exception:
                LOG.exception("TensorRT telemetry observer failed")

    @guarded
    def build(self, model_path: str | Path, *, profile: BatchProfile | None = None,
              shapes: dict[str, tuple[int, ...]] | None = None,
              canary_inputs: Callable[[int], dict[str, np.ndarray]] | None = None,
              selected_provider: str = TRT):
        # Explicit TRT selection is strict even without an environment flag.
        if selected_provider != TRT:
            raise TensorRTBuildError("The strict TensorRT builder requires TensorrtExecutionProvider"
                                     + (" (ROOP_OPT_STRICT_TRT=1)" if os.environ.get("ROOP_OPT_STRICT_TRT") == "1" else ""))
        register_native_libraries()
        import onnx
        import onnxruntime as ort
        path = model_source(Path(model_path))
        model = onnx.load(path, load_external_data=False)
        if any(v.data_location == onnx.TensorProto.EXTERNAL for v in model.graph.initializer):
            raise TensorRTBuildError("External tensor data needs separately verified registry support")
        reason = static_batch_reason(path.name)
        profile = profile or (BatchProfile((1,), 1) if reason else BatchProfile())
        if reason and profile.batches != (1,):
            raise StaticBatchError(reason)
        validate_profile(model, profile)
        if TRT not in ort.get_available_providers():
            raise TensorRTBuildError("ONNX Runtime does not advertise TensorRT; CPU fallback is forbidden")
        precision = precision_for(path)
        cache = cache_directory()
        # Keep native/JIT temporary writes within the checkout as well.
        from roop_3d.models.trt_builder import configure_native_paths
        configure_native_paths(application_root())
        shape_sets = {key: input_shapes(model, batch, shapes) for key, batch in
                      (("min", profile.batches[0]), ("opt", profile.optimum), ("max", profile.batches[-1]))}
        try:
            trt_version = importlib.metadata.version("tensorrt-cu12-libs")
        except importlib.metadata.PackageNotFoundError:
            raise TensorRTBuildError("Packaged tensorrt-cu12-libs is missing; run the NVIDIA installer") from None
        identity = {"model": sha256_file(path)[1], "precision": precision, "profiles": shape_sets,
                    "ort": ort.__version__, "trt": trt_version, "workspace_mib": self.workspace_mib}
        prefix = "roop3d_" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:32]
        provider_options = {
            "device_id": 0, "trt_fp16_enable": precision == "FP16", "trt_int8_enable": False,
            "trt_max_workspace_size": self.workspace_mib * 1048576,
            "trt_engine_cache_enable": True, "trt_engine_cache_path": str(cache),
            "trt_engine_cache_prefix": prefix,
            "trt_timing_cache_enable": True, "trt_timing_cache_path": str(cache),
            "trt_force_timing_cache": False, "trt_force_sequential_engine_build": True,
            "trt_auxiliary_streams": 0, "trt_layer_norm_fp32_fallback": True,
            "trt_detailed_build_log": True,
        }
        for key, shape_set in shape_sets.items():
            provider_options[f"trt_profile_{key}_shapes"] = ",".join(
                f"{name}:" + "x".join(map(str, shape)) for name, shape in shape_set.items() if shape)
        options = ort.SessionOptions()
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        # A profile proves CPU was not used even if an ORT build changes defaults.
        options.enable_profiling = True
        trace = owned_path(cache / f"{prefix}_execution")
        options.profile_file_prefix = str(trace)
        cuda_options = {"use_tf32": False, "arena_extend_strategy": "kSameAsRequested",
                        "cudnn_conv_use_max_workspace": False,
                        "gpu_mem_limit": cuda_memory_budget(self.hardware_profile, self.workspace_mib * 1048576)}
        with self._lock, exclusive(application_root(), "roop_3d/.runtime/strict-trt-build.lock"):
            # Validate existing cache descendants before handing a writable directory to native code.
            for child in cache.rglob("*"):
                owned_path(child)
            existing = {p.name: (p.stat().st_size, p.stat().st_mtime_ns)
                        for p in cache.glob(f"{prefix}*.engine") if p.stat().st_size > 0}
            cold = not existing
            started = time.perf_counter()
            session = None
            try:
                self._emit("COMPILING_TRT" if cold else "RUNNING_CANARY", path.name, precision)
                session = ort.InferenceSession(str(path), sess_options=options,
                    providers=[(TRT, provider_options), ("CUDAExecutionProvider", cuda_options)])
                session.disable_fallback()
                if not session.get_providers() or session.get_providers()[0] != TRT:
                    raise TensorRTBuildError(f"TensorRT activation failed: {session.get_providers()}")
                compile_seconds = 0.0
                for index, batch in enumerate(profile.batches):
                    # Cold engine compilation can happen at creation OR first run.
                    before_run = time.perf_counter()
                    feeds = canary_inputs(batch) if canary_inputs else sample_inputs(model, batch, shapes)
                    expected = input_shapes(model, batch, shapes)
                    if set(feeds) != set(expected) or any(tuple(feeds[n].shape) != s for n, s in expected.items()):
                        raise TensorRTBuildError(f"Canary inputs do not match batch {batch} profile")
                    assert_finite(list(feeds.values()))
                    outputs = session.run(None, feeds)
                    assert_finite(outputs)
                    if index == 0 and cold:
                        compile_seconds = time.perf_counter() - started
                        # Includes session creation and the first synchronous compile/inference call.
                        self._emit("RUNNING_CANARY", path.name, precision)
                    LOG.info("TensorRT canary model=%s batch=%d seconds=%.9f", path.name, batch, time.perf_counter() - before_run)
                engines = [owned_path(p) for p in cache.glob(f"{prefix}*.engine") if p.stat().st_size > 0]
                if not engines:
                    raise TensorRTBuildError("No model/profile/precision-matched .engine file was written; refusing CUDA-only inference")
                if cold and not any(p.name not in existing or (p.stat().st_size, p.stat().st_mtime_ns) != existing[p.name] for p in engines):
                    raise TensorRTBuildError("Cold build did not write a TensorRT engine")
                trace_path = owned_path(Path(session.end_profiling()))
                events = json.loads(trace_path.read_text("utf-8"))
                providers = {e.get("args", {}).get("provider") for e in events if e.get("cat") == "Node"}
                if "CPUExecutionProvider" in providers or TRT not in providers:
                    raise TensorRTBuildError(f"Canary node placement failed strict TensorRT checks: {sorted(p for p in providers if p)}")
                if cold:
                    LOG.info("TensorRT engine compile duration model=%s precision=%s seconds=%.9f scope=session_and_first_run",
                             path.name, precision, compile_seconds)
                self.last_build = {**identity, "engine_prefix": prefix, "cache_hit": not cold,
                                   "compile_seconds": compile_seconds, "checked_batches": list(profile.batches),
                                   "engines": [p.name for p in engines], "node_providers": sorted(p for p in providers if p)}
                return session
            except Exception as error:
                if session is not None:
                    session.end_profiling()
                if isinstance(error, (StaticBatchError, TensorRTBuildError)):
                    raise
                raise TensorRTBuildError(f"TensorRT build/canary failed for {path.name}: {error}") from error
            finally:
                self._emit("IDLE", path.name, precision)


def build_trt_session(model_path: str | Path, **kwargs):
    return TensorRTSessionBuilder().build(model_path, **kwargs)
