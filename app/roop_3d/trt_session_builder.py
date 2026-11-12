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
from roop_3d.core.ort_session import (GpuIoBinding, _fp32_safe_model_path,
                                      _profiles, _static_input_contract, _static_model_path,
                                      _static_graph_inputs, _trt_excluded_ops,
                                      _is_cuda_graph_partition_error, cuda_graphs_enabled)

LOG = logging.getLogger(__name__)
TRT = "TensorrtExecutionProvider"


class StaticBatchError(RuntimeError):
    pass


class TensorRTBuildError(RuntimeError):
    pass


def precision_for(model: str | Path, environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    from roop_3d.trt_fp8_builder import HEAVY_RESTORERS, model_name
    name = model_name(model)
    if name == "inswapper_128" or "motion_extractor" in name or "gpen" in name:
        return "FP32"
    if env.get("ROOP_TRT_FP16") == "0":
        return "FP32"
    if env.get("ROOP_TRT_FP8") == "1" and name in HEAVY_RESTORERS:
        return "FP8"
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


def auto_shape_profiles(model, name: str) -> list[dict[str, tuple[int, ...]]]:
    """Return explicit min/opt/max shapes for dynamic model inputs."""
    initializers = {value.name for value in model.graph.initializer}

    class InputShape:
        def __init__(self, value):
            self.name = value.name
            self.shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param or None
                          for dim in value.type.tensor_type.shape.dim]

    inputs = [InputShape(value) for value in model.graph.input if value.name not in initializers]
    return _profiles(inputs, name)


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

    def _emit(self, state: str, name: str, precision: str, batches: tuple[int, ...] | None = None, opt_shape: object = None) -> None:
        payload = {"engine_state": state, "engine_label": "Compiling TensorRT Engine..." if state != "IDLE" else None,
                   "engine_model": name if state != "IDLE" else None,
                   "engine_precision": precision if state != "IDLE" else None,
                   "engine_since": time.time() if state != "IDLE" else None,
                   "engine_batch_profile": list(batches) if state != "IDLE" and batches else None}
        if opt_shape is not None and state != "IDLE":
            payload["engine_opt_shape"] = opt_shape  # the optimisation-profile shape; only the heartbeat line uses it
        if self.telemetry:
            try:
                self.telemetry(payload)
            except Exception:
                LOG.exception("TensorRT telemetry observer failed")

    @guarded
    def build(self, model_path: str | Path, *, profile: BatchProfile | None = None,
              shapes: dict[str, tuple[int, ...]] | None = None,
              canary_inputs: Callable[[int], dict[str, np.ndarray]] | None = None,
              selected_provider: str = TRT, compute_stream: int | None = None,
              fp8_model_path: str | Path | None = None,
              fp8_validate: Callable[[object], bool] | None = None):
        # Explicit TRT selection is strict even without an environment flag.
        if selected_provider != TRT:
            raise TensorRTBuildError("The strict TensorRT builder requires TensorrtExecutionProvider"
                                     + (" (ROOP_OPT_STRICT_TRT=1)" if os.environ.get("ROOP_OPT_STRICT_TRT") == "1" else ""))
        register_native_libraries()
        import onnx
        import onnxruntime as ort
        path = model_source(Path(model_path))
        model = onnx.load(path, load_external_data=False)
        from roop_3d.trt_fp8_builder import model_name, validate_swapper_fp32, HEAVY_RESTORERS
        if model_name(path) == "inswapper_128":
            try:
                validate_swapper_fp32(model)
            except ValueError as error:
                raise TensorRTBuildError(str(error)) from error
        if any(v.data_location == onnx.TensorProto.EXTERNAL for v in model.graph.initializer):
            raise TensorRTBuildError("External tensor data needs separately verified registry support")
        reason = static_batch_reason(path.name)
        static_contract = _static_input_contract(path.name)
        batch_one_contract = static_contract is not None and static_contract[2]
        profile = profile or (BatchProfile((1,), 1) if reason or batch_one_contract else BatchProfile())
        if reason and profile.batches != (1,):
            raise StaticBatchError(reason)
        validate_profile(model, profile)
        available = set(ort.get_available_providers())
        if TRT not in available:
            raise TensorRTBuildError("ONNX Runtime does not advertise TensorRT; CPU fallback is forbidden")
        if "CUDAExecutionProvider" not in available:
            raise TensorRTBuildError("CUDAExecutionProvider is required for TensorRT graph partitions; CPU fallback is forbidden")
        precision = precision_for(path)
        reference_digest = sha256_file(path)[1]
        fp8_evidence = None
        if precision == "FP8" and fp8_model_path is None:
            raise TensorRTBuildError("FP8 needs a separately calibrated E4M3FN Q/DQ ONNX artifact")
        if fp8_model_path is not None:
            if model_name(path) not in HEAVY_RESTORERS:
                raise TensorRTBuildError("FP8 is restricted to the four heavy restorers; inswapper is FP32 only")
            if not callable(fp8_validate):
                raise TensorRTBuildError("FP8 requires an explicit reference parity acceptance callback")
            from roop_3d.trt_fp8_builder import validate_fp8_graph
            try:
                import torch
                import tensorrt
            except (ImportError, OSError) as error:
                raise TensorRTBuildError(f"Optional FP8 native backend unavailable: {error}") from error
            candidate = owned_path(Path(fp8_model_path))
            if candidate == path or not candidate.is_file():
                raise TensorRTBuildError("FP8 artifact must be a separate owned ONNX file")
            quantized = onnx.load(candidate, load_external_data=False)
            if any(v.data_location == onnx.TensorProto.EXTERNAL for v in quantized.graph.initializer):
                raise TensorRTBuildError("FP8 external tensor data is not admitted")
            try:
                fp8_evidence = validate_fp8_graph(quantized, path.name,
                    torch.cuda.get_device_capability(0), tensorrt.__version__)
            except ValueError as error:
                raise TensorRTBuildError(str(error)) from error
            if input_shapes(model, profile.optimum, shapes) != input_shapes(quantized, profile.optimum, shapes):
                raise TensorRTBuildError("FP8 graph input shapes differ from the reference")
            validate_profile(quantized, profile)
            path, model, precision = candidate, quantized, "FP8"
        cache = cache_directory()
        if precision == "FP16":
            path, safe_model = _fp32_safe_model_path(path, cache, model_name(path))
            if safe_model is not None:
                model = safe_model
        try:
            path, static_model = _static_model_path(path, cache, model_name(path))
        except (ValueError, onnx.checker.ValidationError) as error:
            raise TensorRTBuildError(f"Static detector graph preparation failed: {error}") from error
        if static_model is not None:
            model = static_model
        validate_profile(model, profile)
        # Keep native/JIT temporary writes within the checkout as well.
        from roop_3d.models.trt_builder import configure_native_paths
        configure_native_paths(application_root())
        auto_profiles = auto_shape_profiles(model, path.name)
        shape_sets = {}
        for key, batch, automatic in zip(
                ("min", "opt", "max"),
                (profile.batches[0], profile.optimum, profile.batches[-1]), auto_profiles):
            overrides = dict(automatic)
            overrides.update(shapes or {})
            shape_sets[key] = input_shapes(model, batch, overrides)
        try:
            trt_version = importlib.metadata.version("tensorrt-cu12-libs")
        except importlib.metadata.PackageNotFoundError:
            raise TensorRTBuildError("Packaged tensorrt-cu12-libs is missing; run the NVIDIA installer") from None
        excluded_ops = _trt_excluded_ops(model, path.name)
        static_inputs = _static_graph_inputs(model)
        identity = {"model": sha256_file(path)[1], "precision": precision, "profiles": shape_sets,
                    "ort": ort.__version__, "trt": trt_version, "workspace_mib": self.workspace_mib,
                    "builder_optimization_level": 5, "auxiliary_streams": 2,
                    "force_sequential_engine_build": False, "cuda_graph": static_inputs,
                    "excluded_ops": excluded_ops}
        if fp8_evidence:
            identity.update(reference_model=reference_digest, fp8=fp8_evidence)
        prefix = "roop3d_" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:32]
        provider_options = {
            "device_id": 0, "trt_fp16_enable": precision == "FP16", "trt_int8_enable": False,
            "trt_max_workspace_size": self.workspace_mib * 1048576,
            "trt_engine_cache_enable": True, "trt_engine_cache_path": str(cache),
            "trt_engine_cache_prefix": prefix,
            "trt_timing_cache_enable": True, "trt_timing_cache_path": str(cache),
            "trt_force_timing_cache": False,
            "trt_builder_optimization_level": 5,
            "trt_force_sequential_engine_build": False,
            "trt_auxiliary_streams": 2,
            "trt_cuda_graph_enable": static_inputs and cuda_graphs_enabled(),
            "trt_layer_norm_fp32_fallback": True,
            "trt_op_types_to_exclude": excluded_ops,
            "trt_detailed_build_log": False,
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
        cuda_options = {"use_tf32": False, "cudnn_conv_algo_search": "DEFAULT",
                        "arena_extend_strategy": "kSameAsRequested",
                        "cudnn_conv_use_max_workspace": False,
                        "gpu_mem_limit": cuda_memory_budget(self.hardware_profile, self.workspace_mib * 1048576)}
        if compute_stream is not None:
            if type(compute_stream) is not int or compute_stream <= 0:
                raise TensorRTBuildError('A non-default CUDA compute stream pointer is required')
            provider_options['user_compute_stream'] = str(compute_stream)
            cuda_options['user_compute_stream'] = str(compute_stream)
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
                self._emit("COMPILING_TRT" if cold else "RUNNING_CANARY", path.name, precision, profile.batches, shape_sets["opt"])
                provider_list = [(TRT, provider_options), ("CUDAExecutionProvider", cuda_options)]
                try:
                    session = ort.InferenceSession(str(path), sess_options=options, providers=provider_list)
                except Exception as error:
                    if (not provider_options["trt_cuda_graph_enable"]
                            or not _is_cuda_graph_partition_error(error)):
                        raise
                    # ORT's TensorRT CUDA-graph capture requires the whole session
                    # graph to belong to TRT. Static models with CUDA-only safe or
                    # unsupported nodes retry the same engine policy without graph
                    # capture, retaining TRT/CUDA execution and CPU fallback ban.
                    provider_options["trt_cuda_graph_enable"] = False
                    session = ort.InferenceSession(str(path), sess_options=options,
                        providers=[(TRT, provider_options), ("CUDAExecutionProvider", cuda_options)])
                session._model_path = str(path)
                session.disable_fallback()
                if not session.get_providers() or session.get_providers()[0] != TRT:
                    raise TensorRTBuildError(f"TensorRT activation failed: {session.get_providers()}")
                compile_seconds = 0.0
                canary_runner = GpuIoBinding(session)
                for index, batch in enumerate(profile.batches):
                    # Cold engine compilation can happen at creation OR first run.
                    before_run = time.perf_counter()
                    # Use the optimum spatial profile for representative input
                    # tensors, while still exercising every declared batch size.
                    feeds = canary_inputs(batch) if canary_inputs else sample_inputs(model, batch, shape_sets["opt"])
                    expected = input_shapes(model, batch, shape_sets["opt"])
                    if set(feeds) != set(expected) or any(tuple(feeds[n].shape) != s for n, s in expected.items()):
                        raise TensorRTBuildError(f"Canary inputs do not match batch {batch} profile")
                    assert_finite(list(feeds.values()))
                    device_outputs = canary_runner.run(feeds)
                    outputs = [device_outputs[node.name].numpy() for node in session.get_outputs()]
                    assert_finite(outputs)
                    if index == 0 and cold:
                        compile_seconds = time.perf_counter() - started
                        # Includes session creation and the first synchronous compile/inference call.
                        self._emit("RUNNING_CANARY", path.name, precision, profile.batches)
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
                if fp8_evidence and fp8_validate(session) is not True:
                    raise TensorRTBuildError("FP8 failed reference output parity acceptance")
                if cold:
                    LOG.info("TensorRT engine compile duration model=%s precision=%s seconds=%.9f scope=session_and_first_run",
                             path.name, precision, compile_seconds)
                self.last_build = {**identity, "engine_prefix": prefix, "cache_hit": not cold,
                                   "compile_seconds": compile_seconds, "checked_batches": list(profile.batches),
                                   "cuda_graph_enabled": provider_options["trt_cuda_graph_enable"],
                                   "engines": [p.name for p in engines], "node_providers": sorted(p for p in providers if p)}
                return session
            except Exception as error:
                if session is not None:
                    session.end_profiling()
                if isinstance(error, (StaticBatchError, TensorRTBuildError)):
                    raise
                raise TensorRTBuildError(f"TensorRT build/canary failed for {path.name}: {error}") from error
            finally:
                self._emit("IDLE", path.name, precision, profile.batches)

    def build_reconstruction_models(self, *, disentangled_encoder, lip_regressor, blendshape_bank,
                                    artifact_directory: str | Path | None = None,
                                    face_size: int = 224, mouth_size: int = 96):
        """Export and compile the Stage 7 reconstruction heads as FP16 TRT engines.

        The graph profile covers every batch from 1 through 8 and each batch is
        exercised by the existing strict GPU canary. On a 6 GB profile this
        larger profile is not admitted: callers should keep the existing CUDA
        or CPU reference path there.
        """
        if self.hardware_profile != "12gb":
            raise TensorRTBuildError("Dynamic reconstruction TensorRT batch 1..8 requires the 12gb profile")
        from roop_3d.audit.project_paths import application_root
        from roop_3d.models.owned_files import owned_path
        from roop_3d.reconstruction.monocular_regressors import (
            export_disentangled_face_encoder_onnx, export_lip_phoneme_regressor_onnx,
        )

        directory = owned_path(Path(artifact_directory) if artifact_directory else
                               application_root() / "roop_3d/.runtime/reconstruction_onnx")
        directory.mkdir(parents=True, exist_ok=True)
        profile = BatchProfile(tuple(range(1, 9)), 4)
        artifacts = (
            ("disentangled_face_encoder_dynamic.onnx",
             lambda path: export_disentangled_face_encoder_onnx(disentangled_encoder, path, face_size=face_size)),
            ("lip_phoneme_regressor_dynamic.onnx",
             lambda path: export_lip_phoneme_regressor_onnx(lip_regressor, path, mouth_size=mouth_size)),
            ("reduced_blendshape_bank_dynamic.onnx", lambda path: blendshape_bank.export_onnx(path)),
        )
        sessions = {}
        for filename, export in artifacts:
            path = owned_path(directory / filename)
            export(path)
            sessions[filename.removesuffix("_dynamic.onnx")] = self.build(path, profile=profile)
        return sessions


def build_trt_session(model_path: str | Path, **kwargs):
    return TensorRTSessionBuilder().build(model_path, **kwargs)
