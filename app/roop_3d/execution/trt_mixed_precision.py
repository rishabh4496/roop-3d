"""TensorRT mixed precision sessions with graph-level FP32 pinning and provider failover.

The public seam is :class:`TensorRTMixedPrecisionSessionManager.create`: callers receive
an ORT-compatible session whose ``run`` path retries the same inference on CUDA and then
CPU if the active provider fails. TensorRT uses a derived ONNX graph in which selected
numerically sensitive operators remain FP32; the source model is never modified.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import logging
import os
from pathlib import Path
import threading
from typing import Any, Callable, Mapping, Sequence

import numpy as np

TRT = "TensorrtExecutionProvider"
CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"

FP32_PINNED_OPS = frozenset({
    "RodriguesRotation",
    "SphericalHarmonics",
    "LayerNormalization",
    "InstanceNormalization",
    "Pow",
    "ReduceMean",
    "Softmax",
})
FP16_TENSOR_OPS = frozenset({"Conv", "ConvTranspose", "Gemm", "MatMul"})
MIXED_GRAPH_POLICY_VERSION = "fp16-fp32pinned-v1"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizationProfiles:
    """Resolved ORT TensorRT profile shapes; each shape map is keyed by graph input."""

    minimum: Mapping[str, tuple[int, ...]]
    optimal: Mapping[str, tuple[int, ...]]
    maximum: Mapping[str, tuple[int, ...]]
    dynamic_batch: bool

    def as_options(self) -> dict[str, str]:
        if not self.minimum:
            return {}
        return {
            "trt_profile_min_shapes": _format_shapes(self.minimum),
            "trt_profile_opt_shapes": _format_shapes(self.optimal),
            "trt_profile_max_shapes": _format_shapes(self.maximum),
        }


def _format_shapes(shapes: Mapping[str, Sequence[int]]) -> str:
    return ",".join(f"{name}:" + "x".join(str(int(dim)) for dim in shape)
                    for name, shape in sorted(shapes.items()))


def _family_shape(model_name: str, rank: int) -> tuple[int, ...]:
    folded = model_name.casefold().replace("-", "_")
    if any(token in folded for token in ("detector", "scrfd", "retinaface", "yolo")):
        channels, height, width = 3, 640, 640
    elif any(token in folded for token in ("arcface", "recognition", "adaface", "glintr")):
        channels, height, width = 3, 112, 112
    elif any(token in folded for token in ("3dmm", "smirk", "flame", "craniofacial")):
        channels, height, width = 3, 224, 224
    elif "inswapper" in folded:
        channels, height, width = 3, 128, 128
    elif any(token in folded for token in ("texture_prior", "textureprior", "3d_texture")):
        channels, height, width = 3, 256, 256
    elif any(token in folded for token in ("gpen", "realistic")):
        channels, height, width = 3, 512, 512
    elif "ultramax" in folded:
        channels, height, width = 3, 512, 512
    elif any(token in folded for token in ("inpaint", "background")):
        channels, height, width = 4 if "latent" in folded else 3, 512, 512
    else:
        channels, height, width = 3, 512, 512
    if rank == 4:
        return channels, height, width
    if rank == 3:
        return height, width, channels
    if rank == 2:
        return (512,)
    return tuple(1 for _ in range(max(0, rank - 1)))


def _input_specs(model_path: str | Path) -> list[tuple[str, tuple[int | None, ...]]]:
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    initializers = {value.name for value in model.graph.initializer}
    result = []
    for value in model.graph.input:
        if value.name in initializers:
            continue
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims: list[int | None] = []
        for dim in tensor_type.shape.dim:
            dims.append(int(dim.dim_value) if dim.dim_value > 0 else None)
        if dims:
            result.append((value.name, tuple(dims)))
    return result


def derive_optimization_profiles(
    model_path: str | Path,
    model_name: str | None = None,
    *,
    input_shapes: Mapping[str, Sequence[int]] | None = None,
) -> OptimizationProfiles:
    """Build deterministic B=1/4/8 profiles while preserving fixed model dimensions.

    Fully static inputs are left unprofiled. TensorRT cannot legally widen a static
    batch dimension without changing the model's exported semantics; such models can
    still use TensorRT at their exported shape and report ``dynamic_batch=False``.
    """
    name = model_name or Path(model_path).name
    overrides = {key: tuple(int(dim) for dim in shape) for key, shape in (input_shapes or {}).items()}
    minimum: dict[str, tuple[int, ...]] = {}
    optimal: dict[str, tuple[int, ...]] = {}
    maximum: dict[str, tuple[int, ...]] = {}
    dynamic_batch = False

    for input_name, declared in _input_specs(model_path):
        dims = list(declared)
        if input_name in overrides:
            concrete = list(overrides[input_name])
            if len(concrete) != len(dims) or any(dim <= 0 for dim in concrete):
                raise ValueError(f"Invalid shape override for {input_name!r}: {concrete!r}")
            if any(dimension is not None and dimension != concrete[index]
                   for index, dimension in enumerate(dims)):
                raise ValueError(f"Shape override contradicts static ONNX input {input_name!r}")
            dims = [dimension if dimension is not None else (None if index == 0 else concrete[index])
                    for index, dimension in enumerate(dims)]
        family = _family_shape(name, len(dims))
        base = [int(dim) if dim is not None else
                (family[index - 1] if len(dims) == 4 and 1 <= index <= 3 else
                 family[index - 1] if len(dims) == 3 and 1 <= index <= 2 else
                 family[0] if index == 1 else 1)
                for index, dim in enumerate(dims)]
        if input_name in overrides:
            base = list(overrides[input_name])

        if declared[0] is None:
            dynamic_batch = True
            low, mid, high = list(base), list(base), list(base)
            low[0], mid[0], high[0] = 1, 4, 8
            minimum[input_name], optimal[input_name], maximum[input_name] = tuple(low), tuple(mid), tuple(high)
        elif any(dimension is None for dimension in declared):
            # A non-batch dynamic axis still needs concrete profile values, but
            # production frame batches remain fixed to the export's leading axis.
            fixed = tuple(base)
            minimum[input_name] = optimal[input_name] = maximum[input_name] = fixed

    if not minimum:
        return OptimizationProfiles({}, {}, {}, False)
    return OptimizationProfiles(minimum, optimal, maximum, dynamic_batch)


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_mixed_precision_model(
    model_path: str | Path,
    cache_path: str | Path,
) -> tuple[Path, dict[str, int]]:
    """Write an FP16 graph with explicit FP32 islands for the sensitive operators."""
    import onnx
    from onnx import TensorProto
    from onnxruntime.transformers.float16 import convert_float_to_float16

    source = Path(model_path).resolve(strict=True)
    cache = Path(cache_path).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    digest = _digest_file(source)
    target = cache / f"{source.stem}.{digest[:20]}.{MIXED_GRAPH_POLICY_VERSION}.onnx"
    if target.is_file():
        try:
            cached = onnx.load(str(target), load_external_data=False)
            onnx.checker.check_model(cached)
            _assert_pinned_inputs_are_fp32(cached)
            _assert_tensor_ops_are_fp16(cached)
            return target, _count_pinned_ops(cached)
        except Exception:
            target.unlink(missing_ok=True)

    model = onnx.load(str(source), load_external_data=True)
    pinned_counts: dict[str, int] = {}
    for node in model.graph.node:
        if node.op_type in FP32_PINNED_OPS:
            pinned_counts[node.op_type] = pinned_counts.get(node.op_type, 0) + 1
    converted = convert_float_to_float16(
        model,
        keep_io_types=True,
        op_block_list=sorted(FP32_PINNED_OPS),
        node_block_list=[],
    )
    _sort_graph_nodes(converted.graph)
    try:
        converted = onnx.shape_inference.infer_shapes(converted, strict_mode=False)
    except Exception:
        pass
    _sort_graph_nodes(converted.graph)
    onnx.checker.check_model(converted)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    try:
        onnx.save(converted, str(temporary))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    # Verify that every FP32 island is present in the serialized graph before a
    # TensorRT engine can compile it. A converter API change must fail closed.
    saved = onnx.load(str(target), load_external_data=False)
    if _count_pinned_ops(saved) != pinned_counts:
        target.unlink(missing_ok=True)
        raise RuntimeError("Mixed precision graph changed the sensitive ONNX operators")
    _assert_pinned_inputs_are_fp32(saved)
    _assert_tensor_ops_are_fp16(saved)
    return target, pinned_counts


def _count_pinned_ops(model) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in model.graph.node:
        if node.op_type in FP32_PINNED_OPS:
            counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def _sort_graph_nodes(graph, *, outer_scope=()) -> None:
    """Restore ONNX topological order after ORT's FP16 converter appends Cast nodes."""
    from onnx import AttributeProto

    available = set(outer_scope)
    available.update(value.name for value in graph.input)
    available.update(value.name for value in graph.initializer)
    for sparse in graph.sparse_initializer:
        available.add(sparse.values.name)
        available.add(sparse.indices.name)
    nodes = list(graph.node)
    producers = {output: index for index, node in enumerate(nodes)
                 for output in node.output if output}
    dependencies: list[set[int]] = []
    dependents: list[list[int]] = [[] for _ in nodes]
    for index, node in enumerate(nodes):
        inputs = {name for name in node.input if name}
        unknown = inputs - available - producers.keys()
        if unknown:
            missing = ", ".join(sorted(unknown)[:4])
            raise RuntimeError(f"Converted ONNX node {node.name or node.op_type} has unknown inputs: {missing}")
        deps = {producers[name] for name in inputs if name in producers and producers[name] != index}
        dependencies.append(deps)
        for producer in deps:
            dependents[producer].append(index)
    remaining = [len(deps) for deps in dependencies]
    ready = [index for index, count in enumerate(remaining) if count == 0]
    heapq.heapify(ready)
    ordered = []
    while ready:
        index = heapq.heappop(ready)
        node = nodes[index]
        ordered.append(node)
        available.update(name for name in node.output if name)
        for dependent in dependents[index]:
            remaining[dependent] -= 1
            if remaining[dependent] == 0:
                heapq.heappush(ready, dependent)
    if len(ordered) != len(nodes):
        blocked = ", ".join(nodes[index].name or nodes[index].op_type
                             for index, count in enumerate(remaining) if count)[:400]
        raise RuntimeError(f"Unable to topologically order converted ONNX nodes: {blocked}")
    del graph.node[:]
    graph.node.extend(ordered)
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                _sort_graph_nodes(attribute.g, outer_scope=available)
            elif attribute.type == AttributeProto.GRAPHS:
                for child in attribute.graphs:
                    _sort_graph_nodes(child, outer_scope=available)


def _assert_pinned_inputs_are_fp32(model) -> None:
    from onnx import TensorProto

    types: dict[str, int] = {}
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        if value.type.HasField("tensor_type"):
            types[value.name] = value.type.tensor_type.elem_type
    types.update({value.name: value.data_type for value in model.graph.initializer})
    for node in model.graph.node:
        if node.op_type not in FP32_PINNED_OPS:
            continue
        for input_name in (*node.input, *node.output):
            if input_name and types.get(input_name) == TensorProto.FLOAT16:
                raise RuntimeError(f"FP32-pinned {node.op_type} value {input_name!r} stayed FP16")


def _assert_tensor_ops_are_fp16(model) -> None:
    from onnx import TensorProto

    types: dict[str, int] = {}
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        if value.type.HasField("tensor_type"):
            types[value.name] = value.type.tensor_type.elem_type
    types.update({value.name: value.data_type for value in model.graph.initializer})
    for node in model.graph.node:
        if node.op_type not in FP16_TENSOR_OPS:
            continue
        floating = [types[name] for name in (*node.input, *node.output)
                    if name and name in types and types[name] in (TensorProto.FLOAT, TensorProto.FLOAT16)]
        if floating and any(element_type != TensorProto.FLOAT16 for element_type in floating):
            raise RuntimeError(f"Tensor Core op {node.op_type} was not fully converted to FP16")


def _session_options(ort, intra_op_threads: int, inter_op_threads: int, *, gpu: bool):
    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    options.inter_op_num_threads = inter_op_threads
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    if gpu:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return options


def _new_session_under_build_fence(factory, *args, **kwargs):
    from roop_3d.engine.resources import build_fence

    with build_fence():
        return factory(*args, **kwargs)


def _cuda_options(device_id: int, hardware_profile: str, gpu_memory_limit: int | None,
                  user_compute_stream: int | None = None) -> dict[str, str]:
    options = {
        "device_id": str(device_id),
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "do_copy_in_default_stream": "1",
        "arena_extend_strategy": "kNextPowerOfTwo" if hardware_profile == "12gb" else "kSameAsRequested",
    }
    if gpu_memory_limit is not None:
        options["gpu_mem_limit"] = str(gpu_memory_limit)
    if user_compute_stream is not None:
        options["user_compute_stream"] = str(user_compute_stream)
    return options


class TensorRTMixedPrecisionSessionManager:
    """Create mixed-precision ORT sessions and retry failed providers down the chain."""

    def __init__(
        self,
        *,
        cache_path: str | Path | None = None,
        intra_op_threads: int | None = None,
        inter_op_threads: int = 1,
        gpu_id: int = 0,
        hardware_profile: str = "12gb",
        gpu_memory_limit: int | None = None,
        user_compute_stream: int | None = None,
        session_factory: Callable[..., Any] | None = None,
        ort_module: Any | None = None,
    ):
        cpus = os.cpu_count() or 1
        intra = min(16, max(1, cpus - 1)) if intra_op_threads is None else intra_op_threads
        if not 1 <= intra <= 16 or not 1 <= inter_op_threads <= 8 or gpu_id < 0:
            raise ValueError("Invalid ONNX Runtime thread or GPU settings")
        if (hardware_profile not in ("6gb", "12gb")
                or (gpu_memory_limit is not None and gpu_memory_limit < 1)
                or (user_compute_stream is not None and user_compute_stream < 1)):
            raise ValueError("Invalid hardware profile or CUDA memory budget")
        self.cache_path = Path(cache_path) if cache_path is not None else (
            Path(__file__).resolve().parents[2] / "models" / ".trt_cache"
        )
        self.intra_op_threads, self.inter_op_threads = intra, inter_op_threads
        self.gpu_id, self.hardware_profile = gpu_id, hardware_profile
        self.gpu_memory_limit = gpu_memory_limit
        self.user_compute_stream = user_compute_stream
        self._session_factory = session_factory
        self._ort_module = ort_module

    def _ort(self):
        if self._ort_module is None:
            from roop_3d.models.native_libraries import register_native_libraries
            register_native_libraries()
            import onnxruntime as ort
            try:
                ort.preload_dlls()
            except (AttributeError, OSError):
                pass
            self._ort_module = ort
        return self._ort_module

    def _new_session(self, provider: str, source_path: Path, trt_path: Path,
                     model_name: str, profiles: OptimizationProfiles):
        ort = self._ort()
        gpu = provider != CPU
        session_options = _session_options(ort, self.intra_op_threads, self.inter_op_threads, gpu=gpu)
        if provider == TRT:
            from roop_3d.core.ort_session import TRT_BUILDER_OPTIMIZATION_LEVEL, trt_int8_options
            int8_options = trt_int8_options(model_name, self.cache_path)
            self.cache_path.mkdir(parents=True, exist_ok=True)
            options = {
                "device_id": str(self.gpu_id),
                "trt_fp16_enable": "True",
                "trt_engine_cache_enable": "True",
                "trt_engine_cache_path": str(self.cache_path),
                "trt_timing_cache_enable": "True",
                "trt_timing_cache_path": str(self.cache_path),
                "trt_builder_optimization_level": str(TRT_BUILDER_OPTIMIZATION_LEVEL),
                "trt_max_workspace_size": str((1024 if self.hardware_profile == "6gb" else 2048) * 1024**2),
                "trt_auxiliary_streams": "2",
                "trt_layer_norm_fp32_fallback": "True",
                "trt_op_types_to_exclude": "NonMaxSuppression,GridSample",
                "trt_engine_cache_prefix": "roop3d_" + hashlib.sha256(
                    (f"{_digest_file(source_path)}:{model_name}:{profiles!r}:"
                     f"{TRT_BUILDER_OPTIMIZATION_LEVEL}:{sorted(int8_options.items())}:"
                     f"{sorted(FP32_PINNED_OPS)}:{sorted(FP16_TENSOR_OPS)}:{MIXED_GRAPH_POLICY_VERSION}").encode("utf-8")
                ).hexdigest()[:24],
                **profiles.as_options(),
            }
            options.update(int8_options)
            if self.user_compute_stream is not None:
                options["user_compute_stream"] = str(self.user_compute_stream)
            providers = [(TRT, options), (CUDA, _cuda_options(
                self.gpu_id, self.hardware_profile, self.gpu_memory_limit, self.user_compute_stream))]
            path = trt_path
        elif provider == CUDA:
            providers = [(CUDA, _cuda_options(
                self.gpu_id, self.hardware_profile, self.gpu_memory_limit, self.user_compute_stream))]
            path = source_path
        else:
            providers = [CPU]
            path = source_path
        factory = self._session_factory or ort.InferenceSession
        session = _new_session_under_build_fence(
            factory, str(path), sess_options=session_options, providers=providers)
        active = tuple(session.get_providers()) if hasattr(session, "get_providers") else ()
        if not active or active[0] != provider:
            raise RuntimeError(f"requested {provider}, active providers are {active!r}")
        disable_fallback = getattr(session, "disable_fallback", None)
        if callable(disable_fallback):
            disable_fallback()
        return session

    def create(
        self,
        model_path: str | Path,
        *,
        model_name: str | None = None,
        providers: Sequence[str] = (TRT, CUDA, CPU),
        input_shapes: Mapping[str, Sequence[int]] | None = None,
    ) -> "ResilientORTSession":
        path = Path(model_path).resolve(strict=True)
        name = model_name or path.name
        requested = tuple(provider for provider in (TRT, CUDA, CPU) if provider in providers)
        if not requested:
            raise ValueError("At least one of TensorRT, CUDA, or CPU must be requested")
        ort = self._ort()
        available = set(ort.get_available_providers())
        profiles = derive_optimization_profiles(path, name, input_shapes=input_shapes)
        history: list[str] = []
        trt_path = path
        if TRT in requested and TRT in available:
            try:
                trt_path, _counts = prepare_mixed_precision_model(path, self.cache_path)
            except Exception as error:
                reason = f"{TRT}: FP16 graph preparation failed: {type(error).__name__}: {error}"
                logger.warning("%s", reason)
                history.append(reason)
        elif TRT in requested:
            reason = f"{TRT}: provider unavailable in this ONNX Runtime build"
            logger.warning("%s", reason)
            history.append(reason)

        ordered = [provider for provider in requested if provider in available]
        for provider in requested:
            if provider not in available:
                reason = f"{provider}: provider unavailable in this ONNX Runtime build"
                logger.warning("%s", reason)
                if reason not in history:
                    history.append(reason)
        if not ordered:
            raise RuntimeError("No requested execution provider is available: " + " | ".join(history))

        for index, provider in enumerate(ordered):
            if provider == TRT and trt_path == path and any("FP16 graph preparation failed" in item for item in history):
                continue
            try:
                session = self._new_session(provider, path, trt_path, name, profiles)
                return ResilientORTSession(
                    manager=self,
                    model_path=path,
                    trt_model_path=trt_path,
                    model_name=name,
                    providers=ordered,
                    provider_index=index,
                    session=session,
                    fallback_history=history,
                    profiles=profiles,
                )
            except Exception as error:
                reason = f"{provider}: session initialization failed: {type(error).__name__}: {error}"
                logger.warning("%s", reason)
                history.append(reason)
        raise RuntimeError("No requested ONNX Runtime provider could initialize: " + " | ".join(history))

    create_session = create


class ResilientORTSession:
    """ORT session facade with exact-reason TensorRT -> CUDA -> CPU runtime failover."""

    def __init__(self, *, manager: TensorRTMixedPrecisionSessionManager, model_path: Path,
                 trt_model_path: Path, model_name: str, providers: Sequence[str], provider_index: int,
                 session: Any, fallback_history: Sequence[str], profiles: OptimizationProfiles):
        self.manager, self.model_path, self.trt_model_path = manager, model_path, trt_model_path
        self.model_name, self.providers = model_name, tuple(providers)
        self.user_compute_stream = manager.user_compute_stream
        self._provider_index, self._session = provider_index, session
        self.fallback_history = list(fallback_history)
        self.profiles = profiles
        self._fallback_lock = threading.RLock()

    @property
    def provider(self) -> str:
        return self.providers[self._provider_index]

    @property
    def session(self):
        return self._session

    def get_providers(self):
        return self._session.get_providers()

    def get_inputs(self):
        return self._session.get_inputs()

    def get_outputs(self):
        return self._session.get_outputs()

    def io_binding(self):
        return ResilientIOBinding(self)

    from roop_3d.engine.resources import inference as _inference_guard

    @_inference_guard
    def run(self, output_names, input_feed, **kwargs):
        while True:
            failed_session = self._session
            try:
                feed = _cpu_feed(input_feed) if self.provider == CPU else input_feed
                return failed_session.run(output_names, feed, **kwargs)
            except Exception as error:
                if not self._activate_fallback(failed_session, error, "run"):
                    raise

    @_inference_guard
    def run_with_iobinding(self, binding, **kwargs):
        if not isinstance(binding, ResilientIOBinding) or binding.owner is not self:
            raise TypeError("Use io_binding() from this resilient session so provider failover can replay inputs")
        if self.provider == CPU:
            binding._replay_on_current_provider()
            return None
        failed_session = self._session
        try:
            binding._ensure_current()
            return failed_session.run_with_iobinding(binding.inner, **kwargs)
        except Exception as error:
            if not self._activate_fallback(failed_session, error, "run_with_iobinding"):
                raise
        while True:
            try:
                binding._replay_on_current_provider()
                return None
            except Exception as replay_error:
                if not self._activate_fallback(self._session, replay_error, "iobinding fallback replay"):
                    raise

    def _activate_fallback(self, failed_session, error: BaseException, operation: str) -> bool:
        with self._fallback_lock:
            if self._session is not failed_session:
                return True
            old_index = self._provider_index
            failed_provider = self.providers[old_index]
            for index in range(old_index + 1, len(self.providers)):
                provider = self.providers[index]
                try:
                    session = self.manager._new_session(
                        provider, self.model_path, self.trt_model_path,
                        self.model_name, self.profiles,
                    )
                except Exception as fallback_error:
                    reason = (f"{provider}: fallback initialization failed after {failed_provider} "
                              f"{operation}: {type(fallback_error).__name__}: {fallback_error}")
                    logger.error("%s", reason)
                    self.fallback_history.append(reason)
                    continue
                reason = (f"{failed_provider} -> {provider} after {operation}: "
                          f"{type(error).__name__}: {error}")
                logger.warning("TensorRT mixed precision fallback: %s", reason)
                self.fallback_history.append(reason)
                self._provider_index, self._session = index, session
                return True
            reason = (f"{failed_provider}: {operation} failed and no lower provider initialized: "
                      f"{type(error).__name__}: {error}")
            logger.error("%s", reason)
            self.fallback_history.append(reason)
            return False


class ResilientIOBinding:
    """Reusable binding facade that retains device inputs so a failed run can replay on CPU."""

    def __init__(self, owner: ResilientORTSession):
        self.owner = owner
        self.inner = None
        self._bound_session = None
        self.inputs: dict[str, tuple[str, Any]] = {}
        self.outputs: dict[str, tuple[str, Any]] = {}
        self._fallback_binding = None
        self._fallback_outputs = None

    def _ensure_current(self):
        if self._fallback_binding is not None:
            return
        if self.inner is None or self._bound_session is not self.owner.session:
            self.inner = self.owner.session.io_binding()
            self._bound_session = self.owner.session

    def _bind_gpu(self, operation: str, callback: Callable[[], None]) -> None:
        while self.owner.provider != CPU:
            failed_session = self.owner.session
            try:
                self._ensure_current()
                callback()
                return
            except Exception as error:
                if not self.owner._activate_fallback(failed_session, error, operation):
                    raise

    def clear_binding_inputs(self):
        self.inputs.clear()
        self._fallback_binding = None
        self._fallback_outputs = None
        if self.inner is not None and self._bound_session is self.owner.session:
            self.inner.clear_binding_inputs()
        else:
            self.inner = None
            self._bound_session = None

    def clear_binding_outputs(self):
        self.outputs.clear()
        self._fallback_binding = None
        self._fallback_outputs = None
        if self.inner is not None and self._bound_session is self.owner.session:
            self.inner.clear_binding_outputs()
        else:
            self.inner = None
            self._bound_session = None

    def bind_ortvalue_input(self, name, value):
        self.inputs[name] = ("ortvalue", value)
        self._bind_gpu("bind_ortvalue_input", lambda: self.inner.bind_ortvalue_input(name, value))

    def bind_input_tensor(self, name: str, tensor: Any):
        if not getattr(tensor, "is_cuda", False) or not callable(getattr(tensor, "data_ptr", None)):
            raise ValueError("TensorRT/CUDA I/O binding requires a CUDA tensor")
        if not tensor.is_contiguous():
            raise ValueError("TensorRT/CUDA I/O binding requires a contiguous tensor")
        dtype_map = {
            "torch.float16": np.float16, "torch.float32": np.float32, "torch.float64": np.float64,
            "torch.int8": np.int8, "torch.uint8": np.uint8, "torch.int16": np.int16,
            "torch.int32": np.int32, "torch.int64": np.int64, "torch.bool": np.bool_,
        }
        try:
            dtype = np.dtype(dtype_map[str(tensor.dtype)])
        except KeyError:
            raise TypeError(f"Unsupported CUDA tensor dtype: {tensor.dtype}") from None
        device_id = tensor.device.index if tensor.device.index is not None else self.owner.manager.gpu_id
        if int(device_id) != self.owner.manager.gpu_id:
            raise ValueError(f"CUDA tensor is on device {device_id}, expected {self.owner.manager.gpu_id}")
        shape = tuple(int(dim) for dim in tensor.shape)
        self.inputs[name] = ("tensor", tensor)
        self._bind_gpu("bind_input_tensor", lambda: self.inner.bind_input(
            name, "cuda", device_id, dtype, shape, int(tensor.data_ptr())))

    def bind_cpu_input(self, name, value):
        array = np.ascontiguousarray(value)
        self.inputs[name] = ("numpy", array)

        def bind():
            if self.inputs[name][0] == "numpy":
                ort = self.owner.manager._ort()
                self.inputs[name] = ("ortvalue", ort.OrtValue.ortvalue_from_numpy(
                    array, "cuda", self.owner.manager.gpu_id))
            self.inner.bind_ortvalue_input(name, self.inputs[name][1])

        self._bind_gpu("bind_cpu_input", bind)

    def bind_input(self, name, device_type, device_id, element_type, shape, buffer_ptr):
        # Pointer-only bindings cannot be replayed safely; use bind_input_tensor
        # when runtime failover must retain the allocation.
        self.inputs[name] = ("pointer", (device_type, device_id, element_type, tuple(shape), buffer_ptr))
        self._bind_gpu("bind_input", lambda: self.inner.bind_input(
            name, device_type, device_id, element_type, shape, buffer_ptr))

    def bind_ortvalue_output(self, name, value):
        self.outputs[name] = ("ortvalue", value)
        self._bind_gpu("bind_ortvalue_output", lambda: self.inner.bind_ortvalue_output(name, value))

    def bind_output(self, name, device_type="cuda", device_id=0, *args):
        self.outputs[name] = ("device", (device_type, device_id, args))
        self._bind_gpu("bind_output", lambda: self.inner.bind_output(name, device_type, device_id, *args))

    def synchronize_inputs(self):
        if self.inner is not None and self.owner.provider != CPU:
            self.inner.synchronize_inputs()

    def synchronize_outputs(self):
        if self._fallback_binding is not None:
            self._fallback_binding.synchronize_outputs()
        elif self.inner is not None:
            self.inner.synchronize_outputs()

    def get_outputs(self):
        if self._fallback_outputs is not None:
            return self._fallback_outputs
        return self._fallback_binding.get_outputs() if self._fallback_binding is not None else self.inner.get_outputs()

    def _replay_on_current_provider(self):
        if self.owner.provider != CPU:
            self.inner = self.owner.session.io_binding()
            self._bound_session = self.owner.session
            for name, (kind, value) in self.inputs.items():
                if kind == "ortvalue":
                    self.inner.bind_ortvalue_input(name, value)
                elif kind == "tensor":
                    # A second GPU provider still accepts the same device pointer.
                    dtype_map = {"torch.float16": np.float16, "torch.float32": np.float32,
                                 "torch.float64": np.float64, "torch.int8": np.int8,
                                 "torch.uint8": np.uint8, "torch.int32": np.int32,
                                 "torch.int64": np.int64, "torch.bool": np.bool_}
                    dtype = np.dtype(dtype_map[str(value.dtype)])
                    self.inner.bind_input(name, "cuda", self.owner.manager.gpu_id, dtype,
                                          tuple(value.shape), int(value.data_ptr()))
                elif kind == "pointer":
                    device, device_id, dtype, shape, pointer = value
                    self.inner.bind_input(name, device, device_id, dtype, shape, pointer)
            for name, (kind, value) in self.outputs.items():
                if kind == "ortvalue":
                    self.inner.bind_ortvalue_output(name, value)
                else:
                    device, device_id, args = value
                    self.inner.bind_output(name, device, device_id, *args)
            self.owner.session.run_with_iobinding(self.inner)
            return

        binding = self.owner.session.io_binding()
        for name, (kind, value) in self.inputs.items():
            if kind == "numpy":
                array = value
            elif kind == "tensor":
                array = value.detach().cpu().numpy()
            elif kind == "ortvalue":
                array = value.numpy()
            elif kind == "pointer":
                raise RuntimeError(f"Cannot replay pointer-only I/O binding input {name!r} on CPU")
            else:
                raise RuntimeError(f"Unknown I/O binding input kind {kind!r}")
            binding.bind_cpu_input(name, np.ascontiguousarray(array))
        output_names = tuple(self.outputs)
        for name in output_names:
            binding.bind_output(name, "cpu", 0)
        self.owner.session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        self._fallback_binding = binding
        self._fallback_outputs = binding.get_outputs()


def _cpu_feed(feed: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize inputs only after the session has explicitly degraded to CPU."""
    result = {}
    for name, value in feed.items():
        if callable(getattr(value, "device_name", None)):
            result[name] = value.numpy()
        elif getattr(value, "is_cuda", False) and callable(getattr(value, "detach", None)):
            result[name] = value.detach().cpu().numpy()
        elif callable(getattr(value, "detach", None)):
            result[name] = value.detach().numpy()
        else:
            result[name] = value
    return result
