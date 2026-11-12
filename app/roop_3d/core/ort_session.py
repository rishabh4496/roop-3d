"""TensorRT-first ONNX Runtime sessions with per-model dynamic profiles."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
import time
from pathlib import Path
import os
import threading
from typing import Mapping

import numpy as np

TRT, CUDA, CPU = "TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"
_TRT_MIXED_ALIASES = {
    "tensorrt_mixed", "trt_mixed_precision", "tensorrt_mixed_precision", "tensorRT mixed precision".casefold(),
}


def _provider_name(value: str) -> str:
    return TRT if value.casefold() in _TRT_MIXED_ALIASES else value


def _string_provider_options(options: Mapping[str, object]) -> dict[str, str]:
    """The ORT C API accepts provider options as string-to-string maps."""
    return {key: str(value) for key, value in options.items()}


def _trt_mixed_options(trt_cache_dir: Path, device_id: int = 0) -> dict[str, str]:
    """Required TensorRT FP16 cache options in the string format expected by ORT's C API."""
    return {
        "device_id": str(device_id),
        "trt_fp16_enable": "True",
        "trt_engine_cache_enable": "True",
        "trt_engine_cache_path": str(trt_cache_dir),
        "trt_timing_cache_enable": "True",
        "trt_builder_optimization_level": "3",
    }


def _image_size(name: str) -> tuple[int, int, int]:
    name = name.casefold()
    if "xseg" in name:
        return 3, 256, 256
    if "inswapper" in name:
        return 3, 128, 128
    if any(token in name for token in ("ghost_256", "ghost_1_256", "ghost_2_256", "ghost_3_256",
                                       "blendswap_256", "uniface_256", "hyperswap_1a_256", "hyperswap_1b_256",
                                       "hyperswap_1c_256", "realswap_256", "simswap_256")):
        return 3, 256, 256
    if any(token in name for token in ("realswap_512", "simswap_512")):
        return 3, 512, 512
    if "256" in name and any(x in name for x in ("gpen", "ultramax", "gfpgan", "codeformer")):
        return 3, 256, 256
    if "1024" in name:
        return 3, 1024, 1024
    if any(x in name for x in ("gpen", "ultramax", "gfpgan", "codeformer")):
        return 3, 512, 512
    if any(x in name for x in ("scrfd", "retinaface")):
        return 3, 640, 640
    if any(x in name for x in ("landmark", "2dfan", "1k3d68", "106")):
        return 3, 192, 192
    return 3, 512, 512


def _static_graph_inputs(model) -> bool:
    initializers = {value.name for value in model.graph.initializer}
    inputs = [value for value in model.graph.input if value.name not in initializers]
    return bool(inputs) and all(
        all(dim.dim_value > 0 for dim in value.type.tensor_type.shape.dim)
        for value in inputs
    )


def _trt_excluded_ops(model, name: str) -> str:
    # These operations stay on CUDA when TensorRT cannot safely own them. CUDA EP
    # keeps the fallback on the device; CPU EP placement is disabled per session.
    # Resize stays TRT-eligible when the explicit input profiles make its shape
    # inferable. Unsupported/dynamic instances are partitioned to CUDA by ORT;
    # CPU placement remains disabled at the session level.
    # Exclusions are operator-type-wide in ORT, not per-node. In particular,
    # excluding Add/Mul/Tanh to protect only the enhancer output also moves the
    # same operators in every residual block to CUDA and fragments the graph into
    # dozens of tiny TensorRT engines. The FP32-safe enhancer graph inserts local
    # Cast boundaries around normalization, softmax and terminal scaling instead.
    return ",".join(sorted({"NonMaxSuppression", "GridSample"}))


_LOG = logging.getLogger("roop_3d.ort")


# -- preview session pool ----------------------------------------------------------------------------------------
# A live preview used to build every ORT session from scratch (11 models; LivePortrait's warping network alone took
# 20-40 s under TensorRT), so each preview of an uncached frame cost 40-60 s however cheap the frame was.  Inside
# ``pooled_sessions()`` (entered by the preview service only) ``SessionManager.create`` hands back the session it built
# for the same model and options last time.  Previews are serialised and ORT sessions are safe to ``Run`` from any
# thread, so sharing is sound.  The pool is bounded and is emptied the moment anything else (a render job, a TensorRT
# compile, a re-bind) takes the engine, so it never holds VRAM against real work.
_POOL_LOCK = threading.Lock()
_POOL: "dict[tuple, SessionInfo]" = {}
_POOL_LIMIT = 24
_POOL_SCOPE = threading.local()


def session_pool_enabled() -> bool:
    return os.environ.get("ROOP3D_PREVIEW_SESSION_POOL", "1") != "0"


class pooled_sessions:
    """Context manager: sessions created on this thread are pooled (re-entrant, thread-local)."""

    def __enter__(self):
        _POOL_SCOPE.depth = getattr(_POOL_SCOPE, "depth", 0) + 1
        return self

    def __exit__(self, *exc):
        _POOL_SCOPE.depth -= 1
        return False


def pool_scope_active() -> bool:
    return getattr(_POOL_SCOPE, "depth", 0) > 0 and session_pool_enabled()


_EVICTION_HOOKS: list = []


def on_pool_eviction(callback) -> None:
    """Register something else that is kept warm for previews (e.g. the 3D-head engine) and must go with the sessions."""
    _EVICTION_HOOKS.append(callback)


def evict_pooled_sessions() -> int:
    """Drop every pooled session (their VRAM returns once ORT collects them). Returns how many were dropped."""
    for hook in list(_EVICTION_HOOKS):
        try:
            hook()
        except Exception:
            pass
    with _POOL_LOCK:
        count = len(_POOL)
        _POOL.clear()
    if count:
        import gc
        gc.collect()
        _LOG.info("[ORT] preview session pool released (%d sessions) so the engine is free", count)
    return count


def pooled_session_count() -> int:
    return len(_POOL)


def cuda_graphs_enabled() -> bool:
    """TensorRT-EP CUDA-graph replay is only safe when ONE thread runs inference.

    Capture happens on the first runs of a session; any other thread that synchronises a stream meanwhile
    (the swap, track and composite stages of the async pipeline do) invalidates the capture with CUDA 906
    ("operation would make the legacy stream depend on a capturing blocking stream"), the run fails, the
    runtime rebuilds a session while other stages are mid-inference and the process dies with an access
    violation.  So graphs are off by default; ``ROOP3D_CUDA_GRAPH=1`` opts back in (e.g. single-thread use).
    The engine cache key still records the model's static-ness, so existing engines stay valid.
    """
    import os
    return os.environ.get("ROOP3D_CUDA_GRAPH", "0") == "1"


def _is_cuda_graph_partition_error(error: BaseException) -> bool:
    message = str(error).casefold()
    return "cuda graph" in message and "assigned" in message and "tensorrt" in message


def _cache_prefix(path: Path, name: str, profiles: list[dict], policy: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    identity = f"{name.casefold()}:{digest.hexdigest()}:{profiles!r}:{sorted(policy.items())!r}"
    return "roop3d_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]


_FP32_MODEL_LOCK = threading.Lock()
_CUDA_FP16_MODEL_LOCK = threading.Lock()


def _cuda_fp16_model_path(path: Path, cache_path: Path):
    """Build an owned CUDA mixed-precision graph while preserving its external float32 contract."""
    import onnx
    from onnx import TensorProto
    from onnxruntime.transformers.float16 import convert_float_to_float16

    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    source_digest = digest.hexdigest()
    from roop_3d.models.owned_files import owned_path
    cache_path = owned_path(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)
    target = owned_path(cache_path / f"{path.stem}.{source_digest[:20]}.cuda_fp16_mixed_v1.onnx")
    with _CUDA_FP16_MODEL_LOCK:
        if target.is_file():
            return target, None
        model = onnx.load(str(path), load_external_data=False)
        if any(value.data_location == TensorProto.EXTERNAL for value in model.graph.initializer):
            raise ValueError("CUDA FP16 graph conversion does not admit external ONNX data")
        converted = convert_float_to_float16(model, keep_io_types=True)
        onnx.checker.check_model(converted)
        temporary = owned_path(target.with_name(target.name + f".{os.getpid()}.tmp"))
        onnx.save(converted, str(temporary))
        os.replace(temporary, target)
    return target, converted


def _fp32_safe_model_path(path: Path, cache_path: Path, name: str):
    """Create an owned derived ONNX graph that pins sensitive enhancer ops to FP32.

    ONNX artifacts remain immutable. Explicit Cast nodes surround FP16
    normalization, softmax and terminal output scaling operators, so the CUDA
    partition executes those operations in FP32 while TRT still sees the
    convolution and matrix products as FP16 candidates.
    """
    if not any(token in name.casefold() for token in ("gpen", "ultramax", "codeformer")):
        return path, None
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(str(path), load_external_data=False)
    if any(value.data_location == TensorProto.EXTERNAL for value in model.graph.initializer):
        raise ValueError("FP32 enhancer graph preparation does not admit external ONNX data")
    try:
        model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:
        pass
    graph = model.graph
    value_types = {}
    for value in (*graph.input, *graph.output, *graph.value_info):
        if value.type.HasField("tensor_type"):
            value_types[value.name] = value.type.tensor_type.elem_type
    value_types.update({value.name: value.data_type for value in graph.initializer})

    producer = {output: node for node in graph.node for output in node.output if output}
    terminal_ops = {"Tanh", "Sigmoid", "Mul", "Add", "Sub", "Div", "Clip", "Cast"}
    tail_nodes = set()
    pending = [value.name for value in graph.output]
    initializers = set(value.name for value in graph.initializer)
    while pending:
        output = pending.pop()
        node = producer.get(output)
        if node is None or id(node) in tail_nodes or node.op_type not in terminal_ops:
            continue
        tail_nodes.add(id(node))
        pending.extend(value for value in node.input if value and value not in initializers)
    always_fp32 = {"LayerNormalization", "InstanceNormalization", "Softmax"}
    original_nodes = list(graph.node)
    rewritten = []
    serial = 0
    changed = False
    names = set(value_types) | {name for node in original_nodes for name in (*node.input, *node.output) if name}

    def unique(prefix):
        nonlocal serial
        while True:
            serial += 1
            candidate = f"{prefix}_{serial}"
            if candidate not in names:
                names.add(candidate)
                return candidate

    for node in original_nodes:
        force = node.op_type in always_fp32 or id(node) in tail_nodes
        if not force:
            rewritten.append(node)
            continue
        before, after = [], []
        for index, input_name in enumerate(node.input):
            if input_name and value_types.get(input_name) == TensorProto.FLOAT16:
                cast_name = unique(f"{input_name}_roop3d_fp32")
                before.append(helper.make_node("Cast", [input_name], [cast_name],
                                               to=TensorProto.FLOAT,
                                               name=f"Roop3D_FP32_Input_{serial + 1}"))
                node.input[index] = cast_name
                value_types[cast_name] = TensorProto.FLOAT
                changed = True
        for index, output_name in enumerate(node.output):
            if output_name and value_types.get(output_name) == TensorProto.FLOAT16:
                cast_name = unique(f"{output_name}_roop3d_fp32")
                node.output[index] = cast_name
                after.append(helper.make_node("Cast", [cast_name], [output_name],
                                              to=TensorProto.FLOAT16,
                                              name=f"Roop3D_FP16_Output_{serial + 1}"))
                value_types[cast_name] = TensorProto.FLOAT
                changed = True
        rewritten.extend(before)
        rewritten.append(node)
        rewritten.extend(after)
    if not changed:
        return path, model
    del graph.node[:]
    graph.node.extend(rewritten)
    try:
        model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:
        pass
    onnx.checker.check_model(model)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    cache_path.mkdir(parents=True, exist_ok=True)
    from roop_3d.models.owned_files import owned_path
    source_id = digest.hexdigest()
    target = owned_path(cache_path / f"{path.stem}.{source_id[:20]}.fp32safe_v1.onnx")
    with _FP32_MODEL_LOCK:
        if not target.is_file():
            temporary = owned_path(target.with_name(target.name + f".{os.getpid()}.tmp"))
            onnx.save(model, str(temporary))
            os.replace(temporary, target)
    return target, model


def _static_input_contract(name: str) -> tuple[int, int, bool] | None:
    """Return (height, width, batch_one) for fixed-shape app inference paths."""
    folded = name.casefold()
    if any(token in folded for token in ("scrfd", "retinaface")):
        return 640, 640, True
    if "inswapper" in folded:
        return 128, 128, False
    if any(token in folded for token in ("hyperswap_1a_256", "hyperswap_1b_256", "hyperswap_1c_256",
            "realswap_256", "simswap_256", "ghost_256", "ghost_1_256", "ghost_2_256", "ghost_3_256",
            "blendswap_256", "uniface_256")):
        return 256, 256, False
    if any(token in folded for token in ("realswap_512", "simswap_512")):
        return 512, 512, False
    if "xseg" in folded:
        return 256, 256, True
    if "1k3d68" in folded or "2d106det" in folded or "landmark" in folded:
        return 192, 192, True
    if "motion_extractor" in folded or "appearance_feature_extractor" in folded:
        return 256, 256, True
    if any(token in folded for token in ("gpen", "ultramax", "gfpgan", "codeformer", "bisenet", "face_parser")):
        _, height, width = _image_size(name)
        return height, width, True
    return None


def _static_model_path(path: Path, cache_path: Path, name: str):
    """Specialize dynamic inputs to the app's fixed inference geometry.

    Detector, parser, pose and enhancer adapters run one image at a time. Static
    inputs let ORT fold Shape/Gather/Resize scaffolding that otherwise gets
    assigned to CPU. Swapper batch models retain a dynamic batch while their
    128x128 spatial contract stays fixed.
    """
    contract = _static_input_contract(name)
    if contract is None:
        return path, None
    height, width, batch_one = contract
    import onnx
    from onnx import TensorProto
    model = onnx.load(str(path), load_external_data=False)
    if any(value.data_location == TensorProto.EXTERNAL for value in model.graph.initializer):
        raise ValueError("Static input preparation does not admit external ONNX data")
    initializers = {value.name for value in model.graph.initializer}
    changed = False
    for value in model.graph.input:
        if value.name in initializers:
            continue
        dims = value.type.tensor_type.shape.dim
        if len(dims) == 4 and dims[1].dim_value not in (0, 3):
            continue
        if batch_one and dims and (dims[0].dim_value != 1 or dims[0].HasField("dim_param")):
            dims[0].ClearField("dim_param")
            dims[0].dim_value = 1
            changed = True
        if len(dims) == 4:
            for dim, size in zip(dims[1:], (3, height, width)):
                if dim.dim_value <= 0 or dim.HasField("dim_param"):
                    dim.ClearField("dim_param")
                    dim.dim_value = size
                    changed = True
    if not changed:
        return path, model
    try:
        model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:
        pass
    onnx.checker.check_model(model)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    cache_path.mkdir(parents=True, exist_ok=True)
    from roop_3d.models.owned_files import owned_path
    batch_tag = "b1" if batch_one else "bx"
    target = owned_path(cache_path / f"{path.stem}.{digest.hexdigest()[:20]}.static{height}x{width}_{batch_tag}_v1.onnx")
    with _FP32_MODEL_LOCK:
        if not target.is_file():
            temporary = owned_path(target.with_name(target.name + f".{os.getpid()}.tmp"))
            onnx.save(model, str(temporary))
            os.replace(temporary, target)
    return target, model


def _profiles(nodes, name: str):
    channels, height, width = _image_size(name)
    folded_name = name.casefold()
    batch_sizes = (1, 4, 8) if "inswapper_128" in folded_name else (1, 2, 4)
    result = [{}, {}, {}]
    for node in nodes:
        if not node.shape or not any(not isinstance(d, int) or d <= 0 for d in node.shape):
            continue
        dims = [[], [], []]
        for axis, dim in enumerate(node.shape):
            if isinstance(dim, int) and dim > 0:
                choices = (dim, dim, dim)
            elif axis == 0:
                choices = batch_sizes
            elif len(node.shape) == 4 and axis == 1:
                choices = (channels,) * 3
            elif len(node.shape) == 4 and axis in (2, 3):
                size = height if axis == 2 else width
                # The application adapters submit fixed spatial contracts
                # (640 detector, 128 swapper, fixed parser/restorer sizes).
                # Varying H/W here builds needless engines and exposes dynamic
                # Resize partitions even though production never uses them.
                choices = (size, size, size)
            else:
                choices = (1, 1, 1)
            for dest, value in zip(dims, choices):
                dest.append(value)
        for dest, shape in zip(result, dims):
            dest[node.name] = tuple(shape)
    return result


@dataclass(frozen=True)
class SessionInfo:
    session: object
    provider: str
    requested_providers: tuple[str, ...]
    fallback_errors: tuple[str, ...] = ()


class SessionManager:
    """Build sessions in TensorRT -> CUDA -> CPU order, falling back on init errors."""
    def __init__(self, *, cache_path: str | Path | None = None, intra_op_threads: int = 2,
                 inter_op_threads: int = 1, fp16: bool = True, gpu_id: int = 0,
                 hardware_profile: str = "6gb", gpu_memory_limit: int | None = None):
        if not 1 <= intra_op_threads <= 16 or not 1 <= inter_op_threads <= 8 or gpu_id < 0:
            raise ValueError("Invalid ONNX Runtime thread or GPU settings")
        if hardware_profile not in ("6gb", "12gb") or gpu_memory_limit is not None and gpu_memory_limit < 1:
            raise ValueError("Invalid TensorRT hardware profile or CUDA memory budget")
        self.cache_path = Path(cache_path) if cache_path else Path(__file__).resolve().parents[2] / "models" / ".trt_cache"
        self.intra_op_threads, self.inter_op_threads = intra_op_threads, inter_op_threads
        self.fp16, self.gpu_id = fp16, gpu_id
        self.hardware_profile, self.gpu_memory_limit = hardware_profile, gpu_memory_limit

    def _session_options(self, ort, threads: int, *, gpu: bool = False):
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = threads, self.inter_op_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        if gpu:
            # Prevent a TensorRT/CUDA partition from silently assigning an
            # unsupported node to CPUExecutionProvider.
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        return options

    def _providers(self, ort, path: Path, name: str, provider: str, *, force_fp16: bool = False):
        available = set(ort.get_available_providers())
        if provider not in available:
            return []
        if provider == CPU:
            return [CPU]
        if provider == CUDA:
            options = {"device_id": self.gpu_id, "cudnn_conv_algo_search": "DEFAULT",
                       "do_copy_in_default_stream": "1",
                       "arena_extend_strategy": "kNextPowerOfTwo" if self.hardware_profile == "12gb" else "kSameAsRequested"}
            if self.gpu_memory_limit is not None:
                options["gpu_mem_limit"] = self.gpu_memory_limit
            return [(CUDA, _string_provider_options(options))]
        import onnx
        model = onnx.load(str(path), load_external_data=False)
        initializers = {v.name for v in model.graph.initializer}
        graph_inputs = [v for v in model.graph.input if v.name not in initializers]
        class Node:
            def __init__(self, value):
                self.name = value.name
                self.shape = [d.dim_value if d.dim_value > 0 else d.dim_param or None
                              for d in value.type.tensor_type.shape.dim]
        profile_nodes = [Node(v) for v in graph_inputs]
        profiles = _profiles(profile_nodes, name)
        if _static_graph_inputs(model):
            static_shapes = {node.name: tuple(node.shape) for node in profile_nodes if node.shape}
            profiles = [static_shapes, static_shapes, static_shapes]
        self.cache_path.mkdir(parents=True, exist_ok=True)
        if CUDA not in available:
            return []
        static = _static_graph_inputs(model)
        folded_name = name.casefold()
        fp16_sensitive = any(token in folded_name for token in ("inswapper", "motion_extractor", "gpen"))
        fp16 = force_fp16 or (self.fp16 and not fp16_sensitive)
        excluded_ops = _trt_excluded_ops(model, name)
        cache_policy = {"fp16": fp16, "optimization_level": 3, "auxiliary_streams": 2,
                        "cuda_graph": static, "force_sequential": False, "excluded_ops": excluded_ops}
        trt_cache_dir = self.cache_path
        mixed_options = _trt_mixed_options(trt_cache_dir, self.gpu_id)
        if not fp16:
            mixed_options["trt_fp16_enable"] = "False"
        options = {**mixed_options,
                   "trt_timing_cache_path": str(trt_cache_dir),
                   "trt_engine_cache_prefix": _cache_prefix(path, name, profiles, cache_policy),
                   "trt_max_workspace_size": (1024 if self.hardware_profile == "6gb" else 2048) * 1024**2,
                   "trt_auxiliary_streams": 2,
                   "trt_force_sequential_engine_build": False,
                   "trt_cuda_graph_enable": static and cuda_graphs_enabled(),
                   "trt_layer_norm_fp32_fallback": True,
                   "trt_op_types_to_exclude": excluded_ops}
        for key, profile in zip(("min", "opt", "max"), profiles):
            if profile:
                options[f"trt_profile_{key}_shapes"] = ",".join(
                    f"{tensor}:" + "x".join(map(str, shape)) for tensor, shape in profile.items())
        cuda_options = {"device_id": self.gpu_id, "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": "1",
                        "arena_extend_strategy": "kNextPowerOfTwo" if self.hardware_profile == "12gb" else "kSameAsRequested"}
        if self.gpu_memory_limit is not None:
            cuda_options["gpu_mem_limit"] = self.gpu_memory_limit
        # No explicit CPU entry: GPU sessions set session.disable_cpu_ep_fallback, and ORT >= 1.23 rejects a session
        # that both lists the CPU EP and disables CPU fallback ("Conflicting session configuration"), which made
        # every TensorRT build fail after compiling and silently drop to CUDA.
        return [(TRT, _string_provider_options(options)), (CUDA, _string_provider_options(cuda_options))]

    def create(self, model_path: str | Path, *, model_name: str | None = None,
               providers: tuple[str, ...] = (TRT, CUDA, CPU), intra_op_threads: int | None = None) -> SessionInfo:
        # Loading CUDA / TensorRT libraries and engines while another thread is mid-inference crashes the process
        # (access violation in the driver), so construction takes the exclusive side of the inference fence.
        from roop_3d.engine.resources import build_fence
        from roop_3d.core.transfer_ledger import TRANSFERS
        TRANSFERS.install()          # idempotent; makes every later ORT host<->device copy measurable
        pooled = getattr(_POOL_SCOPE, "depth", 0) > 0 and session_pool_enabled()
        key = None
        if pooled:
            try:
                status = Path(model_path).resolve(strict=True).stat()
                key = (str(Path(model_path).resolve()), status.st_mtime_ns, status.st_size, model_name, tuple(providers),
                       intra_op_threads or self.intra_op_threads, self.fp16, self.hardware_profile, self.gpu_id,
                       str(self.cache_path), cuda_graphs_enabled())   # the memory limit is a measured, moving budget: not identity
            except OSError:
                pooled = False
            else:
                with _POOL_LOCK:
                    hit = _POOL.get(key)
                if hit is not None:
                    _LOG.info("[ORT] %s -> %s (pooled, reused)", model_name or Path(model_path).name,
                              hit.provider.replace("ExecutionProvider", ""))
                    return hit
        with build_fence():
            info = self._create(model_path, model_name=model_name, providers=providers,
                                intra_op_threads=intra_op_threads)
        if pooled and key is not None:
            with _POOL_LOCK:
                while len(_POOL) >= _POOL_LIMIT:
                    _POOL.pop(next(iter(_POOL)))
                _POOL[key] = info
        return info

    def _create(self, model_path: str | Path, *, model_name: str | None = None,
                providers: tuple[str, ...] = (TRT, CUDA, CPU), intra_op_threads: int | None = None) -> SessionInfo:
        from roop_3d.models.native_libraries import register_native_libraries
        register_native_libraries()
        import onnxruntime as ort
        try:
            ort.preload_dlls()
        except (AttributeError, OSError):
            pass
        path = Path(model_path).resolve(strict=True)
        name = model_name or path.name
        mixed_requested = any(p.casefold() in _TRT_MIXED_ALIASES for p in providers)
        normalized = {_provider_name(provider) for provider in providers}
        order = tuple(p for p in (TRT, CUDA, CPU) if p in normalized)
        if not order:
            raise ValueError("At least one provider is required")
        errors = []
        for provider in order:
            attempt_started = time.perf_counter()
            try:
                session_path = path
                if provider == TRT:
                    session_path, _ = _fp32_safe_model_path(path, self.cache_path, name)
                if provider in (TRT, CUDA):
                    session_path, _ = _static_model_path(session_path, self.cache_path, name)
                if provider == CUDA and mixed_requested:
                    session_path, _ = _cuda_fp16_model_path(session_path, self.cache_path / "cuda_fp16")
                provider_options = self._providers(ort, session_path, name, provider,
                                                   force_fp16=mixed_requested)
                if not provider_options:
                    errors.append(f"{provider}: unavailable")
                    continue
                session_options = self._session_options(
                    ort, intra_op_threads or self.intra_op_threads, gpu=provider != CPU)
                try:
                    session = ort.InferenceSession(str(session_path), sess_options=session_options,
                                                   providers=provider_options)
                except Exception as error:
                    if (provider != TRT or provider_options[0][1].get("trt_cuda_graph_enable", "False").casefold() not in ("1", "true")
                            or not _is_cuda_graph_partition_error(error)):
                        raise
                    errors.append("TensorRT CUDA graph capture disabled for a mixed-provider partition")
                    # Provider entries are (name, options) pairs, except a bare provider name such as the CPU fallback.
                    retry_options = [(entry[0], dict(entry[1])) if isinstance(entry, tuple) else entry for entry in provider_options]
                    retry_options[0][1]["trt_cuda_graph_enable"] = "False"
                    session = ort.InferenceSession(str(session_path), sess_options=session_options,
                                                   providers=retry_options)
                active = session.get_providers()
                if not active or active[0] != provider:
                    raise RuntimeError(f"provider activation mismatch: {active}")
                if provider != CPU:
                    session.disable_fallback()
                    session._model_path = str(session_path)
                _LOG.info("[ORT] %s -> %s in %.1f s%s", name, provider.replace("ExecutionProvider", ""),
                          time.perf_counter() - attempt_started,
                          f" (after: {'; '.join(e[:90] for e in errors)})" if errors else "")
                return SessionInfo(session, provider, order, tuple(errors))
            except Exception as error:
                errors.append(f"{provider}: {type(error).__name__}: {error}")
                _LOG.warning("[ORT] %s on %s failed after %.1f s: %s", name, provider.replace("ExecutionProvider", ""),
                             time.perf_counter() - attempt_started, str(error).splitlines()[0][:160] if str(error) else type(error).__name__)
        raise RuntimeError("No requested ONNX Runtime provider could initialize: " + " | ".join(errors))


class GpuIoBinding:
    """Device-aware per-worker I/O binding with cached GPU buffers.

    CUDA/TRT sessions retain inputs and outputs as device OrtValues. CPU
    reference sessions use ORT's CPU binding path, so adapters do not switch
    back to ``session.run`` when exercising a reference provider.
    """
    def __init__(self, session, *, device_id: int = 0,
                 output_buffers: Mapping[str, tuple[tuple[int, ...], object]] | None = None):
        from roop_3d.core.transfer_ledger import TRANSFERS
        TRANSFERS.install()
        active = tuple(session.get_providers()) if hasattr(session, "get_providers") else ()
        self.device_name = "cuda" if active and active[0] in (TRT, CUDA) else "cpu"
        self.session, self.device_id = session, device_id if self.device_name == "cuda" else 0
        self.binding = session.io_binding()
        self._inputs: dict[str, tuple[tuple[int, ...], str, object]] = {}
        self._device_inputs: dict[str, object] = {}
        self._outputs = {}
        if output_buffers is None:
            output_buffers = {}
            for node in session.get_outputs():
                if all(type(dim) is int and dim > 0 for dim in node.shape):
                    dtype = {"tensor(float)": np.float32, "tensor(float16)": np.float16,
                             "tensor(double)": np.float64, "tensor(int64)": np.int64,
                             "tensor(int32)": np.int32}.get(node.type)
                    if dtype is not None:
                        output_buffers[node.name] = (tuple(node.shape), dtype)
        if output_buffers:
            import onnxruntime as ort
            for name, (shape, dtype) in output_buffers.items():
                self._outputs[name] = ort.OrtValue.ortvalue_from_shape_and_type(
                    tuple(shape), np.dtype(dtype), self.device_name, self.device_id)

    def run(self, inputs: Mapping[str, np.ndarray | object], *, output_names: tuple[str, ...] | None = None):
        import onnxruntime as ort
        self.binding.clear_binding_inputs()
        self.binding.clear_binding_outputs()
        self._device_inputs.clear()
        for name, value in inputs.items():
            if isinstance(value, ort.OrtValue) and value.device_name().casefold() == self.device_name:
                tensor = value
            elif (self.device_name == "cuda" and getattr(value, "is_cuda", False)
                  and callable(getattr(value, "data_ptr", None))):
                dtype_names = {"torch.float16": np.float16, "torch.float32": np.float32,
                               "torch.float64": np.float64, "torch.int8": np.int8,
                               "torch.int32": np.int32, "torch.int64": np.int64,
                               "torch.uint8": np.uint8, "torch.bool": np.bool_}
                try:
                    element_type = np.dtype(dtype_names[str(value.dtype)])
                except KeyError:
                    raise TypeError(f"Unsupported CUDA tensor dtype for I/O binding: {value.dtype}") from None
                device_id = value.device.index if value.device.index is not None else self.device_id
                self.binding.bind_input(name, "cuda", device_id, element_type,
                                        tuple(value.shape), value.data_ptr())
                self._device_inputs[name] = value
                continue
            else:
                if getattr(value, "is_cuda", False) and callable(getattr(value, "detach", None)):
                    value = value.detach().cpu().numpy()
                array = np.ascontiguousarray(value)
                if self.device_name == "cpu":
                    self.binding.bind_cpu_input(name, array)
                    self._device_inputs[name] = array
                    continue
                key = (array.shape, array.dtype.str)
                cached = self._inputs.get(name)
                if cached is None or cached[:2] != key:
                    tensor = ort.OrtValue.ortvalue_from_numpy(array, "cuda", self.device_id)
                    self._inputs[name] = (key[0], key[1], tensor)
                else:
                    tensor = cached[2]
                    tensor.update_inplace(array)
            self.binding.bind_ortvalue_input(name, tensor)
        names = output_names or tuple(node.name for node in self.session.get_outputs())
        for name in names:
            if name in self._outputs:
                self.binding.bind_ortvalue_output(name, self._outputs[name])
            else:
                self.binding.bind_output(name, self.device_name, self.device_id)
        self.session.run_with_iobinding(self.binding)
        if self.device_name == "cuda":
            self.binding.synchronize_outputs()
        return dict(zip(names, self.binding.get_outputs()))

    @staticmethod
    def synchronize(stream=None):
        if stream is not None:
            stream.synchronize()
