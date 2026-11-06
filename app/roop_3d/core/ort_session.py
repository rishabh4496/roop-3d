"""TensorRT-first ONNX Runtime sessions with per-model dynamic profiles."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

TRT, CUDA, CPU = "TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"


def _image_size(name: str) -> tuple[int, int, int]:
    name = name.casefold()
    if "xseg" in name:
        return 3, 256, 256
    if any(x in name for x in ("gpen", "ultramax", "gfpgan", "codeformer")):
        return 3, 512, 512
    if any(x in name for x in ("scrfd", "retinaface")):
        return 3, 640, 640
    if any(x in name for x in ("landmark", "2dfan", "1k3d68", "106")):
        return 3, 192, 192
    return 3, 512, 512


def _profiles(nodes, name: str):
    channels, height, width = _image_size(name)
    result = [{}, {}, {}]
    for node in nodes:
        if not node.shape or not any(not isinstance(d, int) or d <= 0 for d in node.shape):
            continue
        dims = [[], [], []]
        for axis, dim in enumerate(node.shape):
            if isinstance(dim, int) and dim > 0:
                choices = (dim, dim, dim)
            elif axis == 0:
                choices = (1, 2, 4)
            elif len(node.shape) == 4 and axis == 1:
                choices = (channels,) * 3
            elif len(node.shape) == 4 and axis in (2, 3):
                size = height if axis == 2 else width
                choices = (max(32, size // 2), size, min(2048, size * 2))
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
                 inter_op_threads: int = 1, fp16: bool = True, gpu_id: int = 0):
        if not 1 <= intra_op_threads <= 16 or not 1 <= inter_op_threads <= 8 or gpu_id < 0:
            raise ValueError("Invalid ONNX Runtime thread or GPU settings")
        self.cache_path = Path(cache_path) if cache_path else Path(__file__).resolve().parents[2] / "models" / ".trt_cache"
        self.intra_op_threads, self.inter_op_threads = intra_op_threads, inter_op_threads
        self.fp16, self.gpu_id = fp16, gpu_id

    def _session_options(self, ort, threads: int):
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = threads, self.inter_op_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        return options

    def _providers(self, ort, path: Path, name: str, provider: str):
        available = set(ort.get_available_providers())
        if provider not in available:
            return []
        if provider == CPU:
            return [CPU]
        if provider == CUDA:
            return [(CUDA, {"device_id": self.gpu_id, "cudnn_conv_algo_search": "EXHAUSTIVE",
                            "do_copy_in_default_stream": True,
                            "arena_extend_strategy": "kNextPowerOfTwo"})]
        import onnx
        model = onnx.load(str(path), load_external_data=False)
        initializers = {v.name for v in model.graph.initializer}
        graph_inputs = [v for v in model.graph.input if v.name not in initializers]
        class Node:
            def __init__(self, value):
                self.name = value.name
                self.shape = [d.dim_value if d.dim_value > 0 else d.dim_param or None
                              for d in value.type.tensor_type.shape.dim]
        profiles = _profiles([Node(v) for v in graph_inputs], name)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        options = {"device_id": self.gpu_id, "trt_fp16_enable": self.fp16,
                   "trt_engine_cache_enable": True, "trt_engine_cache_path": str(self.cache_path),
                   "trt_timing_cache_enable": True, "trt_timing_cache_path": str(self.cache_path),
                   "trt_max_workspace_size": 2 * 1024**3, "trt_auxiliary_streams": 0}
        for key, profile in zip(("min", "opt", "max"), profiles):
            if profile:
                options[f"trt_profile_{key}_shapes"] = ",".join(
                    f"{tensor}:" + "x".join(map(str, shape)) for tensor, shape in profile.items())
        return [(TRT, options)]

    def create(self, model_path: str | Path, *, model_name: str | None = None,
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
        order = tuple(p for p in (TRT, CUDA, CPU) if p in providers)
        if not order:
            raise ValueError("At least one provider is required")
        errors = []
        for provider in order:
            try:
                provider_options = self._providers(ort, path, name, provider)
                if not provider_options:
                    errors.append(f"{provider}: unavailable")
                    continue
                session = ort.InferenceSession(str(path), sess_options=self._session_options(
                    ort, intra_op_threads or self.intra_op_threads), providers=provider_options)
                active = session.get_providers()
                if not active or active[0] != provider:
                    raise RuntimeError(f"provider activation mismatch: {active}")
                return SessionInfo(session, provider, order, tuple(errors))
            except Exception as error:
                errors.append(f"{provider}: {type(error).__name__}: {error}")
        raise RuntimeError("No requested ONNX Runtime provider could initialize: " + " | ".join(errors))


class GpuIoBinding:
    """A per-worker I/O binding that retains GPU input and output OrtValues."""
    def __init__(self, session, *, device_id: int = 0,
                 output_buffers: Mapping[str, tuple[tuple[int, ...], object]] | None = None):
        self.session, self.device_id = session, device_id
        self.binding = session.io_binding()
        self._inputs: dict[str, tuple[tuple[int, ...], str, object]] = {}
        self._outputs = {}
        if output_buffers:
            import onnxruntime as ort
            for name, (shape, dtype) in output_buffers.items():
                self._outputs[name] = ort.OrtValue.ortvalue_from_shape_and_type(
                    tuple(shape), np.dtype(dtype), "cuda", device_id)

    def run(self, inputs: Mapping[str, np.ndarray | object], *, output_names: tuple[str, ...] | None = None):
        import onnxruntime as ort
        self.binding.clear_binding_inputs()
        self.binding.clear_binding_outputs()
        for name, value in inputs.items():
            if isinstance(value, ort.OrtValue) and value.device_name().casefold() == "cuda":
                tensor = value
            else:
                array = np.ascontiguousarray(value)
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
                self.binding.bind_output(name, "cuda", self.device_id)
        self.session.run_with_iobinding(self.binding)
        return dict(zip(names, self.binding.get_outputs()))

    @staticmethod
    def synchronize(stream=None):
        if stream is not None:
            stream.synchronize()
