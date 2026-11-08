"""Explicitly requested strict TRT inference; constructor never loads a model."""
from __future__ import annotations

from pathlib import Path
import copy
import threading

from roop_3d.models.dynamic_shapes import assert_finite
from roop_3d.trt_session_builder import BatchProfile, TensorRTSessionBuilder
from roop_3d.engine.resources import guarded, inference


class OptimizedTRTEngine:
    def __init__(self, model: str | Path, *, builder: TensorRTSessionBuilder | None = None,
                 profile: BatchProfile | None = None, shapes=None):
        self.model, self.builder = Path(model), builder or TensorRTSessionBuilder()
        self.profile, self.shapes = profile, shapes
        self._session = None
        self._io_runner = None
        self._build = None
        self._lock = threading.RLock()

    @guarded
    def prepare(self):
        with self._lock:
            if self._session is None:
                self._session = self.builder.build(self.model, profile=self.profile, shapes=self.shapes)
                from roop_3d.core.ort_session import GpuIoBinding
                self._io_runner = GpuIoBinding(self._session)
                self._build = copy.deepcopy(self.builder.last_build)
            return self._session

    @inference
    def run(self, feeds):
        with self._lock:
            session = self.prepare()
            # Reject out-of-profile requests instead of silently triggering another build.
            batches = self._build["checked_batches"]
            for name, value in feeds.items():
                if name not in ("w", "fidelity_weight") and value.ndim and value.shape[0] not in batches:
                    raise ValueError("Batch must be one of the executed canary batches")
                limits = self._build["profiles"]
                if name not in limits["min"] or len(value.shape) != len(limits["min"][name]) or any(
                    not lo <= n <= hi for lo, n, hi in zip(limits["min"][name], value.shape, limits["max"][name])):
                    raise ValueError(f"Input {name} is outside the compiled shape profile")
            outputs_by_name = self._io_runner.run(feeds)
            outputs = [outputs_by_name[node.name].numpy() for node in session.get_outputs()]
            assert_finite(outputs)
            return outputs

    @inference
    def run_cuda(self, feeds, outputs):
        """Bind caller-owned CUDA tensors directly; never session.run/NumPy pixels.

        Output names, shapes and dtypes must be the model's actual contract.
        Synchronization also supports existing sessions on their own ORT stream.
        """
        import numpy as np
        import torch
        with self._lock:
            session = self.prepare()
            if set(feeds) != {v.name for v in session.get_inputs()} or set(outputs) != {v.name for v in session.get_outputs()}:
                raise ValueError("CUDA binding names must match the full session contract")
            tensors = list(feeds.values()) + list(outputs.values())
            if not tensors or any(not v.is_cuda or not v.is_contiguous() for v in tensors):
                raise ValueError("Bindings require contiguous CUDA tensors")
            device = tensors[0].device
            if any(v.device != device for v in tensors) or device.index != 0:
                raise ValueError("Session bindings must use CUDA device 0")
            limits = self._build["profiles"]
            for name, value in feeds.items():
                if name not in limits["min"] or len(value.shape) != len(limits["min"][name]) or any(
                    not lo <= n <= hi for lo, n, hi in zip(limits["min"][name], value.shape, limits["max"][name])):
                    raise ValueError(f"Input {name} is outside the compiled shape profile")
                if name not in ("w", "fidelity_weight") and value.ndim and value.shape[0] not in self._build["checked_batches"]:
                    raise ValueError("Batch must be one of the executed canary batches")
            types = {torch.float32: np.float32, torch.float16: np.float16, torch.int64: np.int64,
                     torch.int32: np.int32, torch.uint8: np.uint8}
            binding = session.io_binding()
            for values, bind in ((feeds, binding.bind_input), (outputs, binding.bind_output)):
                for name, value in values.items():
                    if value.dtype not in types:
                        raise ValueError("Unsupported CUDA binding dtype")
                    bind(name, "cuda", device.index, types[value.dtype], tuple(value.shape), value.data_ptr())
            torch.cuda.current_stream(device).synchronize()
            session.run_with_iobinding(binding)
            binding.synchronize_outputs()
            return outputs

    @guarded
    def close(self):
        with self._lock:
            self._session = None
            self._build = None
