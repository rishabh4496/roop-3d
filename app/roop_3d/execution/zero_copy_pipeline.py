"""Reusable device buffers and ORT I/O binding for frame-resident GPU stages.

The ring owns each frame's model outputs until its final CUDA consumer records a
completion event. ORT-to-ORT handoffs use persistent OrtValues; outputs consumed
by PyTorch kernels bind directly into persistent CUDA tensors. Host input is
accepted only when the caller explicitly marks it as ingress. Normal stage
handoffs never materialize a CUDA tensor as a NumPy array.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

GPU_PIPELINE_STAGES = (
    "detection",
    "craniofacial_tracking_lbs",
    "silhouette_overhang_inpaint_cage_warp",
    "relit_swap",
    "enhancer",
    "laplacian_blend_4band",
)


class GPUExecutionFallback(RuntimeError):
    """The provider left CUDA during a frame; the owning job should switch to its CPU route."""


@dataclass(frozen=True)
class OutputBufferSpec:
    shape: tuple[int, ...]
    dtype: Any = np.float32
    storage: str = "ortvalue"  # "ortvalue" for ORT handoffs or "torch" for Torch consumers

    def __post_init__(self):
        if not self.shape or any(int(dim) <= 0 for dim in self.shape):
            raise ValueError("GPU output buffers require a concrete, positive shape")
        if self.storage not in ("ortvalue", "torch"):
            raise ValueError("Output storage must be 'ortvalue' or 'torch'")


def _torch_module(value=None):
    if value is not None:
        return value
    import torch
    return torch


def _numpy_dtype(value) -> np.dtype:
    if isinstance(value, str) and value.startswith("tensor("):
        names = {
            "tensor(float)": np.float32, "tensor(float16)": np.float16,
            "tensor(double)": np.float64, "tensor(int64)": np.int64,
            "tensor(int32)": np.int32, "tensor(int16)": np.int16,
            "tensor(int8)": np.int8, "tensor(uint8)": np.uint8,
            "tensor(bool)": np.bool_,
        }
        if value not in names:
            raise TypeError(f"Unsupported ONNX output type: {value}")
        value = names[value]
    return np.dtype(value)


def _output_spec(value) -> OutputBufferSpec:
    if isinstance(value, OutputBufferSpec):
        return value
    if isinstance(value, tuple) and len(value) in (2, 3):
        return OutputBufferSpec(tuple(value[0]), value[1], value[2] if len(value) == 3 else "ortvalue")
    raise TypeError("Output buffers must be OutputBufferSpec or (shape, dtype[, storage])")


def _tree_is_cuda(value, torch) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.is_cuda)
    if isinstance(value, np.ndarray):
        return False
    if isinstance(value, Mapping):
        return all(_tree_is_cuda(child, torch) for child in value.values())
    if isinstance(value, (tuple, list)):
        return all(_tree_is_cuda(child, torch) for child in value)
    if isinstance(value, (bytes, bytearray, memoryview, str, int, float, bool, type(None))):
        return True  # compact metadata / encoded packet, not a pixel tensor
    device_name = getattr(value, "device_name", None)
    if callable(device_name):
        try:
            return str(device_name()).casefold() in ("cuda", "gpu")
        except Exception:
            return False
    return True


class _Slot:
    def __init__(self, index: int):
        self.index = index
        self.buffers: dict[str, dict[str, Any]] = {}
        self.bindings: dict[str, Any] = {}
        self.retire_event = None


class ZeroCopyGPUExecutionRing:
    """Bounded GPU ring with reusable ORT bindings, OrtValues, and Torch buffers.

    Register ONNX stages once with :meth:`register_ort_stage`; each frame lease
    can then run those stages and the in-between CUDA/Torch stages in the declared
    order. Pixel data stays on ``cuda:<device_id>`` throughout the normal path.
    """

    def __init__(self, slots: int = 4, *, device_id: int = 0,
                 stage_order: Sequence[str] = GPU_PIPELINE_STAGES,
                 torch_module=None, ort_module=None, require_cuda: bool = True,
                 stream=None, shared_compute_stream: bool = False):
        if not 2 <= slots <= 32 or device_id < 0:
            raise ValueError("GPU execution ring requires 2..32 slots and a nonnegative device id")
        if not stage_order or len(set(stage_order)) != len(stage_order):
            raise ValueError("GPU stage order must contain unique names")
        self.torch = _torch_module(torch_module)
        if require_cuda and not self.torch.cuda.is_available():
            raise RuntimeError("Zero-copy GPU execution requires a CUDA device")
        if ort_module is None:
            import onnxruntime as ort_module
        self.ort = ort_module
        self.device_id = device_id
        self.device = f"cuda:{device_id}"
        self.stage_order = tuple(stage_order)
        if shared_compute_stream and stream is None:
            raise ValueError("A shared ORT/Torch compute stream must be supplied explicitly")
        self.stream = stream if stream is not None else (
            self.torch.cuda.Stream(device=self.device_id) if require_cuda else None
        )
        self.shared_compute_stream = shared_compute_stream
        self._slots = [_Slot(index) for index in range(slots)]
        self._free = deque(range(slots))
        self._condition = threading.Condition(threading.Lock())
        self._leased: set[int] = set()
        self._stages: dict[str, Any] = {}
        self._closed = False
        self.host_fallback_copies = 0
        self.intermediate_host_copies = 0

    def register_ort_stage(self, name: str, session: Any,
                           output_buffers: Mapping[str, OutputBufferSpec | tuple], *,
                           allow_host_ingress: bool = False) -> None:
        if self._closed or name not in self.stage_order:
            raise ValueError(f"Unknown or closed GPU stage: {name!r}")
        if name in self._stages:
            raise ValueError(f"GPU stage {name!r} is already registered")
        session_stream = getattr(session, "user_compute_stream", None)
        ring_stream = getattr(self.stream, "cuda_stream", None)
        if (self.shared_compute_stream and session_stream is not None
                and int(session_stream) != int(ring_stream)):
            raise ValueError(f"ORT stage {name!r} uses a different CUDA stream than the GPU ring")
        output_order = [output.name for output in session.get_outputs()]
        normalized = {key: _output_spec(output_buffers[key]) for key in output_order if key in output_buffers}
        expected = set(output_order)
        if set(normalized) != expected:
            raise ValueError(f"Output buffer names for {name!r} must match the ONNX outputs; "
                             f"missing={sorted(expected - set(normalized))}, "
                             f"extra={sorted(set(normalized) - expected)}")
        for slot in self._slots:
            buffers = {}
            for output_name, spec in normalized.items():
                dtype = _numpy_dtype(spec.dtype)
                if spec.storage == "ortvalue":
                    value = self.ort.OrtValue.ortvalue_from_shape_and_type(
                        spec.shape, dtype, "cuda", self.device_id)
                else:
                    torch_dtype = self._torch_dtype(dtype)
                    value = self.torch.empty(spec.shape, dtype=torch_dtype, device=self.device)
                buffers[output_name] = {"spec": spec, "value": value}
            slot.buffers[name] = buffers
            slot.bindings[name] = session.io_binding()
        self._stages[name] = {"session": session, "outputs": normalized,
                              "allow_host_ingress": allow_host_ingress}

    def _torch_dtype(self, dtype: np.dtype):
        mapping = {
            np.dtype(np.float16): self.torch.float16, np.dtype(np.float32): self.torch.float32,
            np.dtype(np.float64): self.torch.float64, np.dtype(np.uint8): self.torch.uint8,
            np.dtype(np.int8): self.torch.int8, np.dtype(np.int16): self.torch.int16,
            np.dtype(np.int32): self.torch.int32, np.dtype(np.int64): self.torch.int64,
            np.dtype(np.bool_): self.torch.bool,
        }
        try:
            return mapping[dtype]
        except KeyError:
            raise TypeError(f"Unsupported Torch output dtype: {dtype}") from None

    def acquire(self, *, timeout: float | None = None) -> "GPUExecutionLease":
        with self._condition:
            if self._closed:
                raise RuntimeError("GPU execution ring is closed")
            deadline = None if timeout is None else time.monotonic() + timeout
            while not self._free:
                self._reclaim()
                if self._free or self._closed:
                    break
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("Timed out waiting for a free GPU execution slot")
                self._condition.wait(0.01 if remaining is None else min(0.01, remaining))
            if self._closed:
                raise RuntimeError("GPU execution ring is closed")
            if not self._free:
                raise TimeoutError("Timed out waiting for a free GPU execution slot")
            index = self._free.popleft()
            self._leased.add(index)
            return GPUExecutionLease(self, self._slots[index])

    def _release(self, slot: _Slot, event=None) -> None:
        with self._condition:
            if slot.index not in self._leased:
                raise ValueError("GPU execution slot is not leased")
            self._leased.remove(slot.index)
            slot.retire_event = event
            if event is None:
                self._free.append(slot.index)
            else:
                self._reclaim()
            self._condition.notify_all()

    def _reclaim(self):
        for slot in self._slots:
            if slot.retire_event is not None and slot.retire_event.query():
                slot.retire_event = None
                self._free.append(slot.index)

    def close(self) -> None:
        with self._condition:
            if self._leased:
                raise RuntimeError("Cannot close the GPU execution ring while slots are leased")
            events = tuple(slot.retire_event for slot in self._slots if slot.retire_event is not None)
            self._closed = True
            self._condition.notify_all()
        for event in events:
            event.synchronize()
        if self.stream is not None:
            self.stream.synchronize()
        self._stages.clear()
        for slot in self._slots:
            slot.buffers.clear()
            slot.bindings.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class GPUExecutionLease:
    def __init__(self, ring: ZeroCopyGPUExecutionRing, slot: _Slot):
        self.ring, self.slot = ring, slot
        self._stage_index = 0
        self._released = False

    @property
    def index(self) -> int:
        return self.slot.index

    def _enter_stage(self, name: str):
        if self._released:
            raise RuntimeError("GPU frame lease has been released")
        if self._stage_index >= len(self.ring.stage_order) or self.ring.stage_order[self._stage_index] != name:
            expected = self.ring.stage_order[self._stage_index] if self._stage_index < len(self.ring.stage_order) else "<end>"
            raise ValueError(f"GPU stage order violation: expected {expected!r}, got {name!r}")
        self._stage_index += 1

    def run_torch_stage(self, name: str, function: Callable[..., Any], *args, **kwargs):
        """Run an in-place or functional CUDA stage without staging pixels through host memory."""
        self._enter_stage(name)
        with self.ring.torch.cuda.stream(self.ring.stream):
            result = function(*args, **kwargs)
            if not _tree_is_cuda(result, self.ring.torch):
                raise ValueError(f"GPU stage {name!r} returned a host tensor/array")
        return result

    def run_ort_stage(self, name: str, inputs: Mapping[str, Any]):
        """Run an ORT stage into this slot's persistent device outputs."""
        self._enter_stage(name)
        try:
            spec = self.ring._stages[name]
        except KeyError:
            raise ValueError(f"No ONNX session is registered for GPU stage {name!r}") from None
        session = spec["session"]

        binding = self.slot.bindings[name]
        binding.clear_binding_inputs()
        binding.clear_binding_outputs()
        for input_name, value in inputs.items():
            if getattr(value, "is_cuda", False) and callable(getattr(value, "data_ptr", None)):
                if not value.is_contiguous():
                    raise ValueError(f"ORT input {input_name!r} must be contiguous to bind its CUDA pointer")
                device_index = value.device.index if value.device.index is not None else self.ring.device_id
                if int(device_index) != self.ring.device_id:
                    raise ValueError(f"ORT input {input_name!r} is on cuda:{device_index}, expected {self.ring.device}")
                tensor_bind = getattr(binding, "bind_input_tensor", None)
                if callable(tensor_bind):
                    tensor_bind(input_name, value)
                else:
                    self._bind_tensor(binding, input_name, value)
            elif callable(getattr(value, "device_name", None)):
                device_name = str(value.device_name()).casefold()
                if device_name != "cuda" and not spec["allow_host_ingress"]:
                    raise ValueError(f"ORT input {input_name!r} is not on CUDA")
                if device_name != "cuda":
                    binding.bind_cpu_input(input_name, value.numpy())
                else:
                    binding.bind_ortvalue_input(input_name, value)
            elif isinstance(value, np.ndarray) and spec["allow_host_ingress"]:
                bind_cpu = getattr(binding, "bind_cpu_input", None)
                if not callable(bind_cpu):
                    raise TypeError("This ORT binding cannot accept a host ingress tensor")
                bind_cpu(input_name, value)
            else:
                raise ValueError(f"Intermediate input {input_name!r} must be a CUDA tensor or CUDA OrtValue")

        buffers = self.slot.buffers[name]
        for output_name, output in buffers.items():
            spec_out, value = output["spec"], output["value"]
            if spec_out.storage == "ortvalue":
                binding.bind_ortvalue_output(output_name, value)
            else:
                dtype = _numpy_dtype(spec_out.dtype)
                binding.bind_output(output_name, "cuda", self.ring.device_id,
                                    dtype, spec_out.shape, int(value.data_ptr()))
        session.run_with_iobinding(binding)
        active_provider = getattr(session, "provider", None)
        if not self.ring.shared_compute_stream or active_provider == "CPUExecutionProvider":
            binding.synchronize_outputs()

        outputs = {}
        values = binding.get_outputs()
        # The resilient session can replay a failed CUDA binding on CPU. Copy
        # only that explicit degraded result back into the persistent CUDA slot.
        if len(values) == len(buffers) and any(
                callable(getattr(value, "device_name", None)) and
                str(value.device_name()).casefold() == "cpu" for value in values):
            for (output_name, output), value in zip(buffers.items(), values):
                array = np.ascontiguousarray(value.numpy())
                self.ring.host_fallback_copies += 1
                destination = output["value"]
                if output["spec"].storage == "torch":
                    host = self.ring.torch.from_numpy(array)
                    destination.copy_(host.to(device=self.ring.device, non_blocking=True))
                else:
                    destination.update_inplace(array)
                outputs[output_name] = destination
        else:
            for output_name, output in buffers.items():
                outputs[output_name] = output["value"]
        return outputs

    def _bind_tensor(self, binding, name: str, value: Any):
        dtype_names = {
            "torch.float16": np.float16, "torch.float32": np.float32, "torch.float64": np.float64,
            "torch.int8": np.int8, "torch.uint8": np.uint8, "torch.int16": np.int16,
            "torch.int32": np.int32, "torch.int64": np.int64, "torch.bool": np.bool_,
        }
        try:
            dtype = np.dtype(dtype_names[str(value.dtype)])
        except KeyError:
            raise TypeError(f"Unsupported CUDA input dtype: {value.dtype}") from None
        device_id = value.device.index if value.device.index is not None else self.ring.device_id
        binding.bind_input(name, "cuda", device_id, dtype, tuple(value.shape), int(value.data_ptr()))

    def torch_view(self, value):
        """Return a zero-copy Torch view when this ORT build exposes CUDA DLPack."""
        to_dlpack = getattr(value, "to_dlpack", None)
        if not callable(to_dlpack):
            raise RuntimeError("This ONNX Runtime build has no CUDA OrtValue DLPack export; "
                               "bind that stage's output directly into storage='torch'")
        return self.ring.torch.utils.dlpack.from_dlpack(to_dlpack())

    def release(self, *, stream=None):
        if self._released:
            raise ValueError("GPU frame lease released twice")
        self._released = True
        event = None
        if self.ring.stream is not None:
            event = self.ring.torch.cuda.Event()
            event.record(stream or self.ring.stream)
        self.ring._release(self.slot, event)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()


__all__ = ["GPUExecutionFallback", "GPU_PIPELINE_STAGES", "OutputBufferSpec", "ZeroCopyGPUExecutionRing"]
