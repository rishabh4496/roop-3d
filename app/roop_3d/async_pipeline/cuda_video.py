"""Explicit device-resident frame path over the shared multi-stream scheduler."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Callable

import numpy as np

from roop_3d.async_pipeline.buffers import CudaFrameRing
from roop_3d.async_pipeline.stages import AsyncPipeline, Item

CUDA_STAGE_ORDER = ("track", "rasterize", "swap", "restore", "blend", "encode")


def _frame_and_index(value, fallback_index: int):
    """Accept host decoded frames and explicit CUDA decoder leases without downloading pixels."""
    metadata = None
    if isinstance(value, tuple) and value and hasattr(value[0], "tensor"):
        value, metadata = value[0], value[1:]
    if hasattr(value, "tensor"):
        return value.tensor, int(getattr(value, "index", fallback_index)), (value, metadata)
    if hasattr(value, "image"):
        return value.image, int(getattr(value, "index", fallback_index)), value
    return value, fallback_index, None


class CudaFrameSource:
    """Convert a pull source into ring-backed CUDA ``Item`` objects."""

    def __init__(self, source: Iterable, ring: CudaFrameRing, *, copy_cuda: bool = True):
        self.source, self.ring, self.copy_cuda = source, ring, copy_cuda

    def __iter__(self):
        try:
            for fallback_index, value in enumerate(self.source):
                lease = self.ring.acquire()
                try:
                    frame, index, owner = _frame_and_index(value, fallback_index)
                    if isinstance(frame, np.ndarray):
                        tensor = lease.stage_host(frame)
                    else:
                        tensor = lease.stage_cuda(frame, copy=self.copy_cuda)
                        if owner is not None:
                            lease.keepalive(owner)
                    item = Item(index=index, payload=tensor, event=lease.ready_event)
                    item.meta["device_frame"] = True
                    if owner is not None:
                        item.meta["source_metadata"] = owner
                    item.cleanups.append(lambda item=item, lease=lease: (
                        lease.release_after(item.event) if item.event is not None else lease.release()))
                    yield item
                except BaseException:
                    if not lease._released:
                        if lease.ready_event is not None:
                            lease.release_after(lease.ready_event)
                        else:
                            lease.release()
                    raise
        finally:
            close = getattr(self.source, "close", None)
            if callable(close):
                close()


def _assert_device_pixels(value: Any, torch, *, stage: str) -> None:
    """Fail if a CUDA stage returns raw image pixels to CPU memory."""
    if torch.is_tensor(value):
        if not value.is_cuda:
            raise ValueError(f"CUDA stage {stage!r} returned a host tensor")
    elif isinstance(value, np.ndarray):
        raise ValueError(f"CUDA stage {stage!r} returned a NumPy array")
    elif isinstance(value, Mapping):
        for child in value.values():
            _assert_device_pixels(child, torch, stage=stage)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _assert_device_pixels(child, torch, stage=stage)
    elif isinstance(value, (bytes, bytearray, memoryview, str, int, float, bool, type(None))):
        return  # Encoded packets and small CPU metadata are permitted.
    else:
        tensor = getattr(value, "tensor", None)
        if tensor is not None:
            _assert_device_pixels(tensor, torch, stage=stage)


class CudaFramePipeline:
    """Decode -> track -> rasterize -> swap -> restore -> blend -> encode on CUDA.

    Stage callables are explicit adapters for a model/runtime. Each must keep
    pixel tensors on CUDA; compressed output packets and scalar metadata may be
    host values. If CUDA is unavailable, callers use the existing NumPy
    ``AsyncPipeline``/reference route instead of silently changing residence.
    """

    def __init__(self, source: Iterable, stages: Mapping[str, Callable[[Item], Any]], *,
                 shape: tuple[int, int, int], ring_slots: int = 4, queue_size: int = 4,
                 vram_budget_mb: int = 0, device: int = 0, copy_cuda_frames: bool = True):
        missing = [name for name in CUDA_STAGE_ORDER if name not in stages]
        extra = set(stages) - set(CUDA_STAGE_ORDER)
        if missing or extra:
            raise ValueError(f"CUDA pipeline stages must be exactly {CUDA_STAGE_ORDER}; missing={missing}, extra={sorted(extra)}")
        if vram_budget_mb < 0:
            raise ValueError("vram_budget_mb must be nonnegative")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA pipeline is unavailable; use the CPU reference pipeline")
        frame_bytes = int(np.prod(shape))
        # Host and device copies together stay below 5% of the configured job
        # budget.  Four 4K BGR slots use about 200 MiB across both memories.
        if vram_budget_mb:
            allowed_slots = max(2, (vram_budget_mb * 1024 * 1024 // 20) // (2 * frame_bytes))
            ring_slots = min(ring_slots, allowed_slots)
        self.ring = CudaFrameRing(shape, ring_slots, device=device)
        self.pipeline = AsyncPipeline("cuda_render", None, queue_size=queue_size, device=device)
        self.pipeline.add_pool(self.ring)
        self.pipeline.set_source(CudaFrameSource(source, self.ring, copy_cuda=copy_cuda_frames))
        for name in CUDA_STAGE_ORDER:
            function = stages[name]
            def run(item, function=function, name=name):
                item.payload = function(item)
                _assert_device_pixels(item.payload, torch, stage=name)
                return item.payload
            self.pipeline.add_stage(name, run, stream=True)

    def results(self):
        for item in self.pipeline.results():
            try:
                yield item
            finally:
                item.release()

    def snapshot(self) -> dict:
        return self.pipeline.snapshot()

    def __enter__(self):
        self.pipeline.start()
        return self

    def __exit__(self, *exc):
        try:
            self.pipeline.__exit__(*exc)
        finally:
            self.ring.close()


__all__ = ["CUDA_STAGE_ORDER", "CudaFramePipeline", "CudaFrameSource"]
