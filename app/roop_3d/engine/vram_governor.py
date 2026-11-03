"""Live VRAM admission and process-local disposable allocator cleanup.

The 4096 MiB per-stream reservation is an admission policy, not a measured
model requirement. OOM recovery remains necessary because provider allocations
and concurrent desktop workloads vary between machines.
"""
from __future__ import annotations

from dataclasses import dataclass
import gc
import sys
import threading
from typing import Callable, Literal

from roop_3d.core.hardware import Hardware, probe_hardware, resolve_hardware_profile
from roop_3d.core.settings import Settings

RuntimeProfile = Literal["cpu", "6gb", "12gb"]


@dataclass(frozen=True)
class Admission:
    profile: RuntimeProfile
    provider: Literal["CPUExecutionProvider", "CUDAExecutionProvider"]
    free_vram_mib: int
    reserved_vram_mib: int
    reason: str


class VRAMLease:
    def __init__(self, owner: "VRAMGovernor", key: str, admission: Admission):
        self._owner, self._key = owner, key
        self.admission = admission
        self._released = False
        self._lock = threading.Lock()

    def release(self) -> None:
        with self._lock:
            if not self._released:
                self._released = True
                self._owner._release(self._key)

    def __enter__(self) -> "VRAMLease":
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


class VRAMGovernor:
    """Admit CPU jobs and bounded GPU streams from a fresh hardware snapshot."""

    PER_STREAM_RESERVATION_MIB = 4096

    def __init__(self, hardware_probe: Callable[[Settings], Hardware] = probe_hardware):
        self._probe = hardware_probe
        self._lock = threading.RLock()
        self._active: dict[str, Admission] = {}

    @staticmethod
    def _cpu(reason: str) -> Admission:
        return Admission("cpu", "CPUExecutionProvider", 0, 0, reason)

    def try_acquire(self, key: str, settings: Settings, *, force_profile: RuntimeProfile | None = None
                    ) -> VRAMLease | None:
        if not key:
            raise ValueError("Admission key must be nonempty")
        with self._lock:
            if key in self._active:
                raise ValueError("Admission key already holds a VRAM lease")
            if len(self._active) >= settings.max_concurrent_jobs:
                return None

            if force_profile == "cpu" or settings.hardware_profile == "cpu" or settings.preferred_provider == "CPUExecutionProvider":
                admission = self._cpu("CPU profile selected by job policy")
                self._active[key] = admission
                return VRAMLease(self, key, admission)

            hardware = self._probe(settings)
            if not hardware.gpus:
                admission = self._cpu("No local GPU was verified; CPU reference selected")
                self._active[key] = admission
                return VRAMLease(self, key, admission)

            gpu = hardware.gpus[0]
            free = max(0, int(gpu.effective_vram_budget_mib))
            active_gpu = [value for value in self._active.values() if value.provider == "CUDAExecutionProvider"]
            active_reserved = sum(value.reserved_vram_mib for value in active_gpu)
            usable = max(0, min(free, int(gpu.total_vram_mib) - active_reserved))

            requested = force_profile or settings.hardware_profile
            profile = resolve_hardware_profile(gpu.total_vram_mib, gpu.free_vram_mib, requested)
            if profile == "12gb" and gpu.total_vram_mib < 12288:
                profile = "6gb"

            if usable <= 0:
                if active_gpu:
                    return None
                admission = self._cpu("Free VRAM is below the safety reserve; CPU reference selected")
                self._active[key] = admission
                return VRAMLease(self, key, admission)

            if profile == "12gb" and usable < self.PER_STREAM_RESERVATION_MIB:
                if active_gpu:
                    return None
                admission = self._cpu("Live free VRAM is below the per-stream admission reserve; CPU reference selected")
                self._active[key] = admission
                return VRAMLease(self, key, admission)

            reservation = min(self.PER_STREAM_RESERVATION_MIB, usable)
            if profile == "6gb":
                capacity = 1
            else:
                capacity = min(settings.max_concurrent_jobs,
                               max(1, usable // self.PER_STREAM_RESERVATION_MIB))
            if len(active_gpu) >= capacity:
                return None

            preferred = settings.preferred_provider
            if preferred not in {"auto", "CUDAExecutionProvider", "TensorrtExecutionProvider"}:
                admission = self._cpu("The configured provider is not supported by the reference path")
            elif not hardware.advertised_providers or "CUDAExecutionProvider" not in hardware.advertised_providers:
                admission = self._cpu("CUDAExecutionProvider is unavailable; CPU reference selected")
            else:
                admission = Admission(profile, "CUDAExecutionProvider", usable, reservation,
                                      "Live free VRAM admitted the reference stream")
            self._active[key] = admission
            return VRAMLease(self, key, admission)

    def _release(self, key: str) -> None:
        with self._lock:
            self._active.pop(key, None)

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    @staticmethod
    def evict_disposable() -> dict[str, str | bool]:
        """Collect Python objects and empty an already-loaded CUDA allocator.

        This does not import torch, remove model artifacts, or delete files.
        Inference owners must release their sessions before calling it.
        """
        gc.collect()
        try:
            from roop_3d.engine.adaptive_runtime import AdaptiveModelPool
            pool = AdaptiveModelPool.get_instance()
            if pool.profile.profile_type == "6gb":
                pool.release_heavy()
        except Exception:
            pass
        torch = sys.modules.get("torch")
        state = "NOT_LOADED"
        if torch is not None:
            try:
                cuda = torch.cuda
                if cuda.is_available():
                    cuda.synchronize()
                    cuda.empty_cache()
                    state = "EMPTIED"
                else:
                    state = "CUDA_UNAVAILABLE"
            except (AttributeError, RuntimeError, OSError):
                state = "CLEANUP_FAILED"
        return {"python_gc": True, "torch_allocator": state}


def is_cuda_oom(error: BaseException) -> bool:
    """Recognize known CUDA/allocator OOM signals without classifying all errors."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, MemoryError):
            return True
        name = f"{type(current).__module__}.{type(current).__name__}".casefold()
        message = str(current).casefold()
        if "outofmemory" in name or any(token in message for token in (
                "cuda out of memory", "cuda_error_out_of_memory", "cudaerrormemoryallocation",
                "cuda error: out of memory", "cudnn_status_alloc_failed", "bfc arena allocation",
                "bfcarena::allocaterawinternal", "failed to allocate memory", "memory allocation failed",
                "resource exhausted: oom")):
            return True
        current = current.__cause__ or current.__context__
    return False
