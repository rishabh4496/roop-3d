"""Explicit stream policy and per-thread CUDA primary-context activation."""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import ctypes.util
from dataclasses import dataclass
import os
import threading


@dataclass(frozen=True)
class RenderPolicy:
    batch_size: int = 4
    execution_threads: int = 1
    ring_slots: int = 4
    sparse_interval: int = 25
    range_padding: int = 25
    segment_frames: int = 250
    strict_nvdec: bool = False
    strict_nvenc: bool = False
    ultramax_native_bypass: bool = True
    ultramax_min_pixels: int = 320
    hardware_profile: str = "6gb"

    def __post_init__(self):
        if (not 1 <= self.batch_size <= 8 or not 1 <= self.execution_threads <= 8
                or not 2 <= self.ring_slots <= 32 or self.sparse_interval < 1 or self.range_padding < 0
                or not 1 <= self.segment_frames <= 10000 or self.ultramax_min_pixels < 1
                or self.hardware_profile not in ("cpu", "6gb", "12gb")):
            raise ValueError("Invalid stream render policy")

    @classmethod
    def from_env(cls, **overrides):
        return cls(**{**{"strict_nvdec": os.environ.get("ROOP_STRICT_NVDEC") == "1",
                         "strict_nvenc": os.environ.get("ROOP_STRICT_NVENC") == "1"}, **overrides})


class CudaWorkerContext:
    """Use the runtime's primary context on every executing thread, not a foreign context.

    Sessions and buffers stay owned by one worker. The lock prevents concurrent
    rebinding of that worker's I/O buffers when execution_threads > 1.
    """
    def __init__(self, device: int = 0, *, enabled: bool = True):
        self.device, self.enabled = device, enabled
        self._lock = threading.RLock()
        self._runtime = None

    @contextmanager
    def activate(self):
        with self._lock:
            if self.enabled:
                if self._runtime is None:
                    from roop_3d.audit.project_paths import application_root
                    from roop_3d.models.trt_builder import configure_native_paths
                    configure_native_paths(application_root())
                    from roop_3d.models.native_libraries import register_native_libraries, packaged_library_directories
                    register_native_libraries()
                    candidates = [p for d in packaged_library_directories() for p in d.glob(
                        "cudart64*.dll" if os.name == "nt" else "libcudart.so*")]
                    library = str(candidates[0]) if candidates else ("cudart64_12.dll" if os.name == "nt" else ctypes.util.find_library("cudart"))
                    if not library:
                        raise RuntimeError("CUDA runtime is unavailable; no worker context was created")
                    self._runtime = ctypes.CDLL(library)
                    self._runtime.cudaSetDevice.argtypes = [ctypes.c_int]
                    self._runtime.cudaSetDevice.restype = ctypes.c_int
                    self._runtime.cudaFree.argtypes = [ctypes.c_void_p]
                    self._runtime.cudaFree.restype = ctypes.c_int
                status = self._runtime.cudaSetDevice(self.device)
                if status == 0:
                    status = self._runtime.cudaFree(None)
                if status:
                    raise RuntimeError(f"CUDA worker primary-context activation failed ({status})")
            yield
