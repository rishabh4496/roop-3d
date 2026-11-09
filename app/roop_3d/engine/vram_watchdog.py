"""Dynamic VRAM OOM Watchdog & Micro-Batch Auto-Stepdown Engine.

Monitors CUDA memory allocation (torch.cuda.mem_get_info / NVML) across single-GPU
and dual-GPU setups.
If a 4K frame with multiple faces + UltraMax 1024 + Real-ESRGAN x4 approaches 92% VRAM
capacity or triggers CUDA out of memory, catches the condition inside the worker
without crashing the job:
1. Flushes the TensorRT workspace cache and empties device allocators.
2. Automatically steps down the micro-batch size (B=8 -> B=4 -> B=2 -> B=1).
3. Automatically steps down the tile size (512 -> 256).
4. Records an advisory badge for UI telemetry.
"""
from __future__ import annotations

import ctypes
import gc
import logging
import os
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

from roop_3d.engine.vram_governor import VRAMGovernor, is_cuda_oom

log = logging.getLogger("roop3d.vram_watchdog")

VRAM_OOM_THRESHOLD_PCT = 92.0
MICRO_BATCH_STEPS = (8, 4, 2, 1)
TILE_SIZE_STEPS = (512, 256)


class VRAMWatchdog:
    """Thread-safe dynamic VRAM watchdog and micro-batch auto-stepdown manager."""

    _instance: Optional["VRAMWatchdog"] = None
    _lock = threading.RLock()

    def __init__(self, threshold_pct: float = VRAM_OOM_THRESHOLD_PCT):
        self.threshold_pct = threshold_pct
        self.current_micro_batch: int = 8
        self.current_tile_size: int = 512
        self.advisory_badges: List[Dict[str, Any]] = []
        self.stepdown_count: int = 0
        self._trt_cache_flushed: bool = False
        self._nvml_initialized: Optional[bool] = None

    @classmethod
    def get_instance(cls) -> "VRAMWatchdog":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> "VRAMWatchdog":
        with cls._lock:
            cls._instance = cls()
            return cls._instance

    def _query_torch_cuda(self) -> List[Dict[str, Any]]:
        """Query VRAM across single-GPU and multi-GPU setups using torch.cuda."""
        torch = sys.modules.get("torch")
        if torch is None:
            try:
                import torch as _torch
                torch = _torch
            except ImportError:
                return []

        if not hasattr(torch, "cuda") or not torch.cuda.is_available():
            return []

        gpus = []
        try:
            device_count = torch.cuda.device_count()
            for idx in range(device_count):
                free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
                total_mb = total_bytes / (1024 * 1024)
                free_mb = free_bytes / (1024 * 1024)
                used_mb = total_mb - free_mb
                used_pct = (used_mb / total_mb * 100.0) if total_mb > 0 else 0.0
                gpus.append({
                    "device_index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "total_mb": round(total_mb, 1),
                    "free_mb": round(free_mb, 1),
                    "used_mb": round(used_mb, 1),
                    "used_pct": round(used_pct, 2),
                    "source": "torch.cuda.mem_get_info",
                })
        except Exception as exc:
            log.debug("torch.cuda VRAM probe error: %s", exc)
        return gpus

    def _query_nvml(self) -> List[Dict[str, Any]]:
        """Fallback/supplementary NVML probe via ctypes."""
        if self._nvml_initialized is False:
            return []

        try:
            # Try loading nvml / nvcuda dll on Windows
            nvml_dll = ctypes.windll.LoadLibrary("nvml.dll") if sys.platform == "win32" else ctypes.CDLL("libnvidia-ml.so.1")
            if self._nvml_initialized is None:
                init_res = nvml_dll.nvmlInit_v2()
                self._nvml_initialized = (init_res == 0)
                if not self._nvml_initialized:
                    return []

            device_count = ctypes.c_uint()
            if nvml_dll.nvmlDeviceGetCount_v2(ctypes.byref(device_count)) != 0:
                return []

            class NvmlMemory(ctypes.Structure):
                _fields_ = [
                    ("total", ctypes.c_ulonglong),
                    ("free", ctypes.c_ulonglong),
                    ("used", ctypes.c_ulonglong),
                ]

            gpus = []
            for i in range(device_count.value):
                handle = ctypes.c_void_p()
                if nvml_dll.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(i), ctypes.byref(handle)) == 0:
                    mem = NvmlMemory()
                    if nvml_dll.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem)) == 0:
                        total_mb = mem.total / (1024 * 1024)
                        free_mb = mem.free / (1024 * 1024)
                        used_mb = mem.used / (1024 * 1024)
                        used_pct = (used_mb / total_mb * 100.0) if total_mb > 0 else 0.0
                        gpus.append({
                            "device_index": i,
                            "name": f"NVIDIA GPU {i}",
                            "total_mb": round(total_mb, 1),
                            "free_mb": round(free_mb, 1),
                            "used_mb": round(used_mb, 1),
                            "used_pct": round(used_pct, 2),
                            "source": "nvml",
                        })
            return gpus
        except Exception as exc:
            self._nvml_initialized = False
            log.debug("NVML VRAM probe error: %s", exc)
            return []

    def get_vram_status(self) -> Dict[str, Any]:
        """Get live VRAM status across single-GPU and dual-GPU configurations."""
        with self._lock:
            gpus = self._query_torch_cuda()
            if not gpus:
                gpus = self._query_nvml()

            max_used_pct = max((g["used_pct"] for g in gpus), default=0.0)
            near_oom = max_used_pct >= self.threshold_pct

            return {
                "gpus": gpus,
                "gpu_count": len(gpus),
                "max_used_pct": max_used_pct,
                "near_oom": near_oom,
                "threshold_pct": self.threshold_pct,
                "micro_batch": self.current_micro_batch,
                "tile_size": self.current_tile_size,
                "advisories": list(self.advisory_badges),
            }

    def flush_trt_workspace_cache(self) -> Dict[str, Any]:
        """Flush TensorRT workspace cache and empty CUDA memory allocators."""
        with self._lock:
            flushed_details = {"evicted_disposable": False, "torch_cache_emptied": False}
            try:
                evict_res = VRAMGovernor.evict_disposable()
                flushed_details["evicted_disposable"] = True
                flushed_details["evict_details"] = evict_res
            except Exception as exc:
                log.debug("Governor eviction failed: %s", exc)

            torch = sys.modules.get("torch")
            if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
                    flushed_details["torch_cache_emptied"] = True
                except Exception as exc:
                    log.debug("torch.cuda empty_cache failed: %s", exc)

            # Flush TensorRT execution engine cache / runtime pool if loaded
            try:
                from roop_3d.engine.adaptive_runtime import AdaptiveModelPool
                pool = AdaptiveModelPool.get_instance()
                pool.release_heavy()
                flushed_details["trt_pool_heavy_released"] = True
            except Exception:
                pass

            gc.collect()
            self._trt_cache_flushed = True
            log.info("VRAMWatchdog: TensorRT workspace cache and allocator memory flushed.")
            return flushed_details

    def stepdown(self, reason: str = "VRAM pressure >= 92%") -> Dict[str, Any]:
        """Step down micro-batch size and tile size automatically."""
        with self._lock:
            # 1. Flush TRT and CUDA cache
            flush_res = self.flush_trt_workspace_cache()

            # 2. Step down micro-batch: B=8 -> B=4 -> B=2 -> B=1
            old_b = self.current_micro_batch
            for b in MICRO_BATCH_STEPS:
                if b < old_b:
                    self.current_micro_batch = b
                    break
            else:
                self.current_micro_batch = 1

            # 3. Step down tile size: 512 -> 256
            old_tile = self.current_tile_size
            for t in TILE_SIZE_STEPS:
                if t < old_tile:
                    self.current_tile_size = t
                    break
            else:
                self.current_tile_size = 256

            self.stepdown_count += 1
            badge = {
                "id": f"vram_stepdown_{self.stepdown_count}",
                "level": "advisory",
                "badge": f"⚡ Micro-batch stepped down to B={self.current_micro_batch}, Tile {self.current_tile_size}px (VRAM {self.threshold_pct}%+)",
                "reason": reason,
                "micro_batch": self.current_micro_batch,
                "tile_size": self.current_tile_size,
                "previous_micro_batch": old_b,
                "previous_tile_size": old_tile,
            }
            self.advisory_badges.append(badge)
            log.warning("VRAMWatchdog Stepdown: %s", badge["badge"])
            return {
                "badge": badge,
                "micro_batch": self.current_micro_batch,
                "tile_size": self.current_tile_size,
                "flush": flush_res,
            }

    def check_and_adapt(self) -> Optional[Dict[str, Any]]:
        """Check live VRAM; if >= 92% capacity, automatically trigger stepdown."""
        status = self.get_vram_status()
        if status["near_oom"]:
            return self.stepdown(reason=f"VRAM capacity reached {status['max_used_pct']}% >= {self.threshold_pct}%")
        return None

    def execute_guarded(self, fn, *args, **kwargs):
        """Execute a unit of inference work with watchdog protection.

        If VRAM approaches 92% or CUDA OOM is raised, flush cache, stepdown, and retry.
        """
        # Pre-execution check
        self.check_and_adapt()

        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if is_cuda_oom(exc) or "out of memory" in str(exc).lower():
                log.warning("Caught CUDA Out of Memory in guarded execution: %s", exc)
                stepdown_info = self.stepdown(reason="CUDA out of memory triggered")
                # Retry with stepped-down configuration
                return fn(*args, **kwargs)
            raise


# Global singleton access
watchdog = VRAMWatchdog.get_instance()
