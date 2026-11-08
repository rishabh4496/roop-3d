"""Live GPU / CPU / PCIe utilisation without optional dependencies.

NVML ships with every NVIDIA driver (``nvml.dll`` / ``libnvidia-ml.so``), so it is reached through ``ctypes`` rather
than the ``pynvml`` wheel.  ``nvidia-smi`` is the fallback when NVML cannot be loaded.  CPU utilisation uses
``GetSystemTimes`` on Windows and ``/proc/stat`` elsewhere, so ``psutil`` is not required either.

Every number returned is a *measurement*; a value that the driver does not report is ``None``, never a default.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import threading
import time
from collections import deque
from typing import Any


class _NvmlUtilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _NvmlMemory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class Nvml:
    """Minimal NVML binding: utilisation, memory, encoder/decoder engines, PCIe throughput."""

    PCIE_TX, PCIE_RX = 0, 1

    def __init__(self, index: int = 0) -> None:
        self.index, self.lib, self.handle, self.error = index, None, None, None
        try:
            self._load()
        except Exception as error:   # driver missing / permission: degrade to nvidia-smi
            self.error = f"{type(error).__name__}: {error}"
            self.lib = None

    def _load(self) -> None:
        candidates = (["nvml.dll", os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvml.dll"),
                       r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll"]
                      if sys.platform == "win32" else ["libnvidia-ml.so.1", "libnvidia-ml.so"])
        last: Exception | None = None
        for name in candidates:
            try:
                lib = ctypes.CDLL(name)
                break
            except OSError as error:
                last = error
        else:
            raise last or OSError("NVML not found")
        init = getattr(lib, "nvmlInit_v2", None) or lib.nvmlInit
        if init() != 0:
            raise OSError("nvmlInit failed")
        handle = ctypes.c_void_p()
        getter = getattr(lib, "nvmlDeviceGetHandleByIndex_v2", None) or lib.nvmlDeviceGetHandleByIndex
        if getter(self.index, ctypes.byref(handle)) != 0:
            raise OSError("nvmlDeviceGetHandleByIndex failed")
        self.lib, self.handle = lib, handle

    @property
    def available(self) -> bool:
        return self.lib is not None

    def _call(self, name: str, *args) -> bool:
        function = getattr(self.lib, name, None)
        return function is not None and function(self.handle, *args) == 0

    def sample(self) -> dict[str, Any]:
        out: dict[str, Any] = {"gpu_util_pct": None, "mem_util_pct": None, "vram_used_mib": None, "vram_total_mib": None,
                               "encoder_util_pct": None, "decoder_util_pct": None,
                               "pcie_tx_mb_s": None, "pcie_rx_mb_s": None, "power_w": None, "source": "nvml"}
        util = _NvmlUtilization()
        if self._call("nvmlDeviceGetUtilizationRates", ctypes.byref(util)):
            out["gpu_util_pct"], out["mem_util_pct"] = util.gpu, util.memory
        mem = _NvmlMemory()
        if self._call("nvmlDeviceGetMemoryInfo", ctypes.byref(mem)):
            out["vram_used_mib"], out["vram_total_mib"] = mem.used // 1048576, mem.total // 1048576
        for key, name in (("encoder_util_pct", "nvmlDeviceGetEncoderUtilization"),
                          ("decoder_util_pct", "nvmlDeviceGetDecoderUtilization")):
            value, period = ctypes.c_uint(), ctypes.c_uint()
            if self._call(name, ctypes.byref(value), ctypes.byref(period)):
                out[key] = value.value
        for key, counter in (("pcie_tx_mb_s", self.PCIE_TX), ("pcie_rx_mb_s", self.PCIE_RX)):
            value = ctypes.c_uint()
            if self._call("nvmlDeviceGetPcieThroughput", ctypes.c_int(counter), ctypes.byref(value)):
                out[key] = round(value.value / 1024, 1)        # KB/s -> MB/s
        power = ctypes.c_uint()
        if self._call("nvmlDeviceGetPowerUsage", ctypes.byref(power)):
            out["power_w"] = round(power.value / 1000, 1)
        return out


def _smi_sample(index: int = 0) -> dict[str, Any]:
    keys = ["gpu_util_pct", "mem_util_pct", "vram_used_mib", "vram_total_mib", "power_w"]
    try:
        row = subprocess.check_output(
            ["nvidia-smi", f"--id={index}",
             "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
             "--format=csv,noheader,nounits"], text=True, timeout=3, stderr=subprocess.DEVNULL).strip().splitlines()[0]
        values = [None if item.strip() in ("", "[N/A]") else float(item) for item in row.split(",")]
        out = dict(zip(keys, values))
    except Exception:
        out = dict.fromkeys(keys)
    out.update(encoder_util_pct=None, decoder_util_pct=None, pcie_tx_mb_s=None, pcie_rx_mb_s=None, source="nvidia-smi")
    return out


class _FileTime(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]


def _cpu_times() -> tuple[float, float] | None:
    """(busy, total) cumulative system CPU time in arbitrary but consistent units."""
    try:
        if sys.platform == "win32":
            idle, kernel, user = _FileTime(), _FileTime(), _FileTime()
            if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
                return None
            value = lambda t: (t.high << 32) | t.low
            total = value(kernel) + value(user)        # kernel time includes idle
            return float(total - value(idle)), float(total)
        with open("/proc/stat", encoding="ascii") as handle:
            fields = [float(x) for x in handle.readline().split()[1:]]
        idle = fields[3] + (fields[4] if len(fields) > 4 else 0.0)
        return sum(fields) - idle, sum(fields)
    except Exception:
        return None


class HardwareMonitor:
    """Samples the GPU and CPU on a daemon thread; ``snapshot()`` returns the latest and a rolling summary."""

    def __init__(self, interval: float = 0.2, window: int = 150, index: int = 0) -> None:
        self.interval, self.index = interval, index
        self.nvml = Nvml(index)
        self.history: deque[dict[str, Any]] = deque(maxlen=window)
        self._latest: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._refs = 0
        self._deadline = 0.0
        self._cpu_previous = _cpu_times()

    # -- lifecycle (reference counted so concurrent users share one sampler thread) -------------------------
    def acquire(self) -> "HardwareMonitor":
        with self._lock:
            self._refs += 1
            if self._thread is None or not self._thread.is_alive():
                self._stop.clear()
                self._thread = threading.Thread(target=self._loop, name="r3d-hwmon", daemon=True)
                self._thread.start()
        return self

    def release(self) -> None:
        with self._lock:
            self._refs = max(0, self._refs - 1)
            if self._refs == 0:
                self._stop.set()

    def keepalive(self, seconds: float = 30.0) -> "HardwareMonitor":
        """Sample for ``seconds`` more; polling endpoints call this so an idle server runs no sampler thread."""
        with self._lock:
            self._deadline = max(self._deadline, time.monotonic() + seconds)
            if self._thread is None or not self._thread.is_alive():
                self._stop.clear()
                self._thread = threading.Thread(target=self._loop, name="r3d-hwmon", daemon=True)
                self._thread.start()
        return self

    def _cpu_percent(self) -> float | None:
        now = _cpu_times()
        previous, self._cpu_previous = self._cpu_previous, now
        if now is None or previous is None or now[1] <= previous[1]:
            return None
        return round(100.0 * (now[0] - previous[0]) / (now[1] - previous[1]), 1)

    def sample_once(self) -> dict[str, Any]:
        gpu = self.nvml.sample() if self.nvml.available else _smi_sample(self.index)
        gpu["cpu_util_pct"] = self._cpu_percent()
        gpu["t"] = time.time()
        with self._lock:
            self._latest = gpu
            self.history.append(gpu)
        return gpu

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                idle = self._refs == 0 and time.monotonic() >= self._deadline
            if idle:
                return
            began = time.perf_counter()
            try:
                self.sample_once()
            except Exception:
                pass
            self._stop.wait(max(0.0, self.interval - (time.perf_counter() - began)))

    @staticmethod
    def _summary(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        ordered = sorted(values)
        return {"mean": round(sum(values) / len(values), 1), "min": ordered[0], "max": ordered[-1],
                "p10": ordered[int(0.1 * (len(ordered) - 1))], "p50": ordered[len(ordered) // 2],
                "samples": len(values)}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latest, history = dict(self._latest), list(self.history)
        summary = {key: self._summary([h[key] for h in history if h.get(key) is not None])
                   for key in ("gpu_util_pct", "cpu_util_pct", "encoder_util_pct", "decoder_util_pct",
                               "pcie_tx_mb_s", "pcie_rx_mb_s", "vram_used_mib")}
        return {"latest": latest, "window": summary, "backend": latest.get("source", "unavailable"),
                "nvml_error": self.nvml.error}


_MONITOR: HardwareMonitor | None = None
_MONITOR_LOCK = threading.Lock()


def monitor() -> HardwareMonitor:
    """Process-wide monitor (not started until someone ``acquire()``s it)."""
    global _MONITOR
    with _MONITOR_LOCK:
        if _MONITOR is None:
            _MONITOR = HardwareMonitor()
        return _MONITOR
