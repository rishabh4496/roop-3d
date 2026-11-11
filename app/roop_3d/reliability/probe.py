"""Dependency-free process, GPU and runtime probes used by the Stage 18 reliability tooling.

``psutil`` is deliberately not required: the Windows paths use ``ctypes`` against kernel32/psapi and the Linux
paths read ``/proc``. Every probe degrades to ``None`` (never an exception) when the OS cannot answer, so a
missing metric is visible in a report instead of aborting a soak run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import gc
import os
import subprocess
import sys
import threading
import time
from typing import Any

MIB = 1024 * 1024


# --------------------------------------------------------------------------------------------- Windows ctypes
if os.name == "nt":  # pragma: no branch - exercised on the Windows development machine, skipped elsewhere
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _psapi = ctypes.WinDLL("psapi", use_last_error=True)

    class _MemoryCounters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t)]

    class _ProcessEntry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", wintypes.LONG), ("dwFlags", wintypes.DWORD),
                    ("szExeFile", wintypes.WCHAR * 260)]

    _k32.GetCurrentProcess.restype = wintypes.HANDLE
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    _k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    _k32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry)]
    _k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry)]
    _k32.GetProcessHandleCount.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    _k32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    _k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    _k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_MemoryCounters), wintypes.DWORD]

    _QUERY_LIMITED = 0x1000
    _TERMINATE = 0x0001
    _STILL_ACTIVE = 259

    def _win_processes() -> list[tuple[int, int, int, str]]:
        """(pid, parent pid, os thread count, exe name) for every process, from one ToolHelp snapshot."""
        snapshot = _k32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
        if snapshot in (None, wintypes.HANDLE(-1).value, ctypes.c_void_p(-1).value):
            return []
        rows: list[tuple[int, int, int, str]] = []
        try:
            entry = _ProcessEntry()
            entry.dwSize = ctypes.sizeof(entry)
            ok = _k32.Process32FirstW(snapshot, ctypes.byref(entry))
            while ok:
                rows.append((entry.th32ProcessID, entry.th32ParentProcessID, entry.cntThreads, entry.szExeFile))
                ok = _k32.Process32NextW(snapshot, ctypes.byref(entry))
        finally:
            _k32.CloseHandle(snapshot)
        return rows

    def _win_create_time(pid: int) -> int | None:
        handle = _k32.OpenProcess(_QUERY_LIMITED, False, pid)
        if not handle:
            return None
        try:
            created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
            if not _k32.GetProcessTimes(handle, *(ctypes.byref(item) for item in (created, exited, kernel, user))):
                return None
            return (created.dwHighDateTime << 32) | created.dwLowDateTime
        finally:
            _k32.CloseHandle(handle)

    def _win_alive(pid: int) -> bool:
        handle = _k32.OpenProcess(_QUERY_LIMITED, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return bool(_k32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == _STILL_ACTIVE
        finally:
            _k32.CloseHandle(handle)

    def _win_kill(pid: int) -> bool:
        handle = _k32.OpenProcess(_TERMINATE | _QUERY_LIMITED, False, pid)
        if not handle:
            return False
        try:
            return bool(_k32.TerminateProcess(handle, 1))
        finally:
            _k32.CloseHandle(handle)


# ------------------------------------------------------------------------------------------- process identity
def process_create_time(pid: int) -> int | None:
    """An opaque, OS-stable start stamp. Together with the pid it identifies a process even after pid reuse."""
    if os.name == "nt":
        return _win_create_time(pid)
    try:
        with open(f"/proc/{pid}/stat", "rb") as stream:
            fields = stream.read().rsplit(b")", 1)[1].split()
        return int(fields[19])  # starttime, clock ticks since boot
    except (OSError, IndexError, ValueError):
        return None


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        return _win_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def kill_process(pid: int) -> bool:
    """Force-terminate one process by pid. The caller has already proven that the pid is ours."""
    if pid <= 0 or pid == os.getpid():
        return False
    if os.name == "nt":
        return _win_kill(pid)
    import signal
    try:
        os.kill(pid, signal.SIGKILL)
        return True
    except OSError:
        return False


def child_pids(parent: int | None = None) -> list[int]:
    """Direct children of ``parent`` (default: this process)."""
    parent = os.getpid() if parent is None else parent
    if os.name == "nt":
        return [pid for pid, ppid, _, _ in _win_processes() if ppid == parent and pid != parent]
    found: list[int] = []
    try:
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            try:
                with open(f"/proc/{name}/stat", "rb") as stream:
                    if int(stream.read().rsplit(b")", 1)[1].split()[1]) == parent:
                        found.append(int(name))
            except (OSError, IndexError, ValueError):
                continue
    except OSError:
        pass
    return found


def descendant_pids(parent: int | None = None) -> list[int]:
    """All transitive children (the whole process tree below ``parent``)."""
    parent = os.getpid() if parent is None else parent
    if os.name == "nt":
        edges: dict[int, list[int]] = {}
        for pid, ppid, _, _ in _win_processes():
            edges.setdefault(ppid, []).append(pid)
    else:
        edges = {}
        for pid in [int(name) for name in os.listdir("/proc") if name.isdigit()]:
            try:
                with open(f"/proc/{pid}/stat", "rb") as stream:
                    edges.setdefault(int(stream.read().rsplit(b")", 1)[1].split()[1]), []).append(pid)
            except (OSError, IndexError, ValueError):
                continue
    out, queue = [], [parent]
    while queue:
        for child in edges.get(queue.pop(), []):
            if child not in out and child != parent:
                out.append(child)
                queue.append(child)
    return out


# ----------------------------------------------------------------------------------------------- own process
@dataclass(frozen=True)
class ProcessMemory:
    rss_mib: float | None = None
    private_mib: float | None = None
    peak_rss_mib: float | None = None
    handles: int | None = None
    os_threads: int | None = None


def process_memory() -> ProcessMemory:
    """Working set / private bytes / peak working set, plus open handle (fd) and OS thread counts."""
    if os.name == "nt":
        counters = _MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = _k32.GetCurrentProcess()
        rss = private = peak = None
        if _psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            rss, private, peak = (counters.WorkingSetSize / MIB, counters.PrivateUsage / MIB,
                                  counters.PeakWorkingSetSize / MIB)
        count = wintypes.DWORD()
        handles = count.value if _k32.GetProcessHandleCount(handle, ctypes.byref(count)) else None
        pid = os.getpid()
        threads = next((n for p, _, n, _ in _win_processes() if p == pid), None)
        return ProcessMemory(rss, private, peak, handles, threads)
    rss = peak = threads = None
    try:
        with open("/proc/self/status", "r", encoding="ascii", errors="replace") as stream:
            for line in stream:
                key, _, value = line.partition(":")
                if key == "VmRSS":
                    rss = int(value.split()[0]) / 1024
                elif key == "VmHWM":
                    peak = int(value.split()[0]) / 1024
                elif key == "Threads":
                    threads = int(value.split()[0])
    except (OSError, ValueError):
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak = rss = usage / (MIB if sys.platform == "darwin" else 1024)
        except Exception:
            pass
    try:
        handles = len(os.listdir("/proc/self/fd"))
    except OSError:
        handles = None
    return ProcessMemory(rss, rss, peak, handles, threads)


# ------------------------------------------------------------------------------------------------------ GPU
@dataclass(frozen=True)
class GpuMemory:
    available: bool = False
    used_mib: float | None = None
    total_mib: float | None = None
    # nvidia-smi reports "[N/A]" per-process memory for WDDM (default Windows) GPUs; then only the device total is
    # real and growth must be judged against a baseline taken at the same idle point.
    process_mib: float | None = None
    per_process: bool = False
    name: str | None = None
    driver_version: str | None = None
    reason: str | None = None


_gpu_cache: tuple[float, GpuMemory] | None = None
_gpu_lock = threading.Lock()


def _smi(arguments: list[str]) -> str:
    return subprocess.run(["nvidia-smi", *arguments], capture_output=True, text=True, timeout=5, check=True).stdout


def gpu_memory(*, max_age: float = 1.0, pid: int | None = None) -> GpuMemory:
    """Device and (where the driver allows) per-process VRAM from ``nvidia-smi``. Rate-limited; never raises.

    ``pid`` selects whose per-process figure to report (default: this process). The TensorRT build sidecar asks about its parent."""
    global _gpu_cache
    with _gpu_lock:
        now = time.monotonic()
        if _gpu_cache is not None and now - _gpu_cache[0] < max_age:
            return _gpu_cache[1]
        try:
            line = _smi(["--query-gpu=name,memory.used,memory.total,driver_version",
                         "--format=csv,noheader,nounits"]).strip().splitlines()[0]
            name, used, total, driver = [part.strip() for part in line.split(",")]
            process_mib, per_process = None, False
            try:
                for row in _smi(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]).splitlines():
                    listed, _, memory = row.partition(",")
                    if listed.strip() == str(os.getpid() if pid is None else pid) and memory.strip().replace(".", "").isdigit():
                        process_mib, per_process = float(memory), True
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            value = GpuMemory(True, float(used), float(total), process_mib, per_process, name, driver)
        except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
            value = GpuMemory(False, reason=type(error).__name__)
        _gpu_cache = (now, value)
        return value


def torch_cuda_state() -> dict[str, Any]:
    """Allocator view when torch is already loaded (this app never imports it just to measure).

    ``fragmentation`` is reserved-but-unallocated bytes as a fraction of reserved: the classic caching-allocator
    fragmentation indicator. It is None while torch (or CUDA) is not loaded.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return {"loaded": False, "fragmentation": None}
    try:
        if not torch.cuda.is_available():
            return {"loaded": True, "cuda": False, "fragmentation": None}
        reserved, allocated = torch.cuda.memory_reserved(), torch.cuda.memory_allocated()
        stats = torch.cuda.memory_stats()
        return {"loaded": True, "cuda": True, "allocated_mib": allocated / MIB, "reserved_mib": reserved / MIB,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
                "fragmentation": round((reserved - allocated) / reserved, 4) if reserved else 0.0,
                "alloc_retries": int(stats.get("num_alloc_retries", 0)), "ooms": int(stats.get("num_ooms", 0))}
    except Exception as error:  # a broken CUDA context must not break a diagnostic
        return {"loaded": True, "cuda": False, "fragmentation": None, "error": type(error).__name__}


def count_ort_sessions() -> int:
    """Live ONNX Runtime ``InferenceSession`` objects that still own a native session (a gc scan: on demand only).

    Mirrors ``engine.resources.evict_for_build``: an ORT session that is unreferenced by the application but not yet
    collected is *not* counted, a session still referenced after its owner "released" it is.
    """
    try:
        from roop_3d.engine.resources import count_live_ort_sessions
    except Exception:
        return 0
    return count_live_ort_sessions()


# -------------------------------------------------------------------------------------------------- snapshot
@dataclass(frozen=True)
class ResourceSample:
    elapsed: float
    wall: float
    memory: ProcessMemory
    py_threads: int
    children: int
    gpu: GpuMemory
    gauges: dict[str, int] = field(default_factory=dict)
    ort_sessions: int | None = None
    py_heap_blocks: int | None = None

    def metrics(self) -> dict[str, float]:
        """Flat numeric view used by the growth analysis (None-valued metrics are omitted)."""
        values: dict[str, float | None] = {
            "rss_mib": self.memory.rss_mib, "private_mib": self.memory.private_mib,
            "handles": self.memory.handles, "os_threads": self.memory.os_threads,
            "py_threads": self.py_threads, "children": self.children,
            "gpu_used_mib": self.gpu.used_mib if self.gpu.available else None,
            "gpu_process_mib": self.gpu.process_mib, "ort_sessions": self.ort_sessions,
            "py_heap_blocks": self.py_heap_blocks}
        values.update({f"live.{key}": value for key, value in self.gauges.items()})
        return {key: float(value) for key, value in values.items() if value is not None}


def take_sample(started: float, *, gauges: dict[str, int] | None = None, with_gpu: bool = True,
                with_ort: bool = False, with_heap: bool = False) -> ResourceSample:
    return ResourceSample(
        elapsed=time.monotonic() - started, wall=time.time(), memory=process_memory(),
        py_threads=threading.active_count(), children=len(child_pids()),
        gpu=gpu_memory() if with_gpu else GpuMemory(), gauges=dict(gauges or {}),
        ort_sessions=count_ort_sessions() if with_ort else None,
        py_heap_blocks=sys.getallocatedblocks() if with_heap else None)
