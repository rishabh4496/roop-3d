"""Orphan-process prevention for the FFmpeg children Roop 3D spawns.

Three layers, because no single mechanism covers every way a process can die:

1. **Lifetime binding** (``bind_children_to_process_lifetime``): on Windows the server joins a Job Object flagged
   ``KILL_ON_JOB_CLOSE``; every child (and grandchild) belongs to it, so the OS kills them when the server exits for
   *any* reason, including ``TerminateProcess`` and a crash. On Linux a child asks for ``SIGKILL`` when its parent
   dies (``PR_SET_PDEATHSIG``) via ``spawn_kwargs``.
2. **Registry** (``ProcessRegistry``): each tracked child is recorded on disk with pid *and* creation time. If the
   server vanished before layer 1 could help (an unsupported OS, a job that forbids nesting) the next start sweeps
   any recorded child that is still alive and whose owner is gone. The creation time makes a recycled pid safe.
3. **Termination discipline** at the call sites: ``terminate()`` then bounded ``wait()`` then ``kill()``.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any
import uuid
import weakref

from roop_3d.reliability import probe
from roop_3d.reliability.ledger import LEDGER

log = logging.getLogger("roop3d.reliability")

_job_handle: Any = None
_bind_state: dict[str, Any] = {"bound": False, "reason": "not attempted"}


def bind_children_to_process_lifetime() -> dict[str, Any]:
    """Make the OS reap this process's children when it dies. Idempotent; never raises."""
    global _job_handle
    if _bind_state["bound"]:
        return dict(_bind_state)
    if os.name != "nt":
        _bind_state.update(bound=False, reason="posix: per-child PR_SET_PDEATHSIG via spawn_kwargs()")
        return dict(_bind_state)
    try:
        import ctypes
        from ctypes import wintypes

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in
                        ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                         "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", IoCounters), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        handle = kernel.CreateJobObjectW(None, None)
        if not handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObject failed")
        info = Extended()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
        if not kernel.AssignProcessToJobObject(handle, kernel.GetCurrentProcess()):
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
        _job_handle = handle  # must stay open for the life of the process: closing it is what triggers the kill
        _bind_state.update(bound=True, reason="windows job object (KILL_ON_JOB_CLOSE)")
    except Exception as error:
        _bind_state.update(bound=False, reason=f"unavailable: {type(error).__name__}")
        log.warning("Child-process lifetime binding unavailable (%s); relying on the startup orphan sweep",
                    type(error).__name__)
    return dict(_bind_state)


def spawn_kwargs() -> dict[str, Any]:
    """Extra ``subprocess.Popen`` kwargs that tie the child's lifetime to ours on POSIX (no-op on Windows)."""
    if os.name == "nt" or not hasattr(os, "fork"):
        return {}
    import sys
    if not sys.platform.startswith("linux"):
        return {}

    def die_with_parent() -> None:  # runs in the child between fork and exec
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, 9)  # PR_SET_PDEATHSIG, SIGKILL
    return {"preexec_fn": die_with_parent}


class ProcessRegistry:
    """Disk-backed record of live children. ``root=None`` keeps tracking in memory only (counts, no sweep)."""

    relative = "roop_3d/.runtime/processes"

    def __init__(self, root: Path | None = None):
        self._lock = threading.Lock()
        self._live: dict[int, weakref.ReferenceType] = {}
        self._files: dict[int, Path] = {}
        self.root: Path | None = None
        self.configure(root)

    def configure(self, root: Path | None) -> None:
        self.root = Path(root) if root is not None else None

    def _directory(self) -> Path | None:
        return self.root / self.relative if self.root is not None else None

    # ---- tracking
    def track(self, process: subprocess.Popen, purpose: str) -> subprocess.Popen:
        """Register a freshly spawned child. Returns ``process`` so it composes with ``Popen(...)``."""
        pid = process.pid
        with self._lock:
            self._live[pid] = weakref.ref(process)
        directory = self._directory()
        if directory is not None:
            record = {"pid": pid, "create_time": probe.process_create_time(pid), "purpose": purpose,
                      "owner_pid": os.getpid(), "owner_create_time": probe.process_create_time(os.getpid()),
                      "started": time.time()}
            try:
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f"{pid}-{uuid.uuid4().hex[:8]}.json"
                path.write_text(json.dumps(record), encoding="utf-8")
                with self._lock:
                    self._files[pid] = path
            except OSError:
                pass  # a registry write failure must never stop a render
        return process

    def untrack(self, process: subprocess.Popen) -> None:
        with self._lock:
            self._live.pop(process.pid, None)
            path = self._files.pop(process.pid, None)
        if path is not None:
            path.unlink(missing_ok=True)

    def live_count(self) -> int:
        """Tracked children that are still running. Exited ones are forgotten here, so the count is exact."""
        stale: list[int] = []
        with self._lock:
            items = list(self._live.items())
        alive = 0
        for pid, ref in items:
            process = ref()
            if process is not None and process.poll() is None:
                alive += 1
            else:
                stale.append(pid)
        for pid in stale:
            with self._lock:
                self._live.pop(pid, None)
                path = self._files.pop(pid, None)
            if path is not None:
                path.unlink(missing_ok=True)
        return alive

    # ---- recovery
    def sweep_stale(self) -> list[dict[str, Any]]:
        """Kill recorded children whose owner process is gone. Safe against pid reuse and against a second live
        server instance (whose children are left alone). Returns what was found, for the diagnostics report."""
        directory = self._directory()
        results: list[dict[str, Any]] = []
        if directory is None or not directory.is_dir():
            return results
        for path in sorted(directory.glob("*.json")):
            try:
                record = json.loads(path.read_text("utf-8"))
                pid, stamp = int(record["pid"]), record.get("create_time")
                owner, owner_stamp = int(record["owner_pid"]), record.get("owner_create_time")
            except (OSError, ValueError, KeyError, TypeError):
                path.unlink(missing_ok=True)
                results.append({"file": path.name, "action": "removed_unreadable_record"})
                continue
            item = {"pid": pid, "purpose": record.get("purpose"), "action": "none"}
            owner_alive = (owner == os.getpid() or (probe.process_alive(owner)
                           and (owner_stamp is None or probe.process_create_time(owner) == owner_stamp)))
            child_ours = probe.process_alive(pid) and (stamp is None or probe.process_create_time(pid) == stamp)
            if owner_alive:
                item["action"] = "owner_alive_left_running" if child_ours else "already_exited"
                if not child_ours:
                    path.unlink(missing_ok=True)
            elif child_ours:
                item["action"] = "killed_orphan" if probe.kill_process(pid) else "kill_failed"
                if item["action"] == "killed_orphan":
                    path.unlink(missing_ok=True)
            else:
                item["action"] = "already_exited"
                path.unlink(missing_ok=True)
            results.append(item)
        return results


REGISTRY = ProcessRegistry()
LEDGER.gauge("ffmpeg_process", REGISTRY.live_count)


def terminate(process: subprocess.Popen, *, grace: float = 5.0) -> None:
    """terminate -> bounded wait -> kill. Never leaves the child running, never blocks forever."""
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            log.warning("A child process did not exit after kill()")
    except (ProcessLookupError, OSError):
        pass


def terminate_tree(pid: int, *, grace: float = 5.0) -> int:
    """Kill ``pid`` and everything below it; returns how many processes were signalled. Test and cleanup helper."""
    victims = [*probe.descendant_pids(pid), pid]
    for victim in reversed(victims):
        probe.kill_process(victim)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline and any(probe.process_alive(v) for v in victims):
        time.sleep(0.02)
    return len(victims)
