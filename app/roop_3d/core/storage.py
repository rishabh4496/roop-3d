"""Owned-path, durable atomic JSON storage and cross-process serialization."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
from typing import Any, Iterator
import uuid
import weakref

from roop_3d.audit.isolation import safe_path

# One in-process lock per lock-file path. Held WEAKLY: every render has its own `.roop3d_<job>.mp4.lock`, and a
# `threading.RLock` owns a kernel semaphore on Windows/Python 3.10, so a plain dict leaked one OS handle (plus a
# dict entry) per job forever. A lock now lives exactly as long as some thread is using it, which is all that mutual
# exclusion needs: anyone holding it keeps it alive, so two threads can never end up with different locks for one path.
_threads: "weakref.WeakValueDictionary[str, threading.RLock]" = weakref.WeakValueDictionary()
_guard = threading.Lock()


def owned(root: Path, relative: str) -> Path:
    return safe_path(root, relative, write=True)


def atomic_bytes(root: Path, relative: str, value: bytes) -> None:
    destination = owned(root, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = owned(root, f"{relative}.{uuid.uuid4().hex}.part")
    try:
        with temporary.open("xb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        owned(root, relative)
        os.replace(temporary, destination)
        if os.name != "nt":
            descriptor = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary.exists():
            owned(root, temporary.relative_to(root).as_posix()).unlink()


def atomic_json(root: Path, relative: str, value: Any, *, indent: int | None = 2) -> None:
    """``indent=None`` selects the C JSON encoder (~6x faster on large documents); the default keeps every existing
    file byte-for-byte as it was. Only hot, machine-read files (the job database) opt in."""
    atomic_bytes(root, relative, (json.dumps(value, allow_nan=False, indent=indent) + "\n").encode())


@contextmanager
def exclusive(root: Path, relative: str) -> Iterator[None]:
    """OS locks are released on process death; never truncate an existing lock."""
    path = owned(root, relative)
    with _guard:
        lock = _threads.get(str(path))
        if lock is None:
            lock = _threads[str(path)] = threading.RLock()
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_nlink != 1:
            raise ValueError("Hardlinked lock files are forbidden")
        with path.open("a+b") as stream:
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                stream.seek(0)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
