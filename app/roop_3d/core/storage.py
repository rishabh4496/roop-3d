"""Owned-path, durable atomic JSON storage and cross-process serialization."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
from typing import Any, Iterator
import uuid

from roop_3d.audit.isolation import safe_path

_threads: dict[str, threading.RLock] = {}
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


def atomic_json(root: Path, relative: str, value: Any) -> None:
    atomic_bytes(root, relative, (json.dumps(value, allow_nan=False, indent=2) + "\n").encode())


@contextmanager
def exclusive(root: Path, relative: str) -> Iterator[None]:
    """OS locks are released on process death; never truncate an existing lock."""
    path = owned(root, relative)
    with _guard:
        lock = _threads.setdefault(str(path), threading.RLock())
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
