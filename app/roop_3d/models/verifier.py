"""Bounded SHA-256 verification without deserialization or file mutation."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import stat
from typing import BinaryIO, Iterator

from roop_3d.audit.isolation import safe_path
from roop_3d.core.errors import DomainError


@contextmanager
def verified_stream(path: Path, expected_sha256: str, byte_size: int) -> Iterator[BinaryIO]:
    path = safe_path(path.absolute().parent, path.name)
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size != byte_size:
                raise DomainError("MODEL_DRIFT", "Artifact size or file type differs from the registry")
            digest, total = hashlib.sha256(), 0
            while block := stream.read(min(4 * 1024 * 1024, byte_size - total + 1)):
                total += len(block)
                if total > byte_size:
                    raise DomainError("MODEL_DRIFT", "Artifact exceeded its registered size")
                digest.update(block)
            after = os.fstat(stream.fileno())
            safe_path(path.parent, path.name)
            current = path.stat()
            if ((before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev)
                    != (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev)
                    or (current.st_ino, current.st_dev) != (before.st_ino, before.st_dev)
                    or total != byte_size or digest.hexdigest() != expected_sha256):
                raise DomainError("MODEL_DRIFT", "Artifact bytes differ from the preserved SHA-256")
            stream.seek(0)
            yield stream
    except OSError:
        raise DomainError("MODEL_UNREADABLE", "Registered artifact cannot be opened read-only", 503) from None


def verify_file(path: Path, expected_sha256: str, byte_size: int) -> Path:
    with verified_stream(path, expected_sha256, byte_size):
        return path
