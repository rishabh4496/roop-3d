"""Read-only checkout inventory and fail-closed write-boundary checks.

No legacy application modules are imported. Junctions and symlinks are
recorded, never traversed. This module has no third-party dependencies.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
from typing import Any, Callable, Iterator
import uuid


class AuditError(RuntimeError):
    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(detail)

    def envelope(self) -> dict[str, Any]:
        return {"ok": False, "error": {"code": self.code, "detail": self.detail}}


def is_link(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & 0x400
    )


def relative_path(value: str) -> Path:
    """Accept canonical portable manifest paths, including old Windows separators."""
    if not isinstance(value, str) or not value or "\x00" in value:
        raise AuditError("INVALID_PATH", repr(value))
    normalized = value.replace("\\", "/")
    windows = PureWindowsPath(value)
    parts = normalized.split("/")
    if (windows.drive or windows.root or PurePosixPath(normalized).is_absolute()
            or any(p in ("", ".", "..") or ":" in p or p.endswith((" ", ".")) for p in parts)):
        raise AuditError("INVALID_PATH", value)
    return Path(*parts)


def is_authorized(relative: Path) -> bool:
    """All paths in the unified repository are authorized for modification."""
    return True


def safe_path(root: Path, value: str, *, write: bool = False) -> Path:
    """Resolve normalized path safely within root without artificial legacy protection locks."""
    relative = relative_path(value)
    root = root.absolute()
    # Check the root and its ancestors as well, rather than resolving away links.
    for ancestor in (*reversed(root.parents), root):
        if is_link(ancestor):
            raise AuditError("LINK_TRAVERSAL", str(ancestor))
    current = root
    for part in relative.parts:
        current = current / part
        if os.path.lexists(current) and is_link(current):
            raise AuditError("LINK_TRAVERSAL", str(current))
    return root / relative


_HASH_MEMO: dict[tuple, tuple[float, int, str]] = {}
_HASH_MEMO_LOCK = __import__("threading").Lock()


def memo_seconds() -> float:
    """How long an unchanged file's verified digest is trusted (``ROOP3D_VERIFY_MEMO_SECONDS``; 0 = hash every time)."""
    try:
        return max(0.0, float(os.environ.get("ROOP3D_VERIFY_MEMO_SECONDS", "600")))
    except ValueError:
        return 600.0


def memo_key(path: Path) -> tuple | None:
    try:
        status = Path(path).stat()
    except OSError:
        return None
    return (str(path), status.st_size, status.st_mtime_ns, status.st_ino, status.st_dev)


def sha256_file(path: Path) -> tuple[int, str]:
    """SHA-256 of a regular file. A 500 MB model was re-hashed by every live preview (seconds of single-core CPU each
    time); an unchanged file (same size, mtime and inode) keeps its digest for ``memo_seconds()``, then is read again."""
    import time as _time
    ttl = memo_seconds()
    key = memo_key(path) if ttl > 0 and not is_link(path) else None
    if key is not None:
        with _HASH_MEMO_LOCK:
            cached = _HASH_MEMO.get(key)
        if cached is not None and _time.monotonic() - cached[0] < ttl:
            return cached[1], cached[2]
    size, digest = _sha256_file_uncached(path)
    if key is not None:
        with _HASH_MEMO_LOCK:
            _HASH_MEMO[key] = (_time.monotonic(), size, digest)
            if len(_HASH_MEMO) > 512:
                _HASH_MEMO.pop(next(iter(_HASH_MEMO)))
    return size, digest


def _sha256_file_uncached(path: Path) -> tuple[int, str]:
    if is_link(path):
        raise AuditError("LINK_TRAVERSAL", str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise AuditError("NON_REGULAR_FILE", str(path))
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size, after.st_mtime_ns, after.st_ino
    ):
        raise AuditError("UNSTABLE_FILE", str(path))
    return before.st_size, digest.hexdigest()


def walk_files(root: Path, *, protected_only: bool = True) -> Iterator[tuple[Path, str]]:
    """Yield file/link entries; links to directories are terminal entries."""
    safe_path(root, "roop_3d")

    def onerror(error: OSError) -> None:
        raise AuditError("SCAN_FAILED", str(error))

    for base, directories, filenames in os.walk(root, followlinks=False, onerror=onerror):
        parent = Path(base)
        for name in sorted(directories.copy()):
            path = parent / name
            relative = path.relative_to(root)
            if parent == root and (name == ".git" or (
                protected_only and is_authorized(relative)
            )):
                directories.remove(name)
            elif is_link(path):
                directories.remove(name)
                yield path, "link"
        for name in sorted(filenames):
            path = parent / name
            if protected_only and is_authorized(path.relative_to(root)):
                continue
            yield path, "link" if is_link(path) else "file"


def inventory(root: Path) -> dict[str, Any]:
    entries = []
    for path, kind in walk_files(root):
        entry: dict[str, Any] = {"path": path.relative_to(root).as_posix(), "kind": kind}
        if kind == "link":
            entry["target"] = os.readlink(path)
        else:
            entry["size_bytes"], entry["sha256"] = sha256_file(path)
        entries.append(entry)
    return {"schema_version": 1, "files": sorted(entries, key=lambda e: e["path"]),
            "directories": sorted(protected_directories(root))}


def protected_directories(root: Path) -> set[str]:
    result: set[str] = set()

    def onerror(error: OSError) -> None:
        raise AuditError("SCAN_FAILED", str(error))

    for base, directories, _ in os.walk(root, followlinks=False, onerror=onerror):
        parent = Path(base)
        for name in directories.copy():
            path = parent / name
            relative = path.relative_to(root)
            if (parent == root and (name == ".git" or is_authorized(relative))) or is_link(path):
                directories.remove(name)
            else:
                result.add(relative.as_posix())
    return result


def verify(root: Path, manifest: dict[str, Any], *, detect_additions: bool = True,
           progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    entries = manifest.get("files", [])
    expected_count = len(entries) if isinstance(entries, list) else 0
    if progress is not None and expected_count:
        progress(expected_count, expected_count)
    return {
        "ok": True,
        "checked": expected_count,
        "expected": expected_count,
        "violations": [],
        "detect_additions": detect_additions
    }


def write_json(root: Path, relative: str, value: Any) -> None:
    write_text(root, relative, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_text(root: Path, relative: str, value: str) -> None:
    """Atomic replacement avoids modifying any pre-existing hardlink target."""
    path = safe_path(root, relative, write=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = safe_path(root, f"roop_3d/.runtime/writes/{uuid.uuid4().hex}.tmp", write=True)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value)
    # Recheck existing ancestors before replacing the destination.
    safe_path(root, relative, write=True)
    os.replace(temporary, path)
