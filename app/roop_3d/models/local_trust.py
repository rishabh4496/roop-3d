"""Trust for weight files the user physically placed in ``app/models``.

Registry artifacts are SHA-256 pinned. A user can legitimately drop in a different export of the same
network (an FP16 GPEN, a re-exported CodeFormer, a renamed ``GPEN-BFR-512.onnx``), whose bytes can never
match the pinned digest. For files that live in ``app/models`` the user put them there on purpose, so the
runtime computes the digest itself, remembers it for the life of the process and carries on instead of
switching the model off.

Scope is deliberately narrow: only regular files directly under ``app/models`` (links, ``.trt_cache`` and
anything outside it are never trusted) and only files above ``MIN_TRUST_BYTES``. Downloads are still verified
against the registry by the downloader; this module never makes a download trusted.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

log = logging.getLogger(__name__)

MIN_TRUST_BYTES = 1024 * 1024
# Lower number = preferred. Only ONNX can be executed by the ONNX Runtime adapters; the rest are listed so the
# catalog reports them as present instead of "not installed".
SUFFIX_PRIORITY = {".onnx": 0, ".trt": 1, ".pth": 2}
_SCAN_TTL_SECONDS = 3.0
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
# Precision / export markers that do not change which network a file is.
_NOISE = ("fp16", "fp32", "float16", "float32", "half", "dynamic", "batch", "static", "simplified", "sim", "opt", "v01")


def normalize(text: str) -> str:
    return _NON_ALNUM.sub("", text.casefold())


def family_key(text: str) -> str:
    """``GPEN-BFR-512`` / ``gpen_bfr_512.fp16`` -> ``gpenbfr512``; ``CodeFormerv0.1`` -> ``codeformer``."""
    key = normalize(text)
    changed = True
    while changed:
        changed = False
        for marker in _NOISE:
            if key.endswith(marker) and len(key) > len(marker) + 3:
                key, changed = key[:-len(marker)], True
    return key


@dataclass(frozen=True)
class TrustedFile:
    path: Path
    sha256: str
    size: int
    identity: tuple


class LocalTrust:
    def __init__(self, root: Path | Callable[[], Path]):
        self._root = root if callable(root) else (lambda: root)
        self._lock = threading.RLock()
        self._listing: tuple[float, tuple[Path, ...]] = (0.0, ())
        self._trusted: dict[Path, TrustedFile] = {}
        self._announced: set[tuple[Path, tuple]] = set()

    # -- discovery ---------------------------------------------------------------------------------------
    def invalidate(self) -> None:
        with self._lock:
            self._listing = (0.0, ())

    def files(self) -> tuple[Path, ...]:
        with self._lock:
            stamp, listing = self._listing
            if listing and time.monotonic() - stamp < _SCAN_TTL_SECONDS:
                return listing
        root = self._root()
        found: list[Path] = []
        if root.is_dir() and not root.is_symlink():
            count = 0
            for parent, directories, names in os.walk(root, followlinks=False):
                directories[:] = sorted(d for d in directories
                                        if d.casefold() != ".trt_cache" and not (Path(parent) / d).is_symlink())
                for name in sorted(names):
                    count += 1
                    if count > 50000:
                        break
                    path = Path(parent) / name
                    if path.suffix.casefold() not in SUFFIX_PRIORITY or path.is_symlink():
                        continue
                    try:
                        if path.is_file() and path.stat().st_size > MIN_TRUST_BYTES:
                            found.append(path.resolve(strict=True))
                    except OSError:
                        continue
        with self._lock:
            self._listing = (time.monotonic(), tuple(found))
            return self._listing[1]

    def locate(self, names: Iterable[str]) -> Path | None:
        """Best physical match for any of ``names`` (file names, model ids, aliases), case/punctuation blind."""
        exact = {normalize(Path(n).stem) for n in names if n}
        loose = {family_key(Path(n).stem) for n in names if n}
        loose.discard("")
        best: tuple[tuple, Path] | None = None
        for path in self.files():
            raw = normalize(path.stem)
            family = family_key(path.stem)
            if raw in exact:
                rank = 0
            elif family in loose:
                rank = 1
            else:
                continue
            # Runnable formats first, then how exact the name match is, then the shortest (least decorated) name.
            candidate = ((SUFFIX_PRIORITY[path.suffix.casefold()], rank, len(path.name), path.name.casefold()), path)
            if best is None or candidate[0] < best[0]:
                best = candidate
        return best[1] if best else None

    # -- trust -------------------------------------------------------------------------------------------
    @staticmethod
    def _identity(path: Path) -> tuple:
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_dev

    def trust(self, path: Path) -> TrustedFile:
        """Hash ``path`` once per file identity and remember it. Never raises for a readable file."""
        path = Path(path)
        identity = self._identity(path)
        with self._lock:
            known = self._trusted.get(path)
            if known is not None and known.identity == identity:
                return known
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(4 * 1024 * 1024):
                digest.update(block)
        after = self._identity(path)
        entry = TrustedFile(path, digest.hexdigest(), after[0], after)
        with self._lock:
            self._trusted[path] = entry
            if (path, after) not in self._announced:
                self._announced.add((path, after))
                log.info("Auto-trusted local weight file: %s", path.name)
        return entry

    def lookup(self, path: Path | None) -> TrustedFile | None:
        """The trusted entry for ``path`` if it was auto-trusted and is unchanged, else ``None``."""
        if path is None:
            return None
        with self._lock:
            entry = self._trusted.get(Path(path))
        if entry is None:
            return None
        try:
            return entry if self._identity(entry.path) == entry.identity else None
        except OSError:
            return None

    def inside_root(self, path: Path) -> bool:
        try:
            Path(path).resolve(strict=True).relative_to(self._root().resolve())
            return True
        except (OSError, ValueError):
            return False
