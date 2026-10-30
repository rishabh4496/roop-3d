"""Hardware-keyed metadata only; this module does not compile or load engines."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import uuid
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from roop_3d.audit.isolation import sha256_file
from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract
from roop_3d.core.storage import atomic_json, exclusive, owned
from roop_3d.models.verifier import verify_file


class ShapeRange(Contract):
    minimum: tuple[int, ...] = Field(min_length=1, max_length=16)
    optimum: tuple[int, ...] = Field(min_length=1, max_length=16)
    maximum: tuple[int, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def ordered(self) -> "ShapeRange":
        if (len({len(self.minimum), len(self.optimum), len(self.maximum)}) != 1
                or any(not 0 < a <= b <= c for a, b, c in zip(self.minimum, self.optimum, self.maximum))):
            raise ValueError("Shape profile dimensions must be positive and min <= opt <= max")
        return self


class EngineKey(Contract):
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    precision: Literal["fp32", "fp16", "int8"]
    gpu_compute_capability: str = Field(pattern=r"^[0-9]+\.[0-9]+$")
    driver: str = Field(min_length=1, max_length=128)
    onnxruntime_version: str = Field(min_length=1, max_length=128)
    tensorrt_version: str | None = Field(max_length=128)
    shape_profile: dict[str, ShapeRange] = Field(min_length=1, max_length=64)
    build_signature: str = Field(default="stage03-metadata", min_length=1, max_length=256)

    def digest(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


class EngineMetadata(Contract):
    schema_version: Literal[1] = 1
    key: EngineKey
    kind: Literal["engine", "onnx"]
    compiled_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiled_byte_size: int = Field(gt=0, strict=True)


class BundleFile(Contract):
    name: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,240}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0, le=2**35)


class EngineBundle(Contract):
    schema_version: Literal[1] = 1
    key: EngineKey
    files: tuple[BundleFile, ...] = Field(min_length=1, max_length=256)
    receipt: dict


_lock_guard = threading.Lock()
_local_locks: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


class EngineCache:
    def __init__(self, root: Path):
        self.root = root

    def paths(self, key: EngineKey, kind: Literal["engine", "onnx"] = "engine") -> tuple[Path, Path]:
        if kind not in {"engine", "onnx"}:
            raise ValueError("Unsupported compiled artifact type")
        if kind == "engine" and not key.tensorrt_version:
            raise ValueError("TensorRT engines require an observed TensorRT version")
        base = f"roop_3d/cache/engines/{key.digest()}.{kind}"
        return owned(self.root, base), owned(self.root, base + ".json")

    def record(self, key: EngineKey, kind: Literal["engine", "onnx"] = "engine") -> EngineMetadata:
        artifact, metadata = self.paths(key, kind)
        with exclusive(self.root, metadata.relative_to(self.root).as_posix() + ".lock"):
            artifact, metadata = self.paths(key, kind)
            if artifact.stat().st_nlink != 1:
                raise DomainError("ENGINE_CACHE_UNSAFE", "Compiled artifacts cannot be hardlinked")
            size, digest = sha256_file(artifact)
            value = EngineMetadata(key=key, kind=kind, compiled_sha256=digest, compiled_byte_size=size)
            atomic_json(self.root, metadata.relative_to(self.root).as_posix(), value.model_dump(mode="json"))
            return value

    def lookup(self, key: EngineKey, kind: Literal["engine", "onnx"] = "engine") -> Path | None:
        artifact, metadata = self.paths(key, kind)
        if not artifact.exists() or not metadata.exists():
            return None
        try:
            value = EngineMetadata.model_validate_json(metadata.read_text("utf-8"))
            if value.key != key or value.kind != kind:
                raise ValueError("Cache identity differs")
            return verify_file(artifact, value.compiled_sha256, value.compiled_byte_size)
        except (OSError, ValueError, DomainError):
            raise DomainError("ENGINE_CACHE_INVALID", "Compiled artifact or cache metadata differs from its identity") from None

    @contextmanager
    def lock(self, name: str, *, timeout: float = 600):
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in name) or timeout < 0:
            raise ValueError("Invalid engine lock name/timeout")
        relative = f"roop_3d/cache/engines/.{name}.lock"
        path = owned(self.root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _lock_guard:
            local = _local_locks.get(str(path))
            if local is None:
                local = threading.RLock()
                _local_locks[str(path)] = local
        deadline = time.monotonic() + timeout
        if not local.acquire(timeout=timeout):
            raise DomainError("ENGINE_BUILD_BUSY", "Another engine operation holds this key", 409)
        try:
            if path.exists() and path.stat().st_nlink != 1:
                raise DomainError("ENGINE_CACHE_UNSAFE", "Hardlinked lock file is forbidden", 409)
            with path.open("a+b") as stream:
                if stream.tell() == 0:
                    stream.write(b"0")
                    stream.flush()
                while True:
                    try:
                        stream.seek(0)
                        if os.name == "nt":
                            import msvcrt
                            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                        else:
                            import fcntl
                            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except OSError as error:
                        if error.errno not in (11, 13, 35, 36):
                            raise
                        if time.monotonic() >= deadline:
                            raise DomainError("ENGINE_BUILD_BUSY", "Timed out waiting for engine build lock", 409) from None
                        time.sleep(0.05)
                try:
                    yield
                finally:
                    stream.seek(0)
                    if os.name == "nt":
                        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            local.release()

    def bundle_path(self, key: EngineKey) -> Path:
        if not key.tensorrt_version:
            raise ValueError("Engine bundles require an observed TensorRT version")
        return owned(self.root, f"roop_3d/cache/engines/{key.digest()}.bundle")

    def lookup_bundle(self, key: EngineKey) -> EngineBundle | None:
        directory = self.bundle_path(key)
        if not directory.exists():
            return None
        try:
            manifest = owned(self.root, (directory / "manifest.json").relative_to(self.root).as_posix())
            if not manifest.is_file() or manifest.stat().st_nlink != 1:
                raise ValueError("Unsafe engine manifest")
            value = EngineBundle.model_validate_json(manifest.read_text("utf-8"))
            if value.key != key or {p.name for p in directory.iterdir()} != {f.name for f in value.files} | {"manifest.json"}:
                raise ValueError("Engine bundle identity or contents differ")
            for file in value.files:
                path = owned(self.root, (directory / file.name).relative_to(self.root).as_posix())
                if not path.is_file() or path.stat().st_nlink != 1:
                    raise ValueError("Unsafe engine artifact")
                verify_file(path, file.sha256, file.size_bytes)
            return value
        except (OSError, ValueError, DomainError):
            raise DomainError("ENGINE_CACHE_INVALID", "Engine bundle is incomplete or differs from its recorded hashes", 409) from None

    @contextmanager
    def transaction(self, key: EngineKey):
        """Under key lock, yield a new directory or a previously verified bundle.

        Caller publishes with publish_bundle. Failed builds leave no visible bundle.
        """
        with self.lock(key.digest()):
            existing = self.lookup_bundle(key)
            if existing is not None:
                yield self.bundle_path(key), existing
                return
            temporary = owned(self.root, f"roop_3d/cache/engines/.{key.digest()}-{uuid.uuid4().hex}.part")
            temporary.mkdir(parents=True)
            try:
                yield temporary, None
            finally:
                if temporary.exists():
                    self._remove_temporary(temporary)

    def _remove_temporary(self, directory: Path) -> None:
        cache_root = owned(self.root, "roop_3d/cache/engines")
        path = owned(self.root, directory.relative_to(self.root).as_posix())
        if path.parent != cache_root or not path.name.endswith(".part"):
            raise DomainError("ENGINE_CACHE_UNSAFE", "Only an owned build temporary may be removed", 409)
        for item in path.rglob("*"):
            owned(self.root, item.relative_to(self.root).as_posix())
        shutil.rmtree(path)

    def publish_bundle(self, key: EngineKey, temporary: Path, receipt: dict) -> EngineBundle:
        temporary = owned(self.root, temporary.relative_to(self.root).as_posix())
        target = self.bundle_path(key)
        cache_root = owned(self.root, "roop_3d/cache/engines")
        if temporary.parent != cache_root or not temporary.name.endswith(".part") or target.exists():
            raise DomainError("ENGINE_CACHE_UNSAFE", "Bundle publication requires a new owned build temporary", 409)
        files = []
        for item in sorted(temporary.iterdir()):
            path = owned(self.root, item.relative_to(self.root).as_posix())
            if path.name == "manifest.json" or not path.is_file() or path.stat().st_nlink != 1:
                raise DomainError("ENGINE_CACHE_UNSAFE", "Engine bundle files must be new regular files", 409)
            size, digest = sha256_file(path)
            files.append(BundleFile(name=path.name, sha256=digest, size_bytes=size))
        bundle = EngineBundle(key=key, files=tuple(files), receipt=receipt)
        atomic_json(self.root, (temporary / "manifest.json").relative_to(self.root).as_posix(), bundle.model_dump(mode="json"))
        owned(self.root, temporary.relative_to(self.root).as_posix())
        os.replace(temporary, target)
        return bundle
