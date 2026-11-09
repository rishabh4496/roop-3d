"""Read-only access to preserved locators and configured local search roots."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import BinaryIO, Callable, Iterator

from roop_3d.audit.isolation import is_link, safe_path
from roop_3d.audit.project_paths import application_root, legacy_root
from roop_3d.core.errors import DomainError
from roop_3d.core.settings import runtime_path
from roop_3d.models.downloader import Downloader
from roop_3d.models.local_trust import LocalTrust
from roop_3d.models.registry import ModelRecord, ModelRegistry, authorize
from roop_3d.models.verifier import verified_stream, verify_file


class LocalWeightProvider:
    def __init__(self, registry: ModelRegistry, *, root: Path | None = None,
                 scopes: dict[str, Path] | None = None,
                 search_paths: Callable[[], tuple[str, ...]] | None = None,
                 trust: LocalTrust | None = None):
        self.root = root or application_root()
        self.registry = registry
        self._scopes = scopes
        self.search_paths = search_paths or (lambda: ())
        # Files the user placed in app/models that the registry digest cannot describe (see local_trust.py).
        self.trust = trust

    def scopes(self) -> dict[str, Path]:
        if self._scopes is not None:
            return self._scopes
        home = Path.home()
        return {"repository": legacy_root(), "user_cache": home / ".cache",
                "insightface_cache": home / ".insightface", "torch_cache": home / ".torch"}

    def _locations(self, record: ModelRecord) -> Iterator[Path]:
        authorize(record, action="resolve")
        if not record.expected_sha256 or not record.byte_size:
            raise DomainError("MODEL_IDENTITY_UNKNOWN", "No observed artifact identity is registered")
        if record.locator:
            scope = self.scopes().get(record.locator.scope)
            if scope is None:
                raise DomainError("MODEL_SCOPE_UNKNOWN", "Model locator scope is not configured")
            if os.path.lexists(scope):
                path = safe_path(scope, record.locator.path)
                if path.exists():
                    # Drift at an authoritative locator is fatal, even if another copy exists.
                    yield path
                    return
        cached = Downloader(self.root).cached(record)
        if cached:
            yield cached
            return
        count = 0
        for value in self.search_paths():
            base = runtime_path(value, self.root)
            if not base.exists():
                continue
            safe_path(base, "probe")
            def unreadable(error: OSError) -> None:
                raise DomainError("MODEL_SEARCH_UNREADABLE", "Configured model search root cannot be read", 503)
            for parent, directories, names in os.walk(base, followlinks=False, onerror=unreadable):
                directories[:] = sorted(d for d in directories if not is_link(Path(parent) / d))
                for name in sorted(names):
                    count += 1
                    if count > 50000:
                        raise DomainError("MODEL_SEARCH_LIMIT", "Configured model search exceeds 50000 files")
                    if name in record.filenames:
                        yield safe_path(base, (Path(parent) / name).relative_to(base).as_posix())

    def trusted_path(self, record: ModelRecord) -> Path | None:
        """A physical weight in app/models that matches ``record`` by name, trusted by its own digest.

        Used only when the pinned lookup fails (missing, different export, FP16 variant, renamed file).
        """
        if self.trust is None:
            return None
        names = [*record.filenames, record.model_id, getattr(record, "name", "")]
        path = self.trust.locate(names)
        if path is None:
            return None
        try:
            self.trust.trust(path)
        except OSError:
            return None
        return path

    def trusted_identity(self, path: Path | None) -> tuple[str, int] | None:
        """``(sha256, size)`` of an auto-trusted file, else ``None`` (the registry digest applies)."""
        entry = self.trust.lookup(path) if self.trust is not None else None
        return (entry.sha256, entry.size) if entry else None

    def resolve(self, model: str | ModelRecord) -> Path:
        record = self.registry.get(model) if isinstance(model, str) else model
        try:
            return self._resolve_pinned(record)
        except DomainError:
            # A weight the user put in app/models is never refused for lacking the pinned digest.
            trusted = self.trusted_path(record)
            if trusted is None:
                raise
            return trusted

    def _resolve_pinned(self, record: ModelRecord) -> Path:
        # The locator iterator returns immediately after its first authoritative
        # path; a mutated preserved source can never fall through to another copy.
        mismatch = None
        for path in self._locations(record):
            try:
                return verify_file(path, record.expected_sha256, record.byte_size)  # type: ignore[arg-type]
            except DomainError as error:
                mismatch = error
        if mismatch:
            raise mismatch
        raise DomainError("MODEL_NOT_INSTALLED", "No matching read-only local weight is available", 404)

    @contextmanager
    def open(self, model: str | ModelRecord, *, accepted_license: bool = False,
             commercial: bool = False) -> Iterator[BinaryIO]:
        record = self.registry.get(model) if isinstance(model, str) else model
        path = None
        try:
            authorize(record, action="activate", accepted_license=accepted_license, commercial=commercial)
            path = self.resolve(record)
        except DomainError:
            path = self.trusted_path(record)
            if path is None:
                raise
        identity = self.trusted_identity(path)
        if identity is not None:
            with verified_stream(path, identity[0], identity[1]) as stream:
                yield stream
            return
        from roop_3d.models.governance import ModelCompatibilityScanner
        governance = ModelCompatibilityScanner().scan(record, installed=True, integrity_verified=True)
        if governance["lifecycle_state"] != "READY":
            code = "LICENSE_UNVERIFIED" if governance["lifecycle_state"] == "LICENSE_RESTRICTED" else "MODEL_INCOMPATIBLE"
            raise DomainError(code, "Model is not eligible for inference: " + "; ".join(governance["issues"]), 409)
        with verified_stream(path, record.expected_sha256, record.byte_size) as stream:  # type: ignore[arg-type]
            yield stream
