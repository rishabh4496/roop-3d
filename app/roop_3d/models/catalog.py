"""Compatibility summaries backed by the authoritative Stage 03 registry."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable

from roop_3d.audit.project_paths import application_root
from roop_3d.core.errors import DomainError
from roop_3d.models.downloader import Downloader, DownloadResult
from roop_3d.models.governance import ModelCompatibilityScanner
from roop_3d.models.local_provider import LocalWeightProvider
from roop_3d.models.local_trust import LocalTrust, MIN_TRUST_BYTES
from roop_3d.models.registry import ModelRecord, ModelRegistry, authorize, license_info
from roop_3d.core.storage import atomic_json
from roop_3d.audit.isolation import is_link

# Catalog reads may span several workspaces. Reuse hash evidence only while the
# exact file identity is unchanged. Inference still verifies through the provider.
_CATALOG_VERIFIED: dict[tuple, tuple] = {}

# SHA-256 of weights whose bytes were observed here, keyed by exact file identity. Used only to *report* which
# registry artifact a resolved file is; it never grants trust (that stays with LocalTrust / the pinned verifier).
_OBSERVED_DIGESTS: dict[tuple[str, int, int], str] = {}
_OBSERVED_LOCK = threading.Lock()


def observed_digest(path: Path) -> tuple[str, int]:
    """``(sha256, size)`` of ``path``, hashed once per (path, size, mtime) and re-hashed after any change."""
    path = Path(path)
    before = path.stat()
    key = (str(path), before.st_size, before.st_mtime_ns)
    with _OBSERVED_LOCK:
        cached = _OBSERVED_DIGESTS.get(key)
    if cached is not None:
        return cached, before.st_size
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    after = path.stat()
    if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
        raise DomainError("MODEL_CHANGED", "Model file changed while its identity was being observed", 409)
    with _OBSERVED_LOCK:
        _OBSERVED_DIGESTS[key] = digest.hexdigest()
    return digest.hexdigest(), before.st_size


def render_dependencies(catalog) -> dict[str, str]:
    """Snapshot reference/stream model candidates, including optional local adapters.

    These are dependency identities, not a claim that every candidate ran. Actual activation
    still verifies bytes and reports providers through the existing inference adapters.
    """
    names = {"2d106det.onnx", "1k3d68.onnx", "bisenet_resnet_34.onnx"}
    selected = [catalog.detector(), catalog.recognizer()]
    required = {record["model_id"] for record in selected}
    seen = set()
    for record in catalog.records():
        if record["name"] in names and record["name"] not in seen:
            selected.append(record)
            seen.add(record["name"])
    records = {record["model_id"]: record for record in selected}
    available = {}
    for key, record in records.items():
        try:
            catalog.verify(record)
        except DomainError as error:
            if key not in required and error.code == "MODEL_NOT_INSTALLED":
                continue
            raise
        available[key] = record["sha256"]
    return available


class ModelCatalog:
    def __init__(self, root: Path | None = None, *, cache_root: Path | None = None,
                 search_paths: Callable[[], tuple[str, ...]] | None = None,
                 registry: ModelRegistry | None = None):
        self.root = root or application_root()
        self.registry = registry or ModelRegistry(self.root)
        # Weights the user placed in app/models are auto-trusted by their own digest (models/local_trust.py).
        self.local_trust = LocalTrust(lambda: self.root / "models")
        self.provider = LocalWeightProvider(self.registry, root=cache_root or self.root, search_paths=search_paths,
                                            trust=self.local_trust)
        self.downloader = Downloader(cache_root or self.root)
        self.governance = ModelCompatibilityScanner()
        self.pin_root = cache_root or self.root
        self.unblock_local_models = True
        self._scan_lock = threading.RLock()
        self._local_files: dict[str, Path] = {}
        self._scanned_paths: tuple[Path, ...] = ()
        self._trt_cache_names: tuple[str, ...] = ()
        self._scan_complete = False
        self._vram_loaded: set[str] = set()
        self._provider_files: dict[str, tuple[Path | None, tuple | None]] = {}
        try:
            supported_path = self.root / "roop_3d/supported_models.json"
            if not supported_path.is_file():
                supported_path = Path(__file__).resolve().parents[1] / "supported_models.json"
            supported = json.loads(supported_path.read_text("utf-8"))
            if supported.get("schema_version") != 1 or not isinstance(supported.get("models"), list):
                raise ValueError("Unsupported supported-model catalog")
            self._supported = tuple(supported["models"])
        except (OSError, ValueError, TypeError):
            raise DomainError("MODEL_CATALOG_INVALID", "Supported model catalog is invalid", 503) from None

    @staticmethod
    def _stem(path: Path) -> str:
        return path.stem.casefold().replace(" ", "_")

    def rescan_local(self) -> dict[str, Any]:
        """Discover local model files under app/models without following links.

        This is a visibility/selection scan. The runtime adapter separately checks
        ONNX graph safety before activating an unregistered local file.
        """
        models_root = self.root / "models"
        self.local_trust.invalidate()
        found: list[Path] = []
        if models_root.is_dir() and not is_link(models_root):
            count = 0
            for parent, directories, names in os.walk(models_root, followlinks=False):
                parent_path = Path(parent)
                directories[:] = sorted(name for name in directories
                    if name.casefold() != ".trt_cache" and not is_link(parent_path / name))
                for name in sorted(names):
                    count += 1
                    if count > 50000:
                        raise DomainError("MODEL_SEARCH_LIMIT", "Local model scan exceeds 50000 files", 413)
                    candidate = parent_path / name
                    if candidate.suffix.casefold() not in {".onnx", ".pth", ".trt"} or is_link(candidate):
                        continue
                    try:
                        if candidate.is_file() and candidate.stat().st_size > 0:
                            found.append(candidate.resolve(strict=True))
                    except OSError:
                        continue
        trt_root = models_root / ".trt_cache"
        trt_names = tuple(path.name.casefold() for path in trt_root.iterdir() if path.is_file()) if trt_root.is_dir() else ()
        aliases: dict[str, list[dict[str, Any]]] = {}
        for spec in self._supported:
            for alias in (spec["id"], *spec.get("aliases", ())):
                aliases.setdefault(str(alias).casefold().replace(" ", "_"), []).append(spec)
        local: dict[str, Path] = {}
        matched: set[Path] = set()
        for path in found:
            key = self._stem(path)
            for spec in aliases.get(key, ()):
                existing = local.get(spec["id"])
                priority = {".onnx": 0, ".trt": 1, ".pth": 2}
                if existing is None or priority[path.suffix.casefold()] < priority[existing.suffix.casefold()]:
                    local[spec["id"]] = path
                matched.add(path)
        custom_rows = []
        for path in found:
            if path in matched:
                continue
            relative = path.relative_to(models_root.resolve()).as_posix()
            slug = re.sub(r"[^a-z0-9]+", "_", path.stem.casefold()).strip("_")[:64] or "model"
            suffix = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:8]
            model_id = f"local_{slug}_{suffix}"
            local[model_id] = path
            custom_rows.append({"id": model_id, "model_id": model_id, "name": path.stem,
                "category": "custom", "native_resolution": None, "installed": True, "selectable": True,
                "vram_loaded": False, "trt_cached": path.suffix.casefold() == ".trt", "download_url": None,
                "local_format": path.suffix[1:].casefold(), "state": "CANDIDATE",
                "family_hint": "custom", "license_state": "UNKNOWN",
                "provenance_state": "UNKNOWN", "inference_state": "UNKNOWN",
                "commercial_permission": "UNKNOWN", "sha256": None, "size_bytes": path.stat().st_size,
                "verified_at": None, "role": None, "license_id": "UNKNOWN",
                "commercial_use": "unknown", "license_notice": "Local custom model; runtime graph checks apply",
                "manual_acceptance_required": False})
        with self._scan_lock:
            self._provider_files.clear()
            self._local_files = local
            self._scanned_paths = tuple(found)
            self._trt_cache_names = trt_names
            self._scan_complete = True
        return {"scanned_files": len(found), "registered_local_models": len(local),
                "custom_local_models": custom_rows, "unblock_local_models": self.unblock_local_models}

    def local_path(self, model_id: str) -> Path | None:
        with self._scan_lock:
            return self._local_files.get(model_id)

    def _records_for_spec(self, spec: dict) -> list[ModelRecord]:
        aliases = [str(v).casefold() for v in (spec['id'], *spec.get('aliases', ()))]
        matches = [record for record in self.registry.records() if record.model_id == spec['id'] or
                   any(Path(name).stem.casefold() in aliases for name in record.filenames)]
        return sorted(matches, key=lambda record: (not bool(record.expected_sha256), min(
            (aliases.index(Path(name).stem.casefold()) for name in record.filenames if Path(name).stem.casefold() in aliases),
            default=len(aliases))))

    def weight_available(self, model_id: str, filename: str) -> bool:
        """True when ``filename`` is a registered artifact of the supported entry ``model_id`` that resolves through the
        hash/trust-verifying provider (the same check that makes a rack row VERIFIED_LOCAL)."""
        spec = next((item for item in self._supported if item["id"] == model_id), None)
        if spec is None:
            return False
        wanted = Path(filename).name.casefold()
        return any(wanted in {Path(name).name.casefold() for name in record.filenames}
                   and self._provider_path(record) is not None for record in self._records_for_spec(spec))

    def resolve_local_model(self, model_id: str) -> Path:
        """Resolve a local weight only when it is eligible for inference."""
        if not self.unblock_local_models:
            raise DomainError("LOCAL_MODEL_BLOCKED", "Local model activation is disabled", 403)
        blocked = self._blocked_reason(model_id)
        if blocked is not None:
            raise DomainError("MODEL_BLOCKED", blocked, 403)
        path = self.local_path(model_id)
        if path is None and not self._scan_complete:
            self.rescan_local()
            path = self.local_path(model_id)
        if path is None or path.suffix.casefold() != ".onnx":
            # Registry/search-path artifacts must pass exact-byte verification and the inference governance gate.
            # Files placed directly in app/models use the explicit local-trust path above.
            # Installed bytes alone do not establish rights, compatibility, or runtime readiness.
            # The inference resolver enforces local trust, license, provenance, and compatibility policy.
            # Preserve a structured refusal instead of opening an unreviewed model session.
            spec = next((item for item in self._supported if item['id'] == model_id), None)
            if spec is not None:
                for record in self._records_for_spec(spec):
                    candidate = self._provider_path(record)
                    if candidate is not None and candidate.suffix.casefold() == '.onnx':
                        return self.resolve_for_inference(record.model_id)
            raise DomainError("MODEL_NOT_INSTALLED", self.missing_reason(model_id), 404)
        try:
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to((self.root / "models").resolve(strict=True)) or not resolved.is_file():
                raise ValueError
        except (OSError, ValueError):
            raise DomainError("MODEL_PATH_INVALID", "Local model path is outside app/models", 403) from None
        return resolved

    @staticmethod
    def _file_identity(path: Path) -> tuple:
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino, stat.st_dev

    def _provider_path(self, record: ModelRecord) -> Path | None:
        """Cache catalog verification until rescan or file mutation; activation rehashes."""
        with self._scan_lock:
            previous = self._provider_files.get(record.model_id)
            if previous is not None:
                path, identity = previous
                try:
                    if path is None or self._file_identity(path) == identity:
                        return path
                except OSError:
                    pass
            try:
                from roop_3d.models.verifier import verify_file
                for path in self.provider._locations(record):
                    identity = self._file_identity(path)
                    key = str(path), record.expected_sha256, record.byte_size
                    if _CATALOG_VERIFIED.get(key) != identity:
                        verify_file(path, record.expected_sha256, record.byte_size)
                        _CATALOG_VERIFIED[key] = identity
                    self._provider_files[record.model_id] = path, identity
                    return path
            except (DomainError, OSError):
                pass
            # Not the pinned bytes: a file the user placed in app/models is trusted by its own digest.
            try:
                trusted = self.provider.trusted_path(record)
                if trusted is not None:
                    self._provider_files[record.model_id] = trusted, self._file_identity(trusted)
                    return trusted
            except OSError:
                pass
            self._provider_files[record.model_id] = None, None
            return None

    def _trust_flags(self, provider_path: Path | None, on_disk: Path | None) -> dict[str, Any]:
        """Per-row verification flags. A weight present in app/models is verified, approved and never blocked."""
        local = on_disk is not None
        return {"verified": provider_path is not None or local,
                "governance_approved": True if local else None,
                "blocked": False,
                "local_trust": local}

    def is_local_trusted(self, path: Path | None) -> bool:
        return self.provider.trusted_identity(path) is not None

    def _supported_spec(self, spec_id: str) -> dict | None:
        return next((item for item in self._supported if item["id"] == spec_id), None)

    def artifact_identity(self, path: Path, spec_id: str | None = None) -> dict[str, Any]:
        """Which registry artifact the bytes at ``path`` are, decided by their own SHA-256 and size (never by file name).

        ``REGISTRY_PINNED_SHA256``: digest and size equal a verified, non-blocked registry record, so that artifact's
        reviewed identity applies. ``LOCAL_DIGEST_UNREVIEWED``: bytes the registry cannot describe; they may only run
        under the explicit local-use policy for files placed in app/models. Reporting only, it grants nothing."""
        path = Path(path)
        digest, size = self.provider.trusted_identity(path) or observed_digest(path)
        matches = [record for record in self.registry.records()
                   if record.status in {"verified_local", "verified_remote"}
                   and record.expected_sha256 == digest and record.byte_size == size]
        spec = self._supported_spec(spec_id) if spec_id else None
        preferred = {record.model_id for record in self._records_for_spec(spec)} if spec else set()
        match = next((record for record in matches if record.model_id in preferred), matches[0] if matches else None)
        return {"path": str(path), "filename": path.name, "sha256": digest, "size_bytes": size,
                "registry_model_id": match.model_id if match else None,
                "registry_name": match.name if match else None,
                "artifact_state": "VERIFIED_LOCAL" if match else "LOCAL_UNVERIFIED",
                "trust_basis": "REGISTRY_PINNED_SHA256" if match else "LOCAL_DIGEST_UNREVIEWED"}

    def missing_reason(self, model_id: str) -> str:
        """Why a supported model has no usable weight, naming what was searched and what the registry can verify."""
        spec = self._supported_spec(model_id)
        if spec is None:
            return "Selected model has no installed local ONNX weights"
        records = self._records_for_spec(spec)
        names = sorted({*(name for record in records for name in record.filenames),
                        str(spec.get("aliases", [spec["id"]])[0]) + ".onnx"})
        pinned = [record for record in records if record.expected_sha256 and record.byte_size]
        if not pinned:
            return (f"Selected model has no installed local ONNX weights: none of {', '.join(names)} was found under "
                    "app/models, and the registry holds only an unreviewed catalog candidate with no SHA-256, size or "
                    "public source, so nothing can be verified or downloaded. Place the ONNX file in app/models to run it "
                    "under the local-use policy.")
        return (f"Selected model has no installed local ONNX weights: the {len(pinned)} registered artifact(s) "
                f"({', '.join(names)}) were not found at their locators, in the download cache or in app/models.")

    def _blocked_reason(self, model_id: str) -> str | None:
        """Non-None when the supported entry has registry records and every one is blocked (never overridden by placement)."""
        spec = self._supported_spec(model_id)
        records = self._records_for_spec(spec) if spec is not None else []
        if records and all(record.status == "blocked" for record in records):
            return "Every registry record for this model is blocked: " + (records[0].license_notice or "license forbids use")
        return None

    def set_vram_loaded(self, model_id: str, loaded: bool) -> None:
        with self._scan_lock:
            if loaded:
                self._vram_loaded.add(model_id)
            else:
                self._vram_loaded.discard(model_id)

    @staticmethod
    def _legacy(record: ModelRecord) -> dict[str, Any]:
        return {"model_id": record.model_id, "name": record.name, "sha256": record.expected_sha256,
                "size_bytes": record.byte_size, "state": record.status.upper(), "family_hint": record.family_hint,
                "upstream_url": record.upstream_url,
                "verified_at": record.verified_at, "license_state": "UNKNOWN" if record.license_id == "UNKNOWN" else "VERIFIED_REMOTE",
                "provenance_state": record.provenance_state, "inference_state": "UNKNOWN",
                "commercial_permission": "UNKNOWN", "graph": record.input_output_contract,
                "locator": record.locator.model_dump() if record.locator else None,
                "role": record.role, "license_id": record.license_id, "commercial_use": record.commercial_use,
                "license_notice": record.license_notice, "manual_acceptance_required": record.manual_acceptance_required}

    def records(self) -> list[dict[str, Any]]:
        return [self._legacy(record) for record in self.registry.records() if record.status == "verified_local"]

    def _adapter_support(self, category: str, model_id: str) -> dict[str, Any]:
        """Row fields kept for API shape: there is no model-adapter architecture contract any more (the swapper adapters were removed)."""
        return {"adapter_supported": None, "adapter_unsupported_reason": None}

    def _selection(self, category: str, model_id: str) -> dict[str, Any]:
        """Row fields for ``selectable``: consumer-side checks (installed / verified / blocked flags) decide; presence is not eligibility."""
        return {"selectable": True}

    def public(self) -> list[dict[str, Any]]:
        if not self._scan_complete:
            self.rescan_local()
        registered = [self._legacy(item) for item in self.registry.records()]
        by_name: dict[str, dict[str, Any]] = {}
        for record in registered:
            by_name[record["name"].casefold()] = record
        consumed: set[str] = set()
        rows: list[dict[str, Any]] = []
        local = self._local_files
        for spec in self._supported:
            aliases = {str(value).casefold() for value in (spec["id"], spec["name"], *spec.get("aliases", ()))}
            candidates = self._records_for_spec(spec)
            resolved = next((record for record in candidates if self._provider_path(record) is not None), None)
            source = self._legacy(resolved or candidates[0]) if candidates else {}
            if source:
                consumed.add(source["model_id"])
            local_path = local.get(spec["id"])
            provider_path = self._provider_path(self.registry.get(source['model_id'])) if source else None
            expected_names = {Path(filename).name.casefold()
                              for filename in (self.registry.get(source["model_id"]).filenames if source else ())}
            installed = local_path is not None or provider_path is not None or any(path.name.casefold() in expected_names for path in self._scanned_paths)
            # A weight physically present in app/models is verified, approved and unblocked (local_trust.py).
            on_disk = provider_path if provider_path is not None and self.local_trust.inside_root(provider_path) else None
            if on_disk is None and local_path is not None and local_path.stat().st_size > MIN_TRUST_BYTES:
                on_disk = local_path
            if on_disk is not None and provider_path is None:
                provider_path = on_disk
            observed = self.provider.trusted_identity(on_disk) if on_disk is not None else None
            trt_cached = (local_path is not None and local_path.suffix.casefold() == ".trt") or any(
                any(part in name for part in aliases) for name in self._trt_cache_names)
            rows.append({"id": spec["id"], "model_id": spec["id"], "name": spec["name"],
                **self._trust_flags(provider_path, on_disk), **self._adapter_support(spec["category"], spec["id"]),
                "category": spec["category"], "native_resolution": spec.get("native_resolution"),
                "installed": bool(installed), **self._selection(spec["category"], spec["id"]), "vram_loaded": spec["id"] in self._vram_loaded,
                "trt_cached": trt_cached, "download_url": source.get("upstream_url") if source else None,
                "sha256": source.get("sha256") or (observed[0] if observed else None),
                "size_bytes": source.get("size_bytes") or (observed[1] if observed else (local_path.stat().st_size if local_path else None)),
                "state": "VERIFIED_LOCAL" if provider_path else ("CANDIDATE" if source.get('state') == 'VERIFIED_LOCAL' else source.get('state', 'CANDIDATE')), "family_hint": source.get("family_hint", spec["category"]),
                "verified_at": source.get("verified_at"), "license_state": source.get("license_state", "UNKNOWN"),
                "provenance_state": source.get("provenance_state", "UNKNOWN"),
                "inference_state": source.get("inference_state", "UNKNOWN"),
                "commercial_permission": source.get("commercial_permission", "UNKNOWN"),
                "role": source.get("role"),
                "license_id": source.get("license_id", "UNKNOWN"),
                "commercial_use": source.get("commercial_use", "unknown"),
                "license_notice": source.get("license_notice", "License and provenance have not been reviewed"),
                "manual_acceptance_required": source.get("manual_acceptance_required", False),
                "local_format": (local_path or provider_path).suffix[1:].casefold() if local_path or provider_path else None})
        # Preserve every other first-party registry artifact in the API catalog too.
        supported_ids = {spec["id"] for spec in self._supported}
        for source in registered:
            if source["model_id"] in consumed or source["model_id"] in supported_ids:
                continue
            record = self.registry.get(source["model_id"])
            expected_names = {Path(name).name.casefold() for name in record.filenames}
            provider_path = self._provider_path(record)
            installed = provider_path is not None or any(path.name.casefold() in expected_names for path in self._scanned_paths)
            category = {"detector": "detector", "occlusion_mask": "occluder", "frame_upscaler": "upscaler"}.get(source.get("role"), "other")
            on_disk = provider_path if provider_path is not None and self.local_trust.inside_root(provider_path) else None
            rows.append({**{key: value for key, value in source.items() if key not in {"graph", "locator", "upstream_url"}},
                **({"state": "VERIFIED_LOCAL", "sha256": self.provider.trusted_identity(on_disk)[0]}
                   if on_disk is not None and self.provider.trusted_identity(on_disk) else {}),
                **self._trust_flags(provider_path, on_disk), **self._adapter_support(category, source["model_id"]),
                "id": source["model_id"], "category": category, "native_resolution": None,
                "installed": installed, **self._selection(category, source["model_id"]), "vram_loaded": source["model_id"] in self._vram_loaded,
                "trt_cached": False, "download_url": record.upstream_url, "local_format": None})
        # Unknown files are listed as candidates.
        known_ids = {row["id"] for row in rows}
        for model_id, path in sorted(local.items()):
            if model_id in known_ids or any(spec["id"] == model_id for spec in self._supported):
                continue
            relative = path.relative_to((self.root / "models").resolve()).as_posix()
            slug = re.sub(r"[^a-z0-9]+", "_", path.stem.casefold()).strip("_")[:64] or "model"
            row_id = f"local_{slug}_{hashlib.sha256(relative.encode()).hexdigest()[:8]}"
            rows.append({"id": row_id, "model_id": row_id, "name": path.stem, "category": "custom",
                "native_resolution": None, "installed": True, **self._selection("custom", row_id), "vram_loaded": row_id in self._vram_loaded,
                "trt_cached": path.suffix.casefold() == ".trt", "download_url": None,
                "sha256": None, "size_bytes": path.stat().st_size,
                "state": "VERIFIED_LOCAL" if path.stat().st_size > MIN_TRUST_BYTES else "CANDIDATE",
                **self._trust_flags(path, path if path.stat().st_size > MIN_TRUST_BYTES else None),
                **self._adapter_support("custom", row_id),
                "family_hint": "custom", "verified_at": None, "license_state": "UNKNOWN",
                "provenance_state": "UNKNOWN", "inference_state": "UNKNOWN",
                "commercial_permission": "UNKNOWN", "role": None, "license_id": "UNKNOWN",
                "commercial_use": "unknown", "license_notice": "Local custom model; runtime graph checks apply",
                "manual_acceptance_required": False, "local_format": path.suffix[1:].casefold()})
        return rows

    def get(self, model_id: str) -> dict[str, Any]:
        return self._legacy(self.registry.get(model_id))

    def _path(self, record: dict[str, Any]) -> Path:
        return self.provider.resolve(record["model_id"])

    def verify(self, record: dict[str, Any]) -> None:
        self.provider.resolve(record["model_id"])

    def verified_bytes(self, record: dict[str, Any]) -> bytes:
        model = self.registry.get(record["model_id"])
        if model.input_output_contract.get("external_data"):
            raise DomainError("MODEL_UNSUPPORTED", "Detector cannot load external tensor files")
        path = self.resolve_for_inference(model.model_id)
        with self.provider.open(model) as stream:
            data = stream.read()
        expected = self.provider.trusted_identity(path) or (model.expected_sha256, model.byte_size)
        if len(data) != expected[1] or hashlib.sha256(data).hexdigest() != expected[0]:
            raise DomainError("MODEL_DRIFT", "Model integrity differs from the preserved registry")
        return data

    def ensure_available(self, model_id: str, *, accepted_license: bool = False,
                         commercial: bool = False) -> DownloadResult:
        result = self._ensure_available(model_id, accepted_license=accepted_license, commercial=commercial)
        self.rescan_local()
        return result

    def _ensure_available(self, model_id: str, *, accepted_license: bool = False,
                          commercial: bool = False) -> DownloadResult:
        model = self.registry.get(model_id)
        authorize(model, action="activate", accepted_license=accepted_license, commercial=commercial)
        if model.expected_sha256 and model.byte_size:
            try:
                return DownloadResult(self.provider.resolve(model), False, False)
            except DomainError as error:
                if error.code != "MODEL_NOT_INSTALLED":
                    raise
            cached = self.downloader.cached(model)
            if cached:
                return DownloadResult(cached, False, False)
        return self.downloader.download(model, accepted_license=accepted_license, commercial=commercial)

    def lifecycle(self, model_id: str, *, hardware: dict[str, Any] | None = None) -> dict[str, Any]:
        """Governance state, except that a weight the user placed in app/models runs under the local-use policy.

        That override is explicit in the result: ``governance_state``/``governance_issues`` keep what the governance scan
        found, and ``trust_basis`` says whether the bytes are a reviewed registry artifact (``REGISTRY_PINNED_SHA256``,
        decided by the file's own SHA-256, never by its name) or only trusted by placement (``LOCAL_DIGEST_UNREVIEWED``).
        A blocked (license-forbidden) record is never overridden."""
        result = self._lifecycle(model_id, hardware=hardware)
        if result.get("lifecycle_state") != "READY":
            model = self.registry.get(model_id)
            path = self.provider.trusted_path(model)
            if path is not None:
                identity = self.artifact_identity(path)
                return {**result, "lifecycle_state": "READY", "issues": [], "integrity_verified": True,
                        "governance_state": result.get("lifecycle_state"),
                        "governance_issues": list(result.get("issues", ())),
                        "installed": True, "local_trust": True, "governance_approved": True, "blocked": False,
                        "trust_policy": "LOCAL_USE_TRUST",
                        "trust_basis": identity["trust_basis"], "observed_sha256": identity["sha256"],
                        "registry_match": identity["registry_model_id"]}
        return {**result, "trust_policy": "REGISTRY_GOVERNANCE"}

    def _lifecycle(self, model_id: str, *, hardware: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return governance state; resolving verifies bytes and never loads the model."""
        model = self.registry.get(model_id)
        if model.status == "candidate" or not model.expected_sha256 or not model.byte_size:
            return self.governance.scan(model, installed=False, integrity_verified=None, hardware=hardware)
        try:
            self.provider.resolve(model)
        except DomainError as error:
            if error.code == "MODEL_NOT_INSTALLED":
                return self.governance.scan(model, installed=False, integrity_verified=None, hardware=hardware)
            if error.code == "MODEL_QUARANTINED":
                result = self.governance.scan(model, installed=True, integrity_verified=False, hardware=hardware)
                result["lifecycle_state"] = "QUARANTINED"
                return result
            raise
        return self.governance.scan(model, installed=True, integrity_verified=True, hardware=hardware)

    def resolve_for_inference(self, model_id: str, *, hardware: dict[str, Any] | None = None,
                              precision: str | None = None) -> Path:
        """Resolve only a hash-verified artifact that passes the governance gate."""
        model = self.registry.get(model_id)
        path = self.provider.resolve(model)
        if self.is_local_trusted(path):
            # The user placed this file in app/models: no license/compatibility gate may switch the model off.
            return path
        result = self.governance.scan(model, installed=True, integrity_verified=True,
                                      hardware=hardware, precision=precision)
        if result["lifecycle_state"] != "READY":
            code = "LICENSE_UNVERIFIED" if result["lifecycle_state"] == "LICENSE_RESTRICTED" else "MODEL_INCOMPATIBLE"
            raise DomainError(code, "Model is not eligible for inference: " + "; ".join(result["issues"]), 409)
        authorize(model, action="activate")
        return path

    def pin_version(self, family: str, model_id: str, *, hardware: dict[str, Any] | None = None) -> dict[str, Any]:
        """Atomically pin an already-installed, fully governed immutable version."""
        if not family or len(family) > 96 or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in family):
            raise DomainError("MODEL_VERSION_FAMILY_INVALID", "Version family must be a portable lowercase identifier", 422)
        record = self.registry.get(model_id)
        if record.version_family != family:
            raise DomainError("MODEL_VERSION_FAMILY_MISMATCH", "Model is not registered in this version family", 409)
        state = self.lifecycle(model_id, hardware=hardware)
        if state["lifecycle_state"] != "READY":
            raise DomainError("MODEL_VERSION_NOT_READY", "Only installed, verified, compatible, licensed versions can be pinned", 409)
        relative = "roop_3d/.runtime/model-pins.json"
        path = self.pin_root / relative
        try:
            data = json.loads(path.read_text("utf-8")) if path.exists() else {"schema_version": 1, "families": {}}
            families = data["families"]
            history = list(families.get(family, []))
        except (OSError, ValueError, KeyError, TypeError):
            raise DomainError("MODEL_PINS_INVALID", "Model version pin state is invalid", 503) from None
        if not history or history[-1] != model_id:
            history.append(model_id)
        atomic_json(self.pin_root, relative, {"schema_version": 1, "families": {**families, family: history}})
        return {"family": family, "active_model_id": model_id, "previous_model_id": history[-2] if len(history) > 1 else None}

    def update_version(self, family: str, model_id: str, *, accepted_license: bool = False,
                       commercial: bool = False, hardware: dict[str, Any] | None = None) -> dict[str, Any]:
        """Download an explicitly selected registered revision, then pin it after readiness checks."""
        record = self.registry.get(model_id)
        if record.version_family != family:
            raise DomainError("MODEL_VERSION_FAMILY_MISMATCH", "Model is not registered in this version family", 409)
        result = self.ensure_available(model_id, accepted_license=accepted_license, commercial=commercial)
        pin = self.pin_version(family, model_id, hardware=hardware)
        return {**pin, "downloaded": result.downloaded, "resumed": result.resumed}

    def rollback_version(self, family: str) -> dict[str, Any]:
        """Restore the previous registered pin without replacing either artifact."""
        relative = "roop_3d/.runtime/model-pins.json"
        path = self.pin_root / relative
        try:
            data = json.loads(path.read_text("utf-8"))
            history = list(data["families"][family])
        except (OSError, ValueError, KeyError, TypeError):
            raise DomainError("MODEL_VERSION_ROLLBACK_UNAVAILABLE", "No previous model version pin is available", 409) from None
        if len(history) < 2:
            raise DomainError("MODEL_VERSION_ROLLBACK_UNAVAILABLE", "No previous model version pin is available", 409)
        previous = history[-2]
        state = self.lifecycle(previous)
        if state["lifecycle_state"] != "READY":
            raise DomainError("MODEL_VERSION_ROLLBACK_UNAVAILABLE", "Previous model version is no longer verified and compatible", 409)
        history.append(previous)
        atomic_json(self.pin_root, relative, {"schema_version": 1,
                    "families": {**data["families"], family: history}})
        return {"family": family, "active_model_id": previous, "rolled_back": True}

    def pinned_version(self, family: str) -> ModelRecord:
        relative = "roop_3d/.runtime/model-pins.json"
        try:
            data = json.loads((self.pin_root / relative).read_text("utf-8"))
            model_id = data["families"][family][-1]
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            raise DomainError("MODEL_VERSION_PIN_NOT_FOUND", "No model version is pinned for this family", 404) from None
        return self.registry.get(model_id)

    def select_version(self, record: ModelRecord) -> ModelRecord:
        """Use an explicit family pin when one exists; otherwise preserve registry order."""
        if record.version_family == "UNKNOWN":
            return record
        try:
            pinned = self.pinned_version(record.version_family)
        except DomainError as error:
            if error.code == "MODEL_VERSION_PIN_NOT_FOUND":
                return record
            raise
        return pinned if pinned.role == record.role and pinned.version_family == record.version_family else record

    def license(self, model_id: str) -> dict[str, Any]:
        return license_info(self.registry.get(model_id))

    def detector(self, model_id: str = 'scrfd') -> dict[str, Any]:
        names = {
            'scrfd': ('scrfd_10g_bnkps.onnx', 'scrfd_2.5g_kps.onnx', 'det_10g.onnx'),
            'retinaface': ('retinaface_10g.onnx', 'retinaface_r50.onnx'),
            'yoloface': ('yoloface_8n.onnx',),
        }
        if model_id not in names:
            raise DomainError('DETECTOR_UNAVAILABLE', 'Unknown face detector selection', 422)
        for name in names[model_id]:
            for registered in self.registry.records():
                record = self._legacy(self.select_version(registered))
                if record['name'] == name and self._provider_path(self.registry.get(record['model_id'])) is not None:
                    return record
        # Preserve metadata-only catalog use and let activation surface missing weights.
        for registered in self.registry.records():
            record = self.select_version(registered)
            record = self._legacy(record)
            if record['name'] in names[model_id] and len(record['graph'].get('outputs', [])) == (1 if model_id == 'yoloface' else 9):
                return record
        raise DomainError("DETECTOR_UNAVAILABLE", f"No supported local {model_id} model is registered", 503)

    def recognizer(self) -> dict[str, Any]:
        for preferred in ("w600k_r50.onnx", "arcface_w600k_r50.onnx", "w600k_mbf.onnx"):
            for registered in self.registry.records():
                record = self._legacy(self.select_version(registered))
                if record["name"] == preferred and record.get("role") == "recognizer":
                    return record
        for registered in self.registry.records():
            record = self._legacy(self.select_version(registered))
            if record.get("role") == "recognizer":
                return record
        raise DomainError("RECOGNIZER_UNAVAILABLE", "No supported local recognizer model is registered", 503)
