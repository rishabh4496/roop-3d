"""Hash-bound target-identity locks: (video, inspection, target track) -> authorised source FaceSet.

A lock is created only by an explicit user selection and records every input that
determines which face is replaced and by whom:

* original target video SHA-256 and the inspection (file + identity report) SHA-256,
* the tracking policy fingerprint, detection threshold and detector/recognizer identity,
* the effective pipeline settings that influence tracking/identity transfer,
* for each assignment the track label and the FaceSet id **and archive SHA-256**.

``validate`` recomputes all of it. Any difference marks the lock STALE with machine-readable
reasons; a stale lock is never repaired silently and jobs refuse it. ``digest`` is a
canonical hash of the bound content only (not the lock id, revision or timestamps) so the
same inputs and assignments always reproduce the same digest.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import uuid
from typing import Callable, Literal

from pydantic import Field

from roop_3d.audit.isolation import safe_path, sha256_file
from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract, Settings
from roop_3d.core.storage import atomic_json, exclusive, owned
from roop_3d.engine.detection.track_policy import POLICY_VERSION, TrackPolicy
from roop_3d.identity.report import read_report
from roop_3d.media.schema import FaceInspection, TrackToFaceSetMap

TRACK = r"^track_[0-9]{4}$"
FACESET = r"^[A-Za-z0-9_-]{1,64}$"
SHA = r"^[0-9a-f]{64}$"
LOCK_DIR = "roop_3d/.runtime/identity_locks"
MAX_ASSIGNMENTS = 64


class AssignmentRequest(Contract):
    track_id: str = Field(pattern=TRACK)
    faceset_id: str = Field(pattern=FACESET)


class LockCreate(Contract):
    target_path: str = Field(min_length=1, max_length=1024)
    inspection_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    assignments: tuple[AssignmentRequest, ...] = Field(min_length=1, max_length=MAX_ASSIGNMENTS)
    excluded_tracks: tuple[str, ...] = Field(default=(), max_length=MAX_ASSIGNMENTS)


class LockEdit(Contract):
    revision: int = Field(ge=0, strict=True)
    action: Literal["assign", "unassign", "swap", "exclude", "include"]
    track_id: str = Field(pattern=TRACK)
    other_track_id: str | None = Field(default=None, pattern=TRACK)       # swap partner
    faceset_id: str | None = Field(default=None, pattern=FACESET)         # assign


class LockAssignment(Contract):
    track_id: str = Field(pattern=TRACK)
    faceset_id: str = Field(pattern=FACESET)
    faceset_sha256: str = Field(pattern=SHA)


class IdentityLock(Contract):
    schema_version: Literal[1] = 1
    lock_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    revision: int = Field(ge=0)
    created_at: str
    updated_at: str
    target_path: str
    target_sha256: str = Field(pattern=SHA)
    inspection_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    inspection_sha256: str = Field(pattern=SHA)
    identity_report_sha256: str = Field(pattern=SHA)
    tracking_policy_fingerprint: str = Field(pattern=SHA)
    detection_threshold: float
    detector_model_id: str
    detector_sha256: str = Field(pattern=SHA)
    recognizer_model_id: str
    recognizer_sha256: str = Field(pattern=SHA)
    settings_fingerprint: str = Field(pattern=SHA)
    assignments: tuple[LockAssignment, ...] = Field(max_length=MAX_ASSIGNMENTS)
    excluded_tracks: tuple[str, ...] = Field(default=(), max_length=MAX_ASSIGNMENTS)
    digest: str = Field(pattern=SHA)

    def mappings(self) -> tuple[TrackToFaceSetMap, ...]:
        return tuple(TrackToFaceSetMap(track_id=item.track_id, faceset_id=item.faceset_id)
                     for item in sorted(self.assignments, key=lambda item: item.track_id))


@dataclass(frozen=True)
class LockEnvironment:
    root: Path
    resolve_media: Callable[[str], Path]
    settings: Callable[[], Settings]
    catalog: object            # needs .detector() and .recognizer() -> dict with model_id / sha256


def settings_fingerprint(settings: Settings) -> str:
    """Only settings that change tracking or identity transfer; ports, paths and UI options do not."""
    relevant = {name: getattr(settings, name) for name in (
        "temporal_min_cutoff", "temporal_beta", "temporal_d_cutoff", "pose_hysteresis_degrees", "execution_mode")}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def canonical_digest(content: dict) -> str:
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _bound_content(lock: IdentityLock | dict) -> dict:
    value = lock.model_dump(mode="json") if isinstance(lock, IdentityLock) else dict(lock)
    for volatile in ("lock_id", "revision", "created_at", "updated_at", "digest", "target_path", "schema_version"):
        value.pop(volatile, None)
    value["assignments"] = sorted(value["assignments"], key=lambda item: item["track_id"])
    value["excluded_tracks"] = sorted(value["excluded_tracks"])
    return value


_hashes: dict[tuple[str, int, int], str] = {}


def file_sha256(path: Path, *, fresh: bool = False) -> str:
    status = path.stat()
    key = (str(path), status.st_size, status.st_mtime_ns)
    if fresh or key not in _hashes:
        if len(_hashes) > 128:
            _hashes.clear()
        _hashes[key] = sha256_file(path)[1]
    return _hashes[key]


def faceset_archive(root: Path, faceset_id: str) -> Path:
    """Same resolution order and catalog requirement as the job runner."""
    if not re.fullmatch(FACESET, faceset_id):
        raise DomainError("FACESET_ID_INVALID", "FaceSet identifier is invalid", 422)
    manifest = safe_path(root, f"roop_3d/.runtime/facesets/{faceset_id}.json")
    if not manifest.is_file():
        raise DomainError("FACESET_NOT_FOUND", "Unknown FaceSet", 422)
    for relative in (f"roop_3d/outputs/facesets/{faceset_id}.fsz", f"roop_3d/.runtime/facesets/{faceset_id}.fsz"):
        candidate = safe_path(root, relative)
        if candidate.is_file():
            return candidate
    raise DomainError("FACESET_ARCHIVE_NOT_FOUND", "FaceSet archive is unavailable in owned storage", 422)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _reason(code: str, detail: str, **extra) -> dict:
    return {"code": code, "detail": detail, **extra}


class LockStore:
    def __init__(self, environment: LockEnvironment):
        self.env = environment

    # -- paths -----------------------------------------------------------------------------------
    def _relative(self, lock_id: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{32}", lock_id):
            raise DomainError("IDENTITY_LOCK_NOT_FOUND", "Unknown identity lock", 404)
        return f"{LOCK_DIR}/{lock_id}.json"

    def _inspection_path(self, inspection_id: str) -> Path:
        return safe_path(self.env.root, f"roop_3d/.runtime/inspections/{inspection_id}.json")

    # -- snapshots -------------------------------------------------------------------------------
    def _load_inspection(self, inspection_id: str) -> tuple[FaceInspection, str]:
        path = self._inspection_path(inspection_id)
        if not path.is_file():
            raise DomainError("INSPECTION_NOT_FOUND", "Inspect the target first", 422)
        try:
            inspection = FaceInspection.model_validate_json(path.read_text("utf-8"))
        except ValueError:
            raise DomainError("INSPECTION_INVALID", "Saved inspection is invalid", 422) from None
        return inspection, file_sha256(path, fresh=True)

    def _snapshot_faceset(self, faceset_id: str) -> str:
        return file_sha256(faceset_archive(self.env.root, faceset_id), fresh=True)

    # -- create ----------------------------------------------------------------------------------
    def create(self, value: LockCreate) -> IdentityLock:
        media = self.env.resolve_media(value.target_path)
        target_sha = file_sha256(media, fresh=True)
        inspection, inspection_sha = self._load_inspection(value.inspection_id)
        if inspection.target_sha256 != target_sha:
            raise DomainError("INSPECTION_TARGET_MISMATCH", "Inspection does not belong to the current target video", 409)
        if inspection.tracking_method != POLICY_VERSION or not inspection.identity_report_sha256:
            raise DomainError("IDENTITY_INSPECTION_REQUIRED",
                              "Re-run the target inspection with the identity lock enabled", 422)
        if inspection.tracking_policy_fingerprint != TrackPolicy().fingerprint():
            raise DomainError("IDENTITY_POLICY_CHANGED", "The tracking policy changed since this inspection", 409)
        labels = {track.track_id for track in inspection.tracks}
        self._check_labels(value.assignments, value.excluded_tracks, labels)
        assignments = tuple(LockAssignment(track_id=item.track_id, faceset_id=item.faceset_id,
                                           faceset_sha256=self._snapshot_faceset(item.faceset_id))
                            for item in sorted(value.assignments, key=lambda item: item.track_id))
        settings = self.env.settings()
        detector, recognizer = self.env.catalog.detector(), self.env.catalog.recognizer()
        stamp = _now()
        draft = dict(
            schema_version=1, lock_id=uuid.uuid4().hex, revision=0, created_at=stamp, updated_at=stamp,
            target_path=value.target_path, target_sha256=target_sha, inspection_id=value.inspection_id,
            inspection_sha256=inspection_sha, identity_report_sha256=inspection.identity_report_sha256,
            tracking_policy_fingerprint=inspection.tracking_policy_fingerprint,
            detection_threshold=inspection.detection_threshold if inspection.detection_threshold is not None else 0.5,
            detector_model_id=detector["model_id"], detector_sha256=detector["sha256"],
            recognizer_model_id=recognizer["model_id"], recognizer_sha256=recognizer["sha256"],
            settings_fingerprint=settings_fingerprint(settings),
            assignments=[item.model_dump(mode="json") for item in assignments],
            excluded_tracks=sorted(value.excluded_tracks))
        draft["digest"] = canonical_digest(_bound_content(draft))
        lock = IdentityLock.model_validate(draft)
        # A lock must describe the models/inspection it was created with; reject a mismatch up front.
        if (inspection.detector_sha256 != lock.detector_sha256 or inspection.recognizer_sha256 != lock.recognizer_sha256):
            raise DomainError("IDENTITY_MODEL_CHANGED", "Detector/recognizer differ from the inspection that produced the tracks", 409)
        self._write(lock)
        return lock

    @staticmethod
    def _check_labels(assignments, excluded, labels: set[str]) -> None:
        tracks = [item.track_id for item in assignments]
        if len(set(tracks)) != len(tracks):
            raise DomainError("IDENTITY_TRACK_DUPLICATE", "A target track can be bound to only one source FaceSet", 422)
        unknown = (set(tracks) | set(excluded)) - labels
        if unknown:
            raise DomainError("TRACK_NOT_FOUND", f"Track absent from this inspection: {sorted(unknown)[0]}", 422)
        if set(tracks) & set(excluded):
            raise DomainError("IDENTITY_TRACK_CONFLICT", "A track cannot be both assigned and excluded", 422)

    # -- read / list / delete --------------------------------------------------------------------
    def _write(self, lock: IdentityLock) -> None:
        atomic_json(self.env.root, self._relative(lock.lock_id), lock.model_dump(mode="json"))

    def get(self, lock_id: str) -> IdentityLock:
        path = owned(self.env.root, self._relative(lock_id))
        if not path.is_file():
            raise DomainError("IDENTITY_LOCK_NOT_FOUND", "Unknown identity lock", 404)
        try:
            return IdentityLock.model_validate_json(path.read_text("utf-8"))
        except ValueError:
            raise DomainError("IDENTITY_LOCK_INVALID", "Identity lock file is unreadable", 422) from None

    def list(self, target_path: str | None = None) -> list[IdentityLock]:
        directory = safe_path(self.env.root, LOCK_DIR)
        if not directory.is_dir():
            return []
        found = []
        for path in sorted(directory.glob("*.json"))[:512]:
            try:
                lock = IdentityLock.model_validate_json(path.read_text("utf-8"))
            except (OSError, ValueError):
                continue
            if target_path is None or lock.target_path == target_path:
                found.append(lock)
        return sorted(found, key=lambda item: (item.created_at, item.lock_id))

    def delete(self, lock_id: str) -> None:
        relative = self._relative(lock_id)
        with exclusive(self.env.root, relative + ".lock"):
            path = owned(self.env.root, relative)
            if not path.is_file():
                raise DomainError("IDENTITY_LOCK_NOT_FOUND", "Unknown identity lock", 404)
            path.unlink()

    # -- validation ------------------------------------------------------------------------------
    def validate(self, lock: IdentityLock, *, fresh: bool = False) -> dict:
        """Recompute every bound input. Returns ``{"valid": bool, "reasons": [...]}``."""
        reasons: list[dict] = []
        root = self.env.root
        try:
            media = self.env.resolve_media(lock.target_path)
            if not media.is_file():
                raise DomainError("TARGET_MISSING", "missing", 404)
        except (DomainError, OSError, ValueError):
            reasons.append(_reason("TARGET_MISSING", "The target video is no longer available"))
        else:
            if file_sha256(media, fresh=fresh) != lock.target_sha256:
                reasons.append(_reason("TARGET_CHANGED", "The target video changed after it was inspected"))
        path = self._inspection_path(lock.inspection_id)
        inspection: FaceInspection | None = None
        if not path.is_file():
            reasons.append(_reason("INSPECTION_MISSING", "The saved inspection is missing"))
        elif file_sha256(path, fresh=fresh) != lock.inspection_sha256:
            reasons.append(_reason("INSPECTION_CHANGED", "The saved inspection was modified"))
        else:
            try:
                inspection = FaceInspection.model_validate_json(path.read_text("utf-8"))
            except ValueError:
                reasons.append(_reason("INSPECTION_INVALID", "The saved inspection is invalid"))
        try:
            read_report(root, lock.inspection_id, expected_sha256=lock.identity_report_sha256)
        except DomainError as error:
            reasons.append(_reason("IDENTITY_REPORT_STALE", error.message))
        current_policy = TrackPolicy().fingerprint()
        if lock.tracking_policy_fingerprint != current_policy:
            reasons.append(_reason("POLICY_CHANGED", "The tracking policy changed since the lock was made"))
        for label, getter, model_id, sha in (
                ("detector", self.env.catalog.detector, lock.detector_model_id, lock.detector_sha256),
                ("recognizer", self.env.catalog.recognizer, lock.recognizer_model_id, lock.recognizer_sha256)):
            try:
                record = getter()
            except DomainError:
                reasons.append(_reason("MODEL_UNAVAILABLE", f"The {label} model is not available", model=label))
                continue
            if record["model_id"] != model_id or record["sha256"] != sha:
                reasons.append(_reason("MODEL_CHANGED", f"The {label} model changed", model=label))
        if settings_fingerprint(self.env.settings()) != lock.settings_fingerprint:
            reasons.append(_reason("SETTINGS_CHANGED", "Tracking/identity pipeline settings changed"))
        labels = {track.track_id for track in inspection.tracks} if inspection is not None else None
        for item in lock.assignments:
            if labels is not None and item.track_id not in labels:
                reasons.append(_reason("TRACK_MISSING", "Assigned track is absent from the inspection", track_id=item.track_id))
            try:
                digest = file_sha256(faceset_archive(root, item.faceset_id), fresh=fresh)
            except DomainError:
                reasons.append(_reason("FACESET_MISSING", "The source FaceSet is unavailable",
                                       track_id=item.track_id, faceset_id=item.faceset_id))
                continue
            if digest != item.faceset_sha256:
                reasons.append(_reason("FACESET_CHANGED", "The source FaceSet changed after it was assigned",
                                       track_id=item.track_id, faceset_id=item.faceset_id))
        if canonical_digest(_bound_content(lock)) != lock.digest:
            reasons.append(_reason("LOCK_TAMPERED", "The lock content does not match its digest"))
        return {"valid": not reasons, "reasons": reasons}

    def require_valid(self, lock_id: str, *, fresh: bool = True) -> IdentityLock:
        lock = self.get(lock_id)
        status = self.validate(lock, fresh=fresh)
        if not status["valid"]:
            codes = ", ".join(sorted({item["code"] for item in status["reasons"]}))
            raise DomainError("IDENTITY_LOCK_STALE", f"Identity lock is stale ({codes}); re-inspect and re-confirm the mapping", 409)
        return lock

    # -- manual correction -----------------------------------------------------------------------
    def edit(self, lock_id: str, value: LockEdit) -> IdentityLock:
        relative = self._relative(lock_id)
        with exclusive(self.env.root, relative + ".lock"):
            lock = self.get(lock_id)
            if lock.revision != value.revision:
                raise DomainError("IDENTITY_LOCK_CONFLICT", "Reload the lock before editing", 409)
            status = self.validate(lock, fresh=True)
            # Only a changed/missing FaceSet can be repaired by an explicit re-assignment.
            blocking = [item for item in status["reasons"] if item["code"] not in {"FACESET_CHANGED", "FACESET_MISSING"}]
            if blocking:
                raise DomainError("IDENTITY_LOCK_STALE",
                                  "The target video, inspection or models changed; create a new lock from a fresh inspection", 409)
            inspection, _ = self._load_inspection(lock.inspection_id)
            labels = {track.track_id for track in inspection.tracks}
            if value.track_id not in labels:
                raise DomainError("TRACK_NOT_FOUND", "Track absent from this inspection", 422)
            assignments = {item.track_id: item for item in lock.assignments}
            excluded = set(lock.excluded_tracks)
            if value.action == "assign":
                if value.faceset_id is None:
                    raise DomainError("FACESET_REQUIRED", "Choose the source FaceSet to assign", 422)
                assignments[value.track_id] = LockAssignment(
                    track_id=value.track_id, faceset_id=value.faceset_id,
                    faceset_sha256=self._snapshot_faceset(value.faceset_id))      # re-approves a changed FaceSet
                excluded.discard(value.track_id)
            elif value.action == "unassign":
                if assignments.pop(value.track_id, None) is None:
                    raise DomainError("IDENTITY_TRACK_NOT_ASSIGNED", "Track has no source assignment", 422)
            elif value.action == "swap":
                other = value.other_track_id
                if other is None or other == value.track_id or other not in labels:
                    raise DomainError("TRACK_NOT_FOUND", "Choose a second, different track to swap with", 422)
                if value.track_id not in assignments or other not in assignments:
                    raise DomainError("IDENTITY_TRACK_NOT_ASSIGNED", "Both tracks must have a source assignment to swap", 422)
                first, second = assignments[value.track_id], assignments[other]
                assignments[value.track_id] = first.model_copy(update={"faceset_id": second.faceset_id, "faceset_sha256": second.faceset_sha256})
                assignments[other] = second.model_copy(update={"faceset_id": first.faceset_id, "faceset_sha256": first.faceset_sha256})
            elif value.action == "exclude":
                assignments.pop(value.track_id, None)
                excluded.add(value.track_id)
            else:
                excluded.discard(value.track_id)
            draft = lock.model_dump(mode="json") | dict(
                revision=lock.revision + 1, updated_at=_now(),
                assignments=[item.model_dump(mode="json") for item in sorted(assignments.values(), key=lambda item: item.track_id)],
                excluded_tracks=sorted(excluded))
            draft["digest"] = canonical_digest(_bound_content(draft))
            updated = IdentityLock.model_validate(draft)
            self._write(updated)
            return updated

    # -- job binding ----------------------------------------------------------------------------
    def verify_job(self, request) -> IdentityLock:
        """A lock-bound job must still match the lock exactly, and the lock must still match the disk."""
        lock = self.require_valid(request.identity_lock_id, fresh=True)
        if lock.inspection_id != request.inspection_id or lock.target_path != request.target_path:
            raise DomainError("IDENTITY_LOCK_MISMATCH", "The identity lock belongs to a different target or inspection", 409)
        if {(item.track_id, item.faceset_id) for item in request.face_mappings} !=                 {(item.track_id, item.faceset_id) for item in lock.assignments}:
            raise DomainError("IDENTITY_LOCK_MISMATCH", "Job mappings differ from the locked assignments", 409)
        # Per-interval roles may only NARROW a lock: a segment that swaps must use a locked (track, FaceSet) pair.
        authorised = {(item.track_id, item.faceset_id) for item in lock.assignments}
        for segment in request.face_segments or ():
            if segment.role == "swap" and (segment.track_id, segment.faceset_id) not in authorised:
                raise DomainError("IDENTITY_LOCK_SEGMENT_UNAUTHORIZED",
                                  "A face role swaps a track/FaceSet pair that is not in the identity lock", 409)
        return lock

    # -- API view --------------------------------------------------------------------------------
    def public(self, lock: IdentityLock, *, fresh: bool = False) -> dict:
        return {**lock.model_dump(mode="json"), "status": self.validate(lock, fresh=fresh)}
