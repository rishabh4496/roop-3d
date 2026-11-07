"""Target identity locking from any frame (Stage 12).

    POST /api/v1/target/lock-from-frame   click a face on any frame -> TargetIdentityLock + multi-angle reference bank
    POST /api/v1/target/add-angle-sample  add another angle of the same person to an existing lock ("Add Angle to Target #1")
    GET  /api/v1/target/{lock_id}         the lock, its bank and the keyframe-scan state
    GET  /api/v1/target/{lock_id}/match   which face of a frame is the locked person (scores, not just a boolean)
    GET  /api/v1/target                   locks of a video
    GET  /api/v1/media/faces              full per-face analysis of one frame (embedding, thumbnail, dense landmarks, cluster ids)

The locks are in-memory for the life of the server process; this is separate from the hash-bound, FaceSet-assigning
``/identity/locks`` of the inspection pipeline (see ``identity.target_lock`` for the association rules).
"""
from __future__ import annotations

import logging
import threading
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, Query, Request
from pydantic import Field, model_validator

from roop_3d.core.errors import DomainError
from roop_3d.core.pose import as_pose3d
from roop_3d.core.settings import Contract
from roop_3d.identity.target_lock import BankEntry, FaceRef, TargetIdentityLock, TargetLockStore

log = logging.getLogger("roop3d.target_lock")
router = APIRouter(prefix="/api/v1")
MAX_SCAN_SAMPLES = 48
MIN_ANGLE_SIMILARITY = 0.30      # measured: different people average 0.13 (p99 0.37); same person, any pose, >= 0.49


def store(request: Request) -> TargetLockStore:
    state = request.app.state
    if getattr(state, "target_locks", None) is None:
        state.target_locks = TargetLockStore()
    return state.target_locks


def ref_of(face) -> FaceRef:
    pose = as_pose3d(face.pose_legacy)
    return FaceRef(face.embedding, float(pose.yaw), float(pose.pitch), float(pose.roll), [float(v) for v in face.box])


def annotate(request: Request, path: str, frame_index: int, analyzed: list, faces: list[dict]) -> None:
    """Fill ``cluster_id`` / ``track_id`` / lock flags of the face dicts from the video's person clusters."""
    if not analyzed or any(f.embedding is None for f in analyzed):
        return
    locks = store(request)
    registry = locks.registry(path)
    associations = registry.associate([ref_of(f) for f in analyzed], frame_index, learn=True)
    by_cluster = {l.cluster_id: l for l in locks.locks_for(path)}
    for face, assoc in zip(faces, associations):
        face["cluster_id"] = assoc.cluster_id
        face["cluster_match"] = {"score": round(assoc.score, 4), "margin": round(assoc.margin, 4), "reason": assoc.reason}
        # A single-frame read has no tracker: the track id is the person's persistent cluster id (appearance-linked).
        face["track_id"] = assoc.cluster_id
        face["track_source"] = "appearance_cluster"
        lock = by_cluster.get(assoc.cluster_id)
        face["is_target"] = lock is not None
        face["target_lock_id"] = lock.lock_id if lock else None
        face["target_number"] = lock.target_number if lock else None


def _pick(faces: list[dict], face_index: int | None, point: list[float] | None) -> int:
    if face_index is not None:
        if not 0 <= face_index < len(faces):
            raise DomainError("FACE_NOT_FOUND", f"Frame has {len(faces)} face(s); face_index {face_index} does not exist", 404)
        return face_index
    assert point is not None
    px, py = point
    inside = [i for i, f in enumerate(faces) if f["bbox"][0] <= px <= f["bbox"][2] and f["bbox"][1] <= py <= f["bbox"][3]]
    if inside:
        return min(inside, key=lambda i: (faces[i]["bbox"][2] - faces[i]["bbox"][0]) * (faces[i]["bbox"][3] - faces[i]["bbox"][1]))
    raise DomainError("FACE_NOT_AT_POINT", "No detected face contains that point", 404)


class FaceSelector(Contract):
    path: str = Field(min_length=1, max_length=1024)
    frame_index: int = Field(ge=0, le=100_000_000)
    face_index: int | None = Field(default=None, ge=0, le=31)
    point: list[float] | None = Field(default=None, min_length=2, max_length=2)       # native-frame pixels
    threshold: float = Field(default=0.5, ge=0.1, le=0.99)

    @model_validator(mode="after")
    def one_selector(self):
        if (self.face_index is None) == (self.point is None):
            raise ValueError("Provide exactly one of face_index or point")
        return self


class LockFromFrame(FaceSelector):
    sample_count: int = Field(default=16, ge=0, le=MAX_SCAN_SAMPLES)
    wait: bool = True


class AddAngle(Contract):
    lock_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    frame_index: int = Field(ge=0, le=100_000_000)
    face_index: int | None = Field(default=None, ge=0, le=31)
    point: list[float] | None = Field(default=None, min_length=2, max_length=2)
    threshold: float = Field(default=0.5, ge=0.1, le=0.99)
    force: bool = False

    @model_validator(mode="after")
    def one_selector(self):
        if (self.face_index is None) == (self.point is None):
            raise ValueError("Provide exactly one of face_index or point")
        return self


def _analyzed(request: Request, path: str, frame_index: int, threshold: float):
    frames = request.app.state.media.frames
    result = frames.read(path, frame_index=frame_index)
    analyzed = frames.analyzed_faces(path, result, threshold)
    if analyzed is None:
        raise DomainError("ANALYZER_UNAVAILABLE", "Face analysis is not configured", 503)
    return result, analyzed


def scan_keyframes(request: Request, lock: TargetIdentityLock, sample_count: int, threshold: float) -> None:
    """Expand the reference bank: visit keyframes outward from the anchor and link the person by chained appearance."""
    locks = store(request)
    registry = locks.registry(lock.media_path)
    frames = request.app.state.media.frames
    total = frames.info(lock.media_path).total_frames
    lock.scan = {"state": "running", "requested": sample_count, "sampled": 0, "matched": 0}
    try:
        slots = sorted({min(total - 1, int((k + 0.5) / max(sample_count, 1) * total)) for k in range(sample_count)},
                       key=lambda i: abs(i - lock.anchor_frame_index))
        matched = 0
        for done, index in enumerate(slots, start=1):
            try:
                result = frames.read(lock.media_path, frame_index=index)
                analyzed = frames.analyzed_faces(lock.media_path, result, threshold) or []
                if analyzed and all(f.embedding is not None for f in analyzed):
                    associations = registry.associate([ref_of(f) for f in analyzed], result.frame_index, learn=True)
                    matched += sum(a.cluster_id == lock.cluster_id for a in associations)
            except DomainError as error:
                log.info("keyframe %s skipped: %s", index, error.code)
            lock.scan.update(sampled=done, matched=matched)
        lock.scan.update(state="done", bank_size=len(registry.clusters[lock.cluster_id].bank))
    except Exception as error:                        # a scan failure leaves the anchor-only lock usable
        log.exception("target scan failed")
        lock.scan.update(state="error", message=str(error)[:200])


@router.post("/target/lock-from-frame", status_code=201)
def lock_from_frame(value: LockFromFrame, request: Request) -> dict:
    result, analyzed = _analyzed(request, value.path, value.frame_index, value.threshold)
    frames = request.app.state.media.frames
    h, w = result.image.shape[:2]
    from roop_3d.media.face_analysis import face_to_dict
    dicts = [face_to_dict(f, i, result.frame_index, w, h, detail="basic") for i, f in enumerate(analyzed)]
    if not dicts:
        raise DomainError("NO_FACE_IN_FRAME", "No face was detected in that frame", 404)
    chosen = _pick(dicts, value.face_index, value.point)
    face = analyzed[chosen]
    if face.embedding is None:
        raise DomainError("EMBEDDING_UNAVAILABLE", "The selected face could not be embedded (recognizer unavailable)", 503)
    locks = store(request)
    try:
        lock = locks.create(value.path, result.frame_index, ref_of(face))
    except ValueError as error:
        raise DomainError("EMBEDDING_UNAVAILABLE", str(error), 503) from error
    # Faces of the anchor frame join their clusters too, so the other people on screen are competitors from the start.
    annotate(request, value.path, result.frame_index, analyzed, dicts)
    if value.sample_count > 0:
        if value.wait:
            scan_keyframes(request, lock, value.sample_count, value.threshold)
        else:
            lock.scan = {"state": "running", "requested": value.sample_count, "sampled": 0, "matched": 0}
            threading.Thread(target=scan_keyframes, args=(request, lock, value.sample_count, value.threshold),
                             name=f"target-scan-{lock.lock_id[:6]}", daemon=True).start()
    else:
        lock.scan = {"state": "skipped"}
    return {**lock.public(locks.registry(value.path)), "selected_face_index": chosen, "frame_faces": dicts}


@router.post("/target/add-angle-sample")
def add_angle_sample(value: AddAngle, request: Request) -> dict:
    locks = store(request)
    lock = locks.get(value.lock_id)
    if lock is None:
        raise DomainError("TARGET_LOCK_NOT_FOUND", "Unknown target lock", 404)
    result, analyzed = _analyzed(request, lock.media_path, value.frame_index, value.threshold)
    h, w = result.image.shape[:2]
    from roop_3d.media.face_analysis import face_to_dict
    dicts = [face_to_dict(f, i, result.frame_index, w, h, detail="basic") for i, f in enumerate(analyzed)]
    chosen = _pick(dicts, value.face_index, value.point)
    face = analyzed[chosen]
    if face.embedding is None:
        raise DomainError("EMBEDDING_UNAVAILABLE", "The selected face could not be embedded", 503)
    registry = locks.registry(lock.media_path)
    cluster = registry.clusters[lock.cluster_id]
    similarity, nearest = cluster.score(face.embedding)
    if similarity < MIN_ANGLE_SIMILARITY and not value.force:
        raise DomainError("ANGLE_SAMPLE_DISSIMILAR",
                          f"That face looks nothing like Target #{lock.target_number} (cosine {similarity:.2f} to its closest "
                          "reference). Send force=true if you are sure it is the same person.", 409)
    ref = ref_of(face)
    with registry.lock:
        added = cluster.consider(BankEntry(face.embedding, ref.yaw, ref.pitch, ref.roll, result.frame_index, ref.bbox, "user", similarity),
                                 force=True)
        cluster.seen[result.frame_index] = ref.bbox
    return {**lock.public(registry), "added": added, "similarity_to_bank": round(similarity, 4),
            "warning": "low similarity: accepted because force=true" if similarity < MIN_ANGLE_SIMILARITY else None, "selected_face_index": chosen}


@router.get("/target/{lock_id}")
def read_target(lock_id: str, request: Request) -> dict:
    locks = store(request)
    lock = locks.get(lock_id)
    if lock is None:
        raise DomainError("TARGET_LOCK_NOT_FOUND", "Unknown target lock", 404)
    return lock.public(locks.registry(lock.media_path))


@router.get("/target")
def list_targets(request: Request, path: Annotated[str, Query(min_length=1, max_length=1024)]) -> list[dict]:
    locks = store(request)
    registry = locks.registry(path)
    return [l.public(registry) for l in locks.locks_for(path)]


@router.get("/target/{lock_id}/match")
def match_frame(lock_id: str, request: Request, frame_index: Annotated[int, Query(ge=0, le=100_000_000)],
                threshold: Annotated[float, Query(ge=0.1, le=0.99)] = 0.5) -> dict:
    locks = store(request)
    lock = locks.get(lock_id)
    if lock is None:
        raise DomainError("TARGET_LOCK_NOT_FOUND", "Unknown target lock", 404)
    result, analyzed = _analyzed(request, lock.media_path, frame_index, threshold)
    h, w = result.image.shape[:2]
    from roop_3d.media.face_analysis import face_to_dict
    dicts = [face_to_dict(f, i, result.frame_index, w, h, detail="basic") for i, f in enumerate(analyzed)]
    annotate(request, lock.media_path, result.frame_index, analyzed, dicts)
    target = next((f["face_index"] for f in dicts if f["cluster_id"] == lock.cluster_id), None)
    return {"lock_id": lock_id, "frame_index": result.frame_index, "target_face_index": target, "faces": dicts}


@router.get("/media/faces")
def media_faces(request: Request, path: Annotated[str, Query(min_length=1, max_length=1024)],
                frame_index: Annotated[int | None, Query(ge=0, le=100_000_000)] = None,
                threshold: Annotated[float, Query(ge=0.1, le=0.99)] = 0.5,
                detail: Annotated[str, Query(pattern="^(basic|full)$")] = "full") -> dict:
    frames = request.app.state.media.frames
    result = frames.read(path, frame_index=frame_index)
    analyzed = frames.analyzed_faces(path, result, threshold)
    faces = frames.faces(path, result, threshold, detail=detail)
    faces = [dict(f) for f in faces]                  # the detection cache is shared: annotate copies
    if analyzed:
        annotate(request, path, result.frame_index, analyzed, faces)
    h, w = result.image.shape[:2]
    return {"frame_index": result.frame_index, "width": int(w), "height": int(h), "timestamp_sec": round(result.timestamp_sec, 4),
            "faces": faces}
