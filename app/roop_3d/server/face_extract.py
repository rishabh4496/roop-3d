"""`POST /api/faces/extract_from_frame`: the end-to-end "Use Face From Frame" pipeline.

Takes one face from one frame of a target video or image and turns it into either

* `source_gallery`: a new single-view FaceSet, immediately usable as the active source face;
* `active_faceset`: a new view inserted into the matching (yaw, pitch) pose cell of an existing `.fsz`
  archive, followed by a Stage 0 pose-matrix refresh;
* `target_cluster`: a pinned target-identity cluster centroid whose rank never changes once assigned.

Every path stores a high-resolution aligned crop, the 512-D ArcFace embedding and the `Pose3D`.
`face_index` counts the detected faces of that frame left to right (by box x1), so it is stable.
"""
from __future__ import annotations

import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import uuid
from typing import Any, Literal

import cv2
import numpy as np
from fastapi import APIRouter, Query, Request
from pydantic import Field, model_validator

from roop_3d.audit.isolation import safe_path, sha256_file
from roop_3d.core.errors import DomainError
from roop_3d.core.pose import Pose3D, estimate_pose_5pt, pose_cell
from roop_3d.core.settings import Contract, runtime_path
from roop_3d.core.storage import atomic_json, owned
from roop_3d.engine.volumetric.preprocess import aligned_crop, transform_points
from roop_3d.faceset import pose_matrix as stage0
from roop_3d.faceset.faceset_reader import FaceSetReader, export_v2
from roop_3d.faceset.pose_selector import classify_pose_bin
from roop_3d.faceset.recognizer import Recognizer
from roop_3d.faceset.schemas import FaceSetDocument, FaceView, QualityMetrics

router = APIRouter(prefix="/api/v1")

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CROP_SIZE = 512
CLUSTER_MATCH_SIMILARITY = 0.60  # a new pin this close (cosine) to an existing centroid refines it instead of adding a rank
MAX_CLUSTERS = 64


class ExtractRequest(Contract):
    target_media_path: str = Field(min_length=1, max_length=1024)
    frame_index: int = Field(default=0, ge=0, le=10_000_000)
    face_index: int = Field(default=0, ge=0, le=63)
    destination: Literal["source_gallery", "active_faceset", "target_cluster"]
    faceset_name: str | None = Field(default=None, min_length=1, max_length=128,
                                     description="source_gallery: display name. active_faceset: the id or name of the FaceSet to extend.")

    @model_validator(mode="after")
    def named(self) -> "ExtractRequest":
        if self.destination == "active_faceset" and not self.faceset_name:
            raise ValueError("active_faceset needs faceset_name (the FaceSet id or name)")
        return self


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, np.float64)
    norm = np.linalg.norm(value)
    if not np.isfinite(norm) or norm < 1e-9:
        raise DomainError("IDENTITY_EMBEDDING_INVALID", "Identity embedding must have nonzero norm", 422)
    return value / norm


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")[:40]


class FaceExtractService:
    def __init__(self, state):
        self.state = state
        self.detector = None      # overridable (tests); defaults to the media service's SCRFD detector
        self.recognizer = None    # overridable; defaults to a lazy ArcFace Recognizer
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        self._digests: dict[tuple, str] = {}

    # -- plumbing ---------------------------------------------------------------------
    def _lock(self, key: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(key, threading.Lock())

    def _detector(self):
        return self.detector or self.state.media.detector

    def _recognizer(self):
        if self.recognizer is None:
            self.recognizer = Recognizer(self.state.models)
        return self.recognizer

    def _media_file(self, value: str) -> tuple[Path, bool]:
        root = self.state.root
        suffix = Path(value).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            return self.state.media.path(value), False
        try:
            path = runtime_path(value, root)
        except ValueError:
            raise DomainError("MEDIA_PATH_DENIED", "Media must be a local file inside the owned runtime", 422) from None
        permitted = (root / "roop_3d/.runtime/media", runtime_path(self.state.settings.output_root, root))
        if not any(path.is_relative_to(parent) for parent in permitted):
            raise DomainError("MEDIA_PATH_DENIED", "Only media in runtime/media or runtime/outputs is accepted", 422)
        if not path.is_file():
            raise DomainError("MEDIA_NOT_FOUND", "Local image file does not exist", 404)
        return path, True

    def _frame(self, request: ExtractRequest) -> tuple[np.ndarray, Path]:
        path, is_image = self._media_file(request.target_media_path)
        if is_image:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise DomainError("MEDIA_UNREADABLE", "Image could not be decoded", 422)
            return image, path
        return self.state.media.frame(request.target_media_path, None, request.frame_index), path

    def _target_digest(self, path: Path) -> str:
        status = path.stat()
        key = (str(path), status.st_size, status.st_mtime_ns)
        if key not in self._digests:
            self._digests = {key: sha256_file(path)[1]}
        return self._digests[key]

    # -- extraction -------------------------------------------------------------------
    def extract(self, request: ExtractRequest, *, decoded: tuple[np.ndarray, Path] | None = None) -> dict[str, Any]:
        image, path = decoded if decoded is not None else self._frame(request)
        detections, _ = self._detector().detect(image, 0.5)
        faces = sorted(detections, key=lambda item: float(np.asarray(item[0])[0]))
        if request.face_index >= len(faces):
            raise DomainError("FACE_NOT_FOUND", f"Frame has {len(faces)} detected face(s); face_index {request.face_index} is out of range", 422)
        box, points = faces[request.face_index]
        points = np.asarray(points, np.float64)
        pose = estimate_pose_5pt(points)
        crop = aligned_crop(image, points, CROP_SIZE)
        embedding = _unit(self._recognizer().embed(image, points.astype(np.float32)))
        ok_png, png = cv2.imencode(".png", crop.image)
        ok_jpg, jpg = cv2.imencode(".jpg", cv2.resize(crop.image, (128, 128), interpolation=cv2.INTER_AREA))
        if not (ok_png and ok_jpg):
            raise DomainError("FACE_ENCODE_FAILED", "Face crop could not be encoded", 500)
        gray = cv2.cvtColor(crop.image, cv2.COLOR_BGR2GRAY)
        sharpness = float(min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0))
        box = np.asarray(box, np.float64)
        face = {
            "face_id": "face_" + uuid.uuid4().hex[:12], "frame_index": request.frame_index, "face_index": request.face_index,
            "bbox": [float(v) for v in box], "landmarks5": points.tolist(), "pose": pose.model_dump(),
            "pose_cell": pose_cell(pose).name, "embedding_dim": int(embedding.shape[0]), "sharpness": sharpness,
            "embedding": [float(x) for x in embedding], "thumbnail_base64": base64.b64encode(jpg.tobytes()).decode("ascii"),
        }
        legacy_yaw, legacy_pitch, legacy_roll = pose.legacy_tuple()
        yaw, pitch, roll = max(-90.0, min(90.0, legacy_yaw)), max(-60.0, min(60.0, legacy_pitch)), legacy_roll
        score = round(0.7 * sharpness + 0.3 * 0.95, 4)
        view = FaceView(
            index=0, image_bytes=png.tobytes(), embedding=tuple(float(x) for x in embedding),
            landmarks=tuple((float(x), float(y)) for x, y in transform_points(points, crop.matrix)),
            bbox=(0.0, 0.0, float(CROP_SIZE), float(CROP_SIZE)), yaw=round(yaw, 2), pitch=round(pitch, 2), roll=round(roll, 2),
            expression={"openness": 0.0},
            quality=QualityMetrics(sharpness=round(sharpness, 4), detector_confidence=0.95, score=score,
                                   laplacian_variance=round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
                                   blur=round(1.0 - sharpness, 4), face_pixels=float(max(1.0, (box[2] - box[0]) * (box[3] - box[1])))),
            pose_bin=classify_pose_bin(yaw, pitch)[0], is_synthetic=False)
        self._store_metadata(face, request, jpg.tobytes())
        handler = {"source_gallery": self._to_gallery, "active_faceset": self._to_faceset, "target_cluster": self._to_cluster}[request.destination]
        result = handler(request, face, view, path)
        return {"destination": request.destination, "face": face, **result}

    def _store_metadata(self, face: dict[str, Any], request: ExtractRequest, thumbnail: bytes) -> None:
        root = self.state.root
        base = f"roop_3d/.runtime/extracted_faces/{face['face_id']}"
        owned(root, base + ".jpg").parent.mkdir(parents=True, exist_ok=True)
        owned(root, base + ".jpg").write_bytes(thumbnail)
        face["thumbnail_path"], face["metadata_path"] = base + ".jpg", base + ".json"
        atomic_json(root, base + ".json", {**{k: v for k, v in face.items() if k != "thumbnail_base64"},
                                           "target_media_path": request.target_media_path, "destination": request.destination,
                                           "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()})

    # -- destination: new FaceSet / active source ----------------------------------------
    def _write_faceset(self, document: FaceSetDocument, *, replace: bool) -> None:
        root = self.state.root
        final = owned(root, f"roop_3d/outputs/facesets/{document.faceset_id}.fsz")
        final.parent.mkdir(parents=True, exist_ok=True)
        temporary = final.with_name(f".{document.faceset_id}.{uuid.uuid4().hex}.new.fsz")
        try:
            export_v2(document, temporary, app_root=root)
            if not replace and final.exists():
                raise DomainError("FACESET_EXISTS", "A FaceSet with this id already exists", 409)
            try:
                os.replace(temporary, final)
            except PermissionError:
                raise DomainError("FACESET_BUSY", "The FaceSet archive is in use; try again in a moment", 409) from None
        finally:
            temporary.unlink(missing_ok=True)
        from roop_3d.server.workstation import coverage
        atomic_json(root, f"roop_3d/.runtime/facesets/{document.faceset_id}.json",
                    {"faceset_id": document.faceset_id, "name": document.name[:128], "item_count": len(document.views), "schema_version": 1})
        atomic_json(root, f"roop_3d/.runtime/coverage/{document.faceset_id}.json", coverage(document))

    def _document(self, faceset_id: str, name: str, views: tuple[FaceView, ...], metadata: dict[str, Any],
                  reader: FaceSetReader | None = None) -> FaceSetDocument:
        embeddings = [np.asarray(v.embedding, np.float32) for v in views]
        confident = [i for i, v in enumerate(views) if v.quality.score >= 0.5 and v.quality.detector_confidence >= 0.5]
        helper = reader or FaceSetReader(self.state.models, cache_root=self.state.root / "roop_3d/.runtime/cache")
        canonical, medoid = helper._aggregate_embeddings(embeddings, confident)
        return FaceSetDocument(
            faceset_id=faceset_id, name=name, source_sha256=hashlib.sha256(b"".join(v.image_bytes for v in views)).hexdigest(),
            archive_format="v2_standard", canonical_embedding=tuple(float(x) for x in canonical),
            medoid_embedding=tuple(float(x) for x in medoid), views=views, pose_bins=helper._build_pose_bins(list(views)),
            metadata=metadata)

    def _to_gallery(self, request: ExtractRequest, face: dict[str, Any], view: FaceView, path: Path) -> dict[str, Any]:
        name = request.faceset_name or f"Frame {request.frame_index} · face {request.face_index}"
        faceset_id = (_slug(request.faceset_name) + "_" if request.faceset_name and _slug(request.faceset_name) else "frame_") + uuid.uuid4().hex[:8]
        document = self._document(faceset_id, name, (view,), {"created_via": "extract_from_frame", "face_id": face["face_id"]})
        self._write_faceset(document, replace=False)
        matrix = stage0.preflight(document) if stage0.stage0_enabled() else None
        return {"faceset": {"faceset_id": faceset_id, "name": name, "item_count": 1, "schema_version": 1},
                "active_source": {"faceset_id": faceset_id, "name": name},
                "pose_matrix": matrix.summary() if matrix else None}

    # -- destination: extend an existing FaceSet ------------------------------------------
    def _resolve_faceset(self, key: str) -> tuple[str, Path]:
        root = self.state.root
        catalog = safe_path(root, "roop_3d/.runtime/facesets")
        matches: list[str] = []
        if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", key) and (catalog / f"{key}.json").is_file():
            matches = [key]
        elif catalog.is_dir():
            for manifest in sorted(catalog.glob("*.json")):
                try:
                    if json.loads(manifest.read_text("utf-8")).get("name", "").casefold() == key.casefold():
                        matches.append(manifest.stem)
                except (OSError, ValueError):
                    continue
        if not matches:
            raise DomainError("FACESET_NOT_FOUND", "No FaceSet has that id or name", 404)
        if len(matches) > 1:
            raise DomainError("FACESET_AMBIGUOUS", "Several FaceSets share that name; use the FaceSet id", 409)
        for relative in (f"roop_3d/outputs/facesets/{matches[0]}.fsz", f"roop_3d/.runtime/facesets/{matches[0]}.fsz"):
            archive = safe_path(root, relative)
            if archive.is_file():
                return matches[0], archive
        raise DomainError("FACESET_NOT_FOUND", "FaceSet archive is missing", 404)

    def _to_faceset(self, request: ExtractRequest, face: dict[str, Any], view: FaceView, path: Path) -> dict[str, Any]:
        faceset_id, archive = self._resolve_faceset(request.faceset_name or "")
        with self._lock(faceset_id):
            reader = FaceSetReader(self.state.models, cache_root=self.state.root / "roop_3d/.runtime/cache")
            current = reader.read(archive, faceset_id=faceset_id)
            distance = stage0.cosine_distance(np.asarray(view.embedding), np.asarray(current.canonical_embedding))
            if distance > stage0.GUARD_MAX_COSINE_DISTANCE:
                raise DomainError("IDENTITY_GUARD_VIOLATION",
                                  f"This face is {distance:.2f} cosine distance from the FaceSet identity (limit "
                                  f"{stage0.GUARD_MAX_COSINE_DISTANCE:.2f}); it was not added", 422)
            added = view.model_copy(update={"index": max((v.index for v in current.views), default=-1) + 1})
            document = self._document(faceset_id, current.name, (*current.views, added), {**current.metadata, "extended_via": "extract_from_frame"}, reader)
            self._write_faceset(document, replace=True)
            stage0.invalidate(faceset_id)
        cell = pose_cell(Pose3D.model_validate(face["pose"]))  # already standard frame
        matrix = stage0.preflight(document, label=faceset_id) if stage0.stage0_enabled() else None
        return {"faceset": {"faceset_id": faceset_id, "name": document.name, "item_count": len(document.views), "schema_version": 1},
                "inserted": {"view_index": added.index, "pose_cell": cell.name, "cell_index": cell.index},
                "pose_matrix": matrix.summary() if matrix else None}

    # -- destination: pinned target identity cluster -------------------------------------
    def _to_cluster(self, request: ExtractRequest, face: dict[str, Any], view: FaceView, path: Path) -> dict[str, Any]:
        digest = self._target_digest(path)
        relative = f"roop_3d/.runtime/clusters/{digest[:32]}.json"
        with self._lock("clusters:" + digest):
            store = self._read_clusters(relative, digest, request.target_media_path)
            embedding = np.asarray(face["embedding"], np.float64)
            best, similarity = None, -1.0
            for cluster in store["clusters"]:
                score = float(np.dot(_unit(np.asarray(cluster["centroid"])), embedding))
                if score > similarity:
                    best, similarity = cluster, score
            if best is not None and similarity >= CLUSTER_MATCH_SIMILARITY:
                count = best["count"]
                merged = _unit(np.asarray(best["centroid"]) * count + embedding)  # refine the centroid; the rank stays put
                best.update(centroid=[float(x) for x in merged], count=count + 1, last_face_id=face["face_id"])
                cluster, created = best, False
            else:
                if len(store["clusters"]) >= MAX_CLUSTERS:
                    raise DomainError("CLUSTER_LIMIT", f"At most {MAX_CLUSTERS} pinned clusters per target", 409)
                rank = max((c["rank"] for c in store["clusters"]), default=-1) + 1  # ranks are never reused or renumbered
                cluster = {"rank": rank, "centroid": face["embedding"], "count": 1, "pose": face["pose"], "frame_index": request.frame_index,
                           "face_index": request.face_index, "face_id": face["face_id"], "last_face_id": face["face_id"],
                           "thumbnail_base64": face["thumbnail_base64"],
                           "pinned_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
                store["clusters"].append(cluster)
                created = True
            atomic_json(self.state.root, relative, store)
        return {"cluster": {**cluster, "created": created, "similarity": round(similarity, 4) if similarity >= 0 else None}}

    def _read_clusters(self, relative: str, digest: str, media_path: str) -> dict[str, Any]:
        path = owned(self.state.root, relative)
        if path.is_file():
            try:
                store = json.loads(path.read_text("utf-8"))
                if store.get("target_sha256") == digest and isinstance(store.get("clusters"), list):
                    return store
            except (OSError, ValueError):
                pass
        return {"schema_version": 1, "target_sha256": digest, "target_media_path": media_path, "clusters": []}

    def clusters(self, target_media_path: str) -> dict[str, Any]:
        path, _ = self._media_file(target_media_path)
        digest = self._target_digest(path)
        store = self._read_clusters(f"roop_3d/.runtime/clusters/{digest[:32]}.json", digest, target_media_path)
        return {"target_media_path": target_media_path, "clusters": sorted(store["clusters"], key=lambda c: c["rank"])}


@router.post("/faces/extract_from_frame", status_code=201)
def extract_from_frame(value: ExtractRequest, request: Request) -> dict[str, Any]:
    return request.app.state.face_extract.extract(value)


@router.get("/faces/target_clusters")
def target_clusters(request: Request, target_media_path: str = Query(min_length=1, max_length=1024)) -> dict[str, Any]:
    return request.app.state.face_extract.clusters(target_media_path)


class FillAnglesRequest(Contract):
    target_media_path: str = Field(min_length=1, max_length=1024)
    faceset_id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,64}$')
    frames: tuple[int, ...] | None = Field(default=None, min_length=1, max_length=48)
    max_frames: int = Field(default=100_000,ge=1,le=10_000_000,strict=True)
    time_budget_seconds: float = Field(default=120,ge=1,le=600)

    @model_validator(mode='after')
    def frame_bounds(self):
        if self.frames and any(type(v) is not int or not 0 <= v <= 9999 for v in self.frames):
            raise ValueError('Explicit frame samples must be integers within 0..9999')
        return self


def scan_missing_angles(service, value):
    from contextlib import closing
    import time
    from roop_3d.media.video_reader import VideoReader
    _, archive = service._resolve_faceset(value.faceset_id)
    reader = FaceSetReader(service.state.models, cache_root=service.state.root / 'roop_3d/.runtime/cache')
    document = reader.read(archive, faceset_id=value.faceset_id)
    occupied = {pose_cell((v.yaw,v.pitch,v.roll)).name for v in document.views}
    added, sampled, complete, reason = [], 0, False, 'end_of_video'
    started=time.monotonic()
    path,is_image=service._media_file(value.target_media_path)
    def samples():
        if value.frames is not None or is_image:
            for frame in sorted(set(value.frames or (0,))):
                sample=ExtractRequest(target_media_path=value.target_media_path,frame_index=frame,destination='active_faceset',faceset_name=value.faceset_id)
                yield frame,service._frame(sample)[0]
        else:
            with closing(VideoReader(path,hardware='software').frames()) as frames:
                for decoded in frames:
                    yield decoded.index,decoded.image
    with closing(samples()) as frames:
        for frame,image in frames:
            if len(occupied)==9:
                complete=True;reason='all_cells_populated';break
            if sampled >= value.max_frames or time.monotonic()-started >= value.time_budget_seconds:
                reason='scan_budget';break
            sampled+=1
            detections, _ = service._detector().detect(image, .5)
            for index, (_, points) in enumerate(sorted(detections, key=lambda item: float(item[0][0]))):
                cell = pose_cell(estimate_pose_5pt(points)).name
                if cell in occupied: continue
                embedding = service._recognizer().embed(image, np.asarray(points,np.float32))
                similarity = float(_unit(embedding) @ _unit(np.asarray(document.canonical_embedding)))
                if similarity < CLUSTER_MATCH_SIMILARITY: continue
                sample=ExtractRequest(target_media_path=value.target_media_path,frame_index=frame,face_index=index,
                                      destination='active_faceset',faceset_name=value.faceset_id)
                result=service.extract(sample,decoded=(image,path))
                occupied.add(cell)
                added.append(result)
        else:
            complete=len(occupied)==9 or value.frames is None or is_image
            reason='all_cells_populated' if len(occupied)==9 else 'end_of_video' if complete else 'explicit_samples'
    return {'added':added,'populated_cells':sorted(occupied),'sampled_frames':sampled,
            'scan_complete':complete,'stop_reason':reason}


@router.post('/faces/fill_missing_angles')
def fill_missing_angles(value: FillAnglesRequest, request: Request):
    with request.app.state.engine_gate.preview_section():
        return scan_missing_angles(request.app.state.face_extract,value)


class AnalyzePhotoRequest(Contract):
    image_base64: str = Field(min_length=1, max_length=16_000_000)


def _analyze_photo(value: AnalyzePhotoRequest, request: Request):
    service = request.app.state.face_extract
    try:
        data = base64.b64decode(value.image_base64.split(',')[-1], validate=True)
    except ValueError:
        raise DomainError('IMAGE_DECODE_FAILED', 'Invalid image base64', 422)
    from roop_3d.core.image_bytes import decode_face_image
    image = decode_face_image(data)
    detections,_ = service._detector().detect(image,.5)
    faces=[]
    for box,points in detections[:32]:
        crop=aligned_crop(image,points,CROP_SIZE)
        ok,png=cv2.imencode('.png',crop.image)
        if not ok: continue
        yaw,pitch,roll=estimate_pose_5pt(points).legacy_tuple()
        sharpness=float(min(1,cv2.Laplacian(cv2.cvtColor(crop.image,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()/500))
        faces.append({'id':'photo_'+uuid.uuid4().hex[:12], 'thumbnailUrl':'data:image/png;base64,'+base64.b64encode(png).decode(),
                      'embedding':_unit(service._recognizer().embed(image,np.asarray(points,np.float32))).tolist(),
                      'yaw':yaw,'pitch':pitch,'roll':roll,'sharpness':sharpness,'quality':sharpness,
                      'landmarks':transform_points(np.asarray(points,np.float64),crop.matrix).tolist(), 'bbox':[0,0,CROP_SIZE,CROP_SIZE]})
    return {'faces':faces}


@router.post('/faces/analyze_photo')
def analyze_photo(value: AnalyzePhotoRequest, request: Request):
    with request.app.state.engine_gate.preview_section():
        return _analyze_photo(value,request)


@router.get('/facesets/{faceset_id}/pose_matrix')
def archive_pose_matrix(faceset_id: str, request: Request):
    service=request.app.state.face_extract
    faceset_id,archive=service._resolve_faceset(faceset_id)
    reader=FaceSetReader(service.state.models,cache_root=service.state.root/'roop_3d/.runtime/cache')
    document=reader.read(archive,faceset_id=faceset_id)
    cells={}
    from roop_3d.core.pose import ALL_CELLS
    for cell in ALL_CELLS: cells[cell.name]={'cell':cell.name,'support':0,'views':[]}
    for view in document.views:
        cell=pose_cell((view.yaw,view.pitch,view.roll)).name
        cells[cell]['support']+=1
        cells[cell]['views'].append(view)
    for cell in cells.values():
        views=sorted(cell['views'],key=lambda view:(-view.quality.score,-view.quality.sharpness,view.index))
        best=views[0] if views else None
        thumbnail=None
        if best:
            image=cv2.imdecode(np.frombuffer(best.image_bytes,np.uint8),cv2.IMREAD_COLOR)
            if image is not None:
                ok,data=cv2.imencode('.jpg',cv2.resize(image,(128,128)))
                if ok: thumbnail='data:image/jpeg;base64,'+base64.b64encode(data).decode()
        cell['views']=[{'index':view.index,'yaw':view.yaw,'pitch':view.pitch,'roll':view.roll,
                        'sharpness':view.quality.sharpness,'quality':view.quality.score} for view in views]
        cell['best']=cell['views'][0] if best else None
        cell['thumbnailUrl']=thumbnail
    return {'faceset_id':faceset_id,'item_count':len(document.views),'cells':list(cells.values())}
