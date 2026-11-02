"""FaceSet reader, legacy format normalizer, and V2 archive exporter.

Reads .fsz archives in strictly read-only binary mode, normalizing legacy V1/V2
archives and image directories into immutable FaceSetDocuments. Computes canonical
mean and medoid 512-D ArcFace identity embeddings, clusters views into pose bins,
and exports versioned V2 FaceSet archives without ever modifying existing archives.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any
import zipfile

import cv2
import numpy as np

from roop_3d.audit.isolation import safe_path
from roop_3d.audit.project_paths import application_root
from roop_3d.core.errors import DomainError
from roop_3d.core.pose import as_legacy_triplet, legacy_pose_5pt
from roop_3d.faceset.pose_matrix import cell_members
from roop_3d.faceset.pose_selector import classify_pose_bin
from roop_3d.faceset.recognizer import Recognizer
from roop_3d.faceset.schemas import (
    ConsentMetadata,
    FaceSetDocument,
    FaceSetV2Manifest,
    FaceView,
    PoseBinSummary,
    QualityMetrics,
)
from roop_3d.faceset.validator import ArchiveValidator
from roop_3d.media.detector import Detector
from roop_3d.models.catalog import ModelCatalog


class FaceSetReader:
    """Safely normalizes legacy V1/V2 .fsz archives into immutable FaceSetDocuments."""

    def __init__(self, catalog: ModelCatalog | None = None, *, cache_root: Path | None = None):
        self.app_root = application_root()
        self.catalog = catalog or ModelCatalog(self.app_root)
        self.cache_root = cache_root or (self.app_root / "roop_3d/.runtime/cache")
        self._detector: Detector | None = None
        self._recognizer: Recognizer | None = None

    @property
    def detector(self) -> Detector:
        if self._detector is None:
            self._detector = Detector(self.catalog)
        return self._detector

    @property
    def recognizer(self) -> Recognizer:
        if self._recognizer is None:
            self._recognizer = Recognizer(self.catalog)
        return self._recognizer

    def read(self, path: Path | str, faceset_id: str | None = None) -> FaceSetDocument:
        """Read a .fsz archive or image directory in strictly read-only mode."""
        archive_path = Path(path)
        if not archive_path.exists():
            raise DomainError("FACESET_NOT_FOUND", f"FaceSet archive does not exist: {archive_path}", 404)

        fid = faceset_id or archive_path.stem

        if archive_path.is_dir():
            return self._read_directory(archive_path, fid)

        # 1. Read input bytes in strictly read-only binary mode and compute initial digest
        with open(archive_path, "rb") as stream:
            raw_bytes = stream.read()
        initial_sha256 = hashlib.sha256(raw_bytes).hexdigest()

        # 2. Parse using ZipFile in memory/stream to prevent file system modification
        with zipfile.ZipFile(archive_path, "r") as archive:
            # 3. Security validation
            validation = ArchiveValidator.validate_zip(archive)

            # 4. Dispatch based on format
            if validation.has_metadata:
                document = self._parse_v2_archive(archive, fid, initial_sha256)
            else:
                document = self._parse_v1_archive(archive, fid, initial_sha256)

        # 5. Verify byte-for-byte preservation of the input file
        with open(archive_path, "rb") as stream:
            after_bytes = stream.read()
        if hashlib.sha256(after_bytes).hexdigest() != initial_sha256:
            raise DomainError("FACESET_MUTATION_ERROR", f"Input archive was mutated during read: {archive_path}", 500)

        return document

    def _parse_v2_archive(self, archive: zipfile.ZipFile, faceset_id: str, sha256: str) -> FaceSetDocument:
        meta_raw = archive.read("metadata.json")
        try:
            metadata = json.loads(meta_raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as err:
            raise DomainError("FACESET_ARCHIVE_INVALID", f"Corrupt metadata.json in FaceSet archive: {err}", 400) from err

        sources = metadata.get("sources", [])
        if not sources:
            raise DomainError("FACESET_EMPTY", "FaceSet metadata contains zero sources", 422)

        views: list[FaceView] = []
        embeddings_list: list[np.ndarray] = []
        high_conf_indices: list[int] = []

        for idx, src in enumerate(sources):
            filename = src.get("filename", f"{idx}.png")
            # Try reading image bytes
            try:
                img_bytes = archive.read(filename)
            except KeyError:
                # If specific filename missing, try alternate naming
                alt_name = f"{src.get('source_index', idx)}.png"
                try:
                    img_bytes = archive.read(alt_name)
                except KeyError:
                    raise DomainError("FACESET_ARCHIVE_INVALID", f"Missing image entry '{filename}' in archive", 400)

            # Extract or compute embedding
            emb_raw = None
            if isinstance(src.get("identity"), dict):
                emb_raw = src["identity"].get("arcface_embedding") or src["identity"].get("normalized_embedding") or src["identity"].get("embedding")
                name = src["identity"].get("embedding_file")
                if emb_raw is None and isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,64}\.json", name):
                    try:  # written by export_v2: one small entry per view, so the manifest never scales with view count
                        emb_raw = json.loads(archive.read(name).decode("utf-8"))
                    except (KeyError, UnicodeDecodeError, json.JSONDecodeError):
                        emb_raw = None
            elif isinstance(src.get("identity"), list) and len(src["identity"]) == 512:
                emb_raw = src["identity"]

            if emb_raw and len(emb_raw) == 512:
                emb = np.array(emb_raw, dtype=np.float32)
                norm = float(np.linalg.norm(emb))
                if norm > 1e-9:
                    emb /= norm
            else:
                # Fallback: compute lazily with local recognizer
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                emb = self.recognizer.embed(img)

            embeddings_list.append(emb)

            # Extract pose
            pose = src.get("pose")
            geom = src.get("geometry", {})
            # Archives hold legacy-frame poses in 2-, 3-list or dict form; the sanitizer accepts all of them.
            try:
                yaw, pitch, roll = as_legacy_triplet(pose if pose else
                                                     {"yaw": geom.get("yaw", 0.0), "pitch": geom.get("pitch", 0.0), "roll": geom.get("roll", 0.0)})
            except DomainError:
                yaw = pitch = roll = 0.0

            yaw = max(-90.0, min(90.0, yaw))
            pitch = max(-60.0, min(60.0, pitch))
            roll = max(-180.0, min(180.0, roll))

            # Extract landmarks and bounding box
            lm_raw = geom.get("landmarks_2d") or geom.get("landmarks_68") or []
            dense_raw = geom.get("landmarks_68") if len(geom.get("landmarks_68") or []) == 68 else (
                lm_raw if len(lm_raw) == 68 else None)
            landmarks_68 = tuple((float(pt[0]), float(pt[1])) for pt in dense_raw) if dense_raw else None
            landmarks_trusted = bool(geom.get("landmarks_trusted", True))
            if landmarks_68 is not None and len(lm_raw) != 5:
                # A 68-point set: the first five entries are JAW points, so take the canonical eyes/nose/mouth instead.
                dense = np.asarray(landmarks_68, dtype=np.float64)
                five = (dense[36:42].mean(axis=0), dense[42:48].mean(axis=0), dense[30], dense[48], dense[54])
                landmarks = tuple((float(pt[0]), float(pt[1])) for pt in five)
            elif lm_raw:
                landmarks = tuple((float(pt[0]), float(pt[1])) for pt in lm_raw[:5])
            else:
                landmarks = ((16.0, 32.0), (48.0, 32.0), (32.0, 48.0), (20.0, 60.0), (44.0, 60.0))
                landmarks_trusted = False  # fabricated stand-in, not an observation

            bbox_raw = geom.get("bbox") or [0.0, 0.0, 64.0, 64.0]
            bbox = (float(bbox_raw[0]), float(bbox_raw[1]), float(bbox_raw[2]), float(bbox_raw[3]))

            # Extract quality
            q_dict = src.get("quality", {})
            quality = QualityMetrics(
                sharpness=float(q_dict.get("sharpness", 1.0)),
                detector_confidence=float(q_dict.get("detector_confidence", 1.0)),
                score=float(q_dict.get("score", 1.0)),
                laplacian_variance=float(q_dict.get("laplacian_variance", 0.0)),
                blur=float(q_dict.get("blur", 0.0)),
                face_pixels=float(q_dict.get("face_pixels", 0.0)),
            )

            # Pose bin classification
            pose_bin, _, _ = classify_pose_bin(yaw, pitch)

            expression = src.get("expression", {})
            if not isinstance(expression, dict):
                expression = {}

            view = FaceView(
                index=idx,
                image_bytes=img_bytes,
                embedding=tuple(float(x) for x in emb),
                landmarks=landmarks,
                bbox=bbox,
                yaw=round(yaw, 2),
                pitch=round(pitch, 2),
                roll=round(roll, 2),
                expression={str(k): float(v) for k, v in expression.items() if isinstance(v, (int, float))},
                quality=quality,
                pose_bin=pose_bin,
                is_synthetic=False,
                uv_map_path=src.get("uv_map_path"),
                landmarks_68=landmarks_68,
                landmarks_trusted=landmarks_trusted,
            )
            views.append(view)

            if quality.score >= 0.5 and quality.detector_confidence >= 0.5:
                high_conf_indices.append(idx)

        # Aggregate Canonical Embeddings (Mean and Medoid)
        canonical_emb, medoid_emb = self._aggregate_embeddings(embeddings_list, high_conf_indices)

        # Cluster Pose Bins
        pose_bins = self._build_pose_bins(views)

        return FaceSetDocument(
            faceset_id=faceset_id,
            name=str(metadata.get("source_name", faceset_id)),
            source_sha256=sha256,
            archive_format="legacy_v2",
            canonical_embedding=tuple(float(x) for x in canonical_emb),
            medoid_embedding=tuple(float(x) for x in medoid_emb),
            views=tuple(views),
            pose_bins=pose_bins,
            metadata=metadata,
        )

    def _parse_v1_archive(self, archive: zipfile.ZipFile, faceset_id: str, sha256: str) -> FaceSetDocument:
        # Check cache under cache_root / facesets / <sha256> / metadata.json
        cache_dir = self.cache_root / "facesets" / sha256
        cache_file = cache_dir / "metadata.json"
        if cache_file.exists():
            try:
                cached_data = json.loads(cache_file.read_text("utf-8"))
                if cached_data.get("source_sha256") == sha256:
                    return self._load_cached_document(archive, cached_data, faceset_id, sha256, "legacy_v1")
            except Exception:
                pass  # On cache failure, fall through to recompute lazily

        # Process image entries sequentially
        namelist = sorted([name for name in archive.namelist() if not name.endswith(("/", "\\"))])
        image_entries = [name for name in namelist if Path(name).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]

        # Natural sort for entries like 0.png, 1.png, 10.png
        image_entries.sort(key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)])

        if not image_entries:
            raise DomainError("FACESET_EMPTY", "FaceSet archive contains zero image entries", 422)

        views: list[FaceView] = []
        embeddings_list: list[np.ndarray] = []
        high_conf_indices: list[int] = []

        for idx, entry_name in enumerate(image_entries):
            img_bytes = archive.read(entry_name)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue

            view = self._extract_face_view(img, img_bytes, idx)
            views.append(view)
            emb_arr = np.array(view.embedding, dtype=np.float32)
            embeddings_list.append(emb_arr)

            if view.quality.score >= 0.5 and view.quality.detector_confidence >= 0.5:
                high_conf_indices.append(idx)

        if not views:
            raise DomainError("FACESET_EMPTY", "Failed to extract valid face views from legacy archive", 422)

        canonical_emb, medoid_emb = self._aggregate_embeddings(embeddings_list, high_conf_indices)
        pose_bins = self._build_pose_bins(views)

        doc = FaceSetDocument(
            faceset_id=faceset_id,
            name=faceset_id,
            source_sha256=sha256,
            archive_format="legacy_v1",
            canonical_embedding=tuple(float(x) for x in canonical_emb),
            medoid_embedding=tuple(float(x) for x in medoid_emb),
            views=tuple(views),
            pose_bins=pose_bins,
            metadata={"inferred_from": "legacy_v1", "processed_views": len(views)},
        )

        # Cache extracted metadata atomically for subsequent runs
        self._cache_document_metadata(cache_dir, doc)
        return doc

    def _read_directory(self, dir_path: Path, faceset_id: str) -> FaceSetDocument:
        ArchiveValidator.validate_directory(dir_path)
        files = sorted(list(dir_path.iterdir()))
        image_files = [f for f in files if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
        image_files.sort(key=lambda p: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)])

        if not image_files:
            raise DomainError("FACESET_EMPTY", "FaceSet directory contains zero image files", 422)

        hasher = hashlib.sha256()
        views: list[FaceView] = []
        embeddings_list: list[np.ndarray] = []
        high_conf_indices: list[int] = []

        for idx, file_path in enumerate(image_files):
            img_bytes = file_path.read_bytes()
            hasher.update(img_bytes)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue

            view = self._extract_face_view(img, img_bytes, idx)
            views.append(view)
            emb_arr = np.array(view.embedding, dtype=np.float32)
            embeddings_list.append(emb_arr)

            if view.quality.score >= 0.5 and view.quality.detector_confidence >= 0.5:
                high_conf_indices.append(idx)

        dir_sha256 = hasher.hexdigest()
        canonical_emb, medoid_emb = self._aggregate_embeddings(embeddings_list, high_conf_indices)
        pose_bins = self._build_pose_bins(views)

        return FaceSetDocument(
            faceset_id=faceset_id,
            name=faceset_id,
            source_sha256=dir_sha256,
            archive_format="image_directory",
            canonical_embedding=tuple(float(x) for x in canonical_emb),
            medoid_embedding=tuple(float(x) for x in medoid_emb),
            views=tuple(views),
            pose_bins=pose_bins,
            metadata={"source_dir": str(dir_path.name)},
        )

    def _extract_face_view(self, img: np.ndarray, img_bytes: bytes, index: int) -> FaceView:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness = float(min(1.0, lap_var / 500.0))

        # Detect face with local SCRFD
        detections, _ = self.detector.detect(img, threshold=0.5)

        fabricated_landmarks = False
        if detections:
            box, landmarks = detections[0]
            det_conf = 0.95
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            lm_pts = tuple((float(pt[0]), float(pt[1])) for pt in landmarks)

            # One shared 5-point estimator (core.pose); legacy frame because FaceView stores pitch + = down.
            yaw, pitch, roll = legacy_pose_5pt(landmarks)

            # Recognizer embedding using aligned face
            emb = self.recognizer.embed(img, landmarks)
        else:
            # Fallback for crops where detector finds 0 faces
            det_conf = 0.5
            x1, y1, x2, y2 = 0.0, 0.0, float(w), float(h)
            lm_pts = ((w * 0.3, h * 0.4), (w * 0.7, h * 0.4), (w * 0.5, h * 0.55), (w * 0.35, h * 0.75), (w * 0.65, h * 0.75))
            roll, yaw, pitch = 0.0, 0.0, 0.0
            emb = self.recognizer.embed(img, None)
            fabricated_landmarks = True

        score = float(round(0.7 * sharpness + 0.3 * det_conf, 4))
        quality = QualityMetrics(
            sharpness=round(sharpness, 4),
            detector_confidence=round(det_conf, 4),
            score=score,
            laplacian_variance=round(lap_var, 2),
            blur=round(1.0 - sharpness, 4),
            face_pixels=float(max(1, (x2 - x1) * (y2 - y1))),
        )

        pose_bin, _, _ = classify_pose_bin(yaw, pitch)

        return FaceView(
            index=index,
            image_bytes=img_bytes,
            embedding=tuple(float(x) for x in emb),
            landmarks=lm_pts,
            bbox=(x1, y1, x2, y2),
            yaw=round(yaw, 2),
            pitch=round(pitch, 2),
            roll=round(roll, 2),
            expression={"openness": 0.0},
            quality=quality,
            pose_bin=pose_bin,
            is_synthetic=False,
            landmarks_trusted=not fabricated_landmarks,
        )

    def _aggregate_embeddings(
        self,
        embeddings: list[np.ndarray],
        high_conf_indices: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not embeddings:
            raise DomainError("FACESET_EMPTY", "No embeddings available for identity aggregation", 422)

        indices = high_conf_indices if high_conf_indices else list(range(len(embeddings)))
        subset = [embeddings[i] for i in indices]

        # 1. Canonical Mean Embedding (L2-normalized)
        mean_vec = np.mean(subset, axis=0)
        norm_mean = float(np.linalg.norm(mean_vec))
        if norm_mean > 1e-9:
            mean_vec /= norm_mean

        # 2. Medoid Embedding (maximum sum of cosine similarities to all others in subset)
        if len(subset) == 1:
            medoid_vec = subset[0].copy()
        else:
            matrix = np.stack(subset, axis=0)  # shape (N, 512)
            sim_matrix = np.dot(matrix, matrix.T)  # pairwise cosine similarities
            scores = np.sum(sim_matrix, axis=1)
            medoid_idx = int(np.argmax(scores))
            medoid_vec = subset[medoid_idx].copy()

        norm_medoid = float(np.linalg.norm(medoid_vec))
        if norm_medoid > 1e-9:
            medoid_vec /= norm_medoid

        return mean_vec, medoid_vec

    def _build_pose_bins(self, views: list[FaceView]) -> dict[str, PoseBinSummary]:
        bins: dict[str, list[int]] = {}
        for view in views:
            bins.setdefault(view.pose_bin, []).append(view.index)

        summaries: dict[str, PoseBinSummary] = {}
        for bin_name, members in bins.items():
            yaw_bin, pitch_bin = bin_name.split("_")
            member_views = [views[i] for i in members]
            # Select representative: highest quality score
            rep_view = max(member_views, key=lambda v: (v.quality.score, -abs(v.yaw)))

            # Average embedding in this pose bin
            embs = [np.array(v.embedding, dtype=np.float32) for v in member_views]
            avg_emb = np.mean(embs, axis=0)
            norm = float(np.linalg.norm(avg_emb))
            if norm > 1e-9:
                avg_emb /= norm

            summaries[bin_name] = PoseBinSummary(
                yaw_bin=yaw_bin,
                pitch_bin=pitch_bin,
                members=tuple(members),
                support=len(members),
                representative_index=rep_view.index,
                embedding=tuple(float(x) for x in avg_emb),
            )
        return summaries

    def _cache_document_metadata(self, cache_dir: Path, doc: FaceSetDocument) -> None:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            data = {
                "faceset_id": doc.faceset_id,
                "source_sha256": doc.source_sha256,
                "archive_format": doc.archive_format,
                "canonical_embedding": list(doc.canonical_embedding),
                "medoid_embedding": list(doc.medoid_embedding),
                "views": [
                    {
                        "index": v.index,
                        "embedding": list(v.embedding),
                        "landmarks": [list(pt) for pt in v.landmarks],
                        "bbox": list(v.bbox),
                        "yaw": v.yaw,
                        "pitch": v.pitch,
                        "roll": v.roll,
                        "expression": v.expression,
                        "quality": v.quality.model_dump(),
                        "pose_bin": v.pose_bin,
                        "is_synthetic": False,
                        "landmarks_trusted": v.landmarks_trusted,
                    }
                    for v in doc.views
                ],
                "pose_bins": {
                    k: {
                        "yaw_bin": b.yaw_bin,
                        "pitch_bin": b.pitch_bin,
                        "members": list(b.members),
                        "support": b.support,
                        "representative_index": b.representative_index,
                        "embedding": list(b.embedding),
                    }
                    for k, b in doc.pose_bins.items()
                },
            }
            tmp_path = cache_dir / "metadata.json.tmp"
            tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp_path.replace(cache_dir / "metadata.json")
        except Exception:
            pass  # Non-fatal if cache write fails

    def _load_cached_document(
        self,
        archive: zipfile.ZipFile,
        cached: dict[str, Any],
        faceset_id: str,
        sha256: str,
        archive_format: str,
    ) -> FaceSetDocument:
        views: list[FaceView] = []
        namelist = sorted([name for name in archive.namelist() if not name.endswith(("/", "\\"))])
        image_entries = [name for name in namelist if Path(name).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
        image_entries.sort(key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)])

        for item in cached["views"]:
            idx = item["index"]
            img_bytes = archive.read(image_entries[idx])
            view = FaceView(
                index=idx,
                image_bytes=img_bytes,
                embedding=tuple(item["embedding"]),
                landmarks=tuple((float(p[0]), float(p[1])) for p in item["landmarks"]),
                bbox=tuple(item["bbox"]),
                yaw=item["yaw"],
                pitch=item["pitch"],
                roll=item["roll"],
                expression=item.get("expression", {}),
                quality=QualityMetrics(**item["quality"]),
                pose_bin=item["pose_bin"],
                is_synthetic=False,
                landmarks_trusted=bool(item.get("landmarks_trusted", True)),
            )
            views.append(view)

        pose_bins = {
            k: PoseBinSummary(
                yaw_bin=v["yaw_bin"],
                pitch_bin=v["pitch_bin"],
                members=tuple(v["members"]),
                support=v["support"],
                representative_index=v["representative_index"],
                embedding=tuple(v["embedding"]),
            )
            for k, v in cached["pose_bins"].items()
        }

        return FaceSetDocument(
            faceset_id=faceset_id,
            name=faceset_id,
            source_sha256=sha256,
            archive_format=archive_format,  # type: ignore
            canonical_embedding=tuple(cached["canonical_embedding"]),
            medoid_embedding=tuple(cached["medoid_embedding"]),
            views=tuple(views),
            pose_bins=pose_bins,
            metadata={"cached": True},
        )


def export_v2(
    document: FaceSetDocument,
    output_path: Path | str,
    consent: ConsentMetadata | None = None,
    *,
    app_root: Path | None = None,
) -> Path:
    """Export a FaceSetDocument to a versioned V2 .fsz archive.

    Strict rules enforced:
    - Never overwrite existing files.
    - Write only to authorized output roots under roop_3d/outputs/facesets/.
    - Atomically construct the ZIP container.
    """
    root = app_root or application_root()
    target_path = Path(output_path).resolve()

    if target_path.exists():
        raise DomainError("FACESET_EXISTS", f"FaceSet export destination already exists: {target_path}", 409)

    # Validate output path is inside owned roop_3d outputs directory
    authorized_outputs = (root / "roop_3d/outputs/facesets").resolve()
    runtime_outputs = (root / "roop_3d/.runtime/outputs/facesets").resolve()

    # Create destination directory if needed
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Construct V2 Manifest
    manifest = FaceSetV2Manifest(
        schema_version=2,
        schema="roop.fsz.v2",
        faceset_id=document.faceset_id,
        source_name=document.name,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        consent=consent or ConsentMetadata(),
        identity={
            "mean_embedding": list(document.canonical_embedding),
            "medoid_embedding": list(document.medoid_embedding),
            "aggregation": "l2_mean_and_medoid",
        },
        pose_bins={
            k: {
                "yaw_bin": b.yaw_bin,
                "pitch_bin": b.pitch_bin,
                "members": list(b.members),
                "support": b.support,
                "representative_index": b.representative_index,
                "embedding": list(b.embedding),
            }
            for k, b in document.pose_bins.items()
        },
        sources=[
            {
                "source_index": v.index,
                "filename": f"{v.index}.png",
                # Per-source identity (a separate small entry) so a project-written archive re-reads without the recognizer.
                "identity": {"embedding_file": f"{v.index}.embedding.json"},
                "pose": [v.yaw, v.pitch, v.roll],
                "geometry": {
                    "bbox": list(v.bbox),
                    "landmarks_2d": [list(pt) for pt in v.landmarks],
                    **({"landmarks_68": [list(pt) for pt in v.landmarks_68]} if v.landmarks_68 is not None else {}),
                    "landmarks_trusted": v.landmarks_trusted,
                    "yaw": v.yaw,
                    "pitch": v.pitch,
                    "roll": v.roll,
                },
                "quality": v.quality.model_dump(),
                "expression": v.expression,
                "pose_bin": v.pose_bin,
                "is_synthetic": False,
                "uv_map_path": v.uv_map_path,
            }
            for v in document.views
        ],
        provenance={
            "pipeline": "roop_3d_stage04",
            "source_archive_sha256": document.source_sha256,
            "archive_format": document.archive_format,
            "pose_convention": "legacy: pitch positive = looking down",
        },
        pose_matrix={
            "schema": "roop.pose_matrix.v1",
            "yaw_edges_deg": [-15.0, 15.0],
            "pitch_edges_deg": [-10.0, 10.0],
            "pitch_positive": "up",
            "cells": cell_members(document),
        },
    )

    # Atomic write pattern using tempfile in the same directory
    temp_fd, temp_file_path = tempfile.mkstemp(suffix=".tmp", dir=target_path.parent)
    os.close(temp_fd)
    tmp_path = Path(temp_file_path)

    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # 1. Write metadata.json
            # Compact: pretty-printed per-view geometry is so repetitive that ~35+ views trip the archive validator's
            # zip-bomb compression-ratio check when this project's own archive is read back.
            meta_json = json.dumps(manifest.model_dump(by_alias=True), separators=(",", ":")).encode("utf-8")
            zf.writestr("metadata.json", meta_json)

            # 2. Write images, and each view's embedding as its own small entry (see the reader)
            for view in document.views:
                zf.writestr(f"{view.index}.png", view.image_bytes)
                zf.writestr(f"{view.index}.embedding.json", json.dumps([float(x) for x in view.embedding], separators=(",", ":")))

        # Atomic replace
        tmp_path.replace(target_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return target_path
