"""FaceSet / photo loader for the 3D head pipeline (Stage 11).

Reads a ``.fsz`` archive (roop-ultimate ``legacy_v1``/``legacy_v2`` and roop-3d ``v2_standard`` formats are both handled
by :class:`FaceSetReader`), a directory of photos, or one or more image files, and turns every face sample into a
:class:`FaceSample`:

* 512-D ArcFace identity embedding (``normed_embedding``, L2-normalised);
* 68-point 3D (``1k3d68``) and 106-point 2D (``2d106det``) landmarks when those local ONNX models are installed;
  otherwise the sample keeps the archive's 5-point landmarks and says so (``landmark_source``) - nothing is fabricated;
* Euler pose ``(yaw, pitch, roll)`` in degrees, standard frame (pitch + = up), from a PnP fit of the best landmark layout;
* an aligned 512x512 RGB face crop (ArcFace similarity alignment);
* its place in BOTH pose indexes: the 96-bin geodesic sphere (12 yaw x 8 pitch) and the roop-ultimate 3x3 yaw/pitch
  matrix with roll as the third axis.

:func:`enrich_document` attaches the dense landmarks to the ``FaceView`` objects so the identity fit
(``identity_asset``) and the MakeHuman deformation can use them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import logging
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Sequence

import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.core.pose import PoseCell, pose_cell
from roop_3d.faceset.faceset_reader import FaceSetReader
from roop_3d.faceset.schemas import FaceSetDocument, FaceView
from roop_3d.reconstruction.faceset_3d import PITCH_BINS, TOTAL_BINS, YAW_BINS, _pose, pose_bin
from roop_3d.reconstruction.multiview_fit import five_from_68

log = logging.getLogger("roop3d.faceset_loader")

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ROLL_EDGE_DEG = 8.0
ROLL_NAMES = ("ccw", "level", "cw")
ARCFACE_112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], np.float64)


# --------------------------------------------------------------------------- landmark models
def _model_dirs(root: Path | None) -> list[Path]:
    dirs: list[Path] = []
    env = os.environ.get("ROOP3D_LANDMARK_MODEL_DIR")
    if env:
        dirs.append(Path(env))
    if root is not None:
        dirs += [root / "roop_3d/.runtime/models", root / "roop_3d/.runtime/models/buffalo_l", root / "models/buffalo_l",
                 root.parent / "roop-ultimate/app/models/buffalo_l"]
        # the app root is <pinokio>/api/roop-3d/app, so a sibling launcher's insightface pack sits two levels up
        dirs.append(root.parent.parent / "roop-ultimate/app/models/buffalo_l")
    return [d for d in dirs if d.is_dir()]


class LandmarkEngine:
    """Lazy ``1k3d68`` + ``2d106det`` ONNX runner (insightface protocol: 192 px crop at 1.5x the face box)."""

    def __init__(self, root: Path | None = None, model_dirs: Sequence[Path] | None = None):
        self._dirs = [Path(d) for d in model_dirs] if model_dirs is not None else _model_dirs(root)
        self._sessions: dict[str, Any] = {}
        self._lock = threading.Lock()

    def _path(self, name: str) -> Path | None:
        for d in self._dirs:
            if (d / name).is_file():
                return d / name
        return None

    def _session(self, name: str):
        with self._lock:
            if name not in self._sessions:
                path = self._path(name)
                session = None
                if path is not None:
                    try:
                        import onnxruntime as ort
                        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                    except Exception as err:                       # unreadable model == unavailable, never fabricated
                        log.warning("landmark model %s unusable: %s", name, err)
                self._sessions[name] = session
            return self._sessions[name]

    @property
    def has_68(self) -> bool:
        return self._session("1k3d68.onnx") is not None

    @property
    def has_106(self) -> bool:
        return self._session("2d106det.onnx") is not None

    def signature(self) -> str:
        """Identity of the available models, part of cache keys (dense landmarks change when the model does)."""
        parts = []
        for name in ("1k3d68.onnx", "2d106det.onnx"):
            p = self._path(name)
            parts.append(f"{name}:{p.stat().st_size if p and p.is_file() else 'absent'}")
        return "|".join(parts)

    @staticmethod
    def _crop(image: np.ndarray, bbox: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        side = max(x2 - x1, y2 - y1, 1.0)
        scale = 192.0 / (side * 1.5)
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        matrix = np.array([[scale, 0, 96 - center[0] * scale], [0, scale, 96 - center[1] * scale]], np.float64)
        return cv2.warpAffine(image, matrix, (192, 192), borderValue=0.0), cv2.invertAffineTransform(matrix)

    def landmarks68(self, image: np.ndarray, bbox: Sequence[float]) -> np.ndarray | None:
        """(68, 3): x, y in image pixels, z in the same pixel units (relative depth, + toward the camera ordering of the model)."""
        session = self._session("1k3d68.onnx")
        if session is None:
            return None
        crop, inverse = self._crop(image, bbox)
        blob = cv2.dnn.blobFromImage(crop, 1.0, (192, 192), (0, 0, 0), swapRB=True)
        out = session.run(None, {session.get_inputs()[0].name: blob})[0][0].reshape(-1, 3)[-68:].astype(np.float64)
        xy = (out[:, :2] + 1.0) * 96.0
        z = out[:, 2] * 96.0 * float(np.hypot(inverse[0, 0], inverse[1, 0]))
        xy = xy @ inverse[:, :2].T + inverse[:, 2]
        return np.column_stack([xy, z])

    def landmarks106(self, image: np.ndarray, bbox: Sequence[float]) -> np.ndarray | None:
        session = self._session("2d106det.onnx")
        if session is None:
            return None
        crop, inverse = self._crop(image, bbox)
        blob = cv2.dnn.blobFromImage(crop, 1.0, (192, 192), (0, 0, 0), swapRB=True)
        out = session.run(None, {session.get_inputs()[0].name: blob})[0][0].reshape(-1, 2).astype(np.float64)
        return ((out + 1.0) * 96.0) @ inverse[:, :2].T + inverse[:, 2]


# --------------------------------------------------------------------------- samples
@dataclass(frozen=True)
class FaceSample:
    view_index: int
    normed_embedding: np.ndarray                 # (512,) float32, unit length
    landmarks_5: np.ndarray                      # (5, 2) archive / detector landmarks, image pixels
    landmarks_68: np.ndarray | None              # (68, 3) 1k3d68 (x, y px; z relative depth) or None
    landmarks_106: np.ndarray | None             # (106, 2) 2d106det or None
    landmark_source: str                         # "1k3d68+2d106det" | "1k3d68" | "archive_5pt"
    yaw: float                                   # degrees, standard frame (pitch + = up)
    pitch: float
    roll: float
    reprojection_px: float
    crop_rgb_512: np.ndarray                     # (512, 512, 3) uint8 RGB, ArcFace-aligned
    sphere_bin: int                              # 0..95  (pitch_bin * 12 + yaw_bin)
    matrix_cell: str                             # "left_level" ...
    matrix_index: tuple[int, int, int]           # (yaw 0..2, pitch 0..2, roll 0..2)
    quality: float
    trusted: bool

    def summary(self) -> dict[str, Any]:
        return {"view_index": self.view_index, "yaw": round(self.yaw, 2), "pitch": round(self.pitch, 2),
                "roll": round(self.roll, 2), "sphere_bin": self.sphere_bin, "matrix_cell": self.matrix_cell,
                "matrix_index": list(self.matrix_index), "landmark_source": self.landmark_source,
                "has_68": self.landmarks_68 is not None, "has_106": self.landmarks_106 is not None,
                "embedding_norm": round(float(np.linalg.norm(self.normed_embedding)), 6),
                "reprojection_px": round(self.reprojection_px, 3), "trusted": self.trusted}


@dataclass
class LoadedFaceSet:
    document: FaceSetDocument                    # views carry landmarks_68 when the dense model is available
    samples: list[FaceSample]
    sphere_bins: dict[int, list[int]]            # bin index -> view indices (96-bin geodesic sphere)
    pose_matrix: dict[tuple[int, int, int], list[int]]   # (yaw, pitch, roll) cell -> view indices
    landmark_models: dict[str, bool]
    notes: list[str] = field(default_factory=list)

    @property
    def sphere_coverage_count(self) -> int:
        return len(self.sphere_bins)

    def summary(self) -> dict[str, Any]:
        cells = {}
        for (y, p, r), members in sorted(self.pose_matrix.items()):
            cells[f"{PoseCell(y, p).name}|roll_{ROLL_NAMES[r]}"] = members
        return {"faceset_id": self.document.faceset_id, "archive_format": self.document.archive_format,
                "samples": [s.summary() for s in self.samples], "sphere_total_bins": TOTAL_BINS,
                "sphere_grid": [YAW_BINS, PITCH_BINS], "sphere_coverage_count": self.sphere_coverage_count,
                "sphere_bins": {str(k): v for k, v in sorted(self.sphere_bins.items())},
                "pose_matrix_3x3_roll": cells, "landmark_models": self.landmark_models, "notes": self.notes}


def aligned_crop_512(image_bgr: np.ndarray, landmarks5: np.ndarray) -> np.ndarray:
    """ArcFace similarity alignment rendered at 512 px, returned as RGB."""
    src = np.asarray(landmarks5, np.float64).reshape(5, 2)
    dst = ARCFACE_112 * (512.0 / 112.0)
    matrix, _ = cv2.estimateAffinePartial2D(src.astype(np.float32), dst.astype(np.float32), method=cv2.LMEDS)
    if matrix is None:
        side = min(image_bgr.shape[:2])
        crop = cv2.resize(image_bgr[:side, :side], (512, 512), interpolation=cv2.INTER_AREA)
    else:
        crop = cv2.warpAffine(image_bgr, matrix, (512, 512), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def _roll_index(roll: float) -> int:
    return 0 if roll < -ROLL_EDGE_DEG else 2 if roll > ROLL_EDGE_DEG else 1


def _consistent_with_5pt(points68: np.ndarray, points5: np.ndarray) -> bool:
    """Reject a dense fit that disagrees with the detector's own five points (mean error > 12 % of the eye distance)."""
    iod = float(np.linalg.norm(points5[1] - points5[0]))
    if iod < 1.0:
        return False
    err = float(np.linalg.norm(five_from_68(points68[:, :2]) - points5, axis=1).mean())
    return err <= 0.12 * iod


def sample_view(view: FaceView, engine: LandmarkEngine | None, image: np.ndarray | None = None) -> tuple[FaceSample, FaceView]:
    if image is None:
        image = cv2.imdecode(np.frombuffer(view.image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise DomainError("IMAGE_UNDECODABLE", f"FaceSet view {view.index} cannot be decoded", 422)
    five = np.asarray(view.landmarks, np.float64)
    h, w = image.shape[:2]
    bbox = view.bbox if (view.bbox[2] - view.bbox[0]) > 8 and (view.bbox[3] - view.bbox[1]) > 8 else (0.0, 0.0, float(w), float(h))
    dense68 = dense106 = None
    if view.landmarks_68 is not None:                                   # the archive already carries them: keep as is
        dense68 = np.column_stack([np.asarray(view.landmarks_68, np.float64), np.zeros(68)])
    elif engine is not None and view.landmarks_trusted:
        candidate = engine.landmarks68(image, bbox)
        if candidate is not None and np.isfinite(candidate).all() and _consistent_with_5pt(candidate, five):
            dense68 = candidate
    if engine is not None and view.landmarks_trusted:
        dense106 = engine.landmarks106(image, bbox)
        if dense106 is not None and not np.isfinite(dense106).all():
            dense106 = None
    updated = view.model_copy(update={"landmarks_68": tuple((float(x), float(y)) for x, y in dense68[:, :2])}) \
        if dense68 is not None and view.landmarks_68 is None else view
    yaw, pitch, roll, reproj = _pose(updated)
    embedding = np.asarray(view.embedding, np.float32)
    embedding = embedding / max(float(np.linalg.norm(embedding)), 1e-12)
    cell = pose_cell({"yaw": yaw, "pitch": pitch, "roll": roll}, frame="standard") \
        if False else pose_cell((yaw, -pitch, roll))                    # raw tuples are legacy (pitch + = down)
    source = "archive_5pt" if dense68 is None else ("1k3d68+2d106det" if dense106 is not None else "1k3d68")
    sample = FaceSample(
        view_index=view.index, normed_embedding=embedding, landmarks_5=five, landmarks_68=dense68, landmarks_106=dense106,
        landmark_source=source, yaw=yaw, pitch=pitch, roll=roll, reprojection_px=reproj,
        crop_rgb_512=aligned_crop_512(image, five), sphere_bin=pose_bin(yaw, pitch), matrix_cell=cell.name,
        matrix_index=(cell.yaw_index, cell.pitch_index, _roll_index(roll)), quality=float(view.quality.score),
        trusted=bool(view.landmarks_trusted))
    return sample, updated


def build_loaded(document: FaceSetDocument, engine: LandmarkEngine | None) -> LoadedFaceSet:
    samples, views = [], []
    for view in document.views:
        sample, updated = sample_view(view, engine)
        samples.append(sample)
        views.append(updated)
    bins: dict[int, list[int]] = {}
    matrix: dict[tuple[int, int, int], list[int]] = {}
    for s in samples:
        bins.setdefault(s.sphere_bin, []).append(s.view_index)
        matrix.setdefault(s.matrix_index, []).append(s.view_index)
    notes = []
    models = {"1k3d68": bool(engine and engine.has_68), "2d106det": bool(engine and engine.has_106)}
    if not models["1k3d68"]:
        notes.append("1k3d68 model not installed: geometry fits use the archive's 5-point landmarks (set "
                     "ROOP3D_LANDMARK_MODEL_DIR to an insightface buffalo_l folder).")
    dense = sum(s.landmarks_68 is not None for s in samples)
    if models["1k3d68"] and dense < len(samples):
        notes.append(f"{len(samples) - dense} of {len(samples)} views kept 5-point landmarks (dense fit missing or "
                     "inconsistent with the detector landmarks).")
    enriched = document.model_copy(update={"views": tuple(views)})
    return LoadedFaceSet(document=enriched, samples=samples, sphere_bins=bins, pose_matrix=matrix,
                         landmark_models=models, notes=notes)


# --------------------------------------------------------------------------- entry points
_CACHE: dict[tuple[str, str], LoadedFaceSet] = {}
_CACHE_LOCK = threading.Lock()


def enrich_document(document: FaceSetDocument, engine: LandmarkEngine | None, *, cache: bool = True) -> LoadedFaceSet:
    """Samples + dense landmarks for an already-read document (cached per archive hash and landmark-model signature)."""
    key = (document.source_sha256, engine.signature() if engine else "none")
    if cache:
        with _CACHE_LOCK:
            hit = _CACHE.get(key)
        if hit is not None and hit.document.faceset_id == document.faceset_id:
            return hit
    loaded = build_loaded(document, engine)
    if cache:
        with _CACHE_LOCK:
            if len(_CACHE) > 16:
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[key] = loaded
    return loaded


def load_faceset(source: Path | str | Sequence[Path | str], *, catalog: Any = None, root: Path | None = None,
                 cache_root: Path | None = None, engine: LandmarkEngine | None = None,
                 faceset_id: str | None = None) -> LoadedFaceSet:
    """Load a ``.fsz`` archive, a photo directory, or one/many photos into a :class:`LoadedFaceSet`."""
    engine = engine if engine is not None else LandmarkEngine(root)
    reader = FaceSetReader(catalog, cache_root=cache_root)
    paths = [Path(p) for p in ([source] if isinstance(source, (str, Path)) else source)]
    if not paths:
        raise DomainError("FACESET_SOURCE_EMPTY", "Provide a .fsz archive or at least one image", 422)
    if len(paths) == 1 and paths[0].suffix.casefold() == ".fsz":
        if not paths[0].is_file():
            raise DomainError("FACESET_NOT_FOUND", f"FaceSet archive {paths[0].name} does not exist", 404)
        return enrich_document(reader.read(paths[0], faceset_id=faceset_id or paths[0].stem), engine)
    if len(paths) == 1 and paths[0].is_dir():
        return enrich_document(reader.read(paths[0], faceset_id=faceset_id or paths[0].name[:64]), engine)
    for p in paths:
        if p.suffix.casefold() not in IMAGE_SUFFIXES or not p.is_file():
            raise DomainError("SOURCE_IMAGE_INVALID", f"{p.name} is missing or not a supported image", 422)
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()[:16]
    base = (cache_root or Path(tempfile.gettempdir())) / "stage11_images" / digest
    base.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(paths):
        target = base / f"{i:03d}{p.suffix.casefold()}"
        if not target.exists():
            target.write_bytes(p.read_bytes())
    return enrich_document(reader.read(base, faceset_id=faceset_id or f"photos_{digest}"), engine)
