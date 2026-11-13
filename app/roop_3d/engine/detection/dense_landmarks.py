"""Read-only dense ONNX landmark adapters and evidence-labelled pose estimation."""
from __future__ import annotations

import math
import json
from dataclasses import dataclass
import cv2
import numpy as np

from roop_3d.engine.runtime import OrtModel, RuntimeConfig, image_bgr, tensor
from roop_3d.core.errors import DomainError
from roop_3d.core.pose import legacy_pose_5pt
from roop_3d.engine.volumetric.preprocess import landmark_crop, transform_points
from roop_3d.models.catalog import ModelCatalog
from roop_3d.reconstruction.pose_solver import HeadPose6DoF, solve_head_pose_6dof

POSE_SOURCE_GEOMETRIC = "geometric_5pt"
# A dense PnP fit is rejected (and the 5-point estimate used, reported as such) when its RMS exceeds this fraction of the
# landmark extent, i.e. the canonical model does not explain the observed layout.
PNP_MAX_RMS_EXTENT_FRACTION = 0.08
PNP_MIN_RMS_BUDGET_PX = 5.0


class DenseLandmarks:
    def __init__(self, catalog: ModelCatalog, config: RuntimeConfig = RuntimeConfig()):
        self.catalog, self.config = catalog, config
        self.model68: OrtModel | None = None
        self.model106: OrtModel | None = None
        self.model203: OrtModel | None = None
        self._model68_unavailable = False
        self.mesh478 = None
        self._mesh_checked = False
    def infer68(self, image: np.ndarray, box: np.ndarray) -> np.ndarray | None:
        """Return verified 1k3d68 landmarks in image pixels, or None when the optional model is absent."""
        image_bgr(image)
        box = tensor(np.asarray(box, np.float32), (4,))
        if box[2] <= box[0] or box[3] <= box[1]:
            raise DomainError("LANDMARK_BOX_INVALID", "Dense landmarks require a positive-area face box", 422)
        if self._model68_unavailable:
            return None
        if self.model68 is None:
            try:
                self.model68 = OrtModel(self.catalog, "1k3d68.onnx", self.config)
            except DomainError as error:
                if error.code == "MODEL_NOT_INSTALLED":
                    self._model68_unavailable = True
                    return None
                raise
        x1, y1, x2, y2 = box
        center = np.array(((x1 + x2) / 2, (y1 + y2) / 2))
        scale = 192 / (max(x2 - x1, y2 - y1) * 1.5)
        matrix = np.column_stack((np.eye(2) * scale, 96 - center * scale))
        crop = cv2.warpAffine(image, matrix, (192, 192))
        blob = cv2.dnn.blobFromImage(crop, 1.0, (192, 192), (0, 0, 0), swapRB=True)
        output = next(iter(self.model68.run({self.model68.session.get_inputs()[0].name: blob}).values()))
        raw = np.asarray(output, np.float32)
        if raw.size % 3:
            raise DomainError("LANDMARK_68_OUTPUT_INVALID", "1k3d68 returned a non-xyz landmark tensor", 503)
        flattened = raw.reshape(-1, 3)
        if len(flattened) < 68 or not np.isfinite(flattened).all():
            raise DomainError("LANDMARK_68_OUTPUT_INVALID", "1k3d68 returned an invalid landmark tensor", 503)
        points = flattened[-68:].astype(np.float64)
        xy = (points[:, :2] + 1.0) * 96.0
        depth = points[:, 2] * 96.0 / scale
        inverse = cv2.invertAffineTransform(matrix)
        xy = xy @ inverse[:, :2].T + inverse[:, 2]
        return np.column_stack([xy, depth]).astype(np.float32)


    def infer478(self, image, box):
        """Optional verified local task, cropped per actor to prevent cross-track gaze.

        Install app/models/face_landmarker.task plus a .task.json sidecar with
        sha256 and size_bytes. Nothing is downloaded or borrowed from siblings.
        A missing task returns None; a configured but broken task is an error.
        """
        if not self._mesh_checked:
            path = self.catalog.root / 'models/face_landmarker.task'
            sidecar = path.with_suffix('.task.json')
            if path.is_file():
                try:
                    from roop_3d.flame_3d_engine import MediaPipe478Estimator
                    metadata = json.loads(sidecar.read_text('utf-8'))
                    self.mesh478 = MediaPipe478Estimator(path, metadata['sha256'], metadata['size_bytes'])
                except (OSError, KeyError, ValueError) as error:
                    raise DomainError('IRIS_TASK_INVALID', f'Local iris task needs a valid digest/size sidecar: {error}', 422) from error
            self._mesh_checked = True
        if self.mesh478 is None:
            return None
        h, w = image.shape[:2]
        x1, y1, x2, y2 = np.asarray(box, np.float64)
        margin = max(x2-x1, y2-y1) * .15
        x1, y1 = max(0, int(x1-margin)), max(0, int(y1-margin))
        x2, y2 = min(w, int(x2+margin)), min(h, int(y2+margin))
        if x2-x1 < 2 or y2-y1 < 2:
            return None
        try:
            mesh = self.mesh478.extract(np.ascontiguousarray(image[y1:y2, x1:x2, ::-1]))
        except RuntimeError as error:
            if str(error) == 'No single dense face mesh was detected':
                return None
            raise
        mesh = mesh.copy()
        mesh[:, 0] = mesh[:, 0] * (x2-x1) + x1
        mesh[:, 1] = mesh[:, 1] * (y2-y1) + y1
        mesh[:, 2] *= x2-x1
        return mesh

    def infer106(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        image_bgr(image)
        box = tensor(np.asarray(box, np.float32), (4,))
        if box[2] <= box[0] or box[3] <= box[1]:
            raise DomainError("LANDMARK_BOX_INVALID", "Dense landmarks require a positive-area face box", 422)
        if self.model106 is None:
            self.model106 = OrtModel(self.catalog, "2d106det.onnx", self.config)
        x1, y1, x2, y2 = box
        center = np.array(((x1 + x2) / 2, (y1 + y2) / 2))
        scale = 192 / (max(x2 - x1, y2 - y1) * 1.5)
        matrix = np.column_stack((np.eye(2) * scale, 96 - center * scale))
        crop = cv2.warpAffine(image, matrix, (192, 192))
        # The verified graph starts with scalar Sub/Mul normalization internally.
        blob = cv2.dnn.blobFromImage(crop, 1, (192, 192), swapRB=True)
        output = self.model106.run({"data": blob})["fc1"]
        tensor(output, (1, 212))
        points = (output.reshape(106, 2) + 1) * 96
        return transform_points(points, cv2.invertAffineTransform(matrix)).astype(np.float32)

    def infer203(self, image: np.ndarray, points106: np.ndarray) -> np.ndarray:
        if self.model203 is None:
            self.model203 = OrtModel(self.catalog, "liveportrait_landmark.onnx", self.config)
        crop = landmark_crop(image, points106, 224)
        blob = cv2.dnn.blobFromImage(crop.image, 1 / 255, (224, 224), swapRB=True)
        output = self.model203.run({"input": blob})["856"]
        tensor(output, (1, 406))
        return transform_points(output.reshape(203, 2) * 224, crop.inverse).astype(np.float32)

    def estimate_pose(self, image: np.ndarray, points5: np.ndarray, landmarks68: np.ndarray | None = None,
                      prev_rotation: np.ndarray | None = None) -> "PoseObservation":
        """Observation pose for one face: dense-landmark PnP when 68 landmarks exist, else the 5-point estimate."""
        height, width = image.shape[:2]
        return observation_pose(points5, landmarks68=landmarks68, image_size=(width, height), prev_rotation=prev_rotation)

    def release(self) -> None:
        for model in (self.model106, self.model203):
            if model is not None:
                model.release()
        self.model106 = self.model203 = None
        if self.mesh478 is not None:
            self.mesh478.close()
            self.mesh478 = None
        self._mesh_checked = False
        if self.model68 is not None:
            self.model68.release()
            self.model68 = None
        self._model68_unavailable = False


def geometric_pose(points5: np.ndarray) -> tuple[float, float, float]:
    """Yaw/pitch/roll estimate in the legacy frame (pitch + = down); not calibrated 3D reconstruction.

    Thin wrapper over the single shared estimator `core.pose.estimate_pose_5pt`."""
    return legacy_pose_5pt(points5)


@dataclass(frozen=True)
class PoseObservation:
    """Pose of one observation with its provenance. ``pose`` is the legacy tuple (pitch positive = down) that
    ``FaceObservation.pose`` stores; ``pose6dof`` carries the metric solve when PnP produced it."""
    pose: tuple[float, float, float]
    source: str                     # "pnp_dense" | "geometric_5pt"
    pose6dof: HeadPose6DoF | None
    fallback_reason: str | None     # why the 5-point estimate was used instead of dense PnP; None when PnP succeeded


def observation_pose(points5: np.ndarray, *, landmarks68: np.ndarray | None = None,
                     image_size: tuple[int, int] | None = None,
                     prev_rotation: np.ndarray | None = None) -> PoseObservation:
    """Pose from dense-landmark PnP (``reconstruction.pose_solver``) when >= 68 dense landmarks and ``image_size`` exist.

    Falls back to the 5-point geometric estimate only when no dense set is available or PnP fails / does not explain the
    layout; the returned ``source`` and ``fallback_reason`` state which happened (never a silent substitution)."""
    reason = "no_dense_landmarks" if landmarks68 is None else "image_size_missing" if image_size is None else None
    if reason is None:
        dense = np.asarray(landmarks68, np.float64)
        if dense.ndim != 2 or dense.shape[0] < 68 or dense.shape[1] not in (2, 3):
            reason = "dense_landmarks_invalid_shape"
        else:
            try:
                solved = solve_head_pose_6dof(dense[:, :2], image_size=image_size, prev_rotation=prev_rotation)
                extent = float(np.linalg.norm(dense[:, :2].max(axis=0) - dense[:, :2].min(axis=0)))
                if solved.reproj_rms_px > max(PNP_MIN_RMS_BUDGET_PX, PNP_MAX_RMS_EXTENT_FRACTION * extent):
                    reason = "pnp_reprojection_too_high"
                else:
                    return PoseObservation((solved.yaw_deg, -solved.pitch_deg, solved.roll_deg), solved.source, solved, None)
            except DomainError as error:
                reason = f"pnp_failed:{error.code}"
    return PoseObservation(tuple(float(v) for v in geometric_pose(points5)), POSE_SOURCE_GEOMETRIC, None, reason)
