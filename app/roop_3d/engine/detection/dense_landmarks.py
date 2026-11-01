"""Read-only 106/203-point ONNX adapters and explicitly approximate pose geometry."""
from __future__ import annotations

import math
import cv2
import numpy as np

from roop_3d.engine.runtime import OrtModel, RuntimeConfig, image_bgr, tensor
from roop_3d.core.errors import DomainError
from roop_3d.core.pose import legacy_pose_5pt
from roop_3d.engine.volumetric.preprocess import landmark_crop, transform_points
from roop_3d.models.catalog import ModelCatalog


class DenseLandmarks:
    def __init__(self, catalog: ModelCatalog, config: RuntimeConfig = RuntimeConfig()):
        self.catalog, self.config = catalog, config
        self.model106: OrtModel | None = None
        self.model203: OrtModel | None = None

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

    def release(self) -> None:
        for model in (self.model106, self.model203):
            if model is not None:
                model.release()
        self.model106 = self.model203 = None


def geometric_pose(points5: np.ndarray) -> tuple[float, float, float]:
    """Yaw/pitch/roll estimate in the legacy frame (pitch + = down); not calibrated 3D reconstruction.

    Thin wrapper over the single shared estimator `core.pose.estimate_pose_5pt`."""
    return legacy_pose_5pt(points5)
