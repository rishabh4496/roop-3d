"""SCRFD 3-level/2-anchor decoder; ONNX data loaded only on inspection.

Tensor layout/preprocessing: official InsightFace model_zoo/scrfd.py.
This adapter imports no legacy application code and always uses the CPU provider.
"""
from __future__ import annotations

import threading
from typing import Any
import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.models.catalog import ModelCatalog
from roop_3d.engine.resources import guarded, register


class Detector:
    def __init__(self, catalog: ModelCatalog, session_factory=None):
        self.catalog = catalog
        self._lock = threading.Lock()
        self._session: Any = None
        self._io_runner: Any = None
        self._session_sha: str | None = None
        self._session_factory = session_factory
        register(self, "release_for_build")

    @guarded
    def detect(self, image: np.ndarray, threshold: float) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
        scored, record = self._detect(image, threshold)
        return [(box, points) for box, points, _ in scored], record

    @guarded
    def detect_scored(self, image: np.ndarray, threshold: float) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], dict]:
        """`detect`, plus each face's detector confidence (the SCRFD score the face was kept with)."""
        return self._detect(image, threshold)

    def _detect(self, image: np.ndarray, threshold: float) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], dict]:
        record = self.catalog.detector()
        # Rehash cached model inputs; never continue inference after artifact drift.
        with self._lock:
            if self._session is None or self._session_sha != record["sha256"]:
                if self._session_factory is not None:
                    self._session = self._session_factory(record)
                else:
                    data = self.catalog.verified_bytes(record)
                    import onnxruntime as ort
                    options = ort.SessionOptions()
                    options.intra_op_num_threads = 2
                    self._session = ort.InferenceSession(data, sess_options=options, providers=["CPUExecutionProvider"])
                if hasattr(self._session, "io_binding"):
                    from roop_3d.core.ort_session import GpuIoBinding
                    self._io_runner = GpuIoBinding(self._session)
                self._session_sha = record["sha256"]
            session, io_runner = self._session, self._io_runner
        height, width = image.shape[:2]
        shape = session.get_inputs()[0].shape
        size = int(shape[2]) if isinstance(shape[2], int) else 640
        if size != 640 or (isinstance(shape[3], int) and shape[3] != 640):
            raise DomainError("DETECTOR_LAYOUT_UNSUPPORTED", "SCRFD adapter requires a 640 square input")
        scale = min(size / width, size / height)
        scaled_w, scaled_h = max(1, int(width * scale)), max(1, int(height * scale))
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:scaled_h, :scaled_w] = cv2.resize(image, (scaled_w, scaled_h))
        blob = cv2.dnn.blobFromImage(canvas, 1 / 128.0, (size, size), (127.5, 127.5, 127.5), swapRB=True)
        feeds = {session.get_inputs()[0].name: blob}
        if io_runner is not None:
            names = tuple(node.name for node in session.get_outputs())
            bound = io_runner.run(feeds, output_names=names)
            outputs = [bound[name].numpy() for name in names]
        else:
            # SessionFacade routes through OrtModel's persistent I/O binding;
            # this branch remains for lightweight test doubles.
            outputs = session.run(None, feeds)
        if len(outputs) != 9:
            raise DomainError("DETECTOR_LAYOUT_UNSUPPORTED", "SCRFD output count is unsupported")
        boxes, points, scores = [], [], []
        for level, stride in enumerate((8, 16, 32)):
            count = (size // stride) ** 2 * 2
            confidence = outputs[level].reshape(-1)
            distances = outputs[level + 3].reshape(-1, 4) * stride
            landmarks = outputs[level + 6].reshape(-1, 5, 2) * stride
            if len(confidence) != count or len(distances) != count or len(landmarks) != count:
                raise DomainError("DETECTOR_LAYOUT_UNSUPPORTED", "SCRFD output dimensions do not match the input")
            grid = np.stack(np.mgrid[:size // stride, :size // stride][::-1], axis=-1).reshape(-1, 2)
            centers = np.repeat(grid * stride, 2, axis=0)
            keep = np.flatnonzero(confidence >= threshold)
            bounds = np.column_stack((centers[:, 0] - distances[:, 0], centers[:, 1] - distances[:, 1],
                                      centers[:, 0] + distances[:, 2], centers[:, 1] + distances[:, 3]))
            boxes.extend(bounds[keep] / scale)
            points.extend((landmarks[keep] + centers[keep, None, :]) / scale)
            scores.extend(confidence[keep])
        retained: list[tuple[np.ndarray, np.ndarray, float]] = []
        order = list(np.argsort(scores)[::-1])
        while order and len(retained) < 32:
            index = order.pop(0)
            box = np.asarray(boxes[index]).clip([0, 0, 0, 0], [width, height, width, height])
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            retained.append((box, points[index], float(scores[index])))
            order = [other for other in order if iou(box, boxes[other]) < 0.4]
        return retained, record

    @guarded
    def release_for_build(self) -> None:
        with self._lock:
            self._session = None
            self._io_runner = None
            self._session_sha = None


def iou(left: np.ndarray, right: np.ndarray) -> float:
    x1, y1 = np.maximum(left[:2], right[:2])
    x2, y2 = np.minimum(left[2:], right[2:])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = np.prod(left[2:] - left[:2]) + np.prod(right[2:] - right[:2]) - intersection
    return float(intersection / union) if union > 0 else 0.0
