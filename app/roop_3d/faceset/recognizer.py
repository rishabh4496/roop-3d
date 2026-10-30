"""Face recognition adapter for ArcFace/InsightFace models using local ONNX weights.

All models run with CPUExecutionProvider in read-only mode without downloading
or copying weights. Input faces are aligned using standard 5-point reference
geometry to 112x112, normalized to [-1, 1], and inferred to 512-D L2-normalized embeddings.
"""
from __future__ import annotations

import threading
from typing import Any
import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.models.catalog import ModelCatalog

ARCFACE_REFERENCE_5PTS = np.array([
    [38.2946, 51.6963],  # left eye
    [73.5318, 51.5014],  # right eye
    [56.0252, 71.7366],  # nose tip
    [41.5493, 92.3655],  # left mouth corner
    [70.7299, 92.2041],  # right mouth corner
], dtype=np.float32)


class Recognizer:
    def __init__(self, catalog: ModelCatalog):
        self.catalog = catalog
        self._lock = threading.Lock()
        self._session: Any = None
        self._session_sha: str | None = None
        self._input_name: str | None = None
        self._output_name: str | None = None

    def embed(self, image: np.ndarray, landmarks: np.ndarray | None = None) -> np.ndarray:
        record = self.catalog.recognizer()
        with self._lock:
            data = self.catalog.verified_bytes(record)
            if self._session is None or self._session_sha != record["sha256"]:
                import onnxruntime as ort
                options = ort.SessionOptions()
                options.intra_op_num_threads = 2
                self._session = ort.InferenceSession(data, sess_options=options, providers=["CPUExecutionProvider"])
                self._session_sha = record["sha256"]
                self._input_name = self._session.get_inputs()[0].name
                self._output_name = self._session.get_outputs()[0].name
            session = self._session
            input_name = self._input_name
            output_name = self._output_name

        if image is None or image.size == 0:
            raise DomainError("RECOGNIZER_INPUT_INVALID", "Cannot embed empty image", 400)

        # Align face to 112x112
        aligned = self.align_face(image, landmarks)

        # Preprocess to (1, 3, 112, 112) normalized blob
        blob = cv2.dnn.blobFromImage(
            aligned,
            1.0 / 127.5,
            (112, 112),
            (127.5, 127.5, 127.5),
            swapRB=True,
        )

        outputs = session.run([output_name], {input_name: blob})
        embedding = outputs[0].reshape(-1).astype(np.float32)
        if len(embedding) != 512:
            raise DomainError("RECOGNIZER_OUTPUT_INVALID", f"Expected 512-D embedding, got {len(embedding)}", 500)

        norm = float(np.linalg.norm(embedding))
        if norm > 1e-9:
            embedding /= norm
        return embedding

    @staticmethod
    def align_face(image: np.ndarray, landmarks: np.ndarray | None) -> np.ndarray:
        if landmarks is not None and len(landmarks) == 5:
            pts = np.asarray(landmarks, dtype=np.float32)
            matrix, _ = cv2.estimateAffinePartial2D(pts, ARCFACE_REFERENCE_5PTS, method=cv2.LMEDS)
            if matrix is not None:
                return cv2.warpAffine(image, matrix, (112, 112), borderValue=0.0)
        return cv2.resize(image, (112, 112), interpolation=cv2.INTER_AREA)
