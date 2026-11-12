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
from roop_3d.engine.resources import guarded, inference, register

ARCFACE_REFERENCE_5PTS = np.array([
    [38.2946, 51.6963],  # left eye
    [73.5318, 51.5014],  # right eye
    [56.0252, 71.7366],  # nose tip
    [41.5493, 92.3655],  # left mouth corner
    [70.7299, 92.2041],  # right mouth corner
], dtype=np.float32)


class Recognizer:
    def __init__(self, catalog: ModelCatalog, session_factory=None):
        self.catalog = catalog
        self._lock = threading.Lock()
        self._session: Any = None
        self._session_sha: str | None = None
        self._input_name: str | None = None
        self._output_name: str | None = None
        self._io_runner: Any = None
        self._session_factory = session_factory
        register(self, "release_for_build")

    @inference
    def embed(self, image: np.ndarray, landmarks: np.ndarray | None = None) -> np.ndarray:
        record = self.catalog.recognizer()
        with self._lock:
            if self._session is None or self._session_sha != record["sha256"]:
                if self._session_factory is not None:
                    from roop_3d.engine.runtime import build_with_cpu_fallback
                    self._session = build_with_cpu_fallback(self._session_factory, record, "recognizer")
                else:
                    self._session = None
                if self._session is None:
                    data = self.catalog.verified_bytes(record)
                    import onnxruntime as ort
                    options = ort.SessionOptions()
                    options.intra_op_num_threads = 2
                    from roop_3d.engine.resources import build_fence
                    with build_fence():
                        self._session = ort.InferenceSession(data, sess_options=options, providers=["CPUExecutionProvider"])
                self._session_sha = record["sha256"]
                self._input_name = self._session.get_inputs()[0].name
                self._output_name = self._session.get_outputs()[0].name
                self._io_runner = None
                if hasattr(self._session, "io_binding"):
                    from roop_3d.core.ort_session import GpuIoBinding
                    self._io_runner = GpuIoBinding(self._session)
            session = self._session
            io_runner = self._io_runner
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

        if io_runner is not None:
            value = io_runner.run({input_name: blob}, output_names=(output_name,))[output_name].numpy()
        else:
            # SessionFacade keeps its underlying OrtModel I/O binding.
            value = session.run([output_name], {input_name: blob})[0]
        embedding = value.reshape(-1).astype(np.float32)
        if len(embedding) != 512:
            raise DomainError("RECOGNIZER_OUTPUT_INVALID", f"Expected 512-D embedding, got {len(embedding)}", 500)

        norm = float(np.linalg.norm(embedding))
        if norm > 1e-9:
            embedding /= norm
        return embedding

    @guarded
    def release_for_build(self) -> None:
        with self._lock:
            self._session = None
            self._io_runner = None
            self._session_sha = None
            self._input_name = self._output_name = None

    @staticmethod
    def align_face(image: np.ndarray, landmarks: np.ndarray | None) -> np.ndarray:
        if landmarks is not None and len(landmarks) == 5:
            pts = np.asarray(landmarks, dtype=np.float32)
            matrix, _ = cv2.estimateAffinePartial2D(pts, ARCFACE_REFERENCE_5PTS, method=cv2.LMEDS)
            if matrix is not None:
                return cv2.warpAffine(image, matrix, (112, 112), borderValue=0.0)
        return cv2.resize(image, (112, 112), interpolation=cv2.INTER_AREA)
