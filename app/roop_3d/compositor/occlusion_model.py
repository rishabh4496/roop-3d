"""Local "visible face" networks: XSeg and the FaceFusion face occluder.

Both are 256x256, NHWC, float32 networks that output a single-channel map where **1 means visible
face and 0 means "not visible face"** (background, hands, microphones, hair strands and other
foreground objects). They are lazy, hash-checked through the Stage 03 model catalog and never
downloaded here. A missing model raises a ``DomainError`` that the analyser turns into a reported
capability state; nothing is simulated.

Evidence recorded by ``capability()`` follows the project convention: weights, provenance,
license and inference are separate states. Inference becomes ``VERIFIED_LOCAL`` only after this
process has actually executed the graph.
"""
from __future__ import annotations

import cv2
import numpy as np

from roop_3d.core.errors import DomainError

NET_SIZE = 256
SUPPORTED_NETS = ("xseg_3.onnx", "xseg.onnx", "face_occluder.onnx")


class VisibleFaceNet:
    def __init__(self, catalog=None, config=None, *, model_name: str = "xseg_3.onnx", channel_order: str = "bgr"):
        from roop_3d.engine.runtime import RuntimeConfig
        from roop_3d.models.catalog import ModelCatalog

        if model_name not in SUPPORTED_NETS:
            raise ValueError(f"Unsupported visible-face network: {model_name}")
        if channel_order not in {"bgr", "rgb"}:
            raise ValueError("channel_order must be 'bgr' or 'rgb'")
        self.catalog = catalog or ModelCatalog()
        self.config = config or RuntimeConfig()
        self.model_name, self.channel_order = model_name, channel_order
        self.model = None
        self.inference_verified = False
        self.last_error: str | None = None

    def _load(self):
        if self.model is None:
            from roop_3d.engine.runtime import OrtModel
            self.model = OrtModel(self.catalog, self.model_name, self.config)
        return self.model

    def available(self) -> bool:
        """True when the weights can be loaded; never raises."""
        try:
            self._load()
            return True
        except (DomainError, RuntimeError, OSError, ValueError, KeyError) as error:
            self.last_error = f"{type(error).__name__}: {error}"
            return False

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return an HxW float32 visible-face map (1 = visible face) at the input resolution."""
        from roop_3d.engine.runtime import image_bgr as validate_image

        validate_image(image_bgr)
        height, width = image_bgr.shape[:2]
        source = cv2.resize(image_bgr, (NET_SIZE, NET_SIZE),
                            interpolation=cv2.INTER_AREA if max(height, width) > NET_SIZE else cv2.INTER_CUBIC)
        if self.channel_order == "rgb":
            source = source[:, :, ::-1]
        tensor = np.ascontiguousarray(source.astype(np.float32)[None] / 255.0)
        model = self._load()
        outputs = model.run({model.session.get_inputs()[0].name: tensor})
        for value in outputs.values():
            array = np.asarray(value, dtype=np.float32)
            if array.ndim == 4 and array.shape[0] == 1 and array.shape[1:3] == (NET_SIZE, NET_SIZE) and array.shape[3] == 1:
                if not np.isfinite(array).all():
                    raise DomainError("VISIBLE_FACE_OUTPUT_INVALID", "Network returned non-finite values", 503)
                self.inference_verified = True
                visible = np.clip(array[0, ..., 0], 0.0, 1.0)
                return cv2.resize(visible, (width, height), interpolation=cv2.INTER_LINEAR)
        raise DomainError("VISIBLE_FACE_OUTPUT_UNSUPPORTED", "Network returned no 1x256x256x1 tensor", 503)

    def capability(self) -> dict[str, str]:
        loaded = self.model is not None
        state = {"model": self.model_name,
                 "weights": "VERIFIED_LOCAL" if loaded else ("UNAVAILABLE" if self.last_error else "NOT_LOADED"),
                 "provenance": "UNKNOWN", "license": "UNKNOWN",
                 "inference": "VERIFIED_LOCAL" if self.inference_verified else "UNKNOWN",
                 "input_contract": f"256x256 NHWC float32 {self.channel_order.upper()} /255 (output 1 = visible face)"}
        if self.last_error:
            state["error"] = self.last_error
        return state

    def release(self) -> None:
        if self.model is not None:
            self.model.release()
        self.model = None
