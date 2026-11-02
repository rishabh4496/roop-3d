"""Actual HifiFace-256 inference with its paired ArcFace converter.

The checkpoint learned 3D shape supervision during training. Its exported
runtime inputs are target RGB and identity embedding, not FLAME beta/psi.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roop_3d.flame_3d_engine import rgb_tensor


@dataclass(frozen=True)
class HifiFaceOutput:
    rgb: np.ndarray
    mask: np.ndarray


class HifiFace256:
    def __init__(self, swap_session, converter_session):
        self.swapper, self.converter = swap_session, converter_session

    @classmethod
    def from_registry(cls, *, provider="CPUExecutionProvider", profile="6gb", catalog=None):
        from roop_3d.models.catalog import ModelCatalog
        from roop_3d.models.stage16 import CheckedSession, resolve_artifact
        catalog = catalog or ModelCatalog()
        sessions = []
        for model_id in ("hififace_unofficial_256", "arcface_converter_hififace"):
            record = catalog.registry.get(model_id)
            path = resolve_artifact(model_id, catalog)
            sessions.append(CheckedSession(path, record.expected_sha256, record.byte_size,
                                           provider=provider, profile=profile, batch_dimensions=("batch_size",)))
        return cls(*sessions)

    def convert_identity(self, arcface_embedding):
        source = np.asarray(arcface_embedding, dtype=np.float32).reshape(1, -1)
        if source.shape != (1, 512) or not np.isfinite(source).all() or np.linalg.norm(source) < 1e-8:
            raise ValueError("Supply a finite, nonzero 512-component raw ArcFace embedding")
        inputs = self.converter.session.get_inputs()
        outputs = self.converter.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("ArcFace converter graph differs from the reviewed single-vector contract")
        converted = self.converter.run({inputs[0].name: np.ascontiguousarray(source)})[outputs[0].name]
        if converted.shape != (1, 512) or np.linalg.norm(converted) < 1e-8:
            raise RuntimeError("HifiFace converter emitted an invalid identity")
        # FaceFusion 3.1 converts raw ArcFace then L2 normalizes, in that order.
        return np.ascontiguousarray(converted / np.linalg.norm(converted), dtype=np.float32)

    def swap(self, aligned_rgb256, arcface_embedding) -> HifiFaceOutput:
        if np.asarray(aligned_rgb256).shape != (256, 256, 3):
            raise ValueError("HifiFace requires an aligned 256x256 RGB crop")
        source = self.convert_identity(arcface_embedding)
        output = self.swapper.run({"target": rgb_tensor(aligned_rgb256, 256) * 2 - 1, "source": source})
        image, mask = output["output"], output["mask"]
        if image.shape != (1, 3, 256, 256) or mask.shape != (1, 1, 256, 256):
            raise RuntimeError("HifiFace image/mask graph contract changed")
        rgb = np.rint(np.clip((image[0].transpose(1, 2, 0) + 1) * 127.5, 0, 255)).astype(np.uint8)
        return HifiFaceOutput(rgb, np.clip(mask[0, 0], 0, 1))
