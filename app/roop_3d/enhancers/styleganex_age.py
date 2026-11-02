"""StyleGANEX age inference; latent direction is not calibrated chronological age."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roop_3d.flame_3d_engine import rgb_tensor


@dataclass(frozen=True)
class AgeCalibration:
    """Measured (years, latent direction) pairs, with explicit provenance.

    Upstream FaceFusion's -100..100 direction slider is not a years scale.
    No default conversion between that slider and years is invented here.
    """
    years: tuple[float, ...]
    directions: tuple[float, ...]
    evidence: str

    def __post_init__(self):
        if (len(self.years) < 3 or len(self.years) != len(self.directions) or not self.evidence.strip()
                or not np.isfinite(self.years).all() or not np.isfinite(self.directions).all()
                or any(b <= a for a, b in zip(self.years, self.years[1:]))
                or self.years[0] > -30 or self.years[-1] < 40
                or 0 not in self.years or self.directions[self.years.index(0)] != 0):
            raise ValueError("A measured calibration must cover -30..+40 years and anchor zero")

    def direction(self, offset):
        if not np.isfinite(offset) or not -30 <= offset <= 40:
            raise ValueError("age_offset must be in [-30, 40] years")
        return float(np.interp(offset, self.years, self.directions))


class StyleGANEXAge:
    def __init__(self, session, calibration: AgeCalibration | None = None):
        self.session, self.calibration = session, calibration

    def transform(self, aligned_rgb256, context_rgb384, *, age_offset: float):
        if self.calibration is None:
            raise RuntimeError("Year-based age_offset requires measured StyleGANEX calibration")
        if np.asarray(aligned_rgb256).shape != (256, 256, 3) or np.asarray(context_rgb384).shape != (384, 384, 3):
            raise ValueError("Supply separately aligned 256 RGB face and 384 RGB background crops")
        direction = self.calibration.direction(age_offset)
        inputs = self.session.session.get_inputs()
        scalar = next((v for v in inputs if v.name == "direction"), None)
        if scalar is None or any(not isinstance(s, int) or s < 1 for s in scalar.shape):
            raise ValueError("StyleGANEX direction must have an inspected concrete scalar shape")
        shape = tuple(scalar.shape)
        if int(np.prod(shape)) != 1:
            raise ValueError("StyleGANEX direction must be scalar")
        feeds = {"target": rgb_tensor(aligned_rgb256, 256) * 2 - 1,
                 "target_with_background": rgb_tensor(context_rgb384, 384) * 2 - 1,
                 "direction": np.full(shape, direction, np.float32)}
        outputs = self.session.session.get_outputs()
        if len(outputs) != 1:
            raise ValueError("StyleGANEX output contract requires review")
        output = self.session.run(feeds)[outputs[0].name]
        if output.ndim != 4 or output.shape[0:2] != (1, 3):
            raise RuntimeError("StyleGANEX output must be a single RGB tensor")
        return np.rint(np.clip((output[0].transpose(1, 2, 0) + 1) * 127.5, 0, 255)).astype(np.uint8)

    def preflight_identity(self, face, context, *, age_offset, recognizer):
        """Age first, then re-extract identity; stale embeddings cannot be reused."""
        result = self.transform(face, context, age_offset=age_offset)
        embedding = np.asarray(recognizer(result), np.float32)
        if embedding.shape != (512,) or not np.isfinite(embedding).all() or np.linalg.norm(embedding) < 1e-8:
            raise RuntimeError("Preflight age modifier requires a freshly extracted ArcFace identity")
        return result, embedding
