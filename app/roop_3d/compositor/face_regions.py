"""Named face regions from the local BiSeNet 19-class parser.

What the model can and cannot separate (verified on real frames, see ``docs/OCCLUSION_STAGE5.md``):

* separated: skin, brows, eyes, nose, mouth interior, lips, ears, hair, neck, clothing, glasses, hat;
* NOT separated: **teeth** and **tongue** (both are inside the mouth-interior class; no tongue class or
  model exists) and **foreground objects**
  (hands, microphones, props are not parser classes at all; they come from the visible-face
  networks in ``occlusion_model``).

``teeth_estimate`` is therefore a labelled heuristic, never a parser output, and is only used to
protect pixels from restoration, not to decide what is painted.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from roop_3d.core.errors import DomainError

# CelebAMask-HQ 19-class convention (candidate mapping, checked on real frames by the Stage 5 tests).
REGION_CLASSES: dict[str, tuple[int, ...]] = {
    "skin": (1,),
    "brows": (2, 3),
    "eyes": (4, 5),
    "glasses": (6,),
    "ears": (7, 8, 9),
    "nose": (10,),
    "mouth_interior": (11,),
    "lips": (12, 13),
    "neck": (14, 15),
    "cloth": (16,),
    "hair": (17,),
    "hat": (18,),
}
FACE_INTERIOR_REGIONS = ("skin", "brows", "eyes", "nose", "mouth_interior", "lips")
# Parser classes that are physically in front of, or beside, the face and must be preserved (strict hair/neck/ear exclusion).
FOREGROUND_REGIONS = ("glasses", "hat", "hair", "cloth", "neck", "ears")
STRICT_FORBIDDEN_REGIONS = ("neck", "cloth", "hair", "hat", "ears", "glasses")


@dataclass(frozen=True)
class FaceRegions:
    """Soft (0..1) per-region maps at the resolution of the analysed crop."""

    maps: dict[str, np.ndarray]
    shape: tuple[int, int]

    def __getitem__(self, name: str) -> np.ndarray:
        return self.maps[name]

    def get(self, name: str) -> np.ndarray:
        return self.maps.get(name, np.zeros(self.shape, np.float32))

    @property
    def face_interior(self) -> np.ndarray:
        total = np.zeros(self.shape, np.float32)
        for name in FACE_INTERIOR_REGIONS:
            total += self.get(name)
        return np.clip(total, 0.0, 1.0)

    @property
    def foreground(self) -> np.ndarray:
        total = np.zeros(self.shape, np.float32)
        for name in FOREGROUND_REGIONS:
            total = 1.0 - (1.0 - total) * (1.0 - self.get(name))
        return np.clip(total, 0.0, 1.0)


def build_regions(probabilities: np.ndarray) -> FaceRegions:
    """Sum class probabilities into named regions; raises on an unusable tensor."""
    scores = np.asarray(probabilities, dtype=np.float32)
    needed = max(max(ids) for ids in REGION_CLASSES.values()) + 1
    if scores.ndim != 3 or scores.shape[2] < needed or not np.isfinite(scores).all():
        raise DomainError("REGIONS_PARSER_INVALID", "Parser probabilities must be a finite HxWx19 tensor", 422)
    maps = {name: np.clip(scores[..., list(ids)].sum(axis=-1), 0.0, 1.0).astype(np.float32)
            for name, ids in REGION_CLASSES.items()}
    return FaceRegions(maps, (scores.shape[0], scores.shape[1]))


def teeth_estimate(image_bgr: np.ndarray, mouth_interior: np.ndarray) -> np.ndarray:
    """HEURISTIC teeth map: bright, low-saturation pixels inside the parser's mouth interior.

    Not a model output. Returns zeros when the mouth is closed or no such pixels exist.
    """
    image = np.asarray(image_bgr)
    mouth = np.clip(np.asarray(mouth_interior, dtype=np.float32), 0.0, 1.0)
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[:2] != mouth.shape:
        raise DomainError("REGIONS_INPUT_INVALID", "Image and mouth map must match", 422)
    inside = mouth > 0.5
    if int(inside.sum()) < 12:
        return np.zeros(mouth.shape, np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value, saturation = hsv[..., 2].astype(np.float32), hsv[..., 1].astype(np.float32)
    floor = max(float(np.percentile(value[inside], 55)), 90.0)
    teeth = inside & (value >= floor) & (saturation <= 90.0)
    result = cv2.GaussianBlur(teeth.astype(np.float32), (5, 5), 1.0)
    return np.clip(result * mouth, 0.0, 1.0).astype(np.float32)


def restoration_protection(regions: FaceRegions | None, image_bgr: np.ndarray | None = None, *,
                           grow_px: int = 3) -> np.ndarray | None:
    """Map (1 = do not let a restorer rewrite this pixel): eyes, brows, lips, mouth interior, teeth."""
    if regions is None:
        return None
    protect = np.zeros(regions.shape, np.float32)
    for name in ("eyes", "brows", "mouth_interior", "lips"):
        protect = np.maximum(protect, regions.get(name))
    if image_bgr is not None:
        protect = np.maximum(protect, teeth_estimate(image_bgr, regions.get("mouth_interior")))
    if grow_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow_px + 1, 2 * grow_px + 1))
        protect = cv2.GaussianBlur(cv2.dilate(protect, kernel), (2 * grow_px + 1, 2 * grow_px + 1), grow_px / 2.0)
    return np.clip(protect, 0.0, 1.0).astype(np.float32)
