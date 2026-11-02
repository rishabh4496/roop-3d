"""Explicit optional Stage 16 crop pipeline; default reference inference is intact."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from pydantic import Field

from roop_3d.core.settings import Contract
from roop_3d.flame_3d_engine import Flame3DEngine, FlameParameters, blend_shape


class Stage16Options(Contract):
    shape_retention_weight: float = Field(default=0, ge=0, le=1)
    age_offset: float = Field(default=0, ge=-30, le=40)
    age_position: Literal["preflight", "post_swap"] = "post_swap"
    expression_strength: float = Field(default=1, ge=0, le=1)
    require_independent_gaze: bool = False


class ShapeConditioner(Protocol):
    """A licensed FLAME renderer/projector must implement actual crop geometry."""
    def condition(self, rgb: np.ndarray, source: FlameParameters, blended: FlameParameters) -> np.ndarray: ...


@dataclass(frozen=True)
class Stage16Result:
    rgb: np.ndarray
    flame: FlameParameters
    mask256: np.ndarray
    gaze_verified: bool = False  # Only output-image acceptance can establish accuracy.


class Stage16Pipeline:
    def __init__(self, flame: Flame3DEngine, swapper, portrait, *, shape: ShapeConditioner | None = None,
                 age=None, recognizer=None):
        self.flame, self.swapper, self.portrait = flame, swapper, portrait
        self.shape, self.age, self.recognizer = shape, age, recognizer

    def process(self, source_rgb256, target_rgb256, source_embedding, *, options=Stage16Options(),
                source_context384=None, target_context384=None) -> Stage16Result:
        if options.shape_retention_weight and self.shape is None:
            raise RuntimeError("3D skull preservation requires an activated FLAME geometry conditioner")
        if options.age_offset and self.age is None:
            raise RuntimeError("Age modification requires activated StyleGANEX weights and year calibration")
        if options.age_offset and options.age_position == "preflight":
            if self.recognizer is None or source_context384 is None:
                raise RuntimeError("Preflight age modification requires aligned source context and a recognizer")
            import cv2
            aged, source_embedding = self.age.preflight_identity(
                source_rgb256, source_context384, age_offset=options.age_offset, recognizer=self.recognizer)
            source_rgb256 = cv2.resize(aged, (256, 256))
        source = self.flame.extract(source_rgb256, require_gaze=options.require_independent_gaze)
        target = self.flame.extract(target_rgb256, require_gaze=options.require_independent_gaze)
        blended = blend_shape(source.flame, target.flame, options.shape_retention_weight)
        swapped = self.swapper.swap(target_rgb256, source_embedding)
        image = swapped.rgb
        if self.shape is not None:
            image = self.shape.condition(image, source.flame, blended)
        gaze = {}
        if options.require_independent_gaze:
            # The swapped image can have different eye orientation from the source.
            swapped_head = self.flame.extract(image, require_gaze=True)
            gaze = {"source_gaze": swapped_head.flame.gaze, "driving_gaze": target.flame.gaze}
        image = self.portrait.retarget(image, target_rgb256, strength=options.expression_strength, **gaze)
        if options.age_offset and options.age_position == "post_swap":
            import cv2
            if target_context384 is None:
                raise RuntimeError("Post-swap age modification requires separately aligned target context")
            image = self.age.transform(cv2.resize(image, (256, 256)), target_context384, age_offset=options.age_offset)
        return Stage16Result(image, blended, swapped.mask)
