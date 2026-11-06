"""Explicit optional Stage 16 crop pipeline; default reference inference is intact."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Protocol

import numpy as np
from pydantic import Field

from roop_3d.core.settings import Contract
from roop_3d.flame_3d_engine import EyeGaze, Flame3DEngine, FlameParameters, blend_shape
from roop_3d.temporal import TrackTemporalStabilizer


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
        self.temporal = TrackTemporalStabilizer(min_cutoff=.2, beta=.04, derivative_cutoff=1.,
                                                process_noise=5., measurement_noise=1.5)
        self.frames_processed = 0

    def reset_track(self, track_id: int) -> None:
        self.temporal.forget_track(track_id)

    def process(self, source_rgb256, target_rgb256, source_embedding, *, options=Stage16Options(),
                source_context384=None, target_context384=None, track_id: int = 0,
                timestamp_seconds: float | None = None, scene_cut: bool = False) -> Stage16Result:
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
        timestamp = float(timestamp_seconds if timestamp_seconds is not None else self.frames_processed / 30.0)
        self.frames_processed += 1
        if scene_cut:
            self.reset_track(track_id)
        from roop_3d.reconstruction.geometry_types import rotation_matrix_to_rodrigues, rodrigues_to_rotation_matrix
        target_flame = target.flame
        filtered = replace(target_flame,
            beta=self.temporal.filter(track_id, '3dmm_shape', target_flame.beta, timestamp),
            psi=self.temporal.filter(track_id, '3dmm_expression', target_flame.psi, timestamp),
            jaw_rotation=self.temporal.filter(track_id, '3dmm_jaw', target_flame.jaw_rotation, timestamp),
            rotation=rodrigues_to_rotation_matrix(self.temporal.filter(
                track_id, '3dmm_head_pose', rotation_matrix_to_rodrigues(target_flame.rotation), timestamp)))
        if target_flame.gaze is not None:
            gaze_angles = self.temporal.filter(track_id, '3dmm_eye_gaze', target_flame.gaze.angles(), timestamp)
            filtered = replace(filtered, gaze=EyeGaze(*map(float, gaze_angles)))
        target = replace(target, flame=filtered)
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
