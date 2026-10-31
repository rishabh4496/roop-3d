"""Frame-indexed parameter curves and explicit per-track role intervals."""
from __future__ import annotations

from typing import Literal
from pydantic import Field, model_validator
from roop_3d.core.settings import Contract

CurveName = Literal['swap_weight', 'restorer_opacity', 'mask_feather', 'mask_erosion', 'pose_correction_strength']
BOUNDS = {'swap_weight': (0, 1), 'restorer_opacity': (0, 1), 'mask_feather': (0, 64),
          'mask_erosion': (-32, 32), 'pose_correction_strength': (0, 1)}
FIELDS = {'swap_weight': 'swap_weight', 'restorer_opacity': 'restoration_blend',
          'mask_feather': 'feather_px', 'mask_erosion': 'erosion_px', 'pose_correction_strength': 'pose_strength'}


class CurveKey(Contract):
    frame: int = Field(ge=0, le=100_000_000, strict=True)
    value: float
    easing: Literal['linear', 'ease-in-out', 'hold'] = 'linear'


class ParameterCurves(Contract):
    tracks: dict[CurveName, tuple[CurveKey, ...]] = Field(default_factory=dict)

    @model_validator(mode='after')
    def bounds(self):
        for name, keys in self.tracks.items():
            if len(keys) > 500:
                raise ValueError('At most 500 keys per parameter')
            if [key.frame for key in keys] != sorted({key.frame for key in keys}):
                raise ValueError('Curve frames must be unique and sorted')
            lo, hi = BOUNDS[name]
            if any(not lo <= key.value <= hi for key in keys):
                raise ValueError('Curve value exceeds parameter bounds')
        return self

    def evaluate(self, frame: int, parameters):
        values = parameters.model_dump()
        for name, keys in self.tracks.items():
            if keys:
                values[FIELDS[name]] = evaluate_curve(keys, frame)
        return type(parameters).model_validate(values)


def evaluate_curve(keys: tuple[CurveKey, ...], frame: int) -> float:
    if frame <= keys[0].frame:
        return keys[0].value
    for left, right in zip(keys, keys[1:]):
        if frame >= right.frame:
            continue
        if left.easing == 'hold':
            return left.value
        t = (frame - left.frame) / (right.frame - left.frame)
        if left.easing == 'ease-in-out':
            t = t * t * (3 - 2 * t)
        return left.value + (right.value - left.value) * t
    return keys[-1].value


class FaceSegment(Contract):
    cluster_id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,100}$')
    track_id: str = Field(pattern=r'^track_[0-9]{4}$')
    start_frame: int = Field(ge=0, le=100_000_000, strict=True)
    end_frame: int | None = Field(default=None, ge=0, le=100_000_000, strict=True)
    role: Literal['swap', 'reference', 'exclude'] = 'swap'
    faceset_id: str | None = Field(default=None, pattern=r'^[A-Za-z0-9_-]{1,64}$')
    source_view_index: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode='after')
    def interval(self):
        if self.end_frame is not None and self.end_frame < self.start_frame:
            raise ValueError('Segment ends before it starts')
        if self.role == 'swap' and not self.faceset_id:
            raise ValueError('Swap segments require an explicit source')
        if self.source_view_index is not None and not self.faceset_id:
            raise ValueError('A source face requires a FaceSet')
        return self

    def contains(self, frame: int) -> bool:
        return frame >= self.start_frame and (self.end_frame is None or frame <= self.end_frame)
