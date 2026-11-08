"""Frame-indexed parameter curves and explicit per-track role intervals."""
from __future__ import annotations

from typing import Literal
from functools import lru_cache
from pydantic import Field, model_validator, PrivateAttr
from roop_3d.core.settings import Contract

CurveName = Literal['swap_weight', 'restorer_opacity', 'mask_feather', 'mask_erosion', 'pose_correction_strength', 'tps_expression_weight', 'pose_frontalization', 'swap_opacity', 'mask_feather_radius', 'fidelity_weight', 'identity_blend_t', 'relight_strength', 'occlusion_threshold']
BOUNDS = {'relight_strength': (0, 1), 'occlusion_threshold': (0, 1), 'fidelity_weight': (0,1), 'identity_blend_t': (0,1), 'swap_weight': (0, 1), 'restorer_opacity': (0, 1), 'mask_feather': (0, 64),
          'mask_erosion': (-32, 32), 'pose_correction_strength': (0, 1), 'tps_expression_weight': (0,1), 'pose_frontalization': (0,1), 'swap_opacity': (0,1), 'mask_feather_radius': (0,64)}
FIELDS = {'relight_strength': 'relight_strength', 'occlusion_threshold': 'occlusion_threshold', 'fidelity_weight': 'fidelity_weight', 'identity_blend_t': 'identity_blend_t', 'swap_weight': 'swap_weight', 'restorer_opacity': 'restoration_blend',
          'mask_feather': 'feather_px', 'mask_erosion': 'erosion_px', 'pose_correction_strength': 'pose_strength', 'tps_expression_weight': 'tps_expression_weight', 'pose_frontalization': 'pose_frontalization', 'swap_opacity': 'swap_weight', 'mask_feather_radius': 'feather_px'}


class CurveKey(Contract):
    frame: int = Field(ge=0, le=100_000_000, strict=True)
    value: float
    control1: float = Field(default=0, ge=0, le=1)
    control2: float = Field(default=1, ge=0, le=1)
    easing: Literal['linear', 'ease-in-out', 'hold', 'bezier', 'step'] = 'linear'


class ParameterCurves(Contract):
    _compiled: dict = PrivateAttr(default_factory=dict)
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
                if name not in self._compiled:
                    self._compiled[name] = _compile(tuple((k.frame,k.value,k.easing,k.control1,k.control2) for k in keys))
                values[FIELDS[name]] = evaluate_curve(keys, frame, self._compiled[name])
        return type(parameters).model_validate(values)


@lru_cache(maxsize=128)
def _compile(signature):
    # Fixed-depth radix predecessor index for bounded 32-bit frame indices.
    root = {}
    for index, key in enumerate(signature):
        node = root
        for shift in (24,16,8,0):
            byte = (key[0] >> shift) & 255
            node.setdefault('mask', 0)
            node['mask'] |= 1 << byte
            node = node.setdefault(byte, {})
        node['index'] = index
    return root


def evaluate_curve(keys: tuple[CurveKey, ...], frame: int, signature=None) -> float:
    if not keys:
        raise ValueError('Cannot evaluate an empty curve')
    if frame <= keys[0].frame: return keys[0].value
    if frame >= keys[-1].frame: return keys[-1].value
    signature = signature if signature is not None else tuple((k.frame,k.value,k.easing,k.control1,k.control2) for k in keys)
    root = signature if isinstance(signature, dict) else _compile(signature)
    def predecessor(node, shift, tight=True):
        byte = ((frame >> shift) & 255) if tight else 255
        mask = node['mask'] & ((1 << (byte+1))-1)
        while mask:
            chosen = mask.bit_length()-1
            child = node[chosen]
            if shift == 0: return child['index']
            found = predecessor(child, shift-8, tight and chosen==byte)
            if found is not None: return found
            mask &= ~(1 << chosen)
        return None
    index = predecessor(root,24)
    left,right = keys[index],keys[index+1]
    if left.easing in ('hold','step'): return left.value
    t=(frame-left.frame)/(right.frame-left.frame)
    if left.easing in ('ease-in-out','bezier'):
        t=3*(1-t)**2*t*left.control1+3*(1-t)*t*t*left.control2+t**3
    return left.value+(right.value-left.value)*t


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
