"""Validated, per-target portrait edits shared by preview and export."""
from typing import Annotated
import math
from pydantic import Field, StringConstraints, field_validator
from roop_3d.core.settings import Contract
from roop_3d.reconstruction.arkit_bridge import ARKIT52_NAMES


class FaceEditor(Contract):
    head_pitch: float = Field(default=0, ge=-30, le=30)
    head_yaw: float = Field(default=0, ge=-45, le=45)
    head_roll: float = Field(default=0, ge=-30, le=30)
    gaze_x: float = Field(default=0, ge=-1, le=1)
    gaze_y: float = Field(default=0, ge=-1, le=1)
    blink_left: float = Field(default=0, ge=0, le=1)
    blink_right: float = Field(default=0, ge=0, le=1)
    eyebrow_tilt: float = Field(default=0, ge=-1, le=1)
    eyebrow_raise: float = Field(default=0, ge=-1, le=1)
    mouth_smile: float = Field(default=0, ge=-1, le=1)
    mouth_pout: float = Field(default=0, ge=0, le=1)
    mouth_open: float = Field(default=0, ge=0, le=1)
    jaw_width: float = Field(default=0, ge=-1, le=1)
    chin_length: float = Field(default=0, ge=-1, le=1)
    arkit_blendshapes: dict[str, float] = Field(default_factory=dict, max_length=52,
                                                description="Sparse ARKit 52 sliders; each named value is in [0,1]")

    @field_validator("arkit_blendshapes")
    @classmethod
    def valid_arkit_blendshapes(cls, value: dict[str, float]) -> dict[str, float]:
        unknown = sorted(set(value) - set(ARKIT52_NAMES))
        if unknown:
            raise ValueError(f"Unknown ARKit blendshape(s): {', '.join(unknown)}")
        if any(not math.isfinite(amount) or not 0 <= amount <= 1 for amount in value.values()):
            raise ValueError("ARKit blendshape values must be finite and in [0,1]")
        return value

    def active(self):
        return any(self.model_dump().values())


FaceExpressionOverrides = dict[Annotated[str, StringConstraints(pattern=r'^track_[0-9]{4}$')], FaceEditor]
