"""Validated, per-target portrait edits shared by preview and export."""
from typing import Annotated
from pydantic import Field, StringConstraints
from roop_3d.core.settings import Contract


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

    def active(self):
        return any(self.model_dump().values())


FaceExpressionOverrides = dict[Annotated[str, StringConstraints(pattern=r'^track_[0-9]{4}$')], FaceEditor]
