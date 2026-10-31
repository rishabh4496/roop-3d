"""Bounded workstation parameters shared by previews and durable render requests."""
from typing import Literal
from pydantic import Field
from roop_3d.core.settings import Contract


class WorkstationParameters(Contract):
    restoration_blend: float = Field(default=0, ge=0, le=1)
    feather_px: float = Field(default=5, ge=0, le=64)
    color_strength: float = Field(default=0.65, ge=0, le=1)
    pose_strength: float = Field(default=0.35, ge=0, le=1)
    bypass: tuple[Literal['restorer', 'color'], ...] = Field(default=(), max_length=2)

    def composite_options(self):
        from roop_3d.compositor.pipeline import CompositeOptions
        return CompositeOptions(restoration_blend=0 if 'restorer' in self.bypass else self.restoration_blend,
                                feather_px=self.feather_px,
                                color_strength=0 if 'color' in self.bypass else self.color_strength)
