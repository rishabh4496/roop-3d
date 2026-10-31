"""Bounded workstation parameters shared by previews and durable render requests."""
from typing import Literal
from pydantic import Field, model_serializer
from roop_3d.core.settings import Contract


class WorkstationParameters(Contract):
    restoration_blend: float = Field(default=0, ge=0, le=1)
    feather_px: float = Field(default=5, ge=0, le=64)
    color_strength: float = Field(default=0.65, ge=0, le=1)
    pose_strength: float = Field(default=0.35, ge=0, le=1)
    pose_mode: Literal['retarget', 'mesh_blend'] = 'retarget'
    swap_weight: float = Field(default=1, ge=0, le=1)
    erosion_px: float = Field(default=0, ge=-32, le=32)
    bypass: tuple[Literal['restorer', 'color'], ...] = Field(default=(), max_length=2)

    @model_serializer(mode='wrap')
    def compatible_serialization(self, handler):
        value = handler(self)
        for field in ('swap_weight', 'erosion_px', 'pose_mode'):
            if field not in self.model_fields_set:
                value.pop(field, None)
        return value

    def inference_options(self) -> dict[str, float]:
        # The new curve changes geometric projection, never LivePortrait motion.
        return {'pose_blend': self.pose_strength} if self.pose_mode == 'mesh_blend' else {'strength': self.pose_strength}

    def composite_options(self):
        from roop_3d.compositor.pipeline import CompositeOptions
        return CompositeOptions(restoration_blend=0 if 'restorer' in self.bypass else self.restoration_blend,
                                feather_px=self.feather_px,
                                swap_weight=self.swap_weight, erosion_px=self.erosion_px,
                                color_strength=0 if 'color' in self.bypass else self.color_strength)
