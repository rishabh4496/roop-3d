# SPDX-License-Identifier: MIT
"""Clean-room, local appearance calibration and physical rendering tools."""

from roop_3d.rendering.intrinsic_texture import IntrinsicTextureDecomposer, TextureDecomposition
from roop_3d.rendering.physical_buffers import (
    PhysicalBufferRenderer,
    PhysicalBuffers,
    evaluate_sh_irradiance,
    shade_lambertian_sh,
)
from roop_3d.rendering.skin_calibration import (
    SkinToneEstimate,
    SceneIllumination,
    UnbiasedSkinToneCalibrator,
)

__all__ = [
    "IntrinsicTextureDecomposer",
    "PhysicalBufferRenderer",
    "PhysicalBuffers",
    "SceneIllumination",
    "SkinToneEstimate",
    "TextureDecomposition",
    "UnbiasedSkinToneCalibrator",
    "evaluate_sh_irradiance",
    "shade_lambertian_sh",
]
