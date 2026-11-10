# SPDX-License-Identifier: MIT
"""Clean-room, local appearance calibration and physical rendering tools."""

from roop_3d.rendering.intrinsic_texture import IntrinsicTextureDecomposer, TextureDecomposition
from roop_3d.rendering.hybrid_renderer import (
    BoundaryGaussianSplatRenderer,
    BoundarySplats,
    CanonicalAppearance,
    FastUVFallbackRenderer,
    HybridMeshGaussianRenderer,
    HybridRenderResult,
    MultiViewCanonicalAggregator,
    ReducedBlendshapeBank,
    SplatLayer,
)
from roop_3d.rendering.physical_buffers import (
    DifferentiablePhysicalBuffers,
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
    "BoundaryGaussianSplatRenderer",
    "BoundarySplats",
    "CanonicalAppearance",
    "FastUVFallbackRenderer",
    "HybridMeshGaussianRenderer",
    "HybridRenderResult",
    "MultiViewCanonicalAggregator",
    "DifferentiablePhysicalBuffers",
    "PhysicalBufferRenderer",
    "PhysicalBuffers",
    "ReducedBlendshapeBank",
    "SceneIllumination",
    "SkinToneEstimate",
    "TextureDecomposition",
    "UnbiasedSkinToneCalibrator",
    "SplatLayer",
    "evaluate_sh_irradiance",
    "shade_lambertian_sh",
]
