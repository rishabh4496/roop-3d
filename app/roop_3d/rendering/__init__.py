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

# Seam-free swap tooling is exported lazily: it pulls in scipy and the reconstruction package, which the renderers above
# deliberately import only on demand (reconstruction imports rendering lazily, so an eager import here would be circular).
_LAZY = {
    "StrandAwareOcclusionCompositor": "strand_occlusion", "StrandOcclusionOptions": "strand_occlusion",
    "OcclusionInputs": "strand_occlusion", "OcclusionResult": "strand_occlusion", "OcclusionReport": "strand_occlusion",
    "SwapRegionConfig": "swap_region_3d", "SwapMask3D": "swap_region_3d", "JawTarget": "swap_region_3d",
    "JawAdaptation": "swap_region_3d", "WeldedGraph": "swap_region_3d", "build_swap_mask_3d": "swap_region_3d",
    "adapt_jaw": "swap_region_3d", "jaw_target_from_ratios": "swap_region_3d", "jaw_target_from_landmarks": "swap_region_3d",
    "rasterize_swap_mask": "swap_region_3d", "rasterize_vertex_attributes": "vertex_raster", "RasterBuffers": "vertex_raster",
}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(f"{__name__}.{module}"), name)


__all__ = [
    *_LAZY,
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
