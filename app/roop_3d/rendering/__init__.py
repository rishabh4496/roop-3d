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
    "PhysicalCameraMatcher": "physical_camera", "OpticalMeasurements": "physical_camera",
    "OpticalMatchInfo": "physical_camera", "DominantKeyLightAligner": "physical_camera",
    "KeyLightEstimate": "physical_camera", "RimLightLobe": "physical_camera",
    "TemporalSpatialSwapDenoiser": "physical_camera", "EnvironmentProbe": "physical_camera",
    "CC0PreviewAssets": "physical_camera", "load_cc0_preview_assets": "physical_camera",
    "synthesize_micro_pore_normal": "physical_camera", "sample_micro_pore_normal": "physical_camera",
    "StrandAwareOcclusionCompositor": "strand_occlusion", "StrandOcclusionOptions": "strand_occlusion",
    "OcclusionInputs": "strand_occlusion", "OcclusionResult": "strand_occlusion", "OcclusionReport": "strand_occlusion",
    "SwapRegionConfig": "swap_region_3d", "SwapMask3D": "swap_region_3d", "JawTarget": "swap_region_3d",
    "JawAdaptation": "swap_region_3d", "WeldedGraph": "swap_region_3d", "build_swap_mask_3d": "swap_region_3d",
    "adapt_jaw": "swap_region_3d", "jaw_target_from_ratios": "swap_region_3d", "jaw_target_from_landmarks": "swap_region_3d",
    "rasterize_swap_mask": "swap_region_3d", "rasterize_uv_swap_mask": "swap_region_3d",
    "rasterize_vertex_attributes": "vertex_raster", "RasterBuffers": "vertex_raster",
    "SilhouetteDeltaInpainter": "silhouette_inpaint", "InpaintOptions": "silhouette_inpaint", "DonorFrame": "silhouette_inpaint",
    "InpaintReport": "silhouette_inpaint", "overhang_ring": "silhouette_inpaint",
    "PerimeterCageWarper": "perimeter_cage", "CageOptions": "perimeter_cage", "CageField": "perimeter_cage",
    "perimeter_from_masks": "perimeter_cage", "perimeter_vertex_ids": "perimeter_cage", "mesh_perimeter_pairs": "perimeter_cage",
    "SilhouetteReplacer": "silhouette_replacement", "DonorHistory": "silhouette_replacement", "ScopePlan": "silhouette_replacement",
    "resolve_scope": "silhouette_replacement", "SWAP_SCOPES": "silhouette_replacement",
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
