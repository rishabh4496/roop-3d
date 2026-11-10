# Appearance calibration and hybrid rendering

The modules in this directory are original clean-room implementations
licensed under `RENDERING_MODELS_LICENSE` (MIT). CVTHead, GPAvatar, MeGA,
GAGAvatar and RGBAvatar are design references only: no project source, weights,
datasets or CUDA extensions from those projects are included. New tensor paths
use the standard PyTorch API; the mesh pass continues to use the existing
ModernGL rasterizer.

## Interfaces

`UnbiasedSkinToneCalibrator.decompose` returns linear BGR albedo, median CIE
Lab values, melanin/hemoglobin optical-density coordinates, and an explicit
scene illuminant estimate. Pass an illumination from `SceneIllumination` or a
full scene image (ideally with a neutral patch mask). `calibrate(source_face,
target_face, albedo_preservation_ratio=...)` maps intrinsic tone between crops
and re-illuminates it under the target scene estimate. Ratio 0 uses target
intrinsic tone; ratio 1 uses source intrinsic tone. No demographic skin prior
is applied. The melanin/hemoglobin axes are chromaticity proxies, not measured
biomarkers.

`PhysicalBufferRenderer.render` uses the existing hardware ModernGL mesh
rasterizer and returns camera-space normals, metric z-depth, linear BGR UV
albedo, SH Lambertian plus dielectric specular shading, and geometry/UV
confidence. `render_torch` keeps gradients through UV texture sampling and SH
shading on a fixed visibility buffer. The hardware pass decides triangle
ownership, so it does not provide gradients through mesh visibility or pose.

`IntrinsicTextureDecomposer.canonicalize` attenuates broad log-luminance
shading, removes high-luminance low-chroma outliers, and retains a four-level
Laplacian detail pyramid with metric displacement bands. Canonical identity
asset creation exposes this through
`IdentityAssetSettings(intrinsic_texture_decomposition=True)`. It is opt-in
because single-view albedo and illumination are underconstrained; default atlas
construction preserves the existing measured-texture path.

`MultiViewCanonicalAggregator` cross-attends over per-view UV layers using the
target viewing direction, each texel's surface normal when supplied, view
quality, and observed UV support. `IdentityRig.texture_for_pose` uses the
asset's pose bank in the Stage 4 reenactment path. Identity assets retain a
small, separate UV texture/support layer per selected source view, including
canonical-template assets. Side-profile texels with real support can therefore
contribute at profile yaw; unsupported texels still fall back to the stitched
atlas. Pose-bin selection preserves rare profile references in both templates.
`project_faceset_views` projects a decoded `.fsz` document using fitted cameras
and the existing visibility-tested `UVSynthesizer`.

`ReducedBlendshapeBank.fit` builds up to 20 appearance bases from supplied,
pose-normalized appearance samples and expression vectors. `evaluate` accepts
numpy or torch expression batches. It is not trained automatically from
pose-varying FaceSet views, because those views do not separate expression
changes from pose, illumination and registration errors.

`HybridMeshGaussianRenderer` keeps facial skin on `PhysicalBufferRenderer` and
accepts depth-tested sparse Gaussian splats only for the explicit `eyebrow`,
`eyelash`, `beard_stubble`, and `inner_mouth` regions. Its optional torch pass
keeps gradients through UV texture sampling, SH shading, expression-bank
weights, and point splats. Splats must be supplied with those region labels;
the renderer does not infer eyebrows, eyelashes or facial hair from an atlas.
Triangle visibility and UV interpolation come from the fixed hardware
z-buffer, so gradients do not pass through triangle ownership or mesh pose.
The implementation has not been benchmarked at 120 FPS.

For low-VRAM operation, `FastUVFallbackRenderer` applies optional metric UV
displacement to mesh vertices and performs perspective-correct, depth-tested
atlas warping. It uses the existing bounded rasterizer and needs no splat or
appearance-bank allocation. The active 6 GB reenactment route uses this pass;
the metric displacement atlas is stored when intrinsic texture decomposition
is enabled and is zero otherwise. The 12 GB reenactment route uses the
hardware mesh renderer and reports any context failure through renderer
capability state before reverting to the bounded warp.
