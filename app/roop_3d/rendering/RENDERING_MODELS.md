# Appearance calibration and physical buffers

`skin_calibration.py`, `physical_buffers.py`, and `intrinsic_texture.py` are
original clean-room implementations licensed under `RENDERING_MODELS_LICENSE`.
They contain no research-project source, checkpoints, or datasets. The TRUST
paper motivates resolving scene-light/albedo ambiguity using scene context;
DiffusionRig motivates geometry-aligned physical conditioning buffers; FATE
motivates editable texture-space appearance. These modules do not reproduce
the learned methods or their reported results.

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
rasterizer and returns four same-sized buffers: camera-space normals, metric
z-depth, linear BGR UV albedo, and linear BGR second-order SH Lambertian plus
dielectric specular shading. It also returns geometry/UV confidence. The SH
coefficients use this application's BGR irradiance convention. The specular
term spends a bounded Fresnel energy budget so shaded output does not exceed
incoming irradiance for albedo in `[0, 1]`.

`IntrinsicTextureDecomposer.canonicalize` attenuates broad log-luminance
shading, removes high-luminance low-chroma outliers, and retains a four-level
Laplacian detail pyramid with metric displacement bands. Canonical identity
asset creation exposes this through
`IdentityAssetSettings(intrinsic_texture_decomposition=True)`. It is opt-in
because single-view albedo and illumination are underconstrained; default
atlas construction preserves the existing measured-texture path.

The included tests cover color-temperature/exposure invariance, preservation
ratio endpoints, SH energy bounds, and retention of four texture-detail bands.
