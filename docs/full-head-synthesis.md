# Full-head 3D synthesis

`pose_3d` and `reenactment_3d` sessions now require the bundled MakeHuman CC0
head, measured 68-point target landmarks (`1k3d68.onnx`), and hardware OpenGL
3.3. ModernGL uses a standalone WGL context on Windows and EGL on Linux.
Unavailable geometry, landmarks or GPU context produce an explicit error;
these modes no longer return an ArcFace crop as their fallback.

Source ingestion enriches all archive views with available dense landmarks.
The identity builder jointly fits MakeHuman identity morphs across selected
views, then fits dense landmark residuals onto the complete head topology.
The content-addressed asset records the coefficients and confidence, a
1024px atlas for the 6 GB profile (2048px for 12 GB), and a bounded bank of
256px pose-dependent observation layers and 512-dimensional identity embeddings.
Up to 32 quality-filtered views contribute to an asset. Stored 106- and
468-point landmarks survive archive round trips; they are not reinterpreted as
68-point landmarks without an anatomical correspondence.

Each target frame uses dense PnP with estimated intrinsics, filtered rotation
and translation, and source-template metric scale. Target expression
measurements drive the source rig after neutral calibration. The output ROI
contains every projected head vertex, with padding, even outside the image.
GPU depth testing and perspective UV interpolation render the complete mesh.
The nearest three yaw/pitch/roll observation layers blend over the stitched
atlas. Readback occurs at the existing numpy compositor boundary.

When LivePortrait is installed and retargeting is enabled, its appearance
encoder and SPADE generator operate on the entire 512px UV-rendered portrait.
The multi-scale detail bridge keeps the UV render's low-frequency geometry
and transfers neural detail over the head ROI. The 12 GB path returns a
1024px ROI, with refinement upsampled from the network's native 512px output.
`retarget="required"` requires the neural models; `auto` reports UV-only
operation when they are unavailable. This is an adapter for existing portrait
weights, not a newly trained head-specific latent or spatial feature model.

Full-head replacements automatically use the robust XSeg/BiSeNet/foreground
compositor. Source mesh coverage owns the silhouette; the target's facial
hull cannot cut it back down. Parser and occluder-network availability remain
visible in compositor reports. Hair, glasses and foreground masks veto output;
unrecognized foreground objects still depend on installed occlusion models or
manual exclusion masks.

Unseen texels stay transparent. Archives containing only tight facial crops
cannot provide forehead, ears, scalp or back-of-head texture. Unconstrained
cranial geometry remains a morph-template prior. Estimated intrinsics and
monocular scale are not camera calibration or measured metric anatomy.
The standard 2D swap mode and its optional legacy crop projector remain
available explicitly; they do not claim full-head synthesis.

Regression coverage includes hardware rasterization from -60 to +60 degrees,
archive-to-asset-to-GPU geometry, dense PnP, pose-layer selection, and forehead /
jaw compositing with a foreground veto. Geometry tests deliberately use opaque
texture support to distinguish silhouette clipping from missing observations;
they do not establish photorealistic quality on real footage.
