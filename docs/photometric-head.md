# Full-head photometric synthesis

The full-head MakeHuman route now passes perspective-interpolated camera-space
normals, rest-space normals and metric depth from the hardware rasterizer to
Stage 4. Enable the existing **3D relighting** control (`CompositeOptions.relight`)
to use the hardware OpenGL photometric pass after enhancement and before the
single high-resolution inverse warp. Standard 2D crops keep their existing path.

The relighting pass uses nine RGB coefficients: three SH bands, orders 0–2.
This is the low-frequency diffuse irradiance representation described by
[Ramamoorthi and Hanrahan](https://graphics.stanford.edu/papers/envmap/).
Internally images and coefficients are BGR to match the app's numpy/OpenCV
boundary; the shader performs arithmetic in linear light.

## Calibration and rendering

For each identity the rest mesh is rasterized into UV space. Canonical texture
and confidence produce an estimated diffuse albedo and source SH calibration.
Each selected texture-bank blend is calibrated against this albedo so changes
of source view do not introduce an unaccounted illumination denominator.
The estimator fits at most approximately 8,192 samples with robust weighted
least squares. Calibration runs on the CPU; diffuse-ratio shading, colored
Blinn-Phong highlights, ambient occlusion and directional visibility run in VRAM.
The app's existing numpy renderer/compositor boundaries still perform readback.

Target geometry fits measured dense landmark XY to the MakeHuman depth prior,
independently of the identity silhouette. Visible semantic skin, coverage,
occluder vetoes and measured eye/mouth exclusions weight the target SH solve.
Manual key-light settings remain available. The shader evaluates:

```
enhanced_linear * (target_irradiance * geometric_shadow + colored_specular)
                  / max(source_irradiance_at_rest_normal, 0.04)
```

Normals are welded across UV seams to avoid discontinuous highlights. Gain is
bounded to prevent near-zero source irradiance from exploding. Specular
requires a positive normal/light dot product. A metric, short-range horizon AO
term and twelve key-light depth queries shade nose, brow, lip and jaw
concavities. Target luminance residuals retain cast shadows beyond the smooth
SH prediction. Native enhancer resolution is retained through this pass.
The 6 GB profile still renders 512-pixel mesh ROIs; the 12 GB profile uses 1024.
The photometric shader accepts native enhancer ROIs up to 2048 pixels. Temporary
GPU textures are released each call, and retained calibration is bounded per
active identity rather than per frame.

Single-image intrinsic decomposition is underdetermined. The calibration uses
a diffuse-reflectance prior with luminance 0.5 and robust source chromaticity,
assuming neutral source illumination; it is not measured lighting-invariant material
recovery. Target metric depth remains a template prior. Screen-space AO and
key visibility cannot detect occluders hidden behind visible geometry. Second
order diffuse SH also cannot resolve an arbitrarily narrow neon light source.

Aggressive illumination changes can lower recognition similarity even when
landmarks remain fixed. The full-head compositor therefore measures real
ArcFace similarity against the enhanced baseline. It first tests the physical
result, then searches at most five blend candidates if needed, accepting a
measured cosine of at least 0.80 by default. This constrains identity retention
through relighting; it does not prove that upstream swapping matched the source.
`capability().photometric_identity` reports the raw and accepted similarities,
requested strength and effective strength for each track. Calibration, image
conditioning, this identity guard and existing final fusion remain host-side;
the photometric shader itself runs in VRAM. The guard adds local recognition
inference cost. `CompositeOptions.photometric_identity_floor=0` explicitly
bypasses it for physical rendering diagnostics.

This guard can substantially attenuate harsh lighting when the geometric prior
does not match the real actor. A passing guarded identity test is not evidence
that the full-strength physical rendering met the same threshold. The retained
fraction and raw result must be reviewed together.

## Iris and oral detail

The tracker automatically discovers an optional **local** model at
`app/models/face_landmarker.task`. Supply its adjacent
`face_landmarker.task.json` with the expected bytes and SHA-256:

```json
{"sha256": "<64-character model digest>", "size_bytes": 1234567}
```

The existing MediaPipe adapter verifies this identity before activation; it
also requires the optional `mediapipe` Python package in this app's environment.
There are no downloads in the processing path. Each actor is cropped separately
before extraction. Missing detections provide no gaze measurement for that
frame. A configured corrupt model fails explicitly.

[MediaPipe's attention landmark graph](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_landmark/tensors_to_face_landmarks_with_attention.pbtxt)
provides 478 landmarks including irises. A 468-point mesh supplies an oral
aperture but does **not** supply iris landmarks. Iris centers remain floating
point, masks rasterize with 1/16-pixel vertices and antialiased edges, and each
eye gets an independent geometric pitch/yaw vector. Closed lids suppress the
iris mask. There is no temporal averaging of fast iris motion.

The current neural models have no trained iris-mask/gaze-vector input. The
implementation therefore uses image-space conditioning: measured target iris
pixels and the measured ocular aperture constrain synthesis before enhancement
and are locked again after enhancement/relighting. Replacing the aperture also
clears the previously synthesized pupil, avoiding duplicate irises after a
saccade. This preserves actor eye appearance, rather than
claiming to generate source-identity eyeballs. Gaze is relative to an eyeball
radius prior, not a calibrated point of regard.

When aperture height/width exceeds `tau_mouth = 0.12`, oral synthesis combines
enhanced detail with target high frequencies and target cavity luminance. The
target's dark interdental gaps cap synthesized brightness; tooth details are
never spatially blurred in this operation. A smooth activation ramp avoids
popping near the threshold. 68-point profile landmarks above 65 degrees yaw
do not assert a visible mouth aperture. Existing foreground alpha vetoes apply
to the final composite, including iris and mouth pixels.

## Cranial/neck transition

The open inferior mesh boundary receives an 8 mm submental ribbon with inherited
UVs. Its cubic alpha falloff is guided by surface normal orientation. Closed
templates receive no invented boundary. The source jaw contour remains intact;
the existing target-minus-source silhouette removal still clears the actor's
old jaw. All foreground exclusions remain in force.

## Verification

From `app/`:

```powershell
roop_3d/.venv/Scripts/python.exe -B -m pytest -c roop_3d/pyproject.toml --confcutdir=roop_3d roop_3d/tests/test_photometric_head.py roop_3d/tests/test_extreme_pose_3d_swap.py -q
```

The synthetic checks exercise the actual hardware shader against independent
SH polynomials for harsh red/blue split lighting and overhead shadows. They
also check preserved spatial texture, protected pixels, robust light recovery,
AO, normal rotation, subpixel saccades, blinks, oral gaps and the neck ribbon.
Spatial-texture preservation is not an ArcFace identity measurement.

The separate real ArcFace gate evaluates both physical lighting cases at full
strength, then applies the same identity constraint as the compositor. The
accepted result must exceed cosine similarity **0.78** for each. The test prints
the full-strength cosine, accepted cosine and applied fraction. Set
`ROOP_3D_RELIGHT_CASE` to an absolute path to an owned JSON manifest:

```json
{
  "image": "identity-roi.png",
  "buffers": "fitted-makehuman-buffers.npz",
  "landmarks5": [[0,0], [0,0], [0,0], [0,0], [0,0]]
}
```

Replace the five points with measured ROI-pixel landmarks. The NPZ contains
`normals` (HxWx3, +x right, +y down, +z viewer), `depth` (HxW, positive metric
camera depth, infinity off mesh), `inverse` (2x3 ROI-to-frame matrix) and `camera`
(3x3 intrinsics) captured from the fitted MakeHuman render. Include `landmarks68`
in ROI pixels to exclude measured eye/mouth apertures from skin relighting. The image must
already be in that same square ROI. All paths stay in this checkout; the
recognizer uses the app's verified local model catalog. With no manifest this
gate skips explicitly. Once a manifest is configured, invalid data or missing
weights fails the gate. An owned real FaceSet with measured 68-point landmarks
can prepare the fixture directly:

```powershell
roop_3d/.venv/Scripts/python.exe -B -m roop_3d.tools.prepare_relighting_case facesets/source.fsz
$env:ROOP_3D_RELIGHT_CASE = (Resolve-Path roop_3d/.runtime/acceptance/photometric-head/case.json).Path
```

This fixture tests real-photo identity retention through relighting, rather than
accuracy of a separate identity swap. Passing synthetic checks alone does not establish
the identity threshold or superiority to a 2D pipeline on real videos.

### Local validation, 2026-11-08

On the local RTX 4070, the focused rendering/compositor selection completed
with **136 passed, 4 skipped**. The skips are the absent captured extreme-pose
photo fixtures. Both 512- and 1024-pixel enhancement handoffs were exercised.
The owned real-photo FaceSet fixture and verified `w600k_r50.onnx` produced:

| Scene | Full-strength cosine | Guarded cosine | Retained edit fraction |
| --- | ---: | ---: | ---: |
| Red/blue neon split | 0.444843 | 0.811443 | 0.21875 |
| Overhead key/shadow | 0.347409 | 0.826250 | 0.12500 |

The requested **unattenuated** >0.78 criterion is not met by this fixture.
The identity guard makes the reduced-strength output pass; it does not remove
the need for better actor-specific geometry/intrinsic-image estimation. No
comparative superiority over a 2D pipeline is established by these results.

The broad suite run returned **1,575 passed, 13 failed, 21 skipped** in 28m33s;
it is not a green repository acceptance result. Failures covered identity
tracking, canonical reenactment pose/expression timing, two render jobs reporting
`PROGRESS_REGRESSION`, a legacy import alias, server readiness, model-register
count drift, live ONNX session accounting and an outdated launcher statement.
The 70-degree canonical reenactment failure was reproduced with the unchanged
HEAD versions of the modified renderer/UV/background modules (identical
`ENTERING` result and confidence). The other failures have not all been
baseline-isolated. Final changes were checked with the focused selection above.
