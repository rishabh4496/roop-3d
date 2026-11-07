# Full-face perspective rendering

The active implementation is `app/roop_3d/reconstruction/full_head.py`, called
from `engine/volumetric/session.py`. This checkout has no
`app/processors/face_swapper_3d.py`.

## Geometry and visibility

Full-head modes use dense 3D-to-2D correspondences, RANSAC PnP and LM refinement.
The camera is refitted after GPU depth testing removes hidden correspondences.
Residual rejection uses the landmark extent rather than foreshortened eye
separation. Failed solves raise a domain error; they do not select affine warping.
Standard sessions also enter the full-head path above 30 degrees absolute yaw or
25 degrees absolute pitch. Their first extreme frame can incur an identity-asset
build. Missing prerequisites are reported as errors.

The installed runtime detector is `1k3d68.onnx`. The engine also accepts other
dense layouts, including 468 points, **when explicit correspondences to the actual
mesh vertices are supplied**. An arbitrary 468-point array cannot be treated as
MakeHuman vertex indices. No additional detector or model is downloaded.

`HeadPose.projection_matrix` exposes K[R|t] in the OpenCV object frame. The stored
mesh's viewer-facing z is reflected before this transform. `camera_matrix` may be
supplied to `FullHeadEngine.process`; otherwise focal length is estimated from
frame dimensions. Uncalibrated monocular footage cannot establish an exact camera
or absolute metric identity size.

The hardware OpenGL renderer outputs perspective-correct UVs, metric depth,
appearance confidence and a separate geometric silhouette. GPU point queries
compare each mesh vertex or UV surface sample against the winning depth. A hidden
far-side sample cannot erase the visible near-side pixel that projects to the
same position. UV acquisition uses the fitted scaled-orthographic source camera
with orthographic interpolation; target synthesis uses perspective interpolation.

The silhouette uses the entire MakeHuman head and neck, including cranial and
jaw vertices, without a 68-point or inner-square composite hull. It therefore
does not need a fabricated 86-point cranial correspondence. Temporal filtering
occurs before rasterization; compositing does not filter the resulting ROI
transform a second time.

## Source appearance

MakeHuman asset builds reserve quality-ranked views from nine buckets before
filling remaining slots: frontal, left/right quarter, left/right profile, up,
down, left oblique and right oblique. Both pitch signs are retained within each
oblique bucket. The manifest records actual source indices; empty buckets are
left empty. This bucket scheme is separate from the existing embedding matrix.

The 6 GB profile keeps a 1024-square canonical atlas and 512-square rendering;
the 12 GB profile keeps its existing 2048-square atlas and 1024-square rendering.
Both use bounded 256-square per-view atlas layers. Algorithm version
`identity-mv-4` invalidates old cached identity assets.

Rendering blends the nearest two viewing directions using normalized opposite
spherical arc distances. Two samples define an arc, not a spherical triangle;
off-arc targets use normalized angular-distance weights. Exact views have unit
weight. Roll is omitted from view selection because the observations are already
registered in canonical UV space. Colors are resized premultiplied by visible
support to avoid black edge contamination. Observed view layers can restore
support absent from the canonical atlas.

Missing source observations receive a **diffuse source-color appearance prior**
at runtime. They no longer make the original actor's facial features show through
the silhouette. This completion does not copy a frontal eye or mouth onto an
unseen cheek. Asset observation/confidence states are preserved: these pixels are
not recovered identity detail. Truly seamless photorealistic appearance over an
unobserved region cannot be guaranteed.

The selected installed swapper processes the pose-matched full-head ROI directly,
without target five-point alignment, and contributes a high-frequency residual
inside observed support. Existing optional LivePortrait detail refinement remains
available. These models were trained on their own crop distributions; full-head
residual quality still needs real footage evaluation. The renderer owns the base
anatomy and coverage in every case.

## Silhouette completion and occluders

The ROI encloses both the projected source head and the measured target jaw. A
wider source may expand beyond the original face boundary. For a slimmer source,
the compositor computes target-minus-source support, including trusted semantic
skin where available. GPU jump flooding propagates surrounding background/neck
donors into that ribbon. Source/target head pixels, out-of-frame padding and
foreground occluders are excluded as donors. Only ribbon pixels are composited;
the source silhouette and foreground masks bound the operation.

This is spatial background propagation, not optical-flow history, Telea, or a
LaMa model. It cannot recover unseen background structure. Missing donor pixels
raise an explicit error. BiSeNet/visible-face/manual foreground masks retain
their existing confidence and availability contracts; hand protection depends on
the installed visible-face models, since BiSeNet has no hand class.

## Verification

Run from `app/`:

```powershell
roop_3d/.venv/Scripts/python.exe -B -m pytest -c roop_3d/pyproject.toml --confcutdir=roop_3d roop_3d/tests/test_extreme_pose_3d_swap.py -q
```

The deterministic suite covers 0, 35, 60, 80 and 85 degree yaw in both directions,
and -60/0/+60 pitch, using the actual MakeHuman mesh, PnP, hardware renderer and
compositor. It checks modified skull/cheek/jaw bands, depth-hidden marker rejection
in pixels and vertex queries, angular selection, foreground preservation and old
jaw removal. Synthetic ground truth establishes geometry, not real detector
accuracy or human-perceived identity quality.

Captured-photo acceptance is opt-in. Install authorized `source.fsz` and
`targets.fsz` under `app/roop_3d/tests/fixtures/extreme_pose_real/`. Target photos
must have measured dense landmarks or support local dense detection, actual yaw
labels within five degrees of 0/35/60/80, and pitch within ten degrees of level.
Use different source and target identities. Generated turntables or transformed
frontal photographs are not real-profile fixtures. These four tests explicitly
skip when the archives are absent. No real-profile pass or universal 100 percent
identity-replacement guarantee is implied by the synthetic tests.

## Validation recorded on 2026-11-08

- Final extreme-pose suite: **74 passed, 4 skipped**, using hardware OpenGL.
- Related full-head, identity-asset and robust-occlusion integration run:
  **151 passed, 4 skipped**. Four additional UV-acquisition cases were added
  afterward and passed in the final extreme-pose run above.
- Repository-wide run with `--maxfail=8`: **423 passed, 2 skipped, 8 failed**;
  it stopped early and does not establish a clean full-suite result.
- All eight failures were reproduced in isolated processes loading the original
  `HEAD` versions of every modified Python module. They comprise the null-hash
  diagnostics report, real identity-tracker phantom acceptance, the animated
  render job's `PROGRESS_REGRESSION`, and five legacy reenactment pose/expression
  checks. They were not changed as part of this geometric/compositing refactor.
- `git diff --check` passed. No model downloads, sibling changes, commits or
  pushes were performed.
