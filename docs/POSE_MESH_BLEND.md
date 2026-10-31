# Pose curve: TPS 2.5D to perspective 3D mesh

The pose correction curve now has a geometric mesh-blend implementation in preview,
queued rendering, saved projects and parameter sweeps. New workstation sessions use
the **2.5D / 3D mesh blend (canonical proxy)** method. The canvas method selector
also exposes the original reference LivePortrait retarget path.

## Geometry and interpolation

The mesh path uses the existing explicit 68-landmark canonical head template and
its triangulated surface. It is an **estimated canonical proxy**, not a fitted
identity mesh, a FLAME reconstruction, or a measurement of the person's anatomy.
No restricted weights, downloads, or new dependencies are required.

Only the known five-point landmark correspondence is used. Detector-specific
106/203-point orderings are not reinterpreted as canonical anatomical indices.
The current swapped crop is neutralized using a backward thin-plate spline. That
same observed texture feeds both endpoints:

- **0:** a flat surface deformed by the forward TPS (the 2.5D warp).
- **1:** the canonical 3D surface fitted to observed crop landmarks with SQPnP and
  LM refinement, then perspective projected and rasterized.
- **Between 0 and 1:** interpolate the flat and fitted vertices in camera space,
  project the resulting surface, and rasterize it with perspective-correct texture
  sampling. This is a geometry morph, not an RGB cross-fade.

The crop camera is explicitly estimated: focal length is `1.5 × max(crop width,
crop height)` and the principal point is the crop center. This is a policy estimate,
not supplied camera calibration. Five-point reprojection error is measured and
reported separately. Solver conventions follow the [OpenCV 4.11 pose-estimation
documentation](https://docs.opencv.org/4.11.0/d5/d1f/calib3d_solvePnP.html).

The rasterizer rejects back-facing and behind-camera triangles, resolves overlap
with a z-buffer, and rejects out-of-texture sampling through a coverage mask. The
compositor preserves the target wherever current projection coverage is zero,
including after mask dilation and temporal smoothing. Invalid landmarks, failed
fits, and empty visible surfaces produce explicit errors; they do not silently
switch to retargeting.

Projection is lazy and bounded to 16–512 pixel crops; the current identity-only
mesh path uses the existing 128-pixel swap crop. TPS evaluation uses bounded strips.
API boot does not construct a mesh or activate inference models. Optional FLAME,
diffusion and TensorRT dependencies remain optional.

## Curve and request behavior

`workstation_parameters.pose_mode` selects `mesh_blend` or `retarget`.
`pose_correction_strength` still evaluates into `pose_strength` over source frame
indices, but the selected method determines its meaning. In mesh mode it passes
`pose_blend` into the reference pipeline; it never changes LivePortrait strength.
Mesh-mode sessions disable the optional LivePortrait path so the input texture and
crop transform are consistent across both geometric endpoints.

Older requests, projects and checkpoints without `pose_mode` retain `retarget` and
their previous identity payload. The new field is not inserted into serialized
legacy parameters. Opening an old project explicitly restores the reference mode;
the user can switch it to mesh blending through the normal reversible controls.

The pose pipeline buffer overlays the actual weighted projected mesh and visible
triangles. Metadata identifies the proxy and estimated camera, weight, reprojection
error and visible texture coverage. This does not claim multi-view UV synthesis.

The 3×3 sweep tool includes a pose-blend axis with example values `0, 0.5, 1`.
Tracking and neural swapping run once. The cached TPS endpoint, mesh endpoint,
texture and geometry permit each cell to be rasterized at its exact weight without
another neural inference. Reference-mode pose sweeps are rejected explicitly.

## Validation scope

New tests cover exact endpoints, camera-space interpolation, depth dependence,
perspective-correct sampling, z-buffer occlusion, back-face rejection, degenerate
landmarks, unseen-target preservation, legacy serialization, curve dispatch and
nine-cell pose sweeps.

Real local CPU acceptance exercises detector/identity inference and the compositor
at weights 0, 0.5 and 1. A second test builds a real local FaceSet, inspects a clip,
requests a mesh preview, and completes a queued animated render with encoded video.
Rendered test media remain in ignored, owned runtime directories.

Validation completed on 2026-11-01 (Asia/Calcutta): the full Python suite passed
430 tests and 18 subtests; the frontend passed 82 tests, TypeScript and the Vite
production build. The read-only isolation audit completed at
`2026-10-31T18:30:32Z` with all 68,390 protected files and 2,645 historical entries
verified. Permitted legacy runtime activity is reported separately by audit policy.

This establishes functional CPU execution and output evidence. It does not establish
personalized 3D reconstruction accuracy, improved identity fidelity, speed, or the
separate 6 GB / 12 GB GPU acceptance campaigns. Visual browser/multi-monitor
acceptance is not claimed.
