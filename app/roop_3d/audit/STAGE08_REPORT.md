# Stage 08 — FLAME 3D Reconstruction, UV Synthesis & 2.5D Fallback

Stage 08 software gates passed. The 3D geometry subsystem provides typed coordinate contracts, an always-available license-free 2.5D Landmark Thin-Plate-Spline (TPS) / PnP geometry fallback, a strictly license-gated FLAME 2020 parametric fitter, visibility-weighted multi-view UV texture synthesis, and validated atomic `.obj` and binary `.glb` (glTF 2.0) mesh exports to `roop_3d/outputs/meshes/` for Stage 14's Three.js viewport.

## Verification

- Full suite: **305 passed, 18 subtests passed, 0 failed, 0 skipped**; exit 0. One existing FastAPI/HTTPX deprecation warning.
- All 23 Stage 08 tests are included:
  - Coordinate conversions and round-trips: Rodrigues <-> rotation matrix <-> quaternion <-> Euler (yaw, pitch, roll), OpenCV <-> OpenGL camera transformations.
  - Pinhole camera intrinsics projection/unprojection and FOV validation.
  - Canonical anthropometric 3D landmarks (5, 68, 106, 203 points).
  - PnP 3D pose solver with sub-pixel reprojection accuracy on ground truth.
  - Canonical 3D head mesh generation with counter-clockwise winding, vertex normals, and valid UVs.
  - 2D Thin-Plate-Spline (TPS) point and image deformation minimizing bending energy.
  - Piecewise affine Delaunay triangle warping.
  - FLAME license gate: strictly blocks unverified/unlicensed activation in accordance with Stage 03 registry contracts.
  - Robust Huber and Geman-McClure M-estimator weights downweighting gross outliers.
  - Staged parametric optimization on synthetic clean-room FLAME head models.
  - Visibility-weighted UV synthesis with surface normal dot-product weighting.
  - Per-texel `uv_confidence_mask` distinguishing `OBSERVED`, `INTERPOLATED`, and `UNSEEN` regions. Unseen back-of-head and occluded pixels are strictly masked out and never blended as skin texture.
  - Validated atomic `.obj` / `.mtl` export and binary `.glb` (glTF 2.0) container export with 4-byte chunk padding.
- Final audit: exit 0, **68,391 comprehensive immutable files** and **2,646 historical files** verified byte-for-byte; 0 violations. Legacy files and model baselines remain untouched.

Verification commands:

```powershell
roop_3d/.venv/Scripts/python.exe -B -m pytest -c roop_3d/pyproject.toml --confcutdir=roop_3d roop_3d/tests -q
roop_3d/.venv/Scripts/python.exe -B roop_3d/audit/verify_isolation.py --report
```

## Geometry and Fallback Contracts

1. **Typed Coordinate Conventions (`roop_3d/reconstruction/geometry_types.py`)**:
   - Right-handed OpenCV camera frame (+X right, +Y down, +Z forward) and OpenGL / Three.js viewport frame (+X right, +Y up, -Z forward) with exact involution converters.
   - Metric scale: 1.0 = 1 meter. Anthropometric landmarks and head dimensions span ~0.15m width and ~0.22m height.
   - Pinhole camera model $K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$ with forward projection and depth unprojection.
   - Winding order: counter-clockwise (CCW) front-facing triangles with area-weighted vertex normals.
   - UV origin mapping supporting both `BOTTOM_LEFT` (OpenGL / OBJ / glTF standard) and `TOP_LEFT` (image raster array).

2. **Always-Available 2.5D PnP + TPS Warper (`roop_3d/reconstruction/tps_fallback.py`)**:
   - Dependency-free 3D canonical landmark PnP solver driven by 5, 68, 106, or 203 points. Computes rigid camera pose (Rodrigues, translation, yaw, pitch, roll) and reprojection error without external dependencies.
   - Canonical 3D Delaunay head surface mesh generation with UV texture coordinates and vertex normals, guaranteeing 3D pose visualization in Stage 14 Three.js viewport even when restricted FLAME weights are not installed.
   - 2D Thin-Plate-Spline (TPS) warper solving the biharmonic spline system $\begin{bmatrix} K + \lambda I & P \\ P^T & 0 \end{bmatrix} \begin{bmatrix} W \\ A \end{bmatrix} = \begin{bmatrix} Y \\ 0 \end{bmatrix}$ with backward mapping for artifact-free image interpolation.
   - Piecewise affine Delaunay triangle warper for fast local deformation.

## License-Gated FLAME Fitter & UV Synthesis

1. **Staged Parametric Fitting (`roop_3d/reconstruction/flame_fitter.py`)**:
   - Strictly enforces Stage 03 `flame-2020` model registry policy: manual license acceptance is required, and commercial use is denied. Unlicensed or blocked activation attempts raise `DomainError(..., 403)`.
   - Staged parametric optimization: initializes rigid pose with PnP, optimizes shape coefficients $\beta$ with $L_2$ regularizer $w_{shape} \|\beta\|_2^2$, optimizes expression $\psi$ and jaw rotation $\theta_{jaw}$ with $w_{expr} \|\psi\|_2^2$, and jointly fine-tunes with Huber or Geman-McClure robust landmark loss.
   - Clean-room `SyntheticFLAMEModel` enables complete verification of non-linear parametric fitting and parameter convergence without requiring proprietary MPI assets.

2. **Visibility-Weighted UV Synthesis (`roop_3d/reconstruction/uv_synthesizer.py`)**:
   - Rasterizes UV space triangles to map texels to 3D surface points and normals.
   - Projects multi-angle face views weighted by surface normal dot-product with viewing direction, image sharpness, and optional occlusion masks.
   - Explicitly tracks per-texel confidence mask:
     - `OBSERVED` (2): directly visible by camera with positive normal dot-product ($n_{cam, z} > 0.05$).
     - `INTERPOLATED` (1): unobserved face geometry holes smoothly inpainted from adjacent observed texels.
     - `UNSEEN` (0): back-of-head, ears, or occluded background regions. These pixels are strictly set to zero and never blended into skin texture.

3. **Validated Mesh Export (`roop_3d/reconstruction/mesh_io.py`)**:
   - Atomically exports `.obj` + `.mtl` + `.png` material packages to `roop_3d/outputs/meshes/`.
   - Atomically exports self-contained binary glTF 2.0 (`.glb`) containers with 4-byte chunk padding, vertex positions, normals, UVs, indices, and embedded PNG textures ready for Three.js `GLTFLoader`.

## Files Changed

- `app/roop_3d/reconstruction/geometry_types.py`
- `app/roop_3d/reconstruction/tps_fallback.py`
- `app/roop_3d/reconstruction/flame_fitter.py`
- `app/roop_3d/reconstruction/uv_synthesizer.py`
- `app/roop_3d/reconstruction/mesh_io.py`
- `app/roop_3d/reconstruction/__init__.py`
- `app/roop_3d/tests/test_stage08_geometry.py`
- `app/roop_3d/tests/test_stage08_tps.py`
- `app/roop_3d/tests/test_stage08_flame.py`
- `app/roop_3d/tests/test_stage08_uv.py`
- `app/roop_3d/tests/test_stage08_mesh_io.py`
- `app/roop_3d/audit/STAGE08_REPORT.md`
- `app/roop_3d/audit/STAGE08_VALIDATION.json`
