# Stage 11 - 3D head from `.fsz` facesets and photos (MakeHuman CC0 template)

Verify: `cd app && roop_3d/.venv/Scripts/python.exe ../verify_stage11_3d_faceset_mesh.py` (exit 0 = pass).
Unit tests: `app/roop_3d/tests/test_stage11_faceset_mesh.py`.

## Path mapping (the prompt names files that do not exist)

| Prompt | Real |
|---|---|
| `roop_3d/faceset_loader.py` | `app/roop_3d/faceset_loader.py` (new) |
| `/api/v1/faceset/3d-state` | `app/roop_3d/server/faceset_3d.py` (existing since Stage 3; extended) |
| MakeHuman deformation | `reconstruction/rbf_deform.py`, `makehuman_landmarks.py`, `identity_asset.py` |
| UV stitching | `reconstruction/uv_synthesizer.py` (`blend_mode="multiband"`), `uv_completion.py` |

## What was already true, and what this stage added

Stage 3 already fitted the MakeHuman head from 5 landmarks (`IdentityAssetSettings(template="makehuman")`): the template moved
only ~0.6 mm. Added:

1. **Loader** - per sample: normalised ArcFace embedding, 68-pt 3D (`1k3d68`) and 106-pt (`2d106det`) landmarks, Euler pose,
   aligned 512x512 RGB crop, 96-bin sphere bin and 3x3 (yaw x pitch) cell with roll as the third axis. Both roop-ultimate
   `legacy_v1/v2` and roop-3d `v2_standard` archives go through `FaceSetReader`. Models are searched in
   `ROOP3D_LANDMARK_MODEL_DIR`, `.runtime/models`, and the sibling roop-ultimate `models/buffalo_l`; without them samples keep
   the archive's 5 points and say so (`landmark_source`).
2. **68-pt correspondence on MakeHuman** - *derived*, not annotated: canonical 5-pt anchors registered onto the template's
   five landmark vertices (rms ~4.7 mm), each canonical landmark snapped to the nearest viewer-facing vertex (~3.8 mm mean).
3. **Deformation** - Wendland-RBF interpolation of landmark displacements, Laplacian smoothing of the field on the
   position-welded graph, *local* fold relief (only folded triangles are attenuated), seam-welded normals. On harjot: mean 2.3 mm,
   p95 5.8 mm, max 18.5 mm from the neutral template (was 0.6 mm).
4. **Texture** - Laplacian-pyramid (multiband) blend of per-view layers; black letterbox padding in photos is masked out;
   display atlas completed by bilateral symmetry projection then a harmonic fill that relaxes to the subject's skin tone.
   Output is a 512x512 RGBA PNG (alpha = mesh footprint). The asset's own atlas stays observation-only.
5. **API** - `/faceset/reconstruct-3d` and `/faceset/3d-state` payload adds `faceset_samples`, `mesh_full_3d.deformation`,
   `mesh_full_3d.texture_completion`, `texture_size`.
6. **Viewer** - uses backend welded normals (no UV-seam creases). Tabs, wireframe, observed views, sphere coverage and
   glowing markers already existed from Stage 4.

## Deviations / limits (read these)

- **Pose**: PnP against the canonical 68-pt model flips (pitch -177 deg on a frontal photo). Pose is computed from the five
  points derived from the dense set, as everywhere else.
- **Profile views** (|yaw| ~ 79 deg): `1k3d68` is rejected by a consistency gate there. With MakeHuman, geometry is fitted from
  the dense views and profile views are registered against the fitted head from 5 points: they add texture, not shape.
  So profile-driven nose-tip/chin/forehead depth is *not* achieved on such sets.
- **106-pt**: computed and returned, but no 106 -> template correspondence exists (layout unverified), so it is not used in the fit.
- **Frontal/profile weighting** is the existing MAP bundle adjustment's per-landmark visibility weighting, not a hand-built
  per-feature scheme.
- Completed texels (symmetry / harmonic) are generated for display; they are not observations and are reported as such.
- `test_identity_asset_makehuman.py` was updated: it pinned "MakeHuman has no 68-pt correspondence", which is no longer true.
