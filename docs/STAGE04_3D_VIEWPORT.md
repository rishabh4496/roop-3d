# Stage 4: visible 3D FaceSet reconstruction in the viewport

The Inspector's **3D settings** panel no longer shows a placeholder crystal. It renders your FaceSet (or the faces
extracted from the photos you import) as a textured 3D head, with the pose-coverage sphere around it.

Path mapping, as in earlier stages: the prompt's "frontend directory" is `app/roop_3d_ui`, the viewport component is
`src/components/three/PoseGeometryViewer.jsx`, and the Stage 3 payload comes from `app/roop_3d/server/faceset_3d.py`.

## What you see

| State | Viewport |
|---|---|
| No source chosen | A faint wireframe reference head, the orbital rings, and the line *Load a source face or .fsz faceset to preview 3D reconstruction*. Telemetry reads 0 views, 0 / 96 (0%), `+0.0° Y · +0.0° P`. |
| Reconstructing | *Reconstructing 3D face from the FaceSet…* over the same reference head. |
| Failed | The server's error message and a **Retry** button (nothing is dropped silently into a banner). |
| Ready, **Estimated 2.5D** tab | `mesh_2_5d` with the FaceSet's own photo as texture, a subtle cyan wireframe, and the 68 template landmark points. |
| Ready, **Full 3D Reconstructed** tab | `mesh_full_3d` (head and neck) with the stitched multi-view albedo `uv_texture_data_url`, PBR lighting. Wireframe is off by default and can be toggled. |

Both tabs show the geodesic rings (great circles plus latitude rings every 30° of pitch), the 96-cell coverage lattice
(slate = unobserved, amber = 1–2 views, green = 3 or more) and a **glowing cyan dot at the exact `position_xyz` of every
`observed_pose_points` entry**. The X / Y / Z gizmo is drawn from the camera, so it always matches the orbit.

* **Drag** rotates the head like a trackball (dragging right turns it right), **scroll** zooms, there is no pan.
* **Reset Camera** returns to the frontal view, `+0.0° Y · +0.0° P`.
* **Telemetry bar**: *Observed Views* = `observed_views`; *Sphere Coverage* = `sphere_coverage_count / sphere_coverage_total (sphere_coverage_pct%)`;
  *Live Pose* = the pose of the selected target face (or track, or first target cluster), and while there is none, the pose of
  the head as you are currently viewing it. A white ring on the sphere marks that current pose: when it sits on a cyan dot you are
  looking at the head from an angle your FaceSet actually covers. When a target pose exists, the head itself is turned to it and
  a pink dot marks it on the sphere.

## Render mode binding

| Render mode (saved value) | Viewer tab it selects | Tab badge |
|---|---|---|
| Standard identity swap (`standard`) | Estimated 2.5D | *preview* (the standard path uses no 3D geometry) |
| 2.5D Pose-Aware Swap (`pose_3d`) | Estimated 2.5D | *render* |
| Full 3D Volumetric Re-enactment (`reenactment_3d`) | Full 3D Reconstructed | *render* |

Changing the mode, loading a different source, or showing a built identity asset switches the tab; clicking a tab yourself is
a preview choice that sticks until one of those happens. The saved `render_mode` values did not change, only the labels
and hints (`src/components/workstation/renderModes.ts`).

## Data flow

1. Choosing a FaceSet (or importing a `.fsz` or photos) sets `sourceId` in `App.tsx`.
2. One `POST /api/v1/faceset/reconstruct-3d { faceset_id }` is sent per source. Its response is the Stage 3 payload and is bound
   to the viewer directly. The server caches it by archive hash, and selecting another target face or track no longer
   re-requests or re-parses it (Stage 3's effect re-posted on every selection).
3. `GET /faceset/3d-state` remains the way to re-read the cached payload; the UI does not need it after the POST.

## Defects fixed on the way

* **Wrong coordinate frame.** The payload meshes are in the backend *mesh frame* (+x image-right, **+y down**, +z toward the
  viewer). Stage 3 declared them `opencv` and the viewer flipped y *and* z, which turns the head upside-down and facing away.
  A new `mesh` frame flips y only and swaps triangle winding, so outward faces stay outward.
* **Pitch sign.** Track and detected-face poses are stored in the legacy frame (pitch + = down) but coverage points and the
  sphere use the standard frame (pitch + = up). The viewer converts target poses once (`legacyPoseToStandard`). The head
  rotation (`poseQuaternion`) is now defined so its nose points exactly at the matching sphere dot (unit-tested).
* **Target frames counted as source coverage.** With no coverage payload the old viewer filled *Observed Views* and the lattice
  from detected faces in the target video. The telemetry now describes the source FaceSet only.

## Limits (not hidden)

* The 68 landmarks are the first 68 vertices of the canonical template mesh (`identity_template`), carried onto the fitted
  face. The 106-point layout has no 3D template in the backend, so it is not drawn.
* A single photo gives an anthropometric depth prior, not measured depth. The viewer prints the payload's
  `reconstruction_level` under the canvas, and its first limitation as a tooltip.
* The back of the head and ears are the MakeHuman template prior where no view saw them.

## Verification

* `npm run build` (tsc, eslint, vite): 0 errors.
* `npx vitest run`: 390 tests, 17 new in `src/faceset3dViewer.test.tsx` (frame conversion, winding, pose maths, telemetry,
  idle / loading / error, render-mode tab binding, glow dots).
* Real run: built `dist`, served by the real backend, real Chrome (SwiftShader WebGL), real local FaceSets from `.runtime`.
