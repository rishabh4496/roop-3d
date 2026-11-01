# Stage 5: 3D pose, Stage 0 pose matrix, frontalization, TPS expression, "Use Face From Frame"

As in earlier stages, the prompt's `app/api.py` / `react-ui/` are `app/roop_3d/server/` and `app/roop_3d_ui/`.

## Pose: one structure, one estimator, two frames

* `core/pose.py` is the single home for head pose. `Pose3D(yaw, pitch, roll)` is the public structure
  (degrees; yaw + = nose toward image right; **pitch + = looking up**; roll = in-plane tilt).
* `estimate_pose_5pt(landmarks5)` fits the canonical 3D five-point head model to the ArcFace landmarks
  (scaled-orthographic least squares, orthonormalised, decomposed). It replaces three inconsistent
  estimators (yaw scale x110 in the tracker, x55 in the FaceSet reader, a third in the media service).
  It is uncalibrated (no intrinsics): a few degrees of error for frontal to three-quarter views, and it
  returns the neutral pose, never an exception, for degenerate input.
* **Why two frames.** Stored data (`FaceView.pitch`, archive `pose` lists, `classify_pose_bin`, an existing
  test) uses the *legacy* frame where **pitch + = looking down**. The prompt's matrix (Down < -10, Up > +10)
  is the standard frame. Rather than rewrite stored FaceSets, `as_pose3d` is the one place they meet:
  raw tuples/dicts are read as legacy and converted; `Pose3D` is standard and passes through unchanged.
  `Pose3D.legacy_tuple()` goes the other way. API responses (`faces[].pitch` in the Stage 4 preview
  metadata, `extract_from_frame`) report the **standard** frame.
* **Tuple sanitizer.** `as_pose3d` accepts 2-tuples `(yaw, pitch)`, 3-tuples, lists, numpy arrays, dicts
  (roll optional), `Pose3D`, and attribute objects (including the calibrated rigid pose in
  `reconstruction.geometry_types`, which is a *different* class that happens to share the name). Out-of-range
  angles are clamped/wrapped; wrong lengths and non-finite values raise `DomainError(POSE_INVALID)` (422),
  never a `TypeError`. It is applied in the PoseSelector, the identity swapper, the FaceSet reader (archive
  `pose` entries), the media inspector and the preview readouts.

## Stage 0: the 3x3 pose matrix (`ROOP_STAGE0`)

`faceset/pose_matrix.py`. Cells: yaw Left < -15 deg, Center [-15, 15], Right > 15 and pitch Down < -10,
Level [-10, 10], Up > 10 (centre bands inclusive); roll is carried per cell as the third alignment axis.

* **Pre-flight** (`preflight`, run in `IdentitySwapper.prepare_session` / `set_track_source`, so before any
  render or preview): every view is binned into a cell and each cell's 512-D ArcFace vector is computed once
  (quality-weighted mean), made read-only and fingerprinted (`verify()` detects tampering; the result is
  memoised so repeated previews don't rebuild it). Empty cells are filled from the nearest populated cell and
  flagged `filled_from` with reduced confidence; no identity is ever synthesised.
* **Render time**: `reference_embedding` maps the target pose to its cell in O(1) (`[Stage0] lookup ...` is
  logged at INFO the first time a cell is hit per track, DEBUG after) instead of scanning every view.
* **Identity mixer**: with `identity_blend` the same Slerp/Linear + tangent-dial recipe is applied *per cell*
  across the sources' frozen cell vectors, and the result is stored on the blended document.
* **Identity guard, 0.80**: implemented as a maximum *cosine distance* (1 - cos) of 0.80. For one FaceSet a
  view farther than that from the canonical identity is excluded (and a set with no view inside is rejected,
  `IDENTITY_GUARD_VIOLATION`). For a blend, each cell's vector must be within 0.80 of the dominant source's
  vector in that cell; tangent dials are scaled back (1, 1/2, 1/4, ... 0) until it is, and a blend that fails
  even without dials is rejected. *Interpretation note:* the prompt says "0.80 cosine distance" which could
  also mean similarity >= 0.80; distance <= 0.80 (similarity >= 0.20) is the literal reading and is a loose
  "same person" bound. The constant is `GUARD_MAX_COSINE_DISTANCE` if you want it tighter.
* **Fallback** to the existing per-frame `PoseSelector` when `ROOP_STAGE0=0` (also `false/off/no`), or for
  portfolios with no usable pose spread (several views all at exactly 0/0, or `metadata.custom_angles`).
  Default is **on** (`ROOP_STAGE0` unset behaves as 1).
* **Archives**: `export_v2` now writes each view's embedding as its own small `<n>.embedding.json` entry (a
  project-written `.fsz` previously could not be re-read without running the recognizer model) and a
  `pose_matrix` index (cell -> member views). Readers ignore the index and recompute, so a stale index cannot
  change a render. `metadata.json` is now written compactly: the old pretty-printed form was so repetitive that
  a project-written archive of roughly 35+ views failed the validator's zip-bomb ratio check when read back
  (a latent bug this stage's tests exposed).

## 3D pose frontalization (`pose_frontalization`, 0..1)

`reconstruction/frontalization.py`, applied in `IdentitySwapper.infer_frame` on the aligned 128-px crop
*before* `inswapper_128`. The estimated yaw/pitch are reduced by the weight; the five landmarks are moved to
where the canonical head would put them at that reduced pose (same scale and centre); a thin-plate-spline
warp (image border pinned) carries the crop there, and the swapper's output is warped back with the inverse
correspondence so the existing alignment/compositor are unchanged. Per-point motion is capped (22% of the crop)
so the spline cannot fold. It is a landmark-driven 2D warp on an estimated pose, not reconstruction of unseen
surface; the Stage 4 `frontalization_weight` (post-swap 2.5D to 3D mesh blend) is a different control.

## TPS expression warping (`tps_expression_weight`, 0..1)

`reconstruction/expression_tps.py`. Control points: the face contour (0-32), the nose block (72-86) and a
dense ring on the image border map to themselves; mouth (52-71), both eyes (33-42, 87-96) and both brows
(43-51, 97-105) move toward the target frame's positions by `weight`. The spline interpolates exactly
(regularisation 1e-9), so contour drift is ~1e-5 px; per-point shift is capped at 30% of face width.

The 106-point block layout is the one `landmark_crop` already relies on for eyes and mouth; the project
deliberately does not trust a detector's ordering blindly, so the warp first runs `layout_is_plausible`
(brows above eyes above nose above mouth, in the face's own rotated frame) and does nothing otherwise
(`LAYOUT_UNVERIFIED`). In `ReferencePipeline.infer_frame` the generated crop's own 106 landmarks are detected
(`DenseLandmarks.infer106`) and warped toward the target frame's landmarks; a detection failure leaves the
crop untouched instead of failing the frame.

Both controls are `WorkstationParameters` fields (only serialised when set, so existing request fingerprints
are unchanged), accepted by `POST /api/preview` (`pose_frontalization`, `tps_expression_weight`) and passed to
`ReferencePipeline.infer_frame(frontalization=, tps_expression=)` only when above zero.

## `POST /api/faces/extract_from_frame`

Body: `{target_media_path, frame_index, face_index, destination, faceset_name}`. The target may be a video
(decoded through the media service, frame index 0-9999 like `/media/preview-frame`) or a still image in the
runtime media folder. `face_index` counts the frame's detected faces left to right by box x1. The response's
`face` holds the 512-D embedding, `Pose3D`, `pose_cell`, landmarks, box and thumbnail; the thumbnail and
metadata are also stored under `roop_3d/.runtime/extracted_faces/`.

| `destination` | Effect |
|---|---|
| `source_gallery` | New single-view FaceSet (512-px aligned crop PNG) in `outputs/facesets/` and the library; `active_source` names it. The UI's **Use as source** button selects it as the active source face. |
| `active_faceset` | `faceset_name` is the FaceSet id or its (case-insensitive) name. The view is inserted at the next index, canonical/medoid/pose bins are recomputed, the archive is replaced atomically, Stage 0's memoised matrix is invalidated and rebuilt, and the response includes `inserted {view_index, pose_cell}` and the new `pose_matrix` summary. A face farther than the 0.80 guard from the FaceSet's identity is refused (422) and the archive is not touched. |
| `target_cluster` | Pins a centroid in `roop_3d/.runtime/clusters/<target sha>.json`. A face within cosine similarity 0.60 of an existing pin refines that centroid (running mean) and keeps its rank; otherwise it gets the next rank. Ranks are never reused or renumbered. `GET /api/faces/target_clusters?target_media_path=` lists pins by rank. |

```bash
curl -s -X POST http://127.0.0.1:8030/api/faces/extract_from_frame -H 'Content-Type: application/json' -d '{
  "target_media_path": "roop_3d/.runtime/media/target.mp4", "frame_index": 120, "face_index": 0,
  "destination": "source_gallery", "faceset_name": "Alice" }'
```

```python
import httpx
r = httpx.post("http://127.0.0.1:8030/api/faces/extract_from_frame", json={
    "target_media_path": "roop_3d/.runtime/media/target.mp4", "frame_index": 120, "face_index": 0,
    "destination": "active_faceset", "faceset_name": "Alice"})
print(r.json()["inserted"], r.json()["pose_matrix"]["populated"])
```

```javascript
const r = await fetch('/api/faces/extract_from_frame', { method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ target_media_path, frame_index, face_index: 0, destination: 'source_gallery' }) });
const { active_source, face } = await r.json();   // then POST /api/preview with source_face_ids: [active_source.faceset_id]
```

## Verification

`app/roop_3d/tests/test_remediation_stage5.py` and `app/roop_3d_ui/src/faceFromFrame.test.tsx`. Detection,
recognition and swap inference are deterministic fixtures, so these verify geometry, contracts and plumbing,
not identity fidelity or visual quality on real weights, and the pipeline-level TPS/frontalization hooks have
not been run against the real 106-landmark and `inswapper_128` models.
