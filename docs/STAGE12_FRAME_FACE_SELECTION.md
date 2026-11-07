# Stage 12 - accurate face selection from any frame, rank-preserved target locking

Verify: `cd app && roop_3d/.venv/Scripts/python.exe ../verify_stage12_frame_face_selection.py` (exit 0 = pass; builds a throwaway
video from real photos of two people under `.runtime/media` and deletes it).
Tests: `app/roop_3d/tests/test_stage12_target_lock.py`, UI `stage12Coords.test.ts`, `stage12TargetLock.test.tsx`.

## Path mapping (the prompt names files that do not exist)

| Prompt | Real |
|---|---|
| `app/api.py` | `app/roop_3d/server/media_frames.py` (`/media/frame`), new `server/target_lock.py` |
| `/api/v1/media/inspect` | `/api/v1/media/inspect-faces` is the whole-video tracker (unchanged); per-frame detail is `/api/v1/media/faces` and `/media/frame?detect_faces=true&format=json` |
| `TargetIdentityLock` | `app/roop_3d/identity/target_lock.py` (separate from the hash-bound `/identity/locks`, which assigns FaceSets) |

## What it does

* **Adaptive detection** (`media/face_analysis.py`): one SCRFD pass at 0.30; >= 0.50 is `primary`; 0.30-0.50 is kept as
  `profile_tolerant` only if the five landmarks are geometrically plausible AND >= 4/5 agree with the independent 106-point model.
* **Per face**: `face_id`, `track_id`, `cluster_id`, native-pixel `bbox` (3 decimals) + `bbox_norm`, `kps_5`, `kps_68_3d`
  (null + `kps_68_3d_status` when the model is missing or inconsistent), pose, 512-D `embedding`, `crop_thumbnail_base64`.
  JPEG responses keep the heavy fields out of the `X-Frame-Meta` header (`format=json` / `/media/faces` return everything).
* **Rank-preserved clusters**: one cluster id per person per video; ids never come from left-to-right order (that is `face_index`).
* **Reference bank**: `POST /target/lock-from-frame` anchors the clicked face and samples keyframes outward from the anchor, linking
  by chained appearance so 0 -> 30 -> 65 deg accumulates; `POST /target/add-angle-sample` adds a user angle (refused below cosine
  0.30 unless `force`); `GET /target/{id}`, `/target/{id}/match?frame_index=`, `GET /target?path=`.
* **UI**: sub-pixel `canvasToVideoCoords` / `videoToCanvasCoords` (contain letterbox + zoom + pan); pose pill
  `#1 · Y: -34.2° P: +6.1° R: -2.0°` and nose-tip pose axes; left-click locks the person (`TARGET LOCKED ✓ Target #1`; the 3D viewer
  already follows the selected face's live pose); right-click menu adds Lock as Primary Target / Add This Angle / Bind Source / Exclude.

## Measured, and what it means

ArcFace cosine on the real roop-ultimate facesets: same person < 20 deg apart 0.81+, 50-90 deg apart 0.49-0.61; different people
average 0.13, p99 0.37, **max 0.48**. So there is no single threshold: association accepts >= 0.55, or >= 0.40 when it beats every
alternative (never taken below 0.30) by 0.10. With a frontal-only anchor the profile (cos 0.53) is still bridged in the verify run.

## Limits (read these)

* **Profile-vs-profile of two different people can exceed the weak threshold.** The margin rule protects frames where both appear;
  a lone look-alike profile with no competitor can still merge. Fragmenting is preferred to merging.
* `track_id` for a single-frame read is the person's cluster id (`track_source: appearance_cluster`); there is no temporal tracker
  on a random-access read. Keyframe-scan trajectories are in `observed_frames`.
* Pose is still the SCRFD 5-point estimate (`pose_source`), poor on true profiles; the 68-point PnP is not used (flip ambiguity).
* `kps_68_3d` is rejected on steep profiles (consistency gate), so it is null there.
* Locks and clusters are in memory for the server process; "Exclude" and "Bind source" reuse the existing track/role store.
* The verify video repeats four stills, so it proves association and geometry, not tracking through real motion blur or occlusion.
* Not checked visually in a browser; UI is covered by unit tests and a clean build.
