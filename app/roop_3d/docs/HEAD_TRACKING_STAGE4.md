# Head and gaze tracking (Stage 4, tracking series)

Path mapping: the prompt's `app/roop_3d/tracking/` is real and new. The `/ws/v1/pipeline/status` socket already existed
(`server/ws.py`, `server/pipeline_status.py`). There is no FLAME encoder, gaze model or MediaPipe install on this machine, so
everything below is built on what IS installed (5- and 106-point landmarks from the existing detector) and takes the rest as
optional evidence. Unmeasured quantities are `None`, never placeholders.

## What exists

| Module | Role |
| --- | --- |
| `tracking/video_tracker.py` | `VideoHeadTracker3D`: per-frame pose, shape lock, jaw, asymmetric expression, gaze, SH lighting, shot handling |
| `tracking/filters.py` | One-Euro + Kalman that cannot overshoot; SO(3) rotation filter |
| `tracking/identity_lock.py` | per-shot weighted-median shape lock, outlier rejection, identity break |
| `tracking/gaze.py` | uncalibrated iris-geometry gaze from the 478-point mesh; vergence |
| `tracking/lighting.py` | per-frame second-order SH on the generic head, camera frame |
| `tracking/scene_adaptation.py` | `SceneTrackerAdaptation`: 16 frames, 3 s budget, self-supervised |
| `tracking/runner.py` | `calibrate_video`, `track_video`, `HeadTrackingTap`, adapters over `VideoFrameService` |
| `tracking/telemetry.py` | throttled `tracking.telemetry` payloads (no server imports) |
| `server/ws.py`, `server/pipeline_status.py` | `TrackingTelemetryEvent`; opt-in delivery on `/ws/v1/pipeline/status` |

## Per frame

* **Camera (6 DoF)**: yaw/pitch/roll + scale + 2-D translation of a weak-perspective camera from the 5 landmarks
  (`solve_view_pose`, ~1.5 ms). Depth is the scale, not metric. A supplied FLAME rotation is deliberately ignored: its axis
  convention cannot be verified without weights.
* **beta_target**: locked per shot by weighted-median pooling. FLAME beta if an encoder supplies it, otherwise five (106-point) or
  two (5-point) *facial proportions*, reported as `landmark_proportions_*`, never as FLAME. Ratios involving the nose tip are
  excluded: they move ~50 % over +/-20 deg of pitch.
* **theta_jaw**: FLAME jaw opening if given, else lip drop / 0.085 m lever arm (`jaw_source` says which).
* **psi**: eight asymmetric channels (mouth open/width, smile a/b, blink a/b, brow a/b) against the track's own neutral; FLAME psi
  passes through when supplied.
* **theta_eyes**: per-eye yaw/pitch. `calibrated_estimator` when an `EyeGaze` is supplied, else `iris_geometry_uncalibrated`,
  else `None`. Eyeball torsion is not observable, so the "6 DoF" is two angles per eye. Vergence uses the exact ray intersection
  with a population IPD (63 mm); parallel gaze reports no distance.
* **L_t**: nine RGB SH coefficients in the camera frame, fitted at the projected skin vertices of the generic head every
  `lighting_stride` frames, held between fits.

## Temporal behaviour

`TrackTemporalStabilizer.filter` overshoots (a 70 deg yaw step peaks at 108 deg, a 60 deg turn that stops at 87 deg), so it is not
used. `SmoothedChannel` confines each Kalman step to the segment between the previous output and the One-Euro output. Measured:
no overshoot, no extra lag, and **only <=10 % extra rest smoothing**; its real contribution is bridging dropped frames
(`coast`, <= 0.25 s, decaying velocity). Pose rotation is a pure SO(3) One-Euro (cannot overshoot, no Euler wrap).

A shot boundary resets filters, identity lock and expression baseline. Sources: the caller's `scene_cut`, a 32x32 image
difference/histogram rule (same as `FaceTracker`), a face that jumps > 2.5 inter-ocular distances or rotates > 2700 deg/s in one
frame, and `identity_break` (12 consecutive frames far from the pooled shape). 1400-2700 deg/s is rejected as a measurement error
and bridged. After > 1 s without a face the filters restart; the shot and its lock are kept.

## SceneTrackerAdaptation (SPARK-inspired)

SPARK fine-tunes a capture network; none is installed, so this pass personalises the parametric parts only, from 16 frames
sampled across the video, inside a 3 s budget shared with decoding and detection (`calibrate_video`):

1. the 5 3-D anchors (closed-form MAP shape step alternated with pose re-solves, ~0.3 s), accepted only if the *held-out* frames'
   landmark residual drops >= 2 %;
2. the actor's neutral expression (no warm-up afterwards);
3. a pooled lighting environment with a `lighting_static` verdict;
4. an identity prior the shot locks are compared with.

`AdaptationReport` carries before/after residuals, time, `budget_exhausted`, yaw span, how much depth was observed and warnings.
`SceneAdaptation.to_dict/from_dict` makes it storable with a job. Nothing changes any model's weights.

## Measured (synthetic performer with known truth; `tests/test_head_tracking_*.py`)

* Head 8 % narrower than the template: mean pose error 4.1 / 3.0 / 1.2 deg (yaw/pitch/roll), ~3.5 ms/frame. Head equal to the
  model: 0.13 / 0.12 / 0.06 deg.
* Adaptation on unseen frames: 4.07 / 2.99 / 1.18 -> 2.55 / 2.80 / 1.26 deg; held-out residual 1.05 -> 0.82 mm (floor 0.73 mm).
  Roll is not improved. On a head that already matches it adapts nothing (9/9 trials).
* The pose prior in `solve_view_pose` (default 8 deg) cost 0.3 deg on a matching head and made adaptation harmful; the tracker uses 45 deg.
* SH key-light direction under a mismatched head: 6-31 deg error (median ~15); the ambient term drifts up to 28 % with pose.

## Limits

* Pose accuracy is bounded by 5 landmarks against a generic or 5-point-personalised head; depth is only as observable as the yaw spread.
* Landmark proportions detect a *large* change of person, and only if the new face still fits the head model (a very different face
  biases the pose, which removes its frames from the frontal window). FLAME beta has no such limit.
* Gaze is `None` here unless a mesh478 or calibrated estimator is supplied; the iris estimate is uncalibrated and its vergence is noisy.
* Lighting has no cast shadows and uses generic normals.
* Multi-person video: `calibrate_video` adapts to the largest face unless `pick=` says otherwise.
* **Not wired into the render job runner.** Add `HeadTrackingTap(...).observe(image, tracked.faces, tracked.frame_time, tracked.scene_cut)`
  after `ReferencePipeline.track_frame`, and `calibrate_video(...)` before the main stage, when job-level options for them exist.

## Telemetry: `/ws/v1/pipeline/status`

Frames `{"type": "tracking.telemetry", ...}`: `job_id, track_id, state (tracking|ended), frame_index, timestamp_s, status
(tracked|coasting|lost), reason, confidence, yaw/pitch/roll_deg, jaw_open_deg, gaze_convergence_deg, gaze_convergence_distance_m,
gaze_source, expression_energy, shot_index, shot_cuts, shot_cut, cut_reason, identity_state/kind/frames, lighting_key, adapted`.
Steady frames are coalesced to <= 15 Hz; the first frame, cuts, status changes, lock changes and the end are never coalesced, and
counters are cumulative. They do not consume the global event sequence, are delivered only to this socket, and never evict job
events (a full queue drops the telemetry frame), so `/ws/events` and `/ws/telemetry` clients see no extra message and no gap.
A client connecting mid-run receives the latest frame right after the pipeline snapshot.

```python
publisher = app.state.pipeline.tracking_publisher(job_id=job_id)
tracker = VideoHeadTracker3D(adaptation=calibrate_video(source, extractor), on_frame=publisher,
                             lighting=MeshSHLightingEstimator())
track_video(source, extractor, tracker, publisher=publisher)
```
