# Stage 3 — persistent cross-frame tracking and single-ID locking

One person keeps one target ID (`Face 1`) from the first to the last frame. Three layers, each independently testable.

## 1. Online two-tier tracker (`engine/detection/face_tracker.py`, `motion.py`, `track_policy.py`)

`FaceTracker(policy=TrackPolicy())` associates every frame in two ByteTrack tiers:

| Tier | Detections | Matched against | Evidence | Effect |
|---|---|---|---|---|
| 1 | score >= detection threshold | every live track, including tracks not detected for up to `spatial_bridge_frames` | `0.6 cosine + 0.25 overlap/GIoU + 0.15 landmark layout` | may start a track, feeds the appearance bank |
| 2 | low score (>= 0.1) | tracks tier 1 left over | IoU with the Kalman-predicted box only | keeps the ID alive through occlusion, never swappable, never starts a track |

Assignment is Hungarian (`scipy.optimize.linear_sum_assignment`; only strictly positive scores are admissible).

* **Kalman predictors.** `BoxMotion` filters `(cx, cy, log size, log aspect)`; `CenterMotion3D` filters the metric head centre
  of the 6-DoF pose. Time is the real *frame number* (`FaceTracker.frame_number`), so sparse adaptive sampling predicts across
  its gaps. A missed track coasts on a decaying velocity (or an optical-flow measurement). The 3D-centre Mahalanobis distance
  vetoes an IoU-only link whose metric position jumped.
* **Appearance bank.** Per track: an EMA gallery (updated only on strong, unambiguous matches), diverse anchors, and one
  prototype per 30 deg yaw bin. A pose-binned prototype speaks only for observations at its own yaw (profile embeddings of
  different people resemble each other far more than frontal ones).
* **Head turns.** `PoseTrajectory` flags a steep (>= `turn_yaw`) or sharply changing (>= `turn_step`) yaw, or a trend heading
  there. For such frames the continuity cosine floor drops by `turn_relief` as long as the Kalman-predicted box is continuous,
  and the profile is learned as its own prototype. Without a turn the old floor applies, so a destroyed embedding is still
  withheld.
* A track missing for more than `bridge_frames` (default 90) is parked for strict appearance-only reacquisition.

## 2. Global tracklet consolidation (`identity/consolidation.py`)

Run once over the finished tracklets (`MediaService.inspect` does this after the scan):

1. **Artifacts / noise get no ID**: fewer than `min_identity_frames` detections, or boxes mostly (`artifact_containment`) inside
   a much larger (`artifact_area_ratio`) co-occurring face (an ear, a jaw).
2. **Cannot-link**: tracklets seen on the same frame; tracklets whose well supported frontal prototypes disagree
   (`veto_cosine`).
3. **Links**: pose-matched prototype cosine >= `merge_cosine` (appearance); or continuity (gap <= `continuity_gap_frames`,
   plausible shift/scale, no third face nearby) with cosine >= `merge_weak_cosine`, or >= `merge_turn_cosine` when one side
   meets the other in a steep turn.
4. Strongest links first under union-find, never joining co-occurring clusters.
5. **Canonical IDs** `1, 2, ...` by descending trusted-observation count, then first frame: the main subject is `Face 1`; a
   second ID exists only for a cluster that is genuinely distinct and supported.

The consolidated pose-aware prototypes replace the per-frame learned signature in the identity report, so the render-time
`IdentityGuard` recognises the same person at every angle (its floor is also lowered for steep heads; rivals still veto).

Calibration (`tools.verify_stage3 --calibrate`): pose-matched cosine between *different* people has p50 0.03, p95 0.15-0.22,
p99 <= 0.25 in every yaw bin; the same person scores 0.96 frontal and 0.45 (median) against the frontal prototype in profile.

## 3. Locked target binding (`identity/locked_target.py`)

The render replays a tracker whose IDs are not the canonical ones. `LockedTargetBinder` re-labels each frame's faces:

1. **Pre-pass table** — `ReferencePipeline.scan_locked_targets` (and the existing angle scan) track the whole timeline once;
   the binder consolidates those tracklets globally, matches each identity to a locked signature, and stores frame -> box ->
   canonical ID. The replay looks each face up by frame and box (IoU >= 0.5).
2. **Sticky** raw-ID binding while the face still resembles that person or sits where that person last was.
3. **Embedding fallback** — Hungarian on cosine against the signature, with continuity/turn relief and an ambiguity margin.

A face no canonical ID claims keeps its tracker ID shifted by `UNBOUND_BASE` (never selectable). The runners pass the binder via
`tracking_options()` (built from the inspection's identity report); with a binder the clustering registry no longer renumbers
tracks.

## Verification

```bat
roop_3d/.venv/Scripts/python.exe -B -m roop_3d.tools.stage3_scan      :: per-frame detections -> .runtime/stage3_cache (once)
roop_3d/.venv/Scripts/python.exe -B -m roop_3d.tools.verify_stage3    :: replay + consolidation + binding gate
roop_3d/.venv/Scripts/python.exe -B -m roop_3d.tools.verify_stage3 --live   :: full production stack on every frame
```

Receipt: `.runtime/stage3_verification.json`. Tests: `tests/test_persistent_tracking.py`, `tests/test_identity_tracker_policy.py`.
