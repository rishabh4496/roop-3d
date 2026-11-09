# Persistent target tracking and angular capture

Each media path has a process-local `FaceTrackRegistry` (also exported as
`TargetFaceRegistry` and the compatible `ClusterRegistry`). Internal cluster keys
remain one-based for existing clients; `persistent_face_id` and `face N` labels are
zero-based immutable discovery slots. `face_index` remains a frame-local selector
for extraction APIs and must never be displayed as a persistent identity.

Assignment uses SciPy Hungarian optimization, 0.70 appearance / 0.30 predicted-box
IoU costs, and a hard cosine gate of 0.42. Appearance compares normalized embeddings
against the pose bank, its centroid, and the immutable lock anchor. Unmatched
observations get new slots. Slots are never expired or reused. Motion uses a
constant-velocity Kalman filter; reverse seeks do not rewind its state and gaps
longer than 30 frames discard stale velocity. Render Stage 1 applies this registry
before smoothing, preserving person-specific landmark and rotation filters.

Selecting a target starts a background keyframe scan, then a sequential pass over
remaining frames. The scanner stops early when all 15 buckets pass quality 100,
or completes the pass and reports N/N observed video angles. A skipped frame
prevents an exhaustive-coverage claim. Normal playback/scrub face analysis also
updates the bank. Scanning is local and serial; it does not launch another GPU
model pool. `sample_count=0` explicitly disables the background scan.

The bank keeps the highest-quality measurement per 5 yaw x 3 pitch bucket. Bins are
half-open with the final outer endpoint included. Out-of-grid measurements are
retained as one additional appearance reference but do not claim grid coverage.
Each entry stores a bounded 256px crop, embedding, available 68-point 3D landmarks,
pose, frame, box, quality, and optional occlusion measurement. Quality is Laplacian
variance times detection confidence times (1 - measured occlusion). If occlusion
is unavailable, sharpness times confidence is used and the ratio remains null.
This is not a claim that the face is unoccluded. Optional missing landmark models
are likewise reported explicitly. Perspective 5-point PnP supplies rotation and
translation when its reprojection checks pass; camera intrinsics and metric head
size are estimates, not calibrated depth measurements.

Preview and export sessions share the selected video's registry. Stage 2 queries
the locked bank for the current pose; its measured dense landmarks can fill a
missing-landmark observation, transformed into the current box. Stage 3 receives
that observation for alignment and masks. Current-frame geometry, expression,
occlusion, and lighting measurements take precedence over older reference pixels.
The target bank does not replace the source FaceSet identity or freeze scene light.

No recognition threshold can guarantee identity across arbitrary occlusion,
lookalikes, or missing/incorrect embeddings. Observations below the strict gate
are kept separate. Coverage describes detected poses, and persistence lasts for
the server process; it is not a disk-restored identity database.

Regression tests: `roop_3d/tests/test_persistent_target_registry.py`, plus the Stage
12 target-lock/worker tests and UI target-lock/scrubbing tests.
