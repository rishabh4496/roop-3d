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

## Target Face Bank UI (`#/faceswap`)

The bar above the timeline is the **Target Face Bank**. Every person of the frame on screen
is a card with a filled, colour-coded `FACE n` pill (`n` is the immutable
`persistent_face_id`; the same colour is used for the box and pill on the canvas, the
bank card, the angle grid and the "TARGET LOCKED" chip). Several faces can be locked at once.

- **Click a face** (its canvas box or its card) to toggle its Target Lock: an unlocked face
  is locked and its angle harvest starts, a locked face is unlocked (`DELETE /target/{id}`).
  Unlocking drops the lock only; the `face n` slot and its bank stay, and locking the same
  person again attaches a new lock to the same slot. A source binding made with the card's
  **Source** select (a FaceSet, `.fsz` or image-built) is kept per track and survives an
  unlock. The select is enabled only for a locked face, because the swap is bound to the
  face's track, which locking creates.
- **Capture HUD** (inside the card of every locked face): progress ring, the 5 x 3 grid (yaw
  columns L profile, L quarter, frontal, R quarter, R profile; pitch rows up, level, down;
  "L/R" is the image side), each cell filled with the 256 px crop the harvester kept for it
  (`GET /target/{id}/angle/{yaw}/{pitch}/crop.jpg`). It polls every 1.5 s while the scan runs,
  and a cell pops in again when a better face replaces its crop. The green
  **Angles Locked (100% Coverage)** badge appears only when 15/15 cells are filled. A video
  that never shows a pose ends as `N/15 angles locked`, with the scan's reason (the whole video
  was scanned, or frames were unreadable, or the scan stopped early); coverage is never rounded up.
- **Re-Scan All Angles** (`POST /target/{id}/rescan`) drops the automatically harvested
  references and harvests the video again. The anchor and angles added by hand are kept, so a
  rescan can never leave a lock without a reference; it is refused with 409 while a scan runs.
  A scan stops by itself when its target is unlocked.
- **Hover (or keyboard-focus) a locked card** to list every captured crop with its frame, pose and
  measured similarity to the reference bank. This is evidence, not a guarantee: crops are accepted
  into `face n` above the 0.42 gate described above, and the anchor/user-added crops are labelled.
  Locked faces that are not in the current frame stay in the bank (marked "Not in this frame").

## Engine status bar (`#/faceswap`, `#/facemgr`)

`GET /api/pipeline/status` (alias of `/api/v1/pipeline/status`) is what the bar under the top bar
shows: `active` (the detector, swapper and enhancer session each stage has loaded now),
`selected` (what the next preview or render will load), `precision` (derived from those sessions'
real provider and precision policy: `TensorRT FP16 Mixed` is only reported for TensorRT built
with the `fp16` policy; `auto` is reported as `(auto precision)`), `failures` and `pipeline`
(the same frame as `/ws/v1/pipeline/status`). The enhancer shows `• Verified` when the loaded
weight file has a SHA-256 on record, `• Unverified` otherwise.

A model that cannot be loaded is recorded with the backend's exact `code` and `reason`
(`engine/active_models.switch_guard`, used by the detector, swapper and enhancer builders), the
previous detector session is dropped instead of being kept silently, and the UI raises one error
toast per failure with that reason and the model still in use. A failure is cleared when a working
session of the stage is built or the loaded one serves a request.

Tests: `roop_3d/tests/test_target_ui_endpoints.py`, `src/faceBank.test.tsx`,
`src/engineStatusBar.test.tsx`, `e2e/faceBank.e2e.ts` (real Chrome, REST mocked).
