# Stage 1 — target identification and identity lock

Backend: `app/roop_3d/` · frontend: `app/roop_3d_ui/`. The prompt's "detection and tracking modules" are
`engine/detection/face_tracker.py` (the only tracker), `media/detector.py` (SCRFD) and `faceset/recognizer.py` (ArcFace).
Nothing parallel was created: the policy lives *inside* `FaceTracker`, and inspection, preview and render all replay it.

## What was already correct (kept)

* `ReferencePipeline.infer_frame` already skips tracks outside `selected_ids`, so an unselected face was never swapped.
* `JobRequest.face_mappings` is explicit; jobs validate the inspection's target SHA-256.

## What was defective or missing

| Gap | Consequence before |
|---|---|
| Tracker had no states or confidence; a weak link silently kept the ID, a failed link silently minted a new one | A wrong link swapped the wrong person with no signal |
| A scene cut cleared every track | The same person got a new ID after each cut |
| Track memory ended after `max_missed` (15) frames | A person who left and returned became a new person |
| Nothing recorded who may replace whom, bound to inputs | Mappings were request fields only; no staleness detection |
| The render replayed the tracker and *assumed* its IDs equal the inspection's | An ID drift would swap the wrong face |

## Tracker policy `identity_lock_v1` (opt-in)

`FaceTracker(policy=TrackPolicy())`; `policy=None` is the unchanged legacy path (existing saved inspections keep it).
`TrackedFace` gained `state`, `confidence`, `swap_allowed`, `reason` (defaults: always swappable, so legacy output is identical).

* Association: ArcFace cosine against a per-track **gallery + diverse anchors** (never the last frame, so a hand-corrupted frame
  cannot poison it), box IoU against a motion-predicted box, face-relative 5-point landmark similarity; global Hungarian assignment.
* Optical flow: a track that was not detected has its box carried forward with forward/backward-checked Lucas–Kanade flow
  (constant-velocity fallback when too few features survive). Used only as a spatial prior for re-detection.
* States: `TRACKED`, `UNCERTAIN`, `LOST`, `REACQUIRED`. A face is swappable only in `TRACKED` with confidence ≥ `min_confidence`.
* Ambiguity: when another face/track is a plausible, near-equal identity (margin 0.06) both are `UNCERTAIN`; the flag is cleared
  **only by appearance evidence that separates them**, never by position (spatial continuity proves nothing between look-alikes).
* Reacquisition (leaving/re-entering, occlusion, scene cut): appearance only, cosine ≥ 0.50 and a margin of 0.08 over every other
  face, lost track and visible track. A reacquired track withholds `reacquire_frames` (3) frames; a short detection dropout withholds
  `recovery_frames` (2). A track `UNCERTAIN` for 45 frames becomes `LOST`.
* Scene cut: tracks are parked, not forgotten; the person is found again by the strict rule above.

When identity is unreliable the face is **withheld** (original pixels kept), never given to someone else.

## Measured evidence — and its limits

From the stored FaceSet embeddings of 12 people (120 same-person view pairs spanning up to 180° of yaw, 1,650 different-person pairs):
same person min 0.49 / median 0.69; different people max 0.48 / p99 0.34 / median 0.11. So the 0.50 reacquire floor produced
0 false links but rejects ~3% of same-person view pairs (fail-safe: a new ID). ~0.85% of different-person pairs exceed the 0.35
match floor; those links additionally need spatial continuity. **These thresholds are heuristics on a small, curated sample, not a
calibrated benchmark.** They are part of the policy fingerprint, so any change invalidates saved locks.

## Inspection report and contact sheet

`POST /api/v1/media/inspect-faces` with `{"identity_lock": true}` (the UI sets it; default `false` keeps the old response) runs the
policy tracker over the clip (`time_budget_seconds` up to 900; bounded by `max_samples`/`max_decoded_frames`; `truncated` is reported).
It stores a sidecar `roop_3d/.runtime/identity/<inspection_id>.json` with, per track: visible segments, missing ranges,
confidence mean/min, state and withheld-reason counts, three representative thumbnails (sharp, large, frontal, trusted frames) and
the ArcFace signature taken **only from frames cleared for swapping**. The inspection records the sidecar's SHA-256, policy
fingerprint and detection threshold. Signatures are biometric templates: they stay in the local runtime and are never returned by an
API (`GET /api/v1/identity/inspections/{id}?path=` returns the contact sheet without them).

## Identity lock (`identity/lock.py`)

Created only by explicit `POST /api/v1/identity/locks` `{target_path, inspection_id, assignments:[{track_id, faceset_id}]}`.
It binds: target video SHA-256 · inspection file SHA-256 and id · identity report SHA-256 · policy fingerprint · detection threshold ·
detector and recognizer id + SHA-256 · tracking-relevant settings fingerprint · per assignment the FaceSet id and **archive SHA-256**.
`digest` is a canonical hash of that content (excluding id/revision/timestamps), so the same inputs and assignments reproduce the same
digest. Target identity (a track) and source identity (a FaceSet) are separate fields; one track ↔ one FaceSet, several tracks may share a
FaceSet (a person split across tracks), several tracks may have independent FaceSets.

`validate` recomputes everything and returns machine-readable reasons: `TARGET_CHANGED/MISSING`, `INSPECTION_CHANGED/MISSING/INVALID`,
`IDENTITY_REPORT_STALE`, `POLICY_CHANGED`, `MODEL_CHANGED/UNAVAILABLE`, `SETTINGS_CHANGED`, `TRACK_MISSING`, `FACESET_CHANGED/MISSING`,
`LOCK_TAMPERED`. A stale lock is never repaired silently: jobs are refused (`409 IDENTITY_LOCK_STALE`). Only a changed FaceSet can be
re-approved, by an explicit `assign` edit. Manual correction (`POST /locks/{id}/edit`, revision-checked): `assign`, `unassign`, `swap`,
`exclude`, `include`. `GET /locks/{id}` uses a size/mtime hash cache; `?verify=true` and every job re-hash from disk.

## Render binding

`JobRequest.identity_lock_id` (optional). At job creation and again at job start the lock must be valid, match `inspection_id`,
`target_path` and the job's `face_mappings` exactly, and any `face_segments` may only *narrow* it (a swap segment must use a locked
track/FaceSet pair). The runner replays the tracker with the inspection's recorded policy and threshold and installs an
`IdentityGuard`: before any inference a face is re-checked against the inspection signature of its **locked** track. Wrong person on a
locked ID, or a face that looks more like a differently-assigned person, is withheld. The job manifest
(`stage12.identity_lock`) lists per locked track: frames swapped, withheld frames by reason with ranges, not-detected ranges,
the lock digest, and the number of untargeted face observations that were left untouched. A locked track that was seen but never verified fails
the job (`TRACK_IDENTITY_UNVERIFIED`) instead of producing a silently un-swapped video. Identity-locked inspections also refuse the
live-preview "no selection ⇒ every track" fallback (`TARGET_SELECTION_REQUIRED`).

## UI (`components/facemgr/IdentityLockPanel.tsx`, wired in `App.tsx`)

The existing Target Face Inspector cards now show representative frames, visible time ranges, confidence, un-detected gaps and withheld
frames. A lock bar locks/re-locks/releases the on-screen mapping and shows the digest, assignments and stale reasons. Queueing is
blocked while a lock is stale or the on-screen mapping differs from it. Manual correction is: change the track's source, then Re-lock.
Also fixed: `App.tsx` called `request()` unguarded in render for the viewport preview, so any `request()` error (e.g. duplicate Source A/B)
crashed the whole app; it now uses the already-guarded `previewRequest`.

## Tests (all executed; see the report for counts)

* `tests/test_identity_tracker_policy.py` — analytic scenarios with ground truth: similar people, crossing (distinct/look-alike),
  leaving and re-entering, hand over face (box kept / detection dropout), partial occlusion, camera-angle change, scene cut, missing
  faces, stranger vs lost track, ambiguous lost tracks, timeout, budgets, optical flow. Invariant: `swap_allowed ⇒ correct ID`.
* `tests/test_identity_lock.py` — report, lock binding/digest reproducibility, changed video / inspection / report / FaceSet / models /
  settings / policy, tampering, corrections, job binding, guard, run report, session gating, full API + locked job lifecycle.
* `tests/test_identity_real_inference.py` — REAL SCRFD + ArcFace on composited real faces (crossing, absence > track memory, hand
  dropout, hard cut): no silent switch, reacquisition probation, reproducible inspection, contact sheet, lock on the encoded video.
  Skips with a reason if the local models/reference FaceSets are absent (~3 min).
* `src/identityLock.test.tsx` (Vitest) and `e2e/identityLock.e2e.ts` (Playwright, system Chrome, mocked API).

## Known limitations

* Look-alikes whose appearance never separates after a crossing are withheld until they do (possibly the rest of the clip) rather than guessed.
* A different person who takes over a track's box with cosine ≥ 0.35 and IoU ≥ 0.5 is indistinguishable by ArcFace and will be linked.
* Reacquisition and recovery cost 2–3 un-swapped frames each; reported, not hidden.
* Track *splitting* is not offered; a person split across tracks is handled by assigning several tracks to one FaceSet.
* The guard never re-routes an ID; if render IDs ever diverge from the inspection, frames are withheld, not remapped. Seek-mode previews
  start mid-clip, so their IDs can differ from the render's (pre-existing); with a lock they are withheld, not mis-swapped.
* The locked-job tests drive the real runner/worker/API with a scripted session; the real swapper was not run through a locked job in this
  stage. Real inference was exercised for detection, ArcFace, tracking, inspection and lock creation.
* `exclude`/`swap` lock edits are API-only in the UI. Thumbnails were verified in a mocked browser session, not against a live backend.
* Inspection is bounded (default 32 key-frame samples / 30 s); later appearances of a person in a truncated inspection are unreviewed.
