# Stage 5: timeline scrubbing, filmstrip preview and on-canvas face selection

Scrubbing the timeline shows the exact frame at once, the timeline carries a real filmstrip, and the faces detected in that
frame can be targeted by clicking their box on the video or their card above the timeline.

Path mapping, as in earlier stages (the prompt's paths do not exist): the prompt's `app/api.py` is `app/roop_3d/server/`
(new router `media_frames.py`, the decoder in `media/frame_service.py`), its frontend is `app/roop_3d_ui/`, and its
`verify_stage5_timeline_and_faces.py` is at the repository root next to `verify_stage3_3d_reconstruction.py`.

## What changed for you

* **Scrub and the frame follows.** Dragging the playhead, clicking the timeline or a filmstrip thumbnail, or stepping frames
  updates the preview as fast as frames can be served. On the real 1280x720 test clip that is ~88 frames/s for consecutive
  frames and ~30 ms for a jump to anywhere in the video (median), measured over HTTP.
* **Filmstrip.** Loading a target requests `/media/info` and one `/media/filmstrip` (16 thumbnails, more when the timeline is
  zoomed in). The SMPTE `HH:MM:SS:FF` ruler and the In / Out markers were already there and are unchanged.
* **Faces appear on their own.** When the playhead rests on a frame, the faces in *that* frame are detected and drawn at their
  real position. Before this, boxes came only from "Inspect target faces" and sat where a track *first* appeared.
* **Boxes** (Overlays toggle): unselected = dashed cyan with a `#1`, `#2` badge and a `Y -27° P +17°` pill; hovered = glow;
  the 5-point landmarks (Landmarks toggle); the selected target = solid emerald with `TARGET LOCKED`. `#n` is the same
  left-to-right number *Use as source* uses.
* **Detected Faces in Frame — Click to target**: one card per face right above the timeline (crop, rank, pose, confidence).
  Clicking a card or its box on the canvas does the same thing.
* **Locking.** A face that is already a target is simply selected. A new one is locked: the server records its track (the
  existing `/media/detect-frame-faces`, so it can be rendered), it becomes the selected target, the 3D viewer's **Live Pose**
  follows it (and follows its pose frame to frame), and when a source FaceSet is selected the track is mapped to it.
* **⚡ Live Frame Swap Preview** (next to Compare): swaps the face on this frame and opens the split-wipe comparison canvas; click
  again to go back to side by side.

## API

| Request | Answer |
|---|---|
| `GET /api/v1/media/info?path=` | `{width, height, fps, fps_rational, total_frames, duration_sec, rotation_degrees, frame_count_declared}` (upright size; for a rotated phone clip `width` and `height` are swapped) |
| `GET /api/v1/media/frame?path=&frame_index=N` (or `&timestamp=`) | a JPEG; `X-Frame-Meta` (JSON header) = `{frame_index, timestamp_sec, width, height, image_width, image_height, total_frames, fps, cache, decode_ms}` |
| `…&detect_faces=true&threshold=0.5` | adds `faces: [{face_index, track_id, bbox, bbox_norm, kps_5, pose, pose3d, pose_cell, confidence}]` and `unsupported` |
| `…&format=json` | the same as JSON, the JPEG as a data URL |
| `…&max_size=1280&quality=82` | shrinks the JPEG only; `bbox` and `kps_5` stay in native pixels |
| `GET /api/v1/media/filmstrip?path=&count=16&thumb_height=90` | `{count, thumb_width, thumb_height, total_frames, duration_sec, frames: [{slot, frame_index, timestamp_sec, data_url}]}` |

Errors use the standard envelope: `416` past the end, `422` for a bad selector or a path outside the runtime, `404` missing file.

## Design

**`VideoFrameService`** keeps a small LRU (4) of *open* containers. A request just ahead of the last decoded frame keeps
decoding (no seek); anything else seeks back to the previous keyframe and decodes forward. Decoded frames (192 MB), JPEGs
(64 MB), detections and filmstrips are cached, so going back and forth is free. Colour (matrix, range, HDR tone-mapping) and
rotation come from the production `VideoReader` (`VideoReader.convert` is the one new public method), so a scrubbed frame
looks like a rendered one. `MediaService.frame` (replay from frame 0, bounded to 10 000 frames) is deliberately untouched: it
is the exact path batch code and an existing test rely on.

Frame numbers map to time as `frame_index / fps` with the stream's average rate. That is exact for constant frame rate; the
response always reports the timestamp of the frame really decoded, so a variable-frame-rate file shows up as a difference
rather than a silent substitution. A half-frame tolerance absorbs presentation-time jitter.

**`FrameScrubber`** (`src/frameScrub.ts`) is the "latest frame wins" queue the prompt asked for: one request in flight; when it
lands, the newest wanted frame is requested next. The old preview aborted and re-armed a 100 ms timer on every move, so a
continuous drag never produced a frame. Faces are requested only after the playhead rests (120 ms), as a second request for
the frame already shown (the server answers it from cache). Frames are kept client-side (24, freed on eviction).

**Matching.** A detection overlapping a stored target face is that face (same id), so selection, role, source mapping and the
context menu keep working; the rest are free candidates. The store knows a track by one static box; the frame detection is
the only source of a moving face's real box.

## Not implemented, and why

* **`embedding_norm`.** The recognizer returns L2-normalised vectors, so the norm is always 1.0; reporting it would be
  invented data. It is listed in each response's `unsupported`. "Linking the face's embedding to the swap slot" is done as
  the track-to-FaceSet mapping, which is what the pipeline consumes; the embedding itself is computed at render time.
* **`track_id`** in a per-frame detection is `null`: a single frame has no tracker. Locking creates the track.
* **`POST /media/preview-swap`.** `POST /api/preview` already renders the swapped frame (live swap, cache-first, GPU-aware);
  the new button drives it instead of adding a duplicate endpoint.
* **68-point landmarks** on the canvas: detection provides 5 points (SCRFD).
* **Frame-to-frame identity.** A locked face is shown as locked where it overlaps its track's stored box. Following the same
  person through moving, overlapping faces needs per-frame embeddings, which scrubbing deliberately does not pay for.

## Defects fixed on the way

* Frame indices of 10 000 and above were refused (`FRAME_SELECTOR_INVALID`), so *Use as source* failed on any clip longer than
  ~5.5 minutes at 30 fps. The extractor now seeks for those indices (the real test clip has 12 840 frames).
* `/media/detect-frame-faces` rejected `frame_index > 9999`; it converts to a time, so the cap is lifted.
* `Detector` dropped its confidence scores; `detect_scored` returns them (`detect` is unchanged).
* The filmstrip issued one full decode per thumbnail (a dozen per timeline width); it is one request.
* The per-frame preview reset to blank on every move, which flashed the `<video>` underneath; the previous frame now stays
  until the next arrives.

## Verification

* `npm run build` (tsc, eslint, vite): 0 errors.
* `npx vitest run`: 408 tests (18 new in `src/stage5Scrub.test.tsx`: latest-wins and one-in-flight, settle-then-detect, cache,
  retry, matching, strip).
* `npx playwright test`: real Chrome; 7 new tests in `e2e/stage5Scrub.e2e.ts` (filmstrip, scrub coalescing, strip, lock,
  canvas click through the letterbox, wipe button).
* `pytest roop_3d/tests/test_stage05_timeline_frames.py`: 22 tests on a 10 200-frame clip whose every frame encodes its own
  number, so "which frame did I get" is read from pixels (including indices past 10 000, GOP boundaries, LRU eviction).
* `verify_stage5_timeline_and_faces.py`: real clip, real SCRFD detector, over HTTP. Boxes agree with an independent detector
  run (IoU > 0.85), keypoints lie on the face, frame 1200 matches the replay path (mean abs diff 0.69), frame 12 000 is served,
  one container for the whole run.
* Real UI check: real server, real clip uploaded through the browser, a face locked, Live Pose read `-26.9° Y · +16.5° P` for a
  face the detector measured at Y -27° P +17°.
