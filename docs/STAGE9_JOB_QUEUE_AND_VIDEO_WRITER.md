# Stage 9: persistent job queue, checkpoint recovery, segment joining, NVHardwareVideoWriter

## Path mapping (the prompt's paths do not exist)

| Prompt | Real |
| --- | --- |
| `app/roop/core/job_queue.py` | `app/roop_3d/job_queue.py` (checkpoint + segmented runner); the queue itself is the existing `app/roop_3d/queue_manager.py` |
| `app/temp/<job_id>/segments/` | `app/roop_3d/.runtime/temp/<job_id>/segments/` (all writable state must stay under `.runtime`) |
| `NVHardwareVideoWriter` | `app/roop_3d/media/nv_writer.py` |

## What was already there (kept, not rewritten)

`QueueManager` already provided the SQLite store, WAL, idempotency, cancel / pause / resume / reorder / duplicate / retry, startup recovery, and `segment_dispatcher.py` already did keyframe detection, GOP-aligned cuts and `concat_lossless`.

## What Stage 9 added

- **States**: `ANALYZING` and `RETRYING` (now 12). `FAILED -> RETRYING -> QUEUED` is how `retry_failed_slice` now goes. `PRESCANNING` / `PREPASSING` remain for existing callers. Mirrored in `server/routes.py` and `contracts.ts`.
- **`job_checkpoint.json`** (`JobCheckpoint`): inputs + fingerprint, 3D settings, execution provider, encoder, media facts, frame progress, segment manifest (frames, sha256, status), final output. Written atomically after every segment. `QueueManager.set_checkpoint_path` links it to the queue row; pause no longer discards a non-`.npz` checkpoint path.
- **`SegmentedRenderRunner`**: QUEUED -> PREFLIGHT -> ANALYZING -> RENDERING -> MUXING -> COMPLETED. Pause/cancel are issued through the queue (`queue.pause(job_id)`) and noticed every `interrupt_poll_frames` frames. A partial segment is discarded; finished ones are kept. On resume a segment is skipped only if it is `COMPLETED` **and** its file still has the expected packet count. If the inputs changed since the checkpoint, the run refuses (`CHECKPOINT_MISMATCH`) rather than mixing segments.
- **Segment safety**: a segment is written as `segment_XXXX.mp4.part.mp4` and `os.replace`d into place only when complete. Nothing in the runner deletes segments; `cleanup_segments()` is a separate call that refuses until a verified output exists. The join is `concat -f concat -safe 0 -c copy`, then audio remux; the output is checked (frame count, duration, audio) before publishing and the segments are re-checked afterwards.
- **`NVHardwareVideoWriter`**: `h264_nvenc` / `hevc_nvenc` with a real probe encode; falls back to `libx264` / `libx265` if NVENC is unavailable or dies while the first frames are written (`encoder`, `fallback_reason` report it). `lut_path` + `lut_bake=True` bakes a `.cube` through `lut3d`; `lut_bake=False` ships it as a `<name>.cube` sidecar. `mux_source_audio` adds the source audio trimmed to the timeline In/Out (AAC when trimmed, stream copy when not).

## Wired into the studio render path

The studio UI queue (`QueueDrawer`) is backed by `JobStore` / `JobWorker`, whose streaming renderer is `optimized_processor.StreamingProcessor` with the existing `hardware_streamer.NVHardwareVideoWriter` (NVENC, `lut_path` / `lut_bake`, libx264 fallback) and `SegmentWorkspace` (stream-copy concat + audio remux). Those already existed; `media/nv_writer.py` is a separate standalone writer used by `SegmentedRenderRunner`. What was missing there was resume, which is now added:

- `stream_checkpoint.StreamSegmentCheckpoint` writes `roop_3d/.runtime/stream/<job_id>/job_checkpoint.json` each time a segment closes (inputs, `settings_3d`, execution provider, encoder, frame progress, segment manifest with sha256).
- On a re-run of the same job (pause, crash, recoverable failure) `StreamingProcessor.run` keeps the leading segments that are still intact (file present, sha256 and frame count match, contiguous from frame 0), skips decoding them into the renderer, and continues from that frame. Corrupt or missing segments truncate the resume point. Finished segments are never rewritten and stay on disk after the join (the startup audit only collects them once the job can no longer run).
- `GET /api/v1/jobs/{job_id}/checkpoint` reports the resume point; `ResumeNote.jsx` shows "Resumes at frame N of M after K finished segments" on paused / interrupted / failed cards in the queue drawer.
- Tests: `roop_3d/tests/test_stage09_stream_resume.py` (interrupt a real render, rerun, verify frame 6 resume, untouched segment, corrupt-segment rejection, endpoint).

## Verification

`python verify_stage9_queue_and_video_writer.py` (needs `ffmpeg` on PATH, e.g. `/g/pinokio/bin/miniforge/Library/bin` in bash). 3 segments, pause mid-segment-2, simulated restart, resume at frame 30 with segment 0 byte-identical, 90 frames / 3.0 s / audio, trimmed 60-frame job, RETRYING/cancel, NVENC (or reported fallback), LUT bake.

## Known limits

- `SegmentedRenderRunner` (queue-driven, GOP-aligned) is still not what the studio buttons start; the studio gets resume through `StreamingProcessor` above. Only the stream path resumes at segment level: the reference path keeps its own frame checkpoints. Resuming after the machine switched between NVENC and the CPU encoder concatenates segments from two encoders with `-c copy`, which is not verified.
- Frame indexing assumes constant frame rate sources (`round(pts * fps)`); VFR input is re-timed to its average rate.
- `RETRYING` has no UI styling yet.
