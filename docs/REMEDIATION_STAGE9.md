# Remediation Stage 9: Four-Strategy Batch Matrix, 10-State Persistent Queue, and GOP-Aligned Stream Splitting

Implementation uses the standalone checkout's `app/roop_3d` and `app/roop_3d_ui` paths. The legacy `app/roop` references map seamlessly to `app/roop_3d`.

## Backend-Owned 10-State Persistent Queue

`app/roop_3d/queue_manager.py` implements the persistent queue engine backed by ACID-safe SQLite storage (`roop_3d/.runtime/queue/queue.db` in WAL mode with immediate transactions) synchronized with atomic JSON state snapshots (`roop_3d/.runtime/queue/queue.json`).

### Canonical 10 Job States
The queue manages all 10 canonical job states with strict transition enforcement:
1. `QUEUED`: Enqueued in persistent state awaiting execution.
2. `PREFLIGHT`: Model availability, VRAM allocation, and input verification.
3. `PRESCANNING`: Face detection and identity inspection across targets.
4. `PREPASSING`: Stage 0 pose matrix estimation, sparse keyframe prepass, and landmark caching.
5. `RENDERING`: Swap pipeline execution across frames and segments.
6. `MUXING`: Lossless audio remuxing and instant FFmpeg stream-copy concatenation.
7. `PAUSED`: Execution suspended; checkpoint serialized to dense `.stage2.npz` (`checkpoint.npz`).
8. `COMPLETED`: Processing finished; output video verified.
9. `FAILED`: Terminal error state with actionable error details.
10. `CANCELLED`: Terminal user-cancelled state; resources released.

### Server Restart & Crash Recovery
Closing browser tabs or abruptly killing the FastAPI process preserves `QUEUED` and `PAUSED` jobs intact. On startup, `recover_on_startup()` automatically:
- Preserves all `QUEUED` jobs intact with their original FIFO `queue_order`.
- Preserves all `PAUSED` jobs intact along with their `.stage2.npz` checkpoint references.
- Gracefully transitions any in-flight active states (`PREFLIGHT`, `PRESCANNING`, `PREPASSING`, `RENDERING`, `MUXING`) to `PAUSED` with checkpoints intact so work is never lost.

### Queue Controls & WebSocket Feeds
Exposes REST and WebSocket controls via `/api/v1/queue` and `/ws/queue`:
- Per-job frame progress, fps, and latency telemetry (`POST /api/v1/queue/{job_id}/progress`).
- Per-job cancellation (`POST /api/v1/queue/{job_id}/cancel`).
- Pause and resume from `.stage2.npz` checkpoint (`POST /api/v1/queue/{job_id}/pause`, `POST /api/v1/queue/{job_id}/resume`).
- Drag-and-drop queue reordering (`POST /api/v1/queue/reorder`) executed in a single ACID transaction.
- Job duplication (`POST /api/v1/queue/{job_id}/duplicate`).
- Granular slice retry (`POST /api/v1/queue/{job_id}/retry`).
- Automatic clip joining (`POST /api/v1/queue/{job_id}/join`).

---

## Four-Strategy Combinatorial Batch Matrix Engine (`#/batch`)

`app/roop_3d/batch_matrix.py` implements the combinatorial batch generator, paired with the interactive UI matrix builder in `app/roop_3d_ui/src/screens/BatchMatrixScreen.jsx`:

1. **Strategy A (One-to-Many)**: Applies 1 source identity (or `.fsz` archive) across N target videos/images.
2. **Strategy B (Grouped)**: M source identities mapped to M distinct target groups (e.g. interview subjects vs action sequences).
3. **Strategy C (Per-File Matrix)**: Full Cartesian product (M sources × N targets) with toggleable selection cells, bulk controls (Select All, Clear, Invert, Row/Col toggle), and automatic output naming templates (`{target}_{source}_{restorer}_{timestamp}.mp4`).
4. **Strategy D (Recipe Matrix)**: 1 source × 1 target evaluated across multiple parameter presets:
   - GPEN-512 vs CodeFormer (with scalar fidelity feed `w=0.85`) vs UltraMax Deep Dynamic.
   - TPS / Frontalization sweeps across trimmed clips.
   - Custom user recipe creation with real-time parameter tuning.

---

## GOP-Aligned Video Segment Splitting & Lossless Concat

`app/roop_3d/segment_dispatcher.py` provides segment splitting and concatenation with zero video re-encoding:

1. **GOP Keyframe Boundary Detection**: Keyframe timestamps are resolved from video streams using container packet demuxing and `is_keyframe` inspection.
2. **Zero Generation Loss Stream-Copy**: Chunks are split using FFmpeg stream copy (`-c copy`) along exact keyframe boundaries into timestamped files: `chunk_0001_<ts>.mp4`, `chunk_0002_<ts>.mp4`.
3. **Intermediate Chunk Preservation**: Multi-segment rendering guarantees every intermediate chunk file is strictly preserved on disk throughout processing.
4. **Lossless Concat Demuxer Pass**: Upon completion of all slices, an instant FFmpeg `concat` demuxer stream-copy pass (`-f concat -safe 0 -c copy`) merges the chunks and remuxes the original source audio stream without a second video encode, ensuring frame-accurate audio synchronization.

---

## Verification Gate Summary

1. **Combinatorial Batch Expansion**: All 4 Batch Matrix strategies verified to expand into exact expected job manifests with verified naming templates and parameter mappings.
2. **ACID Queue Durability & Restart Recovery**: Verified that terminating and rebooting the FastAPI service restores `QUEUED` and `PAUSED` jobs intact, supports NPZ checkpoint resume, job duplication, and drag-and-drop reordering.
3. **GOP Splitting & Lossless Concat**: Verified on multi-GOP video with audio that splitting along keyframes preserves intermediate chunks and concat demuxer produces exact frame-accurate output with synchronized stereo audio.
