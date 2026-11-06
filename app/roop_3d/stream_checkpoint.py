"""Stage 9: ``job_checkpoint.json`` for the streaming render path (``optimized_processor.StreamingProcessor``).

The processor already closes a finished segment every ``segment_frames`` frames inside
``roop_3d/.runtime/stream/<job_id>/``. This records each one as it is closed, and lets a re-run of the same job
(after pause, crash, or a recoverable failure) start at the first frame after the last verified segment instead of
frame 0. A checkpoint written for different inputs or settings is ignored; its segments are left on disk for the
startup audit, never deleted here.
"""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any

from roop_3d.core.storage import atomic_json, owned
from roop_3d.job_queue import CHECKPOINT_NAME, JobCheckpoint, SegmentEntry, _sha256, count_video_packets


class StreamSegmentCheckpoint:
    def __init__(self, root: Path, job_id: str, fingerprint: str, *, inputs: dict[str, Any], settings_3d: dict[str, Any],
                 execution_provider: str, fps: float, width: int, height: int, total_frames: int):
        self.root, self.fps = root, fps
        self.relative = f"roop_3d/.runtime/stream/{job_id}/{CHECKPOINT_NAME}"
        self.ck = JobCheckpoint(
            job_id=job_id, inputs=inputs, inputs_fingerprint=fingerprint, settings_3d=settings_3d,
            execution_provider=execution_provider,
            media={"fps": fps, "width": width, "height": height, "total_frames": total_frames},
            frame_progress={"completed_frames": 0, "total_frames": total_frames}, state="RENDERING")

    def resume(self) -> tuple[list[Path], int]:
        """Verified, contiguous finished segments of an earlier attempt, and the frame to continue from."""
        path = owned(self.root, self.relative)
        if not path.is_file():
            return [], 0
        try:
            old = JobCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            return [], 0
        if old.inputs_fingerprint != self.ck.inputs_fingerprint:
            return [], 0
        kept: list[SegmentEntry] = []
        frame = 0
        for seg in sorted(old.segments, key=lambda s: s.index):
            segment = owned(self.root, seg.path)
            ok = (seg.status == "COMPLETED" and seg.start_frame == frame and seg.index == len(kept) and segment.is_file()
                  and seg.sha256 == _sha256(segment) and count_video_packets(segment) == seg.expected_frames)
            if not ok:
                break
            kept.append(seg)
            frame = seg.end_frame
        self.ck.segments = kept
        self._save()
        return [owned(self.root, s.path) for s in kept], frame

    def reset(self) -> None:
        self.ck.segments = []
        self._save()

    def record(self, segment: Path, start_frame: int, frames: int, encoder: str) -> None:
        self.ck.encoder = encoder
        self.ck.segments.append(SegmentEntry(
            index=len(self.ck.segments), start_frame=start_frame, end_frame=start_frame + frames,
            start_seconds=round(start_frame / self.fps, 6), end_seconds=round((start_frame + frames) / self.fps, 6),
            path=Path(os.path.abspath(segment)).relative_to(os.path.abspath(self.root)).as_posix(), status="COMPLETED", frames=frames, sha256=_sha256(segment)))
        self._save()

    def finish(self, output: dict[str, Any]) -> None:
        self.ck.state, self.ck.final_output = "COMPLETED", output
        self._save()

    def _save(self) -> None:
        self.ck.updated_at = datetime.now(timezone.utc).isoformat()
        self.ck.frame_progress["completed_frames"] = sum(s.frames for s in self.ck.segments)
        atomic_json(self.root, self.relative, self.ck.model_dump(mode="json"))
