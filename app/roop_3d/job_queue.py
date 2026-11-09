"""Stage 9: checkpointed, segment-safe render jobs on top of the persistent ``QueueManager``.

``QueueManager`` (``queue_manager.py``) already owns the SQLite queue, the lifecycle states, cancel / pause /
resume / reorder / retry. This module adds what a *long render* needs on top of it:

* ``job_checkpoint.json`` -- exact inputs, 3D settings, frame progress, execution provider and the segment
  manifest, rewritten atomically after every segment, so an interrupted job resumes at the first segment that is
  not already finished and verified instead of at frame 0.
* GOP-aligned segments rendered to ``<job work dir>/segments/segment_XXXX.mp4``. A finished segment is never
  re-rendered, overwritten or deleted by this module: it is written as ``.part`` and renamed only once complete,
  and the final ``concat -c copy`` + audio remux reads the originals. Cleanup is an explicit, separate call that
  refuses to run until the joined output has been verified.

Path mapping: the stage prompt names ``app/roop/core/job_queue.py`` and ``app/temp/<job_id>/``. The real package is
``app/roop_3d`` and all writable state must live under ``roop_3d/.runtime`` (``core.storage.owned``), so jobs work
in ``roop_3d/.runtime/temp/<job_id>/``.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
from typing import Any, Callable, Iterator

import av
import numpy as np
from pydantic import ConfigDict, Field

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract
from roop_3d.core.storage import atomic_json, owned
from roop_3d.engine.vram_watchdog import watchdog
from roop_3d.media.nv_writer import NVHardwareVideoWriter, mux_source_audio
from roop_3d.queue_manager import CanonicalJobState, QueueJob, QueueJobProgress, QueueManager, QueueSlice
from roop_3d.segment_dispatcher import compute_gop_aligned_cuts, concat_lossless, detect_keyframe_timestamps

__all__ = ["JobCheckpoint", "SegmentEntry", "SegmentedRenderRunner", "QueueManager", "CanonicalJobState"]

CHECKPOINT_NAME = "job_checkpoint.json"
FrameTransform = Callable[[np.ndarray, int], np.ndarray]


class SegmentEntry(Contract):
    model_config = ConfigDict(frozen=False)
    index: int  # 0-based; file is segment_{index:04d}.mp4
    start_frame: int
    end_frame: int  # exclusive
    start_seconds: float
    end_seconds: float
    path: str  # relative to the owned root
    status: str = "PENDING"  # PENDING | COMPLETED | FAILED
    frames: int = 0
    sha256: str | None = None
    error: str | None = None

    @property
    def expected_frames(self) -> int:
        return self.end_frame - self.start_frame


class JobCheckpoint(Contract):
    model_config = ConfigDict(frozen=False)
    schema_version: int = 1
    job_id: str
    inputs: dict[str, Any]
    inputs_fingerprint: str
    settings_3d: dict[str, Any] = Field(default_factory=dict)
    execution_provider: str = "CPUExecutionProvider"
    encoder: str = ""
    media: dict[str, Any] = Field(default_factory=dict)  # fps, width, height, total_frames
    frame_progress: dict[str, int] = Field(default_factory=lambda: {"completed_frames": 0, "total_frames": 0})
    segments: list[SegmentEntry] = Field(default_factory=list)
    state: str = "QUEUED"
    final_output: dict[str, Any] | None = None
    updated_at: str = ""


def _default_provider() -> str:
    try:
        import onnxruntime
        return onnxruntime.get_available_providers()[0]
    except Exception:
        return "CPUExecutionProvider"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def count_video_packets(path: Path) -> int:
    """Frames in a file written by our own (B-frame tolerant, one packet per frame) encoders."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return sum(1 for packet in container.demux(stream) if packet.size)


def iter_source_frames(path: Path, start_frame: int, end_frame: int, fps: float) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, bgr)`` for ``start_frame <= index < end_frame`` seeking to the preceding keyframe."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        base = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0.0
        container.seek(max(0, int((start_frame / fps - 0.5) * 1_000_000)), backward=True, any_frame=False)
        for frame in container.decode(stream):
            index = round((float(frame.pts * stream.time_base) - base) * fps)
            if index < start_frame:
                continue
            if index >= end_frame:
                return
            yield index, frame.to_ndarray(format="bgr24")


class SegmentedRenderRunner:
    """Runs one queued job to completion, a pause, a cancel, or a failure. Call again after resume/retry."""

    def __init__(self, queue: QueueManager, *, interrupt_poll_frames: int = 8):
        self.queue = queue
        self.root = queue.root
        self.poll = max(1, interrupt_poll_frames)

    # -- paths --------------------------------------------------------------------------------------------------

    def work_dir(self, job_id: str) -> str:
        return f"roop_3d/.runtime/temp/{job_id}"

    def checkpoint_rel(self, job_id: str) -> str:
        return f"{self.work_dir(job_id)}/{CHECKPOINT_NAME}"

    def load_checkpoint(self, job_id: str) -> JobCheckpoint | None:
        path = owned(self.root, self.checkpoint_rel(job_id))
        if not path.is_file():
            alt = owned(self.root, f"jobs/{job_id}/checkpoint.json")
            if alt.is_file():
                path = alt
            else:
                return None
        try:
            return JobCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as err:
            raise DomainError("CHECKPOINT_INVALID", f"{CHECKPOINT_NAME} is unreadable: {err}", 422) from None

    def _save(self, ck: JobCheckpoint) -> None:
        ck.updated_at = datetime.now(timezone.utc).isoformat()
        ck.frame_progress["completed_frames"] = sum(s.frames for s in ck.segments if s.status == "COMPLETED")
        dump = ck.model_dump(mode="json")
        dump["resume_frame"] = ck.frame_progress["completed_frames"]
        dump["gop_aligned"] = True
        dump["checkpoint_every"] = 120

        # Sync to both checkpoint_rel and app/jobs/<job_id>/checkpoint.json
        atomic_json(self.root, self.checkpoint_rel(ck.job_id), dump)

        jobs_dir = owned(self.root, f"jobs/{ck.job_id}")
        jobs_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(self.root, f"jobs/{ck.job_id}/checkpoint.json", dump)

        # Append to SQLite journal
        try:
            db_path = jobs_dir / "checkpoint.db"
            with sqlite3.connect(str(db_path), timeout=5.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS checkpoint_journal (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        job_id TEXT NOT NULL,
                        frame INTEGER NOT NULL,
                        resume_frame INTEGER NOT NULL,
                        timestamp TEXT NOT NULL,
                        action TEXT NOT NULL,
                        status TEXT NOT NULL,
                        metadata TEXT
                    )
                """)
                conn.execute(
                    "INSERT INTO checkpoint_journal (job_id, frame, resume_frame, timestamp, action, status, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (ck.job_id, ck.frame_progress["completed_frames"], ck.frame_progress["completed_frames"], ck.updated_at, "GOP_CHECKPOINT", "COMMITTED", json.dumps({"state": ck.state}))
                )
        except Exception:
            pass

    def _target(self, request: dict[str, Any]) -> Path:
        raw = request.get("target_path")
        if not raw:
            raise DomainError("TARGET_MISSING", "request.target_path is required", 422)
        path = Path(raw)
        path = path if path.is_absolute() else self.root / path
        if not path.is_file():
            raise DomainError("MEDIA_NOT_FOUND", f"Target video {path} does not exist", 404)
        return path.resolve()

    # -- lifecycle ----------------------------------------------------------------------------------------------

    def _step(self, job_id: str, state: CanonicalJobState) -> None:
        if self.queue.get(job_id).state != state:
            self.queue.transition(job_id, state)

    def _interrupted(self, job_id: str) -> str | None:
        state = self.queue.get(job_id).state
        return state.value if state in (CanonicalJobState.PAUSED, CanonicalJobState.CANCELLED) else None

    def run(self, job_id: str, transform: FrameTransform | None = None) -> dict[str, Any]:
        job = self.queue.get(job_id)
        if job.state not in (CanonicalJobState.QUEUED, CanonicalJobState.RETRYING):
            raise DomainError("ILLEGAL_JOB_TRANSITION", f"Only QUEUED jobs can run (current: {job.state.value})", 422)
        try:
            self._step(job_id, CanonicalJobState.PREFLIGHT)
            target = self._target(job.request)
            self._step(job_id, CanonicalJobState.ANALYZING)
            ck = self._analyze(job, target)
            self._step(job_id, CanonicalJobState.RENDERING)
            outcome = self._render(job_id, ck, target, transform or (lambda frame, _i: frame))
            if outcome != "DONE":
                return {"state": outcome, "completed_segments": sum(s.status == "COMPLETED" for s in ck.segments),
                        "total_segments": len(ck.segments), "checkpoint": self.checkpoint_rel(job_id)}
            self._step(job_id, CanonicalJobState.MUXING)
            output = self._join(job, ck, target)
            self.queue.transition(job_id, CanonicalJobState.COMPLETED, output=output)
            return {"state": "COMPLETED", **output}
        except Exception as err:
            current = self.queue.get(job_id).state
            if current not in (CanonicalJobState.FAILED, CanonicalJobState.CANCELLED, CanonicalJobState.COMPLETED):
                self.queue.transition(job_id, CanonicalJobState.FAILED, error_message=str(err)[:500])
            raise

    # -- ANALYZING ----------------------------------------------------------------------------------------------

    def _analyze(self, job: QueueJob, target: Path) -> JobCheckpoint:
        request = job.request
        inputs = {
            "target_path": str(target), "target_size": target.stat().st_size,
            "trim_in_seconds": request.get("trim_in_seconds"), "trim_out_seconds": request.get("trim_out_seconds"),
            "max_segment_seconds": float(request.get("max_segment_seconds", 15.0)),
            "codec": request.get("codec", "h264"), "lut_path": request.get("lut_path"),
            "lut_bake": bool(request.get("lut_bake", True)), "output_name": request.get("output_name"),
        }
        fingerprint = hashlib.sha256(repr(sorted(inputs.items(), key=lambda kv: kv[0])).encode()).hexdigest()
        existing = self.load_checkpoint(job.job_id)
        if existing is not None:
            if existing.inputs_fingerprint != fingerprint:
                raise DomainError("CHECKPOINT_MISMATCH",
                                  "job_checkpoint.json was written for different inputs; refusing to mix segments", 409)
            existing.execution_provider = request.get("execution_provider") or existing.execution_provider
            return existing

        with av.open(str(target)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate or stream.guessed_rate or 0)
            width, height = stream.codec_context.width, stream.codec_context.height
            duration = float(container.duration) / 1_000_000 if container.duration else 0.0
            total = int(stream.frames or 0) or round(duration * fps)
        if fps <= 0 or total <= 0:
            raise DomainError("MEDIA_UNREADABLE", "Could not determine the target's frame rate / length", 422)
        in_frame = max(0, round(float(inputs["trim_in_seconds"] or 0.0) * fps))
        out_frame = min(total, round(float(inputs["trim_out_seconds"]) * fps)) if inputs["trim_out_seconds"] else total
        if out_frame <= in_frame:
            raise DomainError("TRIM_INVALID", "Timeline Out must be after In", 422)

        keyframes, kf_duration = detect_keyframe_timestamps(target)
        cuts = compute_gop_aligned_cuts(keyframes, kf_duration or total / fps, inputs["max_segment_seconds"])
        bounds = sorted({in_frame, out_frame, *(min(max(round(c[0] * fps), in_frame), out_frame) for c in cuts)})
        segments = [
            SegmentEntry(index=i, start_frame=a, end_frame=b, start_seconds=round(a / fps, 6), end_seconds=round(b / fps, 6),
                         path=f"{self.work_dir(job.job_id)}/segments/segment_{i:04d}.mp4")
            for i, (a, b) in enumerate(zip(bounds, bounds[1:])) if b > a
        ]
        ck = JobCheckpoint(
            job_id=job.job_id, inputs=inputs, inputs_fingerprint=fingerprint,
            settings_3d=dict(request.get("settings_3d") or {}),
            execution_provider=request.get("execution_provider") or _default_provider(),
            media={"fps": fps, "width": width, "height": height, "total_frames": total},
            frame_progress={"completed_frames": 0, "total_frames": out_frame - in_frame},
            segments=segments, state="ANALYZING")
        owned(self.root, f"{self.work_dir(job.job_id)}/segments").mkdir(parents=True, exist_ok=True)
        self._save(ck)
        self.queue.set_checkpoint_path(job.job_id, self.checkpoint_rel(job.job_id))
        self._publish_slices(job.job_id, ck)
        return ck

    def _publish_slices(self, job_id: str, ck: JobCheckpoint) -> None:
        self.queue.set_slices(job_id, [
            QueueSlice(slice_id=f"seg{s.index:04d}", slice_index=s.index, chunk_path=s.path,
                       rendered_chunk_path=s.path if s.status == "COMPLETED" else None,
                       start_seconds=s.start_seconds, end_seconds=s.end_seconds,
                       duration=round(s.end_seconds - s.start_seconds, 6), status=s.status if s.status != "PENDING" else "PENDING",
                       error=s.error, frame_count=s.frames) for s in ck.segments])

    # -- RENDERING ----------------------------------------------------------------------------------------------

    def _segment_verified(self, seg: SegmentEntry) -> bool:
        path = owned(self.root, seg.path)
        return (seg.status == "COMPLETED" and path.is_file() and path.stat().st_size > 0
                and count_video_packets(path) == seg.expected_frames)

    def _render(self, job_id: str, ck: JobCheckpoint, target: Path, transform: FrameTransform) -> str:
        fps, width, height = ck.media["fps"], ck.media["width"], ck.media["height"]
        request = self.queue.get(job_id).request
        for seg in ck.segments:
            if seg.status == "COMPLETED" and self._segment_verified(seg):
                continue  # finished earlier: never touched again
            seg.status, seg.frames, seg.error = "PENDING", 0, None
            final = owned(self.root, seg.path)
            part = final.with_name(final.name + ".part.mp4")
            part.unlink(missing_ok=True)
            writer = NVHardwareVideoWriter(
                part, width, height, fps, codec=ck.inputs["codec"], prefer_hardware=bool(request.get("prefer_hardware", True)),
                lut_path=ck.inputs["lut_path"], lut_bake=ck.inputs["lut_bake"], gop=max(1, round(fps)))
            written, interrupted = 0, None
            try:
                for index, frame in iter_source_frames(target, seg.start_frame, seg.end_frame, fps):
                    if written % 15 == 0:
                        watchdog.check_and_adapt()
                    try:
                        transformed = transform(frame, index)
                    except Exception as exc:
                        if "out of memory" in str(exc).lower():
                            watchdog.stepdown(reason="OOM during frame transform")
                            transformed = transform(frame, index)
                        else:
                            raise
                    writer.write(np.ascontiguousarray(transformed))
                    written += 1
                    if written % 120 == 0:
                        self._save(ck)
                        advisory_msg = watchdog.advisory_badges[-1]["badge"] if watchdog.advisory_badges else None
                        self.queue.update_progress(job_id, QueueJobProgress(
                            frame=ck.frame_progress["completed_frames"] + written,
                            total_frames=ck.frame_progress["total_frames"],
                            fps=fps, stage="RENDERING", advisory=advisory_msg
                        ))
                    if written % self.poll == 0:
                        interrupted = self._interrupted(job_id)
                        if interrupted:
                            break
                if not interrupted:
                    interrupted = self._interrupted(job_id)
            except BaseException as err:
                writer.__exit__(type(err), err, None)
                seg.status, seg.error = "FAILED", str(err)[:300]
                self._save(ck)
                self._publish_slices(job_id, ck)
                raise
            if interrupted:
                writer.__exit__(RuntimeError, None, None)  # discard the partial segment only
                part.unlink(missing_ok=True)
                ck.state = interrupted
                self._save(ck)
                self._publish_slices(job_id, ck)
                return interrupted
            writer.close()
            if written != seg.expected_frames or count_video_packets(part) != written:
                part.unlink(missing_ok=True)
                seg.status, seg.error = "FAILED", f"decoded {written} of {seg.expected_frames} frames"
                self._save(ck)
                self._publish_slices(job_id, ck)
                raise DomainError("SEGMENT_FRAME_MISMATCH", f"segment {seg.index}: {seg.error}", 500)
            ck.encoder = writer.encoder
            os.replace(part, final)  # published only when complete
            seg.status, seg.frames, seg.sha256 = "COMPLETED", written, _sha256(final)
            ck.state = "RENDERING"
            self._save(ck)
            self._publish_slices(job_id, ck)
            advisory_msg = watchdog.advisory_badges[-1]["badge"] if watchdog.advisory_badges else None
            self.queue.update_progress(job_id, QueueJobProgress(
                frame=ck.frame_progress["completed_frames"], total_frames=ck.frame_progress["total_frames"],
                fps=fps, stage="RENDERING", advisory=advisory_msg))
        return "DONE"

    # -- MUXING -------------------------------------------------------------------------------------------------

    def _join(self, job: QueueJob, ck: JobCheckpoint, target: Path) -> dict[str, Any]:
        missing = [s.index for s in ck.segments if not self._segment_verified(s)]
        if missing:
            raise DomainError("SLICE_INCOMPLETE", f"segments {missing} are missing or fail verification", 422)
        fps = ck.media["fps"]
        work = self.work_dir(job.job_id)
        joined = owned(self.root, f"{work}/joined_video.mp4")
        concat_lossless([owned(self.root, s.path) for s in ck.segments], None, joined, preserve_intermediates=True)

        name = ck.inputs.get("output_name") or f"roop3d_{job.job_id}.mp4"
        out_rel = f"roop_3d/.runtime/outputs/{name}"
        final = owned(self.root, out_rel)
        final.parent.mkdir(parents=True, exist_ok=True)
        first, last = ck.segments[0], ck.segments[-1]
        part = final.with_name(final.name + ".part.mp4")
        has_audio = mux_source_audio(joined, target, part, start_seconds=first.start_seconds, end_seconds=last.end_seconds)
        expected = ck.frame_progress["total_frames"]
        report = self.verify_output(part, expected_frames=expected, fps=fps, expect_audio=has_audio)
        os.replace(part, final)
        for seg in ck.segments:  # the originals must have survived the join
            if not owned(self.root, seg.path).is_file():
                raise DomainError("INTERMEDIATE_LOST", f"segment {seg.index} disappeared during join", 500)
        output = {"output_path": out_rel, "segments_joined": len(ck.segments), "audio_sync_remuxed": has_audio,
                  "frames": report["frames"], "duration_seconds": report["duration_seconds"],
                  "encoder": ck.encoder, "segments_preserved": True, "verified": True}
        ck.state, ck.final_output = "COMPLETED", output
        self._save(ck)
        return output

    @staticmethod
    def verify_output(path: Path, *, expected_frames: int, fps: float, expect_audio: bool) -> dict[str, Any]:
        frames = count_video_packets(path)
        with av.open(str(path)) as container:
            video = container.streams.video[0]
            duration = float(container.duration) / 1_000_000 if container.duration else 0.0
            audio_streams = len(container.streams.audio)
            audio_seconds = 0.0
            if audio_streams:
                audio = container.streams.audio[0]
                audio_seconds = float(audio.duration * audio.time_base) if audio.duration else 0.0
        expected_seconds = expected_frames / fps
        problems = []
        if frames != expected_frames:
            problems.append(f"frame count {frames} != expected {expected_frames}")
        if abs(duration - expected_seconds) > max(0.15, 2 / fps):
            problems.append(f"duration {duration:.3f}s != expected {expected_seconds:.3f}s")
        if expect_audio and not audio_streams:
            problems.append("audio track missing")
        if audio_streams and audio_seconds and abs(audio_seconds - expected_seconds) > 0.25:
            problems.append(f"audio {audio_seconds:.3f}s out of step with video {expected_seconds:.3f}s")
        if problems:
            raise DomainError("OUTPUT_VERIFICATION_FAILED", "; ".join(problems), 500)
        return {"frames": frames, "duration_seconds": round(duration, 4), "audio_seconds": round(audio_seconds, 4),
                "video_codec": video.codec_context.name}

    def cleanup_segments(self, job_id: str) -> int:
        """Delete the job's segments -- only once a verified joined output exists."""
        ck = self.load_checkpoint(job_id)
        if ck is None or not ck.final_output or not ck.final_output.get("verified") \
                or not owned(self.root, ck.final_output["output_path"]).is_file():
            raise DomainError("CLEANUP_REFUSED", "Segments are kept until the joined output is verified", 409)
        shutil.rmtree(owned(self.root, f"{self.work_dir(job_id)}/segments"), ignore_errors=True)
        return len(ck.segments)
