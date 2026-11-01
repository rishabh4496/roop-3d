"""GOP-aligned stream-copy video segment splitting and lossless concat demuxer.

Features:
  - Exact Group of Pictures (GOP) keyframe boundary detection via PyAV / ffprobe.
  - Zero generation loss stream-copy splitting (-c copy) into timestamped chunks:
    chunk_0001_<ts>.mp4, chunk_0002_<ts>.mp4, etc.
  - Multi-segment rendering pipeline preserving intermediate chunk files.
  - Instant FFmpeg concat demuxer pass with original audio remuxing (zero video re-encoding).
"""
from __future__ import annotations

from fractions import Fraction
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Callable
import uuid

import av
from pydantic import ConfigDict, Field

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract


def find_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path:
        return path
    for candidate in [
        r"G:\pinokio\bin\miniforge\Library\bin\ffmpeg.exe",
        r"C:\pinokio\bin\miniforge\Library\bin\ffmpeg.exe",
    ]:
        if Path(candidate).is_file():
            return candidate
    return "ffmpeg"


def find_ffprobe() -> str:
    path = shutil.which("ffprobe")
    if path:
        return path
    for candidate in [
        r"G:\pinokio\bin\miniforge\Library\bin\ffprobe.exe",
        r"C:\pinokio\bin\miniforge\Library\bin\ffprobe.exe",
    ]:
        if Path(candidate).is_file():
            return candidate
    return "ffprobe"


class SegmentSlice(Contract):
    model_config = ConfigDict(frozen=False)
    slice_index: int  # 1-indexed: 1, 2, ...
    slice_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    chunk_path: str
    rendered_chunk_path: str | None = None
    start_seconds: float
    end_seconds: float | None = None
    duration: float = 0.0
    status: str = "PENDING"  # PENDING, RENDERING, COMPLETED, FAILED
    error: str | None = None
    retry_count: int = 0
    frame_count: int = 0


def detect_keyframe_timestamps(video_path: Path | str) -> tuple[list[float], float]:
    """Inspect video container and return sorted keyframe times (seconds) and total duration."""
    video_path = Path(video_path)
    if not video_path.is_file():
        raise DomainError("MEDIA_NOT_FOUND", f"Video file {video_path} does not exist", 404)

    keyframes: list[float] = []
    total_duration = 0.0

    try:
        with av.open(str(video_path)) as container:
            if not container.streams.video:
                raise DomainError("MEDIA_NO_VIDEO", "File contains no video stream", 422)
            stream = container.streams.video[0]
            tb = Fraction(stream.time_base) if stream.time_base else Fraction(1, 1000)
            start_pts = stream.start_time or 0

            if container.duration is not None:
                total_duration = float(container.duration) / 1000000.0
            elif stream.duration is not None:
                total_duration = float(stream.duration * tb)

            for packet in container.demux(stream):
                if packet.is_keyframe and packet.pts is not None:
                    sec = float((packet.pts - start_pts) * tb)
                    if sec >= 0.0 and (not keyframes or sec - keyframes[-1] >= 0.05):
                        keyframes.append(round(sec, 3))
    except Exception as err:
        # Fallback to ffprobe if container demux encounters codec-specific errors
        keyframes, total_duration = _detect_keyframes_ffprobe(video_path)

    if not keyframes:
        keyframes = [0.0]
    elif keyframes[0] > 0.05:
        keyframes.insert(0, 0.0)

    if total_duration <= 0.0 and keyframes:
        total_duration = keyframes[-1]

    return sorted(set(keyframes)), total_duration


def _detect_keyframes_ffprobe(video_path: Path) -> tuple[list[float], float]:
    ffprobe_exe = find_ffprobe()
    cmd = [
        ffprobe_exe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "packet=pts_time,flags",
        "-of", "csv=print_section=0", str(video_path)
    ]
    keyframes: list[float] = [0.0]
    duration = 0.0
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        for line in proc.stdout.splitlines():
            parts = line.strip().split(",")
            if len(parts) >= 2:
                pts_str, flags = parts[0], parts[1]
                try:
                    pts = float(pts_str)
                    if "K" in flags and pts >= 0.0:
                        keyframes.append(round(pts, 3))
                    duration = max(duration, pts)
                except ValueError:
                    continue
    except Exception:
        pass
    return sorted(set(keyframes)), duration


def compute_gop_aligned_cuts(
    keyframes: list[float],
    total_duration: float,
    max_segment_duration: float = 15.0,
) -> list[tuple[float, float]]:
    """Partition [0, total_duration] into intervals aligned exactly to keyframe boundaries."""
    if total_duration <= max_segment_duration or len(keyframes) <= 1:
        return [(0.0, round(total_duration, 3))]

    cuts: list[float] = [0.0]
    curr_target = max_segment_duration

    while curr_target < total_duration - 0.5:
        candidates = [k for k in keyframes if k > cuts[-1] + 0.5 and k <= curr_target + max_segment_duration * 0.5]
        if candidates:
            best = min(candidates, key=lambda k: abs(k - curr_target))
            cuts.append(best)
            curr_target = best + max_segment_duration
        else:
            after = [k for k in keyframes if k > cuts[-1] + 0.5]
            if after:
                cuts.append(after[0])
                curr_target = after[0] + max_segment_duration
            else:
                break

    if cuts[-1] < total_duration:
        cuts.append(round(total_duration, 3))

    intervals = []
    for i in range(len(cuts) - 1):
        s = round(cuts[i], 3)
        e = round(cuts[i + 1], 3)
        intervals.append((s, e))

    return intervals


def split_video_gop(
    video_path: Path | str,
    output_dir: Path | str,
    max_segment_duration: float = 15.0,
    timestamp: str | None = None,
) -> list[SegmentSlice]:
    """Split input video along exact GOP keyframe boundaries using FFmpeg stream-copy with zero generation loss."""
    video_path = Path(video_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = timestamp or str(int(time.time()))
    keyframes, total_duration = detect_keyframe_timestamps(video_path)
    cuts = compute_gop_aligned_cuts(keyframes, total_duration, max_segment_duration)
    ffmpeg_exe = find_ffmpeg()

    slices: list[SegmentSlice] = []

    for idx, (start_sec, end_sec) in enumerate(cuts, start=1):
        chunk_name = f"chunk_{idx:04d}_{ts}.mp4"
        chunk_file = output_dir / chunk_name

        # Exact keyframe stream copy:
        # Placing -ss before -i when seeking to keyframes enables instant seek with zero re-encode
        if start_sec <= 0.01:
            cmd = [
                ffmpeg_exe, "-y",
                "-i", str(video_path),
                "-t", str(end_sec),
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                str(chunk_file)
            ]
        else:
            duration = end_sec - start_sec
            cmd = [
                ffmpeg_exe, "-y",
                "-ss", str(start_sec),
                "-i", str(video_path),
                "-t", str(duration),
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                str(chunk_file)
            ]

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as err:
            raise DomainError("SPLIT_FAILED", f"FFmpeg stream copy split failed for {chunk_name}: {err.stderr[-300:] if err.stderr else 'unknown'}", 500)

        if not chunk_file.is_file() or chunk_file.stat().st_size == 0:
            raise DomainError("SPLIT_FAILED", f"Chunk file {chunk_name} was not created or is empty", 500)

        # Probe chunk frame count
        frame_count = 0
        try:
            with av.open(str(chunk_file)) as c:
                if c.streams.video:
                    frame_count = c.streams.video[0].frames or 0
        except Exception:
            pass

        dur = round(end_sec - start_sec, 3)
        slices.append(
            SegmentSlice(
                slice_index=idx,
                chunk_path=str(chunk_file.resolve().as_posix()),
                start_seconds=start_sec,
                end_seconds=end_sec,
                duration=dur,
                frame_count=frame_count,
            )
        )

    return slices


def has_audio_stream(path: Path | str) -> bool:
    try:
        with av.open(str(path)) as c:
            return bool(c.streams.audio)
    except Exception:
        return False


def concat_lossless(
    slice_paths: list[Path | str],
    source_audio_video: Path | str | None,
    output_path: Path | str,
    preserve_intermediates: bool = True,
) -> dict[str, Any]:
    """Execute instant FFmpeg concat demuxer stream-copy pass and remux original audio without a second video encode.

    Intermediate chunk files are preserved on disk.
    """
    if not slice_paths:
        raise DomainError("SEGMENTS_EMPTY", "No slice paths supplied for concatenation", 422)

    resolved_slices = [Path(p).resolve() for p in slice_paths]
    for p in resolved_slices:
        if not p.is_file():
            raise DomainError("SEGMENT_FILE_MISSING", f"Segment file {p.name} does not exist", 404)

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_exe = find_ffmpeg()
    token = uuid.uuid4().hex[:12]
    concat_list_file = output_path.parent / f"concat_{token}.txt"

    lines = [f"file '{p.as_posix()}'\n" for p in resolved_slices]
    concat_list_file.write_text("".join(lines), encoding="utf-8")

    audio_source = Path(source_audio_video).resolve() if source_audio_video else None
    has_audio = audio_source is not None and audio_source.is_file() and has_audio_stream(audio_source)

    try:
        if has_audio:
            # Concat demuxer stream copy video (0:v:0) + original source audio (1:a:0)
            # Try stream copy for both audio and video first
            cmd = [
                ffmpeg_exe, "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_list_file),
                "-i", str(audio_source),
                "-c:v", "copy",
                "-c:a", "copy",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                str(output_path),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                # If audio copy failed (codec mismatch), transcode audio with aac while preserving video stream copy
                cmd = [
                    ffmpeg_exe, "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", str(concat_list_file),
                    "-i", str(audio_source),
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-map", "0:v:0",
                    "-map", "1:a:0?",
                    str(output_path),
                ]
                subprocess.run(cmd, capture_output=True, text=True, check=True)
        else:
            # Video stream copy only
            cmd = [
                ffmpeg_exe, "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_list_file),
                "-c", "copy",
                str(output_path),
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as err:
        raise DomainError("CONCAT_FAILED", f"FFmpeg concat demuxer failed: {err.stderr[-300:] if err.stderr else 'unknown'}", 500)
    finally:
        concat_list_file.unlink(missing_ok=True)

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise DomainError("CONCAT_FAILED", "Concatenated output file was not created or is empty", 500)

    # Calculate total frames and verified audio
    total_frames = 0
    audio_present = False
    try:
        with av.open(str(output_path)) as c:
            if c.streams.video:
                total_frames = c.streams.video[0].frames or 0
            audio_present = bool(c.streams.audio)
    except Exception:
        pass

    # Intermediate chunks must be preserved
    if preserve_intermediates:
        for p in resolved_slices:
            if not p.is_file():
                raise DomainError("INTERMEDIATE_LOST", f"Intermediate chunk {p.name} was prematurely removed", 500)

    return {
        "output_path": output_path.as_posix(),
        "total_frames": total_frames,
        "segments_concatenated": len(resolved_slices),
        "preserved_chunks": [p.as_posix() for p in resolved_slices],
        "audio_sync_remuxed": audio_present,
    }


class SegmentDispatcher:
    """Orchestrates GOP-splitting, sequential/distributed slice rendering, and final concat."""

    def __init__(self, root: Path, work_dir: Path | None = None):
        self.root = root
        self.work_dir = work_dir or (root / "roop_3d/.runtime/segments")
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def prepare_slices(self, video_path: Path | str, max_segment_duration: float = 15.0) -> list[SegmentSlice]:
        return split_video_gop(video_path, self.work_dir, max_segment_duration=max_segment_duration)

    def execute_and_concat(
        self,
        video_path: Path | str,
        render_slice_fn: Callable[[SegmentSlice], str],
        output_path: Path | str,
        max_segment_duration: float = 15.0,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Split video, execute slice renderer on each slice preserving chunks, and perform lossless concat."""
        slices = self.prepare_slices(video_path, max_segment_duration=max_segment_duration)
        rendered_chunks: list[str] = []

        for idx, sl in enumerate(slices):
            sl.status = "RENDERING"
            try:
                out_chunk = render_slice_fn(sl)
                sl.rendered_chunk_path = out_chunk
                sl.status = "COMPLETED"
                rendered_chunks.append(out_chunk)
            except Exception as err:
                sl.status = "FAILED"
                sl.error = str(err)
                raise
            if on_progress:
                on_progress(idx + 1, len(slices))

        # Perform lossless concat and remux audio from original input video
        result = concat_lossless(rendered_chunks, source_audio_video=video_path, output_path=output_path, preserve_intermediates=True)
        result["slices"] = [s.model_dump() for s in slices]
        return result
