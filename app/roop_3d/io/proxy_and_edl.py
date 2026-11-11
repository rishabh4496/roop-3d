# SPDX-License-Identifier: MIT
"""Local video proxies, filmstrips, CMX 3600 EDL interchange and render ETA."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Callable, Mapping

import numpy as np

from roop_3d.vfx_export import TimelinePlan, export_edl


def _rate(rate: str | int | float) -> tuple[int, int, float]:
    if isinstance(rate, str):
        parts = rate.split("/")
        if len(parts) not in (1, 2):
            raise ValueError("frame rate must be numeric or a rational such as 30000/1001")
        numerator = int(parts[0])
        denominator = int(parts[1]) if len(parts) == 2 else 1
    elif isinstance(rate, int):
        numerator, denominator = rate, 1
    else:
        # Preserve common broadcast rates instead of converting to an integer rate.
        common = {23.976: (24000, 1001), 29.97: (30000, 1001), 59.94: (60000, 1001)}
        numerator, denominator = common.get(round(float(rate), 3), (round(float(rate) * 1000), 1000))
    if numerator <= 0 or denominator <= 0:
        raise ValueError("frame rate must be positive")
    return numerator, denominator, numerator / denominator


def timecode_to_frame(value: str, frame_rate: str | int | float, *, drop_frame: bool | None = None) -> int:
    """Convert SMPTE HH:MM:SS:FF or HH:MM:SS;FF into a zero-based frame number."""
    text = value.strip()
    match = re.fullmatch(r"(\d{2,}):(\d{2}):(\d{2})([:;])(\d{2})", text)
    if not match:
        raise ValueError(f"invalid SMPTE timecode {value!r}")
    hours, minutes, seconds, separator, frames = match.groups()
    h, m, s, f = int(hours), int(minutes), int(seconds), int(frames)
    if m >= 60 or s >= 60:
        raise ValueError(f"invalid SMPTE timecode {value!r}")
    numerator, denominator, fps = _rate(frame_rate)
    nominal = int(round(fps))
    if f >= nominal:
        raise ValueError(f"frame field must be less than nominal frame rate {nominal}")
    is_drop = separator == ";" if drop_frame is None else bool(drop_frame)
    if is_drop and nominal not in (30, 60):
        raise ValueError("drop-frame timecode is supported for 29.97 or 59.94 fps")
    if is_drop:
        dropped = (2 if nominal == 30 else 4) * (h * 60 + m - (h * 60 + m) // 10)
        drop_per_minute = 2 if nominal == 30 else 4
        if m % 10 and s == 0 and f < drop_per_minute:
            raise ValueError("timecode names a frame skipped by drop-frame numbering")
        return ((h * 3600 + m * 60 + s) * nominal + f) - dropped
    return (h * 3600 + m * 60 + s) * nominal + f


def frame_to_timecode(frame: int, frame_rate: str | int | float, *, drop_frame: bool = False) -> str:
    if type(frame) is not int or frame < 0:
        raise ValueError("frame must be a nonnegative integer")
    _, _, fps = _rate(frame_rate)
    nominal = int(round(fps))
    label = frame
    if drop_frame:
        if nominal not in (30, 60):
            raise ValueError("drop-frame timecode is supported for 29.97 or 59.94 fps")
        drop = 2 if nominal == 30 else 4
        frames_per_10m = nominal * 600 - drop * 9
        frames_per_minute = nominal * 60 - drop
        ten_minute_blocks, remainder = divmod(frame, frames_per_10m)
        add = drop * 9 * ten_minute_blocks
        if remainder >= drop:
            add += drop * ((remainder - drop) // frames_per_minute)
        label += add
    f = label % nominal
    total_seconds = label // nominal
    s = total_seconds % 60
    m = total_seconds // 60 % 60
    h = total_seconds // 3600
    delimiter = ";" if drop_frame else ":"
    return f"{h:02d}:{m:02d}:{s:02d}{delimiter}{f:02d}"


@dataclass(frozen=True)
class EDLEvent:
    event_number: str
    reel: str
    track: str
    edit_type: str
    source_in: int
    source_out: int
    record_in: int
    record_out: int
    comment: str = ""


@dataclass(frozen=True)
class EDLDocument:
    title: str
    frame_rate: str
    drop_frame: bool
    events: tuple[EDLEvent, ...]
    swapped_ranges: tuple[tuple[int, int], ...]
    scene_cuts: tuple[int, ...]


def parse_edl(text: str, *, frame_rate: str | int | float | None = None) -> EDLDocument:
    """Parse common CMX 3600 CUT/DISSOLVE events and retain record-frame ranges."""
    if not isinstance(text, str) or len(text.encode("utf-8")) > 16 * 1024 * 1024:
        raise ValueError("EDL text must be a string no larger than 16 MiB")
    title = "Imported EDL"
    detected_rate = str(frame_rate or "25/1")
    drop = False
    events: list[EDLEvent] = []
    comments: list[str] = []
    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            comments.clear()
            continue
        upper = line.upper()
        if upper.startswith("TITLE:"):
            title = line.split(":", 1)[1].strip() or title
            continue
        if upper.startswith("FCM:"):
            drop = "DROP" in upper and "NON-DROP" not in upper
            continue
        if upper.startswith("* FRAME RATE:"):
            detected_rate = line.split(":", 1)[1].strip() or detected_rate
            continue
        if line.startswith("*"):
            comments.append(line[1:].strip())
            continue
        parts = line.split()
        if not parts[0].isdigit():
            continue
        if len(parts) < 8:
            raise ValueError(f"EDL event on line {line_no} has too few columns")
        event_no, reel, track, edit_type = parts[:4]
        timecode_start = 4
        if edit_type.upper() in {"D", "W", "K"}:
            if len(parts) < 9:
                raise ValueError(f"transition event on line {line_no} is missing duration or timecodes")
            timecode_start = 5  # transition duration precedes the four timecodes
        codes = parts[timecode_start:timecode_start + 4]
        if len(codes) != 4:
            raise ValueError(f"EDL event on line {line_no} is missing source/record timecodes")
        use_drop = drop or any(";" in code for code in codes)
        try:
            values = [timecode_to_frame(code, detected_rate, drop_frame=use_drop) for code in codes]
        except ValueError as error:
            raise ValueError(f"invalid EDL event on line {line_no}: {error}") from error
        if values[1] < values[0] or values[3] < values[2]:
            raise ValueError(f"EDL event on line {line_no} has reversed in/out points")
        events.append(EDLEvent(event_no, reel, track, edit_type.upper(), *values, comment=" | ".join(comments)))
        comments.clear()
    if not events:
        raise ValueError("EDL contains no supported events")
    ranges = sorted((event.record_in, event.record_out) for event in events
                    if event.reel.casefold() in {"swap", "swapped", "face_swap"}
                    or "swapped" in event.comment.casefold())
    cuts = sorted({event.record_in for event in events if event.track.upper().startswith("V")})
    return EDLDocument(title, detected_rate, drop, tuple(events), tuple(ranges), tuple(cuts))


class CalibratedRenderTimeEstimator:
    """Online render ETA with an eight-frame calibration window.

    The first eight observed frames establish a median frame cost. Later frames
    update a bounded recent window. ETA remains unavailable until calibration
    completes and is recomputed from measured render frames thereafter.
    """

    def __init__(self, total_frames: int | None = None, *, warmup_frames: int = 8, window: int = 64):
        if total_frames is not None and (type(total_frames) is not int or total_frames < 1):
            raise ValueError("total_frames must be a positive integer or None")
        if type(warmup_frames) is not int or warmup_frames < 1 or type(window) is not int or window < warmup_frames:
            raise ValueError("warmup_frames must be positive and window must be at least warmup_frames")
        self.total_frames = total_frames
        self.warmup_frames = warmup_frames
        self.window = window
        self._durations: list[float] = []
        self._started_at = time.monotonic()
        self._elapsed_observed = 0.0

    @property
    def completed_frames(self) -> int:
        return len(self._durations)

    @property
    def calibrated(self) -> bool:
        return self.completed_frames >= self.warmup_frames

    def observe_frame(self, duration_seconds: float, *, total_frames: int | None = None) -> dict[str, Any]:
        duration = float(duration_seconds)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("frame duration must be finite and nonnegative")
        if total_frames is not None:
            if type(total_frames) is not int or total_frames < self.completed_frames + 1:
                raise ValueError("total_frames must be an integer no smaller than the new completion count")
            self.total_frames = total_frames
        self._durations.append(max(duration, 1e-6))
        self._durations = self._durations[-self.window:]
        self._elapsed_observed += duration
        return self.snapshot()

    def snapshot(self, *, total_frames: int | None = None) -> dict[str, Any]:
        if total_frames is not None:
            if type(total_frames) is not int or total_frames < self.completed_frames:
                raise ValueError("total_frames must be an integer no smaller than completed_frames")
            self.total_frames = total_frames
        remaining = None if self.total_frames is None else max(0, self.total_frames - self.completed_frames)
        elapsed = max(self._elapsed_observed, time.monotonic() - self._started_at)
        if not self.calibrated:
            return {"calibrated": False, "warmup_remaining": self.warmup_frames - self.completed_frames,
                    "completed_frames": self.completed_frames, "total_frames": self.total_frames,
                    "elapsed_seconds": elapsed, "seconds_per_frame": None, "eta_seconds": None,
                    "estimated_total_seconds": None}
        seconds_per_frame = float(np.median(self._durations))
        eta = None if remaining is None else float(seconds_per_frame * remaining)
        return {"calibrated": True, "warmup_remaining": 0, "completed_frames": self.completed_frames,
                "total_frames": self.total_frames, "elapsed_seconds": elapsed,
                "seconds_per_frame": seconds_per_frame, "eta_seconds": eta,
                "estimated_total_seconds": None if eta is None else elapsed + eta}


class VideoProxyAndEDLManager:
    """Create local FFmpeg proxy/contact sheet assets and exchange CMX EDLs."""

    def __init__(self, *, ffmpeg: str | None = None,
                 runner: Callable[..., Any] = subprocess.run, timeout_seconds: float = 900):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if ffmpeg is None:
            from roop_3d.segment_dispatcher import find_ffmpeg
            ffmpeg = find_ffmpeg()
        self.ffmpeg = str(ffmpeg)
        self.runner = runner
        self.timeout_seconds = float(timeout_seconds)

    @staticmethod
    def _local_file(value: str | Path) -> Path:
        path = Path(value).absolute()
        if not path.is_file() or any(part.is_symlink() for part in (*reversed(path.parents), path)):
            raise ValueError("video source must be a regular local file")
        return path

    def _run(self, command: list[str]) -> None:
        try:
            result = self.runner(command, capture_output=True, text=True, timeout=self.timeout_seconds, check=False)
        except FileNotFoundError as error:
            raise RuntimeError("FFmpeg is unavailable for proxy generation") from error
        if result.returncode != 0:
            detail = (getattr(result, "stderr", "") or "").strip()
            raise RuntimeError(f"FFmpeg failed to generate import media{': ' + detail[-1200:] if detail else ''}")

    def create_proxy(self, source: str | Path, destination: str | Path, *, max_edge: int = 960,
                     crf: int = 30) -> Path:
        src, dest = self._local_file(source), Path(destination).absolute()
        if dest.suffix.casefold() not in {".mp4", ".mov", ".mkv"} or max_edge < 256 or not 18 <= crf <= 40:
            raise ValueError("proxy output, max_edge or CRF is invalid")
        dest.parent.mkdir(parents=True, exist_ok=True)
        filt = f"scale={max_edge}:{max_edge}:force_original_aspect_ratio=decrease:force_divisible_by=2"
        self._run([self.ffmpeg, "-y", "-v", "error", "-i", str(src), "-map", "0:v:0", "-an", "-sn", "-dn",
                   "-vf", filt, "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", str(dest)])
        if not dest.is_file() or dest.stat().st_size == 0:
            raise RuntimeError("FFmpeg did not create a proxy file")
        return dest

    def create_filmstrip(self, source: str | Path, destination: str | Path, *, duration_seconds: float,
                         count: int = 16, thumb_width: int = 240, columns: int = 4) -> Path:
        src, dest = self._local_file(source), Path(destination).absolute()
        if not np.isfinite(duration_seconds) or duration_seconds <= 0 or not 1 <= count <= 64:
            raise ValueError("duration_seconds must be positive and count must be 1..64")
        if not 64 <= thumb_width <= 1024 or not 1 <= columns <= count:
            raise ValueError("thumb_width must be 64..1024 and columns must be within 1..count")
        dest.parent.mkdir(parents=True, exist_ok=True)
        fps_filter = f"fps={count}/{float(duration_seconds):.9f}"
        filt = f"{fps_filter},scale={thumb_width}:-2,tile={columns}x{math.ceil(count / columns)}:nb_frames={count}"
        self._run([self.ffmpeg, "-y", "-v", "error", "-i", str(src), "-map", "0:v:0", "-an", "-vf", filt,
                   "-frames:v", "1", "-q:v", "3", str(dest)])
        if not dest.is_file() or dest.stat().st_size == 0:
            raise RuntimeError("FFmpeg did not create a filmstrip image")
        return dest

    def prepare_import(self, source: str | Path, asset_directory: str | Path, *, duration_seconds: float | None,
                       asset_key: str | None = None, count: int = 16) -> dict[str, Any]:
        src = self._local_file(source)
        key = asset_key or re.sub(r"[^A-Za-z0-9_-]+", "_", src.stem).strip("_")[:80] or "video"
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", key):
            raise ValueError("asset_key may contain only letters, numbers, underscores and hyphens")
        root = Path(asset_directory).absolute()
        root.mkdir(parents=True, exist_ok=True)
        outcomes: dict[str, Any] = {"proxy": None, "filmstrip": None, "errors": {}}

        def filmstrip() -> Path:
            if duration_seconds is None or not np.isfinite(duration_seconds) or duration_seconds <= 0:
                raise ValueError("filmstrip requires a known positive video duration")
            return self.create_filmstrip(src, root / "filmstrips" / f"{key}.jpg",
                                         duration_seconds=duration_seconds, count=count)

        for name, action in (
            ("proxy", lambda: self.create_proxy(src, root / "proxies" / f"{key}.mp4")),
            ("filmstrip", filmstrip),
        ):
            try:
                outcomes[name] = str(action())
            except (OSError, RuntimeError, ValueError) as error:
                outcomes["errors"][name] = str(error)
        outcomes["state"] = "ready" if all(outcomes[name] for name in ("proxy", "filmstrip")) else "partial"
        return outcomes

    @staticmethod
    def export_edl(plan: TimelinePlan | dict[str, Any], *, drop_frame: bool = False) -> dict[str, str]:
        return export_edl(plan, drop_frame=drop_frame)

    @staticmethod
    def import_edl(text: str, *, frame_rate: str | int | float | None = None) -> EDLDocument:
        return parse_edl(text, frame_rate=frame_rate)

    def export_to_directory(self, plan: TimelinePlan | dict[str, Any], directory: str | Path, *,
                            drop_frame: bool = False) -> list[Path]:
        output = Path(directory).absolute()
        output.mkdir(parents=True, exist_ok=True)
        written = []
        for name, content in self.export_edl(plan, drop_frame=drop_frame).items():
            path = output / name
            path.write_text(content, encoding="utf-8", newline="\n")
            written.append(path)
        return written
