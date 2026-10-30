"""Local-only media probing with exact timing and explicit sampling evidence."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import time
from typing import Iterator

import av

from roop_3d.audit.isolation import is_link
from roop_3d.core.errors import DomainError

OPTIONS = {"protocol_whitelist": "", "enable_drefs": "0", "use_absolute_path": "0",
           "format_whitelist": "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,avi,mpegts"}


@contextmanager
def open_media(path: Path, **kwargs) -> Iterator[av.container.InputContainer]:
    path = Path(path).absolute()
    try:
        if any(is_link(p) for p in (*reversed(path.parents), path)):
            raise DomainError("MEDIA_PATH_DENIED", "Linked input media is forbidden", 422)
        with path.open("rb") as handle, av.open(handle, mode="r", options=OPTIONS, **kwargs) as container:
            if not container.streams.video:
                raise DomainError("VIDEO_STREAM_MISSING", "Container has no video stream", 422)
            for stream in container.streams.video:
                if not 0 < stream.width * stream.height <= 16777216:
                    raise DomainError("VIDEO_DIMENSIONS_DENIED", "Video dimensions exceed decode budget", 422)
            yield container
    except FileNotFoundError:
        raise DomainError("MEDIA_NOT_FOUND", "Local media file does not exist", 404) from None
    except (av.FFmpegError, OSError):
        raise DomainError("MEDIA_DECODE_FAILED", "Local media could not be decoded", 422) from None


@dataclass(frozen=True)
class AudioInfo:
    index: int
    codec: str
    time_base: Fraction
    sample_rate: int
    layout: str
    channels: int
    language: str | None
    start_time: int | None
    duration: int | None


@dataclass(frozen=True)
class VideoInfo:
    index: int
    codec: str
    time_base: Fraction
    nominal_rate: Fraction | None
    average_rate: Fraction | None
    width: int
    height: int
    pixel_format: str | None
    colorspace: int
    color_range: int
    color_primaries: int
    color_trc: int
    rotation: int
    frame_count: int | None
    start_time: int | None
    duration: int | None
    vfr: bool | None
    sampled_frames: int
    sample_complete: bool

    @property
    def color_space_name(self) -> str:
        return {1: "BT.709", 5: "BT.601 (BT.470BG)", 6: "BT.601 (SMPTE 170M)"}.get(self.colorspace, "UNKNOWN")


@dataclass(frozen=True)
class ProbeResult:
    container: str
    duration: Fraction | None
    videos: tuple[VideoInfo, ...]
    audio: tuple[AudioInfo, ...]


def probe(path: Path, sample_frames: int = 300, *, max_scan_seconds: float = 10) -> ProbeResult:
    if not 2 <= sample_frames <= 1000000:
        raise ValueError("Timing sample must contain 2..1000000 frames")
    if max_scan_seconds <= 0:
        raise ValueError("Probe scan time must be positive")
    with open_media(path) as container:
        audio = tuple(AudioInfo(s.index, s.codec_context.name, Fraction(s.time_base),
                                s.codec_context.sample_rate, s.codec_context.layout.name,
                                len(s.codec_context.layout.channels), s.metadata.get("language"),
                                s.start_time, s.duration) for s in container.streams.audio)
        video = []
        for stream in container.streams.video:
            # Independent demuxers avoid cross-stream seek/discard effects.
            with open_media(path) as sampled:
                selected = sampled.streams[stream.index]
                ticks: list[Fraction] = []
                rotation, complete, missing = 0, True, False
                began = time.monotonic()
                for index, frame in enumerate(sampled.decode(selected)):
                    rotation = frame.rotation
                    if frame.pts is not None:
                        ticks.append(frame.pts * Fraction(frame.time_base))
                    else:
                        missing = True
                    if index + 1 >= sample_frames or time.monotonic() - began >= max_scan_seconds:
                        complete = False
                        break
                deltas = [b - a for a, b in zip(ticks, ticks[1:])]
                varying = bool(deltas) and max(deltas) - min(deltas) > Fraction(stream.time_base)
                vfr = True if varying else (False if complete and not missing and len(deltas) >= 2 else None)
            ctx = stream.codec_context
            video.append(VideoInfo(stream.index, ctx.name, Fraction(stream.time_base), stream.base_rate,
                                   stream.average_rate, stream.width, stream.height, ctx.pix_fmt,
                                   ctx.colorspace, ctx.color_range, ctx.color_primaries, ctx.color_trc,
                                   rotation, stream.frames or None, stream.start_time, stream.duration,
                                   vfr, len(ticks), complete))
        duration = Fraction(container.duration, av.time_base) if container.duration is not None else None
        return ProbeResult(container.format.name, duration, tuple(video), audio)
