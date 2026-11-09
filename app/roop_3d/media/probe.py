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
from roop_3d.color import codes
from roop_3d.color.description import (
    ColorDescription, HdrStaticMetadata, merge_hdr_metadata, parse_content_light, parse_mastering_display,
    pixel_format_traits, resolve_color)
from roop_3d.core.errors import DomainError

OPTIONS = {"protocol_whitelist": "", "enable_drefs": "0", "use_absolute_path": "0",
           "format_whitelist": "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,avi,mpegts"}


@contextmanager
def open_media(path: Path, *, require_video: bool = True, **kwargs) -> Iterator[av.container.InputContainer]:
    path = Path(path).absolute()
    try:
        if any(is_link(p) for p in (*reversed(path.parents), path)):
            raise DomainError("MEDIA_PATH_DENIED", "Linked input media is forbidden", 422)
        options = OPTIONS if require_video else {**OPTIONS, "format_whitelist": OPTIONS["format_whitelist"] + ",wav,mp3"}
        with path.open("rb") as handle, av.open(handle, mode="r", options=options, **kwargs) as container:
            if require_video and not container.streams.video:
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
    sample_format: str | None = None
    bit_rate: int | None = None


TEXT_SUBTITLE_CODECS = frozenset({"subrip", "srt", "ass", "ssa", "webvtt", "mov_text", "text", "subviewer",
                                  "subviewer1", "microdvd", "sami", "realtext", "jacosub", "mpl2", "vplayer",
                                  "stl", "pjs", "aqtitle"})


@dataclass(frozen=True)
class SubtitleInfo:
    index: int
    codec: str
    kind: str  # "text" | "bitmap"
    time_base: Fraction
    language: str | None
    title: str | None
    start_time: int | None
    duration: int | None
    default: bool
    forced: bool


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
    bit_depth: int = 8
    chroma: str = "unknown"
    sample_aspect_ratio: Fraction = Fraction(1)
    hdr: HdrStaticMetadata | None = None

    @property
    def color_space_name(self) -> str:
        return {1: "BT.709", 5: "BT.601 (BT.470BG)", 6: "BT.601 (SMPTE 170M)",
                9: "BT.2020 (non-constant luminance)", 0: "RGB (identity)"}.get(self.colorspace, "UNKNOWN")

    @property
    def color(self) -> ColorDescription:
        """Resolved colour interpretation: tagged fields as tagged, the rest by recorded convention."""
        return resolve_color(primaries=self.color_primaries, transfer=self.color_trc, matrix=self.colorspace,
                             range_=self.color_range, width=self.width, height=self.height,
                             pix_fmt=self.pixel_format, hdr=self.hdr)


@dataclass(frozen=True)
class ProbeResult:
    container: str
    duration: Fraction | None
    videos: tuple[VideoInfo, ...]
    audio: tuple[AudioInfo, ...]
    subtitles: tuple[SubtitleInfo, ...] = ()
    start_time: Fraction | None = None
    other_streams: int = 0


def probe(path: Path, sample_frames: int = 300, *, max_scan_seconds: float = 10) -> ProbeResult:
    if not 2 <= sample_frames <= 1000000:
        raise ValueError("Timing sample must contain 2..1000000 frames")
    if max_scan_seconds <= 0:
        raise ValueError("Probe scan time must be positive")
    with open_media(path) as container:
        audio = tuple(AudioInfo(s.index, s.codec_context.name, Fraction(s.time_base),
                                s.codec_context.sample_rate, s.codec_context.layout.name,
                                len(s.codec_context.layout.channels), s.metadata.get("language"),
                                s.start_time, s.duration, s.codec_context.format.name if s.codec_context.format else None,
                                s.bit_rate or None) for s in container.streams.audio)
        subtitles = tuple(SubtitleInfo(
            s.index, s.codec_context.name, "text" if s.codec_context.name in TEXT_SUBTITLE_CODECS else "bitmap",
            Fraction(s.time_base) if s.time_base else Fraction(1, 1000), s.metadata.get("language"),
            s.metadata.get("title"), s.start_time, s.duration,
            bool(s.disposition & av.stream.Disposition.default), bool(s.disposition & av.stream.Disposition.forced))
            for s in container.streams.subtitles)
        video = []
        for stream in container.streams.video:
            # Independent demuxers avoid cross-stream seek/discard effects.
            with open_media(path) as sampled:
                selected = sampled.streams[stream.index]
                ticks: list[Fraction] = []
                rotation, complete, missing = 0, True, False
                frame_tags = mastering = light = None
                began = time.monotonic()
                for index, frame in enumerate(sampled.decode(selected)):
                    rotation = frame.rotation
                    if index == 0:
                        frame_tags = (int(frame.colorspace), int(frame.color_primaries),
                                      int(frame.color_trc), int(frame.color_range))
                        for item in frame.side_data:
                            if item.type.name == "MASTERING_DISPLAY_METADATA":
                                mastering = parse_mastering_display(bytes(item))
                            elif item.type.name == "CONTENT_LIGHT_LEVEL":
                                light = parse_content_light(bytes(item))
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
            # In-band signalling (H.264/HEVC VUI) is only known after the first decode: container or codec
            # parameters win when they are specific, and the decoded frame fills in what they leave open.
            matrix, primaries, trc, rng = ctx.colorspace, ctx.color_primaries, ctx.color_trc, ctx.color_range
            if frame_tags is not None:
                matrix = matrix if not codes.is_unspecified("matrix", matrix) else frame_tags[0]
                primaries = primaries if not codes.is_unspecified("primaries", primaries) else frame_tags[1]
                trc = trc if not codes.is_unspecified("transfer", trc) else frame_tags[2]
                rng = rng if not codes.is_unspecified("range", rng) else frame_tags[3]
            traits = pixel_format_traits(ctx.pix_fmt)
            aspect = Fraction(stream.sample_aspect_ratio) if stream.sample_aspect_ratio else Fraction(1)
            video.append(VideoInfo(stream.index, ctx.name, Fraction(stream.time_base), stream.base_rate,
                                   stream.average_rate, stream.width, stream.height, ctx.pix_fmt,
                                   matrix, rng, primaries, trc,
                                   rotation, stream.frames or None, stream.start_time, stream.duration,
                                   vfr, len(ticks), complete, traits.bit_depth, traits.chroma, aspect,
                                   merge_hdr_metadata(mastering, light)))
        duration = Fraction(container.duration, av.time_base) if container.duration is not None else None
        start = Fraction(container.start_time, av.time_base) if container.start_time is not None else None
        other = len(container.streams) - len(container.streams.video) - len(audio) - len(subtitles)
        return ProbeResult(container.format.name, duration, tuple(video), audio, subtitles, start, other)
