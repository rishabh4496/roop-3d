"""Validation of media files: inputs (``validate_media``) and finished outputs (``validate_output``).

Both return structured evidence rather than a boolean. Input validation never refuses a file for being
unusual -- it reports what it found, with a severity -- so the caller can decide. Output validation
raises a ``DomainError`` with a specific code for anything that would make the file wrong.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from threading import Event
from typing import Callable

import av

from roop_3d.color import codes
from roop_3d.core.errors import DomainError
from roop_3d.media.probe import ProbeResult, open_media, probe
from roop_3d.media.video_reader import VideoReader, check_cancel


@dataclass(frozen=True)
class Issue:
    severity: str  # "error" | "warning" | "info"
    code: str
    message: str

    def as_dict(self) -> dict:
        return {"severity": self.severity, "code": self.code, "message": self.message}


def validate_media(path: Path, *, deep: bool = False, max_deep_frames: int = 20000) -> dict:
    """Structured findings for an input file. ``deep`` decodes every frame and audio packet."""
    info = probe(path)
    video = info.videos[0]
    color = video.color
    issues: list[Issue] = []

    def add(severity: str, code: str, message: str) -> None:
        issues.append(Issue(severity, code, message))

    if video.rotation % 90:
        add("error", "ROTATION_NOT_RIGHT_ANGLE", f"Display rotation {video.rotation} is not a multiple of 90 degrees")
    elif video.rotation:
        add("info", "ROTATION_METADATA", f"Display rotation {video.rotation} degrees is applied to produce upright pixels")
    if video.vfr:
        add("info", "VARIABLE_FRAME_RATE", "Variable frame rate: source PTS are preserved, never resampled by default")
    elif video.vfr is None:
        add("warning", "FRAME_RATE_UNVERIFIED", "Frame timing could not be fully sampled; constant frame rate is not asserted")
    if video.start_time not in (None, 0):
        add("info", "NONZERO_START", f"Video starts at {float(video.start_time * video.time_base):.6f}s; the origin is preserved")
    for name in color.assumed_fields:
        add("warning", "COLOR_ASSUMED", f"{name} is not tagged in the file ({color.provenance[name]})")
    if color.is_hdr:
        add("warning", "HDR_INPUT", f"{color.dynamic_range}: rendering tone-maps to SDR BT.709 unless HDR passthrough is chosen "
                                   "for unmodified media")
        if color.hdr is None or not color.hdr.has_mastering_display:
            add("info", "HDR_METADATA_ABSENT", "No mastering-display metadata: source peak is assumed to be 1000 cd/m2")
    elif color.wide_gamut:
        add("info", "WIDE_GAMUT_SDR", "Wide-gamut SDR: frames keep the file's own primaries; no gamut conversion is applied")
    if color.bit_depth > 8:
        add("info", "HIGH_BIT_DEPTH", f"{color.bit_depth}-bit source: face processing runs at 8 bits; "
                                      "the media layer decodes and encodes the higher depth")
    if video.width % 2 or video.height % 2:
        add("warning", "ODD_DIMENSIONS", "Odd dimensions: 4:2:0 outputs cannot be written without cropping")
    if len(info.audio) > 1:
        add("info", "MULTIPLE_AUDIO", f"{len(info.audio)} audio tracks: all are preserved")
    video_start = (video.start_time or 0) * video.time_base
    for track in info.audio:
        if track.start_time is not None:
            offset = track.start_time * track.time_base - video_start
            if abs(offset) > Fraction(1, 10):
                add("warning", "AUDIO_VIDEO_OFFSET", f"Audio track {track.index} starts {float(offset):+.3f}s from the video; "
                                                     "the offset is preserved")
    dropped = [s for s in info.subtitles if s.codec not in ("mov_text", "subrip", "srt", "ass", "ssa", "webvtt",
                                                            "dvd_subtitle", "hdmv_pgs_subtitle", "dvb_subtitle")]
    for track in dropped:
        add("warning", "SUBTITLE_UNSUPPORTED", f"Subtitle track {track.index} ({track.codec}) cannot be preserved")
    if len(info.videos) > 1:
        add("warning", "MULTIPLE_VIDEO_STREAMS", f"{len(info.videos)} video streams: only the first is processed")
    if info.other_streams:
        add("info", "OTHER_STREAMS", f"{info.other_streams} data/attachment stream(s) are not carried into outputs")

    deep_report = None
    if deep:
        deep_report = _deep(path, info, issues, max_deep_frames)

    return {"container": info.container, "duration_seconds": None if info.duration is None else float(info.duration),
            "video": {"codec": video.codec, "width": video.width, "height": video.height, "pixel_format": video.pixel_format,
                      "bit_depth": video.bit_depth, "chroma": video.chroma, "rotation": video.rotation,
                      "time_base": str(video.time_base), "average_rate": str(video.average_rate) if video.average_rate else None,
                      "vfr": video.vfr, "color": color.as_dict()},
            "audio": [{"stream": a.index, "codec": a.codec, "sample_rate": a.sample_rate, "layout": a.layout,
                       "language": a.language} for a in info.audio],
            "subtitles": [{"stream": s.index, "codec": s.codec, "kind": s.kind, "language": s.language}
                          for s in info.subtitles],
            "deep": deep_report, "issues": [i.as_dict() for i in issues],
            "ok": not any(i.severity == "error" for i in issues)}


def _deep(path: Path, info: ProbeResult, issues: list[Issue], limit: int) -> dict:
    previous: Fraction | None = None
    count = 0
    bad_order = missing = 0
    reader = VideoReader(path, hardware="software", hdr_policy="raw")
    complete = True
    try:
        for decoded in reader.frames(native=True):
            if count >= limit:
                complete = False
                break
            count += 1
            at = decoded.timestamp.seconds
            if previous is not None and at <= previous:
                bad_order += 1
            previous = at
    except DomainError as error:
        issues.append(Issue("error", error.code, error.message))
        missing = 1
    declared = info.videos[0].frame_count
    if bad_order:
        issues.append(Issue("error", "PTS_NOT_INCREASING", f"{bad_order} frame(s) have non-increasing PTS"))
    if complete and declared and declared != count:
        issues.append(Issue("warning", "FRAME_COUNT_MISMATCH", f"Container declares {declared} frames, decoding found {count}"))
    audio_errors = 0
    for track in info.audio:
        try:
            with open_media(path) as container:
                samples = sum(frame.samples for frame in container.decode(container.streams[track.index]))
            if samples == 0:
                audio_errors += 1
                issues.append(Issue("error", "AUDIO_EMPTY", f"Audio track {track.index} decodes to no samples"))
        except (av.FFmpegError, DomainError):
            audio_errors += 1
            issues.append(Issue("error", "AUDIO_DECODE_FAILED", f"Audio track {track.index} cannot be decoded"))
    return {"frames_decoded": count, "complete": complete, "declared_frames": declared,
            "pts_order_violations": bad_order, "decode_errors": missing, "audio_errors": audio_errors}


@dataclass
class OutputExpectation:
    frame_times: list[Fraction]
    video_end: Fraction
    width: int
    height: int
    pix_fmt: str | None
    tags: dict
    sample_aspect_ratio: Fraction
    display_rotation: int = 0
    audio_tracks: int = 0
    subtitle_tracks: int = 0
    pts_tolerance: Fraction = Fraction(0)
    notes: list = field(default_factory=list)


def validate_output(path: Path, expected: OutputExpectation, *, cancel: Event | None = None,
                    on_frame: Callable[[int, av.VideoFrame], None] | None = None) -> dict:
    """Decode the finished file completely and compare every property that must not have changed.

    ``on_frame(index, frame)`` sees every decoded frame once, so callers can compare pixels without a second pass.
    """
    observed = probe(path)
    video = observed.videos[0]
    if (video.width, video.height) != (expected.width, expected.height):
        raise DomainError("OUTPUT_DIMENSIONS", "Output dimensions differ from the encoded frames", 422)
    # H.264 reports full-range 4:2:0 as the deprecated yuvj420p; the range tag is checked separately.
    if expected.pix_fmt and (video.pixel_format or "").replace("yuvj", "yuv") != expected.pix_fmt.replace("yuvj", "yuv"):
        raise DomainError("OUTPUT_PIXEL_FORMAT", f"Output pixel format is {video.pixel_format}, expected {expected.pix_fmt}", 422)
    got = {"primaries": video.color_primaries, "transfer": video.color_trc, "matrix": video.colorspace,
           "range": video.color_range}
    if got != expected.tags:
        raise DomainError("OUTPUT_COLOR_TAGS", f"Output colour tags {got} differ from the intended {expected.tags}", 422)
    if video.sample_aspect_ratio != expected.sample_aspect_ratio:
        raise DomainError("OUTPUT_ASPECT_DRIFT", "Output pixel aspect differs from the upright source", 422)
    if len(observed.audio) != expected.audio_tracks:
        raise DomainError("AUDIO_TRACK_LOSS", "Output audio track count differs from input", 422)
    if len(observed.subtitles) != expected.subtitle_tracks:
        raise DomainError("SUBTITLE_TRACK_LOSS", "Output subtitle track count differs from the plan", 422)

    count, worst = 0, Fraction(0)
    rotation_seen = None
    reader = VideoReader(path, hardware="software", hdr_policy="raw")
    iterator = reader.frames(native=True)
    try:
        for decoded in iterator:
            check_cancel(cancel)
            if count >= len(expected.frame_times):
                raise DomainError("OUTPUT_FRAME_LOSS", "Output contains more frames than were encoded", 422)
            error = abs(decoded.timestamp.seconds - expected.frame_times[count])
            worst = max(worst, error)
            if error > expected.pts_tolerance:
                raise DomainError("OUTPUT_PTS_DRIFT", "Output PTS differ by more than one source time-base tick", 422)
            if rotation_seen is None:
                rotation_seen = decoded.frame.rotation
            if on_frame is not None:
                on_frame(count, decoded.frame)
            count += 1
    finally:
        iterator.close()
    if count != len(expected.frame_times):
        raise DomainError("OUTPUT_FRAME_LOSS", "Output frame count differs from encoded frame count", 422)
    if (rotation_seen or 0) % 360 != expected.display_rotation % 360:
        raise DomainError("OUTPUT_ROTATION", f"Output display rotation {rotation_seen} differs from the plan "
                                             f"({expected.display_rotation})", 422)
    # Frame PTS were verified exactly above. Container duration is edit-list / start-offset dependent, so it is
    # checked for the failure that matters (a truncated tail) and otherwise reported.
    clip = expected.video_end - expected.frame_times[0]
    duration_error = None
    if observed.duration is not None:
        if observed.duration + Fraction(1, 10) < clip:
            raise DomainError("OUTPUT_DURATION", f"Output container is {float(observed.duration):.3f}s but the encoded "
                                                 f"video alone is {float(clip):.3f}s", 422)
        duration_error = observed.duration - clip
    for stream in observed.audio:
        with open_media(path) as track:
            if next(track.decode(track.streams[stream.index]), None) is None:
                raise DomainError("AUDIO_EMPTY", "An output audio track cannot be decoded", 422)
    return {"frames": count, "max_pts_error_seconds": float(worst), "video_duration_error_seconds":
            None if duration_error is None else float(duration_error), "pixel_format": video.pixel_format,
            "tags": got, "display_rotation": rotation_seen or 0, "audio_tracks": len(observed.audio),
            "subtitle_tracks": len(observed.subtitles), "decoded_completely": True}
