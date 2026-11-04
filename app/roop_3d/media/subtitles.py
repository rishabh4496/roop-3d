"""Subtitle stream preservation: packet copy where the output container can hold the codec, otherwise an
explicit, recorded drop. This PyAV build cannot open the text subtitle encoders (srt / webvtt / mov_text),
so there is no transcoding; nothing is converted, mislabelled or silently lost.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from threading import Event
from typing import Iterator, Literal

import av

from roop_3d.core.errors import DomainError
from roop_3d.media.probe import SubtitleInfo
from roop_3d.media.timestamps import rescale
from roop_3d.media.video_reader import check_cancel

SubtitlePolicy = Literal["copy", "require", "drop"]

#: Codecs each container can carry as a straight packet copy (FFmpeg muxer capabilities, verified in tests).
COPYABLE = {
    ".mp4": frozenset({"mov_text"}),
    ".mov": frozenset({"mov_text"}),
    ".mkv": frozenset({"subrip", "srt", "ass", "ssa", "webvtt", "dvd_subtitle", "hdmv_pgs_subtitle", "dvb_subtitle"}),
}


@dataclass(frozen=True)
class SubtitlePlan:
    source_index: int
    codec: str
    kind: str
    language: str | None
    action: str  # "copy" | "drop"
    reason: str


def plan_subtitles(tracks: tuple[SubtitleInfo, ...], suffix: str, policy: SubtitlePolicy = "copy") -> tuple[SubtitlePlan, ...]:
    if policy not in ("copy", "require", "drop"):
        raise ValueError("Subtitle policy must be 'copy', 'require' or 'drop'")
    plans = []
    for track in tracks:
        if policy == "drop":
            action, reason = "drop", "subtitle policy is drop"
        elif track.codec in COPYABLE.get(suffix, frozenset()):
            action, reason = "copy", f"{track.codec} can be stream-copied into {suffix}"
        else:
            action = "drop"
            reason = f"{suffix} cannot hold {track.codec} subtitles and this build has no subtitle transcoder"
        plans.append(SubtitlePlan(track.index, track.codec, track.kind, track.language, action, reason))
    if policy == "require":
        lost = [p for p in plans if p.action == "drop"]
        if lost:
            raise DomainError("SUBTITLE_NOT_PRESERVABLE",
                              f"Subtitle stream(s) {[p.source_index for p in lost]} cannot be preserved in {suffix}: "
                              f"{lost[0].reason}", 422)
    return tuple(plans)


def subtitle_packets(container: av.container.InputContainer, output: av.container.OutputContainer,
                     plans: tuple[SubtitlePlan, ...], *, start: Fraction | None = None, end: Fraction | None = None,
                     cancel: Event | None = None) -> Iterator[av.Packet]:
    """Yield retimed subtitle packets (trim origin subtracted). Output streams are created on first ``next``."""
    mapping = {}
    for plan in plans:
        if plan.action != "copy":
            continue
        source = container.streams[plan.source_index]
        target = output.add_stream_from_template(source)
        target.metadata.update(source.metadata)
        target.disposition = source.disposition
        mapping[plan.source_index] = target
    if not mapping:
        return
    origin = start or Fraction(0)
    for packet in container.demux([container.streams[i] for i in mapping]):
        check_cancel(cancel)
        if packet.pts is None or not packet.size:
            continue
        at = packet.pts * Fraction(packet.time_base)
        if (start is not None and at < start) or (end is not None and at >= end):
            continue
        offset = rescale(origin.numerator, Fraction(1, origin.denominator), Fraction(packet.time_base))
        packet.pts -= offset
        if packet.dts is not None:
            packet.dts -= offset
        packet.stream = mapping[packet.stream.index]
        yield packet
