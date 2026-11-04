"""Explicit all-track audio planning and packet/sample-accurate local muxing."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import ceil
from pathlib import Path
from threading import Event
from typing import Iterator, Literal

import av
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.media.probe import open_media
from roop_3d.media.timestamps import rescale
from roop_3d.media.video_reader import check_cancel

AudioPolicy = Literal["copy", "aac", "pcm", "transcode", "drop"]
COMPATIBLE = {".mp4": {"aac", "mp3", "ac3", "eac3", "alac"},
              ".mov": {"aac", "mp3", "ac3", "alac", "pcm_s16le", "pcm_s24le"},
              ".mkv": {"aac", "mp3", "ac3", "eac3", "alac", "pcm_s16le", "pcm_s24le", "flac", "opus", "vorbis"}}


#: Standard layout for a channel count, used when a container stores only the count (PCM in Matroska, AVI,
#: WAV) and FFmpeg reports an unlabelled layout such as "2 channels", which no encoder accepts by name.
_DEFAULT_LAYOUTS = {1: "mono", 2: "stereo", 3: "3.0", 4: "4.0", 5: "5.0", 6: "5.1", 7: "6.1", 8: "7.1"}


def transcode_layout(layout: av.AudioLayout) -> str:
    """Layout name safe to hand to an encoder and resampler for this source layout."""
    count = len(layout.channels)
    if not layout.name.endswith(" channels"):
        return layout.name
    if count not in _DEFAULT_LAYOUTS:
        raise DomainError("AUDIO_LAYOUT_UNSUPPORTED", f"No standard channel layout for {count} unlabelled channels", 422)
    return _DEFAULT_LAYOUTS[count]


@dataclass(frozen=True)
class AudioPlan:
    source_index: int
    source_codec: str
    codec: str
    action: str
    reason: str


def plan_audio(source: Path, suffix: str, policy: AudioPolicy = "copy", *,
               start: Fraction | None = None, end: Fraction | None = None,
               cancel: Event | None = None) -> tuple[AudioPlan, ...]:
    if policy not in ("copy", "aac", "pcm", "transcode", "drop") or suffix not in COMPATIBLE:
        raise ValueError("Unsupported audio/container policy")
    if policy == "pcm" and suffix == ".mp4":
        raise DomainError("AUDIO_POLICY_INCOMPATIBLE", "PCM output requires MOV or MKV", 422)
    with open_media(source) as container:
        streams = list(container.streams.audio)
        if policy == "drop" and streams:
            raise DomainError("AUDIO_DROP_DENIED", "Stage 05 preserves every audio track", 422)
        aligned = {s.index: {"start": start is None, "end": end is None} for s in streams}
        if streams and (start is not None or end is not None):
            for packet in container.demux(streams):
                check_cancel(cancel)
                if packet.pts is None or not packet.size:
                    continue
                at = packet.pts * Fraction(packet.time_base)
                stop = at + packet.duration * Fraction(packet.time_base)
                if at == start:
                    aligned[packet.stream.index]["start"] = True
                if stop == end:
                    aligned[packet.stream.index]["end"] = True
        result = []
        for stream in streams:
            codec = stream.codec_context.name
            copy = policy == "copy" and codec in COMPATIBLE[suffix] and all(aligned[stream.index].values())
            chosen = codec if copy else ("pcm_s16le" if policy == "pcm" else "aac")
            reason = "codec and packet boundaries compatible" if copy else "explicit policy or incompatible codec/trim boundaries"
            result.append(AudioPlan(stream.index, codec, chosen, "copy" if copy else "transcode", reason))
        return tuple(result)


def audio_packets(container: av.container.InputContainer, output: av.container.OutputContainer,
                  plans: tuple[AudioPlan, ...], *, start: Fraction | None = None,
                  end: Fraction | None = None, cancel: Event | None = None) -> Iterator[av.Packet]:
    streams, mapping, resamplers = [], {}, {}
    origin = start or Fraction(0)
    for plan in plans:
        source = container.streams[plan.source_index]
        streams.append(source)
        if plan.action == "copy":
            target = output.add_stream_from_template(source)
        else:
            target = output.add_stream(plan.codec, rate=source.codec_context.sample_rate)
            target.layout = transcode_layout(source.codec_context.layout)
            target.bit_rate = max(320000, 128000 * len(source.codec_context.layout.channels))
            target.time_base = Fraction(1, target.codec_context.sample_rate)
            target.codec_context.time_base = target.time_base
            format = "fltp" if plan.codec == "aac" else "s16"
            resamplers[source.index] = av.AudioResampler(format=format, layout=target.layout, rate=target.codec_context.sample_rate)
        target.metadata.update(source.metadata)
        target.disposition = source.disposition
        mapping[source.index] = (plan, target)

    def encode_samples(frame: av.AudioFrame, target: av.audio.stream.AudioStream) -> Iterator[av.Packet]:
        check_cancel(cancel)
        if frame.pts is None:
            raise DomainError("AUDIO_TIMESTAMP_MISSING", "Audio samples have no timestamp", 422)
        base = Fraction(1, frame.sample_rate)
        # Exact sample boundaries; start rounds upward and end downward.
        at = frame.pts * Fraction(frame.time_base)
        lower = max(0, ceil((start - at) / base)) if start is not None else 0
        upper = min(frame.samples, (end - at) // base) if end is not None else frame.samples
        if upper <= lower:
            return
        data = frame.to_ndarray()
        channels = len(frame.layout.channels)
        factor = 1 if frame.format.is_planar else channels
        cropped = np.ascontiguousarray(data[:, lower * factor:upper * factor])
        clipped = av.AudioFrame.from_ndarray(cropped, format=frame.format.name, layout=frame.layout.name)
        clipped.sample_rate = frame.sample_rate
        clipped.time_base = base
        clipped.pts = rescale(frame.pts, Fraction(frame.time_base), base) + lower - rescale(origin.numerator, Fraction(1, origin.denominator), base)
        yield from target.encode(clipped)

    if streams:
        for packet in container.demux(streams):
            check_cancel(cancel)
            plan, target = mapping[packet.stream.index]
            if plan.action == "copy":
                if packet.pts is None or not packet.size:
                    continue
                at = packet.pts * Fraction(packet.time_base)
                if (start is not None and at < start) or (end is not None and at >= end):
                    continue
                offset = rescale(origin.numerator, Fraction(1, origin.denominator), Fraction(packet.time_base))
                packet.pts -= offset
                if packet.dts is not None:
                    packet.dts -= offset
                packet.stream = target
                yield packet
            else:
                for frame in packet.decode():
                    for converted in resamplers[plan.source_index].resample(frame):
                        yield from encode_samples(converted, target)
        for plan in plans:
            if plan.action != "copy":
                target = mapping[plan.source_index][1]
                for converted in resamplers[plan.source_index].resample(None):
                    yield from encode_samples(converted, target)
                yield from target.encode(None)
