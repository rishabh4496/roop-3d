"""Validated local encode, all-track audio mux, and atomic output publication."""
from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction
from math import lcm
from pathlib import Path
from threading import Event
from typing import Callable, Literal
import os
import uuid
import json

import av
import numpy as np

from roop_3d.audit.isolation import sha256_file
from roop_3d.audit.project_paths import application_root
from roop_3d.core.errors import DomainError
from roop_3d.core.settings import runtime_path
from roop_3d.core.storage import atomic_json, owned, exclusive
from roop_3d.media.audio import AudioPolicy, audio_packets, plan_audio
from roop_3d.media.frame_map import FrameMap
from roop_3d.media.probe import open_media, probe
from roop_3d.media.timestamps import rescale
from roop_3d.media.video_reader import DecodedFrame, VideoReader, check_cancel


def _encoder(codec: str, hardware: str, width: int, height: int, base: Fraction) -> tuple[str, str]:
    if hardware == "auto":
        name = "h264_nvenc" if codec == "h264" else "hevc_nvenc"
        try:
            context = av.CodecContext.create(name, "w")
            context.width, context.height, context.pix_fmt, context.time_base = width, height, "yuv420p", base
            context.max_b_frames = 0
            context.open()
            frame = av.VideoFrame.from_ndarray(np.zeros((height, width, 3), np.uint8), format="bgr24")
            frame.pts, frame.time_base = 0, base
            packets = context.encode(frame) + context.encode(None)
            if not packets:
                raise RuntimeError("Hardware encoder produced no packet")
            decoder = av.CodecContext.create("h264" if codec == "h264" else "hevc", "r")
            decoded = []
            for packet in packets:
                decoded.extend(decoder.decode(packet))
            decoded.extend(decoder.decode(None))
            if not decoded or decoded[0].width != width or decoded[0].height != height:
                raise RuntimeError("Hardware output failed decode validation")
            return name, "VERIFIED_LOCAL"
        except (av.FFmpegError, RuntimeError, ValueError):
            pass
    return ("libx264" if codec == "h264" else "libx265"), "UNSUPPORTED" if hardware == "auto" else "UNKNOWN"


def encode_video(source: Path, filename: str, *, root: Path | None = None,
                 codec: Literal["h264", "hevc"] = "h264", hardware: Literal["auto", "software"] = "auto",
                 timing: Literal["preserve_source_timing", "explicit_cfr"] = "preserve_source_timing",
                 rate: Fraction | None = None, audio_policy: AudioPolicy = "copy",
                 start: Fraction | None = None, end: Fraction | None = None,
                 transform: Callable[[DecodedFrame], np.ndarray] | None = None,
                 cancel: Event | None = None, output_dir: str | None = None,
                 manifest_metadata: dict | None = None,
                 before_publish: Callable[[], None] | None = None) -> dict:
    """Trim times are absolute source seconds, start inclusive and end exclusive.

    A trim rebases both video and audio by exactly start. Otherwise source PTS,
    including a nonzero origin, are retained. The caller owns any transform.
    """
    root = root or application_root()
    if Path(filename).name != filename or Path(filename).suffix.lower() not in (".mp4", ".mov", ".mkv"):
        raise ValueError("Output must be a simple MP4, MOV, or MKV filename")
    if codec not in ("h264", "hevc") or hardware not in ("auto", "software"):
        raise ValueError("Invalid encoder policy")
    if timing not in ("preserve_source_timing", "explicit_cfr") or timing == "explicit_cfr" and (rate is None or rate <= 0):
        raise ValueError("Explicit CFR requires a positive rational rate")
    if any(value is not None and not isinstance(value, Fraction) for value in (start, end, rate)):
        raise ValueError("Rates and trim boundaries must be Fraction values")
    if start is not None and start < 0 or end is not None and end <= (start or 0):
        raise ValueError("Invalid trim interval")
    relative_dir = "roop_3d/outputs" if output_dir is None else output_dir
    if output_dir is None:
        output_directory = owned(root, relative_dir)
    else:
        try:
            output_directory = runtime_path(relative_dir, root)
        except ValueError:
            raise DomainError("OUTPUT_PATH_DENIED", "Output directory must stay inside the owned runtime", 422) from None
        relative_dir = output_directory.relative_to(root).as_posix()
    output_relative = (Path(relative_dir) / filename).as_posix()
    final = owned(root, output_relative)
    manifest_path = output_relative + ".manifest.json"
    owned(root, manifest_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary = owned(root, f"roop_3d/outputs/{token}.video{final.suffix}")
    muxed = owned(root, f"roop_3d/outputs/{token}.mux{final.suffix}")
    published = False
    with exclusive(root, f"roop_3d/outputs/.{filename}.lock"):
        if final.exists() or owned(root, manifest_path).exists():
            raise DomainError("OUTPUT_EXISTS", "Output and manifest must be new files", 409)
        frames = None
        audio = None
        try:
            check_cancel(cancel)
            digest = sha256_file(source)[1]
            info = probe(source)
            video = info.videos[0]
            plans = plan_audio(source, final.suffix.lower(), audio_policy, start=start, end=end, cancel=cancel)
            origin = start or Fraction(0)
            base = Fraction(1, lcm(video.time_base.denominator, origin.denominator,
                                   rate.numerator if timing == "explicit_cfr" else 1))
            mux_options = {"avoid_negative_ts": "disabled"}
            if final.suffix.lower() in (".mp4", ".mov"):
                # The default MOV movie timescale (1000) rounds edit-list
                # offsets even when the video track uses exact ticks.
                movie_scale = lcm(base.denominator, *(a.time_base.denominator for a in info.audio))
                mux_options.update(movie_timescale=str(movie_scale), video_track_timescale=str(base.denominator))
            width, height = (video.height, video.width) if video.rotation % 180 else (video.width, video.height)
            if width % 2 or height % 2 or video.rotation % 90:
                raise DomainError("ENCODE_DIMENSIONS", "YUV420 output requires even upright dimensions and right-angle rotation", 422)
            encoder, hardware_state = _encoder(codec, hardware, width, height, base)
            reader = VideoReader(source, hardware=hardware, cancel=cancel)
            source_count = 0
            frame_map = FrameMap(rate, origin) if timing == "explicit_cfr" else None

            def selected():
                nonlocal source_count
                iterator = reader.frames()
                try:
                    for frame in iterator:
                        source_count += 1
                        at = frame.timestamp.seconds
                        if start is not None and at < start:
                            continue
                        if end is not None and at >= end:
                            break
                        if end is not None:
                            duration = min(frame.duration, rescale((end - at).numerator, Fraction(1, (end - at).denominator), frame.timestamp.time_base))
                            frame = DecodedFrame(frame.index, frame.timestamp, duration, frame.key_frame, frame.image)
                        yield frame
                finally:
                    iterator.close()

            frames = frame_map.select(selected()) if frame_map else ((frame, frame.timestamp) for frame in selected())
            expected: list[Fraction] = []
            with av.open(str(temporary), "w", options=mux_options) as output:
                stream = output.add_stream(encoder, rate=rate or video.average_rate or Fraction(25))
                stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
                stream.time_base = stream.codec_context.time_base = base
                stream.codec_context.max_b_frames = 0
                stream.codec_context.thread_count = 1
                ctx = stream.codec_context
                ctx.colorspace, ctx.color_range = video.colorspace, video.color_range
                ctx.color_primaries, ctx.color_trc = video.color_primaries, video.color_trc
                stream.options = {"preset": "fast"} if "nvenc" not in encoder else {"preset": "p4"}
                output.metadata["comment"] = "Synthetic media processed locally by Roop 3D"
                for decoded, stamp in frames:
                    check_cancel(cancel)
                    image = transform(decoded) if transform else decoded.image
                    if not isinstance(image, np.ndarray) or image.shape != (height, width, 3) or image.dtype != np.uint8:
                        raise DomainError("ENCODE_FRAME_INVALID", "Transform must return upright uint8 BGR with unchanged dimensions", 422)
                    frame = av.VideoFrame.from_ndarray(image, format="bgr24")
                    conversion = {"dst_color_range": video.color_range} if video.color_range in (1, 2) else {}
                    if video.colorspace in (1, 4, 5, 6, 7):
                        conversion["dst_colorspace"] = video.colorspace
                    frame = frame.reformat(format="yuv420p", **conversion)
                    relative = stamp.seconds - origin
                    frame.pts, frame.time_base = rescale(relative.numerator, Fraction(1, relative.denominator), base), base
                    duration = Fraction(1, 1) / rate if frame_map else decoded.duration * decoded.timestamp.time_base
                    frame.duration = rescale(duration.numerator, Fraction(1, duration.denominator), base)
                    frame.colorspace, frame.color_range = video.colorspace, video.color_range
                    expected.append(relative)
                    for packet in stream.encode(frame):
                        output.mux(packet)
                if not expected:
                    raise DomainError("ENCODE_EMPTY", "No frames in selected interval", 422)
                for packet in stream.encode(None):
                    output.mux(packet)
            with open_media(temporary) as encoded, open_media(source) as original, av.open(
                    str(muxed), "w", options=mux_options) as output:
                target = output.add_stream_from_template(encoded.streams.video[0])
                output.metadata["comment"] = "Synthetic media processed locally by Roop 3D"
                audio = audio_packets(original, output, plans, start=start, end=end, cancel=cancel)
                # Prime audio to create every output stream before header emission.
                pending = next(audio, None)
                for packet in encoded.demux(encoded.streams.video[0]):
                    check_cancel(cancel)
                    if packet.dts is None:
                        continue
                    while pending is not None and (pending.dts or 0) * pending.time_base <= packet.dts * packet.time_base:
                        output.mux(pending)
                        pending = next(audio, None)
                    packet.stream = target
                    output.mux(packet)
                while pending is not None:
                    check_cancel(cancel)
                    output.mux(pending)
                    pending = next(audio, None)
            observed = probe(muxed)
            if len(observed.audio) != len(info.audio):
                raise DomainError("AUDIO_TRACK_LOSS", "Output audio track count differs from input", 422)
            verified_count = 0
            verification = VideoReader(muxed, hardware="software")
            for decoded in verification.frames():
                check_cancel(cancel)
                if verified_count >= len(expected) or abs(decoded.timestamp.seconds - expected[verified_count]) > video.time_base:
                    raise DomainError("OUTPUT_PTS_DRIFT", "Output PTS differ by more than one source time-base tick", 422)
                verified_count += 1
            if verified_count != len(expected):
                raise DomainError("OUTPUT_FRAME_LOSS", "Output frame count differs from encoded frame count", 422)
            with open_media(muxed) as playable:
                for audio_stream in playable.streams.audio:
                    # Reopen per track: decoding one stream consumes demux state.
                    with open_media(muxed) as track:
                        if next(track.decode(track.streams[audio_stream.index]), None) is None:
                            raise DomainError("AUDIO_EMPTY", "An output audio track cannot be decoded", 422)
            if sha256_file(source)[1] != digest:
                raise DomainError("MEDIA_CHANGED", "Input media changed during encoding", 409)
            try:
                metadata = json.loads(json.dumps(manifest_metadata or {}, allow_nan=False))
                if not isinstance(metadata, dict) or len(json.dumps(metadata).encode("utf-8")) > 65536:
                    raise ValueError("metadata must be a small JSON object")
            except (TypeError, ValueError):
                raise DomainError("MANIFEST_METADATA_INVALID", "Output manifest metadata is invalid", 422) from None
            manifest = {"schema_version": 1, "input_sha256": digest, "output_sha256": sha256_file(muxed)[1],
                        "output_path": output_relative, "manifest_path": manifest_path,
                        "encoder": encoder, "nvenc_state": hardware_state, "decoder": reader.backend,
                        "width": width, "height": height, "pixel_format": "yuv420p",
                        "nvdec_state": reader.hardware_state, "source_frames_read": source_count,
                        "encoded_frames": len(expected), "verified_frames": verified_count,
                        "source_time_base": str(video.time_base), "output_time_base": str(observed.videos[0].time_base),
                        "timing_policy": timing, "audio_policy": audio_policy, "audio_tracks": [asdict(p) for p in plans],
                        "trim": {"start": str(start) if start is not None else None, "end": str(end) if end is not None else None},
                        "frame_map": [asdict(entry) for entry in frame_map.entries] if frame_map else [],
                        "provenance": {"synthetic_media": True, "processor": "Roop 3D", "processing": "local",
                                       "transformed": transform is not None, "notice": "Synthetic media processed locally by Roop 3D"}}
            if metadata:
                manifest["stage12"] = metadata
            check_cancel(cancel)
            if before_publish is not None:
                before_publish()
            owned(root, final.relative_to(root).as_posix())
            os.replace(muxed, final)
            published = True
            atomic_json(root, manifest_path, manifest)
            return manifest
        except BaseException as error:
            if published:
                owned(root, final.relative_to(root).as_posix()).unlink(missing_ok=True)
            if isinstance(error, (av.FFmpegError, OSError)):
                raise DomainError("MEDIA_ENCODE_FAILED", "Local encode or mux failed", 422) from None
            raise
        finally:
            if frames is not None:
                frames.close()
            if audio is not None:
                audio.close()
            for path in (temporary, muxed):
                owned(root, path.relative_to(root).as_posix()).unlink(missing_ok=True)
