"""Validated local encode, all-track audio / subtitle mux, and atomic output publication.

Two frame paths share one muxer and one validator:

* **RGB path** (a ``transform`` is applied, or the profile is 8-bit): 8-bit BGR in, explicit YCbCr matrix and
  range out, tagged with the colour the pipeline actually produced.
* **Native path** (no transform, profile allows it): decoded YCbCr planes are re-encoded without an RGB
  round trip, which is the only way PQ/HLG material can pass through unmodified.

Nothing is promised that was not verified: the finished file is decoded completely and checked for PTS,
frame count, dimensions, pixel format, colour tags, rotation, audio / subtitle track count, fidelity
against the encoder input, and audio payload / signal synchronisation.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, dataclass
from fractions import Fraction
from heapq import merge as merge_sorted
from math import lcm
from pathlib import Path
from threading import Event
from typing import Callable, Literal
import json
import os
import shutil
import uuid

import av
import numpy as np

from roop_3d.audit.isolation import sha256_file
from roop_3d.audit.project_paths import application_root
from roop_3d.color import codes
from roop_3d.color.description import ColorDescription, pixel_format_traits, working_color
from roop_3d.core.errors import DomainError
from roop_3d.core.settings import runtime_path
from roop_3d.core.storage import atomic_json, owned, exclusive
from roop_3d.media import av_sync, fidelity
from roop_3d.media.audio import AudioPolicy, audio_packets, plan_audio
from roop_3d.media.convert import RgbFrameConverter, clone_frame, promote_planes
from roop_3d.media.export_profiles import ExportProfile, SEQUENCE_FORMATS, get_profile, legacy_profile
from roop_3d.media.frame_map import FrameMap
from roop_3d.media.probe import open_media, probe
from roop_3d.media.sequence import (
    SequenceReader, SequenceWriter, discover_sequence, sequence_color, sequence_digest, verify_sequence, write_sidecar)
from roop_3d.media.subtitles import SubtitlePolicy, plan_subtitles, subtitle_packets
from roop_3d.media.timestamps import rescale
from roop_3d.media.validation import OutputExpectation, validate_output
from roop_3d.media.video_reader import DecodedFrame, VideoReader, check_cancel
from roop_3d.media.finishing import Finisher, FinishingOptions
from roop_3d.media.provenance import recipe_claim, authenticate_manifest, sign_c2pa
from roop_3d.reliability.disk import is_disk_full
from roop_3d.reliability.ledger import LEDGER

AV_SYNC_WINDOW_SECONDS = 15.0
# Signal-lag bounds in seconds. A stream copy is bit-identical, so its lag is zero. Re-encoded audio (AAC with
# its encoder delay compensated by the MP4 edit list, or PCM) measured 0.0 ms at the 0.125 ms resolution of the
# 8 kHz correlation; 2 ms leaves headroom for that resolution without hiding a real offset.
COPY_LAG_TOLERANCE = 0.0005
TRANSCODE_LAG_TOLERANCE = 0.002
#: Near-lossless defaults for an encoder a profile switches to (colour-preserving HDR passthrough -> HEVC Main10).
SWITCHED_ENCODER_OPTIONS = {"libx265": {"preset": "slow", "crf": "8", "x265-params": "log-level=none"}}
FFV1_NATIVE = frozenset({"yuv420p", "yuv422p", "yuv444p", "yuv420p10le", "yuv422p10le", "yuv444p10le",
                         "yuv420p12le", "yuv422p12le", "yuv444p12le"})


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


@dataclass(frozen=True)
class _Facts:
    """Everything the encoder needs to know about the source, from a video file or an image sequence."""
    width: int
    height: int
    rotation: int
    time_base: Fraction
    average_rate: Fraction | None
    sample_aspect_ratio: Fraction
    pixel_format: str | None
    color: ColorDescription
    audio: tuple
    subtitles: tuple
    is_sequence: bool = False


def _output_format(profile: ExportProfile, desc: ColorDescription, native: bool, hdr_pass: bool) -> tuple[str, str]:
    """(encoder name, pixel format) for a profile on this source and path. Raises when it cannot be honoured."""
    if profile.id == "lossless_intermediate":
        return ("ffv1", "bgr0") if not native else ("ffv1", None)  # None: keep the source layout (caller resolves)
    if profile.id == "color_preserving":
        bits = min(max(desc.bit_depth, 10), 12)
        if hdr_pass:
            chroma = "444" if desc.chroma in ("444", "rgb") else ("422" if desc.chroma == "422" else "420")
            return "libx265", f"yuv{chroma}p{bits}le"
        return "prores_ks", "yuv444p10le" if desc.chroma in ("444", "rgb") else "yuv422p10le"
    if profile.id == "archival_hevc10" and native and desc.bit_depth > 10:
        return profile.encoder, f"yuv420p{min(desc.bit_depth, 12)}le"
    return profile.encoder, profile.pix_fmt


def _full_range(tags: dict) -> bool:
    return tags["range"] == codes.RANGE_FULL


def encode_video(source: Path, filename: str, *, root: Path | None = None,
                 codec: Literal["h264", "hevc"] = "h264", hardware: Literal["auto", "software"] = "auto",
                 timing: Literal["preserve_source_timing", "explicit_cfr"] = "preserve_source_timing",
                 rate: Fraction | None = None, audio_policy: AudioPolicy = "copy",
                 start: Fraction | None = None, end: Fraction | None = None,
                 transform: Callable[[DecodedFrame], np.ndarray] | None = None,
                 cancel: Event | None = None, output_dir: str | None = None,
                 manifest_metadata: dict | None = None,
                 before_publish: Callable[[], None] | None = None,
                 finishing: FinishingOptions | None = None,
                 profile: str | ExportProfile | None = None,
                 hdr_policy: Literal["tonemap_sdr", "reject", "raw", "passthrough"] = "tonemap_sdr",
                 subtitles: SubtitlePolicy = "copy", sequence_rate: Fraction | None = None,
                 sequence_format: str = "png", verify_av_sync: bool = True) -> dict:
    """Trim times are absolute source seconds, start inclusive and end exclusive.

    A trim rebases both video and audio by exactly start. Otherwise source PTS,
    including a nonzero origin, are retained. The caller owns any transform.

    ``profile`` selects an export preset (see ``media.export_profiles``); omitted, the pre-Stage-15
    encoder settings apply. ``source`` may be a directory holding a numbered image sequence, which then
    needs an explicit ``sequence_rate``. ``hdr_policy`` governs PQ/HLG input; ``passthrough`` re-encodes
    unmodified media (no ``transform``) on a profile that can carry HDR.
    """
    root = Path(os.path.abspath(root or application_root()))
    if not root.is_relative_to(application_root().absolute()):
        from roop_3d.audit.isolation import AuditError
        raise AuditError("OUTPUT_ROOT_DENIED", "Output root is outside the standalone application")
    finisher = Finisher(finishing or FinishingOptions())
    claim = None
    suffix = Path(filename).suffix.lower()
    chosen = get_profile(profile) if profile is not None else legacy_profile(suffix, codec)
    if Path(filename).name != filename:
        raise ValueError("Output must be a simple filename")
    if chosen.is_sequence:
        if suffix:
            raise ValueError("An image-sequence output name is a directory name and has no extension")
        if sequence_format not in SEQUENCE_FORMATS:
            raise ValueError(f"Sequence format must be one of {sorted(SEQUENCE_FORMATS)}")
    elif suffix not in (".mp4", ".mov", ".mkv") or (profile is not None and suffix != chosen.container):
        raise ValueError(f"Output must be a simple {chosen.container if profile is not None else 'MP4, MOV, or MKV'} filename")
    if codec not in ("h264", "hevc") or hardware not in ("auto", "software"):
        raise ValueError("Invalid encoder policy")
    if timing not in ("preserve_source_timing", "explicit_cfr") or timing == "explicit_cfr" and (rate is None or rate <= 0):
        raise ValueError("Explicit CFR requires a positive rational rate")
    if any(value is not None and not isinstance(value, Fraction) for value in (start, end, rate, sequence_rate)):
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
    container_suffix = suffix if not chosen.is_sequence else ".mkv"
    temporary = owned(root, f"roop_3d/outputs/{token}.video{container_suffix}")
    muxed = owned(root, f"roop_3d/outputs/{token}.mux{container_suffix}")
    signed = owned(root, f"roop_3d/outputs/{token}.signed{container_suffix}")
    sequence_work = owned(root, f"roop_3d/outputs/{token}.sequence")
    audio_sidecar = sequence_work / "audio.mka"
    published = False
    with exclusive(root, f"roop_3d/outputs/.{filename}.lock"):
        if final.exists() or owned(root, manifest_path).exists():
            raise DomainError("OUTPUT_EXISTS", "Output and manifest must be new files", 409)
        frames = None
        audio = None
        encoder_token = LEDGER.open("video_encoder", filename)
        try:
            check_cancel(cancel)
            source = Path(source)
            is_sequence = source.is_dir()
            sequence = None
            if is_sequence:
                if sequence_rate is None:
                    raise DomainError("SEQUENCE_RATE_REQUIRED",
                                      "An image sequence has no clock: pass an explicit rational sequence_rate", 422)
                sequence = discover_sequence(source, sequence_rate)
                digest = sequence_digest(sequence)
                facts = _Facts(sequence.width, sequence.height, 0, sequence.time_base, sequence.rate, Fraction(1),
                               sequence.source_pix_fmt, sequence_color(sequence), (), (), True)
            else:
                digest = sha256_file(source)[1]
                info = probe(source)
                v = info.videos[0]
                facts = _Facts(v.width, v.height, v.rotation, v.time_base, v.average_rate, v.sample_aspect_ratio,
                               v.pixel_format, v.color, info.audio, info.subtitles)
            if finishing and finishing.provenance != 'none':
                claim = recipe_claim({'input_sha256': digest, 'finishing': finisher.metadata(),
                                      'render': manifest_metadata or {}}, finishing.provenance)
            desc = facts.color
            hdr_pass = desc.is_hdr and hdr_policy == "passthrough"
            native_ok = transform is None and not finisher.active and chosen.native_passthrough and not is_sequence
            if hdr_pass and not (native_ok and chosen.hdr_passthrough):
                reason = ("a transform or finishing is applied (HDR cannot be processed by the 8-bit pipeline)"
                          if transform is not None or finisher.active else
                          f"profile {chosen.id} cannot carry PQ/HLG")
                raise DomainError("HDR_PASSTHROUGH_UNSUPPORTED", f"HDR passthrough is unavailable: {reason}", 422)
            native = native_ok and (not desc.is_hdr or hdr_pass)
            working = working_color(desc, "passthrough" if hdr_pass else hdr_policy, (facts.width, facts.height))
            plans = plan_audio(source, container_suffix, audio_policy, start=start, end=end, cancel=cancel) \
                if not is_sequence else ()
            sub_plans = plan_subtitles(facts.subtitles, container_suffix, subtitles) \
                if not chosen.is_sequence else ()
            origin = start or Fraction(0)
            base = Fraction(1, lcm(facts.time_base.denominator, origin.denominator,
                                   rate.numerator if timing == "explicit_cfr" else 1))
            mux_options = {"avoid_negative_ts": "disabled"}
            if container_suffix in (".mp4", ".mov"):
                # The default MOV movie timescale (1000) rounds edit-list
                # offsets even when the video track uses exact ticks.
                audio_bases = tuple(a.time_base.denominator for a in (facts.audio if not is_sequence else ()))
                movie_scale = lcm(base.denominator, *audio_bases)
                mux_options.update(movie_timescale=str(movie_scale), video_track_timescale=str(base.denominator))
            rotated_pixels = not native
            if facts.rotation % 90:
                raise DomainError("ENCODE_DIMENSIONS", "Rotation must be a right angle", 422)
            width, height = ((facts.height, facts.width) if facts.rotation % 180 and rotated_pixels
                             else (facts.width, facts.height))
            output_aspect = (1 / facts.sample_aspect_ratio if facts.rotation % 180 and rotated_pixels
                             else facts.sample_aspect_ratio)
            display_rotation = 0 if rotated_pixels else facts.rotation
            if native and display_rotation and container_suffix not in (".mp4", ".mov"):
                raise DomainError("NATIVE_ROTATION_UNSUPPORTED",
                                  f"Rotated media cannot be carried unmodified in {container_suffix}: use MOV or MP4", 422)

            encoder = out_pix = hardware_state = None
            if chosen.is_sequence:
                encoder, out_pix, hardware_state = "image2", "rgb24", "UNKNOWN"
            else:
                encoder, out_pix = _output_format(chosen, desc, native, hdr_pass)
                if out_pix is None:  # lossless native: keep the source layout
                    out_pix = (facts.pixel_format or "").replace("yuvj", "yuv")
                    if out_pix not in FFV1_NATIVE:
                        raise DomainError("ENCODE_FORMAT_UNSUPPORTED",
                                          f"FFV1 cannot store {facts.pixel_format} unmodified", 422)
                if chosen.allow_hardware and out_pix == "yuv420p" and encoder in ("libx264", "libx265"):
                    encoder, hardware_state = _encoder("h264" if encoder == "libx264" else "hevc", hardware,
                                                       width, height, base)
                else:
                    hardware_state = "NOT_REQUESTED"
                try:
                    av.codec.Codec(encoder, "w")
                except av.FFmpegError:
                    raise DomainError("ENCODER_UNAVAILABLE", f"Encoder {encoder} is not available in this build", 422) from None
                chroma = pixel_format_traits(out_pix).chroma
                if chroma == "420" and (width % 2 or height % 2) or chroma == "422" and width % 2:
                    raise DomainError("ENCODE_DIMENSIONS", f"{out_pix} output requires even upright dimensions", 422)

            rgb_output = (not chosen.is_sequence) and av.VideoFormat(out_pix).is_rgb
            if rgb_output:
                out_tags = {"primaries": working.primaries, "transfer": working.transfer, "matrix": codes.MAT_RGB,
                            "range": codes.RANGE_FULL}
            else:
                out_tags = dict(working.tags)
            out_bits = 8 if chosen.is_sequence else av.VideoFormat(out_pix).components[0].bits
            reader = (SequenceReader(sequence, cancel=cancel) if is_sequence else
                      VideoReader(source, hardware=hardware, cancel=cancel, hdr_policy="passthrough" if hdr_pass else hdr_policy,
                                  source_color=desc))
            source_count = 0
            frame_map = FrameMap(rate, origin) if timing == "explicit_cfr" else None

            def selected():
                nonlocal source_count
                iterator = reader.frames(native=True) if native else reader.frames()
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
                            frame = DecodedFrame(frame.index, frame.timestamp, duration, frame.key_frame, frame.image, frame.frame)
                        yield frame
                finally:
                    iterator.close()

            frames = frame_map.select(selected()) if frame_map else ((frame, frame.timestamp) for frame in selected())
            expected: list[Fraction] = []
            durations: list[Fraction] = []
            sampler = fidelity.FidelitySampler()
            native_samples: dict[int, av.VideoFrame] = {}
            native_tail = None

            if chosen.is_sequence:
                sequence_work.mkdir(parents=True)
                writer = SequenceWriter(sequence_work, sequence_format)
                for decoded, stamp in frames:
                    check_cancel(cancel)
                    image = transform(decoded) if transform else decoded.image
                    if not isinstance(image, np.ndarray) or image.shape != (height, width, 3) or image.dtype != np.uint8:
                        raise DomainError("ENCODE_FRAME_INVALID", "Transform must return upright uint8 BGR with unchanged dimensions", 422)
                    image = finisher.apply(image, decoded.index)
                    relative = stamp.seconds - origin
                    span = Fraction(1, 1) / rate if frame_map else decoded.duration * decoded.timestamp.time_base
                    sampler.offer(len(expected), image)
                    writer.write(len(expected), image, pts=relative, duration=span)
                    expected.append(relative)
                    durations.append(span)
                if not expected:
                    raise DomainError("ENCODE_EMPTY", "No frames in selected interval", 422)
            else:
                converter = None if native else RgbFrameConverter(out_pix, matrix=working.tags["matrix"] if not rgb_output else codes.MAT_RGB,
                                                                   full_range=_full_range(out_tags), width=width, height=height)
                with av.open(str(temporary), "w", options=mux_options) as output:
                    stream = output.add_stream(encoder, rate=rate or facts.average_rate or Fraction(25))
                    stream.width, stream.height, stream.pix_fmt = width, height, out_pix
                    stream.time_base = stream.codec_context.time_base = base
                    stream.codec_context.max_b_frames = 0
                    stream.codec_context.thread_count = 1
                    ctx = stream.codec_context
                    ctx.sample_aspect_ratio = output_aspect
                    ctx.colorspace, ctx.color_range = out_tags["matrix"], out_tags["range"]
                    ctx.color_primaries, ctx.color_trc = out_tags["primaries"], out_tags["transfer"]
                    # Private options belong to the encoder, not the profile: a profile may switch encoder
                    # (colour-preserving ProRes -> HEVC for PQ/HLG) and must not hand one encoder another's options.
                    if "nvenc" in encoder:
                        options = {"preset": "p4"}
                    elif encoder == chosen.encoder:
                        options = dict(chosen.options)
                    else:
                        options = dict(SWITCHED_ENCODER_OPTIONS.get(encoder, {}))
                    if encoder == "prores_ks" and out_pix.startswith("yuv444"):
                        options["profile"] = "4"
                    if encoder == "libx265" and hdr_pass:
                        extra = desc.hdr.to_x265_params() if desc.hdr else None
                        params = options.get("x265-params", "log-level=none")
                        options = {**options, "x265-params": f"{params}:repeat-headers=1:hdr10=1" + (f":{extra}" if extra else "")}
                    stream.options = options
                    if display_rotation:
                        stream.set_display_rotation(display_rotation)
                    output.metadata["comment"] = "Synthetic media processed locally by Roop 3D"
                    for decoded, stamp in frames:
                        check_cancel(cancel)
                        index = len(expected)
                        if native:
                            frame = promote_planes(decoded.frame, out_pix, full_range=_full_range(out_tags))
                            if frame is decoded.frame:  # never stamp the decoder's own frame: CFR may reuse it
                                frame = clone_frame(frame)
                            if index in fidelity.KEEP:
                                native_samples[index] = decoded.frame
                            native_tail = (index, decoded.frame)
                        else:
                            image = transform(decoded) if transform else decoded.image
                            if not isinstance(image, np.ndarray) or image.shape != (height, width, 3) or image.dtype != np.uint8:
                                raise DomainError("ENCODE_FRAME_INVALID", "Transform must return upright uint8 BGR with unchanged dimensions", 422)
                            image = finisher.apply(image, decoded.index)
                            sampler.offer(index, image)
                            frame = converter(image)
                        relative = stamp.seconds - origin
                        frame.pts, frame.time_base = rescale(relative.numerator, Fraction(1, relative.denominator), base), base
                        duration = Fraction(1, 1) / rate if frame_map else decoded.duration * decoded.timestamp.time_base
                        frame.duration = rescale(duration.numerator, Fraction(1, duration.denominator), base)
                        frame.colorspace, frame.color_range = out_tags["matrix"], out_tags["range"]
                        frame.color_primaries, frame.color_trc = out_tags["primaries"], out_tags["transfer"]
                        expected.append(relative)
                        durations.append(duration)
                        for packet in stream.encode(frame):
                            output.mux(packet)
                    if not expected:
                        raise DomainError("ENCODE_EMPTY", "No frames in selected interval", 422)
                    for packet in stream.encode(None):
                        output.mux(packet)
                if native_tail is not None:
                    native_samples[native_tail[0]] = native_tail[1]

            audio_fields = {"audio_tracks": [asdict(p) for p in plans], "audio_sync": [], "audio_integrity": []}
            sub_report = [asdict(p) for p in sub_plans]
            if chosen.is_sequence:
                if plans:
                    with ExitStack() as stack:
                        original = stack.enter_context(open_media(source))
                        audio_out = stack.enter_context(av.open(str(audio_sidecar), "w", format="matroska"))
                        audio = audio_packets(original, audio_out, plans, start=start, end=end, cancel=cancel)
                        for packet in audio:
                            audio_out.mux(packet)
            else:
                with open_media(temporary) as encoded, av.open(str(muxed), "w", options={**mux_options, **chosen.mux_options}) as output, \
                        ExitStack() as stack:
                    target = output.add_stream_from_template(encoded.streams.video[0])
                    if display_rotation:
                        target.set_display_rotation(display_rotation)
                    output.metadata["comment"] = "Synthetic media processed locally by Roop 3D"
                    if claim:
                        # The recipe hash is embedded before computing the final file hash.
                        output.metadata['comment'] += '\n' + json.dumps(claim, sort_keys=True)
                    audio = (audio_packets(stack.enter_context(open_media(source)), output, plans, start=start, end=end,
                                           cancel=cancel) if plans else iter(()))
                    subtitle_stream = (subtitle_packets(stack.enter_context(open_media(source)), output, sub_plans,
                                                        start=start, end=end, cancel=cancel)
                                       if any(p.action == "copy" for p in sub_plans) else iter(()))
                    # Prime both generators: every output stream must exist before the header is emitted.
                    side = merge_sorted(audio, subtitle_stream, key=lambda p: (p.dts or 0) * p.time_base)
                    pending = next(side, None)
                    for packet in encoded.demux(encoded.streams.video[0]):
                        check_cancel(cancel)
                        if packet.dts is None:
                            continue
                        while pending is not None and (pending.dts or 0) * pending.time_base <= packet.dts * packet.time_base:
                            output.mux(pending)
                            pending = next(side, None)
                        packet.stream = target
                        output.mux(packet)
                    while pending is not None:
                        check_cancel(cancel)
                        output.mux(pending)
                        pending = next(side, None)
            c2pa_result = None
            if finishing and finishing.provenance == 'c2pa' and not chosen.is_sequence:
                c2pa_result = sign_c2pa(muxed, signed, claim)
                os.replace(signed, muxed)

            expected_end = expected[-1] + durations[-1]
            tolerance = facts.time_base
            validation: dict = {}
            fidelity_report: list[dict] = []
            sync_report: list[dict] = []
            clip = float(min(AV_SYNC_WINDOW_SECONDS, max(0.0, float(expected_end - expected[0]))))
            hdr_note = None

            def check_audio(target_path: Path) -> None:
                """Payload identity (copied tracks) and signal lag (all tracks) against the produced audio."""
                with av.open(str(target_path)) as produced_container:
                    produced_indices = [s.index for s in produced_container.streams.audio]
                for plan, produced_index in zip(plans, produced_indices):
                    check = av_sync.check_track(
                        source, target_path, plan, produced_index, origin=origin, source_start=start, source_end=end,
                        seconds=clip, tolerance=COPY_LAG_TOLERANCE if plan.action == "copy" else TRANSCODE_LAG_TOLERANCE)
                    sync_report.append(check.as_dict())
                    if not check.ok:
                        raise DomainError("OUTPUT_AV_DESYNC",
                                          f"Audio track {plan.source_index} is not preserved ({check.as_dict()})", 422)

            if chosen.is_sequence:
                samples = sampler.finish()
                frame_rate = rate or facts.average_rate
                document = {"schema_version": 1, "format": sequence_format, "frame_count": len(expected),
                            "frame_rate": f"{frame_rate.numerator}/{frame_rate.denominator}",
                            "timing": "pts_seconds are exact rationals relative to the clip start; the files carry no clock",
                            "source_sha256": digest, "color": {"working": working.as_dict()},
                            "audio": ["audio.mka"] if plans else [], "frames": writer.records}
                if plans:
                    with av.open(str(audio_sidecar)) as sidecar:
                        if len(sidecar.streams.audio) != len(plans):
                            raise DomainError("AUDIO_TRACK_LOSS", "Audio sidecar track count differs from input", 422)
                        document["audio_tracks"] = [{"codec": s.codec_context.name, "language": s.metadata.get("language")}
                                                    for s in sidecar.streams.audio]
                    if verify_av_sync:
                        check_audio(audio_sidecar)
                sidecar_hash = write_sidecar(sequence_work, document)
                validation = verify_sequence(sequence_work, document, samples)
                fidelity_report = [{"frame": i, "exact": True} for i in sorted(samples)]
                output_sha = sidecar_hash
                observed_time_base = str(base)
            else:
                expectation = OutputExpectation(
                    frame_times=expected, video_end=expected_end, width=width, height=height, pix_fmt=out_pix,
                    tags=out_tags, sample_aspect_ratio=output_aspect, display_rotation=display_rotation,
                    audio_tracks=len(facts.audio), subtitle_tracks=sum(1 for p in sub_plans if p.action == "copy"),
                    pts_tolerance=tolerance)
                wanted = sampler.finish()

                def on_frame(index: int, decoded_frame) -> None:
                    if native:
                        if index in native_samples:
                            fidelity_report.append(fidelity.compare_luma(
                                index, native_samples[index], decoded_frame, bias_tolerance=chosen.bias_tolerance,
                                tail_tolerance=chosen.tail_tolerance))
                        return
                    if index in wanted:
                        fidelity_report.append(fidelity.compare_frames(
                            index, wanted[index], decoded_frame.to_ndarray(format="bgr24"),
                            bias_tolerance=chosen.bias_tolerance, tail_tolerance=chosen.tail_tolerance))

                validation = validate_output(muxed, expectation, cancel=cancel, on_frame=on_frame)
                observed = probe(muxed)
                if hdr_pass and desc.hdr is not None:
                    carried = observed.videos[0].hdr
                    if encoder == "libx265":
                        if carried is None or carried.max_cll != desc.hdr.max_cll or carried.max_luminance != desc.hdr.max_luminance:
                            raise DomainError("OUTPUT_HDR_METADATA", "HDR static metadata was not preserved in the output", 422)
                    else:
                        hdr_note = (f"{encoder} cannot carry HDR static metadata (MaxCLL / mastering display); "
                                    "the PQ/HLG colour tags and pixel values are preserved")
                observed_time_base = str(observed.videos[0].time_base)
                if plans and verify_av_sync:
                    check_audio(muxed)
                if not fidelity_report:
                    raise DomainError("OUTPUT_FIDELITY", "No frame could be compared with the encoder input", 422)
                output_sha = None
            audio_fields["audio_sync"] = sync_report
            if (sequence_digest(sequence) if is_sequence else sha256_file(source)[1]) != digest:
                raise DomainError("MEDIA_CHANGED", "Input media changed during encoding", 409)
            try:
                metadata = json.loads(json.dumps(manifest_metadata or {}, allow_nan=False))
                if not isinstance(metadata, dict) or len(json.dumps(metadata).encode("utf-8")) > 65536:
                    raise ValueError("metadata must be a small JSON object")
            except (TypeError, ValueError):
                raise DomainError("MANIFEST_METADATA_INVALID", "Output manifest metadata is invalid", 422) from None
            color_report = {
                "source": desc.as_dict(), "working": working.as_dict(), "hdr_policy": "passthrough" if hdr_pass else hdr_policy,
                "path": "native_planes" if native else "rgb8", "output_tags": {
                    "primaries": codes.name_of("primaries", out_tags["primaries"]), "transfer": codes.name_of("transfer", out_tags["transfer"]),
                    "matrix": codes.name_of("matrix", out_tags["matrix"]), "range": codes.name_of("range", out_tags["range"])},
                "output_pixel_format": out_pix, "output_bit_depth": out_bits,
                "processing_bit_depth": "unmodified" if native else 8,
                "assumptions": list(desc.assumed_fields), "conversion": working.conversion,
                "notes": [*working.notes, *([hdr_note] if hdr_note else [])],
                "claims": {
                    "hdr_output": "PASSTHROUGH_VERIFIED" if hdr_pass else ("NOT_PRODUCED_TONE_MAPPED_TO_SDR" if desc.is_hdr else "NOT_APPLICABLE"),
                    "high_bit_depth_processing": "UNSUPPORTED_PIPELINE_IS_8_BIT" if not native else "NOT_APPLICABLE_PIXELS_UNMODIFIED",
                    "container_bit_depth": out_bits}}
            manifest = {"schema_version": 1, "input_sha256": digest, "output_sha256": output_sha or sha256_file(muxed)[1],
                        "output_path": output_relative, "manifest_path": manifest_path,
                        "encoder": encoder, "nvenc_state": hardware_state, "decoder": reader.backend,
                        "width": width, "height": height, "pixel_format": out_pix,
                        "source_rotation": facts.rotation, "orientation_policy": "upright_pixels" if rotated_pixels else "rotation_metadata_preserved",
                        "source_sample_aspect_ratio": str(facts.sample_aspect_ratio), "output_sample_aspect_ratio": str(output_aspect),
                        "nvdec_state": reader.hardware_state, "source_frames_read": source_count,
                        "encoded_frames": len(expected), "verified_frames": validation.get("frames", len(expected)),
                        "source_time_base": str(facts.time_base), "output_time_base": observed_time_base,
                        "timing_policy": timing, "audio_policy": audio_policy, "audio_tracks": audio_fields["audio_tracks"],
                        "trim": {"start": str(start) if start is not None else None, "end": str(end) if end is not None else None},
                        "frame_map": [asdict(entry) for entry in frame_map.entries] if frame_map else [],
                        "provenance": {"synthetic_media": True, "processor": "Roop 3D", "processing": "local",
                                       "transformed": transform is not None, "notice": "Synthetic media processed locally by Roop 3D"},
                        "stage15": {"profile": chosen.id, "profile_definition": chosen.as_dict(), "color": color_report,
                                    "validation": validation, "fidelity": fidelity_report, "audio_sync": sync_report,
                                    "subtitles": sub_report, "sequence": ({"format": sequence_format, "directory": output_relative}
                                                                          if chosen.is_sequence else None)}}
            if metadata:
                manifest["stage12"] = metadata
                if "model_hashes" in metadata:
                    manifest["model_hashes"] = metadata["model_hashes"]
            if finishing:
                manifest['finishing'] = finisher.metadata()
            if claim:
                manifest['render_claim'] = claim
                manifest['c2pa'] = c2pa_result
                authenticate_manifest(manifest, finishing.provenance)
            check_cancel(cancel)
            if before_publish is not None:
                before_publish()
            owned(root, final.relative_to(root).as_posix())
            os.replace(sequence_work if chosen.is_sequence else muxed, final)
            published = True
            atomic_json(root, manifest_path, manifest)
            return manifest
        except BaseException as error:
            if published:
                target = owned(root, final.relative_to(root).as_posix())
                shutil.rmtree(target, ignore_errors=True) if target.is_dir() else target.unlink(missing_ok=True)
            if is_disk_full(error):  # keep the cause visible: it is resumable, unlike a corrupt-media encode failure
                raise DomainError("DISK_FULL", "The output volume ran out of space while encoding. "
                                  "Free disk space and resume the job.", 507) from None
            if isinstance(error, (av.FFmpegError, OSError)):
                if os.environ.get("ROOP3D_MEDIA_TRACE"):  # opt-in diagnostics; the API message stays generic
                    import traceback
                    traceback.print_exc()
                raise DomainError("MEDIA_ENCODE_FAILED", "Local encode or mux failed", 422) from None
            raise
        finally:
            encoder_token.close()
            if frames is not None:
                frames.close()
            if audio is not None and hasattr(audio, "close"):
                audio.close()
            for path in (temporary, muxed, signed):
                owned(root, path.relative_to(root).as_posix()).unlink(missing_ok=True)
            if sequence_work.exists():
                shutil.rmtree(sequence_work, ignore_errors=True)
