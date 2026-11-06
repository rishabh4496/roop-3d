"""Pull-based upright RGB/BGR decode. No background workers or frame queue."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import os
from pathlib import Path
from threading import Event
from typing import Iterator, Literal

import av
import cv2
import numpy as np

from roop_3d.color import codes
from roop_3d.color.description import (
    HDR_POLICIES, ColorDescription, HdrPolicy, WorkingColor, merge_hdr_metadata, parse_content_light, parse_mastering_display,
    resolve_color, working_color)
from roop_3d.color.hdr import HdrToSdr, ToneMapParams
from roop_3d.core.errors import DomainError
from roop_3d.media.probe import open_media
from roop_3d.media.timestamps import Timestamp, TimestampEngine, rescale
from roop_3d.reliability.ledger import LEDGER

#: Matrices PyAV can request from swscale on a frame; BT.2020 is applied natively when the file tags it.
_FRAME_MATRICES = (1, 4, 5, 6, 7)


def check_cancel(cancel: Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise DomainError("MEDIA_CANCELLED", "Media operation was cancelled", 409)


def describe_frame_color(frame: av.VideoFrame, stream: av.video.stream.VideoStream) -> ColorDescription:
    """Resolve the colour interpretation of a stream from its first software-decoded frame.

    Container / codec parameters win when specific; the decoded frame's own tags (in-band VUI) fill
    what they leave open. HDR static metadata comes from frame side data.
    """
    ctx = stream.codec_context
    matrix, primaries, trc, rng = ctx.colorspace, ctx.color_primaries, ctx.color_trc, ctx.color_range
    matrix = matrix if not codes.is_unspecified("matrix", matrix) else int(frame.colorspace)
    primaries = primaries if not codes.is_unspecified("primaries", primaries) else int(frame.color_primaries)
    trc = trc if not codes.is_unspecified("transfer", trc) else int(frame.color_trc)
    rng = rng if not codes.is_unspecified("range", rng) else int(frame.color_range)
    mastering = light = None
    for item in frame.side_data:
        if item.type.name == "MASTERING_DISPLAY_METADATA":
            mastering = parse_mastering_display(bytes(item))
        elif item.type.name == "CONTENT_LIGHT_LEVEL":
            light = parse_content_light(bytes(item))
    return resolve_color(primaries=primaries, transfer=trc, matrix=matrix, range_=rng, width=frame.width,
                         height=frame.height, pix_fmt=frame.format.name, hdr=merge_hdr_metadata(mastering, light))


def rotate_upright(image: np.ndarray, angle: int) -> np.ndarray:
    """Rotate an image by the (counter-clockwise) display rotation so its pixels are upright."""
    if angle % 90:
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
        w, h = int(np.ceil(height * sine + width * cosine)), int(np.ceil(height * cosine + width * sine))
        matrix[:, 2] += ((w - width) / 2, (h - height) / 2)
        return cv2.warpAffine(image, matrix, (w, h))
    return np.ascontiguousarray(np.rot90(image, angle // 90))


def upright(frame: av.VideoFrame, format: str = "bgr24", rotation: int | None = None) -> np.ndarray:
    image = frame.to_ndarray(format=format)
    angle = frame.rotation if rotation is None else rotation
    if angle % 90:
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
        w, h = int(np.ceil(height * sine + width * cosine)), int(np.ceil(height * cosine + width * sine))
        matrix[:, 2] += ((w - width) / 2, (h - height) / 2)
        return cv2.warpAffine(image, matrix, (w, h))
    return np.ascontiguousarray(np.rot90(image, angle // 90))


@dataclass(frozen=True)
class DecodedFrame:
    index: int
    timestamp: Timestamp
    duration: int
    key_frame: bool
    image: np.ndarray | None
    # Set instead of ``image`` by ``frames(native=True)``: the decoded frame in its own pixel format.
    frame: av.VideoFrame | None = None


class VideoReader:
    def __init__(self, path: Path, *, hardware: Literal["auto", "software"] = "auto",
                 max_frame_bytes: int = 64 * 1024 * 1024, format: Literal["bgr24", "rgb24"] = "bgr24",
                 cancel: Event | None = None, thread_count: int | None = None,
                 hdr_policy: HdrPolicy = "tonemap_sdr", source_color: ColorDescription | None = None):
        """``hdr_policy`` decides what PQ/HLG input becomes: ``tonemap_sdr`` (default, explicit BT.2390
        conversion to BT.709), ``reject`` (raise HDR_INPUT_REJECTED) or ``raw`` (legacy: HDR code values
        are treated as SDR, kept only so verification can decode a file bit-for-bit as written).
        SDR input is never converted: the 8-bit RGB frames carry the file's own primaries and transfer.
        ``source_color`` lets a caller that already resolved the description (the encoder) pin it.
        """
        if hardware not in ("auto", "software") or format not in ("bgr24", "rgb24") or max_frame_bytes <= 0:
            raise ValueError("Invalid reader policy")
        if hdr_policy not in HDR_POLICIES:
            raise ValueError(f"hdr_policy must be one of {HDR_POLICIES}")
        self.path, self.hardware, self.max_frame_bytes, self.format = Path(path), hardware, max_frame_bytes, format
        self.cancel = cancel
        self.thread_count = thread_count
        self.backend, self.hardware_state = "software", "UNKNOWN"
        self.fallback_reason: str | None = None
        self.hdr_policy = hdr_policy
        self.source_color: ColorDescription | None = source_color
        self.working: WorkingColor | None = None
        self._converter: HdrToSdr | None = None
        self._matrix_override: int | None = None

    def _prepare_color(self, frame: av.VideoFrame, stream: av.video.stream.VideoStream) -> None:
        """Fix the colour interpretation once, from the first software-decoded frame."""
        if self.working is not None:
            return
        description = self.source_color or describe_frame_color(frame, stream)
        self.source_color = description
        self.working = working_color(description, self.hdr_policy, (frame.width, frame.height))
        if self.working.conversion is not None:
            self._converter = HdrToSdr(description.transfer, description.primaries,
                                       ToneMapParams(source_peak_nits=self.working.tone_map["source_peak_nits"]))
        ctx = stream.codec_context
        # swscale falls back to BT.601 for an untagged matrix whatever the resolution; apply the
        # resolved (recorded) convention explicitly instead.
        if (codes.is_unspecified("matrix", ctx.colorspace) and codes.is_unspecified("matrix", int(frame.colorspace))
                and description.matrix in _FRAME_MATRICES):
            self._matrix_override = description.matrix

    def _image(self, frame: av.VideoFrame, rotation: int | None) -> np.ndarray:
        if self._converter is not None:
            rgb = self._converter.convert(frame.to_ndarray(format="rgb48le"))
            if self.format == "bgr24":
                rgb = rgb[..., ::-1]
            angle = frame.rotation if rotation is None else rotation
            return rotate_upright(np.ascontiguousarray(rgb), angle)
        if self._matrix_override is not None:
            frame.colorspace = self._matrix_override
        return upright(frame, self.format, rotation)

    def convert(self, frame: av.VideoFrame, stream: av.video.stream.VideoStream, rotation: int | None = None) -> np.ndarray:
        """A software-decoded frame as an upright image under this reader's colour policy.

        For callers that keep a container open and own the decode loop (the scrubbing frame service) but must
        not re-implement HDR tone mapping, matrix selection or rotation.
        """
        self._prepare_color(frame, stream)
        image = self._image(frame, rotation)
        if image.nbytes > self.max_frame_bytes:
            raise DomainError("FRAME_MEMORY_BUDGET", "Upright frame exceeds byte budget", 422)
        return image

    def frames(self, *, seek: Timestamp | None = None, keyframes_only: bool = False,
               native: bool = False) -> Iterator[DecodedFrame]:
        """Indices count emitted frames in this iterator; seek results use PTS as identity.

        ``native=True`` yields the decoded ``av.VideoFrame`` in its source pixel format (software decode,
        no RGB conversion, no rotation applied) for encoders that must not round-trip through 8-bit RGB.
        """
        last: Fraction | None = None
        index = 0
        rotation = None
        hardware_ok = self.hardware == "auto" and not native
        if self.hardware == "auto":
            # CUDA transfer frames can omit DISPLAYMATRIX side data. Read the
            # source orientation with the reference software decoder first.
            check_cancel(self.cancel)
            with open_media(self.path) as reference:
                stream = reference.streams.video[0]
                stream.codec_context.thread_count = 1
                if stream.width * stream.height * 3 > self.max_frame_bytes:
                    raise DomainError("FRAME_MEMORY_BUDGET", "Frame exceeds configured byte budget", 422)
                first = next(reference.decode(stream), None)
                rotation = first.rotation if first is not None else 0
                if first is not None:
                    self._prepare_color(first, stream)
                    # Hardware transfer frames lose HDR side data and need a 10-bit-aware conversion
                    # that has not been validated; HDR is always decoded in software.
                    hardware_ok = not self.source_color.is_hdr
        attempts = [True, False] if hardware_ok else [False]
        for hardware in attempts:
            check_cancel(self.cancel)
            try:
                kwargs = {"hwaccel": av.codec.hwaccel.HWAccel("cuda", allow_software_fallback=False)} if hardware else {}
                with LEDGER.hold("video_decoder", self.path.name), open_media(self.path, **kwargs) as container:
                    stream = container.streams.video[0]
                    threads = self.thread_count or (1 if hardware else max(1, min(8, (os.cpu_count() or 4) - 2)))
                    stream.codec_context.thread_count = threads
                    if stream.width * stream.height * 3 > self.max_frame_bytes:
                        raise DomainError("FRAME_MEMORY_BUDGET", "Frame exceeds configured byte budget", 422)
                    if seek is not None:
                        container.seek(seek.in_base(Fraction(stream.time_base)), stream=stream, backward=True)
                    if keyframes_only:
                        stream.codec_context.skip_frame = "NONKEY"
                    engine = TimestampEngine()
                    metadata: dict[Fraction, tuple[bool, int, Fraction]] = {}
                    for packet in container.demux(stream):
                        check_cancel(self.cancel)
                        if packet.pts is not None and (not keyframes_only or packet.is_keyframe):
                            metadata[packet.pts * Fraction(packet.time_base)] = (
                                packet.is_keyframe, packet.duration, Fraction(packet.time_base))
                        if len(metadata) > 256:
                            raise DomainError("DECODE_METADATA_BUDGET", "Decoder exceeded pending timestamp budget", 422)
                        for frame in packet.decode():
                            check_cancel(self.cancel)
                            stamp = engine.accept(frame.pts, Fraction(frame.time_base))
                            key, duration, duration_base = metadata.pop(stamp.seconds, (
                                frame.key_frame, frame.duration, Fraction(frame.time_base)))
                            if last is not None and stamp.seconds <= last:
                                continue  # Software fallback resumes without duplicate delivery.
                            if keyframes_only and not (key or frame.key_frame):
                                continue
                            accelerated = hardware and stream.codec_context.is_hwaccel
                            if accelerated:
                                # PyAV 17's CUDA transfer copies only PTS. Restore
                                # color interpretation from the stream context.
                                ctx = stream.codec_context
                                frame.colorspace, frame.color_range = ctx.colorspace, ctx.color_range
                                frame.color_primaries, frame.color_trc = ctx.color_primaries, ctx.color_trc
                            else:
                                self._prepare_color(frame, stream)
                            image = None if native else self._image(frame, rotation)
                            if image is not None and image.nbytes > self.max_frame_bytes:
                                raise DomainError("FRAME_MEMORY_BUDGET", "Upright frame exceeds byte budget", 422)
                            self.backend = "nvdec" if accelerated else "software"
                            self.hardware_state = "VERIFIED_LOCAL" if accelerated else (
                                "UNSUPPORTED" if self.hardware == "auto" else "UNKNOWN")
                            duration = frame.duration or rescale(duration, duration_base, stamp.time_base)
                            yield DecodedFrame(index, stamp, duration, key or frame.key_frame, image,
                                               frame if native else None)
                            last, index = stamp.seconds, index + 1
                return
            except (DomainError, av.FFmpegError, RuntimeError, ValueError) as error:
                if not hardware or isinstance(error, DomainError) and error.code != "MEDIA_DECODE_FAILED":
                    raise
                self.fallback_reason = "CUDA decode unavailable for this media; software decoder selected"
                self.hardware_state = "UNSUPPORTED"

    def at(self, timestamp: Timestamp, max_decode_frames: int = 10000) -> DecodedFrame:
        iterator = self.frames(seek=timestamp)
        try:
            for index, frame in enumerate(iterator):
                if index >= max_decode_frames:
                    raise DomainError("DECODE_BUDGET_EXCEEDED", "Seek exceeds frame budget", 422)
                if frame.timestamp.seconds >= timestamp.seconds:
                    return frame
        finally:
            iterator.close()
        raise DomainError("FRAME_OUT_OF_RANGE", "Timestamp is beyond the video", 416)

    def thumbnails(self, timestamps: list[Timestamp], max_size: tuple[int, int] = (320, 180)) -> Iterator[tuple[Timestamp, bytes]]:
        if len(timestamps) > 128 or min(max_size) < 1 or max(max_size) > 2048:
            raise ValueError("Thumbnail request exceeds budget")
        for timestamp in timestamps:
            frame = self.at(timestamp)
            h, w = frame.image.shape[:2]
            scale = min(max_size[0] / w, max_size[1] / h, 1)
            image = cv2.resize(frame.image, (max(1, int(w * scale)), max(1, int(h * scale))))
            if self.format == "rgb24":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            ok, data = cv2.imencode(".jpg", image)
            if not ok:
                raise DomainError("PREVIEW_ENCODE_FAILED", "Thumbnail encoding failed", 422)
            yield frame.timestamp, data.tobytes()
