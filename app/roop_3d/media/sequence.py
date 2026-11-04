"""Numbered image sequences, as input and as an export profile.

A sequence has no clock. Input therefore requires an explicit rational frame rate, and frame ``i`` is at
exactly ``i / rate`` seconds (integer ticks, no float). Output writes one image per frame plus a sidecar
that records each frame's exact timestamp and SHA-256, the colour description and any audio sidecar --
the information the image files themselves cannot hold.

Validated here: PNG, TIFF and DPX in and out; JPEG in (8-bit, lossy); EXR in (scene-linear float,
re-encoded with the sRGB OETF and reported as such). 16-bit samples are reduced to 8 bits with
round-to-nearest because the face pipeline works at 8 bits.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
from threading import Event
from typing import Iterator

import av
import numpy as np

from roop_3d.audit.isolation import is_link
from roop_3d.color import codes, transfer
from roop_3d.color.description import ColorDescription
from roop_3d.core.errors import DomainError
from roop_3d.media.export_profiles import SEQUENCE_FORMATS
from roop_3d.media.timestamps import Timestamp
from roop_3d.media.video_reader import DecodedFrame, check_cancel

_NAME = re.compile(r"^(?P<prefix>.*?)(?P<number>\d{1,10})(?P<ext>\.[A-Za-z0-9]+)$")
INPUT_FORMATS = {".png": "png_pipe", ".tif": "tiff_pipe", ".tiff": "tiff_pipe", ".dpx": "dpx_pipe",
                 ".jpg": "jpeg_pipe", ".jpeg": "jpeg_pipe", ".exr": "exr_pipe"}
MAX_FRAMES = 200_000
MAX_PIXELS = 16_777_216
MAX_FILE_BYTES = 512 * 1024 * 1024
_OPTIONS = {"protocol_whitelist": ""}


@dataclass(frozen=True)
class SequenceInfo:
    directory: Path
    prefix: str
    digits: int
    extension: str
    first_number: int
    files: tuple[Path, ...]
    width: int
    height: int
    source_pix_fmt: str
    source_bit_depth: int
    rate: Fraction

    @property
    def count(self) -> int:
        return len(self.files)

    @property
    def time_base(self) -> Fraction:
        return Fraction(1, self.rate.numerator)

    @property
    def frame_ticks(self) -> int:
        return self.rate.denominator

    def timestamp(self, index: int) -> Timestamp:
        return Timestamp(index * self.rate.denominator, self.time_base)


def _open_image(path: Path):
    handle = path.open("rb")
    try:
        return handle, av.open(handle, mode="r", format=INPUT_FORMATS[path.suffix.lower()], options=_OPTIONS)
    except (av.FFmpegError, OSError):
        handle.close()
        raise DomainError("SEQUENCE_FRAME_UNREADABLE", f"Image {path.name} could not be decoded", 422) from None


def _decode_image(path: Path) -> av.VideoFrame:
    if is_link(path) or path.stat().st_size > MAX_FILE_BYTES:
        raise DomainError("SEQUENCE_FRAME_DENIED", f"Image {path.name} is linked or too large", 422)
    handle, container = _open_image(path)
    try:
        frame = next(container.decode(video=0), None)
        if frame is None:
            raise DomainError("SEQUENCE_FRAME_UNREADABLE", f"Image {path.name} contains no picture", 422)
        return frame
    finally:
        container.close()
        handle.close()


def to_bgr8(frame: av.VideoFrame, extension: str) -> np.ndarray:
    """Decode one image frame to uint8 BGR. Float EXR is treated as scene-linear and sRGB-encoded."""
    if extension == ".exr":
        rgb = frame.reformat(format="gbrpf32le").to_ndarray()
        # gbrp planes are G, B, R; PyAV stacks them in that order.
        linear = np.stack([rgb[..., 2], rgb[..., 0], rgb[..., 1]], axis=-1)
        encoded = transfer.srgb_inverse_eotf(np.nan_to_num(linear, nan=0.0, posinf=1.0, neginf=0.0))
        return np.ascontiguousarray(np.rint(encoded[..., ::-1] * 255).astype(np.uint8))
    bits = frame.format.components[0].bits
    if bits > 8:
        wide = frame.reformat(format="bgr48le").to_ndarray().astype(np.uint32)
        return ((wide * 255 + 32767) // 65535).astype(np.uint8)
    return frame.reformat(format="bgr24").to_ndarray()


def discover_sequence(directory: Path, rate: Fraction) -> SequenceInfo:
    directory = Path(directory)
    if not isinstance(rate, Fraction) or rate <= 0:
        raise DomainError("SEQUENCE_RATE_REQUIRED", "An image sequence has no clock: pass an explicit rational frame rate", 422)
    if not directory.is_dir() or is_link(directory):
        raise DomainError("MEDIA_NOT_FOUND", "Image sequence directory does not exist", 404)
    groups: dict[tuple[str, str, int], dict[int, Path]] = {}
    for path in directory.iterdir():
        match = _NAME.match(path.name)
        if not match or not path.is_file() or path.suffix.lower() not in INPUT_FORMATS:
            continue
        key = (match["prefix"], path.suffix.lower(), len(match["number"]))
        groups.setdefault(key, {})[int(match["number"])] = path
    if not groups:
        raise DomainError("SEQUENCE_EMPTY", "No numbered PNG/TIFF/DPX/JPEG/EXR images found", 422)
    if len(groups) > 1:
        raise DomainError("SEQUENCE_AMBIGUOUS", "The directory holds more than one numbered image series", 422)
    (prefix, extension, digits), numbered = next(iter(groups.items()))
    numbers = sorted(numbered)
    if len(numbers) > MAX_FRAMES:
        raise DomainError("SEQUENCE_TOO_LONG", f"At most {MAX_FRAMES} frames are accepted", 422)
    missing = [n for n in range(numbers[0], numbers[-1] + 1) if n not in numbered]
    if missing:
        raise DomainError("SEQUENCE_GAP", f"Missing frame number(s) {missing[:8]}: a gap would silently shift timing", 422)
    first = _decode_image(numbered[numbers[0]])
    if not 0 < first.width * first.height <= MAX_PIXELS:
        raise DomainError("VIDEO_DIMENSIONS_DENIED", "Image dimensions exceed decode budget", 422)
    bits = 32 if extension == ".exr" else first.format.components[0].bits
    return SequenceInfo(directory, prefix, digits, extension, numbers[0], tuple(numbered[n] for n in numbers),
                        first.width, first.height, first.format.name, bits, rate)


def sequence_color(info: SequenceInfo) -> ColorDescription:
    """Stills carry no reliable colour tags; the assumption is recorded, never silent."""
    how = "assumed(still-image default: sRGB, full range)"
    if info.extension == ".exr":
        return ColorDescription(codes.PRI_BT709, 8, codes.MAT_RGB, codes.RANGE_FULL, 32, "rgb", "SDR",
                                {"primaries": how, "transfer": "assumed(EXR is scene-linear)", "matrix": "assumed(RGB)",
                                 "range": "assumed(full)"})
    return ColorDescription(codes.PRI_BT709, codes.TRC_SRGB, codes.MAT_RGB, codes.RANGE_FULL, info.source_bit_depth, "rgb",
                            "SDR", {"primaries": how, "transfer": how, "matrix": "assumed(RGB)", "range": how})


class SequenceReader:
    """Same iteration contract as ``VideoReader.frames`` (exact rational PTS, upright BGR frames)."""

    def __init__(self, info: SequenceInfo, *, cancel: Event | None = None):
        self.info, self.cancel = info, cancel
        self.backend, self.hardware_state = "software", "UNKNOWN"
        self.source_color = sequence_color(info)

    def frames(self, *, native: bool = False) -> Iterator[DecodedFrame]:
        if native:
            raise DomainError("SEQUENCE_NATIVE_UNSUPPORTED", "Image sequences are always processed as RGB", 422)
        for index, path in enumerate(self.info.files):
            check_cancel(self.cancel)
            frame = _decode_image(path)
            if (frame.width, frame.height) != (self.info.width, self.info.height):
                raise DomainError("SEQUENCE_DIMENSIONS_MISMATCH", f"{path.name} differs in size from the first frame", 422)
            yield DecodedFrame(index, self.info.timestamp(index), self.info.frame_ticks, True,
                               to_bgr8(frame, self.info.extension))


def sequence_digest(info: SequenceInfo) -> str:
    """Tree hash over the ordered (name, SHA-256) pairs of every frame file."""
    outer = hashlib.sha256()
    for path in info.files:
        outer.update(path.name.encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return outer.hexdigest()


class SequenceWriter:
    """Writes numbered frames to an existing directory and returns the frame records for the sidecar."""

    def __init__(self, directory: Path, image_format: str = "png", stem: str = "frame", start_number: int = 0):
        if image_format not in SEQUENCE_FORMATS:
            raise ValueError(f"Sequence format must be one of {sorted(SEQUENCE_FORMATS)}")
        self.directory, self.stem, self.start_number = Path(directory), stem, start_number
        self.codec, self.pix_fmt, self.suffix = SEQUENCE_FORMATS[image_format]
        self.image_format = image_format
        self.records: list[dict] = []

    def name(self, index: int) -> str:
        return f"{self.stem}_{self.start_number + index:08d}{self.suffix}"

    def write(self, index: int, bgr: np.ndarray, *, pts: Fraction, duration: Fraction) -> None:
        path = self.directory / self.name(index)
        frame = av.VideoFrame.from_ndarray(bgr, format="bgr24").reformat(format=self.pix_fmt)
        frame.pts, frame.time_base = 0, Fraction(1, 1)
        with av.open(str(path), "w") as container:
            stream = container.add_stream(self.codec)
            stream.width, stream.height, stream.pix_fmt = bgr.shape[1], bgr.shape[0], self.pix_fmt
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)
        self.records.append({"index": index, "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "pts_seconds": f"{pts.numerator}/{pts.denominator}",
                             "duration_seconds": f"{duration.numerator}/{duration.denominator}"})


def write_sidecar(directory: Path, document: dict) -> str:
    data = json.dumps(document, indent=2, sort_keys=True).encode()
    (directory / "sequence.json").write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def verify_sequence(directory: Path, document: dict, expected_bgr: dict[int, np.ndarray]) -> dict:
    """Every recorded file exists with its recorded hash and decodes to the right size; samples match exactly."""
    records = document["frames"]
    for record in records:
        path = directory / record["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise DomainError("OUTPUT_FRAME_LOSS", f"{record['file']} does not match its recorded hash", 422)
    for index, expected in expected_bgr.items():
        frame = _decode_image(directory / records[index]["file"])
        if (frame.width, frame.height) != (expected.shape[1], expected.shape[0]):
            raise DomainError("OUTPUT_DIMENSIONS", "Sequence frame dimensions differ from the encoder input", 422)
        decoded = to_bgr8(frame, Path(records[index]["file"]).suffix.lower())
        if not np.array_equal(decoded, expected):
            raise DomainError("OUTPUT_FIDELITY", f"Sequence frame {index} does not decode to the encoder input", 422)
    return {"frames": len(records), "hashes_verified": len(records), "samples_exact": sorted(expected_bgr)}
