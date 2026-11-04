"""Colour descriptions of media: what the file says, what is assumed, and what that means.

Nothing here assumes material is SDR BT.709. Each of primaries, transfer, matrix and range is either
``tagged`` (the file states it) or ``assumed(<rule>)`` with the rule named, so a report can always say
how much of an interpretation was a guess. An untagged file is never classified as HDR: HDR needs an
explicit PQ or HLG transfer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import struct
from typing import Literal

from roop_3d.color import codes
from roop_3d.core.errors import DomainError

DynamicRange = Literal["SDR", "HDR_PQ", "HDR_HLG"]
#: ``passthrough`` keeps PQ/HLG untouched and is only meaningful when no pixel is processed.
HdrPolicy = Literal["tonemap_sdr", "reject", "raw", "passthrough"]
HDR_POLICIES = ("tonemap_sdr", "reject", "raw", "passthrough")


@dataclass(frozen=True)
class PixelFormatTraits:
    bit_depth: int
    chroma: str  # "420" | "422" | "444" | "440" | "411" | "rgb" | "gray" | "unknown"
    family: str  # "yuv" | "rgb" | "gray" | "unknown"
    implies_full_range: bool


_YUV = re.compile(r"^yuv(?:a)?(?P<j>j)?(?P<chroma>\d{3})p(?P<bits>\d+)?(?:le|be)?$")
_RGB = re.compile(r"^(?:gbr|rgb|bgr|argb|abgr|rgba|bgra|x2rgb|x2bgr)")
_RGB_BITS = re.compile(r"(\d+)(?:le|be)?$")


def pixel_format_traits(name: str | None) -> PixelFormatTraits:
    """Bit depth, chroma layout and family of an FFmpeg pixel format name."""
    if not name:
        return PixelFormatTraits(8, "unknown", "unknown", False)
    match = _YUV.match(name)
    if match:
        return PixelFormatTraits(int(match["bits"] or 8), match["chroma"], "yuv", bool(match["j"]))
    if name.startswith("gray") or name.startswith("ya"):
        bits = _RGB_BITS.search(name)
        return PixelFormatTraits(int(bits[1]) if bits and int(bits[1]) > 8 else 8, "gray", "gray", True)
    if name in ("nv12", "nv21"):
        return PixelFormatTraits(8, "420", "yuv", False)
    if name in ("nv16",):
        return PixelFormatTraits(8, "422", "yuv", False)
    match = re.match(r"^p(?P<chroma>[024])(?P<bits>\d{2})(?:le|be)$", name)  # p010 = 4:2:0 10-bit, p216 = 4:2:2 16-bit
    if match:
        return PixelFormatTraits(int(match["bits"]), {"0": "420", "2": "422", "4": "444"}[match["chroma"]], "yuv", False)
    if name in ("y210le", "y212le"):
        return PixelFormatTraits(int(name[2:4]), "422", "yuv", False)
    if name in ("xv30le", "xv36le", "vuyx", "vuya"):
        return PixelFormatTraits(10 if name == "xv30le" else (12 if name == "xv36le" else 8), "444", "yuv", False)
    if _RGB.match(name):
        bits = _RGB_BITS.search(name)
        depth = int(bits[1]) if bits and name.endswith(("le", "be")) else 8
        # packed 48/64-bit RGB names carry total bits (rgb48 = 16 per channel)
        depth = {48: 16, 64: 16, 96: 32, 128: 32}.get(depth, depth)
        return PixelFormatTraits(depth if depth in range(8, 33) else 8, "rgb", "rgb", True)
    return PixelFormatTraits(8, "unknown", "unknown", False)


@dataclass(frozen=True)
class HdrStaticMetadata:
    """SMPTE ST 2086 mastering display and CTA-861.3 content light level, as carried in the stream."""
    max_cll: int | None = None
    max_fall: int | None = None
    primaries: tuple[tuple[float, float], ...] | None = None  # R, G, B (x, y)
    white_point: tuple[float, float] | None = None
    min_luminance: float | None = None
    max_luminance: float | None = None

    @property
    def has_mastering_display(self) -> bool:
        return self.primaries is not None and self.white_point is not None and self.max_luminance is not None

    def to_x265_params(self) -> str | None:
        """x265 ``master-display`` / ``max-cll`` option fragments, or None when nothing is carried."""
        parts = []
        if self.has_mastering_display:
            (rx, ry), (gx, gy), (bx, by) = self.primaries
            wx, wy = self.white_point

            def chroma(value: float) -> int:
                return int(round(value * 50000))

            parts.append("master-display=G(%d,%d)B(%d,%d)R(%d,%d)WP(%d,%d)L(%d,%d)" % (
                chroma(gx), chroma(gy), chroma(bx), chroma(by), chroma(rx), chroma(ry),
                chroma(wx), chroma(wy), int(round(self.max_luminance * 10000)),
                int(round((self.min_luminance or 0.0) * 10000))))
        if self.max_cll is not None and self.max_fall is not None:
            parts.append(f"max-cll={self.max_cll},{self.max_fall}")
        return ":".join(parts) if parts else None

    def as_dict(self) -> dict:
        return {"max_cll": self.max_cll, "max_fall": self.max_fall, "primaries": self.primaries,
                "white_point": self.white_point, "min_luminance": self.min_luminance,
                "max_luminance": self.max_luminance}


def parse_mastering_display(data: bytes) -> HdrStaticMetadata | None:
    """Parse an ``AVMasteringDisplayMetadata`` blob (88 bytes: ten rationals and two flags)."""
    if len(data) != 88:
        return None
    values = struct.unpack("<20i2i", data)
    rationals = [(values[i], values[i + 1]) for i in range(0, 20, 2)]
    has_primaries, has_luminance = values[20], values[21]

    def ratio(item: tuple[int, int]) -> float | None:
        return item[0] / item[1] if item[1] else None

    primaries = white = None
    if has_primaries:
        points = [ratio(r) for r in rationals[:8]]
        if None not in points:
            primaries = ((points[0], points[1]), (points[2], points[3]), (points[4], points[5]))
            white = (points[6], points[7])
    low = high = None
    if has_luminance:
        low, high = ratio(rationals[8]), ratio(rationals[9])
    if primaries is None and high is None:
        return None
    return HdrStaticMetadata(primaries=primaries, white_point=white, min_luminance=low, max_luminance=high)


def parse_content_light(data: bytes) -> tuple[int, int] | None:
    """Parse an ``AVContentLightMetadata`` blob (MaxCLL, MaxFALL as two uint32)."""
    if len(data) != 8:
        return None
    return struct.unpack("<II", data)


def merge_hdr_metadata(mastering: HdrStaticMetadata | None, light: tuple[int, int] | None) -> HdrStaticMetadata | None:
    if mastering is None and light is None:
        return None
    base = mastering or HdrStaticMetadata()
    if light is None:
        return base
    return HdrStaticMetadata(light[0], light[1], base.primaries, base.white_point, base.min_luminance, base.max_luminance)


@dataclass(frozen=True)
class ColorDescription:
    """Resolved colour interpretation of a video stream. Codes are FFmpeg / H.273 values."""
    primaries: int
    transfer: int
    matrix: int
    range: int  # 1 limited, 2 full (never 0 once resolved)
    bit_depth: int
    chroma: str
    dynamic_range: DynamicRange
    provenance: dict = field(default_factory=dict, hash=False, compare=False)
    hdr: HdrStaticMetadata | None = None

    @property
    def assumed_fields(self) -> tuple[str, ...]:
        return tuple(name for name, how in self.provenance.items() if how != "tagged")

    @property
    def is_hdr(self) -> bool:
        return self.dynamic_range != "SDR"

    @property
    def wide_gamut(self) -> bool:
        return self.primaries in (9, 11, 12)

    def as_dict(self) -> dict:
        return {
            "primaries": codes.name_of("primaries", self.primaries), "primaries_code": self.primaries,
            "transfer": codes.name_of("transfer", self.transfer), "transfer_code": self.transfer,
            "matrix": codes.name_of("matrix", self.matrix), "matrix_code": self.matrix,
            "range": codes.name_of("range", self.range), "range_code": self.range,
            "bit_depth": self.bit_depth, "chroma": self.chroma, "dynamic_range": self.dynamic_range,
            "provenance": dict(self.provenance), "assumed_fields": list(self.assumed_fields),
            "hdr_static_metadata": self.hdr.as_dict() if self.hdr else None}


def _heuristic_matrix(width: int, height: int) -> tuple[int, str]:
    """BT.709 for HD, BT.470BG for 625-line, SMPTE 170M for 525-line (the convention players use)."""
    if width > 1024 or height > 576:
        return codes.MAT_BT709, "HD (>1024 wide or >576 high) assumed BT.709"
    if height >= 560:
        return codes.MAT_BT470BG, "625-line SD assumed BT.601 (BT.470 B/G)"
    return codes.MAT_SMPTE170M, "525-line SD assumed BT.601 (SMPTE 170M)"


def resolve_color(*, primaries: int | None, transfer: int | None, matrix: int | None, range_: int | None,
                  width: int, height: int, pix_fmt: str | None,
                  hdr: HdrStaticMetadata | None = None) -> ColorDescription:
    """Resolve possibly-missing tags into a complete description, recording every assumption."""
    traits = pixel_format_traits(pix_fmt)
    provenance: dict[str, str] = {}

    pq_or_hlg = transfer in (codes.TRC_PQ, codes.TRC_HLG)
    if traits.family == "rgb" and codes.is_unspecified("matrix", matrix):
        matrix_value = codes.MAT_RGB
        provenance["matrix"] = "assumed(RGB pixel format carries no YCbCr matrix)"
    elif codes.is_unspecified("matrix", matrix):
        if pq_or_hlg or primaries == codes.PRI_BT2020:
            matrix_value, rule = codes.MAT_BT2020NC, "BT.2020 primaries or HDR transfer imply BT.2020 non-constant luminance"
        else:
            matrix_value, rule = _heuristic_matrix(width, height)
        provenance["matrix"] = f"assumed({rule})"
    else:
        matrix_value = int(matrix)
        provenance["matrix"] = "tagged"

    if codes.is_unspecified("primaries", primaries):
        primaries_value = {codes.MAT_BT709: codes.PRI_BT709, codes.MAT_BT470BG: codes.PRI_BT470BG,
                           codes.MAT_SMPTE170M: codes.PRI_SMPTE170M, codes.MAT_BT2020NC: codes.PRI_BT2020,
                           10: codes.PRI_BT2020, 7: 7}.get(matrix_value, codes.PRI_BT709)
        provenance["primaries"] = "assumed(derived from the YCbCr matrix)"
    else:
        primaries_value = int(primaries)
        provenance["primaries"] = "tagged"

    if codes.is_unspecified("transfer", transfer):
        transfer_value = {codes.MAT_BT709: codes.TRC_BT709, codes.MAT_BT470BG: codes.TRC_SMPTE170M,
                          codes.MAT_SMPTE170M: codes.TRC_SMPTE170M, codes.MAT_BT2020NC: codes.TRC_BT2020_10,
                          codes.MAT_RGB: codes.TRC_SRGB}.get(matrix_value, codes.TRC_BT709)
        provenance["transfer"] = "assumed(SDR; untagged material is never treated as HDR)"
    else:
        transfer_value = int(transfer)
        provenance["transfer"] = "tagged"

    if codes.is_unspecified("range", range_):
        range_value = codes.RANGE_FULL if traits.implies_full_range else codes.RANGE_LIMITED
        provenance["range"] = "assumed(%s)" % ("full for JPEG-range or RGB pixel format" if traits.implies_full_range
                                               else "limited (MPEG) for YCbCr")
    else:
        range_value = int(range_)
        provenance["range"] = "tagged"

    dynamic: DynamicRange = ("HDR_PQ" if transfer_value == codes.TRC_PQ
                             else "HDR_HLG" if transfer_value == codes.TRC_HLG else "SDR")
    return ColorDescription(primaries_value, transfer_value, matrix_value, range_value, traits.bit_depth,
                            traits.chroma, dynamic, provenance, hdr)


@dataclass(frozen=True)
class WorkingColor:
    """What the 8-bit RGB frames handed to a transform mean, and how the output must be tagged.

    ``source`` is the interpretation of the input file. For SDR input the working space equals it: the
    pipeline is colour-transparent, with no gamut or transfer conversion. For HDR input under the
    ``tonemap_sdr`` policy the working space is BT.709 SDR and ``conversion`` names the operator.
    """
    primaries: int
    transfer: int
    matrix: int
    range: int
    source: ColorDescription
    conversion: str | None = None
    tone_map: dict | None = None
    notes: tuple[str, ...] = ()

    @property
    def tags(self) -> dict:
        return {"primaries": self.primaries, "transfer": self.transfer, "matrix": self.matrix, "range": self.range}

    def as_dict(self) -> dict:
        return {"primaries": codes.name_of("primaries", self.primaries), "transfer": codes.name_of("transfer", self.transfer),
                "matrix": codes.name_of("matrix", self.matrix), "range": codes.name_of("range", self.range),
                "tags": self.tags, "conversion": self.conversion, "tone_map": self.tone_map,
                "processing_bit_depth": 8, "notes": list(self.notes), "source": self.source.as_dict()}


def working_color(source: ColorDescription, hdr_policy: HdrPolicy = "tonemap_sdr",
                  frame_size: tuple[int, int] | None = None) -> WorkingColor:
    """The colour contract between decode and encode for a source under an HDR policy.

    ``tags`` are what a YCbCr output must carry. An RGB source has no YCbCr matrix, so one is chosen by the
    same convention as for untagged video (BT.709 for HD) with studio range, and the choice is recorded.
    """
    if hdr_policy not in HDR_POLICIES:
        raise ValueError(f"hdr_policy must be one of {HDR_POLICIES}")
    if not source.is_hdr and source.matrix == codes.MAT_RGB:
        matrix, rule = _heuristic_matrix(*frame_size) if frame_size else (codes.MAT_BT709, "default BT.709")
        return WorkingColor(source.primaries, source.transfer, matrix, codes.RANGE_LIMITED, source,
                            notes=(f"RGB source has no YCbCr matrix: YCbCr outputs use {codes.name_of('matrix', matrix)} "
                                   f"({rule}) with limited range",))
    if not source.is_hdr:
        return WorkingColor(source.primaries, source.transfer, source.matrix, source.range, source)
    if hdr_policy == "reject":
        raise DomainError("HDR_INPUT_REJECTED",
                          "HDR (PQ/HLG) input is rejected by policy; choose tonemap_sdr to render an SDR result", 422)
    if hdr_policy == "passthrough":
        return WorkingColor(source.primaries, source.transfer, source.matrix, source.range, source,
                            notes=("HDR passthrough: pixels are re-encoded unmodified in the source's own YCbCr planes",))
    if hdr_policy == "raw":
        return WorkingColor(source.primaries, source.transfer, source.matrix, source.range, source,
                            notes=("HDR code values are processed as if they were SDR; output is not a valid HDR rendition",))
    from roop_3d.color.hdr import ALGORITHM, ToneMapParams
    peak = source.hdr.max_luminance if source.hdr and source.hdr.max_luminance else None
    params = ToneMapParams(source_peak_nits=max(100.0, min(peak, 10000.0)) if peak else 1000.0)
    notes = ["HDR input is tone-mapped to SDR BT.709 before processing; the output is SDR, not HDR"]
    if peak is None:
        notes.append("No mastering-display metadata: source peak assumed 1000 cd/m^2")
    return WorkingColor(codes.PRI_BT709, codes.TRC_BT709, codes.MAT_BT709, codes.RANGE_LIMITED, source,
                        conversion=f"{source.dynamic_range.lower()}_to_sdr_bt709:{ALGORITHM}",
                        tone_map={"algorithm": ALGORITHM, "source_peak_nits": params.source_peak_nits,
                                  "sdr_peak_nits": params.sdr_peak_nits, "reference_white_nits": params.reference_white_nits},
                        notes=tuple(notes))
