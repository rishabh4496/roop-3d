"""Declarative export presets. A profile states what it *can* carry; it never promises more.

Everything the swap pipeline produces is 8-bit display-referred RGB, so no profile claims "HDR output"
or ">8-bit processing" for rendered frames. Profiles that use a 10-bit container say so as
``container_bit_depth``: it reduces encode-stage quantisation, it does not recover precision the 8-bit
pipeline discarded. HDR can only pass through *unmodified* (``hdr_passthrough``) and only on profiles
whose codec, pixel format and bit depth can represent it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

SEQUENCE = "sequence"
ProfileId = Literal["standard_mp4", "high_quality_mp4", "archival_hevc10", "mov_workflow",
                    "lossless_intermediate", "image_sequence", "color_preserving"]


@dataclass(frozen=True)
class ExportProfile:
    id: str
    label: str
    description: str
    container: str                      # ".mp4" | ".mov" | ".mkv" | "sequence"
    encoder: str                        # preferred software encoder (or image codec name)
    codec_family: str                   # "h264" | "hevc" | "prores" | "ffv1" | "image"
    pix_fmt: str | None                 # fixed output pixel format; None = derived from the source
    container_bit_depth: int            # bit depth of the encoded stream
    options: Mapping[str, str] = field(default_factory=dict)
    mux_options: Mapping[str, str] = field(default_factory=dict)
    lossless: bool = False
    rgb_native: bool = False            # encodes RGB directly, no YCbCr matrix at all
    allow_hardware: bool = False        # NVENC may stand in for libx264 / libx265
    native_passthrough: bool = False    # may re-encode source planes without an RGB round trip
    hdr_passthrough: bool = False       # may carry PQ/HLG unmodified (needs native_passthrough)
    audio_default: str = "copy"
    # Output-decode fidelity tolerances, in 8-bit code values, for the self-check against the encoder input.
    bias_tolerance: float = 2.5
    tail_tolerance: float = 10.0

    @property
    def is_sequence(self) -> bool:
        return self.container == SEQUENCE

    def as_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "description": self.description, "container": self.container,
                "encoder": self.encoder, "codec_family": self.codec_family, "pix_fmt": self.pix_fmt,
                "container_bit_depth": self.container_bit_depth, "options": dict(self.options),
                "lossless": self.lossless, "rgb_native": self.rgb_native, "native_passthrough": self.native_passthrough,
                "hdr_passthrough": self.hdr_passthrough, "audio_default": self.audio_default,
                "processing_bit_depth": 8,
                "limits": ["Rendered frames are 8-bit display-referred RGB; this profile does not add precision."
                           if self.container_bit_depth > 8 else "8-bit output."]}


PROFILES: dict[str, ExportProfile] = {p.id: p for p in (
    ExportProfile(
        "standard_mp4", "Standard MP4",
        "H.264 8-bit 4:2:0 in MP4: broadest playback compatibility. Same encoder settings as the original default output.",
        ".mp4", "libx264", "h264", "yuv420p", 8, {"preset": "fast"}, allow_hardware=True, bias_tolerance=6.0, tail_tolerance=14.0),
    ExportProfile(
        "high_quality_mp4", "High-quality MP4",
        "H.264 High profile, slow preset, CRF 16, 8-bit 4:2:0, fast-start MP4.",
        ".mp4", "libx264", "h264", "yuv420p", 8, {"preset": "slow", "crf": "16", "profile": "high"},
        {"movflags": "+faststart"}, bias_tolerance=4.0, tail_tolerance=12.0),
    ExportProfile(
        "archival_hevc10", "High-bitrate archival (HEVC 10-bit)",
        "HEVC Main10, CRF 10, slow preset, in Matroska. Visually transparent; keeps every audio and subtitle stream.",
        ".mkv", "libx265", "hevc", "yuv420p10le", 10,
        {"preset": "slow", "crf": "10", "x265-params": "log-level=none"},
        native_passthrough=True, hdr_passthrough=True, bias_tolerance=3.0, tail_tolerance=10.0),
    ExportProfile(
        "mov_workflow", "MOV workflow (ProRes 422 HQ)",
        "Apple ProRes 422 HQ, 10-bit 4:2:2 in QuickTime MOV, for editing and finishing applications.",
        ".mov", "prores_ks", "prores", "yuv422p10le", 10, {"profile": "3", "vendor": "apl0"},
        native_passthrough=True, bias_tolerance=3.0, tail_tolerance=10.0),
    ExportProfile(
        "lossless_intermediate", "Lossless intermediate (FFV1)",
        "FFV1 version 3 in Matroska. Rendered frames are stored as RGB, so decoding returns the encoder input exactly; "
        "untouched media is stored in its own YCbCr planes.",
        ".mkv", "ffv1", "ffv1", None, 8, {"level": "3", "slicecrc": "1", "coder": "1"},
        lossless=True, rgb_native=True, native_passthrough=True, hdr_passthrough=True, bias_tolerance=0.0, tail_tolerance=0.0),
    ExportProfile(
        "image_sequence", "Image sequence",
        "One numbered image per frame (PNG, TIFF or DPX, 8-bit RGB) plus a sidecar manifest holding exact rational "
        "timestamps, colour description and audio sidecar.",
        SEQUENCE, "png", "image", None, 8, {}, lossless=True, rgb_native=True, bias_tolerance=0.0, tail_tolerance=0.0),
    ExportProfile(
        "color_preserving", "Professional colour-preserving",
        "ProRes 422 HQ / 4444 (10-bit) in MOV with the source's primaries, transfer, matrix and range tags kept "
        "exactly. Untouched PQ/HLG media is carried as HEVC Main10 with its HDR10 static metadata.",
        ".mov", "prores_ks", "prores", None, 10, {"profile": "3", "vendor": "apl0"},
        native_passthrough=True, hdr_passthrough=True, bias_tolerance=3.0, tail_tolerance=10.0),
)}
DEFAULT_PROFILE = "standard_mp4"
SEQUENCE_FORMATS = {"png": ("png", "rgb24", ".png"), "tiff": ("tiff", "rgb24", ".tif"), "dpx": ("dpx", "rgb24", ".dpx")}


def get_profile(profile: str | ExportProfile | None) -> ExportProfile:
    if isinstance(profile, ExportProfile):
        return profile
    name = profile or DEFAULT_PROFILE
    if name not in PROFILES:
        raise ValueError(f"Unknown export profile {name!r}; available: {sorted(PROFILES)}")
    return PROFILES[name]


def legacy_profile(suffix: str, codec: str) -> ExportProfile:
    """The pre-Stage-15 behaviour (no profile argument): same encoder settings, container from the filename."""
    hevc = codec == "hevc"
    return ExportProfile(
        "legacy", "Legacy default", "Pre-Stage-15 encoder settings for the requested container and codec.",
        suffix, "libx265" if hevc else "libx264", "hevc" if hevc else "h264", "yuv420p", 8, {"preset": "fast"},
        allow_hardware=True, bias_tolerance=6.0, tail_tolerance=14.0)
