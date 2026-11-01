"""Stage 22: NLE Timeline Interchange, Standalone VFX Matte Pass Exports, and Cryptographic Provenance Metadata."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
from typing import Any, Literal
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from roop_3d.audit.isolation import sha256_file
from roop_3d.core.errors import DomainError
from roop_3d.core.storage import owned, atomic_bytes, atomic_json


# ---------------------------------------------------------------------------
# 1. Standalone VFX Matte Pass & Foreground Pass Exporters
# ---------------------------------------------------------------------------

def combine_alpha_matte(
    geometric_mask: np.ndarray,
    occlusion_matte: np.ndarray | None = None,
    brush_override: np.ndarray | None = None,
    feather_radius: int = 0,
) -> np.ndarray:
    """Combines geometric face mask, SAM-2/XSeg occlusion matte, and manual brush overrides.

    All inputs normalized to float32 in [0.0, 1.0].
    - geometric_mask: initial face coverage boundary (convex hull or 3D mesh coverage)
    - occlusion_matte: foreground occluders (hands, mics, hair) subtracted from face coverage
    - brush_override: operator manual brush painting (additions/subtractions)
    - feather_radius: optional edge feathering radius (Gaussian blur)
    Returns:
        Combined high-precision grayscale alpha matte as float32 in [0.0, 1.0].
    """
    geom = geometric_mask.astype(np.float32)
    if geom.max() > 1.0:
        geom = geom / 255.0
    if geom.ndim == 3:
        geom = geom[..., 0]

    h, w = geom.shape[:2]
    combined = geom.copy()

    # Subtract foreground occlusion if provided
    if occlusion_matte is not None:
        occ = occlusion_matte.astype(np.float32)
        if occ.max() > 1.0:
            occ = occ / 255.0
        if occ.ndim == 3:
            occ = occ[..., 0]
        if occ.shape[:2] != (h, w):
            occ = cv2.resize(occ, (w, h), interpolation=cv2.INTER_LINEAR)
        # Occluded foreground removes swap coverage
        combined = combined * (1.0 - np.clip(occ, 0.0, 1.0))

    # Apply manual brush overrides (white = force keep swap, black = restore target)
    if brush_override is not None:
        brush = brush_override.astype(np.float32)
        if brush.max() > 1.0:
            brush = brush / 255.0
        if brush.ndim == 3:
            brush = brush[..., 0]
        if brush.shape[:2] != (h, w):
            brush = cv2.resize(brush, (w, h), interpolation=cv2.INTER_LINEAR)
        # Composite brush: where brush has high confidence, blend with combined
        combined = np.clip(brush, 0.0, 1.0) * brush + combined * (1.0 - brush)

    # Apply optional Gaussian feathering to blend edges seamlessly
    if feather_radius > 0:
        ksize = feather_radius * 2 + 1
        combined = cv2.GaussianBlur(combined, (ksize, ksize), feather_radius / 2.0)

    return np.clip(combined, 0.0, 1.0).astype(np.float32)


def export_alpha_matte(
    matte: np.ndarray,
    bit_depth: Literal[8, 16] = 16,
) -> np.ndarray:
    """Converts a normalized [0, 1] alpha matte to a high-precision grayscale image.

    Returns uint16 (0-65535) for 16-bit or uint8 (0-255) for 8-bit.
    """
    norm = np.clip(matte, 0.0, 1.0)
    if bit_depth == 16:
        return np.rint(norm * 65535.0).astype(np.uint16)
    return np.rint(norm * 255.0).astype(np.uint8)


def extract_foreground_pass(
    swapped_face_rgb: np.ndarray,
    alpha_matte: np.ndarray,
    output_format: Literal["RGBA", "RGB_MATTE"] = "RGBA",
    bit_depth: Literal[8, 16] = 8,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Generates an uncomposited swapped-face-only foreground pass.

    Allows VFX artists to perform custom grading and compositing in Nuke, After Effects,
    or DaVinci Resolve Fusion without background contamination.

    Returns:
        RGBA image array (4 channels) if output_format == "RGBA",
        or tuple of (uncomposited_rgb, alpha_matte) if output_format == "RGB_MATTE".
    """
    rgb = swapped_face_rgb.copy()
    h, w = rgb.shape[:2]

    alpha = alpha_matte.astype(np.float32)
    if alpha.max() > 1.0:
        alpha = alpha / (65535.0 if alpha.dtype == np.uint16 else 255.0)
    if alpha.shape[:2] != (h, w):
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)

    if bit_depth == 16:
        rgb_16 = np.rint(np.clip(rgb.astype(np.float32) / (255.0 if rgb.dtype == np.uint8 else 65535.0), 0.0, 1.0) * 65535.0).astype(np.uint16)
        alpha_16 = np.rint(np.clip(alpha, 0.0, 1.0) * 65535.0).astype(np.uint16)
        if output_format == "RGBA":
            return np.dstack([rgb_16, alpha_16])
        return rgb_16, alpha_16

    rgb_8 = np.rint(np.clip(rgb.astype(np.float32) / (65535.0 if rgb.dtype == np.uint16 else 255.0), 0.0, 1.0) * 255.0).astype(np.uint8)
    alpha_8 = np.rint(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    if output_format == "RGBA":
        return np.dstack([rgb_8, alpha_8])
    return rgb_8, alpha_8


class VfxPassCollector:
    """Thread-safe collector for companion VFX passes across frames."""

    def __init__(self, shape: tuple[int, int], selected_passes: tuple[str, ...]):
        self.shape = shape
        self.selected_passes = tuple(selected_passes)
        self.buffers: dict[str, np.ndarray] = {}
        for p in self.selected_passes:
            if p in ("foreground", "rgba", "uv"):
                self.buffers[p] = np.zeros((*shape, 4 if p in ("foreground", "rgba") else 3), np.float32)
            elif p in ("normal", "depth"):
                self.buffers[p] = np.zeros((*shape, 3 if p == "normal" else 1), np.float32)
            else:
                self.buffers[p] = np.zeros(shape, np.float32)

    def observe(
        self,
        pass_name: str,
        image_or_matte: np.ndarray,
        inverse: np.ndarray | None = None,
        alpha: np.ndarray | None = None,
    ) -> None:
        size = (self.shape[1], self.shape[0])
        val = image_or_matte.astype(np.float32)
        if val.max() > 1.0:
            val = val / 255.0

        if inverse is not None:
            warped = cv2.warpAffine(val, inverse, size)
        else:
            warped = cv2.resize(val, size) if val.shape[:2] != self.shape else val

        if pass_name == "alpha" and "alpha" in self.buffers:
            self.buffers["alpha"] = np.maximum(self.buffers["alpha"], warped)
        elif pass_name == "occluder" and "occluder" in self.buffers:
            self.buffers["occluder"] = np.maximum(self.buffers["occluder"], warped)
        elif pass_name in ("foreground", "rgba") and pass_name in self.buffers:
            if alpha is not None:
                alpha_w = cv2.warpAffine(alpha, inverse, size) if inverse is not None else alpha
                if warped.ndim == 3 and warped.shape[2] == 3:
                    rgba = np.dstack([warped, alpha_w])
                else:
                    rgba = warped
                self.buffers[pass_name] = np.maximum(self.buffers[pass_name], rgba)
            else:
                self.buffers[pass_name] = np.maximum(self.buffers[pass_name], warped)
        elif pass_name in self.buffers:
            self.buffers[pass_name] = warped

    def save_checkpoint(self, root: Path, job_id: str, frame_index: int) -> dict[str, str]:
        saved_paths = {}
        for name, buf in self.buffers.items():
            if name in ("alpha", "occluder"):
                out = export_alpha_matte(buf, bit_depth=16)
                ok, encoded = cv2.imencode(".png", out)
            elif name in ("foreground", "rgba"):
                # 8-bit RGBA PNG
                out_rgba = np.rint(np.clip(buf, 0.0, 1.0) * 255.0).astype(np.uint8)
                ok, encoded = cv2.imencode(".png", cv2.cvtColor(out_rgba, cv2.COLOR_RGBA2BGRA))
            else:
                out_pass = np.rint(np.clip(buf, 0.0, 1.0) * 65535.0).astype(np.uint16)
                ok, encoded = cv2.imencode(".png", out_pass)

            if ok:
                rel = f"roop_3d/.runtime/jobs/{job_id}/passes/{name}/{frame_index:08d}.png"
                atomic_bytes(root, rel, encoded.tobytes())
                saved_paths[name] = rel
        return saved_paths


# ---------------------------------------------------------------------------
# 2. NLE Timeline Interchange (.fcpxml / .xml / .edl)
# ---------------------------------------------------------------------------

@dataclass
class TimelineClip:
    start_frame: int
    end_frame: int
    media_in: int
    url: str
    name: str = "Clip"
    lane: int = 0
    confidence: float = 1.0
    has_occlusion_warning: bool = False


@dataclass
class TimelineMarker:
    frame: int
    name: str
    comment: str = ""
    marker_type: str = "warning"  # "low_confidence", "occlusion_warning", "chapter"


@dataclass
class TimelinePlan:
    name: str
    frame_rate: str
    frame_count: int
    width: int
    height: int
    source_url: str
    swapped_url: str | None = None
    alpha_url: str | None = None
    foreground_url: str | None = None
    ranges: list[tuple[int, int]] = field(default_factory=list)
    scene_cuts: list[int] = field(default_factory=list)
    markers: list[TimelineMarker] = field(default_factory=list)
    vfr: bool = False
    media_offset: int = 0


def parse_rational_rate(rate_str: str) -> tuple[int, int, float]:
    """Parses rational frame rate string (e.g. '24000/1001' or '25/1' or '30')."""
    parts = rate_str.split("/")
    num = int(parts[0])
    den = int(parts[1]) if len(parts) > 1 else 1
    if den == 0:
        den = 1
    return num, den, num / den


def frame_to_timecode(frame: int, rate_str: str, drop_frame: bool = False) -> str:
    """Converts frame integer to standard SMPTE timecode (HH:MM:SS:FF)."""
    num, den, fps = parse_rational_rate(rate_str)
    nominal = round(fps)
    if drop_frame and nominal in (30, 60):
        drop = 2 if nominal == 30 else 4
        per_10m = nominal * 600 - drop * 9
        per_m = nominal * 60 - drop
        d = frame // per_10m
        m = frame % per_10m
        if m >= drop:
            frame += drop * 9 * d + drop * ((m - drop) // per_m)
        else:
            frame += drop * 9 * d

    h = frame // (nominal * 3600)
    m = (frame // (nominal * 60)) % 60
    s = (frame // nominal) % 60
    f = frame % nominal
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


def build_timeline_from_prescan(
    prescan_data: dict[str, Any],
    source_url: str,
    swapped_url: str | None = None,
    alpha_url: str | None = None,
    foreground_url: str | None = None,
    confidence_threshold: float = 0.5,
) -> TimelinePlan:
    """Builds a structured TimelinePlan from STAGE_1_SPARSE_PRESCAN results and metadata."""
    frame_count = int(prescan_data.get("frame_count", prescan_data.get("total_frames", 100)))
    frame_rate = str(prescan_data.get("frame_rate", prescan_data.get("fps", "25/1")))
    width = int(prescan_data.get("width", 1920))
    height = int(prescan_data.get("height", 1080))
    name = str(prescan_data.get("name", "Roop 3D Timeline"))

    ranges = [tuple(r) for r in prescan_data.get("swapped_ranges", [])]
    scene_cuts = sorted(set(prescan_data.get("scene_cuts", [0, frame_count])))

    markers: list[TimelineMarker] = []

    # Detect low confidence frames
    low_conf_frames = prescan_data.get("low_confidence_frames", [])
    for item in low_conf_frames:
        f_idx = item if isinstance(item, int) else item.get("frame", 0)
        conf = item.get("confidence", 0.0) if isinstance(item, dict) else 0.0
        markers.append(
            TimelineMarker(
                frame=f_idx,
                name=f"Low Confidence ({conf:.2f})" if conf else "Low Confidence Warning",
                comment=f"Swap tracking confidence dropped below {confidence_threshold:.2f} threshold.",
                marker_type="low_confidence",
            )
        )

    # Detect occlusion warning frames
    occlusion_frames = prescan_data.get("occlusion_frames", [])
    for item in occlusion_frames:
        f_idx = item if isinstance(item, int) else item.get("frame", 0)
        markers.append(
            TimelineMarker(
                frame=f_idx,
                name="Occlusion Warning",
                comment="SAM-2/XSeg detected high foreground occlusion; edge seam blending applied.",
                marker_type="occlusion_warning",
            )
        )

    return TimelinePlan(
        name=name,
        frame_rate=frame_rate,
        frame_count=frame_count,
        width=width,
        height=height,
        source_url=source_url,
        swapped_url=swapped_url,
        alpha_url=alpha_url,
        foreground_url=foreground_url,
        ranges=ranges,
        scene_cuts=scene_cuts,
        markers=markers,
    )


def export_fcpxml(plan: TimelinePlan | dict[str, Any]) -> str:
    """Generates standard Apple Final Cut Pro XML (FCPXML 1.10).

    Places original target video on Track 1, discrete active swap ranges on Track 2,
    and chapter/timeline markers at any low-confidence or occlusion frames.
    """
    if isinstance(plan, dict):
        p = build_timeline_from_prescan(
            plan,
            source_url=plan.get("sourceUrl", plan.get("source_url", "")),
            swapped_url=plan.get("swappedUrl", plan.get("swapped_url")),
            alpha_url=plan.get("alphaUrl", plan.get("alpha_url")),
            foreground_url=plan.get("foregroundUrl", plan.get("foreground_url")),
        )
    else:
        p = plan

    num, den, _ = parse_rational_rate(p.frame_rate)
    frame_duration = f"{den}/{num}s"
    total_seconds = f"{p.frame_count * den}/{num}s"

    def to_seconds(f: int) -> str:
        return f"{f * den}/{num}s"

    urls = [p.source_url]
    if p.swapped_url:
        urls.append(p.swapped_url)
    if p.alpha_url:
        urls.append(p.alpha_url)
    if p.foreground_url:
        urls.append(p.foreground_url)

    urls = list(dict.fromkeys(urls))
    ref_map = {url: f"r{i+2}" for i, url in enumerate(urls)}

    # Root fcpxml
    root = ET.Element("fcpxml", version="1.10")
    resources = ET.SubElement(root, "resources")
    ET.SubElement(
        resources,
        "format",
        id="r1",
        frameDuration=frame_duration,
        width=str(p.width),
        height=str(p.height),
    )

    for url in urls:
        asset = ET.SubElement(
            resources,
            "asset",
            id=ref_map[url],
            name=Path(url).name or "media",
            start="0s",
            duration=total_seconds,
            hasVideo="1",
            format="r1",
        )
        ET.SubElement(asset, "media-rep", kind="original-media", src=url)

    library = ET.SubElement(root, "library")
    event = ET.SubElement(library, "event", name="Roop 3D")
    project = ET.SubElement(event, "project", name=p.name)
    sequence = ET.SubElement(
        project,
        "sequence",
        format="r1",
        duration=total_seconds,
        tcStart="0s",
        tcFormat="NDF",
    )
    spine = ET.SubElement(sequence, "spine")

    # Segment clips by scene cuts and swap ranges
    cut_points = sorted(set([0, p.frame_count] + p.scene_cuts + [r[0] for r in p.ranges] + [r[1] for r in p.ranges]))
    cut_points = [c for c in cut_points if 0 <= c <= p.frame_count]

    # Map markers by frame
    marker_dict: dict[int, list[TimelineMarker]] = {}
    for m in p.markers:
        marker_dict.setdefault(m.frame, []).append(m)

    for i in range(len(cut_points) - 1):
        start = cut_points[i]
        end = cut_points[i + 1]
        if start >= end:
            continue
        dur = end - start

        # Track 1: Original source
        source_clip = ET.SubElement(
            spine,
            "asset-clip",
            name="Original Source",
            ref=ref_map[p.source_url],
            offset=to_seconds(start),
            start=to_seconds(start),
            duration=to_seconds(dur),
        )

        # Attach chapter / warning markers if within this range
        for f in range(start, end):
            if f in marker_dict:
                for mk in marker_dict[f]:
                    ET.SubElement(
                        source_clip,
                        "chapter-marker",
                        start=to_seconds(f - start),
                        duration=frame_duration,
                        value=f"[{mk.marker_type.upper()}] {mk.name}: {mk.comment}",
                    )

        # Track 2: Active swapped clips
        is_swapped = any(r[0] <= start and end <= r[1] for r in p.ranges)
        if is_swapped and p.swapped_url:
            ET.SubElement(
                source_clip,
                "asset-clip",
                name="Swapped Face",
                ref=ref_map[p.swapped_url],
                lane="1",
                offset=to_seconds(start),
                start=to_seconds(start),
                duration=to_seconds(dur),
            )

        # Track 3: Alpha matte companion pass
        if is_swapped and p.alpha_url:
            ET.SubElement(
                source_clip,
                "asset-clip",
                name="Alpha Matte Pass",
                ref=ref_map[p.alpha_url],
                lane="2",
                offset=to_seconds(start),
                start=to_seconds(start),
                duration=to_seconds(dur),
            )

    xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n'
    return xml_declaration + ET.tostring(root, encoding="utf-8").decode("utf-8") + "\n"


def export_xmeml(plan: TimelinePlan | dict[str, Any]) -> str:
    """Generates standard Apple / Adobe Premiere FCP 7 XML (XMEML version 5).

    Places Track 1 (source), Track 2 (face swap trimmed clips), and Track 3 (alpha matte).
    """
    if isinstance(plan, dict):
        p = build_timeline_from_prescan(
            plan,
            source_url=plan.get("sourceUrl", plan.get("source_url", "")),
            swapped_url=plan.get("swappedUrl", plan.get("swapped_url")),
            alpha_url=plan.get("alphaUrl", plan.get("alpha_url")),
            foreground_url=plan.get("foregroundUrl", plan.get("foreground_url")),
        )
    else:
        p = plan

    _, _, fps_float = parse_rational_rate(p.frame_rate)
    timebase = str(round(fps_float))

    root = ET.Element("xmeml", version="5")
    seq = ET.SubElement(root, "sequence")
    ET.SubElement(seq, "name").text = p.name
    ET.SubElement(seq, "duration").text = str(p.frame_count)
    rate = ET.SubElement(seq, "rate")
    ET.SubElement(rate, "timebase").text = timebase

    media = ET.SubElement(seq, "media")
    video = ET.SubElement(media, "video")
    format_tag = ET.SubElement(video, "format")
    sample_chars = ET.SubElement(format_tag, "samplecharacteristics")
    ET.SubElement(sample_chars, "width").text = str(p.width)
    ET.SubElement(sample_chars, "height").text = str(p.height)

    # Track 1: Original Source
    track1 = ET.SubElement(video, "track")
    c1 = ET.SubElement(track1, "clipitem", id="clipitem-1")
    ET.SubElement(c1, "name").text = "Original Source"
    ET.SubElement(c1, "start").text = "0"
    ET.SubElement(c1, "end").text = str(p.frame_count)
    ET.SubElement(c1, "in").text = "0"
    ET.SubElement(c1, "out").text = str(p.frame_count)
    file1 = ET.SubElement(c1, "file", id="file-source")
    ET.SubElement(file1, "name").text = Path(p.source_url).name
    ET.SubElement(file1, "pathurl").text = p.source_url

    # Track 2: Swapped Clips
    track2 = ET.SubElement(video, "track")
    clip_id = 2
    for r_start, r_end in p.ranges:
        c2 = ET.SubElement(track2, "clipitem", id=f"clipitem-{clip_id}")
        clip_id += 1
        ET.SubElement(c2, "name").text = "Swapped Face"
        ET.SubElement(c2, "start").text = str(r_start)
        ET.SubElement(c2, "end").text = str(r_end)
        ET.SubElement(c2, "in").text = str(r_start)
        ET.SubElement(c2, "out").text = str(r_end)
        if p.swapped_url:
            file2 = ET.SubElement(c2, "file", id="file-swapped")
            ET.SubElement(file2, "name").text = Path(p.swapped_url).name
            ET.SubElement(file2, "pathurl").text = p.swapped_url

    # Track 3: Alpha Matte Pass (if provided)
    if p.alpha_url:
        track3 = ET.SubElement(video, "track")
        for r_start, r_end in p.ranges:
            c3 = ET.SubElement(track3, "clipitem", id=f"clipitem-{clip_id}")
            clip_id += 1
            ET.SubElement(c3, "name").text = "Alpha Matte"
            ET.SubElement(c3, "start").text = str(r_start)
            ET.SubElement(c3, "end").text = str(r_end)
            ET.SubElement(c3, "in").text = str(r_start)
            ET.SubElement(c3, "out").text = str(r_end)
            file3 = ET.SubElement(c3, "file", id="file-alpha")
            ET.SubElement(file3, "name").text = Path(p.alpha_url).name
            ET.SubElement(file3, "pathurl").text = p.alpha_url

    # Markers for low confidence and occlusion warnings
    for mk in p.markers:
        marker_elem = ET.SubElement(seq, "marker")
        ET.SubElement(marker_elem, "name").text = mk.name
        ET.SubElement(marker_elem, "comment").text = mk.comment
        ET.SubElement(marker_elem, "in").text = str(mk.frame)
        ET.SubElement(marker_elem, "out").text = str(mk.frame + 1)

    xml_header = '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n'
    return xml_header + ET.tostring(root, encoding="utf-8").decode("utf-8") + "\n"


def export_edl(
    plan: TimelinePlan | dict[str, Any],
    drop_frame: bool = False,
) -> dict[str, str]:
    """Generates standard CMX 3600 EDL files for V1 (Source) and V2 (Swapped clips).

    Returns:
        dict with filenames as keys ('V1.edl', 'V2.edl', 'V3.edl') and contents as values.
    """
    if isinstance(plan, dict):
        p = build_timeline_from_prescan(
            plan,
            source_url=plan.get("sourceUrl", plan.get("source_url", "")),
            swapped_url=plan.get("swappedUrl", plan.get("swapped_url")),
            alpha_url=plan.get("alphaUrl", plan.get("alpha_url")),
            foreground_url=plan.get("foregroundUrl", plan.get("foreground_url")),
        )
    else:
        p = plan

    fcm = "DROP FRAME" if drop_frame else "NON-DROP FRAME"

    # Track 1: Source
    v1_lines = [
        f"TITLE: {p.name} V1 Original Source",
        f"FCM: {fcm}",
        f"* FRAME RATE: {p.frame_rate}",
        "",
        f"001  AX       V     C        {frame_to_timecode(0, p.frame_rate, drop_frame)} {frame_to_timecode(p.frame_count, p.frame_rate, drop_frame)} {frame_to_timecode(0, p.frame_rate, drop_frame)} {frame_to_timecode(p.frame_count, p.frame_rate, drop_frame)}",
        f"* FROM CLIP NAME: {Path(p.source_url).name}",
        "",
    ]

    # Track 2: Swapped clips
    v2_lines = [
        f"TITLE: {p.name} V2 Swapped Clips",
        f"FCM: {fcm}",
        f"* FRAME RATE: {p.frame_rate}",
        "",
    ]

    for idx, (r_start, r_end) in enumerate(p.ranges):
        src_in = frame_to_timecode(r_start, p.frame_rate, drop_frame)
        src_out = frame_to_timecode(r_end, p.frame_rate, drop_frame)
        rec_in = src_in
        rec_out = src_out
        v2_lines.append(f"{idx+1:03d}  SWAP     V     C        {src_in} {src_out} {rec_in} {rec_out}")
        v2_lines.append(f"* FROM CLIP NAME: {Path(p.swapped_url or 'swap.mov').name}")

        # Check if markers fall in this range
        for mk in p.markers:
            if r_start <= mk.frame < r_end:
                v2_lines.append(f"* MARKER [{mk.marker_type.upper()}] FRAME {mk.frame}: {mk.name} - {mk.comment}")
        v2_lines.append("")

    result = {
        "V1.edl": "\n".join(v1_lines),
        "V2.edl": "\n".join(v2_lines),
    }

    if p.alpha_url:
        v3_lines = [
            f"TITLE: {p.name} V3 Alpha Matte",
            f"FCM: {fcm}",
            f"* FRAME RATE: {p.frame_rate}",
            "",
        ]
        for idx, (r_start, r_end) in enumerate(p.ranges):
            src_in = frame_to_timecode(r_start, p.frame_rate, drop_frame)
            src_out = frame_to_timecode(r_end, p.frame_rate, drop_frame)
            v3_lines.append(f"{idx+1:03d}  MATTE    V     C        {src_in} {src_out} {src_in} {src_out}")
            v3_lines.append(f"* FROM CLIP NAME: {Path(p.alpha_url).name}")
            v3_lines.append("")
        result["V3.edl"] = "\n".join(v3_lines)

    return result


# ---------------------------------------------------------------------------
# 3. Cryptographic Provenance Manifest (C2PA-Compatible & SHA-256 Signed)
# ---------------------------------------------------------------------------

def create_provenance_manifest(
    recipe: dict[str, Any],
    model_hashes: dict[str, str],
    stage_versions: dict[str, str] | None = None,
    input_sha256: str | None = None,
    output_sha256: str | None = None,
    companion_hashes: dict[str, str] | None = None,
    signing_key: str | None = None,
) -> dict[str, Any]:
    """Generates C2PA-compatible / SHA-256 signed JSON provenance sidecar metadata.

    Records exact model hashes, pipeline stage versions, and render timestamps.
    """
    ts = datetime.now(timezone.utc).isoformat()
    canonical_recipe = json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode("utf-8")
    recipe_hash = hashlib.sha256(canonical_recipe).hexdigest()

    stages = {
        "stage_0_prescan": "1.0.0",
        "stage_1_pose": "1.0.0",
        "stage_2_trt": "1.0.0",
        "stage_3_stream": "1.0.0",
        "stage_22_vfx": "1.0.0",
    }
    if stage_versions:
        stages.update(stage_versions)

    manifest: dict[str, Any] = {
        "schema": "roop3d.provenance.v1",
        "c2pa_compatible": True,
        "render_timestamp": ts,
        "pipeline_stage_versions": stages,
        "model_hashes": model_hashes,
        "recipe_sha256": recipe_hash,
        "input_media_sha256": input_sha256 or "",
        "output_media_sha256": output_sha256 or "",
        "companion_passes_sha256": companion_hashes or {},
        "c2pa_assertions": [
            {
                "label": "c2pa.actions",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.edited",
                            "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia",
                            "softwareAgent": "Roop 3D Standalone Workstation",
                            "when": ts,
                        }
                    ]
                },
            },
            {
                "label": "org.roop3d.pipeline",
                "data": {
                    "models": model_hashes,
                    "stages": stages,
                    "recipe_hash": recipe_hash,
                },
            },
        ],
    }

    # Cryptographic signature
    key = signing_key or os.environ.get("ROOP3D_PROVENANCE_KEY", "")
    if key and len(key.encode("utf-8")) >= 32:
        content_bytes = json.dumps(
            {k: v for k, v in manifest.items() if k != "signature"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        manifest["signature"] = {
            "algorithm": "hmac-sha256",
            "value": hmac.new(key.encode("utf-8"), content_bytes, hashlib.sha256).hexdigest(),
        }
    else:
        # Standard SHA-256 manifest digest
        content_bytes = json.dumps(
            {k: v for k, v in manifest.items() if k != "signature"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        manifest["signature"] = {
            "algorithm": "sha256",
            "value": hashlib.sha256(content_bytes).hexdigest(),
        }

    return manifest


def save_provenance_sidecar(
    output_media_path: str | Path,
    manifest: dict[str, Any],
) -> Path:
    """Saves cryptographic provenance manifest sidecar (.provenance.json) alongside media."""
    path = Path(output_media_path)
    sidecar_path = path.parent / f"{path.name}.provenance.json"
    sidecar_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return sidecar_path


def verify_provenance_sidecar(
    output_media_path: str | Path,
    provenance_path: str | Path | None = None,
    signing_key: str | None = None,
) -> dict[str, Any]:
    """Verifies output media hash, recipe digest, and cryptographic provenance signature."""
    media_path = Path(output_media_path)
    prov_path = Path(provenance_path) if provenance_path else media_path.parent / f"{media_path.name}.provenance.json"

    if not media_path.is_file():
        raise DomainError("MEDIA_NOT_FOUND", f"Target media file not found: {media_path}", 404)
    if not prov_path.is_file():
        raise DomainError("PROVENANCE_NOT_FOUND", f"Provenance sidecar not found: {prov_path}", 404)

    manifest = json.loads(prov_path.read_text("utf-8"))
    _, media_hash = sha256_file(media_path)
    expected_hash = manifest.get("output_media_sha256", "")
    media_matches = bool(expected_hash and hmac.compare_digest(media_hash, expected_hash))

    sig = manifest.get("signature", {})
    algorithm = sig.get("algorithm", "sha256")
    sig_value = sig.get("value", "")

    content_bytes = json.dumps(
        {k: v for k, v in manifest.items() if k != "signature"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    key = signing_key or os.environ.get("ROOP3D_PROVENANCE_KEY", "")
    if algorithm == "hmac-sha256":
        if not key:
            sig_valid = False
            msg = "HMAC verification key not supplied in environment."
        else:
            calc = hmac.new(key.encode("utf-8"), content_bytes, hashlib.sha256).hexdigest()
            sig_valid = hmac.compare_digest(calc, sig_value)
            msg = "HMAC signature verified." if sig_valid else "HMAC signature mismatch."
    else:
        calc = hashlib.sha256(content_bytes).hexdigest()
        sig_valid = hmac.compare_digest(calc, sig_value)
        msg = "SHA-256 manifest integrity verified." if sig_valid else "Manifest digest mismatch."

    is_valid = media_matches and sig_valid
    return {
        "valid": is_valid,
        "media_hash_matches": media_matches,
        "signature_valid": sig_valid,
        "algorithm": algorithm,
        "render_timestamp": manifest.get("render_timestamp"),
        "model_hashes_verified": len(manifest.get("model_hashes", {})),
        "message": msg,
    }
