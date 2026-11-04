"""Strict file type, content magic bytes, and path safety validation."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Literal

from roop_3d.core.errors import DomainError

MAX_HEADER_SCAN_BYTES = 4096

DISALLOWED_FILE_HEADERS: tuple[tuple[bytes, str], ...] = (
    (b"MZ", "DOS/Windows executable (PE)"),
    (b"\x7fELF", "Linux ELF executable"),
    (b"\xca\xfe\xba\xbe", "Java/Mach-O binary"),
    (b"\xfe\xed\xfa\xce", "Mach-O 32-bit binary"),
    (b"\xfe\xed\xfa\xcf", "Mach-O 64-bit binary"),
    (b"\xcf\xfa\xed\xfe", "Mach-O reverse binary"),
    (b"#!", "Shell script"),
    (b"<?php", "PHP script"),
    (b"<script", "Web script payload"),
    (b"\x80\x02", "Python pickle v2"),
    (b"\x80\x03", "Python pickle v3"),
    (b"\x80\x04", "Python pickle v4"),
    (b"\x80\x05", "Python pickle v5"),
)


def validate_safe_filename(value: str) -> str:
    """Validate that filename is a portable basename without path traversal or reserved names."""
    reserved = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
    if (not value or len(value) > 255 or value in {".", ".."}
            or any(c in value for c in '/\\:%?*"<>|') or value.endswith((" ", "."))
            or any(ord(c) < 32 for c in value) or value.split(".")[0].casefold() in reserved):
        raise DomainError("FILENAME_SECURITY_VIOLATION", f"Filename '{value}' violates portability/traversal rules", 400)
    return value


def validate_safe_path(path: Path | str, base_dir: Path, allow_symlinks: bool = False) -> Path:
    """Ensure path resolves strictly within base_dir and contains no symlink or traversal escape."""
    base_resolved = base_dir.resolve()
    target = Path(path)
    if not target.is_absolute():
        target = (base_dir / target).resolve()
    else:
        target = target.resolve()

    try:
        target.relative_to(base_resolved)
    except ValueError:
        raise DomainError("PATH_TRAVERSAL_DETECTED", f"Path '{path}' escapes base directory '{base_dir}'", 403) from None

    if not allow_symlinks:
        current = target
        while current != base_resolved and current.parent != current:
            if current.is_symlink():
                raise DomainError("SYMLINK_ATTACK_DETECTED", f"Symlink rejected at '{current}'", 403)
            current = current.parent
    return target


def validate_image_header(data: bytes, expected_format: str | None = None) -> str:
    """Inspect magic bytes for image formats (PNG, JPEG, WebP); fail closed on mismatch."""
    if not data or len(data) < 4:
        raise DomainError("IMAGE_HEADER_INVALID", "Image data is empty or too short for header validation", 400)

    # Check for executable or script disguises
    for sig, desc in DISALLOWED_FILE_HEADERS:
        if data.startswith(sig):
            raise DomainError("DISALLOWED_PAYLOAD", f"Image disguised as {desc}", 400)

    detected: str | None = None
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        detected = "png"
    elif data.startswith(b"\xff\xd8\xff"):
        detected = "jpeg"
    elif len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        detected = "webp"

    if detected is None:
        raise DomainError("IMAGE_FORMAT_UNRECOGNIZED", "Unrecognized image format signature", 400)

    if expected_format:
        norm_expected = expected_format.lower().lstrip(".")
        if norm_expected == "jpg":
            norm_expected = "jpeg"
        if detected != norm_expected:
            raise DomainError("IMAGE_FORMAT_MISMATCH", f"Expected image format {expected_format}, detected {detected}", 400)

    return detected


def validate_video_container(data: bytes, filename: str = "video.mp4") -> str:
    """Inspect container magic bytes for video formats (MP4, MOV, MKV, WebM, AVI, TS)."""
    if not data or len(data) < 8:
        raise DomainError("VIDEO_HEADER_INVALID", "Video data is empty or too short for container verification", 400)

    for sig, desc in DISALLOWED_FILE_HEADERS:
        if data.startswith(sig):
            raise DomainError("DISALLOWED_PAYLOAD", f"Video disguised as {desc}", 400)

    suffix = Path(filename).suffix.lower()

    # MP4 / M4V / MOV
    if suffix in {".mp4", ".m4v", ".mov"}:
        # ISO Base Media File Format: length (4 bytes) + box type (4 bytes)
        # Typically box type is ftyp, moov, wide, skip, free, mdat
        has_iso_box = False
        if len(data) >= 8:
            box_type = data[4:8]
            if box_type in {b"ftyp", b"moov", b"wide", b"skip", b"free", b"mdat", b"pnot"}:
                has_iso_box = True
            elif data.startswith(b"\x00\x00\x00") and len(data) >= 12:
                # Some QuickTime variants
                inner_box = data[8:12]
                if inner_box in {b"ftyp", b"moov", b"wide", b"free"}:
                    has_iso_box = True
        if not has_iso_box:
            raise DomainError("VIDEO_CONTAINER_INVALID", f"File '{filename}' lacks valid MP4/MOV container headers", 400)
        return "mp4" if suffix != ".mov" else "mov"

    # MKV / WebM
    if suffix in {".mkv", ".webm"}:
        if not data.startswith(b"\x1a\x45\xdf\xa3"):
            raise DomainError("VIDEO_CONTAINER_INVALID", f"File '{filename}' lacks valid EBML header for MKV/WebM", 400)
        return "webm" if suffix == ".webm" else "mkv"

    # AVI
    if suffix == ".avi":
        if not (data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"AVI "):
            raise DomainError("VIDEO_CONTAINER_INVALID", f"File '{filename}' lacks valid RIFF/AVI container header", 400)
        return "avi"

    # MPEG-TS
    if suffix == ".ts":
        if data[0] != 0x47:
            raise DomainError("VIDEO_CONTAINER_INVALID", f"File '{filename}' lacks MPEG-TS sync byte 0x47", 400)
        return "ts"

    raise DomainError("VIDEO_FORMAT_UNSUPPORTED", f"Unsupported video format suffix '{suffix}'", 400)


def validate_faceset_archive_bytes(data: bytes) -> None:
    """Validate that raw bytes start with valid zip archive magic."""
    if not data or len(data) < 4:
        raise DomainError("FACESET_ARCHIVE_INVALID", "FaceSet archive is empty or too short", 400)
    if not (data.startswith(b"PK\x03\x04") or data.startswith(b"PK\x05\x06") or data.startswith(b"PK\x07\x08")):
        raise DomainError("FACESET_ARCHIVE_INVALID", "FaceSet archive lacks valid ZIP magic bytes", 400)
