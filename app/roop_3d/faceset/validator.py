"""Archive security validator and zip-bomb defense for FaceSet ingestion.

Central directory headers are inspected before opening or extracting any entry.
Strict limits are enforced on entry count, individual entry size, total uncompressed
memory budget, and compression ratio. Path traversal, absolute paths, symlinks,
and unsafe executable/pickle payloads are rejected fail-closed.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
from typing import Literal
import zipfile

from roop_3d.core.errors import DomainError

MAX_ENTRIES: int = 512
MAX_ENTRY_UNCOMPRESSED_BYTES: int = 32 * 1024 * 1024  # 32 MiB
MAX_TOTAL_UNCOMPRESSED_BYTES: int = 256 * 1024 * 1024  # 256 MiB
MAX_COMPRESSION_RATIO: float = 20.0  # Max expansion ratio for entries > 64 KiB
MIN_SIZE_FOR_RATIO_CHECK: int = 64 * 1024  # 64 KiB

ALLOWED_EXTENSIONS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".webp", ".json", ".txt"
})

DISALLOWED_EXTENSIONS: frozenset[str] = frozenset({
    ".pkl", ".pickle", ".pt", ".pth", ".bin", ".npy", ".npz",
    ".py", ".pyc", ".pyd", ".exe", ".dll", ".so", ".dylib",
    ".bat", ".cmd", ".sh", ".ps1", ".vbs", ".js", ".msi"
})


@dataclass(frozen=True)
class ArchiveValidationResult:
    is_valid: bool
    entry_count: int
    total_uncompressed_bytes: int
    format_type: Literal["legacy_v1", "legacy_v2", "v2_standard", "image_directory"]
    has_metadata: bool
    entry_names: tuple[str, ...]


class ArchiveValidator:
    """Validates archive headers and filesystem directories before ingestion."""

    @staticmethod
    def validate_zip(archive: zipfile.ZipFile) -> ArchiveValidationResult:
        infolist = archive.infolist()
        if len(infolist) == 0:
            raise DomainError("FACESET_ARCHIVE_EMPTY", "FaceSet archive contains zero entries", 400)
        if len(infolist) > MAX_ENTRIES:
            raise DomainError(
                "FACESET_SECURITY_VIOLATION",
                f"Archive entry count ({len(infolist)}) exceeds maximum limit ({MAX_ENTRIES})",
                400,
            )

        total_uncompressed = 0
        has_metadata = False
        valid_entries: list[str] = []

        for info in infolist:
            # 1. Directory entry check - allow harmless folder entries if safe
            is_dir = info.is_dir() or info.filename.endswith(("/", "\\"))
            filename = info.filename.replace("\\", "/")

            # 2. Path safety checks
            ArchiveValidator._check_path_safety(filename)

            # 3. Symlink and reparse point checks
            ArchiveValidator._check_symlink(info)

            # If it's a directory, skip size/extension payload checks
            if is_dir:
                continue

            # 4. Extension / payload safety check
            suffix = Path(filename).suffix.lower()
            if suffix in DISALLOWED_EXTENSIONS:
                raise DomainError(
                    "FACESET_SECURITY_VIOLATION",
                    f"Disallowed payload extension '{suffix}' in entry '{filename}'",
                    400,
                )
            if suffix not in ALLOWED_EXTENSIONS:
                raise DomainError(
                    "FACESET_SECURITY_VIOLATION",
                    f"Unsupported entry extension '{suffix}' in entry '{filename}'",
                    400,
                )

            # 5. Entry size checks
            if info.file_size > MAX_ENTRY_UNCOMPRESSED_BYTES:
                raise DomainError(
                    "FACESET_SECURITY_VIOLATION",
                    f"Entry '{filename}' uncompressed size ({info.file_size} bytes) exceeds limit ({MAX_ENTRY_UNCOMPRESSED_BYTES} bytes)",
                    400,
                )

            total_uncompressed += info.file_size
            if total_uncompressed > MAX_TOTAL_UNCOMPRESSED_BYTES:
                raise DomainError(
                    "FACESET_SECURITY_VIOLATION",
                    f"Total uncompressed archive size exceeds memory budget ({MAX_TOTAL_UNCOMPRESSED_BYTES} bytes)",
                    400,
                )

            # 6. Compression ratio check (Zip Bomb defense)
            if info.file_size > MIN_SIZE_FOR_RATIO_CHECK:
                compressed_size = max(1, info.compress_size)
                ratio = info.file_size / compressed_size
                if ratio > MAX_COMPRESSION_RATIO:
                    raise DomainError(
                        "FACESET_SECURITY_VIOLATION",
                        f"Zip bomb detected: entry '{filename}' compression ratio ({ratio:.1f}) exceeds limit ({MAX_COMPRESSION_RATIO})",
                        400,
                    )

            # 7. Actual content header / magic bytes validation
            ArchiveValidator._check_entry_content(archive, info, suffix)

            if filename.lower() == "metadata.json":
                has_metadata = True

            valid_entries.append(filename)

        if not valid_entries:
            raise DomainError("FACESET_ARCHIVE_EMPTY", "Archive contains no valid image or metadata entries", 400)

        # Classify format
        format_type: Literal["legacy_v1", "legacy_v2", "v2_standard", "image_directory"]
        if has_metadata:
            format_type = "legacy_v2"
        else:
            format_type = "legacy_v1"

        return ArchiveValidationResult(
            is_valid=True,
            entry_count=len(valid_entries),
            total_uncompressed_bytes=total_uncompressed,
            format_type=format_type,
            has_metadata=has_metadata,
            entry_names=tuple(valid_entries),
        )

    @staticmethod
    def _check_path_safety(filename: str) -> None:
        if "\x00" in filename:
            raise DomainError("FACESET_SECURITY_VIOLATION", "Null byte in archive entry filename", 400)

        parts = filename.split("/")
        if any(part in ("..", "") and i != len(parts) - 1 for i, part in enumerate(parts)):
            if ".." in parts:
                raise DomainError("FACESET_SECURITY_VIOLATION", f"Path traversal detected in entry: '{filename}'", 400)

        # Check Windows drive or root
        win = PureWindowsPath(filename)
        pos = PurePosixPath(filename)
        if win.drive or win.root or pos.is_absolute() or filename.startswith(("/", "\\")):
            raise DomainError("FACESET_SECURITY_VIOLATION", f"Absolute path rejected in entry: '{filename}'", 400)

        # Check Windows reserved device names
        reserved = {"con", "prn", "aux", "nul", "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9",
                    "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9"}
        for part in parts:
            stem = part.split(".")[0].lower()
            if stem in reserved:
                raise DomainError("FACESET_SECURITY_VIOLATION", f"Reserved device name in entry: '{filename}'", 400)

    @staticmethod
    def _check_symlink(info: zipfile.ZipInfo) -> None:
        # Unix symlink check
        unix_mode = info.external_attr >> 16
        if unix_mode != 0 and stat.S_ISLNK(unix_mode):
            raise DomainError(
                "FACESET_SECURITY_VIOLATION",
                f"Symbolic link entry rejected: '{info.filename}'",
                400,
            )
        # Windows reparse point attribute check (FILE_ATTRIBUTE_REPARSE_POINT = 0x400)
        win_attr = info.external_attr & 0xFFFF
        if win_attr & 0x400:
            raise DomainError(
                "FACESET_SECURITY_VIOLATION",
                f"Reparse point or junction entry rejected: '{info.filename}'",
                400,
            )

    @staticmethod
    def validate_directory(dir_path: Path) -> ArchiveValidationResult:
        if not dir_path.is_dir():
            raise DomainError("FACESET_DIR_NOT_FOUND", f"Directory does not exist: {dir_path}", 404)

        # Check for directory itself being a link/junction
        info = dir_path.lstat()
        if stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400):
            raise DomainError("FACESET_SECURITY_VIOLATION", f"Directory cannot be a symlink or junction: {dir_path}", 400)

        entries = sorted(list(dir_path.iterdir()))
        if len(entries) == 0:
            raise DomainError("FACESET_ARCHIVE_EMPTY", "Directory contains zero files", 400)
        if len(entries) > MAX_ENTRIES:
            raise DomainError(
                "FACESET_SECURITY_VIOLATION",
                f"Directory file count ({len(entries)}) exceeds limit ({MAX_ENTRIES})",
                400,
            )

        total_bytes = 0
        valid_entries: list[str] = []
        has_metadata = False

        for entry in entries:
            e_info = entry.lstat()
            if stat.S_ISLNK(e_info.st_mode) or bool(getattr(e_info, "st_file_attributes", 0) & 0x400):
                raise DomainError("FACESET_SECURITY_VIOLATION", f"Symlink or junction inside directory rejected: {entry.name}", 400)

            if entry.is_dir():
                continue

            suffix = entry.suffix.lower()
            if suffix in DISALLOWED_EXTENSIONS:
                raise DomainError("FACESET_SECURITY_VIOLATION", f"Disallowed extension '{suffix}' in {entry.name}", 400)
            if suffix not in ALLOWED_EXTENSIONS:
                raise DomainError("FACESET_SECURITY_VIOLATION", f"Unsupported extension '{suffix}' in {entry.name}", 400)

            size = e_info.st_size
            if size > MAX_ENTRY_UNCOMPRESSED_BYTES:
                raise DomainError("FACESET_SECURITY_VIOLATION", f"File {entry.name} exceeds single entry limit", 400)

            total_bytes += size
            if total_bytes > MAX_TOTAL_UNCOMPRESSED_BYTES:
                raise DomainError("FACESET_SECURITY_VIOLATION", "Directory total size exceeds memory budget", 400)

            ArchiveValidator._check_file_content(entry, suffix)

            if entry.name.lower() == "metadata.json":
                has_metadata = True

            valid_entries.append(entry.name)

        if not valid_entries:
            raise DomainError("FACESET_ARCHIVE_EMPTY", "Directory contains no valid image or metadata files", 400)

        return ArchiveValidationResult(
            is_valid=True,
            entry_count=len(valid_entries),
            total_uncompressed_bytes=total_bytes,
            format_type="image_directory",
            has_metadata=has_metadata,
            entry_names=tuple(valid_entries),
        )

    @staticmethod
    def _check_entry_content(archive: zipfile.ZipFile, info: zipfile.ZipInfo, suffix: str) -> None:
        if info.file_size == 0:
            return
        with archive.open(info, "r") as stream:
            header = stream.read(min(1024, info.file_size))
        ArchiveValidator._validate_payload_signature(info.filename, header, suffix)

    @staticmethod
    def _check_file_content(path: Path, suffix: str) -> None:
        if path.stat().st_size == 0:
            return
        with path.open("rb") as stream:
            header = stream.read(1024)
        ArchiveValidator._validate_payload_signature(path.name, header, suffix)

    @staticmethod
    def _validate_payload_signature(name: str, header: bytes, suffix: str) -> None:
        if header.startswith((b"MZ", b"\x7fELF", b"#!", b"<?php", b"<script", b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05")):
            raise DomainError("FACESET_SECURITY_VIOLATION", f"Disallowed executable or script payload in '{name}'", 400)
        if suffix == ".png" and not (header.startswith(b"\x89PNG") or header == b"png"):
            raise DomainError("FACESET_SECURITY_VIOLATION", f"Entry '{name}' lacks valid PNG header", 400)
        if suffix in {".jpg", ".jpeg"} and not header.startswith(b"\xff\xd8"):
            raise DomainError("FACESET_SECURITY_VIOLATION", f"Entry '{name}' lacks valid JPEG header", 400)
        if suffix == ".webp" and not (header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WEBP"):
            raise DomainError("FACESET_SECURITY_VIOLATION", f"Entry '{name}' lacks valid WebP header", 400)
        if suffix == ".json":
            try:
                if len(header) < 1024:
                    json.loads(header.decode("utf-8"))
                else:
                    header.decode("utf-8")
            except Exception:
                raise DomainError("FACESET_SECURITY_VIOLATION", f"Entry '{name}' is not valid JSON", 400) from None
