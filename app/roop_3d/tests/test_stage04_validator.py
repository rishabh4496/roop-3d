"""Unit tests for archive security validator and zip-bomb defense."""
from __future__ import annotations

import io
from pathlib import Path
import stat
import zipfile
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.faceset.validator import (
    MAX_COMPRESSION_RATIO,
    MAX_ENTRIES,
    MAX_ENTRY_UNCOMPRESSED_BYTES,
    MAX_TOTAL_UNCOMPRESSED_BYTES,
    ArchiveValidator,
)


def _make_zip(entries: dict[str, bytes], compress_type=zipfile.ZIP_DEFLATED) -> zipfile.ZipFile:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=compress_type) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    bio.seek(0)
    return zipfile.ZipFile(bio, "r")


def test_valid_v1_archive():
    zf = _make_zip({"0.png": b"\x89PNG\r\n\x1a\nfake1", "1.png": b"\x89PNG\r\n\x1a\nfake2"})
    result = ArchiveValidator.validate_zip(zf)
    assert result.is_valid is True
    assert result.entry_count == 2
    assert result.format_type == "legacy_v1"
    assert result.has_metadata is False


def test_valid_v2_archive():
    zf = _make_zip({"metadata.json": b'{"version": 2}', "0.png": b"\x89PNG\r\n\x1a\nfake1"})
    result = ArchiveValidator.validate_zip(zf)
    assert result.is_valid is True
    assert result.entry_count == 2
    assert result.format_type == "legacy_v2"
    assert result.has_metadata is True


def test_empty_archive_rejected():
    zf = _make_zip({})
    with pytest.raises(DomainError) as exc_info:
        ArchiveValidator.validate_zip(zf)
    assert exc_info.value.code == "FACESET_ARCHIVE_EMPTY"


def test_entry_count_limit_rejected():
    entries = {f"{i}.png": b"a" for i in range(MAX_ENTRIES + 1)}
    zf = _make_zip(entries)
    with pytest.raises(DomainError) as exc_info:
        ArchiveValidator.validate_zip(zf)
    assert exc_info.value.code == "FACESET_SECURITY_VIOLATION"
    assert "entry count" in exc_info.value.message


def test_path_traversal_rejected():
    zf = _make_zip({"../payload.png": b"123"})
    with pytest.raises(DomainError) as exc_info:
        ArchiveValidator.validate_zip(zf)
    assert exc_info.value.code == "FACESET_SECURITY_VIOLATION"
    assert "traversal" in exc_info.value.message.lower()


def test_absolute_path_posix_rejected():
    zf = _make_zip({"/etc/shadow.png": b"123"})
    with pytest.raises(DomainError) as exc_info:
        ArchiveValidator.validate_zip(zf)
    assert exc_info.value.code == "FACESET_SECURITY_VIOLATION"
    assert "absolute path" in exc_info.value.message.lower()


def test_absolute_path_windows_rejected():
    zf = _make_zip({"C:\\Windows\\System32\\calc.png": b"123"})
    with pytest.raises(DomainError) as exc_info:
        ArchiveValidator.validate_zip(zf)
    assert exc_info.value.code == "FACESET_SECURITY_VIOLATION"
    assert "absolute path" in exc_info.value.message.lower()


def test_null_byte_rejected():
    with pytest.raises(DomainError) as exc_info:
        ArchiveValidator._check_path_safety("image\x00.png")
    assert exc_info.value.code == "FACESET_SECURITY_VIOLATION"
    assert "null byte" in exc_info.value.message.lower()


def test_disallowed_extension_pickle_rejected():
    zf = _make_zip({"model.pkl": b"\x80\x04\x95..."})
    with pytest.raises(DomainError) as exc_info:
        ArchiveValidator.validate_zip(zf)
    assert exc_info.value.code == "FACESET_SECURITY_VIOLATION"
    assert ".pkl" in exc_info.value.message


def test_disallowed_extension_script_rejected():
    zf = _make_zip({"exploit.py": b"import os; os.system('calc')"})
    with pytest.raises(DomainError) as exc_info:
        ArchiveValidator.validate_zip(zf)
    assert exc_info.value.code == "FACESET_SECURITY_VIOLATION"
    assert ".py" in exc_info.value.message


def test_symlink_rejected():
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as zf:
        zinfo = zipfile.ZipInfo("link.png")
        # Set POSIX symlink attribute (S_IFLNK)
        zinfo.external_attr = stat.S_IFLNK << 16
        zf.writestr(zinfo, "/target/path")
    bio.seek(0)
    zf = zipfile.ZipFile(bio, "r")
    with pytest.raises(DomainError) as exc_info:
        ArchiveValidator.validate_zip(zf)
    assert exc_info.value.code == "FACESET_SECURITY_VIOLATION"
    assert "symbolic link" in exc_info.value.message.lower()


def test_zip_bomb_compression_ratio_rejected():
    # 1 MB of zeros compresses down to very few bytes with Deflate
    uncompressed_size = 1024 * 1024  # 1 MiB (> 64 KiB)
    data = b"\x00" * uncompressed_size
    zf = _make_zip({"bomb.png": data}, compress_type=zipfile.ZIP_DEFLATED)
    # The ratio of 1 MB zeros in deflate is > 100x (limit is 20.0)
    with pytest.raises(DomainError) as exc_info:
        ArchiveValidator.validate_zip(zf)
    assert exc_info.value.code == "FACESET_SECURITY_VIOLATION"
    assert "zip bomb" in exc_info.value.message.lower()


def test_directory_validation(tmp_path: Path):
    # Valid directory
    img_dir = tmp_path / "valid_faces"
    img_dir.mkdir()
    (img_dir / "0.png").write_bytes(b"\x89PNG1")
    (img_dir / "1.jpg").write_bytes(b"\xFF\xD8\xFF1")

    res = ArchiveValidator.validate_directory(img_dir)
    assert res.is_valid is True
    assert res.entry_count == 2
    assert res.format_type == "image_directory"

    # Empty directory
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(DomainError) as exc_info:
        ArchiveValidator.validate_directory(empty_dir)
    assert exc_info.value.code == "FACESET_ARCHIVE_EMPTY"

    # Directory with dangerous extension
    bad_dir = tmp_path / "bad_dir"
    bad_dir.mkdir()
    (bad_dir / "test.pkl").write_bytes(b"pickle")
    with pytest.raises(DomainError) as exc_info:
        ArchiveValidator.validate_directory(bad_dir)
    assert exc_info.value.code == "FACESET_SECURITY_VIOLATION"
