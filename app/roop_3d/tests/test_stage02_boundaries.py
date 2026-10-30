from __future__ import annotations

import os
from pathlib import Path
import subprocess

from pydantic import ValidationError
import pytest

from roop_3d.audit.isolation import AuditError
from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Settings
from roop_3d.core.storage import atomic_json, owned
from roop_3d.media.service import MediaService
from roop_3d.models.catalog import ModelCatalog


def test_real_directory_link_is_rejected_before_reads_and_writes(tmp_path: Path) -> None:
    target = tmp_path / "roop_3d/.runtime/target"
    target.mkdir(parents=True)
    folder = tmp_path / "roop_3d/.runtime/media"
    folder.mkdir()
    link = folder / "link"
    if os.name == "nt":
        env = dict(os.environ, ROOP3D_TEST_LINK=str(link), ROOP3D_TEST_TARGET=str(target))
        result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                                 "New-Item -ItemType Junction -Path $env:ROOP3D_TEST_LINK -Target $env:ROOP3D_TEST_TARGET | Out-Null"],
                                env=env, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
    else:
        os.symlink(target, link, target_is_directory=True)
    try:
        with pytest.raises(AuditError) as error:
            atomic_json(tmp_path, "roop_3d/.runtime/media/link/forbidden.json", {"secret": "not written"})
        assert error.value.code == "LINK_TRAVERSAL"
        assert not (target / "forbidden.json").exists()
        with pytest.raises(ValidationError):
            Settings(cache_root=str(link))
        with pytest.raises(DomainError):
            MediaService(tmp_path, ModelCatalog()).path("roop_3d/.runtime/media/link/video.mp4")
    finally:
        # Remove the link itself, never recursively traverse its target.
        if os.name == "nt":
            os.rmdir(link)
        else:
            link.unlink()


def test_atomic_replace_does_not_modify_hardlinked_input(tmp_path: Path) -> None:
    target = owned(tmp_path, "roop_3d/.runtime/immutable.json")
    target.parent.mkdir(parents=True)
    target.write_bytes(b"immutable source")
    destination = owned(tmp_path, "roop_3d/.runtime/config.json")
    os.link(target, destination)
    atomic_json(tmp_path, "roop_3d/.runtime/config.json", {"replacement": True})
    assert target.read_bytes() == b"immutable source"
    assert destination.read_bytes() != target.read_bytes()
