"""Unit tests for Stage 16 reset safety and boundary isolation."""
from __future__ import annotations

from pathlib import Path
import socket
import sys
from unittest.mock import MagicMock, patch

import pytest

from roop_3d.audit.project_paths import application_root, legacy_root, project_root

if str(project_root()) not in sys.path:
    sys.path.insert(0, str(project_root()))

from tools.project_admin import check_reset, owned


def test_owned_rejects_external_paths() -> None:
    root = project_root()
    # Relative path with '..' escaping root
    with pytest.raises(RuntimeError, match="Path is not inside this Roop 3D app"):
        owned("../roop-ultimate")

    # Absolute path pointing outside
    with pytest.raises(RuntimeError, match="Path is not inside this Roop 3D app"):
        owned(str(legacy_root()))


def test_check_reset_rejects_when_port_occupied() -> None:
    with patch("socket.socket") as mock_socket_cls:
        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 0  # Port is occupied
        mock_socket_cls.return_value.__enter__.return_value = mock_socket

        with pytest.raises(RuntimeError, match="The API port is still occupied"):
            check_reset()


def test_reset_preserves_facesets_and_outputs() -> None:
    root = project_root()
    facesets_dir = root / "roop_3d/outputs/facesets"
    outputs_dir = root / "outputs"

    # Verify reset targets do NOT include facesets or outputs
    reset_script = (root / "reset.js").read_text("utf-8")
    assert "outputs" not in reset_script
    assert "facesets" not in reset_script
    assert "roop-ultimate" not in reset_script
    assert "app/roop_3d/.venv" in reset_script
    assert "cache/install-ready.json" in reset_script
