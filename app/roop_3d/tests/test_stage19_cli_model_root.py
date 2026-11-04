"""Stage 19: the offline model commands must use the server's data root.

Found while writing docs/OFFLINE_MODELS.md: ``tools/project_admin.py model-import`` built its ModelManager from the
project root, so a bundle was copied to ``<project>/roop_3d/models/cache`` while the server (root = ``app/``) only
reads ``<project>/app/roop_3d/models/cache``. The imported weights were never found.
"""
from __future__ import annotations

import sys
from pathlib import Path

import roop_3d.installer.model_manager as model_manager
import roop_3d.installer.offline_cache as offline_cache
from roop_3d.audit.project_paths import application_root, project_root

if str(project_root()) not in sys.path:
    sys.path.insert(0, str(project_root()))

from tools import project_admin

SERVER_ROOT = Path(__file__).resolve().parents[2]       # app/, the root roop_3d.server.app uses


def test_cli_root_for_models_is_the_server_root():
    assert project_admin.APP_ROOT == SERVER_ROOT
    assert project_admin.APP_ROOT == project_admin.ROOT / "app"


def test_model_import_verify_and_offline_health_use_the_server_root(monkeypatch, capsys):
    seen: dict[str, Path] = {}

    class Recorder:
        def __init__(self, root, registry=None):
            seen["manager"] = Path(root)
        def import_bundle(self, path):
            return {"imported": str(path)}
        def verify_all_models(self):
            return {"verified": True}

    monkeypatch.setattr(model_manager, "ModelManager", Recorder)
    monkeypatch.setattr(offline_cache, "check_offline_readiness", lambda root, registry=None: seen.setdefault("offline", Path(root)) and {"ok": True})
    project_admin.model_import("bundle.zip")
    project_admin.model_verify()
    project_admin.offline_health()
    capsys.readouterr()
    assert seen["manager"] == SERVER_ROOT
    assert seen["offline"] == SERVER_ROOT
    assert application_root() == SERVER_ROOT
