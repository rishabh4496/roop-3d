"""Resolve the independent app and explicitly read-only legacy model source."""
from __future__ import annotations

import json
import os
from pathlib import Path


def application_root() -> Path:
    return Path(__file__).absolute().parents[2]


def project_root() -> Path:
    return application_root().parent


def legacy_root(app_root: Path | None = None) -> Path:
    app_root = app_root or application_root()
    settings_path = app_root / "roop_3d/audit/model_sources.json"
    settings = json.loads(settings_path.read_text("utf-8"))
    if settings.get("read_only") is not True:
        raise ValueError("External model sources must be explicitly read-only")
    if settings.get("root"):
        path = Path(settings["root"])
        return (path if path.is_absolute() else app_root / path).absolute()
    pinokio = os.environ.get("PINOKIO_HOME")
    if not pinokio:
        config = json.loads((Path.home() / ".pinokio/config.json").read_text("utf-8"))
        pinokio = config["home"]
    return Path(pinokio).absolute() / "api" / settings["pinokio_app_id"]
