"""Own-project launcher preflight; never operates on another Pinokio app."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).absolute().parents[1]


def owned(relative: str) -> Path:
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts:
        raise RuntimeError("Path is not inside this Roop 3D app")
    path = ROOT / part
    for ancestor in (*reversed(ROOT.parents), ROOT, *[ROOT.joinpath(*part.parts[:n]) for n in range(1, len(part.parts) + 1)]):
        if os.path.lexists(ancestor):
            info = ancestor.lstat()
            if ancestor.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
                raise RuntimeError("Refusing a symlink/junction in the standalone app write path")
    if not path.absolute().is_relative_to(ROOT):
        raise RuntimeError("External write rejected")
    return path


def preflight() -> None:
    marker = json.loads(owned(".roop3d-project.json").read_text("utf-8"))
    if marker.get("app_id") != "roop-3d":
        raise RuntimeError("Wrong project identity")
    sys.path.insert(0, str(owned("app")))
    from roop_3d.tools.git_guard import check_branch
    check_branch(ROOT)
    for relative in ("app/roop_3d", "app/roop_3d/.venv", "cache", "cache/uv", "cache/tmp"):
        path = owned(relative)
        if relative.startswith("cache"):
            path.mkdir(parents=True, exist_ok=True)


def mark_ready() -> None:
    preflight()
    if Path(sys.prefix).absolute() != owned("app/roop_3d/.venv"):
        raise RuntimeError("Install must run inside this app's own virtual environment")
    for package in ("fastapi", "uvicorn", "pydantic", "numpy", "av", "opencv-python-headless", "httpx", "PyYAML"):
        importlib.metadata.version(package)
    sys.path.insert(0, str(owned("app")))
    from roop_3d.tools.install_git_guard import install
    install(ROOT)
    digest = hashlib.sha256(owned("app/roop_3d/requirements.lock").read_bytes()).hexdigest()
    owned("cache/install-ready.json").write_text(json.dumps({"app_id": "roop-3d", "requirements_sha256": digest,
                                                           "environment": "app/roop_3d/.venv"}) + "\n", encoding="utf-8")


def check_reset() -> None:
    preflight()
    # These are the complete reset targets. No source, model or output directory is removed.
    owned("app/roop_3d/.venv")
    owned("cache/install-ready.json")
    with socket.socket() as probe:
        probe.settimeout(0.5)
        if probe.connect_ex(("127.0.0.1", int(os.environ.get("ROOP3D_PORT", "8030")))) == 0:
            raise RuntimeError("The API port is still occupied; no environment files were deleted")
    print("Verified reset targets are strictly inside this independent Roop 3D project", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["preflight", "mark-ready", "check-reset"])
    arguments = parser.parse_args()
    try:
        {"preflight": preflight, "mark-ready": mark_ready, "check-reset": check_reset}[arguments.action]()
        print("Roop 3D project check: PASS", flush=True)
        return 0
    except Exception as error:
        print("ROOP3D_GUARD_ERROR: " + str(error), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
