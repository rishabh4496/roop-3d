"""Own-project launcher preflight, offline manager, and recovery admin.

Never operates on another Pinokio app (especially roop-ultimate).
"""
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

ROOT = Path(__file__).resolve().parents[1]


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
    if not path.resolve().is_relative_to(ROOT):
        raise RuntimeError("External write rejected")
    return path


def ensure_sys_path() -> None:
    app_dir = str(owned("app"))
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)


def preflight(verbose: bool = False) -> None:
    marker = json.loads(owned(".roop3d-project.json").read_text("utf-8"))
    if marker.get("app_id") != "roop-3d":
        raise RuntimeError("Wrong project identity")

    ensure_sys_path()
    from roop_3d.tools.git_guard import check_branch
    check_branch(ROOT)

    for relative in ("app/roop_3d", "app/roop_3d/.venv", "cache", "cache/uv", "cache/tmp", "cache/backups"):
        path = owned(relative)
        if relative.startswith("cache"):
            path.mkdir(parents=True, exist_ok=True)

    # Run the comprehensive 12-point environment preflight
    from roop_3d.installer.preflight import run_environment_preflight
    report = run_environment_preflight(ROOT)

    if verbose or report["status"] == "FAIL":
        for check in report["checks"]:
            prefix = "[PASS]" if check["status"] == "PASS" else (f"[{check['status']}]")
            print(f"{prefix} {check['name']}: {check['message']}", flush=True)

    if not report["ready"]:
        failed = [c["name"] for c in report["checks"] if c["status"] == "FAIL"]
        raise RuntimeError(f"Environment preflight failed critical checks: {', '.join(failed)}")

    print(f"Roop 3D preflight: PASS (Profile: {report['resolved_hardware_profile']})", flush=True)


def check_env() -> None:
    ensure_sys_path()
    from roop_3d.installer.preflight import run_environment_preflight
    report = run_environment_preflight(ROOT)
    print(json.dumps(report, indent=2), flush=True)


def mark_ready() -> None:
    preflight()
    if Path(sys.prefix).resolve() != owned("app/roop_3d/.venv").resolve():
        raise RuntimeError("Install must run inside this app's own virtual environment")

    ensure_sys_path()
    from roop_3d.tools.install_git_guard import install
    install(ROOT)

    from roop_3d.installer.verification import verify_installation
    res = verify_installation(ROOT)
    print(f"Installation verified: {res['packages_count']} packages checked. Dist built.", flush=True)


def verify_install() -> None:
    ensure_sys_path()
    from roop_3d.installer.verification import verify_installation
    res = verify_installation(ROOT)
    print(json.dumps(res, indent=2), flush=True)


def offline_health() -> None:
    ensure_sys_path()
    from roop_3d.installer.offline_cache import check_offline_readiness
    res = check_offline_readiness(ROOT)
    print(json.dumps(res, indent=2), flush=True)


def model_import(bundle_path: str) -> None:
    ensure_sys_path()
    from roop_3d.installer.model_manager import ModelManager
    manager = ModelManager(ROOT)
    res = manager.import_bundle(Path(bundle_path))
    print(json.dumps(res, indent=2), flush=True)


def model_verify() -> None:
    ensure_sys_path()
    from roop_3d.installer.model_manager import ModelManager
    manager = ModelManager(ROOT)
    res = manager.verify_all_models()
    print(json.dumps(res, indent=2), flush=True)


def check_update() -> None:
    ensure_sys_path()
    from roop_3d.installer.update_manager import check_update_status
    res = check_update_status(ROOT)
    print(json.dumps(res, indent=2), flush=True)


def backup_config() -> None:
    ensure_sys_path()
    from roop_3d.installer.update_manager import create_configuration_backup
    res = create_configuration_backup(ROOT)
    print(json.dumps(res, indent=2), flush=True)


def rollback(backup_id: str | None = None) -> None:
    ensure_sys_path()
    from roop_3d.installer.update_manager import rollback_to_backup
    res = rollback_to_backup(ROOT, backup_id)
    print(json.dumps(res, indent=2), flush=True)


def recover() -> None:
    ensure_sys_path()
    from roop_3d.installer.verification import recover_interrupted_installation
    res = recover_interrupted_installation(ROOT)
    print(json.dumps(res, indent=2), flush=True)


def check_reset() -> None:
    preflight()
    # These are the complete reset targets. No source, model, FaceSet, or output directory is removed.
    owned("app/roop_3d/.venv")
    owned("cache/install-ready.json")
    with socket.socket() as probe:
        probe.settimeout(0.5)
        if probe.connect_ex(("127.0.0.1", int(os.environ.get("ROOP3D_PORT", "8030")))) == 0:
            raise RuntimeError("The API port is still occupied; no environment files were deleted")
    print("Verified reset targets are strictly inside this independent Roop 3D project", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=[
        "preflight", "check-env", "mark-ready", "verify-install",
        "offline-health", "model-import", "model-verify",
        "check-update", "backup-config", "rollback", "recover",
        "check-reset",
    ])
    parser.add_argument("--bundle", default=None, help="Path to model bundle (ZIP or directory) for model-import")
    parser.add_argument("--backup-id", default=None, help="Backup ID for rollback")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose preflight results")

    arguments = parser.parse_args()
    try:
        if arguments.action == "preflight":
            preflight(verbose=arguments.verbose)
        elif arguments.action == "check-env":
            check_env()
        elif arguments.action == "mark-ready":
            mark_ready()
        elif arguments.action == "verify-install":
            verify_install()
        elif arguments.action == "offline-health":
            offline_health()
        elif arguments.action == "model-import":
            if not arguments.bundle:
                raise RuntimeError("--bundle argument is required for model-import")
            model_import(arguments.bundle)
        elif arguments.action == "model-verify":
            model_verify()
        elif arguments.action == "check-update":
            check_update()
        elif arguments.action == "backup-config":
            backup_config()
        elif arguments.action == "rollback":
            rollback(arguments.backup_id)
        elif arguments.action == "recover":
            recover()
        elif arguments.action == "check-reset":
            check_reset()

        print("Roop 3D project check: PASS", flush=True)
        return 0
    except Exception as error:
        print("ROOP3D_GUARD_ERROR: " + str(error), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
