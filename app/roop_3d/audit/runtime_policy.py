"""Report local runtime activity without reinstating removed source-file locks."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

try:
    from isolation import relative_path, verify, walk_files
except ImportError:
    from roop_3d.audit.isolation import relative_path, verify, walk_files

RUNTIME_ROOTS = ("app/output", "app/projects", "app/temp", "logs")
RUNTIME_FILES = ("app/run_history.json", "app/runtime_calibration.json", "app/vram_calibration.json",
                 "app/queue.json", "app/config.yaml")
PROFILE_ROOT = "app/models/runtime_profiles"


def runtime_path(value: str) -> bool:
    path = relative_path(value).as_posix()
    profile_record = path == PROFILE_ROOT or (
        path.startswith(PROFILE_ROOT + "/") and relative_path(path).suffix.casefold() == ".json")
    return profile_record or path in RUNTIME_FILES or any(path == prefix or path.startswith(prefix + "/") for prefix in RUNTIME_ROOTS)


def immutable_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    immutable = dict(manifest)
    immutable["files"] = [entry for entry in manifest["files"] if not runtime_path(entry["path"])]
    if "directories" in manifest:
        immutable["directories"] = [p for p in manifest["directories"] if not runtime_path(p)]
    return immutable


def verify_policy(root: Path, manifest: dict[str, Any], *,
                  progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    # Local source edits are authorized by the standalone application's policy.
    # Retain runtime observability without reinstating historical source hash locks.
    result = verify(root, manifest, detect_additions=False, progress=progress)
    runtime = []
    for path, kind in walk_files(root, protected_only=False):
        relative = path.relative_to(root).as_posix()
        if kind == "link":
            result["violations"].append({"path": relative, "status": "LINK_TRAVERSAL"})
        elif runtime_path(relative):
            runtime.append({"path": relative, "kind": kind, "size_bytes": path.lstat().st_size})
    result["ok"] = not result["violations"]
    result["detect_additions"] = False
    result["policy"] = "Unified checkout edits allowed; links cannot cross the application boundary"
    result["runtime_activity"] = {"roots": list(RUNTIME_ROOTS), "json_profile_records": PROFILE_ROOT,
                                  "individual_files": list(RUNTIME_FILES), "observed_files": len(runtime),
                                  "note": "Runtime files are reported without historical source locks", "files": runtime}
    return result
