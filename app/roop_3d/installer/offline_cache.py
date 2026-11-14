"""Read-only offline input checks, separate from provider and render execution."""
from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any

from packaging.markers import default_environment
from packaging.requirements import Requirement, InvalidRequirement
from packaging.utils import canonicalize_name

from roop_3d.models.catalog import ModelCatalog
from roop_3d.models.pure3d import resolve_components
from roop_3d.models.registry import ModelRegistry


def check_offline_dependency_cache(root: Path) -> dict[str, Any]:
    """Compare the selected venv's distribution metadata against the pinned lock.

    Metadata presence/version is not a DLL import, wheelhouse-completeness, or
    successful inference claim. No subprocess or network request is made.
    """
    package_root = root / "roop_3d" if (root / "roop_3d").is_dir() else root / "app/roop_3d"
    lock = package_root / "requirements.lock"
    venv = package_root / ".venv"
    uv_cache = root / "cache/uv" if (root / "cache/uv").is_dir() else root.parent / "cache/uv"
    present = (venv / "pyvenv.cfg").is_file()
    sites = [venv / "Lib/site-packages", *venv.glob("lib/python*/site-packages")]
    installed = {canonicalize_name(d.metadata["Name"]): d.version
                 for d in metadata.distributions(path=[str(p) for p in sites if p.is_dir()])
                 if d.metadata.get("Name")}
    missing = []
    mismatched = []
    invalid = []
    environment = default_environment()
    if present:
        for line in (venv / "pyvenv.cfg").read_text("utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() in {"version", "version_info"}:
                version = value.strip()
                environment.update(python_full_version=version, python_version=".".join(version.split(".")[:2]))
    checked = 0
    if lock.is_file():
        for line in lock.read_text("utf-8").splitlines():
            if not line or line[0].isspace() or line.startswith(("#", "-")):
                continue
            text = line.rstrip().removesuffix("\\").strip()
            try:
                requirement = Requirement(text)
            except InvalidRequirement:
                invalid.append(text)
                continue
            if requirement.marker and not requirement.marker.evaluate(environment):
                continue
            checked += 1
            name = canonicalize_name(requirement.name)
            actual = installed.get(name)
            if actual is None:
                missing.append(requirement.name)
            elif actual not in requirement.specifier:
                mismatched.append({"name": requirement.name, "expected": str(requirement.specifier), "actual": actual})
    complete = present and checked > 0 and not missing and not mismatched and not invalid
    archives = sum(1 for p in uv_cache.rglob("*") if p.is_file() and (p.suffix == ".whl" or p.name.endswith(".tar.gz"))) if uv_cache.is_dir() else 0
    return {"status": "READY" if complete else "WARN", "requirements_lock_present": lock.is_file(),
            "uv_cache_present": uv_cache.is_dir(), "cached_archives_count": archives,
            "venv_present": present, "venv_path": str(venv), "installed_metadata_matches_lock": complete,
            "requirements_checked": checked, "missing_distributions": missing, "version_mismatches": mismatched,
            "unparsed_requirements": invalid, "runtime_imports_verified": False, "wheelhouse_complete": None,
            "message": ("Installed distribution metadata matches the lock; native imports and model forwards remain unverified."
                        if complete else "The selected virtual environment is missing or differs from the dependency lock.")}


def check_offline_readiness(root: Path, registry: ModelRegistry | None = None) -> dict[str, Any]:
    """Resolve every required pure-3D input without claiming the renderer can run."""
    registry = registry or ModelRegistry(root)
    catalog = ModelCatalog(root, registry=registry)
    components = resolve_components(catalog, root=root)
    missing = [row["component"] for row in components if row["required"] and row["state"] != "READY"]
    dependencies = check_offline_dependency_cache(root)
    from roop_3d.core.true3d_guard import NOT_WIRED_CODE, NOT_WIRED_MESSAGE, renderer_is_wired
    wired = renderer_is_wired()
    inputs_ready = not missing and dependencies["installed_metadata_matches_lock"]
    # A resolver cannot establish successful sessions/forwards. Require those
    # independently; a wired hook alone is not execution evidence either.
    return {"offline_ready": False, "offline_inputs_ready": inputs_ready, "status": "WARN",
            "core_models": components, "missing_count": len(missing), "missing_models": missing,
            "dependencies": dependencies, "zero_remote_calls": True,
            "renderer_wired": wired, "execution_verified": False,
            "blockers": ([{"code": NOT_WIRED_CODE, "message": NOT_WIRED_MESSAGE}] if not wired else [])
                        + [{"code": "COMPONENT_FORWARDS_NOT_VERIFIED",
                            "message": "Provider/session and standalone component forwards require separate verification."}],
            "message": "Offline input resolution is reported separately; this check does not prove a working render or successful inference."}
