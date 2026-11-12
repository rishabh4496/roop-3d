"""Project-owned model outputs, with no link or hardlink traversal."""
from __future__ import annotations

import os
from pathlib import Path

from roop_3d.audit.project_paths import project_root


def owned_path(path: Path) -> Path:
    root = project_root().absolute()
    path = Path(os.path.abspath(path))
    if not path.is_relative_to(root):
        raise ValueError("Model output must stay inside the standalone Roop 3D checkout")
    for part in (*reversed(path.parents), path):
        if os.path.lexists(part):
            info = part.lstat()
            if part.name not in {".runtime", ".venv", "cache"} and (part.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400):
                raise ValueError(f"Link traversal rejected: {part}")
            if part.is_file() and info.st_nlink != 1:
                raise ValueError(f"Hardlinked output rejected: {part}")
    return path


def cache_directory() -> Path:
    path = owned_path(project_root() / "app/models/.trt_cache")
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_source(path: Path) -> Path:
    """External weights are read only through the preserved hashed registry."""
    path = Path(os.path.abspath(path))
    if path.is_relative_to(project_root()):
        return owned_path(path)
    from roop_3d.models.catalog import ModelCatalog
    catalog = ModelCatalog()
    for record in catalog.registry.records():
        if record.name == path.name or path.name in record.filenames:
            registered = catalog.provider.resolve(record)
            if registered.absolute() == path:
                return registered
    raise ValueError("External model is not an exact verified hashed-registry locator")
