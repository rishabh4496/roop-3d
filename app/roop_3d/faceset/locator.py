"""One resolver for "which archive is this FaceSet?": the runtime catalog and the ``app/facesets`` library.

FaceSets exist in two places. Uploaded / extracted ones are catalogued (``roop_3d/.runtime/facesets/<id>.json`` + an
archive under ``roop_3d/outputs/facesets``). The library directory ``app/facesets`` holds ready ``<name>.fsz`` archives
(``harjot.fsz``) that the FaceSet builder lists by file stem but that have no catalog entry, so a lookup that demanded a
catalog manifest answered ``FACESET_NOT_FOUND`` for a FaceSet the user can see. Resolution order for a key:

1. a catalogued FaceSet id;
2. a library archive whose stem is the key;
3. a catalogued FaceSet whose display name is the key (case-insensitive; ambiguity is an error, never a guess).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

from roop_3d.audit.isolation import safe_path
from roop_3d.core.errors import DomainError

_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
LIBRARY_DIR = "facesets"


@dataclass(frozen=True)
class ResolvedFaceSet:
    faceset_id: str
    archive: Path
    source: str                       # "catalog" | "library"


def _catalog_archive(root: Path, faceset_id: str) -> Path | None:
    for relative in (f"roop_3d/outputs/facesets/{faceset_id}.fsz", f"roop_3d/.runtime/facesets/{faceset_id}.fsz"):
        candidate = safe_path(root, relative)
        if candidate.is_file():
            return candidate
    return None


def resolve_faceset(root: Path, key: str) -> ResolvedFaceSet:
    if not _ID.fullmatch(key or ""):
        raise DomainError("FACESET_ID_INVALID", "FaceSet id is invalid", 422)
    catalog = safe_path(root, "roop_3d/.runtime/facesets")
    if (catalog / f"{key}.json").is_file():
        archive = _catalog_archive(root, key)
        if archive is None:
            raise DomainError("FACESET_ARCHIVE_NOT_FOUND", "FaceSet archive is missing", 404)
        return ResolvedFaceSet(key, archive, "catalog")
    library = safe_path(root, f"{LIBRARY_DIR}/{key}.fsz")
    if library.is_file():
        return ResolvedFaceSet(key, library, "library")
    named: list[str] = []
    if catalog.is_dir():
        for manifest in sorted(catalog.glob("*.json")):
            try:
                if str(json.loads(manifest.read_text("utf-8")).get("name", "")).casefold() == key.casefold():
                    named.append(manifest.stem)
            except (OSError, ValueError):
                continue
    if len(named) > 1:
        raise DomainError("FACESET_AMBIGUOUS", "Several FaceSets share that name; use the FaceSet id", 409)
    if named:
        archive = _catalog_archive(root, named[0])
        if archive is None:
            raise DomainError("FACESET_ARCHIVE_NOT_FOUND", "FaceSet archive is missing", 404)
        return ResolvedFaceSet(named[0], archive, "catalog")
    raise DomainError("FACESET_NOT_FOUND", "No FaceSet has that id or name", 404)
