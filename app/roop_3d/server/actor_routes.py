"""Stage 23: multi-person target actors ("Person #1" ... "Person #N") for one inspection.

Read-only on the inspection; nothing here runs inference. Identity embeddings stay in the local sidecar and are never
sent over HTTP, so the clustering evidence available here is the tracker's ``persistent_id`` (identity-lock
inspections) and otherwise one actor per track.
"""
from __future__ import annotations

import re

from fastapi import APIRouter, Request
from pydantic import Field

from roop_3d.audit.isolation import safe_path
from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract
from roop_3d.media.actors import ActorMappingResult, ActorReport, actor_face_mappings, cluster_actors
from roop_3d.media.schema import FaceInspection

router = APIRouter(prefix="/api/v1/actors")
_FACESET_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ActorAssignments(Contract):
    # actor_id -> FaceSet id. Each FaceSet is a distinct source identity with its own 3D skull lock and atlas.
    assignments: dict[str, str] = Field(max_length=64)


def _inspection(request: Request, inspection_id: str) -> FaceInspection:
    if not re.fullmatch(r"[0-9a-f]{32}", inspection_id):
        raise DomainError("INSPECTION_NOT_FOUND", "Inspect the target first", 404)
    path = safe_path(request.app.state.root, f"roop_3d/.runtime/inspections/{inspection_id}.json")
    if not path.is_file():
        raise DomainError("INSPECTION_NOT_FOUND", "Inspect the target first", 404)
    return FaceInspection.model_validate_json(path.read_text("utf-8"))


@router.get("/{inspection_id}", response_model=ActorReport)
def list_actors(inspection_id: str, request: Request) -> ActorReport:
    """Distinct people in the target, most prominent first."""
    return cluster_actors(_inspection(request, inspection_id))


@router.post("/{inspection_id}/mappings", response_model=ActorMappingResult)
def map_actors(inspection_id: str, value: ActorAssignments, request: Request) -> ActorMappingResult:
    """Turn ``{actor -> FaceSet}`` into per-track ``face_mappings`` ready for a job request."""
    bad = sorted(item for item in value.assignments.values() if not _FACESET_ID.fullmatch(item))
    if bad:
        raise DomainError("FACESET_ID_INVALID", "FaceSet ids may only use letters, digits, '_' and '-'", 422)
    report = cluster_actors(_inspection(request, inspection_id))
    return actor_face_mappings(report, value.assignments)
