"""Stage 14 dry-run API: build a Pipeline plan and its decision report without starting a job.

* ``POST /api/v1/orchestrator/plan``  select Engines (manual, balanced or quality) and return the report.
* ``GET  /api/v1/orchestrator/engines``  the capability registry with per-tier eligibility.

Read-only: it never loads a Model, reserves VRAM or submits work. Live free VRAM does not enter selection;
the tier comes from the request, else from the resolved hardware profile.
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import Field
from starlette.concurrency import run_in_threadpool

from roop_3d.core.hardware import probe_hardware
from roop_3d.core.settings import Contract
from roop_3d.orchestrator import (CapabilityRegistry, EvidenceStore, ManualPick, MediaFacts, OrchestratorRegistryError,
                                  SelectionRequest, Selector, build_report, render_text, segments_from_tracks)
from roop_3d.orchestrator.schema import ADAPTIVE_PROFILE, Category

router = APIRouter(prefix="/api/v1/orchestrator")


class PickBody(Contract):
    category: Category
    engine_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]{0,95}$")
    provider: str | None = None
    precision: str | None = None


class TrackBody(Contract):
    track_id: str = Field(pattern=r"^track_[0-9]{4}$")
    scene_id: int = Field(default=0, ge=0)
    first_frame: int = Field(default=0, ge=0)
    last_frame: int = Field(default=0, ge=0)
    yaw_degrees: tuple[float, float] | None = None
    method: Literal["eye_line_roll_only", "five_point_geometric_estimate"] = "five_point_geometric_estimate"


class MediaBody(Contract):
    width: int = Field(ge=1, le=100_000)
    height: int = Field(ge=1, le=100_000)


class PlanBody(Contract):
    mode: Literal["manual", "balanced", "quality"] = "balanced"
    tier: Literal["cpu", "6gb", "12gb"] | None = None
    profile_id: str = ADAPTIVE_PROFILE
    manual: tuple[PickBody, ...] = Field(default=(), max_length=16)
    budget_mib: int | None = Field(default=None, ge=0, le=1_000_000)
    media: MediaBody | None = None
    tracks: tuple[TrackBody, ...] = Field(default=(), max_length=256)


class _Pose:
    def __init__(self, yaw, method):
        self.yaw_degrees, self.method = yaw, method


class _Track:
    def __init__(self, body: TrackBody):
        self.track_id, self.scene_id = body.track_id, body.scene_id
        self.first_frame, self.last_frame = body.first_frame, body.last_frame
        self.pose_range = _Pose(body.yaw_degrees, body.method)


def registry_for(request: Request) -> CapabilityRegistry:
    cached = getattr(request.app.state, "orchestrator_registry", None)
    if cached is None:
        catalog = request.app.state.models
        cached = CapabilityRegistry.load(models=catalog.registry,
                                         lifecycle=lambda model_id, hardware: catalog.lifecycle(model_id, hardware=hardware))
        request.app.state.orchestrator_registry = cached
    return cached


def _invalid(error: OrchestratorRegistryError) -> JSONResponse:
    return JSONResponse(status_code=error.status, content={"code": error.code, "message": error.message,
                                                            "issues": [item.as_dict() for item in error.issues]})


def _tier(request: Request, body: PlanBody) -> str:
    return body.tier or probe_hardware(request.app.state.settings).resolved_hardware_profile


@router.get("/engines")
async def engines(request: Request, tier: Literal["cpu", "6gb", "12gb"] | None = None):
    def build():
        registry = registry_for(request)
        chosen = tier or probe_hardware(request.app.state.settings).resolved_hardware_profile
        from roop_3d.orchestrator.selector import TIER_BUDGET_MIB
        rows = []
        for engine in registry.engines.values():
            reasons = registry.engine_reasons(engine, chosen, TIER_BUDGET_MIB[chosen])
            rows.append({**engine.model_dump(mode="json"), "digest": engine.digest(), "tier": chosen,
                         "available": not reasons,
                         "unavailable_reasons": [{"code": c.value, "detail": d} for c, d in reasons]})
        return {"registry": dict(registry.digests), "engines": rows}
    try:
        return await run_in_threadpool(build)
    except OrchestratorRegistryError as error:
        return _invalid(error)


@router.post("/plan")
async def plan(request: Request, body: PlanBody):
    def build():
        registry = registry_for(request)
        selector = Selector(registry, EvidenceStore.load())
        selection = selector.select(SelectionRequest(
            mode=body.mode, tier=_tier(request, body), profile_id=body.profile_id,
            manual=tuple(ManualPick(p.category, p.engine_id, p.provider, p.precision) for p in body.manual),
            budget_mib=body.budget_mib,
            media=None if body.media is None else MediaFacts(body.media.width, body.media.height),
            segments=segments_from_tracks([_Track(t) for t in body.tracks], registry.scene_rules)))
        report = build_report(registry, selection)
        return {"report": report, "text": render_text(report)}
    try:
        return await run_in_threadpool(build)
    except OrchestratorRegistryError as error:
        return _invalid(error)
