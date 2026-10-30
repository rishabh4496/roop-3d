"""Local REST contracts; catalog adapters never import legacy application code."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Header, Query, Request, Response

from roop_3d.audit.isolation import safe_path, sha256_file
from roop_3d.core.errors import DomainError
from roop_3d.core.settings import ConfigStore, Settings
from roop_3d.jobs.schema import FaceSetSummary, JobRecord, JobRequest
from roop_3d.media.schema import FaceInspection, InspectRequest, MediaProbe, MediaRequest
from roop_3d.server.contracts import Capabilities, ModelAvailability, ModelSummary
from roop_3d.server.ws import EVENT_ADAPTER, event_stream

router = APIRouter(prefix="/api/v1")


def facesets(root: Path) -> list[FaceSetSummary]:
    base = safe_path(root, "roop_3d/.runtime/facesets")
    if not base.exists():
        return []
    result = []
    try:
        for path in sorted(base.glob("*.json")):
            safe_path(root, path.relative_to(root).as_posix())
            item = FaceSetSummary.model_validate_json(path.read_text("utf-8"))
            if path.stem != item.faceset_id:
                raise ValueError("FaceSet manifest name mismatch")
            result.append(item)
    except (ValueError, OSError):
        raise DomainError("FACESET_CATALOG_INVALID", "Local FaceSet catalog contains an invalid manifest") from None
    return result


@router.get("/config", response_model=Settings)
def config(request: Request) -> Settings:
    return request.app.state.settings


@router.put("/config", response_model=Settings)
def save_config(value: Settings, request: Request) -> Settings:
    with request.app.state.config_lock:
        current = request.app.state.settings
        if any(getattr(value, field) != getattr(current, field) for field in ("host", "port", "dev_cors", "max_body_bytes")):
            raise DomainError("TRANSPORT_RESTART_REQUIRED", "Transport settings must be changed offline before restart")
        ConfigStore(request.app.state.root).save(value)
        request.app.state.settings = value
        return value


@router.get("/system/capabilities", response_model=Capabilities)
def capabilities(request: Request) -> Capabilities:
    return Capabilities(enabled_modes=("media_probe", "frame_preview", "face_inspection", "persistent_job_queue"),
                        models=tuple(ModelSummary.model_validate(item) for item in request.app.state.models.public()),
                        legal_parameter_bounds=Settings.model_json_schema(),
                        inspection_parameter_bounds=InspectRequest.model_json_schema(),
                        job_parameter_bounds=JobRequest.model_json_schema(),
                        event_schema=EVENT_ADAPTER.json_schema())


@router.get("/models", response_model=list[ModelSummary])
def models(request: Request) -> list[dict]:
    return request.app.state.models.public()


@router.post("/models/{model_id}/download", response_model=ModelAvailability)
def download(model_id: str, request: Request) -> ModelAvailability:
    record = request.app.state.models.get(model_id)
    # Local reuse satisfies availability without any media transfer or duplicate weights.
    request.app.state.models.verify(record)
    return ModelAvailability(model_id=model_id, state="VERIFIED_LOCAL")


@router.get("/facesets", response_model=list[FaceSetSummary])
def get_facesets(request: Request) -> list[FaceSetSummary]:
    return facesets(request.app.state.root)


@router.post("/media/probe", response_model=MediaProbe)
def probe(value: MediaRequest, request: Request) -> MediaProbe:
    return request.app.state.media.probe(value.path)


@router.post("/media/inspect-faces", response_model=FaceInspection)
def inspect(value: InspectRequest, request: Request) -> FaceInspection:
    return request.app.state.media.inspect(value)


@router.get("/media/preview-frame", response_class=Response,
            responses={200: {"content": {"image/jpeg": {"schema": {"type": "string", "format": "binary"}}}}})
def preview(request: Request, path: Annotated[str, Query(min_length=1, max_length=1024)],
            timestamp: Annotated[float | None, Query(ge=0, le=86400)] = None,
            frame_index: Annotated[int | None, Query(ge=0, le=9999)] = None,
            comparison_path: Annotated[str | None, Query(max_length=1024)] = None) -> Response:
    image = request.app.state.media.preview(path, timestamp, frame_index, comparison_path)
    return Response(image, media_type="image/jpeg", headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})


def validate_job(request: Request, value: JobRequest) -> None:
    state = request.app.state
    if value.processing == "remote":
        if not state.settings.remote_processing_enabled:
            raise DomainError("REMOTE_PROCESSING_DISABLED", "Remote processing is disabled")
        raise DomainError("REMOTE_ADAPTER_UNSUPPORTED", "No remote provider adapter is installed")
    path = state.media.path(value.target_path)
    receipt = safe_path(state.root, f"roop_3d/.runtime/inspections/{value.inspection_id}.json")
    if not receipt.is_file():
        raise DomainError("INSPECTION_NOT_FOUND", "Inspect the target video before mapping its tracks", 422)
    try:
        inspection = FaceInspection.model_validate_json(receipt.read_text("utf-8"))
    except ValueError:
        raise DomainError("INSPECTION_INVALID", "Saved inspection is invalid") from None
    if inspection.inspection_id != value.inspection_id or inspection.target_sha256 != sha256_file(path)[1]:
        raise DomainError("INSPECTION_TARGET_MISMATCH", "Inspection does not match the current target video")
    track_ids = {item.track_id for item in inspection.tracks}
    faceset_ids = {item.faceset_id for item in facesets(state.root)}
    if any(item.track_id not in track_ids for item in value.face_mappings):
        raise DomainError("TRACK_NOT_FOUND", "Mapping refers to a track absent from this inspection", 422)
    if any(item.faceset_id not in faceset_ids for item in value.face_mappings):
        raise DomainError("FACESET_NOT_FOUND", "Mapping refers to an unknown FaceSet", 422)


@router.post("/jobs", response_model=JobRecord, status_code=201)
def create_job(value: JobRequest, request: Request, response: Response,
               idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=128)]) -> JobRecord:
    record, created = request.app.state.jobs.create(value, idempotency_key, lambda: validate_job(request, value))
    response.status_code = 201 if created else 200
    response.headers["Location"] = f"/api/v1/jobs/{record.job_id}"
    return record


@router.get("/jobs", response_model=list[JobRecord])
def get_jobs(request: Request) -> list[JobRecord]:
    return request.app.state.jobs.list()


@router.get("/jobs/{job_id}", response_model=JobRecord)
def get_job(job_id: str, request: Request) -> JobRecord:
    return request.app.state.jobs.get(job_id)


@router.post("/jobs/{job_id}/cancel", response_model=JobRecord)
def cancel(job_id: str, request: Request) -> JobRecord:
    return request.app.state.jobs.control(job_id, "cancel")


@router.post("/jobs/{job_id}/pause", response_model=JobRecord)
def pause(job_id: str, request: Request) -> JobRecord:
    return request.app.state.jobs.control(job_id, "pause")


@router.post("/jobs/{job_id}/resume", response_model=JobRecord)
def resume(job_id: str, request: Request) -> JobRecord:
    return request.app.state.jobs.control(job_id, "resume")


router.add_api_websocket_route("/ws/events", event_stream)
