"""REST endpoints for the 10-state backend-owned persistent queue and 4-strategy Batch Matrix."""
from __future__ import annotations

from typing import Annotated, Any
from fastapi import APIRouter, Body, Header, Request, Response
from pydantic import Field

from roop_3d.batch_matrix import (
    BUILTIN_RECIPES,
    BatchMatrixEngine,
    JobManifest,
    RecipeConfig,
)
from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract
from roop_3d.queue_manager import CanonicalJobState, QueueJob, QueueSlice, SegmentSlice
from roop_3d.segment_dispatcher import concat_lossless, split_video_gop

router = APIRouter(prefix="/api/v1")


class SubmitJobRequest(Contract):
    request: dict[str, Any]
    name: str | None = None
    slices: list[dict[str, Any]] | None = None


class ReorderRequest(Contract):
    ordered_job_ids: list[str] = Field(min_length=1)


class RetrySliceRequest(Contract):
    slice_index_or_id: int | str


class ProgressUpdateRequest(Contract):
    frame: int = 0
    total_frames: int | None = None
    fps: float = 0.0
    stage: str = "RENDERING"
    stage_latency_ms: float = 0.0


class BatchGenerateRequest(Contract):
    strategy: str
    source: str | None = None
    sources: list[str] | None = None
    target: Any | None = None
    targets: list[Any] | None = None
    groups: list[dict[str, Any]] | None = None
    recipes: list[dict[str, Any]] | None = None
    output_template: str | None = None
    restorer: str = "auto"
    active_cells: list[Any] | None = None
    workstation_parameters: dict[str, Any] | None = None


class BatchSubmitRequest(BatchGenerateRequest):
    auto_split: bool = False
    max_segment_duration: float = 15.0


class SplitVideoRequest(Contract):
    video_path: str
    max_segment_duration: float = 15.0
    timestamp: str | None = None


class ConcatVideoRequest(Contract):
    slice_paths: list[str] = Field(min_length=1)
    source_audio_video: str | None = None
    output_path: str


# =========================================================================
# Queue Endpoints
# =========================================================================

@router.get("/queue", response_model=list[QueueJob])
def list_queue_jobs(request: Request) -> list[QueueJob]:
    return request.app.state.queue.list_jobs()


@router.post("/queue", response_model=QueueJob, status_code=201)
def submit_queue_job(
    value: SubmitJobRequest,
    request: Request,
    response: Response,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> QueueJob:
    job, created = request.app.state.queue.submit(
        request=value.request,
        idempotency_key=idempotency_key,
        name=value.name,
        slices=value.slices,
    )
    response.status_code = 201 if created else 200
    response.headers["Location"] = f"/api/v1/queue/{job.job_id}"
    return job


@router.get("/queue/{job_id}", response_model=QueueJob)
def get_queue_job(job_id: str, request: Request) -> QueueJob:
    return request.app.state.queue.get(job_id)


@router.post("/queue/{job_id}/cancel", response_model=QueueJob)
def cancel_queue_job(job_id: str, request: Request) -> QueueJob:
    return request.app.state.queue.cancel(job_id)


@router.post("/queue/{job_id}/pause", response_model=QueueJob)
def pause_queue_job(job_id: str, request: Request) -> QueueJob:
    return request.app.state.queue.pause(job_id)


@router.post("/queue/{job_id}/resume", response_model=QueueJob)
def resume_queue_job(job_id: str, request: Request) -> QueueJob:
    return request.app.state.queue.resume(job_id)


@router.post("/queue/{job_id}/progress", response_model=QueueJob)
def update_job_progress(job_id: str, value: ProgressUpdateRequest, request: Request) -> QueueJob:
    return request.app.state.queue.update_progress(job_id, value.model_dump())


@router.post("/queue/reorder", response_model=list[QueueJob])
def reorder_queue(value: ReorderRequest, request: Request) -> list[QueueJob]:
    return request.app.state.queue.reorder(value.ordered_job_ids)


@router.post("/queue/{job_id}/duplicate", response_model=QueueJob, status_code=201)
def duplicate_queue_job(job_id: str, request: Request) -> QueueJob:
    return request.app.state.queue.duplicate(job_id)


@router.post("/queue/{job_id}/retry", response_model=QueueJob)
def retry_failed_slice_route(job_id: str, value: RetrySliceRequest, request: Request) -> QueueJob:
    return request.app.state.queue.retry_failed_slice(job_id, value.slice_index_or_id)


@router.post("/queue/{job_id}/join", response_model=QueueJob)
def join_clip_slices(job_id: str, request: Request) -> QueueJob:
    return request.app.state.queue.automatic_clip_joining(job_id)


# =========================================================================
# Batch Matrix Endpoints
# =========================================================================

@router.get("/batch/recipes", response_model=list[dict[str, Any]])
def get_recipes() -> list[dict[str, Any]]:
    return BUILTIN_RECIPES


@router.post("/batch/generate", response_model=list[JobManifest])
def generate_batch_manifests(value: BatchGenerateRequest) -> list[JobManifest]:
    spec = value.model_dump(exclude_none=True)
    return BatchMatrixEngine.expand_batch(spec)


@router.post("/batch/submit", response_model=list[QueueJob], status_code=201)
def submit_batch(value: BatchSubmitRequest, request: Request) -> list[QueueJob]:
    spec = value.model_dump(exclude_none=True)
    manifests = BatchMatrixEngine.expand_batch(spec)
    queued_jobs: list[QueueJob] = []

    for m in manifests:
        req_payload = m.to_job_payload()
        slices = None
        if value.auto_split:
            target_path = request.app.state.media.path(m.target_path)
            if target_path.is_file():
                try:
                    split_slices = split_video_gop(
                        target_path,
                        request.app.state.root / "roop_3d/.runtime/segments",
                        max_segment_duration=value.max_segment_duration,
                    )
                    slices = [s.model_dump() for s in split_slices]
                except Exception:
                    slices = None

        job, _ = request.app.state.queue.submit(
            request=req_payload,
            name=m.name,
            slices=slices,
        )
        queued_jobs.append(job)

    return queued_jobs


# =========================================================================
# GOP Video Segment Splitting & Lossless Concat Endpoints
# =========================================================================

@router.post("/segments/split", response_model=list[SegmentSlice])
def split_video_endpoint(value: SplitVideoRequest, request: Request) -> list[SegmentSlice]:
    media_path = request.app.state.media.path(value.video_path)
    output_dir = request.app.state.root / "roop_3d/.runtime/segments"
    return split_video_gop(
        media_path,
        output_dir,
        max_segment_duration=value.max_segment_duration,
        timestamp=value.timestamp,
    )


@router.post("/segments/concat", response_model=dict[str, Any])
def concat_video_endpoint(value: ConcatVideoRequest, request: Request) -> dict[str, Any]:
    resolved_slices = [request.app.state.media.path(p) for p in value.slice_paths]
    source_audio = request.app.state.media.path(value.source_audio_video) if value.source_audio_video else None
    out_path = request.app.state.media.path(value.output_path)
    return concat_lossless(
        slice_paths=resolved_slices,
        source_audio_video=source_audio,
        output_path=out_path,
        preserve_intermediates=True,
    )
