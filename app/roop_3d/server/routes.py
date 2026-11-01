"""Local REST contracts; catalog adapters never import legacy application code."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Body, Header, Query, Request, Response
from pydantic import Field

from roop_3d.audit.isolation import safe_path, sha256_file
from roop_3d.core.errors import DomainError
from roop_3d.core.settings import ConfigStore, Contract, Settings
from roop_3d.jobs.schema import FaceSetSummary, JobRecord, JobRequest
from roop_3d.media.schema import FaceInspection, InspectRequest, MediaProbe, MediaRequest
from roop_3d.models.registry import ModelRecord
from roop_3d.server.contracts import (Capabilities, ModelAvailability, ModelSummary,
                                     ModelDownloadRequest, ModelLicenseNotice)
from roop_3d.server.diagnostics import BenchmarkRequest, PreFlightLintRequest
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
    from roop_3d.jobs.schema import JobState
    from roop_3d.core.hardware import probe_hardware
    from roop_3d.server.workstation import CHUNK_BYTES, MAX_ASSET_BYTES
    state = request.app.state
    hardware = probe_hardware(state.settings)
    available = {item["name"] for item in state.models.public()}
    gpu = bool(hardware.gpus) and "CUDAExecutionProvider" in hardware.advertised_providers
    lp = all(name in available for name in ("liveportrait_appearance_feature_extractor.onnx", "liveportrait_motion_extractor.onnx", "liveportrait_warping_spade.onnx"))
    stream = state.settings.render_backend == "stream" or state.settings.render_backend == "auto" and gpu
    import importlib.metadata
    try:
        importlib.metadata.version("tensorrt-cu12-libs")
        trt = gpu and stream
    except importlib.metadata.PackageNotFoundError:
        trt = False
    try:
        from roop_3d.compositor.stream_restoration import load_ultramax_spec
        ultramax = load_ultramax_spec(state.root).model_name in available
    except (ValueError, OSError, TypeError, KeyError):
        ultramax = False
    options = {
        "execution_mode": {"swap": None, "volumetric": None if lp else "LivePortrait weights are unavailable", "hybrid": "No verified diffusion loader is configured", "diffusion_vfx": "No verified diffusion loader is configured"},
        "preferred_provider": {"auto": None, "CPUExecutionProvider": None, "CUDAExecutionProvider": None if gpu else "CUDA is unavailable", "TensorrtExecutionProvider": None if trt else "TensorRT requires the stream backend, NVIDIA GPU and packaged TensorRT libraries"},
        "hardware_profile": {"auto": None, "cpu": None, "6gb": None if gpu else "No CUDA GPU is available", "12gb": None if gpu and hardware.gpus[0].total_vram_mib >= 12288 else "A 12 GB CUDA GPU is required"},
        "precision_policy": {"auto": None, "fp32": None, "fp16": "The reference session uses observed FP32 graph contracts"},
        "face_restoration_policy": {"off": None, "auto": None, "gfpgan": "No verified GFPGAN adapter",
            "gpen": None if stream and "gpen_bfr_512.onnx" in available else "GPEN requires the stream backend and registered weights",
            "codeformer": None if stream and "CodeFormerv0.1.onnx" in available else "CodeFormer requires the stream backend and registered weights",
            "ultramax": None if stream and ultramax else "UltraMax requires an explicit registered model and preprocessing contract"},
        "render_backend": {"auto": None, "reference": None, "stream": None},
        "lip_sync_mode": {"off": None, "wav2lip": "No verified Wav2Lip inference adapter"},
        "audio_policy": {"copy": None, "transcode": None, "drop": "The workstation preserves source audio; use copy or transcode"},
    }
    workstation = {"features": {key: {"enabled": True, "reason": None} for key in
        ("uploads", "frame_preview", "face_inspection", "pose_inspection", "pose_mesh_blend", "reconstruction_viewer", "projects", "trim", "video_playback")},
        "controls": options, "read_only_parameters": {key: "Configure this infrastructure setting offline" for key in
        ("schema_version", "host", "port", "dev_cors", "max_body_bytes", "cache_root", "output_root", "model_search_paths", "diagnostics_level")},
        "disabled_parameters": {"remote_processing_enabled": "No remote adapter is configured", "synthetic_media_notice": "Synthetic media notice is required"},
        "job_states": [item.value for item in JobState],
        "job_controls": {"pause": ["queued", "preparing", "running"], "resume": ["paused", "interrupted"],
                         "cancel": ["queued", "preparing", "running", "pausing", "paused", "interrupted"]},
        "upload_chunk_bytes": min(CHUNK_BYTES, state.settings.max_body_bytes), "max_asset_bytes": MAX_ASSET_BYTES}
    workstation["features"]["queue"] = {"enabled": state.worker is not None, "reason": None if state.worker is not None else "Background worker is disabled"}
    return Capabilities(enabled_modes=("media_probe", "frame_preview", "face_inspection", "persistent_job_queue",
                                       "local_video_render", "workstation", "projects"), workstation=workstation,
                        rendering_worker_available=request.app.state.worker is not None,
                        models=tuple(ModelSummary.model_validate(item) for item in request.app.state.models.public()),
                        legal_parameter_bounds=Settings.model_json_schema(),
                        inspection_parameter_bounds=InspectRequest.model_json_schema(),
                        job_parameter_bounds=JobRequest.model_json_schema(),
                        event_schema=EVENT_ADAPTER.json_schema())


@router.get("/models", response_model=list[ModelSummary])
def models(request: Request) -> list[dict]:
    return request.app.state.models.public()


@router.get("/models/registry", response_model=list[ModelRecord])
def model_registry(request: Request) -> tuple[ModelRecord, ...]:
    return request.app.state.models.registry.records()


@router.get("/models/{model_id}/license", response_model=ModelLicenseNotice)
def model_license(model_id: str, request: Request) -> dict:
    return request.app.state.models.license(model_id)


@router.post("/models/{model_id}/download", response_model=ModelAvailability)
def download(model_id: str, request: Request,
             value: Annotated[ModelDownloadRequest | None, Body()] = None) -> ModelAvailability:
    value = value or ModelDownloadRequest()
    result = request.app.state.models.ensure_available(model_id, accepted_license=value.accepted_license,
                                                       commercial=value.commercial)
    return ModelAvailability(model_id=model_id, state="VERIFIED_LOCAL", downloaded=result.downloaded,
                             resumed=result.resumed)


@router.get("/facesets", response_model=list[FaceSetSummary])
def get_facesets(request: Request) -> list[FaceSetSummary]:
    return facesets(request.app.state.root)


@router.post("/media/probe", response_model=MediaProbe)
def probe(value: MediaRequest, request: Request) -> MediaProbe:
    return request.app.state.media.probe(value.path)


@router.post("/media/inspect-faces", response_model=FaceInspection)
def inspect(value: InspectRequest, request: Request) -> FaceInspection:
    return request.app.state.media.inspect(value)


class FrameFaceDetectionRequest(Contract):
    path: str = Field(min_length=1, max_length=1024)
    timestamp: float | None = Field(default=None, ge=0, le=86400)
    frame_index: int | None = Field(default=None, ge=0, le=9999)
    threshold: float = Field(default=0.5, ge=0.1, le=0.99)


@router.post("/media/detect-frame-faces")
def detect_frame_faces(value: FrameFaceDetectionRequest, request: Request) -> dict:
    return request.app.state.media.detect_frame_faces(
        value.path, value.timestamp, value.frame_index, value.threshold
    )


@router.get("/media/preview-frame", response_class=Response,
            responses={200: {"content": {"image/jpeg": {"schema": {"type": "string", "format": "binary"}}}}})
def preview(request: Request, path: Annotated[str, Query(min_length=1, max_length=1024)],
            timestamp: Annotated[float | None, Query(ge=0, le=86400)] = None,
            frame_index: Annotated[int | None, Query(ge=0, le=9999)] = None,
            comparison_path: Annotated[str | None, Query(max_length=1024)] = None) -> Response:
    image = request.app.state.media.preview(path, timestamp, frame_index, comparison_path)
    return Response(image, media_type="image/jpeg", headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})


class SplitBoundariesRequest(Contract):
    target_path: str = Field(min_length=1, max_length=1024)
    max_segment_duration: float = Field(default=15.0, gt=0, le=3600)


class ConcatSegmentsRequest(Contract):
    segment_paths: list[str] = Field(min_length=1, max_length=256)
    output_filename: str | None = None


@router.post("/media/split-boundaries")
def split_boundaries(value: SplitBoundariesRequest, request: Request) -> list[dict]:
    return request.app.state.media.segment_boundaries(value.target_path, value.max_segment_duration)


@router.post("/media/concat-segments")
def concat_segments(value: ConcatSegmentsRequest, request: Request) -> dict:
    return request.app.state.media.concat_segments(value.segment_paths, value.output_filename)


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
    if value.face_segments:
        if any(item.track_id not in track_ids for item in value.face_segments):
            raise DomainError('TRACK_NOT_FOUND', 'Face role refers to an unknown track', 422)
        if any(item.faceset_id and item.faceset_id not in faceset_ids for item in value.face_segments):
            raise DomainError('FACESET_NOT_FOUND', 'Face role refers to an unknown FaceSet', 422)


@router.post("/jobs", response_model=JobRecord, status_code=201)
def create_job(value: JobRequest, request: Request, response: Response,
               idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=128)]) -> JobRecord:
    record, created = request.app.state.jobs.create(value, idempotency_key, lambda: validate_job(request, value))
    if created and request.app.state.worker is not None:
        request.app.state.worker.wake()
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
    record = request.app.state.jobs.control(job_id, "cancel")
    if request.app.state.worker is not None:
        request.app.state.worker.wake()
    return record


@router.post("/jobs/{job_id}/pause", response_model=JobRecord)
def pause(job_id: str, request: Request) -> JobRecord:
    record = request.app.state.jobs.control(job_id, "pause")
    if request.app.state.worker is not None:
        request.app.state.worker.wake()
    return record


@router.post("/jobs/{job_id}/resume", response_model=JobRecord)
def resume(job_id: str, request: Request) -> JobRecord:
    record = request.app.state.jobs.control(job_id, "resume")
    if request.app.state.worker is not None:
        request.app.state.worker.wake()
    return record


@router.post("/system/preflight-lint")
def preflight_lint(request: Request, value: PreFlightLintRequest | None = None) -> dict:
    from roop_3d.server.diagnostics import PreFlightLintRequest as PReq, run_preflight_lint
    val = value or PReq(target_path="roop_3d/.runtime/media/default.mp4")
    return run_preflight_lint(request.app.state.root, request.app.state, val)


class OptimizeModelsRequest(Contract):
    models: list[str] | None = None


@router.post("/system/optimize-models")
def optimize_models_route(request: Request, value: OptimizeModelsRequest | None = None) -> dict:
    from roop_3d.server.diagnostics import optimize_models
    return optimize_models(request.app.state, (value.models if value else None))


@router.post("/system/benchmark")
def run_benchmark(request: Request, value: BenchmarkRequest | None = None) -> dict:
    from roop_3d.server.diagnostics import BenchmarkRequest as BReq, run_synthetic_benchmark
    val = value or BReq()
    return run_synthetic_benchmark(request.app.state, val)


@router.get("/system/diagnostics-report")
def diagnostics_report(request: Request) -> dict:
    from roop_3d.server.diagnostics import generate_14_section_report
    return generate_14_section_report(request.app.state.root, request.app.state)


router.add_api_websocket_route("/ws/events", event_stream)
router.add_api_websocket_route("/ws/telemetry", event_stream)
