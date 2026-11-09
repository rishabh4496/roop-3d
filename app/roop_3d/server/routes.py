"""Local REST contracts; catalog adapters never import legacy application code."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Body, Header, Query, Request, Response
from fastapi.responses import FileResponse
from pydantic import Field

from roop_3d.audit.isolation import safe_path, sha256_file
from roop_3d.core.errors import DomainError
from roop_3d.core.settings import ConfigStore, Contract, Settings
from roop_3d.core.storage import owned
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
    from roop_3d.core.hardware import MIN_12GB_PROFILE_VRAM_MIB, probe_hardware
    from roop_3d.server.workstation import CHUNK_BYTES, MAX_ASSET_BYTES
    state = request.app.state
    hardware = probe_hardware(state.settings)
    precision_status = hardware.tensorrt_mixed_precision
    model_rows = state.models.public()
    available = {item['id'] for item in model_rows if item.get('installed') and item.get('state') == 'VERIFIED_LOCAL'}
    available.update(item['name'] for item in model_rows if item.get('installed') and item.get('state') == 'VERIFIED_LOCAL')
    gpu = ("CUDAExecutionProvider" in (hardware.advertised_providers or [])
           and (bool(hardware.gpus) or bool(precision_status.get("cuda_fp16_available"))))
    lp = all(name in available for name in ("liveportrait_appearance_feature_extractor.onnx", "liveportrait_motion_extractor.onnx", "liveportrait_warping_spade.onnx"))
    trt = bool(precision_status.get("tensorrt_available"))
    cuda_fp16 = bool(precision_status.get("cuda_fp16_available"))
    mixed_precision = bool(precision_status.get("available") or trt or cuda_fp16)
    try:
        from roop_3d.compositor.stream_restoration import load_ultramax_spec
        ultramax = load_ultramax_spec(state.root).model_name in available
    except (ValueError, OSError, TypeError, KeyError):
        ultramax = False
    options = {
        "swapper_model_id": {item["id"]: None for item in state.models.public() if item["category"] == "swapper"},
        "execution_mode": {"swap": None, "volumetric": None if lp else "LivePortrait weights are unavailable", "hybrid": "No verified diffusion loader is configured", "diffusion_vfx": "No verified diffusion loader is configured"},
        "preferred_provider": {"auto": None, "CPUExecutionProvider": None, "CUDAExecutionProvider": None if gpu else "CUDA is unavailable", "TensorrtExecutionProvider": None if trt else "TensorRT requires an NVIDIA GPU, its execution provider and packaged TensorRT libraries"},
        "hardware_profile": {"auto": None, "cpu": None, "6gb": None if gpu else "No CUDA GPU is available", "12gb": None if gpu and hardware.gpus and hardware.gpus[0].total_vram_mib >= MIN_12GB_PROFILE_VRAM_MIB else "A CUDA GPU with at least 10 GB of video memory is required"},
        "precision_policy": {"auto": None, "fp32": None, "fp16": None if mixed_precision else "FP16 mixed precision requires TensorRT or CUDA FP16 support"},
        "detector_model_id": {key: None for key in ('scrfd', 'retinaface', 'yoloface')},
        "face_restoration_policy": {"off": None, "auto": None, "gfpgan": None if 'gfpgan_v14' in available else "GFPGAN weights are unavailable",
            "gpen": None if 'gpen_bfr_512' in available else "GPEN weights are unavailable",
            "codeformer": None if 'codeformer' in available else "CodeFormer weights are unavailable",
            "ultramax": None if ultramax else "UltraMax requires an installed model and preprocessing contract"},
        "render_backend": {"auto": None, "reference": None, "stream": None},
        "lip_sync_mode": {"off": None, "wav2lip": None, "wav2lip_gan_fp16": None,
                          "musetalk": "Whisper/VAE engine bundle unavailable", "edtalk": "Pose/emotion engine bundle unavailable"},
        "audio_policy": {"copy": None, "transcode": None, "drop": "The workstation preserves source audio; use copy or transcode"},
    }
    workstation = {"features": {key: {"enabled": True, "reason": None} for key in
        ("uploads", "frame_preview", "face_inspection", "pose_inspection", "pose_mesh_blend", "reconstruction_viewer", "projects", "trim", "video_playback")},
        "controls": options, "read_only_parameters": {key: "Configure this infrastructure setting offline" for key in
        ("schema_version", "host", "port", "dev_cors", "max_body_bytes", "cache_root", "output_root", "model_search_paths", "diagnostics_level")},
        "disabled_parameters": {"remote_processing_enabled": "No remote adapter is configured", "synthetic_media_notice": "Synthetic media notice is required"},
        "job_states": [item.value for item in JobState],
        "canonical_job_states": ["QUEUED", "PREFLIGHT", "ANALYZING", "PRESCANNING", "PREPASSING", "RENDERING", "MUXING", "PAUSED", "COMPLETED", "FAILED", "RETRYING", "CANCELLED"],
        "job_controls": {"pause": ["queued", "preparing", "running"], "resume": ["paused", "interrupted"],
                         "cancel": ["queued", "preparing", "running", "pausing", "paused", "interrupted"]},
        "upload_chunk_bytes": min(CHUNK_BYTES, state.settings.max_body_bytes), "max_asset_bytes": MAX_ASSET_BYTES}
    workstation["features"]["queue"] = {"enabled": state.worker is not None, "reason": None if state.worker is not None else "Background worker is disabled"}
    workstation["tensorrt_mixed_precision"] = precision_status
    workstation["active_hardware_profile"] = state.settings.hardware_profile
    workstation["resolved_hardware_profile"] = getattr(hardware, "resolved_hardware_profile", "auto")
    return Capabilities(enabled_modes=("media_probe", "frame_preview", "face_inspection", "persistent_job_queue",
                                       "local_video_render", "workstation", "projects"), workstation=workstation,
                        rendering_worker_available=request.app.state.worker is not None,
                        models=tuple(ModelSummary.model_validate(item) for item in request.app.state.models.public()),
                        legal_parameter_bounds=Settings.model_json_schema(),
                        inspection_parameter_bounds=InspectRequest.model_json_schema(),
                        job_parameter_bounds=JobRequest.model_json_schema(),
                        event_schema=EVENT_ADAPTER.json_schema())


class BatchModelDownloadRequest(Contract):
    model_id: str | None = None
    models: list[str] = Field(default_factory=list)
    accepted_license: bool = False
    commercial: bool = False


class ModelVerifyRequest(Contract):
    model_id: str | None = None
    models: list[str] = Field(default_factory=list)


class ModelCompileTrtRequest(Contract):
    model_id: str | None = None
    models: list[str] = Field(default_factory=list)
    precision: Literal['fp16', 'fp32'] = "fp16"


@router.get("/models", response_model=list[ModelSummary])
def models(request: Request) -> list[dict]:
    from roop_3d.core.hardware import probe_hardware
    status = probe_hardware(request.app.state.settings).tensorrt_mixed_precision
    return [{**item, "tensorrt_mixed_precision": status}
            for item in request.app.state.models.public()]


@router.get("/models/active")
def active_model_readback(request: Request) -> dict[str, Any]:
    """The model file each stage has loaded right now, next to what the saved settings select.

    ``active`` is null for a stage that has no session in memory (nothing rendered or previewed yet, or its
    session was released); ``selected`` is what the next preview/render will load.
    """
    from roop_3d.engine import active_models
    return {"selected": _selected_models(request.app.state.settings), "active": active_models.snapshot()}


def _selected_models(settings: Settings) -> dict[str, str]:
    return {"swapper_model": settings.swapper_model_id, "detector_model": settings.detector_model_id,
            "enhancer_model": settings.face_restoration_policy}


_PROVIDER_NAMES = {"TensorrtExecutionProvider": "TensorRT", "CUDAExecutionProvider": "CUDA", "CPUExecutionProvider": "CPU",
                   "DmlExecutionProvider": "DirectML", "ROCMExecutionProvider": "ROCm", "CoreMLExecutionProvider": "CoreML"}


def precision_label(provider: str | None, policy: str | None) -> str | None:
    """What a loaded session really runs as, from its execution provider and the precision policy it was built with.

    TensorRT with the ``fp16`` policy is built through ``tensorrt_mixed`` (FP16 mixed precision); with ``auto`` it still builds
    FP16 engines but without the mixed-precision options, and the policy is named rather than guessed."""
    if not provider:
        return None
    name = _PROVIDER_NAMES.get(provider, provider.replace("ExecutionProvider", ""))
    policy = policy or "auto"
    if provider == "CPUExecutionProvider":
        return "CPU FP32"
    if policy == "fp32":
        return f"{name} FP32"
    if provider == "TensorrtExecutionProvider" and policy == "fp16":
        return "TensorRT FP16 Mixed"
    if policy == "fp16":
        return f"{name} FP16"
    return f"{name} (auto precision)"


@router.get("/pipeline/status")
def pipeline_status(request: Request) -> dict[str, Any]:
    """Live engine read-back for the pipeline bar (`GET /api/pipeline/status`).

    ``active`` is what each stage has loaded right now, ``precision`` is derived from those sessions' real provider and
    precision policy, and ``failures`` carries the exact backend code and reason of the latest model that could not be
    loaded for a stage (the previous model is not kept silently). ``pipeline`` is the same frame the WebSocket streams."""
    from roop_3d.engine import active_models
    state = request.app.state
    active = active_models.snapshot()
    loaded = [active[stage] for stage in ("swapper", "enhancer", "detector") if active.get(stage)]
    primary = loaded[0] if loaded else None
    return {"pipeline": state.pipeline.snapshot().model_dump(mode="json"),
            "selected": _selected_models(state.settings), "active": active,
            "precision": {"label": precision_label(primary["provider"], primary["precision"]) if primary else None,
                          "stage": primary["stage"] if primary else None,
                          "stages": {item["stage"]: precision_label(item["provider"], item["precision"]) for item in loaded},
                          "policy": state.settings.precision_policy},
            "failures": active_models.failures()}


@router.post("/models/rescan")
def rescan_models(request: Request) -> dict[str, Any]:
    result = request.app.state.models.rescan_local()
    rows = request.app.state.models.public()
    result["models"] = len(rows)
    result["installed"] = sum(bool(item.get("installed")) for item in rows)
    return result


@router.get("/models/hub")
def model_hub(request: Request) -> dict[str, Any]:
    return request.app.state.model_hub.snapshot()


@router.post("/models/{model_id}/downloads", status_code=202)
def start_model_download(model_id: str, request: Request,
                         value: Annotated[ModelDownloadRequest | None, Body()] = None) -> dict[str, Any]:
    value = value or ModelDownloadRequest()
    return request.app.state.model_downloads.start(model_id, accepted_license=value.accepted_license,
                                                   commercial=value.commercial)


@router.get("/models/downloads/{download_id}")
def model_download_status(download_id: str, request: Request) -> dict[str, Any]:
    return request.app.state.model_downloads.get(download_id)


@router.get("/models/downloads")
def active_model_downloads(request: Request) -> list[dict[str, Any]]:
    return request.app.state.model_downloads.active()


@router.delete("/models/downloads/{download_id}")
def cancel_model_download(download_id: str, request: Request) -> dict[str, Any]:
    return request.app.state.model_downloads.cancel(download_id)


@router.get("/models/registry")
def model_registry(request: Request, view: str | None = None) -> Any:
    is_aliased = request.scope.get("original_api_path") == "/api/models/registry"
    if view == "unified" or (is_aliased and view != "legacy"):
        from roop_3d.model_registry import UNIFIED_MANIFEST_MAP, UnifiedModelRegistry
        report = UnifiedModelRegistry(request.app.state.root).get_full_readiness_report()
        for item in report:
            if item.get("model_id") in UNIFIED_MANIFEST_MAP:
                _clear_unverified_legacy_trt_claim(item)
        return report
    return request.app.state.models.registry.records()


def _clear_unverified_legacy_trt_claim(report: dict[str, Any]) -> None:
    """Keep legacy readiness fields, but don't trust an engine filename as an inference receipt."""
    if report.get("readiness_state") == "READY + TRT ENGINE":
        report["readiness_state"] = "DOWNLOADED (NEEDS TRT)"
        report["trt_engine_path"] = None
        report["engine_presence_verified"] = False
        report["actual_inference_available"] = False


@router.post("/models/download")
def batch_download_models(
    request: Request,
    value: Annotated[BatchModelDownloadRequest | None, Body()] = None,
) -> dict[str, Any]:
    req = value or BatchModelDownloadRequest()
    target_ids = list(dict.fromkeys(([req.model_id] if req.model_id else []) + req.models))
    if not target_ids:
        raise DomainError("MODEL_SELECTION_REQUIRED", "Select one or more registered model IDs; bulk download is disabled", 422)
    results = []
    for model_id in target_ids:
        result = request.app.state.models.ensure_available(model_id, accepted_license=req.accepted_license,
                                                           commercial=req.commercial)
        results.append({"model_id": model_id, "status": "VERIFIED_LOCAL", "downloaded": result.downloaded,
                        "resumed": result.resumed})

    return {"status": "ok", "results": results}


def _legacy_model_verification(root: Path, target_ids: list[str]) -> dict[str, Any]:
    """Read-only compatibility response for existing /models/verify callers."""
    from roop_3d.model_registry import UNIFIED_MANIFEST_MAP, UNIFIED_MODEL_ENTRIES, UnifiedModelRegistry
    legacy_registry = UnifiedModelRegistry(root)
    if not target_ids:
        target_ids = [entry.model_id for entry in UNIFIED_MODEL_ENTRIES]
    reports = []
    for model_id in target_ids:
        entry = UNIFIED_MANIFEST_MAP.get(model_id)
        if entry is None:
            reports.append({"model_id": model_id, "readiness_state": "MISSING", "error": "Unknown model ID"})
            continue
        report = legacy_registry.inspect_model_readiness(entry)
        # The old inspector infers TensorRT readiness from a filename only.
        # Do not claim executable acceleration without a verified build receipt.
        _clear_unverified_legacy_trt_claim(report)
        reports.append(report)
    return {"status": "ok", "models": reports}


@router.post("/models/verify")
def verify_models(
    request: Request,
    value: Annotated[ModelVerifyRequest | None, Body()] = None,
) -> dict[str, Any]:
    req = value or ModelVerifyRequest()
    target_ids = list(dict.fromkeys(([req.model_id] if req.model_id else []) + req.models))
    return _legacy_model_verification(request.app.state.root, target_ids)


@router.post("/models/hub/verify")
def verify_hub_models(
    request: Request,
    value: Annotated[ModelVerifyRequest | None, Body()] = None,
) -> dict[str, Any]:
    """Verify canonical hub IDs without changing the legacy verify endpoint contract."""
    req = value or ModelVerifyRequest()
    target_ids = list(dict.fromkeys(([req.model_id] if req.model_id else []) + req.models))
    if not target_ids:
        raise DomainError("MODEL_SELECTION_REQUIRED", "Select one or more registered model IDs to verify", 422)
    reports = []
    for model_id in target_ids:
        try:
            path = request.app.state.models.provider.resolve(model_id)
            evidence = request.app.state.model_hub.model(model_id)
            reports.append({"model_id": model_id, "status": evidence["status"], "bytes_verified": True,
                            "actual_inference_available": evidence["actual_inference_available"],
                            "filename": path.name})
        except DomainError as error:
            try:
                evidence = request.app.state.model_hub.model(model_id)
                state = evidence["status"]
                inference_available = evidence["actual_inference_available"]
            except DomainError as lookup_error:
                state = "NOT_INSTALLED" if lookup_error.code == "MODEL_NOT_FOUND" else "UNVERIFIED"
                inference_available = False
            reports.append({"model_id": model_id, "status": state, "bytes_verified": False,
                            "actual_inference_available": inference_available, "error_code": error.code})

    return {"status": "ok", "models": reports}


@router.post("/models/compile_trt")
def compile_trt_models(
    request: Request,
    value: Annotated[ModelCompileTrtRequest | None, Body()] = None,
) -> dict[str, Any]:
    state = request.app.state
    if state.worker is None:
        raise DomainError('TRT_BUILD_UNAVAILABLE', 'TensorRT compilation requires the rendering worker', 409)
    req = value or ModelCompileTrtRequest()
    ids = list(dict.fromkeys(([req.model_id] if req.model_id else []) + req.models))
    if not ids or len(ids) > 16:
        raise DomainError('MODEL_SELECTION_REQUIRED', 'Select between one and sixteen local models to compile', 422)
    from roop_3d.core.hardware import probe_hardware
    from roop_3d.engine.runtime import RuntimeConfig
    from roop_3d.server.engine_gate import EngineBusy
    from roop_3d.server.model_runtime import compile_model
    from roop_3d.server.app import release_media_models
    hardware = probe_hardware(state.settings)
    if not hardware.gpus or 'TensorrtExecutionProvider' not in hardware.advertised_providers or state.settings.hardware_profile == 'cpu':
        raise DomainError('TRT_BUILD_UNAVAILABLE', 'TensorRT compilation requires an available NVIDIA GPU and TensorRT execution provider', 409)
    if state.engine_gate.busy_reason():
        raise EngineBusy(state.engine_gate.busy_reason())
    config = RuntimeConfig(provider='TensorrtExecutionProvider', hardware_profile=hardware.resolved_hardware_profile,
        precision_policy=req.precision, reserve_mib=state.settings.memory_safety_reserve_mb)
    results = []
    with state.engine_gate.owner('TENSORRT_COMPILATION'):
        release_media_models(state.media, state.worker)
        try:
            for model_id in ids:
                state.pipeline.set_engine('COMPILING_TRT', label=f'{model_id} {req.precision}', model=model_id, precision=req.precision)
                results.append(compile_model(state.models, model_id, config))
        finally:
            state.pipeline.set_engine('IDLE')
    return {'status': 'compiled', 'results': results}


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


@router.get("/models/required_status", include_in_schema=False)
@router.get("/models/required-status")
def models_required_status(request: Request) -> dict[str, Any]:
    from roop_3d.core.model_downloader import verify_all_required_models
    return {
        "status": "ok",
        "models": verify_all_required_models(request.app.state.root),
    }


class DownloadMissingModelsRequest(Contract):
    models: list[str] | None = None


@router.post("/models/download_missing", include_in_schema=False)
@router.post("/models/download-missing")
def download_missing_models_endpoint(
    request: Request,
    value: Annotated[DownloadMissingModelsRequest | None, Body()] = None,
) -> dict[str, Any]:
    from roop_3d.core.model_downloader import REQUIRED_MODELS, download_model, find_model_path
    req_models = value.models if (value and value.models) else list(REQUIRED_MODELS.keys())
    results = []
    root = request.app.state.root
    dest_dir = root / "app/models"
    dest_dir.mkdir(parents=True, exist_ok=True)

    for mname in req_models:
        mspec = REQUIRED_MODELS.get(mname)
        if not mspec:
            continue
        existing = find_model_path(mspec, root)
        if existing is not None:
            results.append({"name": mspec.name, "status": "ALREADY_PRESENT", "path": str(existing)})
            continue
        try:
            path = download_model(mspec.name, destination_dir=dest_dir, app_state=request.app.state)
            results.append({"name": mspec.name, "status": "DOWNLOADED", "path": str(path)})
        except Exception as error:
            results.append({"name": mspec.name, "status": "FAILED", "error": str(error)})

    return {"status": "ok", "results": results}


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
    frame_index: int | None = Field(default=None, ge=0, le=10_000_000)   # converted to a timestamp, so any index of a long video works
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
    state.model_hub.engines_registry.validate(value.engine_id)
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
    if value.identity_lock_id:
        from roop_3d.identity.lock import LockEnvironment, LockStore
        LockStore(LockEnvironment(state.root, state.media.path, lambda: state.settings,
                                  state.media.detector.catalog)).verify_job(value)
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


@router.get("/jobs/{job_id}/checkpoint")
def job_checkpoint(job_id: str, request: Request) -> dict:
    """Stage 9 & 14: what a re-run of this job would resume from."""
    try:
        request.app.state.jobs.get(job_id)
    except Exception:
        pass
    import json
    candidates = [
        safe_path(request.app.state.root, f"jobs/{job_id}/checkpoint.json"),
        safe_path(request.app.state.root, f"roop_3d/.runtime/jobs/{job_id}/checkpoint.json"),
        safe_path(request.app.state.root, f"roop_3d/.runtime/temp/{job_id}/job_checkpoint.json"),
        safe_path(request.app.state.root, f"roop_3d/.runtime/stream/{job_id}/job_checkpoint.json"),
    ]
    target_file = next((p for p in candidates if p.is_file()), None)
    if not target_file:
        return {"available": False}
    try:
        data = json.loads(target_file.read_text("utf-8"))
    except (OSError, ValueError):
        return {"available": False}
    done = [s for s in data.get("segments", []) if s.get("status") == "COMPLETED"]
    completed_segments = len(done)
    completed_frames = data.get("completed_frame")
    if completed_frames is None:
        completed_frames = data.get("frame_progress", {}).get("completed_frames", 0)
    resume_frame = done[-1]["end_frame"] if done else data.get("resume_frame", (completed_frames + 1 if isinstance(completed_frames, int) and completed_frames >= 0 else 0))
    total_frames = data.get("total_frames") or data.get("frame_progress", {}).get("total_frames", 0)
    return {
        "available": True,
        "state": data.get("state"),
        "execution_provider": data.get("execution_provider"),
        "encoder": data.get("encoder"),
        "completed_segments": completed_segments,
        "resume_frame": resume_frame,
        "completed_frames": completed_frames,
        "total_frames": total_frames,
        "updated_at": data.get("updated_at"),
        "gop_aligned": data.get("gop_aligned", True),
    }


@router.get("/jobs/{job_id}/artifacts/{kind}")
def job_artifact(job_id: str, kind: str, request: Request):
    record = request.app.state.jobs.get(job_id)
    if kind not in {"video", "manifest", "sequence"}:
        raise DomainError("JOB_ARTIFACT_NOT_FOUND", "Unknown render artifact", 404)
    if record.state.value != "succeeded" or record.output is None:
        raise DomainError("JOB_OUTPUT_NOT_READY", "A verified completed render is required", 409)
    if kind == "video" and not record.output.baked_video:
        raise DomainError("JOB_VIDEO_NOT_RENDERED", "This render exported passes only", 409)
    output = owned(request.app.state.root, record.output.output_path)
    manifest = owned(request.app.state.root, record.output.manifest_path)
    if output.is_dir():
        # An image-sequence export: the directory's integrity hash is that of its sidecar (which lists every frame hash).
        sidecar = output / "sequence.json"
        if kind == "video":
            raise DomainError("JOB_OUTPUT_IS_SEQUENCE", "This render is an image sequence; fetch the 'sequence' artifact", 409)
        if not sidecar.is_file() or not manifest.is_file() or sha256_file(sidecar)[1] != record.output.output_sha256:
            raise DomainError("JOB_OUTPUT_CHANGED", "Published output is missing or its integrity changed", 409)
        path = sidecar if kind == "sequence" else manifest
        return FileResponse(path, filename=path.name, headers={"Cache-Control": "no-store"}, media_type="application/json")
    if kind == "sequence":
        raise DomainError("JOB_ARTIFACT_NOT_FOUND", "This render is not an image sequence", 404)
    if not output.is_file() or not manifest.is_file() or sha256_file(output)[1] != record.output.output_sha256:
        raise DomainError("JOB_OUTPUT_CHANGED", "Published output is missing or its integrity changed", 409)
    path = output if kind == "video" else manifest
    video_types = {".mp4": "video/mp4", ".mov": "video/quicktime", ".mkv": "video/x-matroska"}
    return FileResponse(path, filename=path.name, headers={"Cache-Control": "no-store"},
                        media_type=video_types.get(path.suffix.lower(), "application/octet-stream") if kind == "video" else "application/json")


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


class CheckCompatibilityRequest(Contract):
    force_remote: bool = False


@router.get("/system/diagnostics-report", include_in_schema=False)
@router.get("/system/diagnostics_report")
def diagnostics_report(request: Request) -> dict:
    from roop_3d.server.diagnostics import generate_14_section_report
    return generate_14_section_report(request.app.state.root, request.app.state)


@router.get("/system/environment-health", include_in_schema=False)
@router.get("/system/environment_health")
def environment_health(request: Request) -> dict:
    from roop_3d.server.diagnostics import get_environment_health_card
    return get_environment_health_card(request.app.state.root, request.app.state)


@router.post("/system/purge-vram", include_in_schema=False)
@router.post("/system/purge_vram")
def purge_vram_route(request: Request) -> dict:
    from roop_3d.server.diagnostics import purge_idle_vram
    return purge_idle_vram(request.app.state)


@router.post("/system/check-compatibility", include_in_schema=False)
@router.post("/system/check_compatibility")
def check_compatibility_route(request: Request, value: CheckCompatibilityRequest | None = None) -> dict:
    from roop_3d.server.diagnostics import check_offline_compatibility
    force = value.force_remote if value else False
    return check_offline_compatibility(request.app.state.settings, force_remote=force)


router.add_api_websocket_route("/ws/events", event_stream)
router.add_api_websocket_route("/ws/telemetry", event_stream)
