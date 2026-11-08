"""Stage 12 job pipeline: validated prepare, resumable frame processing, and mux."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import re
import time
from typing import Literal
import zipfile

import cv2
import numpy as np
from pydantic import Field

from roop_3d.async_pipeline import AsyncPipeline, Item, PinnedFramePool
from roop_3d.async_pipeline.config import async_pipeline_enabled, async_queue_depth, pinned_enabled
from roop_3d.async_pipeline.reference import ReferenceStages
from roop_3d.audit.isolation import safe_path, sha256_file
from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract, Settings
from roop_3d.core.storage import atomic_bytes, atomic_json, owned
from roop_3d.engine.runtime import RuntimeConfig
from roop_3d.engine.volumetric.session import ReferencePipeline
from roop_3d.engine.vram_governor import Admission, RuntimeProfile, is_cuda_oom
from roop_3d.faceset.faceset_reader import FaceSetReader
from roop_3d.faceset.schemas import FaceSetDocument
from roop_3d.jobs.schema import JobOutput, JobProgress, JobRecord, JobState
from roop_3d.media.schema import FaceInspection
from roop_3d.media.video_reader import VideoReader
from roop_3d.media.video_writer import encode_video
from roop_3d.models.catalog import ModelCatalog, render_dependencies
from roop_3d.compositor.pipeline import CompositeOptions
from roop_3d.compositor.restoration import GpenRestorer
from roop_3d.engine.edit_plan import FrameEditPlan
from roop_3d.engine.detection.track_policy import POLICY_VERSION, TrackPolicy


class PipelineCheckpoint(Contract):
    schema_version: Literal[1] = 1
    job_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    settings_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    faceset_sha256: dict[str, str] = Field(default_factory=dict)
    completed_frame: int = Field(default=-1, ge=-1, strict=True)
    total_frames: int | None = Field(default=None, ge=1, strict=True)
    runtime_profile: RuntimeProfile = "cpu"
    retry_profile: RuntimeProfile | None = None
    oom_retries: int = Field(default=0, ge=0, le=1, strict=True)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    input_hashes: dict[str, str] = Field(default_factory=dict)
    faceset_ids: tuple[str, ...] = Field(default_factory=tuple)
    pose_cache: dict[str, list[float]] = Field(default_factory=dict)
    keyframe_curves: dict = Field(default_factory=dict)
    manual_mask: str | None = Field(default=None)
    execution_provider: str = Field(default="CPUExecutionProvider")
    completed_segment: int = Field(default=-1, ge=-1, strict=True)
    total_segments: int | None = Field(default=None, ge=1, strict=True)
    # Exact request inputs, 3D pose/Stage 0 matrix settings and hardware state, so a project can be resumed after a restart.
    inputs: dict = Field(default_factory=dict)
    pose_settings: dict = Field(default_factory=dict)
    hardware_state: dict = Field(default_factory=dict)
    model_digests: dict[str, str] = Field(default_factory=dict)
    inspection_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    identity_assets: dict[str, str | None] = Field(default_factory=dict)


CHECKPOINT_NPZ_FORMAT = "roop3d.checkpoint.npz/1"


def checkpoint_npz_bytes(checkpoint: PipelineCheckpoint) -> bytes:
    """NPZ twin of checkpoint.json: the JSON manifest plus every pose-cache vector as an exact float64 array."""
    manifest = json.dumps({"format": CHECKPOINT_NPZ_FORMAT, "checkpoint": checkpoint.model_dump(mode="json")},
                          sort_keys=True, allow_nan=False).encode("utf-8")
    arrays = {"manifest": np.frombuffer(manifest, np.uint8)}
    for index, (key, values) in enumerate(sorted(checkpoint.pose_cache.items())):
        arrays[f"pose_cache_{index:05d}"] = np.asarray(values, np.float64)
    arrays["pose_cache_keys"] = np.frombuffer(json.dumps(sorted(checkpoint.pose_cache)).encode("utf-8"), np.uint8)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def read_checkpoint_npz(data: bytes) -> PipelineCheckpoint:
    """Load and validate an NPZ checkpoint (no pickles); raises ValueError when it is not a valid checkpoint."""
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            manifest = json.loads(archive["manifest"].tobytes().decode("utf-8"))
            keys = json.loads(archive["pose_cache_keys"].tobytes().decode("utf-8"))
            cache = {key: [float(v) for v in archive[f"pose_cache_{index:05d}"]] for index, key in enumerate(keys)}
    except (OSError, EOFError, KeyError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError("Invalid NPZ checkpoint") from error
    if manifest.get("format") != CHECKPOINT_NPZ_FORMAT:
        raise ValueError("Unsupported NPZ checkpoint format")
    value = PipelineCheckpoint.model_validate({**manifest["checkpoint"], "pose_cache": cache})
    return value


class PipelinePaused(Exception):
    pass


class PipelineCancelled(Exception):
    pass


class PipelineInterrupted(Exception):
    pass


class RecoverablePipelineOOM(Exception):
    """A completed-frame checkpoint is ready for the one CPU retry."""


@dataclass(frozen=True)
class PreparedPipeline:
    target: Path
    input_sha256: str
    total_frames: int | None
    sources_by_track: dict[int, FaceSetDocument]
    labels_to_ids: dict[str, int]
    faceset_ids: tuple[str, ...]
    checkpoint: PipelineCheckpoint
    settings: Settings
    edit_plan: FrameEditPlan | None = None
    # Identity lock (None when the job is not bound to a lock) and tracker options replayed from the inspection.
    identity_lock: object | None = None
    identity_guard: object | None = None
    tracking: Mapping | None = None


class MasterPipelineRunner:
    """Run the local reference pipeline without loading models during API boot.

    Completed BGR frames are held as PNG files under the job's owned runtime
    directory. That lets pause/OOM retries reuse completed output while the
    reference inference path is replayed from the beginning to restore tracker,
    LivePortrait, and temporal-compositor state.
    """

    CHECKPOINT_EVERY = 30
    MAX_FACESETS = 64

    def __init__(self, root: Path, catalog: ModelCatalog, media, settings_provider: Callable[[], Settings], *,
                 faceset_reader: FaceSetReader | None = None, pipeline_factory=ReferencePipeline,
                 writer=encode_video, checkpoint_every: int = CHECKPOINT_EVERY, hardware_probe=None):
        if checkpoint_every < 1:
            raise ValueError("checkpoint_every must be positive")
        self.root, self.catalog, self.media = root, catalog, media
        self.settings_provider = settings_provider
        self.faceset_reader = faceset_reader or FaceSetReader(catalog, cache_root=root / "roop_3d/.runtime/cache")
        self.pipeline_factory, self.writer = pipeline_factory, writer
        self.checkpoint_every = checkpoint_every
        self.hardware_probe = hardware_probe
        self.live_hub = None                    # set by the server: a LiveFrameHub that receives every finished frame
        self.last_pipeline = None               # the most recent AsyncPipeline, kept for post-run telemetry
        self.last_pipeline_snapshot = None

    @staticmethod
    def _request_digest(record: JobRecord) -> str:
        payload = json.dumps(record.request.identity_payload(), sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _checkpoint_relative(job_id: str) -> str:
        return f"roop_3d/.runtime/jobs/{job_id}/checkpoint.json"

    @staticmethod
    def _frame_relative(job_id: str, frame: int) -> str:
        return f"roop_3d/.runtime/jobs/{job_id}/frames/{frame:08d}.png"

    def retry_profile(self, job_id: str) -> RuntimeProfile | None:
        path = owned(self.root, self._checkpoint_relative(job_id))
        if not path.exists():
            return None
        try:
            return PipelineCheckpoint.model_validate_json(path.read_text("utf-8")).retry_profile
        except (OSError, ValueError):
            raise DomainError("JOB_CHECKPOINT_INVALID", "Saved job checkpoint is invalid and was preserved", 409) from None

    def _read_checkpoint(self, job_id: str, input_sha256: str, request_sha256: str,
                         total_frames: int | None, profile: RuntimeProfile) -> PipelineCheckpoint:
        path = owned(self.root, self._checkpoint_relative(job_id))
        if path.exists():
            try:
                checkpoint = PipelineCheckpoint.model_validate_json(path.read_text("utf-8"))
            except (OSError, ValueError):
                raise DomainError("JOB_CHECKPOINT_INVALID", "Saved job checkpoint is invalid and was preserved", 409) from None
            if (checkpoint.job_id != job_id or checkpoint.input_sha256 != input_sha256
                    or checkpoint.request_sha256 != request_sha256):
                raise DomainError("JOB_CHECKPOINT_MISMATCH", "Checkpoint does not match this job input", 409)
            if total_frames is not None and checkpoint.completed_frame >= total_frames:
                raise DomainError("JOB_CHECKPOINT_INVALID", "Checkpoint frame exceeds the probed input", 409)
        else:
            checkpoint = PipelineCheckpoint(job_id=job_id, input_sha256=input_sha256,
                                            request_sha256=request_sha256, total_frames=total_frames,
                                            runtime_profile=profile)
            self._write_checkpoint(checkpoint)
        if checkpoint.completed_frame >= 0:
            for frame in range(checkpoint.completed_frame + 1):
                path = owned(self.root, self._frame_relative(job_id, frame))
                if not path.is_file() or path.stat().st_size <= 0:
                    raise DomainError("JOB_CHECKPOINT_FRAMES_MISSING", "Checkpoint frame data is incomplete", 409)
        return checkpoint

    def _write_checkpoint(self, checkpoint: PipelineCheckpoint) -> None:
        atomic_json(self.root, self._checkpoint_relative(checkpoint.job_id),
                    checkpoint.model_dump(mode="json"))
        bin_rel = f"roop_3d/.runtime/jobs/{checkpoint.job_id}/checkpoint.bin"
        json_bytes = checkpoint.model_dump_json().encode("utf-8")
        binary_payload = bytearray(b"R3DCP001")
        binary_payload.extend(len(json_bytes).to_bytes(4, "big"))
        binary_payload.extend(json_bytes)
        if checkpoint.manual_mask:
            mask_bytes = checkpoint.manual_mask.encode("utf-8")
            binary_payload.extend(len(mask_bytes).to_bytes(4, "big"))
            binary_payload.extend(mask_bytes)
        atomic_bytes(self.root, bin_rel, bytes(binary_payload))
        atomic_bytes(self.root, f"roop_3d/.runtime/jobs/{checkpoint.job_id}/checkpoint.npz", checkpoint_npz_bytes(checkpoint))

    def _checkpoint(self, checkpoint: PipelineCheckpoint, *, completed_frame: int | None = None,
                    profile: RuntimeProfile | None = None, retry_profile: RuntimeProfile | None = None,
                    oom_retries: int | None = None, provider: str | None = None,
                    completed_segment: int | None = None, pose_cache: dict | None = None) -> PipelineCheckpoint:
        updates: dict = {
            "completed_frame": checkpoint.completed_frame if completed_frame is None else completed_frame,
            "runtime_profile": checkpoint.runtime_profile if profile is None else profile,
            "retry_profile": retry_profile if retry_profile is not None else checkpoint.retry_profile,
            "oom_retries": checkpoint.oom_retries if oom_retries is None else oom_retries,
            "updated_at": datetime.now(timezone.utc),
        }
        if provider is not None:
            updates["execution_provider"] = provider
        if completed_segment is not None:
            updates["completed_segment"] = completed_segment
        if pose_cache is not None:
            updates["pose_cache"] = pose_cache
        value = checkpoint.model_copy(update=updates)
        self._write_checkpoint(value)
        return value

    def _faceset_path(self, faceset_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", faceset_id):
            raise DomainError("FACESET_ID_INVALID", "FaceSet identifier is invalid", 422)
        candidates = (f"roop_3d/outputs/facesets/{faceset_id}.fsz",
                     f"roop_3d/.runtime/facesets/{faceset_id}.fsz")
        for relative in candidates:
            path = safe_path(self.root, relative)
            if path.is_file():
                return path
        raise DomainError("FACESET_ARCHIVE_NOT_FOUND", "Mapped FaceSet archive is unavailable in owned storage", 404)

    def lock_store(self, settings: Settings):
        from roop_3d.identity.lock import LockEnvironment, LockStore
        return LockStore(LockEnvironment(self.root, self.media.path, lambda: settings, self.catalog))

    def tracking_options(self, request) -> dict:
        """Tracker configuration the render must replay: exactly what the inspection was made with.

        Legacy inspections (no policy) keep the legacy tracker and its default threshold, so existing
        saved inspections, projects and previews behave exactly as before.
        """
        path = safe_path(self.root, f"roop_3d/.runtime/inspections/{request.inspection_id}.json")
        try:
            inspection = FaceInspection.model_validate_json(path.read_text("utf-8"))
        except (OSError, ValueError):
            raise DomainError("INSPECTION_INVALID", "Saved inspection is invalid", 422) from None
        if inspection.tracking_method != POLICY_VERSION:
            return {}
        policy = TrackPolicy()
        if inspection.tracking_policy_fingerprint != policy.fingerprint():
            raise DomainError("IDENTITY_POLICY_CHANGED", "The tracking policy changed since this inspection; inspect again", 409)
        return {"track_policy": policy, "detection_threshold": inspection.detection_threshold or 0.5}

    def identity_context(self, request, labels_to_ids: Mapping[str, int], settings: Settings):
        """(lock, guard) for a lock-bound job, re-verified against the files on disk right now."""
        if request.identity_lock_id is None:
            return None, None
        from roop_3d.identity.guard import IdentityGuard
        from roop_3d.identity.report import read_report
        lock = self.lock_store(settings).verify_job(request)
        report = read_report(self.root, lock.inspection_id, expected_sha256=lock.identity_report_sha256)
        guard = IdentityGuard.from_report(report, labels_to_ids, {item.track_id: item.faceset_id for item in lock.assignments})
        return lock, guard

    def _prepare_inputs(self, record: JobRecord, settings: Settings
                        ) -> tuple[Path, str, int | None, dict[int, FaceSetDocument], dict[str, int], list[str]]:
        request = record.request
        target = self.media.path(request.target_path)
        digest = sha256_file(target)[1]
        inspection_path = safe_path(self.root, f"roop_3d/.runtime/inspections/{request.inspection_id}.json")
        if not inspection_path.is_file():
            raise DomainError("INSPECTION_NOT_FOUND", "Saved inspection is unavailable", 422)
        try:
            inspection = FaceInspection.model_validate_json(inspection_path.read_text("utf-8"))
        except (OSError, ValueError):
            raise DomainError("INSPECTION_INVALID", "Saved inspection is invalid", 422) from None
        if inspection.inspection_id != request.inspection_id or inspection.target_sha256 != digest:
            raise DomainError("INSPECTION_TARGET_MISMATCH", "Inspection does not match the current target video", 409)
        probe = self.media.probe(request.target_path)
        total_frames = probe.declared_frame_count if probe.declared_frame_count and probe.declared_frame_count > 0 else None
        tracks = {item.track_id: item for item in inspection.tracks}
        track_ids: dict[str, int] = {}
        documents: dict[int, FaceSetDocument] = {}
        seen_sources: set[str] = set()
        if len(request.face_mappings) > self.MAX_FACESETS:
            raise DomainError("FACESET_MAPPING_LIMIT", "Job maps too many target tracks", 422)
        for mapping in request.face_mappings:
            track = tracks.get(mapping.track_id)
            if track is None:
                raise DomainError("TRACK_NOT_FOUND", "Mapped track is absent from saved inspection", 422)
            persistent_id = track.persistent_id
            if persistent_id is None:
                persistent_id = int(mapping.track_id.rsplit("_", 1)[1])
            if persistent_id in documents:
                raise DomainError("TRACK_MAPPING_AMBIGUOUS", "Inspection maps multiple labels to one tracker identity", 422)
            track_ids[mapping.track_id] = persistent_id
            if mapping.faceset_id not in seen_sources:
                path = self._faceset_path(mapping.faceset_id)
                document = self.faceset_reader.read(path, faceset_id=mapping.faceset_id)
                if document.faceset_id != mapping.faceset_id:
                    raise DomainError("FACESET_ID_MISMATCH", "Mapped FaceSet archive identity does not match its catalog ID", 422)
                documents[persistent_id] = document
                seen_sources.add(mapping.faceset_id)
            else:
                existing = next(value for value in documents.values() if value.faceset_id == mapping.faceset_id)
                documents[persistent_id] = existing
        if request.identity_blend is not None:
            from roop_3d.faceset.identity_blend import blend_documents
            recipe = request.identity_blend
            pair = [self.faceset_reader.read(self._faceset_path(item), faceset_id=item) for item in recipe.source_face_ids]
            blended = blend_documents(pair, recipe)
            documents = {track: blended for track in documents}
        if settings.execution_mode == "diffusion_vfx":
            raise DomainError("DIFFUSION_VFX_UNAVAILABLE",
                              "Stage 11 requires a verified local family loader; no loader is configured", 503)
        return target, digest, total_frames, documents, track_ids, [item.faceset_id for item in request.face_mappings]

    @staticmethod
    def _runtime_config(admission: Admission, settings: Settings) -> RuntimeConfig:
        provider = admission.provider
        if admission.profile != "cpu":
            if settings.preferred_provider == "TensorrtExecutionProvider":
                provider = settings.preferred_provider
            elif settings.preferred_provider == "auto":
                from roop_3d.core.hardware import probe_hardware, resolve_accelerator
                hardware = probe_hardware(settings)
                provider = resolve_accelerator("auto", hardware.advertised_providers, bool(hardware.gpus))
        return RuntimeConfig(provider=provider, hardware_profile=admission.profile,
                             reserve_mib=settings.memory_safety_reserve_mb)

    def prepare(self, record: JobRecord, settings: Settings | None = None) -> PreparedPipeline:
        """Validate media, probe timing, load read-only FaceSets and check resume data."""
        settings = settings or self.settings_provider()
        if record.request.export:
            from roop_3d.media.provenance import require_mode
            require_mode(record.request.export.finishing.provenance)
            if {'normal', 'depth'} & set(record.request.export.passes) and record.request.parameters_at(0).pose_mode != 'mesh_blend':
                raise DomainError('VFX_GEOMETRY_UNAVAILABLE', 'Normal/depth exports require mesh blend mode', 422)
            if record.request.export.passes and settings.video_timing_policy == 'cfr':
                raise DomainError('VFX_TIMING_POLICY', 'Pass exports require preserved source timing; conform VFR media before rendering for NLE interchange', 422)
        target, digest, total, sources, labels_to_ids, faceset_ids = self._prepare_inputs(record, settings)
        edit_plan = FrameEditPlan(self, record.request, sources)
        request_digest = self._request_digest(record)
        checkpoint = self._read_checkpoint(record.job_id, digest, request_digest, total,
                                           "cpu")
        self._require_disk_space(record, target, total, checkpoint)
        # Bind actual verified local dependencies, not registry presence. Missing optional models stay optional.
        models = render_dependencies(self.catalog)
        inspection_digest = sha256_file(owned(self.root,
            f"roop_3d/.runtime/inspections/{record.request.inspection_id}.json"))[1]
        if checkpoint.completed_frame >= 0 and not checkpoint.model_digests:
            raise DomainError("JOB_CHECKPOINT_MODELS_UNKNOWN",
                              "This older checkpoint has no model identity; queue a new render", 409)
        if checkpoint.model_digests and checkpoint.model_digests != models:
            raise DomainError("JOB_MODELS_CHANGED", "Model selection changed since the saved checkpoint", 409)
        if checkpoint.inspection_sha256 and checkpoint.inspection_sha256 != inspection_digest:
            raise DomainError("JOB_INSPECTION_CHANGED", "Target inspection changed since the saved checkpoint", 409)
        from roop_3d.jobs.project_store import settings_digest
        fingerprint = settings_digest(settings)
        source_fingerprints = {document.faceset_id: document.source_sha256 for document in sources.values()}
        source_fingerprints.update({document.faceset_id: document.source_sha256 for document in edit_plan.documents.values()})
        for document in (*sources.values(), *edit_plan.documents.values()):
            source_fingerprints.update(document.metadata.get('identity_curve_sources', {}))
        if checkpoint.faceset_sha256 and checkpoint.faceset_sha256 != source_fingerprints:
            raise DomainError("JOB_SOURCE_CHANGED", "A source FaceSet changed since the saved checkpoint", 409)
        if checkpoint.settings_sha256 is not None and checkpoint.settings_sha256 != fingerprint:
            raise DomainError("JOB_SETTINGS_CHANGED", "Restore the job's effective settings before resuming", 409)
        curves_dict = record.request.parameter_curves.model_dump(mode="json") if record.request.parameter_curves else {}
        input_hashes_dict = {"target": digest, "target_path": record.request.target_path, "inspection_id": record.request.inspection_id}
        input_hashes_dict.update(source_fingerprints)
        if record.request.identity_lock_id:
            input_hashes_dict["identity_lock_id"] = record.request.identity_lock_id
        checkpoint = checkpoint.model_copy(update={
            "settings_sha256": fingerprint,
            "faceset_sha256": source_fingerprints,
            "input_hashes": input_hashes_dict,
            "faceset_ids": tuple(faceset_ids),
            "keyframe_curves": curves_dict,
            "manual_mask": record.request.mask_base64,
            "execution_provider": "CPUExecutionProvider",
            "inputs": self._checkpoint_inputs(record),
            "pose_settings": self._pose_settings(record, {**sources, **edit_plan.documents}),
            "hardware_state": self._hardware_state(settings),
            "model_digests": models,
            "inspection_sha256": inspection_digest,
        })
        self._write_checkpoint(checkpoint)
        workdir = owned(self.root, f"roop_3d/.runtime/jobs/{record.job_id}/frames")
        workdir.mkdir(parents=True, exist_ok=True)
        numeric_ids = set(labels_to_ids.values())
        by_track = {track_id: sources[track_id] for track_id in numeric_ids}
        lock, guard = self.identity_context(record.request, labels_to_ids, settings)
        return PreparedPipeline(target, digest, total, by_track, labels_to_ids,
                                tuple(faceset_ids), checkpoint, settings, edit_plan,
                                lock, guard, self.tracking_options(record.request))

    def _require_disk_space(self, record: JobRecord, target: Path, total: int | None,
                            checkpoint: PipelineCheckpoint) -> None:
        """Stage 18: refuse a job the volume cannot hold *before* any frame is stored (resumes are discounted)."""
        from roop_3d.reliability import disk
        try:
            with self.media.open(record.request.target_path) as container:
                stream = container.streams.video[0]
                width, height = stream.width, stream.height
                frames = total or (int(float(stream.duration * stream.time_base) * float(stream.average_rate or 0))
                                   if stream.duration and stream.average_rate else 0)
        except (OSError, IndexError, TypeError, ValueError):
            return  # geometry unknown: the runtime ENOSPC handling still protects the job
        needed = disk.estimate(width, height, frames, target.stat().st_size,
                               remaining_frames=frames - (checkpoint.completed_frame + 1))
        disk.require(self.root, needed)

    @staticmethod
    def _checkpoint_inputs(record: JobRecord) -> dict:
        request = record.request
        return {"target_path": request.target_path, "inspection_id": request.inspection_id,
                "face_mappings": [item.model_dump(mode="json") for item in request.face_mappings]}

    @staticmethod
    def _pose_settings(record: JobRecord, documents: Mapping[int, FaceSetDocument]) -> dict:
        """Pose/TPS parameters plus the Stage 0 3x3 pose-matrix fingerprint of every mapped FaceSet."""
        parameters = record.request.workstation_parameters
        values = parameters.model_dump(mode="json") if parameters is not None else {}
        keep = ("pose_mode", "pose_strength", "pose_frontalization", "tps_expression_weight", "render_mode")
        matrices: dict[str, dict] = {}
        from roop_3d.faceset import pose_matrix
        for document in {item.faceset_id: item for item in documents.values()}.values():
            try:
                matrix = pose_matrix.preflight(document)
            except Exception:  # Matrix evidence is advisory; a checkpoint must still be written.
                matrix = None
            if matrix is not None:
                matrices[document.faceset_id] = {"fingerprint": matrix.fingerprint, "mode": matrix.mode,
                                                 "populated": matrix.populated, "filled": matrix.filled}
        return {**{key: values[key] for key in keep if key in values}, "pose_matrices": matrices}

    def _hardware_state(self, settings: Settings) -> dict:
        try:
            from roop_3d.core.hardware import probe_hardware
            hardware = self.hardware_probe(settings) if self.hardware_probe is not None else probe_hardware(settings)
        except Exception:
            return {"probe": "UNAVAILABLE"}
        return {"probe": hardware.gpu_probe_state, "gpus": [{"name": gpu.name, "total_vram_mib": gpu.total_vram_mib,
                                                               "free_vram_mib": gpu.free_vram_mib} for gpu in hardware.gpus],
                "advertised_providers": hardware.advertised_providers or [], "preferred_provider": settings.preferred_provider,
                "hardware_profile": settings.hardware_profile, "execution_threads": settings.execution_threads}

    @staticmethod
    def _control_signal(control: Callable[[], object]) -> type[Exception] | None:
        state = control()
        value = state.value if isinstance(state, JobState) else str(state or "")
        if value == JobState.pausing.value:
            return PipelinePaused
        if value == JobState.cancelling.value:
            return PipelineCancelled
        if value in {"shutdown", JobState.interrupted.value}:
            return PipelineInterrupted
        return None

    def _check_control(self, control: Callable[[], object], checkpoint: PipelineCheckpoint,
                       completed_frame: int) -> None:
        signal = self._control_signal(control)
        if signal is not None:
            self._checkpoint(checkpoint, completed_frame=completed_frame)
            raise signal()

    @staticmethod
    def _stage_timings(result, session, decode_ms: float, encode_ms: float, timing: dict | None = None) -> dict[str, float]:
        """Measured per-stage milliseconds for one frame; a stage the engine did not time is omitted.

        ``timing`` is the compositor's per-frame timing captured by the composite stage itself; reading
        ``session.compositor.last_timing`` here would report a later frame once the stages overlap."""
        def number(value) -> float | None:
            return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0 and value == value and value != float('inf') else None
        latency = getattr(result, 'latency', None)
        timing = timing if timing is not None else (getattr(getattr(session, 'compositor', None), 'last_timing', None) or {})
        found: dict[str, float | None] = {
            'decode': number(decode_ms), 'detection': number(getattr(latency, 'detection_ms', None)),
            'inswapper': number(getattr(latency, 'swap_ms', None)), 'masking': number(timing.get('masking')),
            'restorer': number(timing.get('restorer')), 'encode': number(encode_ms)}
        retarget, projection = number(getattr(latency, 'retarget_ms', None)), number(getattr(latency, 'projection_ms', None))
        if retarget is not None or projection is not None:
            found['pose_tps'] = (retarget or 0.0) + (projection or 0.0)
        return {key: round(value, 3) for key, value in found.items() if value is not None}

    def _save_frame(self, job_id: str, index: int, image: np.ndarray, expected_shape: tuple[int, ...]) -> None:
        if image.dtype != np.uint8 or image.shape != expected_shape:
            raise DomainError("PIPELINE_FRAME_INVALID", "Compositor must return upright uint8 BGR at source dimensions", 422)
        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise DomainError("PIPELINE_FRAME_ENCODE_FAILED", "Completed frame could not be checkpointed", 500)
        atomic_bytes(self.root, self._frame_relative(job_id, index), encoded.tobytes())

    def run(self, record: JobRecord, admission: Admission, *, control: Callable[[], object],
            mark_running: Callable[[], None], report_progress: Callable[[JobProgress], None]) -> JobOutput:
        prepared = self.prepare(record, self.settings_provider())
        return self.run_prepared(record, admission, prepared=prepared, control=control,
                                 mark_running=mark_running, report_progress=report_progress)

    def run_prepared(self, record: JobRecord, admission: Admission, *, prepared: PreparedPipeline,
                     control: Callable[[], object], mark_running: Callable[[], None],
                     report_progress: Callable[[JobProgress], None]) -> JobOutput:
        settings = prepared.settings
        target, input_sha256 = prepared.target, prepared.input_sha256
        total_frames = prepared.total_frames
        sources, labels_to_ids, faceset_ids = prepared.sources_by_track, prepared.labels_to_ids, prepared.faceset_ids
        checkpoint = prepared.checkpoint
        if checkpoint.retry_profile == "cpu" and admission.profile != "cpu":
            raise DomainError("JOB_RETRY_PROFILE_MISMATCH", "OOM retry must be admitted on the CPU profile", 409)
        checkpoint = self._checkpoint(checkpoint, profile=admission.profile, provider=admission.provider)
        workdir = owned(self.root, f"roop_3d/.runtime/jobs/{record.job_id}/frames")
        workdir.mkdir(parents=True, exist_ok=True)
        numeric_ids = set(labels_to_ids.values())
        source_by_track = sources
        base_source = next(iter(source_by_track.values()))
        runtime_config = self._runtime_config(admission, settings)
        session = self.pipeline_factory(self.catalog, runtime_config)
        select_swapper = getattr(session, "select_swapper", None)
        if callable(select_swapper):
            select_swapper(record.request.swapper_model_id or settings.swapper_model_id)
        mapped_seen: set[int] = set()
        started = time.monotonic()
        decoded_count = 0
        expected_shape: tuple[int, ...] | None = None
        last_completed = checkpoint.completed_frame
        capability: dict = {}
        restorer_sha256: str | None = None
        export = record.request.export
        hdr_policy = export.hdr_policy if export else 'tonemap_sdr'
        if hdr_policy != 'tonemap_sdr':  # raises HDR_INPUT_REJECTED for PQ/HLG input before any work is done
            self.media.working_color(record.request.target_path, hdr_policy)
        scene_cuts: list[int] = []
        swapped_ranges: list[list[int]] = []
        identity_report = None

        try:
            signal = self._control_signal(control)
            if signal is not None:
                self._checkpoint(checkpoint, completed_frame=last_completed)
                raise signal()
            session.prepare_session(base_source, track_ids=numeric_ids,
                                    sources_by_track=source_by_track,
                                    retarget='off' if record.request.parameters_at(0).uses_mesh_projection() else 'auto',
                                    **record.request.parameters_at(0).session_options(),
                                    **(prepared.tracking or {}))
            if getattr(session, 'compositor', None) is not None:
                session.compositor.track_labels = {number: label for label, number in labels_to_ids.items()}
            assets = session.capability().get("render_modes", {}).get("identity_assets", {})
            asset_keys = {key: value.get("build_key") if value.get("available") else None for key, value in assets.items()}
            if checkpoint.completed_frame >= 0 and checkpoint.identity_assets != asset_keys:
                raise DomainError("JOB_IDENTITY_ASSET_CHANGED", "3D identity geometry changed since the saved checkpoint", 409)
            checkpoint = checkpoint.model_copy(update={"identity_assets": asset_keys})
            self._write_checkpoint(checkpoint)
            if prepared.identity_guard is not None:
                from roop_3d.identity.guard import IdentityRunReport
                session.set_identity_guard(prepared.identity_guard)
                identity_report = IdentityRunReport(prepared.identity_lock, labels_to_ids)
            if hasattr(session, 'configure_temporal'):
                session.configure_temporal(settings)
            architecture = record.request.parameters_at(0).restorer_model
            if architecture == "auto":
                # Each family keeps its own engine: UltraMax / GFPGAN are never silently served by GPEN.
                architecture = settings.face_restoration_policy
            if architecture not in {"off", "auto", "gpen", "codeformer", "gfpgan", "ultramax"}:
                raise DomainError("RESTORER_UNAVAILABLE", "The selected restoration architecture has no verified local adapter", 503)
            animated_restoration = bool(record.request.parameter_curves and any(key.value > 0 for key in record.request.parameter_curves.tracks.get('restorer_opacity', ())))
            if architecture == 'ultramax':
                from roop_3d.compositor.restoration import UltraMaxPipelineRestorer
                session.compositor.restorer = UltraMaxPipelineRestorer(self.catalog, runtime_config)
            elif architecture == 'codeformer':
                from roop_3d.compositor.restoration import CodeFormerRestorer
                session.compositor.restorer = CodeFormerRestorer(self.catalog, runtime_config)
            elif architecture == 'gfpgan':
                from roop_3d.compositor.restoration import GfpganRestorer
                session.compositor.restorer = GfpganRestorer(self.catalog, runtime_config)
            elif architecture in ('auto', 'gpen') or animated_restoration or (record.request.workstation_parameters and record.request.workstation_parameters.composite_options().restoration_blend > 0):
                try:
                    gpen = next((item for item in self.catalog.records()
                                 if item["name"] == "gpen_bfr_512.onnx" and item["state"] == "VERIFIED_LOCAL"), None)
                except (KeyError, TypeError):
                    gpen = None
                if gpen is not None:
                    try:
                        self.catalog.verify(gpen)
                    except DomainError:
                        gpen = None
                    if gpen is not None:
                        session.compositor.restorer = GpenRestorer(self.catalog, runtime_config)
            mark_running()
            if checkpoint.completed_frame >= 0:
                # Show the saved position immediately; inference replays the earlier frames to rebuild tracker state.
                report_progress(JobProgress(frame=checkpoint.completed_frame + 1, total_frames=total_frames, stage="resume"))

            angle_scan = getattr(session, "scan_angle_trajectories", None)
            install_angle_scan = getattr(session, "set_angle_trajectories", None)
            if callable(angle_scan) and callable(install_angle_scan):
                media_info = self.media.probe(record.request.target_path)
                try:
                    scan_fps = float(media_info.fps) if media_info.fps else 30.0
                except (TypeError, ValueError):
                    scan_fps = 30.0

                def scan_progress(scanned: int) -> None:
                    self._check_control(control, checkpoint, last_completed)
                    if scanned % 30 == 0 or scanned == total_frames:
                        reported = min(scanned, total_frames) if total_frames is not None else scanned
                        report_progress(JobProgress(frame=reported, total_frames=total_frames, stage="angle_scan"))

                trajectories = angle_scan(target, numeric_ids, fps=scan_fps, hdr_policy=hdr_policy,
                    min_cutoff=settings.temporal_min_cutoff, beta=settings.temporal_beta,
                    derivative_cutoff=settings.temporal_d_cutoff, on_frame=scan_progress)
                install_angle_scan(trajectories)
                if total_frames is None:
                    report_progress(JobProgress(frame=max((len(items) for items in trajectories.values()), default=0),
                                                total_frames=None, stage="angle_scan"))

            reader = VideoReader(target, hardware="auto", hdr_policy=hdr_policy)
            frames = reader.frames()
            # Frames <= resume_after were saved by an earlier attempt: they are still inferred (tracker / temporal state)
            # but never written again.
            resume_after = checkpoint.completed_frame
            pipelined = hasattr(session, "track_frame") and async_pipeline_enabled()   # stubs run inline
            request = record.request

            # -- the four stages of one frame; identical whether they run on four threads or inline ------------------
            def save_passes(collector, result, index):
                collector.geometry(result.replacements, request.parameters_at(index).pose_strength)
                collector.save(self.root, record.job_id, index)

            frame_stages = ReferenceStages(
                session, request=request, edit_plan=prepared.edit_plan, export=export, identity_report=identity_report,
                lip_sync=settings.lip_sync_mode == "wav2lip", working_transfer=lambda: reader.working.transfer,
                resume_after=resume_after, scene_cuts=scene_cuts, swapped_ranges=swapped_ranges, save_passes=save_passes,
                write_frame=lambda index, image, shape: self._save_frame(record.job_id, index, image, shape))
            stages = frame_stages.stages()
            depth = async_queue_depth()
            pipeline = AsyncPipeline("render", None, queue_size=depth) if pipelined else None

            def decoded_items():
                pool = None
                try:
                    for decoded in frames:
                        image = decoded.image
                        if pool is None and pinned_enabled():
                            pool = PinnedFramePool(image.shape, depth + len(stages) + 2, cancel=pipeline.cancel_event)
                            pipeline.add_pool(pool)
                        if pool is not None and image.shape == pool.shape:
                            slot, buffer = pool.acquire()        # blocks when every buffer is in flight: bounded memory
                            np.copyto(buffer, image)
                            decoded = replace(decoded, image=buffer)
                            yield Item(index=decoded.index, payload=decoded, cleanups=[lambda number=slot: pool.release(number)])
                        else:
                            yield Item(index=decoded.index, payload=decoded)
                finally:
                    frames.close()

            def inline_items():
                try:
                    iterator = iter(frames)
                    while True:
                        began = time.perf_counter()
                        decoded = next(iterator, None)
                        if decoded is None:
                            return
                        item = Item(index=decoded.index, payload=decoded)
                        item.wall_ms["decode"] = (time.perf_counter() - began) * 1000
                        for name, stage in stages:
                            began = time.perf_counter()
                            item.payload = stage(item)
                            item.wall_ms[name] = (time.perf_counter() - began) * 1000
                        yield item
                finally:
                    frames.close()

            if pipelined:
                pipeline.set_source(decoded_items())
                for name, stage in stages:
                    pipeline.add_stage(name, stage, stream=name != "write")
                self.last_pipeline = pipeline
            hub = getattr(self, "live_hub", None)
            try:
                with (pipeline if pipelined else nullcontext()):
                    for item in (pipeline.results() if pipelined else inline_items()):
                        expected_shape = frame_stages.expected_shape
                        decoded, result = item.meta["decoded"], item.meta["result"]
                        walls = item.wall_ms
                        self._check_control(control, checkpoint, last_completed)
                        mapped_seen.update(entry.track_id for entry in result.replacements if entry.track_id in numeric_ids)
                        decoded_count += 1
                        if hub is not None:
                            hub.publish(item.payload, frame_index=decoded.index, timestamp_s=float(decoded.timestamp.seconds),
                                        stage_ms=(walls.get("track", 0.0), walls.get("swap", 0.0), walls.get("composite", 0.0)))
                        if item.meta.get("saved"):
                            last_completed = decoded.index
                            completed = decoded.index + 1
                            elapsed = max(time.monotonic() - started, 1e-6)
                            work_ms = sum(walls.get(name, 0.0) for name, _ in stages)
                            report_progress(JobProgress(frame=completed, total_frames=total_frames,
                                fps=completed / elapsed, stage="render", stage_latency_ms=work_ms,
                                stage_timings_ms=self._stage_timings(result, session, walls.get("decode", 0.0),
                                                                      walls.get("write", 0.0), item.meta.get("timing"))))
                            if completed % self.checkpoint_every == 0:
                                checkpoint = self._checkpoint(checkpoint, completed_frame=last_completed,
                                                              profile=admission.profile)
                        item.release()
                        self._check_control(control, checkpoint, last_completed)
            finally:
                frames.close()
                if pipelined:
                    self.last_pipeline_snapshot = pipeline.snapshot()
            if decoded_count == 0:
                raise DomainError("MEDIA_EMPTY", "Target video contains no decodable frames", 422)
            required_ids = numeric_ids if record.request.face_segments is None else {prepared.edit_plan.ids[item.track_id] for item in record.request.face_segments if item.role == 'swap'}
            missing = required_ids - mapped_seen
            if missing:
                if identity_report is not None and not identity_report.never_seen() and identity_report.never_verified():
                    raise DomainError("TRACK_IDENTITY_UNVERIFIED",
                                      "A locked track was seen but its identity was never verified; no face was replaced for it", 422)
                raise DomainError("TRACK_NOT_DETECTED", "One or more mapped tracks were not detected during rendering", 422)
            checkpoint = self._checkpoint(checkpoint, completed_frame=last_completed,
                                          profile=admission.profile)
            capability = session.capability()
            # Capture the actual activated restoration model before release clears its session.
            restorer = getattr(session.compositor, "restorer", None)
            model = getattr(restorer, "model", None) or getattr(getattr(restorer, "runner", None), "model", None)
            record_used = getattr(model, "record", None)
            if record_used:
                restorer_sha256 = record_used.get("sha256")
        except BaseException as error:
            if is_cuda_oom(error):
                if checkpoint.oom_retries == 0 and admission.profile != "cpu":
                    checkpoint = self._checkpoint(checkpoint, completed_frame=last_completed,
                                                  profile=admission.profile, retry_profile="cpu", oom_retries=1)
                    raise RecoverablePipelineOOM("CUDA allocation failed; checkpoint is ready for one CPU retry") from None
                raise DomainError("PIPELINE_OOM", "Pipeline exhausted the available memory profiles", 503) from None
            raise
        finally:
            session.release_session()

        self._check_control(control, checkpoint, last_completed)

        report_progress(JobProgress(frame=decoded_count, total_frames=total_frames, stage="mux"))

        def completed_frame(decoded):
            self._check_control(control, checkpoint, last_completed)
            path = owned(self.root, self._frame_relative(record.job_id, decoded.index))
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or expected_shape is None or image.shape != expected_shape:
                raise DomainError("JOB_CHECKPOINT_FRAME_INVALID", "Processed frame checkpoint failed validation", 409)
            return image

        if settings.video_timing_policy == "cfr":
            fps = self.media.probe(record.request.target_path).fps_rational
            try:
                rate = Fraction(fps) if fps else None
            except (ValueError, ZeroDivisionError):
                rate = None
            if rate is None or rate <= 0:
                raise DomainError("VIDEO_RATE_UNKNOWN", "CFR output requires a verified rational input frame rate", 422)
            timing, output_rate = "explicit_cfr", rate
        else:
            timing, output_rate = "preserve_source_timing", None

        model_hashes: dict[str, str] = {}
        try:
            detector_rec = self.catalog.detector()
            if detector_rec and "sha256" in detector_rec:
                model_hashes["detector"] = detector_rec["sha256"]
        except Exception:
            pass
        try:
            recognizer_rec = self.catalog.recognizer()
            if recognizer_rec and "sha256" in recognizer_rec:
                model_hashes["recognizer"] = recognizer_rec["sha256"]
        except Exception:
            pass
        if capability.get("identity_swapper", {}).get("sha256"):
            model_hashes["identity_swapper"] = capability["identity_swapper"]["sha256"]
        if restorer_sha256:
            model_hashes["restorer"] = restorer_sha256

        metadata = {
            "stage": "12",
            "job_id": record.job_id,
            "runtime_profile": admission.profile,
            "provider": admission.provider,
            "identity_transfer": capability.get("identity_swapper", {}).get("inference_state", "UNKNOWN"),
            "liveportrait": capability.get("portrait", {}).get("state", "UNKNOWN"),
            "stage08_uv": "UNSUPPORTED" if settings.execution_mode in {"volumetric", "hybrid"} else "NOT_REQUESTED",
            "pose_projection": capability.get('pose_projection', {}),
            "pose_mode": record.request.parameters_at(0).pose_mode,
            "stage10_lip_sync": capability.get("lip_sync", {}).get("state", "UNKNOWN"),
            "stage11_diffusion_vfx": "UNSUPPORTED" if settings.execution_mode in {"hybrid", "diffusion_vfx"} else "NOT_REQUESTED",
            "face_mappings": [{"track_id": label, "faceset_id": source_id}
                               for (label, source_id) in zip(labels_to_ids, faceset_ids)],
            "model_hashes": model_hashes,
            "request_sha256": self._request_digest(record),
            "parameters": record.request.parameters_at(0).model_dump(mode="json"),
            "source_faceset_sha256": checkpoint.faceset_sha256,
            "effective_settings": settings.model_dump(mode="json"),
            "dependency_model_digests": checkpoint.model_digests,
            "inspection_sha256": checkpoint.inspection_sha256,
            "identity_assets": capability.get("render_modes", {}).get("identity_assets", {}),
            "restoration": capability.get("full_frame_compositor", {}).get("restorer", {}),
        }
        if identity_report is not None:
            metadata['identity_lock'] = identity_report.manifest()
        if export:
            metadata['timeline'] = {'scene_cuts': scene_cuts, 'swapped_ranges': swapped_ranges}
        pass_result = None
        if export and export.passes:
            from roop_3d.media.vfx import publish_passes
            pass_result = publish_passes(self.root, record.job_id, target, export.passes, settings.output_root,
                start=Fraction(str(record.request.trim_start_seconds)) if record.request.trim_start_seconds is not None else None,
                end=Fraction(str(record.request.trim_end_seconds)) if record.request.trim_end_seconds is not None else None,
                check=lambda: self._check_control(control, checkpoint, last_completed), timeline=metadata['timeline'])
            metadata['vfx_manifest'] = pass_result['manifest_path']
            metadata['vfx_manifest_sha256'] = sha256_file(owned(self.root, pass_result['manifest_path']))[1]
            if not export.baked_video:
                first = export.passes[0]
                result = JobOutput(output_path=pass_result['paths'][first], manifest_path=pass_result['manifest_path'],
                    output_sha256=pass_result['metadata']['output_sha256'][first], frames=len(pass_result['metadata']['frames']),
                    audio_policy='drop', passes=pass_result['paths'], baked_video=False,
                    pass_manifest_path=pass_result['manifest_path'])
                if self._control_signal(control) is not None:
                    self.discard_output(result)
                    self._check_control(control, checkpoint, last_completed)
                return result
        try:
            stage15 = {}
            output_name = f"roop3d_{record.job_id}.mp4"
            if export and export.profile:
                from roop_3d.media.export_profiles import get_profile
                container = get_profile(export.profile).container
                output_name = f"roop3d_{record.job_id}" + ('' if container == 'sequence' else container)
                stage15 = {'profile': export.profile, 'sequence_format': export.sequence_format}
            if export and export.subtitles != 'copy':
                stage15['subtitles'] = export.subtitles
            if hdr_policy != 'tonemap_sdr':
                stage15['hdr_policy'] = hdr_policy
            manifest = self.writer(target, output_name, root=self.root,
                hardware="software" if admission.profile == "cpu" else "auto", timing=timing,
                rate=output_rate, audio_policy=settings.audio_policy, transform=completed_frame,
                output_dir=settings.output_root, manifest_metadata=metadata,
                start=Fraction(str(record.request.trim_start_seconds)) if record.request.trim_start_seconds is not None else None,
                end=Fraction(str(record.request.trim_end_seconds)) if record.request.trim_end_seconds is not None else None,
                before_publish=lambda: self._check_control(control, checkpoint, last_completed),
                **({'finishing': export.finishing} if export else {}), **stage15)
        except BaseException as error:
            if pass_result:
                from roop_3d.media.vfx import discard_sequences
                discard_sequences(self.root, pass_result['manifest_path'])
                for relative in (*pass_result['paths'].values(), pass_result['manifest_path']):
                    owned(self.root, relative).unlink(missing_ok=True)
            if is_cuda_oom(error):
                if checkpoint.oom_retries == 0 and admission.profile != "cpu":
                    self._checkpoint(checkpoint, completed_frame=last_completed, profile=admission.profile,
                                     retry_profile="cpu", oom_retries=1)
                    raise RecoverablePipelineOOM("CUDA allocation failed during mux; checkpoint is ready for one CPU retry") from None
                raise DomainError("PIPELINE_OOM", "Output encoding exhausted the available memory profiles", 503) from None
            raise
        result = JobOutput(output_path=manifest["output_path"], manifest_path=manifest["manifest_path"],
                           output_sha256=manifest["output_sha256"], frames=manifest["verified_frames"],
                           audio_policy=manifest["audio_policy"], passes=pass_result['paths'] if pass_result else {},
                           pass_manifest_path=pass_result['manifest_path'] if pass_result else None)
        signal = self._control_signal(control)
        if signal is not None:
            self.discard_output(result)
            raise signal()
        return result

    def discard_output(self, output: JobOutput) -> None:
        if output.pass_manifest_path:
            from roop_3d.media.vfx import discard_sequences
            discard_sequences(self.root, output.pass_manifest_path)
        artifacts = {output.output_path, output.manifest_path, *output.passes.values()}
        if output.pass_manifest_path:
            artifacts.add(output.pass_manifest_path)
        for relative in artifacts:
            path = owned(self.root, relative)
            if path.exists():
                if path.is_symlink():
                    raise DomainError("JOB_OUTPUT_INVALID", "Published job output is not an owned regular file", 409)
                if relative == output.output_path and path.is_dir():  # an image-sequence export is a directory
                    if any(item.is_symlink() or item.is_dir() for item in path.iterdir()):
                        raise DomainError("JOB_OUTPUT_INVALID", "Image sequence holds links or nested directories", 409)
                    for item in path.iterdir():
                        item.unlink()
                    path.rmdir()
                    continue
                if not path.is_file():
                    raise DomainError("JOB_OUTPUT_INVALID", "Published job output is not an owned regular file", 409)
                path.unlink()

    def cleanup(self, job_id: str) -> None:
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise ValueError("Invalid job identifier")
        relative = f"roop_3d/.runtime/jobs/{job_id}"
        directory = owned(self.root, relative)
        if not directory.exists():
            return
        if directory.is_symlink() or not directory.is_dir():
            raise DomainError("JOB_WORKDIR_INVALID", "Job work directory is not an owned directory", 409)
        for path in sorted(directory.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if path.is_symlink():
                raise DomainError("JOB_WORKDIR_INVALID", "Symlink found in job work directory", 409)
            relative_path = path.relative_to(self.root).as_posix()
            owned(self.root, relative_path)
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        directory.rmdir()
