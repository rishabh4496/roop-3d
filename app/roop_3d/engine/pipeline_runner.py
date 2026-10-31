"""Stage 12 job pipeline: validated prepare, resumable frame processing, and mux."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Literal

import cv2
import numpy as np
from pydantic import Field

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
from roop_3d.models.catalog import ModelCatalog
from roop_3d.compositor.pipeline import CompositeOptions
from roop_3d.compositor.restoration import GpenRestorer


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
                 writer=encode_video, checkpoint_every: int = CHECKPOINT_EVERY):
        if checkpoint_every < 1:
            raise ValueError("checkpoint_every must be positive")
        self.root, self.catalog, self.media = root, catalog, media
        self.settings_provider = settings_provider
        self.faceset_reader = faceset_reader or FaceSetReader(catalog, cache_root=root / "roop_3d/.runtime/cache")
        self.pipeline_factory, self.writer = pipeline_factory, writer
        self.checkpoint_every = checkpoint_every

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

    def _checkpoint(self, checkpoint: PipelineCheckpoint, *, completed_frame: int | None = None,
                    profile: RuntimeProfile | None = None, retry_profile: RuntimeProfile | None = None,
                    oom_retries: int | None = None) -> PipelineCheckpoint:
        value = checkpoint.model_copy(update={
            "completed_frame": checkpoint.completed_frame if completed_frame is None else completed_frame,
            "runtime_profile": checkpoint.runtime_profile if profile is None else profile,
            "retry_profile": retry_profile if retry_profile is not None else checkpoint.retry_profile,
            "oom_retries": checkpoint.oom_retries if oom_retries is None else oom_retries,
            "updated_at": datetime.now(timezone.utc),
        })
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
        if settings.execution_mode == "diffusion_vfx":
            raise DomainError("DIFFUSION_VFX_UNAVAILABLE",
                              "Stage 11 requires a verified local family loader; no loader is configured", 503)
        return target, digest, total_frames, documents, track_ids, [item.faceset_id for item in request.face_mappings]

    @staticmethod
    def _runtime_config(admission: Admission, settings: Settings) -> RuntimeConfig:
        return RuntimeConfig(provider=admission.provider, hardware_profile=admission.profile,
                             reserve_mib=settings.memory_safety_reserve_mb)

    def prepare(self, record: JobRecord, settings: Settings | None = None) -> PreparedPipeline:
        """Validate media, probe timing, load read-only FaceSets and check resume data."""
        settings = settings or self.settings_provider()
        target, digest, total, sources, labels_to_ids, faceset_ids = self._prepare_inputs(record, settings)
        request_digest = self._request_digest(record)
        checkpoint = self._read_checkpoint(record.job_id, digest, request_digest, total,
                                           "cpu")
        from roop_3d.jobs.project_store import settings_digest
        fingerprint = settings_digest(settings)
        source_fingerprints = {document.faceset_id: document.source_sha256 for document in sources.values()}
        if checkpoint.faceset_sha256 and checkpoint.faceset_sha256 != source_fingerprints:
            raise DomainError("JOB_SOURCE_CHANGED", "A source FaceSet changed since the saved checkpoint", 409)
        if checkpoint.settings_sha256 is not None and checkpoint.settings_sha256 != fingerprint:
            raise DomainError("JOB_SETTINGS_CHANGED", "Restore the job's effective settings before resuming", 409)
        checkpoint = checkpoint.model_copy(update={"settings_sha256": fingerprint, "faceset_sha256": source_fingerprints})
        self._write_checkpoint(checkpoint)
        workdir = owned(self.root, f"roop_3d/.runtime/jobs/{record.job_id}/frames")
        workdir.mkdir(parents=True, exist_ok=True)
        numeric_ids = set(labels_to_ids.values())
        by_track = {track_id: sources[track_id] for track_id in numeric_ids}
        return PreparedPipeline(target, digest, total, by_track, labels_to_ids,
                                tuple(faceset_ids), checkpoint, settings)

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
        checkpoint = self._checkpoint(checkpoint, profile=admission.profile)
        workdir = owned(self.root, f"roop_3d/.runtime/jobs/{record.job_id}/frames")
        workdir.mkdir(parents=True, exist_ok=True)
        numeric_ids = set(labels_to_ids.values())
        source_by_track = sources
        base_source = next(iter(source_by_track.values()))
        runtime_config = self._runtime_config(admission, settings)
        session = self.pipeline_factory(self.catalog, runtime_config)
        mapped_seen: set[int] = set()
        started = time.monotonic()
        decoded_count = 0
        expected_shape: tuple[int, ...] | None = None
        last_completed = checkpoint.completed_frame
        capability: dict = {}

        try:
            signal = self._control_signal(control)
            if signal is not None:
                self._checkpoint(checkpoint, completed_frame=last_completed)
                raise signal()
            session.prepare_session(base_source, track_ids=numeric_ids,
                                    sources_by_track=source_by_track, retarget="auto")
            if settings.face_restoration_policy in {"gfpgan", "codeformer"}:
                raise DomainError("RESTORER_UNAVAILABLE", "The selected restoration architecture has no verified local adapter", 503)
            if settings.face_restoration_policy == "auto" or (record.request.workstation_parameters and record.request.workstation_parameters.composite_options().restoration_blend > 0):
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

            reader = VideoReader(target, hardware="auto")
            frames = reader.frames()
            try:
                for decoded in frames:
                    self._check_control(control, checkpoint, last_completed)
                    expected_shape = decoded.image.shape
                    frame_started = time.perf_counter()
                    parameters = record.request.workstation_parameters
                    result = (session.infer_frame(decoded.image, strength=parameters.pose_strength)
                              if parameters else session.infer_frame(decoded.image))
                    output = session.composite_frame(
                        decoded.image, result,
                        options=parameters.composite_options() if parameters else CompositeOptions(restoration_blend=(0.35 if session.compositor.restorer else 0.0)),
                        lip_sync_enabled=settings.lip_sync_mode == "wav2lip")
                    mapped_seen.update(item.track_id for item in result.replacements if item.track_id in numeric_ids)
                    decoded_count += 1
                    if decoded.index > checkpoint.completed_frame:
                        self._save_frame(record.job_id, decoded.index, output, expected_shape)
                        last_completed = decoded.index
                        completed = decoded.index + 1
                        elapsed = max(time.monotonic() - started, 1e-6)
                        report_progress(JobProgress(frame=completed, total_frames=total_frames,
                            fps=completed / elapsed, stage="render", stage_latency_ms=(time.perf_counter() - frame_started) * 1000))
                        if completed % self.checkpoint_every == 0:
                            checkpoint = self._checkpoint(checkpoint, completed_frame=last_completed,
                                                          profile=admission.profile)
                    self._check_control(control, checkpoint, last_completed)
            finally:
                frames.close()
            if decoded_count == 0:
                raise DomainError("MEDIA_EMPTY", "Target video contains no decodable frames", 422)
            missing = numeric_ids - mapped_seen
            if missing:
                raise DomainError("TRACK_NOT_DETECTED", "One or more mapped tracks were not detected during rendering", 422)
            checkpoint = self._checkpoint(checkpoint, completed_frame=last_completed,
                                          profile=admission.profile)
            capability = session.capability()
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
        if getattr(session, "compositor", None) and getattr(session.compositor, "restorer", None):
            res_rec = getattr(session.compositor.restorer, "record", None)
            if res_rec and "sha256" in res_rec:
                model_hashes["restorer"] = res_rec["sha256"]

        metadata = {
            "stage": "12",
            "job_id": record.job_id,
            "runtime_profile": admission.profile,
            "provider": admission.provider,
            "identity_transfer": capability.get("identity_swapper", {}).get("inference_state", "UNKNOWN"),
            "liveportrait": capability.get("portrait", {}).get("state", "UNKNOWN"),
            "stage08_uv": "UNSUPPORTED" if settings.execution_mode in {"volumetric", "hybrid"} else "NOT_REQUESTED",
            "stage10_lip_sync": capability.get("lip_sync", {}).get("state", "UNKNOWN"),
            "stage11_diffusion_vfx": "UNSUPPORTED" if settings.execution_mode in {"hybrid", "diffusion_vfx"} else "NOT_REQUESTED",
            "face_mappings": [{"track_id": label, "faceset_id": source_id}
                               for (label, source_id) in zip(labels_to_ids, faceset_ids)],
            "model_hashes": model_hashes,
        }
        try:
            manifest = self.writer(target, f"roop3d_{record.job_id}.mp4", root=self.root,
                hardware="software" if admission.profile == "cpu" else "auto", timing=timing,
                rate=output_rate, audio_policy=settings.audio_policy, transform=completed_frame,
                output_dir=settings.output_root, manifest_metadata=metadata,
                start=Fraction(str(record.request.trim_start_seconds)) if record.request.trim_start_seconds is not None else None,
                end=Fraction(str(record.request.trim_end_seconds)) if record.request.trim_end_seconds is not None else None,
                before_publish=lambda: self._check_control(control, checkpoint, last_completed))
        except BaseException as error:
            if is_cuda_oom(error):
                if checkpoint.oom_retries == 0 and admission.profile != "cpu":
                    self._checkpoint(checkpoint, completed_frame=last_completed, profile=admission.profile,
                                     retry_profile="cpu", oom_retries=1)
                    raise RecoverablePipelineOOM("CUDA allocation failed during mux; checkpoint is ready for one CPU retry") from None
                raise DomainError("PIPELINE_OOM", "Output encoding exhausted the available memory profiles", 503) from None
            raise
        result = JobOutput(output_path=manifest["output_path"], manifest_path=manifest["manifest_path"],
                           output_sha256=manifest["output_sha256"], frames=manifest["verified_frames"],
                           audio_policy=manifest["audio_policy"])
        signal = self._control_signal(control)
        if signal is not None:
            self.discard_output(result)
            raise signal()
        return result

    def discard_output(self, output: JobOutput) -> None:
        for relative in (output.output_path, output.manifest_path):
            path = owned(self.root, relative)
            if path.exists():
                if path.is_symlink() or not path.is_file():
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
