"""Versioned owned .r3dp manifests; raw media, embeddings and secrets stay outside."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Literal
import uuid

from pydantic import Field

from roop_3d.audit.isolation import sha256_file
from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract, Settings
from roop_3d.core.storage import atomic_json, exclusive, owned
from roop_3d.jobs.schema import JobRequest


def settings_digest(settings: Settings) -> str:
    return hashlib.sha256(json.dumps(settings.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ProjectDraft(Contract):
    name: str = Field(min_length=1, max_length=128)
    request: JobRequest
    checkpoint_job_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")


class ProjectInput(Contract):
    path: str = Field(max_length=1024)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class ProjectManifest(Contract):
    schema_version: Literal[1] = 1
    project_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    name: str = Field(min_length=1, max_length=128)
    saved_at: datetime
    request: JobRequest
    inputs: tuple[ProjectInput, ...] = Field(min_length=1, max_length=66)
    model_digests: dict[str, str]
    effective_settings: Settings
    checkpoint_job_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    last_completed_frame: int = Field(default=-1, ge=-1)


class ProjectStore:
    def __init__(self, root: Path, media, catalog, jobs):
        self.root, self.media, self.catalog, self.jobs = root, media, catalog, jobs

    @staticmethod
    def relative(project_id: str) -> str:
        if len(project_id) != 32 or any(c not in "0123456789abcdef" for c in project_id):
            raise DomainError("PROJECT_ID_INVALID", "Project ID is invalid", 422)
        return f"roop_3d/.runtime/projects/{project_id}.r3dp"

    def _inputs(self, request: JobRequest) -> tuple[ProjectInput, ...]:
        target = self.media.path(request.target_path)
        paths = [target]
        inspection = owned(self.root, f"roop_3d/.runtime/inspections/{request.inspection_id}.json")
        if not inspection.is_file():
            raise DomainError("INSPECTION_NOT_FOUND", "Inspect the target before saving a mapped project", 422)
        paths.append(inspection)
        for source in sorted({item.faceset_id for item in request.face_mappings}):
            candidates = [owned(self.root, f"roop_3d/{base}/facesets/{source}.fsz") for base in ("outputs", ".runtime")]
            path = next((path for path in candidates if path.is_file()), None)
            if path is None:
                raise DomainError("FACESET_ARCHIVE_NOT_FOUND", "Mapped FaceSet archive is unavailable", 422)
            paths.append(path)
        return tuple(ProjectInput(path=path.relative_to(self.root).as_posix(), sha256=sha256_file(path)[1]) for path in paths)

    def _models(self) -> dict[str, str]:
        names = {"inswapper_128.onnx", "2d106det.onnx", "liveportrait_landmark.onnx", "liveportrait_appearance_feature_extractor.onnx",
                 "liveportrait_motion_extractor.onnx", "liveportrait_warping_spade.onnx", "liveportrait_stitching.onnx",
                 "liveportrait_stitching_eye.onnx", "liveportrait_stitching_lip.onnx", "bisenet_resnet_34.onnx", "gpen_bfr_512.onnx"}
        selected = [self.catalog.detector(), self.catalog.recognizer()]
        seen = set()
        for record in self.catalog.records():
            if record["name"] in names and record["name"] not in seen:
                selected.append(record)
                seen.add(record["name"])
        result = {record["model_id"]: record["sha256"] for record in selected}
        for model_id in result:
            self.catalog.verify(self.catalog.get(model_id))
        return result

    def save(self, draft: ProjectDraft, settings: Settings) -> ProjectManifest:
        inputs = self._inputs(draft.request)
        frame = -1
        if draft.checkpoint_job_id:
            record = self.jobs.get(draft.checkpoint_job_id)
            if record.request != draft.request:
                raise DomainError("PROJECT_JOB_MISMATCH", "Checkpoint job uses different project inputs", 409)
            path = owned(self.root, f"roop_3d/.runtime/jobs/{record.job_id}/checkpoint.json")
            if not path.is_file():
                raise DomainError("PROJECT_CHECKPOINT_MISSING", "No resumable checkpoint exists for this job", 409)
            try:
                from roop_3d.engine.pipeline_runner import PipelineCheckpoint
                checkpoint = PipelineCheckpoint.model_validate_json(path.read_text("utf-8"))
                expected_request = hashlib.sha256(json.dumps(draft.request.identity_payload(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                if (checkpoint.settings_sha256 != settings_digest(settings) or checkpoint.job_id != record.job_id
                        or checkpoint.input_sha256 != inputs[0].sha256 or checkpoint.request_sha256 != expected_request):
                    raise ValueError()
                frame = checkpoint.completed_frame
            except (ValueError, OSError):
                raise DomainError("PROJECT_CHECKPOINT_INVALID", "Checkpoint settings do not match this project", 409) from None
        manifest = ProjectManifest(project_id=uuid.uuid4().hex, name=draft.name,
            saved_at=datetime.now(timezone.utc), request=draft.request, inputs=inputs, model_digests=self._models(),
            effective_settings=settings, checkpoint_job_id=draft.checkpoint_job_id, last_completed_frame=frame)
        atomic_json(self.root, self.relative(manifest.project_id), manifest.model_dump(mode="json"))
        return manifest

    def list(self) -> list[dict]:
        base = owned(self.root, "roop_3d/.runtime/projects")
        if not base.exists():
            return []
        result = []
        for path in sorted(base.glob("*.r3dp")):
            item = self.load(path.stem, verify=False)
            result.append({"project_id": item.project_id, "name": item.name, "saved_at": item.saved_at,
                           "last_completed_frame": item.last_completed_frame})
        return result

    def load(self, project_id: str, *, verify: bool = True) -> ProjectManifest:
        path = owned(self.root, self.relative(project_id))
        if not path.is_file():
            raise DomainError("PROJECT_NOT_FOUND", "Project does not exist", 404)
        try:
            if path.stat().st_size > 512 * 1024:
                raise ValueError()
            item = ProjectManifest.model_validate_json(path.read_text("utf-8"))
            if item.project_id != project_id:
                raise ValueError()
        except (ValueError, OSError):
            raise DomainError("PROJECT_INVALID", "Project manifest is invalid and was preserved", 422) from None
        if verify:
            expected = self._inputs(item.request)
            if expected != item.inputs:
                raise DomainError("PROJECT_INPUT_CHANGED", "A target, inspection, or FaceSet hash changed; checkpoint continuation is blocked", 409)
            for model_id, digest in item.model_digests.items():
                record = self.catalog.get(model_id)
                if record["sha256"] != digest:
                    raise DomainError("PROJECT_MODEL_CHANGED", "A project model identity changed", 409)
                self.catalog.verify(record)
        return item

    def resume(self, project_id: str, settings: Settings):
        with exclusive(self.root, self.relative(project_id) + ".lock"):
            item = self.load(project_id)
            if settings != item.effective_settings:
                raise DomainError("PROJECT_SETTINGS_CHANGED", "Restore the saved effective settings before continuing", 409)
            if not item.checkpoint_job_id:
                raise DomainError("PROJECT_CHECKPOINT_MISSING", "Project has no saved job checkpoint; queue it as a new job", 409)
            record = self.jobs.get(item.checkpoint_job_id)
            if record.request != item.request:
                raise DomainError("PROJECT_JOB_MISMATCH", "Saved checkpoint belongs to another request", 409)
            from roop_3d.engine.pipeline_runner import PipelineCheckpoint
            path = owned(self.root, f"roop_3d/.runtime/jobs/{record.job_id}/checkpoint.json")
            try:
                checkpoint = PipelineCheckpoint.model_validate_json(path.read_text("utf-8"))
                request_digest = hashlib.sha256(json.dumps(item.request.identity_payload(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                if (checkpoint.job_id != record.job_id or checkpoint.completed_frame < item.last_completed_frame
                        or checkpoint.input_sha256 != item.inputs[0].sha256 or checkpoint.request_sha256 != request_digest
                        or checkpoint.settings_sha256 != settings_digest(settings)):
                    raise ValueError()
            except (OSError, ValueError):
                raise DomainError("PROJECT_CHECKPOINT_INVALID", "Checkpoint identity or completed frame is invalid", 409) from None
            for index in range(checkpoint.completed_frame + 1):
                frame = owned(self.root, f"roop_3d/.runtime/jobs/{record.job_id}/frames/{index:08d}.png")
                if not frame.is_file() or frame.stat().st_size == 0:
                    raise DomainError("PROJECT_CHECKPOINT_FRAMES_MISSING", "Completed checkpoint frame data is missing", 409)
            return self.jobs.control(record.job_id, "resume")
