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
from roop_3d.jobs.schema import JobRequest, JobOutput
from roop_3d.models.catalog import render_dependencies


def settings_digest(settings: Settings) -> str:
    return hashlib.sha256(json.dumps(settings.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ViewportPreferences(Contract):
    live_swap: bool = True
    view_mode: Literal["side", "compare"] = "side"
    compare_mode: Literal["wipe", "crossfade", "diff", "blend", "loupe"] = "wipe"
    show_landmarks: bool = True
    show_pose: bool = True
    show_overlays: bool = True
    grid_mode: Literal["off", "thirds", "crosshair", "safe"] = "off"
    zoom: float = Field(default=1, ge=1, le=4)


class ProjectPreview(Contract):
    timestamp: float = Field(default=0, ge=0, le=86400)
    selected_track: str = Field(default="", max_length=128)
    source_faceset_id: str = Field(default="", max_length=64)
    viewport: ViewportPreferences = Field(default_factory=ViewportPreferences)


class ProjectRender(Contract):
    job_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    state: str
    created_at: datetime
    output: JobOutput | None = None


class ProjectDraft(Contract):
    name: str = Field(min_length=1, max_length=128)
    request: JobRequest
    checkpoint_job_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    preview: ProjectPreview = Field(default_factory=ProjectPreview)
    revision: int | None = Field(default=None, ge=0)


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
    revision: int = Field(default=0, ge=0)
    preview: ProjectPreview = Field(default_factory=ProjectPreview)
    identity_assets: dict[str, str] = Field(default_factory=dict)
    output_history: tuple[ProjectRender, ...] = ()


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
        source_ids = {item.faceset_id for item in request.face_mappings} | {item.faceset_id for item in request.face_segments or () if item.faceset_id}
        if request.identity_blend:
            source_ids.update(request.identity_blend.source_face_ids)
        for source in sorted(source_ids):
            candidates = [owned(self.root, f"roop_3d/{base}/facesets/{source}.fsz") for base in ("outputs", ".runtime")]
            path = next((path for path in candidates if path.is_file()), None)
            if path is None:
                raise DomainError("FACESET_ARCHIVE_NOT_FOUND", "Mapped FaceSet archive is unavailable", 422)
            paths.append(path)
        return tuple(ProjectInput(path=path.relative_to(self.root).as_posix(), sha256=sha256_file(path)[1]) for path in paths)

    def _models(self) -> dict[str, str]:
        return render_dependencies(self.catalog)

    def _assets(self, inputs: tuple[ProjectInput, ...]) -> dict[str, str]:
        """Reference existing verified assets; saving never builds one or copies its arrays."""
        from roop_3d.reconstruction.identity_asset import IdentityAssetManifest, IdentityAssetStore
        base = owned(self.root, "roop_3d/.runtime/identity_assets")
        store = IdentityAssetStore(base)
        result = {}
        for item in inputs:
            if not item.path.endswith(".fsz"):
                continue
            faceset_id = Path(item.path).stem
            directory = owned(self.root, f"roop_3d/.runtime/identity_assets/{faceset_id}")
            if not directory.is_dir():
                continue
            for candidate in sorted(directory.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                path = owned(self.root, candidate.relative_to(self.root).as_posix())
                try:
                    manifest = IdentityAssetManifest.model_validate_json(path.read_text("utf-8"))
                    if (manifest.faceset_source_sha256 == item.sha256 and manifest.faceset_id == faceset_id
                            and store.load(faceset_id, manifest.build_key) is not None):
                        result[faceset_id] = manifest.build_key
                        break
                except (ValueError, OSError):
                    continue
        return result

    def history(self, project_id: str) -> tuple[ProjectRender, ...]:
        return tuple(ProjectRender(job_id=job.job_id, state=job.state.value, created_at=job.created_at,
                                   output=job.output if job.state.value == "succeeded" else None)
                     for job in self.jobs.list() if job.project_id == project_id)

    def save(self, draft: ProjectDraft, settings: Settings, *, project_id: str | None = None) -> ProjectManifest:
        project_id = project_id or uuid.uuid4().hex
        with exclusive(self.root, self.relative(project_id) + ".lock"):
            previous = self.load(project_id, verify=False) if owned(self.root, self.relative(project_id)).exists() else None
            if previous is not None and draft.revision != previous.revision:
                raise DomainError("PROJECT_REVISION_CONFLICT", "Project changed; reopen it before saving", 409)
            return self._save(draft, settings, project_id, previous)

    def _save(self, draft: ProjectDraft, settings: Settings, project_id: str,
              previous: ProjectManifest | None) -> ProjectManifest:
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
        manifest = ProjectManifest(project_id=project_id, name=draft.name,
            saved_at=datetime.now(timezone.utc), request=draft.request, inputs=inputs, model_digests=self._models(),
            effective_settings=settings, checkpoint_job_id=draft.checkpoint_job_id, last_completed_frame=frame,
            revision=previous.revision + 1 if previous else 0, preview=draft.preview,
            identity_assets=self._assets(inputs), output_history=self.history(project_id))
        if len(manifest.model_dump_json().encode("utf-8")) > 512 * 1024:
            raise DomainError("PROJECT_TOO_LARGE", "Project exceeds the 512 KiB manifest limit", 422)
        import re
        if re.search(r'(?i)"(api_key|secret|password|consent_token|private_key|token)"\s*:\s*"[^"]+"', manifest.model_dump_json()):
            raise DomainError("PROJECT_CREDENTIALS_REJECTED", "Project manifest must not contain credentials or secrets", 422)
        atomic_json(self.root, self.relative(manifest.project_id), manifest.model_dump(mode="json"))
        return manifest

    def validate_project(self, project_id: str) -> dict:
        item = self.load(project_id, verify=False)
        verified = True
        status = "Inputs Verified"
        reason = None
        missing_inputs: list[str] = []
        try:
            expected = self._inputs(item.request)
            if expected != item.inputs:
                verified = False
                status = "Missing Source Media"
                reason = "A target, inspection, or FaceSet hash changed"
        except DomainError as err:
            verified = False
            status = "Missing Source Media"
            reason = err.message
            missing_inputs.append(err.message)
        except Exception:
            verified = False
            status = "Missing Source Media"
            reason = "Source media or inspection file is missing"
            missing_inputs.append("Source file missing")

        if verified:
            for model_id, digest in item.model_digests.items():
                try:
                    record = self.catalog.get(model_id)
                    if record["sha256"] != digest:
                        verified = False
                        status = "Missing Source Media"
                        reason = f"Model {model_id} identity changed"
                        break
                    self.catalog.verify(record)
                except Exception:
                    verified = False
                    status = "Missing Source Media"
                    reason = f"Model {model_id} is missing"
                    break

        checkpoint_data = None
        if item.checkpoint_job_id:
            cp_path = owned(self.root, f"roop_3d/.runtime/jobs/{item.checkpoint_job_id}/checkpoint.json")
            bin_path = owned(self.root, f"roop_3d/.runtime/jobs/{item.checkpoint_job_id}/checkpoint.bin")
            if cp_path.is_file():
                try:
                    from roop_3d.engine.pipeline_runner import PipelineCheckpoint
                    cp = PipelineCheckpoint.model_validate_json(cp_path.read_text("utf-8"))
                    checkpoint_data = cp.model_dump(mode="json")
                    checkpoint_data["has_binary_checkpoint"] = bin_path.is_file()
                except Exception:
                    pass

        return {
            "project_id": item.project_id,
            "name": item.name,
            "saved_at": item.saved_at,
            "last_completed_frame": item.last_completed_frame,
            "inputs_verified": verified,
            "status": status,
            "verification_reason": reason,
            "missing_inputs": missing_inputs,
            "checkpoint_job_id": item.checkpoint_job_id,
            "checkpoint": checkpoint_data,
            "target_path": item.request.target_path,
            "face_mappings": [m.model_dump(mode="json") for m in item.request.face_mappings],
            "trim_start_seconds": item.request.trim_start_seconds,
            "trim_end_seconds": item.request.trim_end_seconds,
            "has_mask": bool(item.request.mask_base64),
            "has_curves": bool(item.request.parameter_curves),
        }

    def list(self) -> list[dict]:
        base = owned(self.root, "roop_3d/.runtime/projects")
        if not base.exists():
            return []
        result = []
        for path in sorted(base.glob("*.r3dp")):
            try:
                result.append(self.validate_project(path.stem))
            except Exception:
                result.append({
                    "project_id": path.stem,
                    "name": f"Unreadable project {path.stem[:8]}",
                    "saved_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc),
                    "last_completed_frame": -1,
                    "inputs_verified": False,
                    "status": "Missing Source Media",
                    "verification_reason": "Failed to load project manifest",
                    "missing_inputs": ["Corrupted or invalid project manifest"],
                    "checkpoint_job_id": None,
                    "checkpoint": None,
                    "target_path": None,
                    "face_mappings": [],
                    "trim_start_seconds": None,
                    "trim_end_seconds": None,
                    "has_mask": False,
                    "has_curves": False,
                })
        return result

    def load(self, project_id: str, *, verify: bool = True) -> ProjectManifest:
        path = owned(self.root, self.relative(project_id))
        if not path.is_file():
            raise DomainError("PROJECT_NOT_FOUND", "Project does not exist", 404)
        try:
            if path.stat().st_size > 512 * 1024:
                raise ValueError()
            raw_text = path.read_text("utf-8")
            import re
            if re.search(r'(?i)"(api_key|secret|password|consent_token|private_key|token)"\s*:\s*"[^"]+"', raw_text):
                raise DomainError("PROJECT_CREDENTIALS_REJECTED", "Project manifest must not contain credentials or secrets", 422)
            item = ProjectManifest.model_validate_json(raw_text)
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
        history = self.history(project_id)
        return item.model_copy(update={"output_history": history or item.output_history})

    def queue(self, project_id: str, settings: Settings, idempotency_key: str, validate=None):
        with exclusive(self.root, self.relative(project_id) + ".lock"):
            item = self.load(project_id)
            if settings != item.effective_settings:
                raise DomainError("PROJECT_SETTINGS_CHANGED", "Save or restore project settings before rendering", 409)
            return self.jobs.create(item.request, idempotency_key,
                (lambda: validate(item.request)) if validate else None,
                project_id=project_id, effective_settings=item.effective_settings)

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
