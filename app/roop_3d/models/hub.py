"""Read-only presentation of registered models and compatible engine adapters."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from roop_3d.core.errors import DomainError
from roop_3d.models.catalog import ModelCatalog
from roop_3d.models.governance import LIFECYCLE_STATES
from roop_3d.models.registry import ModelRecord, authorize

MODEL_STATES = frozenset({"AVAILABLE", "NOT_INSTALLED", "LICENSE_RESTRICTED", "INCOMPATIBLE",
                          "UNVERIFIED", "BROKEN", "QUARANTINED", "READY_FOR_INFERENCE"})
DOWNLOAD_BLOCKERS = {
    "DOWNLOAD_EVIDENCE_REQUIRED": "No public source with identical bytes (SHA-256 and size) is registered. This file is a local or "
                                  "modified build, so it can be used from disk but cannot be downloaded.",
    "LICENSE_UNVERIFIED": "The license for these weights is not reviewed, and the source does not publish one.",
    "MODEL_BLOCKED": "This model is blocked by its license.",
    "DOWNLOAD_FORMAT_BLOCKED": "Pickle-based formats (.pt/.pth) are not downloaded because loading them can execute code.",
}


class ModelHubService:
    """Build status from the canonical registry and real local hash checks.

    This service never downloads or loads a model. Hash checks happen only when a
    user opens the hub, and inference availability is only exposed for adapters
    already wired into the reference pipeline.
    """

    def __init__(self, catalog: ModelCatalog):
        from roop_3d.engine.registry import EngineRegistry
        self.engines_registry = EngineRegistry()
        self.catalog = catalog
        self._core: dict[str, tuple[ModelRecord, str]] = {}
        self._core_cache: dict[str, tuple[str, bool, str]] = {}
        self._index_core_models()

    def _index_core_models(self) -> None:
        try:
            for name, summary in (("detector", self.catalog.detector()),
                                  ("recognizer", self.catalog.recognizer())):
                self._core[name] = (self.catalog.registry.get(summary["model_id"]),
                                    "SCRFD face detection" if name == "detector" else "ArcFace identity embedding")
        except DomainError:
            pass

    def _is_core(self, record: ModelRecord) -> bool:
        return any(item.model_id == record.model_id for item, _ in self._core.values())

    def _core_status(self, record: ModelRecord) -> tuple[str, bool, str]:
        if record.model_id in self._core_cache:
            return self._core_cache[record.model_id]
        try:
            path = self.catalog.resolve_for_inference(record.model_id)
            identity = self.catalog.artifact_identity(path)
            result = ("AVAILABLE", False,
                      f"Input bytes resolved at {path}; trust={identity['trust_basis']}. "
                      "No successful provider/session forward is established by registry or test-source presence.")
        except DomainError as error:
            state = ("BROKEN" if error.code == "MODEL_DRIFT" else
                     "NOT_INSTALLED" if error.code == "MODEL_NOT_INSTALLED" else
                     "LICENSE_RESTRICTED" if error.code in {"LICENSE_UNVERIFIED", "MODEL_BLOCKED"} else "UNVERIFIED")
            result = (state, False, f"{error.code}: {error.message}")
        self._core_cache[record.model_id] = result
        return result

    def _local_artifact_state(self, record: ModelRecord) -> tuple[str, str | None]:
        """Resolve through the canonical provider, including configured read-only roots."""
        try:
            return "AVAILABLE", self.catalog.provider.resolve(record).name
        except DomainError as error:
            if error.code == "MODEL_NOT_INSTALLED":
                return "NOT_INSTALLED", None
            if error.code in {"MODEL_DRIFT", "MODEL_IDENTITY_UNKNOWN"}:
                return "BROKEN", None
            if error.code == "MODEL_QUARANTINED":
                return "QUARANTINED", None
            return "UNVERIFIED", None

    def _entry(self, record: ModelRecord) -> dict[str, Any]:
        blocker: str | None = None
        try:
            authorize(record, action="download", accepted_license=True)
            download_eligible = True
        except DomainError as error:
            download_eligible = False
            # Report the most fundamental reason: no identical public bytes beats an unreviewed license.
            blocker = ("DOWNLOAD_EVIDENCE_REQUIRED" if not record.upstream_url and record.status != "blocked"
                       else error.code)
        if download_eligible and record.serialization_format.upper() in {"PICKLE", "PYTORCH", "TORCH", "PT", "PTH", "CKPT", "PKL", "BIN"}:
            download_eligible, blocker = False, "DOWNLOAD_FORMAT_BLOCKED"
        code_path = self._is_core(record)
        artifact_installed: bool | None = None
        integrity_verified: bool | None = None
        compatibility = "NO_REGISTERED_ADAPTER"
        adapter_spec = None      # the 2D swapper adapters were removed: no artifact has a registered face-swap adapter
        inference_note = "No verified inference path for this model in Roop 3D."
        if record.status == "blocked":
            state, inference, evidence = "LICENSE_RESTRICTED", False, record.license_notice
        elif record.status == "candidate":
            state, inference, evidence = "UNVERIFIED", False, record.evidence
        elif code_path:
            state, inference, evidence = self._core_status(record)
            compatibility = "REFERENCE_PIPELINE"
            artifact_installed = state not in {"NOT_INSTALLED", "FAILED"}
            integrity_verified = artifact_installed and state not in {"BROKEN", "QUARANTINED"}
        elif record.model_family in {"MiniMax H3 Ref2VA LoRA", "Qwen-Image-2.1 face-swap LoRA"}:
            state, filename = self._local_artifact_state(record)
            artifact_installed, integrity_verified = state == "AVAILABLE", state == "AVAILABLE"
            if state == "AVAILABLE":
                evidence = (f"Adapter bytes are hash-verified at {filename}; its workflow remains unavailable to production jobs.")
            elif state == "NOT_INSTALLED":
                evidence = "No hash-matching adapter file was found in the download cache or configured local model paths."
            elif state == "BROKEN":
                evidence = "A local adapter file failed its registered size or SHA-256 check."
            elif state == "QUARANTINED":
                evidence = "A corrupt managed-cache artifact was moved to quarantine."
            else:
                evidence = "The configured local model paths could not be fully verified."
            inference = False
            compatibility = "NOT_INTEGRATED"
        elif record.status == "verified_remote" or adapter_spec is not None or (
                record.upstream_url and record.provenance_state == "VERIFIED_REMOTE"):
            local_state, filename = self._local_artifact_state(record)
            artifact_installed, integrity_verified = local_state == "AVAILABLE", local_state == "AVAILABLE"
            if local_state == "BROKEN":
                state, inference, evidence = "BROKEN", False, "A local file failed its registered size or SHA-256 check."
            elif local_state == "QUARANTINED":
                state, inference, evidence = "QUARANTINED", False, "A corrupt managed-cache artifact was moved to quarantine."
            elif local_state == "UNVERIFIED":
                state, inference, evidence = "UNVERIFIED", False, "The configured local model paths could not be fully verified."
            else:
                state = ("AVAILABLE" if local_state == "AVAILABLE" else
                         "LICENSE_RESTRICTED" if record.license_id == "UNKNOWN" and not record.rights_unpublished
                         else "NOT_INSTALLED")
                inference = False
                compatibility = "NO_REGISTERED_ADAPTER"
                if local_state == "AVAILABLE":
                    evidence = f"Artifact hash-verified at {filename}; no compatible inference adapter is registered."
                else:
                    evidence = "Reviewed artifact identity is registered; it has not been installed in the cache or configured local paths."
        elif record.license_id == "UNKNOWN":
            state, inference, evidence = "LICENSE_RESTRICTED", False, record.license_notice
        else:
            state, inference, evidence = "INCOMPATIBLE", False, "No compatible Roop 3D engine adapter is registered for this artifact."

        governance = self.catalog.governance.scan(record, installed=artifact_installed,
                                                  integrity_verified=integrity_verified)
        if state == "QUARANTINED":
            governance["lifecycle_state"] = "QUARANTINED"
        inference = inference and governance["lifecycle_state"] == "READY"
        if not inference and state == "READY_FOR_INFERENCE":
            state, evidence = governance["lifecycle_state"], "Governance gate prevents inference: " + "; ".join(governance["issues"])

        source = record.source_repository
        if source is None and record.upstream_url:
            parsed = urlsplit(record.upstream_url)
            source = f"{parsed.scheme}://{parsed.netloc}"
        weight_files = [{"filename": name,
                         "sha256": record.expected_sha256,
                         "size_bytes": record.byte_size,
                         "source_url": record.upstream_url}
                        for name in record.filenames]
        core_purpose = next((purpose for core_record, purpose in self._core.values()
                             if core_record.model_id == record.model_id), None)
        core = core_purpose is not None
        family = (record.model_family if record.model_family != "UNKNOWN" else
                  f"InsightFace-compatible {record.family_hint}" if core else record.family_hint)
        engine_type = (record.engine_type if record.engine_type != "UNKNOWN" else
                       "ONNX Runtime" if core else "ONNX artifact" if any(name.lower().endswith(".onnx") for name in record.filenames)
                       else "UNKNOWN")
        precision = list(record.supported_precision)
        if core and not precision:
            precision = ["FP32 graph contract"]
        providers = list(record.supported_providers)
        provider_evidence = record.provider_evidence
        if core:
            provider_evidence = "Session initialization and standalone forward execution are not established by this registry view."
        dependencies = list(record.dependencies)
        if core and not dependencies:
            dependencies = ["ONNX Runtime", "Roop 3D ReferencePipeline"]
        return {
            "model_id": record.model_id,
            "name": record.name,
            "version": (record.version if record.version != "UNKNOWN" else
                        "SHA-256 identified" if record.status == "verified_local" else "UNKNOWN"),
            "family": family,
            "original_repository": record.original_repository,
            "source_revision": record.source_revision,
            "architecture": record.architecture,
            "required_base_model": record.required_base_model,
            "engine_type": engine_type,
            "purpose": core_purpose or (record.purpose if record.purpose != "UNKNOWN" else record.role_evidence),
            "source_repository": source,
            "weight_files": weight_files,
            "sha256": record.expected_sha256,
            "license": {"id": record.license_id, "commercial_use": record.commercial_use,
                        "url": record.license_url, "notice": record.license_notice,
                        "redistribution_permission": record.redistribution_permission,
                        "attribution_requirements": record.attribution_requirements,
                        "base_model": {"name": record.required_base_model,
                                       "license_id": record.base_model_license_id,
                                       "license_url": record.base_model_license_url,
                                       "commercial_use": record.base_model_commercial_use,
                                       "redistribution_permission": record.base_model_redistribution_permission,
                                       "attribution_requirements": record.base_model_attribution_requirements},
                        "acceptance_required": record.manual_acceptance_required,
                        "related_urls": list(record.related_license_urls)},
            "minimum_vram_gib": record.minimum_vram_gib,
            "recommended_vram_gib": record.recommended_vram_gib,
            "hardware_requirements": record.hardware_requirements,
            "dependencies": dependencies,
            "supported_precision": precision,
            "precision_requirements": list(record.precision_requirements),
            "dependency_compatibility": record.dependency_compatibility,
            "serialization_format": record.serialization_format,
            "security_status": record.security_status,
            "supported_execution_providers": providers,
            "provider_evidence": provider_evidence,
            "status": state,
            "lifecycle_state": governance["lifecycle_state"],
            "governance": governance,
            "status_evidence": evidence,
            "benchmark_status": record.benchmark_status,
            "actual_inference_available": inference,
            "engine_compatibility": compatibility,
            "inference_evidence": (evidence if inference else inference_note),
            "download_allowed": download_eligible and state == "NOT_INSTALLED",
            "download_blocker": (DOWNLOAD_BLOCKERS.get(blocker or "", "Download is not available for this model.")
                                 if not download_eligible else
                                 None if state == "NOT_INSTALLED" else
                                 "Already installed locally; no download needed." if state in {"AVAILABLE", "LICENSE_RESTRICTED", "READY_FOR_INFERENCE"}
                                 else "Download is not available in the model's current state."),
            "role": record.role,
        }

    def models(self) -> list[dict[str, Any]]:
        return [self._entry(record) for record in self.catalog.registry.records()]

    def model(self, model_id: str) -> dict[str, Any]:
        """Return one model's evidence without implying that byte verification means inference."""
        record = self.catalog.registry.get(model_id)
        self._core_cache.pop(model_id, None)
        return self._entry(record)

    def engines(self) -> list[dict[str, Any]]:
        required = [record for record, _ in self._core.values()]
        checks = [self._core_status(record) for record in required]
        ready = bool(required) and len(required) == 2 and all(state == "READY_FOR_INFERENCE" for state, _, _ in checks)
        states = {state for state, _, _ in checks}
        reference_state = ("READY_FOR_INFERENCE" if ready else "BROKEN" if "BROKEN" in states else
                           "LICENSE_RESTRICTED" if "LICENSE_RESTRICTED" in states else
                           "NOT_INSTALLED" if "NOT_INSTALLED" in states else "UNVERIFIED")
        return [
            {
                "engine_id": "reference_onnx",
                "name": "Roop 3D Reference ONNX",
                "family": "InsightFace-compatible detector and ArcFace recognizer (tracking / identity gate; rendering needs the true-3D head renderer)",
                "engine_type": "ONNX Runtime",
                "purpose": "Local tracking, identity gate and compositing path. Head rendering requires the true-3D renderer, which is not wired.",
                "model_ids": [record.model_id for record in required],
                "minimum_vram_gib": None,
                "recommended_vram_gib": None,
                "dependencies": ["Registered detector and recognizer artifacts", "ONNX Runtime execution provider", "True-3D head renderer (not wired)"],
                "supported_precision": ["FP32 graph contract; provider-specific reduced precision is unverified"],
                "supported_execution_providers": sorted({provider for record in required for provider in record.supported_providers}),
                "status": reference_state,
                "benchmark_status": "NOT_RUN",
                "actual_inference_available": False,
                "selectable": self.engines_registry.selectable("reference_onnx"),
                "selected": True,
                "evidence": "Resolved detector/recognizer bytes are not proof of an executable head renderer or successful model forwards.",
            },
            {
                "engine_id": "photoreal_3d_ffmpeg",
                "name": "Photoreal 3D (FFmpeg stream)",
                "family": "Same detector and ArcFace; then true-3D head render layer, quotient-image relight and Laplacian composite",
                "engine_type": "ONNX Runtime + FFmpeg decode/encode pipes",
                "purpose": "Full-head mesh rendering, semantic hair exclusion and multiband blending through the guarded true-3D path.",
                "model_ids": [record.model_id for record in required],
                "minimum_vram_gib": None,
                "recommended_vram_gib": None,
                "dependencies": ["Reference detector and ArcFace", "FFmpeg", "Measured dense landmarks",
                                 "BiSeNet / XSeg semantic masks", "True-3D head renderer (not wired)"],
                "supported_precision": ["Follows the saved precision policy"],
                "supported_execution_providers": sorted({provider for record in required for provider in record.supported_providers}),
                "status": reference_state,
                "benchmark_status": "NOT_RUN",
                "actual_inference_available": False,
                "selectable": self.engines_registry.selectable("photoreal_3d_ffmpeg"),
                "selected": False,
                "evidence": "The true-3D renderer remains unwired. No model-selection or artifact-presence check establishes production rendering.",
            },
            {
                "engine_id": "minimax_h3_ref2va_experimental",
                "name": "MiniMax H3 Ref2VA (experimental video generation)",
                "family": "ComfyUI LoRA workflow",
                "engine_type": "ComfyUI custom-node video generation",
                "purpose": "Reference-conditioned generative video; not frame-preserving identity replacement.",
                "model_ids": ["minimax_h3_ref2va_faceswap_lora"],
                "minimum_vram_gib": None,
                "recommended_vram_gib": None,
                "dependencies": ["MiniMax H3 base model and separate license", "CRT-Nodes MiniMax H3 nodes", "VideoHelperSuite", "H3 Turbo LoRA", "Local video input and video output bridge"],
                "supported_precision": [],
                "supported_execution_providers": [],
                "status": "UNVERIFIED",
                "benchmark_status": "NOT_RUN",
                "actual_inference_available": False,
                "selectable": False,
                "selected": False,
                "evidence": "The current ComfyUI bridge returns images and does not provide the local video upload/output path required by this workflow. Base weights, compatible node versions, output/audio behavior and RTX 3060/4070 requirements have not been verified here.",
            },
            {
                "engine_id": "bfs_qwen_image21_experimental",
                "name": "BFS Qwen-Image-2.1 (experimental still/keyframe edit)",
                "family": "ComfyUI image-edit LoRA",
                "engine_type": "Diffusion image editing",
                "purpose": "Possible still-image or manually selected keyframe correction; not a video-rate swapper.",
                "model_ids": ["bfs_qwen21_dare60_avgrank30", "bfs_qwen21_prune20_dare50_avgrank26", "bfs_qwen21_prune20_dare60_avgrank30"],
                "minimum_vram_gib": None,
                "recommended_vram_gib": None,
                "dependencies": ["Qwen-Image-2.1 base and separate Qwen Research License", "Compatible ComfyUI Qwen-Image workflow"],
                "supported_precision": [],
                "supported_execution_providers": [],
                "status": "INCOMPATIBLE",
                "benchmark_status": "NOT_RUN",
                "actual_inference_available": False,
                "selectable": False,
                "selected": False,
                "evidence": "No Qwen image-edit adapter or keyframe consistency integration is present in the current job pipeline. The base model is research-only for non-commercial use; the adapter MIT license does not replace it.",
            },
        ]

    def snapshot(self) -> dict[str, Any]:
        self._core_cache.clear()
        return {"models": self.models(), "engines": self.engines(),
                "model_states": sorted(MODEL_STATES),
                "lifecycle_states": sorted(LIFECYCLE_STATES),
                "execution_policy": "offline_by_default",
                "download_policy": "explicit_user_action_hash_verified_cancellable"}
