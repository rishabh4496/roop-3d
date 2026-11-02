"""Lazy, hash-checked Stage 16 artifacts and provider execution evidence."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Mapping

import numpy as np

from roop_3d.audit.project_paths import application_root, project_root
from roop_3d.core.errors import DomainError
from roop_3d.models.catalog import ModelCatalog
from roop_3d.models.owned_files import owned_path, model_source
from roop_3d.models.registry import ModelRegistry, authorize, license_info
from roop_3d.models.verifier import verify_file
from roop_3d.engine.resources import guarded, register

FACEFUSION_REPOSITORIES = tuple(f"facefusion/models-{v}" for v in ("3.0.0", "3.1.0", "3.2.0", "3.3.0"))
STAGE16_IDS = (
    "hififace_unofficial_256", "arcface_converter_hififace", "styleganex_age",
    "bisenet_resnet_18", "xseg_1", "xseg_2", "swin2_sr_x4", "real_web_photo_x4",
    "stage16_liveportrait_appearance", "stage16_liveportrait_motion", "stage16_liveportrait_generator",
    "stage16_liveportrait_stitch", "stage16_liveportrait_eye", "stage16_liveportrait_lip",
    "smirk_em1", "emoca_flame",
)


def prepare_generator(catalog: ModelCatalog | None = None) -> tuple[Path, str, int]:
    """Content-address an owned derivative of the preserved reference generator."""
    import hashlib
    from roop_3d.core.storage import atomic_json, exclusive
    from roop_3d.models.grid_sample import convert_grid_sample
    catalog = catalog or ModelCatalog()
    record = next((r for r in catalog.registry.records()
                   if r.name == "liveportrait_warping_spade.onnx" and r.locator), None)
    if record is None:
        raise RuntimeError("No inspected reference LivePortrait generator is registered")
    source = catalog.provider.resolve(record)
    relative = f"roop_3d/.runtime/stage16/generator_{record.expected_sha256}_4d_v1.onnx"
    path = owned_path(application_root() / relative)
    evidence_relative = relative + ".json"
    evidence = owned_path(application_root() / evidence_relative)
    with exclusive(application_root(), relative + ".lock"):
        if path.exists() or evidence.exists():
            if not (path.is_file() and evidence.is_file()):
                raise RuntimeError("Incomplete derivative; preserve it for inspection and select a new conversion version")
            metadata = json.loads(evidence.read_text("utf-8"))
            if metadata.get("source_sha256") != record.expected_sha256 or metadata.get("conversion") != "4d_v1":
                raise RuntimeError("Derivative provenance does not match the preserved source")
            verify_file(path, metadata["sha256"], metadata["size"])
            return path, metadata["sha256"], metadata["size"]
        changed = convert_grid_sample(source, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        size = path.stat().st_size
        atomic_json(application_root(), evidence_relative, {"source_sha256": record.expected_sha256,
                    "sha256": digest, "size": size, "conversion": "4d_v1", "replaced": changed,
                    "provider_evidence": "UNVERIFIED"})
        return path, digest, size


def resolve_artifact(model_id: str, catalog: ModelCatalog | None = None, *,
                     accepted_license: bool = False, commercial: bool = False) -> Path:
    catalog = catalog or ModelCatalog()
    record = catalog.registry.get(model_id)
    authorize(record, action="activate", accepted_license=accepted_license, commercial=commercial)
    # An identical preserved artifact is a read-only input, never copied or renamed.
    for local in catalog.registry.records():
        if local.locator and record.expected_sha256 and local.expected_sha256 == record.expected_sha256:
            return catalog.provider.resolve(local)
    return catalog.provider.resolve(record)


def readiness_report(root: Path | None = None) -> list[dict]:
    from roop_3d.audit.isolation import safe_path
    root = root or application_root()
    if not safe_path(root, "roop_3d/models/stage16_models.json").is_file():
        return []  # Isolated roots without this catalogue do not acquire global model state.
    registry = ModelRegistry(root)
    report = []
    for record in registry.records():
        if record.model_id not in STAGE16_IDS:
            continue
        category = {"identity_swapper": "swapper", "embedding_converter": "detector_embedder",
                    "age_modifier": "enhancer", "frame_upscaler": "enhancer", "face_parser": "occlusion_masking",
                    "occlusion_mask": "occlusion_masking"}.get(record.role, "3d_geometry")
        report.append({"model_id": record.model_id, "name": record.name, "category": category,
                       "readiness_state": "MISSING", "hash_valid": False,
                       "artifact_evidence": record.provenance_state, "graph_evidence": record.input_output_contract.get("state"),
                       "inference_evidence": "UNVERIFIED", "gaze_evidence": "UNVERIFIED",
                       "expected_sha256": record.expected_sha256, "expected_size_bytes": record.byte_size,
                       "primary_url": record.upstream_url, "fallback_url": None,
                       "vram_footprint_mb": None, "precision_policy": "FP32",
                       "supports_dynamic_batch": None, "model_path": None, "trt_engine_path": None,
                       "description": record.evidence, "rights": license_info(record)})
        # Presence is resolved only through verified identities, including preserved locators.
        if record.expected_sha256:
            try:
                path = resolve_artifact(record.model_id, ModelCatalog(root=registry.root, registry=registry))
                report[-1].update(readiness_state="DOWNLOADED (NEEDS TRT)", hash_valid=True, model_path=str(path))
            except DomainError as error:
                if error.code == "MODEL_DRIFT":
                    report[-1]["readiness_state"] = "CORRUPT HASH"
                elif error.code not in {"MODEL_NOT_INSTALLED", "LICENSE_ACCEPTANCE_REQUIRED"}:
                    raise
    return report


class CheckedSession:
    """Explicitly created sessions only; never instantiated during API startup.

    Provider selection alone is not proof of native execution. Call
    finish_evidence after representative inference to inspect every kernel.
    """
    @guarded
    def __init__(self, path: Path, digest: str, size: int, *, provider: str = "CPUExecutionProvider",
                 profile: str = "6gb", evidence_dir: Path | None = None,
                 batch_dimensions: tuple[str, ...] = ()):
        if provider not in {"CPUExecutionProvider", "CUDAExecutionProvider", "TensorrtExecutionProvider"}:
            raise ValueError("Unsupported provider")
        if profile not in {"6gb", "12gb"}:
            raise ValueError("Choose the independent 6gb or 12gb profile")
        self.path = model_source(path)
        verify_file(self.path, digest, size)
        from roop_3d.models.trt_builder import configure_native_paths
        from roop_3d.models.native_libraries import register_native_libraries
        if provider != "CPUExecutionProvider":
            configure_native_paths(application_root())
            register_native_libraries()
        import onnxruntime as ort
        if provider not in ort.get_available_providers():
            raise RuntimeError(f"Required provider is unavailable: {provider}")
        self.provider = provider
        self.evidence_dir = owned_path(evidence_dir or project_root() / "cache/stage16/providers")
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        options = ort.SessionOptions()
        options.log_severity_level = 3
        for dimension in batch_dimensions:
            options.add_free_dimension_override_by_name(dimension, 1)
        options.enable_profiling = True
        options.profile_file_prefix = str(owned_path(self.evidence_dir / (self.path.stem + "_" + uuid.uuid4().hex + "_execution")))
        if provider != "CPUExecutionProvider":
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        selected: list = [provider]
        if provider == "TensorrtExecutionProvider":
            import tensorrt
            if not str(tensorrt.__version__).startswith("10."):
                raise RuntimeError("Stage 16 acceptance requires TensorRT 10")
            cache = owned_path(self.evidence_dir / f"trt_{profile}_{digest}")
            cache.mkdir(parents=True, exist_ok=True)
            selected = [(provider, {"trt_engine_cache_enable": True, "trt_engine_cache_path": str(cache),
                                   "trt_max_workspace_size": {"6gb": 256, "12gb": 512}[profile] * 1024**2,
                                   "trt_fp16_enable": False, "trt_min_subgraph_size": 1})]
        self.session = ort.InferenceSession(str(self.path), sess_options=options, providers=selected)
        self.session.disable_fallback()
        if provider not in self.session.get_providers():
            raise RuntimeError("Requested native provider failed to initialize")
        self.runs = 0
        self._evidence = None
        self._profile_closed = False
        register(self, "close")

    @guarded
    def run(self, feeds: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._profile_closed or self.session is None:
            raise RuntimeError("Provider evidence is finalized; prepare a new session for further inference")
        expected = self.session.get_inputs()
        if set(feeds) != {v.name for v in expected}:
            raise ValueError("Feed names differ from the inspected ONNX contract")
        for node in expected:
            value = np.asarray(feeds[node.name])
            dtype = {"tensor(float)": np.float32, "tensor(float16)": np.float16}.get(node.type)
            if dtype is None or value.dtype != dtype or not np.isfinite(value).all():
                raise ValueError(f"Invalid dtype or nonfinite input: {node.name}")
            if len(node.shape) != value.ndim or any(isinstance(s, int) and s != actual
                                                   for s, actual in zip(node.shape, value.shape)):
                raise ValueError(f"Shape differs from inspected ONNX contract: {node.name}")
        values = self.session.run(None, dict(feeds))
        if not values or any(not np.isfinite(v).all() for v in values):
            raise RuntimeError("Model emitted empty or nonfinite outputs")
        self.runs += 1
        return dict(zip((v.name for v in self.session.get_outputs()), values))

    @guarded
    def finish_evidence(self) -> dict:
        if self._evidence is not None:
            return self._evidence.copy()
        if self.runs == 0:
            raise RuntimeError("Representative inference must execute before provider acceptance")
        if self._profile_closed or self.session is None:
            raise RuntimeError("Execution profile was closed without native acceptance")
        profile_path = owned_path(Path(self.session.end_profiling()))
        self._profile_closed = True
        events = json.loads(profile_path.read_text("utf-8"))
        kernels = [e["args"]["provider"] for e in events if e.get("cat") == "Node"
                   and e.get("args", {}).get("provider")]
        if not kernels or any(p != self.provider for p in kernels):
            raise RuntimeError(f"Native execution acceptance failed; observed providers: {sorted(set(kernels))}")
        self._evidence = {"provider": self.provider, "runs": self.runs, "kernels": len(kernels),
                          "native": True, "profile": str(profile_path)}
        return self._evidence.copy()

    @guarded
    def close(self) -> None:
        if self.session is not None:
            if not self._profile_closed:
                self.session.end_profiling()
                self._profile_closed = True
            self.session = None
