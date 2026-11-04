"""Capability registry: declarative Engine data validated fail-closed against the governed Model registry.

Two failure levels (Stage 14 design):

* A malformed manifest, a duplicate id, or an Engine that contradicts its Models makes the *whole* registry
  invalid (``ORCHESTRATOR_REGISTRY_INVALID``, 503). Every issue is collected before raising.
* A well-formed Engine whose adapter is missing, whose Models are not Stage 12 ``READY``, or which does not
  support the requested tier stays registered but *unavailable*; the reasons feed the decision report.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping

from pydantic import ValidationError

from roop_3d.core.errors import DomainError
from roop_3d.orchestrator.schema import (ADAPTIVE_PROFILE, BASELINE_PROFILE, CATEGORY_ORDER, CPU_PROVIDERS,
                                         EngineEntry, EngineManifest, ReasonCode, RenderProfile,
                                         RenderProfiles, SceneRules, digest_of)

PACKAGE_DIR = Path(__file__).resolve().parent

# adapter_id -> "module:attribute". Capability facts live in the manifest; this table only says which code runs.
ADAPTERS: dict[str, str] = {
    "pyav_video_reader": "roop_3d.media.video_reader:VideoReader",
    "ffmpeg_stream_source": "roop_3d.hardware_streamer:NvdecFrameSource",
    "scrfd_detector": "roop_3d.media.detector:Detector",
    "arcface_recognizer": "roop_3d.faceset.recognizer:Recognizer",
    "landmarks_106": "roop_3d.engine.detection.dense_landmarks:DenseLandmarks",
    "inswapper_reference": "roop_3d.engine.volumetric.identity_swapper:IdentitySwapper",
    "inswapper_batch_stream": "roop_3d.vectorized_pipeline:GpuBatchPipeline",
    "identity_asset_builder": "roop_3d.reconstruction.identity_asset:IdentityAssetStore",
    "pose_blend_projector": "roop_3d.reconstruction.pose_blend:PoseBlendProjector",
    "liveportrait_retarget": "roop_3d.engine.volumetric.reference_adapter:LivePortraitAdapter",
    "tps_expression": "roop_3d.reconstruction.expression_tps:apply_expression_to_crop",
    "reenactment_3d": "roop_3d.reconstruction.reenactment:ReenactmentEngine",
    "gpen_reference": "roop_3d.compositor.restoration:GpenRestorer",
    "codeformer_reference": "roop_3d.compositor.restoration:CodeFormerRestorer",
    "stream_restorer_gpen": "roop_3d.compositor.stream_restoration:DecoupledRestorer",
    "stream_restorer_codeformer": "roop_3d.compositor.stream_restoration:DecoupledRestorer",
    "stream_restorer_ultramax": "roop_3d.compositor.stream_restoration:DecoupledRestorer",
    "bisenet_face_parser": "roop_3d.compositor.masks:BisenetFaceParser",
    "xseg_visible_face": "roop_3d.compositor.occlusion_analysis:OcclusionAnalyzer",
    "pyav_video_writer": "roop_3d.media.video_writer:encode_video",
    "ffmpeg_stream_writer": "roop_3d.hardware_streamer:NVHardwareVideoWriter",
}


def adapter_present(adapter_id: str) -> bool:
    """True when the adapter's module exists and defines the named class or function (no import, no model load)."""
    target = ADAPTERS.get(adapter_id)
    if not target:
        return False
    module, _, name = target.partition(":")
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return False
    if spec is None or not spec.origin or not Path(spec.origin).is_file():
        return False
    try:
        text = Path(spec.origin).read_text("utf-8", errors="replace")
    except OSError:
        return False
    return re.search(rf"^(?:class|def)\s+{re.escape(name)}\b", text, re.MULTILINE) is not None


@dataclass(frozen=True)
class Issue:
    code: str
    engine_id: str | None
    path: str

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "engine_id": self.engine_id, "path": self.path}


class OrchestratorRegistryError(DomainError):
    def __init__(self, issues: Iterable[Issue]):
        self.issues = sorted(set(issues), key=lambda item: (item.engine_id or "", item.path, item.code))
        super().__init__("ORCHESTRATOR_REGISTRY_INVALID",
                         f"Orchestrator registry is invalid ({len(self.issues)} issue(s))", 503)


def job_runner_ids() -> frozenset[str]:
    from roop_3d.engine.registry import REFERENCE_ENGINE, UNAVAILABLE_ENGINES
    return frozenset({REFERENCE_ENGINE, *UNAVAILABLE_ENGINES})


def legacy_residency_sets() -> tuple[frozenset[str], frozenset[str]]:
    from roop_3d.engine.adaptive_runtime import HEAVY_MODELS, LIGHT_MODELS
    return frozenset(HEAVY_MODELS), frozenset(LIGHT_MODELS)


def _schema_issues(error: ValidationError, source: str) -> list[Issue]:
    return [Issue("SCHEMA_INVALID", None, source + "." + ".".join(str(part) for part in item["loc"]))
            for item in error.errors(include_input=False, include_url=False)]


LifecycleFn = Callable[[str, dict[str, Any]], Mapping[str, Any]]


class CapabilityRegistry:
    def __init__(self, engines: EngineManifest, profiles: RenderProfiles, scene_rules: SceneRules, models: Any,
                 *, lifecycle: LifecycleFn | None = None, adapter_check: Callable[[str], bool] = adapter_present):
        issues = self.validate(engines, profiles, scene_rules, models)
        if issues:
            raise OrchestratorRegistryError(issues)
        self.manifest, self.profiles, self.scene_rules = engines, profiles, scene_rules
        self.engines: dict[str, EngineEntry] = {item.engine_id: item for item in sorted(engines.engines,
                                                                                         key=lambda e: e.engine_id)}
        self._models = models
        self._lifecycle = lifecycle
        self._adapter_check = adapter_check
        self._lifecycle_cache: dict[tuple[str, str], str] = {}
        self.digests = {"engines": digest_of(engines.model_dump(mode="json")),
                        "render_profiles": digest_of(profiles.model_dump(mode="json")),
                        "scene_rules": digest_of(scene_rules.model_dump(mode="json"))}
        self.model_hashes = {ref.model_id: models.get(ref.model_id).expected_sha256
                             for engine in self.engines.values() for ref in engine.models}

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, models: Any = None, *, directory: Path | None = None, lifecycle: LifecycleFn | None = None,
             adapter_check: Callable[[str], bool] = adapter_present) -> "CapabilityRegistry":
        directory = directory or PACKAGE_DIR
        issues: list[Issue] = []
        parsed: dict[str, Any] = {}
        for name, model in (("engines_manifest.json", EngineManifest), ("render_profiles.json", RenderProfiles),
                            ("scene_rules.json", SceneRules)):
            try:
                parsed[name] = model.model_validate_json((directory / name).read_text("utf-8"))
            except OSError:
                issues.append(Issue("FILE_UNREADABLE", None, name))
            except ValidationError as error:
                issues.extend(_schema_issues(error, name))
        if issues:
            raise OrchestratorRegistryError(issues)
        if models is None:
            from roop_3d.models.registry import ModelRegistry
            models = ModelRegistry()
        return cls(parsed["engines_manifest.json"], parsed["render_profiles.json"], parsed["scene_rules.json"],
                   models, lifecycle=lifecycle, adapter_check=adapter_check)

    # ------------------------------------------------------------- validation
    @staticmethod
    def validate(engines: EngineManifest, profiles: RenderProfiles, scene_rules: SceneRules, models: Any) -> list[Issue]:
        issues: list[Issue] = []
        by_id: dict[str, EngineEntry] = {}
        reserved = job_runner_ids()
        heavy_names, light_names = legacy_residency_sets()
        for engine in engines.engines:
            eid = engine.engine_id
            if eid in by_id:
                issues.append(Issue("DUPLICATE_ENGINE_ID", eid, "engines"))
            by_id[eid] = engine
            if eid in reserved:
                issues.append(Issue("ENGINE_ID_RESERVED", eid, "engine_id"))
            if any(tier == "cpu" for tier in engine.hardware_tiers) and not CPU_PROVIDERS.intersection(engine.providers):
                issues.append(Issue("TIER_PROVIDER_MISMATCH", eid, "hardware_tiers"))
            names: list[str] = []
            for index, ref in enumerate(engine.models):
                try:
                    record = models.get(ref.model_id)
                except DomainError:
                    issues.append(Issue("MODEL_UNKNOWN", eid, f"models.{index}.model_id"))
                    continue
                names.append(record.name)
                supported = {item for item in record.supported_providers}
                if supported and not set(engine.providers) <= supported:
                    issues.append(Issue("PROVIDER_CONTRADICTS_MODEL", eid, f"providers[{ref.model_id}]"))
                declared = {item.upper() for item in (*record.supported_precision, *record.precision_requirements)}
                if declared and not {p.upper() for p in engine.precisions} <= declared:
                    issues.append(Issue("PRECISION_CONTRADICTS_MODEL", eid, f"precisions[{ref.model_id}]"))
                if "FP16" in engine.precisions:
                    from roop_3d.models.precision_policy import policy_for
                    if not policy_for(record.name, record.role).fp16_candidate or "inswapper" in record.name.lower():
                        issues.append(Issue("PRECISION_CONTRADICTS_MODEL", eid, f"precisions[{ref.model_id}]"))
            if names:
                if any(n in heavy_names for n in names) and engine.residency != "heavy":
                    issues.append(Issue("RESIDENCY_CONTRADICTS_LEGACY", eid, "residency"))
                elif all(n in light_names for n in names) and engine.residency == "heavy":
                    issues.append(Issue("RESIDENCY_CONTRADICTS_LEGACY", eid, "residency"))
        order = {name: index for index, name in enumerate(CATEGORY_ORDER)}
        for engine in engines.engines:
            for other in engine.exclusive_with:
                if other not in by_id:
                    issues.append(Issue("EXCLUSIVE_UNKNOWN", engine.engine_id, "exclusive_with"))
                elif engine.engine_id not in by_id[other].exclusive_with:
                    issues.append(Issue("EXCLUSIVE_NOT_SYMMETRIC", engine.engine_id, "exclusive_with"))
            target = engine.fallback.target_engine_id
            if target is not None and (target not in by_id or order[by_id[target].category] > order[engine.category]):
                issues.append(Issue("FALLBACK_TARGET_INVALID", engine.engine_id, "fallback.target_engine_id"))
        seen_profiles: set[str] = set()
        for profile in profiles.profiles:
            if profile.profile_id in seen_profiles:
                issues.append(Issue("DUPLICATE_PROFILE_ID", None, f"profiles.{profile.profile_id}"))
            seen_profiles.add(profile.profile_id)
            for rule in (*profile.required, *profile.optional):
                for eid in rule.allowed_engines:
                    if eid not in by_id:
                        issues.append(Issue("PROFILE_ENGINE_UNKNOWN", eid, f"profiles.{profile.profile_id}"))
                    elif by_id[eid].category != rule.category:
                        issues.append(Issue("PROFILE_ENGINE_WRONG_CATEGORY", eid, f"profiles.{profile.profile_id}"))
        baseline = next((p for p in profiles.profiles if p.profile_id == BASELINE_PROFILE), None)
        if baseline is None:
            issues.append(Issue("BASELINE_PROFILE_MISSING", None, "profiles"))
        elif baseline.optional or any(len(rule.allowed_engines) != 1 for rule in baseline.required):
            issues.append(Issue("BASELINE_NOT_PINNED", None, "profiles.baseline"))
        if not any(p.profile_id == ADAPTIVE_PROFILE for p in profiles.profiles):
            issues.append(Issue("ADAPTIVE_PROFILE_MISSING", None, "profiles"))
        from roop_3d.evaluation.dataset import SceneCategory
        valid_categories = {item.value for item in SceneCategory}
        for item in scene_rules.trait_map:
            if not set(item.scene_categories) <= valid_categories:
                issues.append(Issue("SCENE_CATEGORY_UNKNOWN", None, f"trait_map.{item.trait}"))
        return issues

    # ------------------------------------------------------------ introspection
    def engine(self, engine_id: str) -> EngineEntry:
        try:
            return self.engines[engine_id]
        except KeyError:
            raise DomainError("ENGINE_NOT_FOUND", "Unknown orchestrator engine", 422) from None

    def profile(self, profile_id: str) -> RenderProfile:
        for item in self.profiles.profiles:
            if item.profile_id == profile_id:
                return item
        raise DomainError("PROFILE_NOT_FOUND", "Unknown render profile", 422)

    def by_category(self, category: str) -> list[EngineEntry]:
        return [e for e in self.engines.values() if e.category == category]

    def default_execution(self, engine: EngineEntry, tier: str, provider: str | None = None,
                          precision: str | None = None) -> tuple[str, str] | list[tuple[ReasonCode, str]]:
        """Deterministic provider and precision for a tier, or reasons why the request cannot be honoured."""
        allowed = [p for p in engine.providers if tier != "cpu" or p in CPU_PROVIDERS]
        reasons: list[tuple[ReasonCode, str]] = []
        if provider is not None and provider not in allowed:
            reasons.append((ReasonCode.PROVIDER_UNSUPPORTED, f"{provider} is not available for {engine.engine_id} on {tier}"))
        if precision is not None and precision not in engine.precisions:
            reasons.append((ReasonCode.PRECISION_FORBIDDEN, f"{precision} is not permitted for {engine.engine_id}"))
        if not allowed:
            reasons.append((ReasonCode.PROVIDER_UNSUPPORTED, f"no provider for {engine.engine_id} on {tier}"))
        if reasons:
            return reasons
        return provider or allowed[0], precision or ("FP32" if "FP32" in engine.precisions else engine.precisions[0])

    def model_states(self, engine: EngineEntry, tier: str, budget_mib: int) -> dict[str, str]:
        hardware = {"vram_total_gb": round(budget_mib / 1024, 3), "profile": tier}
        states: dict[str, str] = {}
        for ref in engine.models:
            key = (ref.model_id, tier)
            if key not in self._lifecycle_cache:
                try:
                    result = self._lifecycle_fn()(ref.model_id, hardware)
                    self._lifecycle_cache[key] = str(result.get("lifecycle_state", "UNKNOWN"))
                except DomainError:
                    self._lifecycle_cache[key] = "UNKNOWN"
            states[ref.model_id] = self._lifecycle_cache[key]
        return states

    def _lifecycle_fn(self) -> LifecycleFn:
        if self._lifecycle is None:
            from roop_3d.models.catalog import ModelCatalog
            catalog = ModelCatalog()
            self._lifecycle = lambda model_id, hardware: catalog.lifecycle(model_id, hardware=hardware)
        return self._lifecycle

    def engine_reasons(self, engine: EngineEntry, tier: str, budget_mib: int) -> list[tuple[ReasonCode, str]]:
        """Why an Engine cannot be used on a tier, independent of any plan. Empty means eligible."""
        reasons: list[tuple[ReasonCode, str]] = []
        if not self._adapter_check(engine.adapter_id):
            reasons.append((ReasonCode.ADAPTER_NOT_REGISTERED, engine.adapter_id))
        if tier not in engine.hardware_tiers:
            reasons.append((ReasonCode.TIER_UNSUPPORTED, f"{engine.engine_id} does not support {tier}"))
        else:
            execution = self.default_execution(engine, tier)
            if isinstance(execution, list):
                reasons.extend(execution)
        not_ready = {mid: state for mid, state in self.model_states(engine, tier, budget_mib).items() if state != "READY"}
        if not_ready:
            reasons.append((ReasonCode.NOT_GOVERNANCE_READY,
                            ", ".join(f"{mid}={state}" for mid, state in sorted(not_ready.items()))))
        return reasons
