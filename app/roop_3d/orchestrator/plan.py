"""Pipeline plans: one Engine per category, with the structural checks every plan must pass."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from roop_3d.orchestrator.registry import CapabilityRegistry
from roop_3d.orchestrator.schema import (CATEGORY_ORDER, NON_IMAGE_CATEGORIES, PIPELINE_DERIVATIONS, SOURCE_KINDS,
                                         EngineEntry, ReasonCode, digest_of)

Reason = tuple[ReasonCode, str, str | None]  # (code, detail, engine_id)


@dataclass(frozen=True)
class PlanEngine:
    category: str
    engine_id: str
    provider: str
    precision: str
    entry_digest: str
    model_sha256s: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"category": self.category, "engine_id": self.engine_id, "provider": self.provider,
                "precision": self.precision, "entry_digest": self.entry_digest,
                "model_sha256s": list(self.model_sha256s)}


@dataclass(frozen=True)
class Plan:
    profile_id: str
    tier: str
    engines: tuple[PlanEngine, ...]  # in CATEGORY_ORDER

    @property
    def digest(self) -> str:
        return digest_of({"tier": self.tier, "engines": [e.as_dict() for e in self.engines]})

    @property
    def key(self) -> tuple[str, ...]:
        return tuple(e.engine_id for e in self.engines)

    def engine_for(self, category: str) -> PlanEngine | None:
        return next((e for e in self.engines if e.category == category), None)

    def engine_ids(self) -> frozenset[str]:
        return frozenset(e.engine_id for e in self.engines)


def plan_engine(registry: CapabilityRegistry, engine: EngineEntry, provider: str, precision: str) -> PlanEngine:
    hashes = tuple(sorted(registry.model_hashes[ref.model_id] or "" for ref in engine.models))
    return PlanEngine(engine.category, engine.engine_id, provider, precision, engine.digest(), hashes)


def make_plan(profile_id: str, tier: str, engines: list[PlanEngine]) -> Plan:
    return Plan(profile_id, tier, tuple(sorted(engines, key=lambda e: CATEGORY_ORDER.index(e.category))))


def _derive(available: set[str]) -> None:
    changed = True
    while changed:
        changed = False
        for needs, produces in PIPELINE_DERIVATIONS:
            if produces not in available and needs <= available:
                available.add(produces)
                changed = True


def structural_reasons(registry: CapabilityRegistry, plan: Plan, long_side: int | None) -> list[Reason]:
    """Contract chain, exclusivity and resolution. Order-stable; every violation is reported."""
    reasons: list[Reason] = []
    available = set(SOURCE_KINDS)
    for item in plan.engines:
        engine = registry.engine(item.engine_id)
        _derive(available)
        missing = [kind for kind in engine.inputs if kind not in available]
        if missing:
            reasons.append((ReasonCode.CONTRACT_MISMATCH, f"{engine.engine_id} needs {', '.join(missing)}", engine.engine_id))
        available.update(engine.outputs)
        if long_side is not None and engine.category not in NON_IMAGE_CATEGORIES:
            low, high = engine.resolution.min, engine.resolution.max
            if (low is not None and long_side < low) or (high is not None and long_side > high):
                reasons.append((ReasonCode.RESOLUTION_UNSUPPORTED,
                                f"{engine.engine_id} supports {low}-{high}px, media is {long_side}px", engine.engine_id))
    _derive(available)
    if "encoded_video" not in available:
        reasons.append((ReasonCode.CONTRACT_MISMATCH, "plan does not produce encoded_video", None))
    ids = plan.engine_ids()
    for item in plan.engines:
        for other in registry.engine(item.engine_id).exclusive_with:
            if other in ids and item.engine_id < other:
                reasons.append((ReasonCode.EXCLUSIVE_CONFLICT, f"{item.engine_id} excludes {other}", item.engine_id))
    return reasons


@dataclass(frozen=True)
class Admission:
    admitted: bool
    peak_vram_mib: int | None
    peak_source: str          # measured | declared | none
    budget_mib: int
    reserve_mib: int
    schedule: str             # sequential | co_resident | cpu
    residency: tuple[tuple[str, str], ...]
    reasons: tuple[tuple[ReasonCode, str], ...]
    notes: tuple[str, ...] = ()


def declared_peak(registry: CapabilityRegistry, plan: Plan) -> tuple[int | None, list[tuple[ReasonCode, str]], tuple[str, ...]]:
    """Peak from declared per-engine figures under the tier's residency schedule (unmeasured, labelled so)."""
    heavy: list[int] = []
    light: list[int] = []
    unknown: list[str] = []
    notes: list[str] = []
    for item in plan.engines:
        engine = registry.engine(item.engine_id)
        if engine.residency == "none":
            if engine.category in NON_IMAGE_CATEGORIES:
                notes.append(f"{engine.engine_id}: hardware codec VRAM is not accounted by the governor")
            continue
        if engine.declared_peak_vram_mib is None:
            unknown.append(engine.engine_id)
        else:
            (heavy if engine.residency == "heavy" else light).append(engine.declared_peak_vram_mib)
    if unknown:
        return None, [(ReasonCode.PEAK_VRAM_UNKNOWN, "no declared peak for " + ", ".join(sorted(unknown)))], tuple(notes)
    if plan.tier == "6gb":
        total = (max(heavy) if heavy else 0) + sum(light)
    else:
        total = sum(heavy) + sum(light)
    return total, [], tuple(notes)


def admit(registry: CapabilityRegistry, plan: Plan, *, budget_mib: int, reserve_mib: int,
          measured_peak_mib: int | None, source: str) -> Admission:
    residency = tuple((e.engine_id, registry.engine(e.engine_id).residency) for e in plan.engines)
    if plan.tier == "cpu":
        gpu = [e.engine_id for e in plan.engines if e.provider not in {"CPUExecutionProvider", "SoftwareCodec", "CPUNative"}]
        reasons = tuple((ReasonCode.PROVIDER_UNSUPPORTED, f"{i} uses a GPU provider on the cpu tier") for i in gpu)
        return Admission(not reasons, 0, "none", 0, 0, "cpu", residency, reasons)
    schedule = "sequential" if plan.tier == "6gb" else "co_resident"
    if source == "measured":
        peak, extra, notes = measured_peak_mib, ([] if measured_peak_mib is not None else
                                                  [(ReasonCode.PEAK_VRAM_UNKNOWN, "evidence records no peak VRAM")]), ()
    else:
        peak, extra, notes = declared_peak(registry, plan)
    reasons = list(extra)
    if peak is not None and peak + reserve_mib > budget_mib:
        reasons.append((ReasonCode.EXCEEDS_VRAM_BUDGET,
                        f"peak {peak} MiB + reserve {reserve_mib} MiB exceeds budget {budget_mib} MiB"))
    return Admission(not reasons, peak, source if peak is not None else "none", budget_mib, reserve_mib, schedule,
                     residency, tuple(reasons), notes)
