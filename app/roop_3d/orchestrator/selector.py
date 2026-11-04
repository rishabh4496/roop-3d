"""Deterministic Engine selection: Manual, Balanced and Quality.

Selection is a pure function of a frozen snapshot (registry digests, mode, picks, tier, configured budget,
media facts, scene segments and the evidence set). Live VRAM never enters here: it may later *refuse* an
admitted plan at run time, but it can never change which Engines a plan contains. Nothing is substituted
silently: a request that cannot be honoured is reported ``INFEASIBLE`` with every reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import itertools
from typing import Any, Sequence

from roop_3d.core.errors import DomainError
from roop_3d.core.hardware import LOW_VRAM_CEILING_MIB, STANDARD_VRAM_CEILING_MIB
from roop_3d.orchestrator.compile import compile_plan
from roop_3d.orchestrator.evidence import EvidenceMatch, EvidenceStore
from roop_3d.orchestrator.plan import (Admission, Plan, Reason, admit, make_plan, plan_engine, structural_reasons)
from roop_3d.orchestrator.registry import CapabilityRegistry
from roop_3d.orchestrator.scene import SceneSegment, whole_target_segment
from roop_3d.orchestrator.schema import (ADAPTIVE_PROFILE, BASELINE_PROFILE, CATEGORY_ORDER, TIERS, Mode,
                                         ReasonCode, digest_of)

TIER_BUDGET_MIB = {"cpu": 0, "6gb": LOW_VRAM_CEILING_MIB, "12gb": STANDARD_VRAM_CEILING_MIB}
DEFAULT_RESERVE_MIB = 768
MAX_PLANS = 4096


@dataclass(frozen=True)
class ManualPick:
    category: str
    engine_id: str
    provider: str | None = None
    precision: str | None = None


@dataclass(frozen=True)
class MediaFacts:
    width: int
    height: int

    @property
    def long_side(self) -> int:
        return max(self.width, self.height)


@dataclass(frozen=True)
class SelectionRequest:
    mode: Mode
    tier: str
    profile_id: str = ADAPTIVE_PROFILE
    manual: tuple[ManualPick, ...] = ()
    budget_mib: int | None = None
    reserve_mib: int = DEFAULT_RESERVE_MIB
    media: MediaFacts | None = None
    segments: tuple[SceneSegment, ...] = ()


@dataclass
class PlanEval:
    plan: Plan
    structural: list[Reason]
    evidence: EvidenceMatch
    admission: Admission | None
    stage: int  # 0 structural failure, 1 no usable evidence, 2 inadmissible, 3 scorable

    @property
    def feasible(self) -> bool:
        return self.stage == 3


@dataclass
class Decision:
    segment: SceneSegment
    plan: Plan | None
    basis: str                      # manual | evidence_quality | evidence_balanced | baseline_unscored | none
    reason_code: str                # why this plan: closed vocabulary string
    evidence: EvidenceMatch | None
    admission: Admission | None
    score: float | None
    score_note: str
    problems: list[Reason]
    rejected: list[dict[str, Any]]
    notes: list[str] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return self.plan is not None


@dataclass
class Selection:
    request: SelectionRequest
    snapshot: dict[str, Any]
    decisions: list[Decision]
    budget_mib: int
    eligibility: dict[str, list[Reason]]
    candidates_considered: int

    @property
    def status(self) -> str:
        return "FEASIBLE" if self.decisions and all(d.feasible for d in self.decisions) else "INFEASIBLE"


def _dedupe(reasons: Sequence[Reason]) -> list[Reason]:
    seen: set[tuple[str, str, str | None]] = set()
    result: list[Reason] = []
    for code, detail, engine_id in reasons:
        key = (code.value, detail, engine_id)
        if key not in seen:
            seen.add(key)
            result.append((code, detail, engine_id))
    return sorted(result, key=lambda r: (r[2] or "", r[0].value, r[1]))


class Selector:
    def __init__(self, registry: CapabilityRegistry, evidence: EvidenceStore | None = None):
        self.registry = registry
        self.evidence = evidence or EvidenceStore()

    # ------------------------------------------------------------------ snapshot
    def snapshot(self, request: SelectionRequest, segments: Sequence[SceneSegment], budget: int) -> dict[str, Any]:
        body = {"registry": dict(self.registry.digests),
                "model_hashes_digest": digest_of(sorted(self.registry.model_hashes.items())),
                "evidence_digest": digest_of([r.model_dump(mode="json") for r in self.evidence.records]),
                "mode": request.mode, "tier": request.tier, "profile_id": request.profile_id,
                "manual": [[p.category, p.engine_id, p.provider, p.precision] for p in request.manual],
                "budget_mib": budget, "reserve_mib": request.reserve_mib,
                "media": None if request.media is None else [request.media.width, request.media.height],
                "segments": [s.as_dict() for s in segments]}
        return {**body, "snapshot_digest": digest_of(body)}

    # ----------------------------------------------------------------- selection
    def select(self, request: SelectionRequest) -> Selection:
        if request.tier not in TIERS:
            raise DomainError("TIER_INVALID", "Hardware tier must be cpu, 6gb or 12gb", 422)
        if request.mode not in ("manual", "balanced", "quality"):
            raise DomainError("MODE_INVALID", "Mode must be manual, balanced or quality", 422)
        if request.mode != "manual" and request.manual:
            raise DomainError("MANUAL_PICKS_NOT_ALLOWED", "Engine picks are only valid in manual mode", 422)
        budget = TIER_BUDGET_MIB[request.tier] if request.budget_mib is None else request.budget_mib
        if budget < 0 or request.reserve_mib < 0:
            raise DomainError("BUDGET_INVALID", "VRAM budget and reserve must not be negative", 422)
        segments = tuple(request.segments) or (whole_target_segment(),)
        profile = self.registry.profile(request.profile_id)
        eligibility = {eid: _dedupe([(c, d, eid) for c, d in self.registry.engine_reasons(e, request.tier, budget)])
                       for eid, e in self.registry.engines.items()}
        snapshot = self.snapshot(request, segments, budget)
        if request.mode == "manual":
            decisions, considered = self._manual(request, profile, segments, budget, eligibility), 1
        else:
            decisions, considered = self._auto(request, profile, segments, budget, eligibility)
        return Selection(request, snapshot, decisions, budget, eligibility, considered)

    # ------------------------------------------------------------------- helpers
    def _long_side(self, request: SelectionRequest) -> int | None:
        return None if request.media is None else request.media.long_side

    def _build_plan(self, request: SelectionRequest, profile_id: str, picks: Sequence[ManualPick]
                    ) -> tuple[Plan | None, list[Reason]]:
        engines, problems = [], []
        for pick in picks:
            engine = self.registry.engines[pick.engine_id]
            execution = self.registry.default_execution(engine, request.tier, pick.provider, pick.precision)
            if isinstance(execution, list):
                problems.extend((code, detail, pick.engine_id) for code, detail in execution)
                continue
            engines.append(plan_engine(self.registry, engine, *execution))
        return (None, problems) if problems else (make_plan(profile_id, request.tier, engines), [])

    def _evaluate(self, request: SelectionRequest, plan: Plan, budget: int) -> PlanEval:
        structural = structural_reasons(self.registry, plan, self._long_side(request))
        structural += [(ReasonCode.RUNNER_CANNOT_EXPRESS, text, None) for text in compile_plan(plan).unsupported]
        structural = _dedupe(structural)
        evidence = self.evidence.match(plan)
        if structural:
            return PlanEval(plan, structural, evidence, None, 0)
        if not evidence.usable:
            return PlanEval(plan, [], evidence, None, 1)
        admission = admit(self.registry, plan, budget_mib=budget, reserve_mib=request.reserve_mib,
                          measured_peak_mib=evidence.record.peak_vram_mib, source="measured")
        return PlanEval(plan, [], evidence, admission, 3 if admission.admitted else 2)

    def _declared_admission(self, request: SelectionRequest, plan: Plan, budget: int) -> Admission:
        return admit(self.registry, plan, budget_mib=budget, reserve_mib=request.reserve_mib,
                     measured_peak_mib=None, source="declared")

    # ------------------------------------------------------------------- manual
    def _manual(self, request: SelectionRequest, profile, segments, budget, eligibility) -> list[Decision]:
        problems: list[Reason] = []
        rules = {r.category: r for r in (*profile.required, *profile.optional)}
        required = {r.category for r in profile.required}
        picked: dict[str, ManualPick] = {}
        for pick in request.manual:
            engine = self.registry.engines.get(pick.engine_id)
            if engine is None:
                problems.append((ReasonCode.NOT_IN_PROFILE, "unknown engine", pick.engine_id))
            elif pick.category != engine.category or engine.category not in rules or \
                    engine.engine_id not in rules[engine.category].allowed_engines:
                problems.append((ReasonCode.NOT_IN_PROFILE, f"not allowed for {pick.category} in profile {profile.profile_id}",
                                 pick.engine_id))
            elif pick.category in picked:
                problems.append((ReasonCode.NOT_IN_PROFILE, f"second pick for {pick.category}", pick.engine_id))
            else:
                picked[pick.category] = pick
                problems.extend(eligibility[pick.engine_id])
                execution = self.registry.default_execution(engine, request.tier, pick.provider, pick.precision)
                if isinstance(execution, list):
                    problems.extend((code, detail, pick.engine_id) for code, detail in execution)
        problems.extend((ReasonCode.REQUIRED_CATEGORY_MISSING, c, None) for c in CATEGORY_ORDER
                        if c in required and c not in picked)
        plan: Plan | None = None
        evaluation: PlanEval | None = None
        admission: Admission | None = None
        if not problems:
            plan, built = self._build_plan(request, profile.profile_id, [picked[c] for c in CATEGORY_ORDER if c in picked])
            problems.extend(built)
            if plan is not None:
                evaluation = self._evaluate(request, plan, budget)
                problems.extend(evaluation.structural)
                admission = self._declared_admission(request, plan, budget)
                problems.extend((c, d, None) for c, d in admission.reasons)
        problems = _dedupe(problems)
        chosen = plan if not problems else None
        decisions = []
        for segment in segments:
            rejected = self._manual_rejected(profile, picked, chosen, eligibility)
            decisions.append(Decision(segment, chosen, "manual" if chosen else "none",
                                      "MANUAL" if chosen else "INFEASIBLE",
                                      evaluation.evidence if evaluation else None, admission,
                                      None, "manual selection is not scored", problems, rejected,
                                      ["manual picks are never substituted"]))
        return decisions

    def _manual_rejected(self, profile, picked, chosen, eligibility) -> list[dict[str, Any]]:
        out = []
        for rule in (*profile.required, *profile.optional):
            for eid in sorted(rule.allowed_engines):
                if chosen is not None and eid in chosen.engine_ids():
                    continue
                reasons = list(eligibility[eid]) or [(ReasonCode.NOT_SELECTED_MANUAL, "not picked", eid)]
                out.append({"category": rule.category, "engine_id": eid, "reasons": _reason_dicts(reasons)})
        return out

    # --------------------------------------------------------------------- auto
    def _auto(self, request: SelectionRequest, profile, segments, budget, eligibility):
        candidates = self._enumerate(request, profile, eligibility)
        evals = [self._evaluate(request, plan, budget) for plan in candidates]
        baseline_plan, baseline_problems = self._baseline(request, eligibility)
        baseline_eval = self._evaluate(request, baseline_plan, budget) if baseline_plan else None
        decisions = [self._decide(request, profile, segment, evals, baseline_plan, baseline_eval, baseline_problems,
                                  budget, eligibility) for segment in segments]
        return decisions, len(candidates)

    def _enumerate(self, request, profile, eligibility) -> list[Plan]:
        axes: list[list[Any]] = []
        for rule, required in ([(r, True) for r in profile.required] + [(r, False) for r in profile.optional]):
            options = [eid for eid in sorted(rule.allowed_engines) if not eligibility[eid]]
            if required and not options:
                return []
            axes.append(([] if required else [None]) + options)
        order = {r.category: i for i, r in enumerate([*profile.required, *profile.optional])}
        rules = sorted([*profile.required, *profile.optional], key=lambda r: CATEGORY_ORDER.index(r.category))
        axes = [axes[order[r.category]] for r in rules]
        size = 1
        for axis in axes:
            size *= len(axis)
            if size > MAX_PLANS:
                raise DomainError("ORCHESTRATOR_SEARCH_TOO_LARGE", "The profile admits too many candidate plans", 422)
        plans = []
        for combo in itertools.product(*axes):
            picks = [ManualPick(self.registry.engines[eid].category, eid) for eid in combo if eid is not None]
            plan, problems = self._build_plan(request, profile.profile_id, picks)
            if plan is not None and not problems:
                plans.append(plan)
        return plans

    def _baseline(self, request, eligibility) -> tuple[Plan | None, list[Reason]]:
        profile = self.registry.profile(BASELINE_PROFILE)
        problems: list[Reason] = []
        picks = []
        for rule in profile.required:
            eid = rule.allowed_engines[0]
            if eligibility[eid]:
                problems.extend(eligibility[eid])
                problems.append((ReasonCode.NO_ELIGIBLE_ENGINE, f"baseline needs {rule.category}", eid))
            picks.append(ManualPick(rule.category, eid))
        if problems:
            return None, _dedupe(problems)
        plan, built = self._build_plan(request, BASELINE_PROFILE, picks)
        return plan, _dedupe(built)

    @staticmethod
    def _score(evaluation: PlanEval, segment: SceneSegment) -> tuple[float, str]:
        record = evaluation.evidence.record
        scores = {item.category: item.score for item in record.scene_scores}
        if segment.scoring_categories and all(c in scores for c in segment.scoring_categories):
            return min(scores[c] for c in segment.scoring_categories), \
                "worst case over " + ", ".join(segment.scoring_categories)
        return record.composite_overall, "overall composite (no category-specific evidence for this scene)"

    def _decide(self, request, profile, segment, evals, baseline_plan, baseline_eval, baseline_problems,
                budget, eligibility) -> Decision:
        usable = [e for e in evals if e.feasible]
        floor = None
        if request.mode == "balanced" and baseline_eval is not None and baseline_eval.feasible:
            floor = self._score(baseline_eval, segment)[0]
        chosen: PlanEval | None = None
        basis, code = "none", "INFEASIBLE"
        if usable and (request.mode == "quality" or floor is not None):
            if request.mode == "quality":
                chosen = min(usable, key=lambda e: (-self._score(e, segment)[0], e.evidence.record.latency_ms_p50, e.plan.key))
                basis, code = "evidence_quality", "HIGHEST_SCORE"
            else:
                pool = [e for e in usable if self._score(e, segment)[0] >= floor]
                chosen = min(pool, key=lambda e: (e.evidence.record.latency_ms_p50, -self._score(e, segment)[0], e.plan.key))
                basis, code = "evidence_balanced", "LOWEST_LATENCY_AT_OR_ABOVE_BASELINE"
        notes: list[str] = []
        problems: list[Reason] = []
        plan = admission = evidence = score = None
        score_note = ""
        if chosen is not None:
            plan, admission, evidence = chosen.plan, chosen.admission, chosen.evidence
            score, score_note = self._score(chosen, segment)
        elif baseline_plan is not None:
            declared = self._declared_admission(request, baseline_plan, budget)
            baseline_structural = baseline_eval.structural if baseline_eval else []
            if baseline_structural or not declared.admitted:
                problems = _dedupe([*baseline_structural, *((c, d, None) for c, d in declared.reasons)])
            else:
                plan, admission, evidence = baseline_plan, declared, baseline_eval.evidence if baseline_eval else None
                basis, code = "baseline_unscored", (evidence.code.value if evidence and evidence.code
                                                    else ReasonCode.NO_EVIDENCE_FOR_PROFILE.value)
                score_note = "baseline declared; no applicable real evidence"
                notes.append("no plan has applicable real evidence on this tier; the declared baseline plan is used")
        else:
            problems = baseline_problems
            if not evals:
                problems = _dedupe([*problems, (ReasonCode.NO_ELIGIBLE_ENGINE, "profile has an unfillable required category", None)])
        rejected = self._rejected(request, profile, plan, evals, floor, segment, eligibility)
        if request.mode == "balanced" and floor is None and usable:
            notes.append("balanced mode needs applicable evidence for the baseline plan to set its quality floor")
        return Decision(segment, plan, basis, code, evidence, admission, score, score_note, problems, rejected, notes)

    def _rejected(self, request, profile, plan, evals, floor, segment, eligibility) -> list[dict[str, Any]]:
        out = []
        chosen_ids = plan.engine_ids() if plan else frozenset()
        for rule in (*sorted(profile.required, key=lambda r: CATEGORY_ORDER.index(r.category)),
                     *sorted(profile.optional, key=lambda r: CATEGORY_ORDER.index(r.category))):
            for eid in sorted(rule.allowed_engines):
                if eid in chosen_ids:
                    continue
                if eligibility[eid]:
                    reasons = list(eligibility[eid])
                else:
                    containing = [e for e in evals if eid in e.plan.engine_ids()]
                    if not containing:
                        empty = sorted(r.category for r in profile.required if all(eligibility[x] for x in r.allowed_engines))
                        reasons = [(ReasonCode.NO_ELIGIBLE_ENGINE,
                                    "no plan can be formed: no eligible engine for " + ", ".join(empty), eid)]
                    else:
                        rank = {"none": 0, "stale": 1, "not_real": 2, "veto": 3, "valid": 4}
                        top = max(x.stage for x in containing)
                        # Among equally advanced plans prefer the most informative evidence status, then a fixed order.
                        best = sorted((e for e in containing if e.stage == top),
                                      key=lambda e: (-rank[e.evidence.status], e.plan.key))[0]
                        if best.stage == 0:
                            reasons = list(best.structural)
                        elif best.stage == 1:
                            reasons = [(best.evidence.code, best.evidence.detail, eid)]
                        elif best.stage == 2:
                            reasons = [(c, d, eid) for c, d in best.admission.reasons]
                        elif floor is not None and self._score(best, segment)[0] < floor:
                            reasons = [(ReasonCode.BELOW_BASELINE, "scores below the baseline plan", eid)]
                        else:
                            reasons = [(ReasonCode.DOMINATED_BY_SELECTED, "a selected plan scores or performs better", eid)]
                out.append({"category": rule.category, "engine_id": eid, "reasons": _reason_dicts(_dedupe(reasons))})
        return out


def _reason_dicts(reasons: Sequence[Reason]) -> list[dict[str, Any]]:
    return [{"code": code.value, "detail": detail, **({"engine_id": engine_id} if engine_id else {})}
            for code, detail, engine_id in reasons]
