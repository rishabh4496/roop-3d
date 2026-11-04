"""Regression Detection and Baseline Improvement Gate (Stage 11).

Enforces the non-negotiable acceptance contract:
'A new model cannot become the default merely because it produces a technically valid output.
Demonstrate measurable improvement against the existing baseline using both quantitative metrics
and visual review.'

Evaluates:
- Metric regression boundaries (Identity, Pose, Landmarks, Temporal flicker, Background, Latency)
- Safety veto enforcement
- Minimum measurable improvement thresholds over the active baseline
- Composite quality score progression
- Integration with blinded human review win-rate data
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from roop_3d.core.errors import DomainError
from roop_3d.evaluation.benchmark_db import BenchmarkRecord
from roop_3d.evaluation.blind_review import BlindReviewSummary
from roop_3d.evaluation.metrics import CompositeQualityScore, SceneMetricSummary


class AcceptanceVerdict(str, Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"


@dataclass
class RegressionCheckResult:
    """The outcome of an individual metric regression check."""
    metric_name: str
    baseline_value: float
    candidate_value: float
    relative_change_pct: float
    passed: bool
    is_regression: bool
    threshold_pct: float
    failure_detail: str = ""


@dataclass
class CandidateEvaluationReport:
    """Complete evaluation report for a candidate model/pipeline against baseline."""
    candidate_id: str
    baseline_id: str
    verdict: AcceptanceVerdict
    candidate_composite_score: float
    baseline_composite_score: float
    composite_improvement_pts: float
    safety_vetoes_passed: bool
    checks: list[RegressionCheckResult] = field(default_factory=list)
    measurable_improvements: list[str] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)
    blocking_reasons: list[str] = field(default_factory=list)
    blind_review_summary: BlindReviewSummary | None = None
    verdict_summary: str = ""


class RegressionDetector:
    """Detects metric regressions and evaluates candidate acceptance criteria."""

    def __init__(
        self,
        max_identity_drop_pct: float = 2.0,
        max_pose_mae_increase_deg: float = 2.5,
        max_landmark_rmse_increase_iod: float = 0.03,
        max_temporal_flicker_increase_pct: float = 15.0,
        max_background_psnr_drop_db: float = 1.0,
        max_latency_increase_pct: float = 25.0,
        min_required_improvement_pct: float = 3.0,
    ):
        self.max_identity_drop_pct = max_identity_drop_pct
        self.max_pose_mae_increase_deg = max_pose_mae_increase_deg
        self.max_landmark_rmse_increase_iod = max_landmark_rmse_increase_iod
        self.max_temporal_flicker_increase_pct = max_temporal_flicker_increase_pct
        self.max_background_psnr_drop_db = max_background_psnr_drop_db
        self.max_latency_increase_pct = max_latency_increase_pct
        self.min_required_improvement_pct = min_required_improvement_pct

    def evaluate_candidate(
        self,
        candidate_summary: SceneMetricSummary,
        candidate_composite: CompositeQualityScore,
        baseline_summary: SceneMetricSummary,
        baseline_composite: CompositeQualityScore,
        blind_review: BlindReviewSummary | None = None,
        candidate_id: str = "candidate_run",
        baseline_id: str = "baseline_run",
    ) -> CandidateEvaluationReport:
        """Evaluate whether a candidate model qualifies to replace the baseline default."""
        checks: list[RegressionCheckResult] = []
        regressions: list[str] = []
        improvements: list[str] = []
        blockers: list[str] = []

        # 0. Safety Veto Check
        if not candidate_composite.passed_safety_vetoes:
            for reason in candidate_composite.veto_reasons:
                blockers.append(f"SAFETY_VETO: {reason}")

        # Helper for percentage delta
        def p_delta(cand: float, base: float) -> float:
            if abs(base) < 1e-6:
                return 0.0
            return ((cand - base) / abs(base)) * 100.0

        # 1. Identity Similarity Check (higher is better)
        id_delta = p_delta(candidate_summary.identity_similarity_mean, baseline_summary.identity_similarity_mean)
        id_regressed = id_delta < -self.max_identity_drop_pct
        if id_regressed:
            regressions.append(f"Identity similarity dropped by {abs(id_delta):.1f}% (limit: {self.max_identity_drop_pct}%)")
        elif id_delta >= self.min_required_improvement_pct:
            improvements.append(f"Identity similarity increased by +{id_delta:.1f}%")
        checks.append(RegressionCheckResult(
            metric_name="identity_similarity",
            baseline_value=baseline_summary.identity_similarity_mean,
            candidate_value=candidate_summary.identity_similarity_mean,
            relative_change_pct=id_delta,
            passed=not id_regressed,
            is_regression=id_regressed,
            threshold_pct=self.max_identity_drop_pct,
        ))

        # 2. Pose MAE Check (lower is better)
        pose_delta = candidate_summary.pose_error_mae_mean_deg - baseline_summary.pose_error_mae_mean_deg
        pose_regressed = pose_delta > self.max_pose_mae_increase_deg
        if pose_regressed:
            regressions.append(f"Pose error increased by +{pose_delta:.2f} deg (limit: {self.max_pose_mae_increase_deg} deg)")
        elif pose_delta <= -0.5:
            improvements.append(f"Pose accuracy improved by {abs(pose_delta):.2f} deg")
        checks.append(RegressionCheckResult(
            metric_name="pose_error_mae_deg",
            baseline_value=baseline_summary.pose_error_mae_mean_deg,
            candidate_value=candidate_summary.pose_error_mae_mean_deg,
            relative_change_pct=p_delta(candidate_summary.pose_error_mae_mean_deg, baseline_summary.pose_error_mae_mean_deg),
            passed=not pose_regressed,
            is_regression=pose_regressed,
            threshold_pct=self.max_pose_mae_increase_deg,
        ))

        # 3. Landmark RMSE Check (lower is better)
        lm_delta = candidate_summary.landmark_rmse_iod_mean - baseline_summary.landmark_rmse_iod_mean
        lm_regressed = lm_delta > self.max_landmark_rmse_increase_iod
        if lm_regressed:
            regressions.append(f"Landmark RMSE increased by +{lm_delta:.3f} IOD (limit: {self.max_landmark_rmse_increase_iod} IOD)")
        elif lm_delta <= -0.01:
            improvements.append(f"Landmark alignment improved by {abs(lm_delta):.3f} IOD")
        checks.append(RegressionCheckResult(
            metric_name="landmark_rmse_iod",
            baseline_value=baseline_summary.landmark_rmse_iod_mean,
            candidate_value=candidate_summary.landmark_rmse_iod_mean,
            relative_change_pct=p_delta(candidate_summary.landmark_rmse_iod_mean, baseline_summary.landmark_rmse_iod_mean),
            passed=not lm_regressed,
            is_regression=lm_regressed,
            threshold_pct=self.max_landmark_rmse_increase_iod,
        ))

        # 4. Temporal Flicker Check (lower is better)
        flicker_delta = p_delta(candidate_summary.temporal_flicker_mean, baseline_summary.temporal_flicker_mean)
        flicker_regressed = flicker_delta > self.max_temporal_flicker_increase_pct
        if flicker_regressed:
            regressions.append(f"Temporal flicker increased by +{flicker_delta:.1f}% (limit: {self.max_temporal_flicker_increase_pct}%)")
        elif flicker_delta <= -self.min_required_improvement_pct:
            improvements.append(f"Temporal flicker reduced by {abs(flicker_delta):.1f}%")
        checks.append(RegressionCheckResult(
            metric_name="temporal_flicker",
            baseline_value=baseline_summary.temporal_flicker_mean,
            candidate_value=candidate_summary.temporal_flicker_mean,
            relative_change_pct=flicker_delta,
            passed=not flicker_regressed,
            is_regression=flicker_regressed,
            threshold_pct=self.max_temporal_flicker_increase_pct,
        ))

        # 5. Background PSNR Check (higher is better)
        bg_delta = candidate_summary.background_psnr_mean_db - baseline_summary.background_psnr_mean_db
        bg_regressed = bg_delta < -self.max_background_psnr_drop_db
        if bg_regressed:
            regressions.append(f"Background PSNR degraded by {abs(bg_delta):.2f} dB (limit: {self.max_background_psnr_drop_db} dB)")
        elif bg_delta >= 1.0:
            improvements.append(f"Background preservation improved by +{bg_delta:.2f} dB")
        checks.append(RegressionCheckResult(
            metric_name="background_psnr_db",
            baseline_value=baseline_summary.background_psnr_mean_db,
            candidate_value=candidate_summary.background_psnr_mean_db,
            relative_change_pct=bg_delta,
            passed=not bg_regressed,
            is_regression=bg_regressed,
            threshold_pct=self.max_background_psnr_drop_db,
        ))

        # 6. Latency Check (lower is better)
        lat_delta = p_delta(candidate_summary.latency_p50_ms, baseline_summary.latency_p50_ms)
        lat_regressed = lat_delta > self.max_latency_increase_pct
        if lat_regressed:
            regressions.append(f"Warm frame latency increased by +{lat_delta:.1f}% (limit: {self.max_latency_increase_pct}%)")
        elif lat_delta <= -self.min_required_improvement_pct:
            improvements.append(f"Throughput increased by {abs(lat_delta):.1f}% speedup")
        checks.append(RegressionCheckResult(
            metric_name="latency_p50_ms",
            baseline_value=baseline_summary.latency_p50_ms,
            candidate_value=candidate_summary.latency_p50_ms,
            relative_change_pct=lat_delta,
            passed=not lat_regressed,
            is_regression=lat_regressed,
            threshold_pct=self.max_latency_increase_pct,
        ))

        # 7. Blind Review Integration
        if blind_review is not None:
            if blind_review.model_2_win_rate_pct < 45.0:
                regressions.append(f"Blinded human preference rejected candidate: {blind_review.model_2_win_rate_pct:.1f}% win rate")
            elif blind_review.model_2_win_rate_pct >= 55.0:
                improvements.append(f"Blinded human preference favored candidate: {blind_review.model_2_win_rate_pct:.1f}% win rate")

        # 8. Composite Quality Progression
        score_diff = candidate_composite.overall_score - baseline_composite.overall_score

        # Acceptance Decision Logic
        if blockers:
            verdict = AcceptanceVerdict.BLOCKED
            summary_msg = f"Candidate is BLOCKED due to {len(blockers)} safety vetoes: {'; '.join(blockers)}"
        elif regressions:
            verdict = AcceptanceVerdict.REJECTED
            summary_msg = f"Candidate REJECTED due to {len(regressions)} metric regressions: {'; '.join(regressions)}"
        elif score_diff <= 0.0 or not improvements:
            verdict = AcceptanceVerdict.REJECTED
            summary_msg = "Candidate REJECTED: Technically valid output, but failed to demonstrate measurable improvement against baseline."
        else:
            verdict = AcceptanceVerdict.ACCEPTED
            summary_msg = f"Candidate ACCEPTED (+{score_diff:.2f} pts composite improvement): Demonstrates measurable gains ({', '.join(improvements)}) with zero regressions."

        return CandidateEvaluationReport(
            candidate_id=candidate_id,
            baseline_id=baseline_id,
            verdict=verdict,
            candidate_composite_score=candidate_composite.overall_score,
            baseline_composite_score=baseline_composite.overall_score,
            composite_improvement_pts=round(score_diff, 2),
            safety_vetoes_passed=len(blockers) == 0,
            checks=checks,
            measurable_improvements=improvements,
            regressions=regressions,
            blocking_reasons=blockers,
            blind_review_summary=blind_review,
            verdict_summary=summary_msg,
        )
