"""Roop 3D Objective AI Visual Quality Evaluation Framework (Stage 11).

Provides:
- Controlled, consented golden-video dataset generation and registry
- Multi-dimensional objective metric evaluation (Identity, Pose, Landmarks,
  Expressions, Temporal stability, Mask quality, Occlusion, Background,
  Perceptual SSIM/LPIPS, Artifacts, Latency)
- Blinded A/B human-reviewed comparison harness
- Persistent historical benchmark database with full model SHA-256 provenance
- Automated regression detection and baseline improvement gating
"""
from __future__ import annotations

from roop_3d.evaluation.dataset import (
    GoldenDatasetRegistry,
    GoldenScene,
    SceneCategory,
    generate_synthetic_scene,
)
from roop_3d.evaluation.metrics import (
    CompositeQualityScore,
    FrameMetricResult,
    ObjectiveMetricsEvaluator,
    SceneMetricSummary,
)
from roop_3d.evaluation.harness import (
    EngineProfile,
    EvaluationHarness,
    HarnessRunResult,
)
from roop_3d.evaluation.blind_review import (
    BlindComparisonPair,
    BlindComparisonSession,
    BlindedReviewSystem,
    HumanReviewScore,
)
from roop_3d.evaluation.benchmark_db import (
    BenchmarkDatabase,
    BenchmarkRecord,
    ModelProvenance,
)
from roop_3d.evaluation.regression import (
    AcceptanceVerdict,
    RegressionCheckResult,
    RegressionDetector,
)
from roop_3d.evaluation.reports import QualityReportGenerator

__all__ = [
    "AcceptanceVerdict",
    "BenchmarkDatabase",
    "BenchmarkRecord",
    "BlindComparisonPair",
    "BlindComparisonSession",
    "BlindedReviewSystem",
    "CompositeQualityScore",
    "EngineProfile",
    "EvaluationHarness",
    "FrameMetricResult",
    "GoldenDatasetRegistry",
    "GoldenScene",
    "HarnessRunResult",
    "HumanReviewScore",
    "ModelProvenance",
    "ObjectiveMetricsEvaluator",
    "QualityReportGenerator",
    "RegressionCheckResult",
    "RegressionDetector",
    "SceneCategory",
    "SceneMetricSummary",
    "generate_synthetic_scene",
]
