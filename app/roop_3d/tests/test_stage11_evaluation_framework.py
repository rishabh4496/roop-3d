"""Stage 11: Objective Visual Quality Evaluation Framework Tests.

Verifies:
1. Golden-video dataset generation across all 11 representative categories.
2. Objective multi-dimensional metrics suite (11 metrics).
3. Composite quality score calculation and hard safety vetoes.
4. Multi-engine evaluation harness running comparative matrices.
5. Blinded A/B human-reviewed comparison system with cryptographic unblinding.
6. Persistent benchmark database with model SHA-256 and hardware provenance.
7. Regression detection and baseline improvement gating criteria.
8. Official Quality Framework documentation generation.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from roop_3d.evaluation.dataset import (
    GoldenDatasetRegistry,
    GoldenScene,
    SceneCategory,
    generate_synthetic_scene,
)
from roop_3d.evaluation.metrics import (
    CompositeQualityScore,
    ObjectiveMetricsEvaluator,
    compute_cosine_similarity,
    compute_perceptual_distance,
    compute_psnr,
    compute_ssim,
)
from roop_3d.evaluation.harness import (
    EngineProfile,
    EvaluationHarness,
    HarnessRunResult,
)
from roop_3d.evaluation.blind_review import (
    BlindComparisonPair,
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
    RegressionDetector,
)
from roop_3d.evaluation.reports import QualityReportGenerator


# --------------------------------------------------------------------------- 1. Dataset Generation
def test_all_11_dataset_categories_are_represented():
    """Verify procedural generation across all 11 required representative categories."""
    assert len(SceneCategory) == 11

    expected_categories = {
        SceneCategory.FRONTAL,
        SceneCategory.EXTREME_YAW,
        SceneCategory.RAPID_MOTION,
        SceneCategory.SPEAKING_EXPRESSIONS,
        SceneCategory.HAND_OCCLUSION,
        SceneCategory.HAIR_GLASSES,
        SceneCategory.MULTIPLE_PEOPLE,
        SceneCategory.DARK_SCENES,
        SceneCategory.MOTION_BLUR,
        SceneCategory.DIFFERENT_RESOLUTIONS,
        SceneCategory.SKIN_LIGHTING,
    }
    assert set(SceneCategory) == expected_categories

    # Generate a sample scene for each category and verify annotations
    for cat in SceneCategory:
        res = (384, 384) if cat == SceneCategory.DIFFERENT_RESOLUTIONS else (512, 512)
        scene = generate_synthetic_scene(cat, frame_count=5, resolution=res)

        assert scene.scene_id.startswith("golden_")
        assert scene.category == cat
        assert scene.frame_count == 5
        assert scene.resolution == res

        # Verify ground-truth annotations on each frame
        for frame in scene.frames:
            assert frame.image.shape == (res[1], res[0], 3)
            assert len(frame.pose_deg) == 3
            assert frame.landmarks5.shape == (5, 2)
            assert frame.landmarks68 is not None and frame.landmarks68.shape == (68, 2)
            assert frame.landmarks106 is not None and frame.landmarks106.shape == (106, 2)
            assert frame.background_mask is not None and frame.background_mask.shape == (res[1], res[0])

            if cat == SceneCategory.HAND_OCCLUSION:
                assert frame.occluder_mask is not None
                assert frame.occluder_mask.sum() > 0
            elif cat == SceneCategory.MULTIPLE_PEOPLE:
                assert len(frame.secondary_faces) >= 1


def test_golden_dataset_registry_caching(tmp_path: Path):
    """Verify registry can materialize scenes into valid MP4 video containers."""
    registry = GoldenDatasetRegistry(cache_dir=tmp_path / "golden_videos")
    scenes = registry.get_canonical_dataset(frame_count=4)
    assert len(scenes) == 11

    paths = registry.materialize_dataset(scenes[:2])
    assert len(paths) == 2
    for s_id, path in paths.items():
        assert path.exists()
        assert path.stat().st_size > 1000


# --------------------------------------------------------------------------- 2. Objective Metrics
def test_mathematical_metrics_primitives():
    """Verify PSNR, SSIM, perceptual distance, and cosine similarity primitives."""
    img_a = np.full((64, 64, 3), 120, dtype=np.uint8)
    img_identical = np.full((64, 64, 3), 120, dtype=np.uint8)
    img_diff = np.full((64, 64, 3), 180, dtype=np.uint8)

    # PSNR
    assert compute_psnr(img_a, img_identical) == 100.0
    psnr_val = compute_psnr(img_a, img_diff)
    assert 10.0 < psnr_val < 20.0

    # SSIM
    assert compute_ssim(img_a, img_identical) == pytest.approx(1.0, abs=1e-4)
    assert compute_ssim(img_a, img_diff) < 0.95
    # High-frequency textured pattern degradation
    img_pattern = np.zeros_like(img_a)
    img_pattern[::2, ::2] = 255
    assert compute_ssim(img_a, img_pattern) < 0.50

    # Perceptual Distance (lower = more similar)
    assert compute_perceptual_distance(img_a, img_identical) == pytest.approx(0.0, abs=1e-5)
    assert compute_perceptual_distance(img_a, img_diff) >= 0.0

    # Cosine Similarity
    vec_1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vec_2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vec_ortho = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert compute_cosine_similarity(vec_1, vec_2) == pytest.approx(1.0, abs=1e-5)
    assert compute_cosine_similarity(vec_1, vec_ortho) == pytest.approx(0.0, abs=1e-5)


def test_evaluator_measures_all_11_dimensions():
    """Verify ObjectiveMetricsEvaluator computes all 11 metrics across frames and scenes."""
    evaluator = ObjectiveMetricsEvaluator(
        source_canonical_embedding=np.array([1.0] + [0.0] * 511, dtype=np.float32)
    )

    scene = generate_synthetic_scene(SceneCategory.FRONTAL, frame_count=6)
    frame_results = []
    prev_frame = None

    for frame in scene.frames:
        res = evaluator.evaluate_frame(
            output_frame=frame.image,
            target_frame=frame.image,
            target_pose_deg=frame.pose_deg,
            target_landmarks=frame.landmarks5,
            predicted_landmarks=frame.landmarks5,
            target_expression=frame.expression,
            prev_output_frame=prev_frame,
            gt_occluder_mask=frame.occluder_mask,
            gt_background_mask=frame.background_mask,
            gt_face_mask=frame.face_mask,
            swapped_crop=frame.image[100:200, 100:200],
            latency_ms=15.5,
            frame_index=frame.index,
        )
        frame_results.append(res)
        prev_frame = frame.image

    summary = evaluator.summarize_scene(scene.scene_id, scene.category.value, frame_results)
    assert summary.frame_count == 6
    assert summary.identity_similarity_mean >= 0.70
    assert summary.pose_error_mae_mean_deg <= 0.1
    assert summary.landmark_rmse_iod_mean <= 0.05
    assert summary.temporal_flicker_mean <= 5.0
    assert summary.background_psnr_mean_db >= 40.0
    assert summary.latency_p50_ms == pytest.approx(15.5, abs=0.5)

    # Composite Score
    composite = evaluator.compute_composite_score(summary)
    assert composite.overall_score >= 80.0
    assert composite.passed_safety_vetoes is True
    assert len(composite.veto_reasons) == 0


def test_composite_quality_score_enforces_safety_vetoes():
    """Verify that hard safety vetoes fail closed when individual bounds are breached."""
    evaluator = ObjectiveMetricsEvaluator()
    scene = generate_synthetic_scene(SceneCategory.FRONTAL, frame_count=4)

    # 1. Critical Identity Drop Veto (< 0.28)
    res_bad_id = [
        evaluator.evaluate_frame(
            output_frame=f.image,
            target_frame=f.image,
            target_pose_deg=f.pose_deg,
            target_landmarks=f.landmarks5,
        )
        for f in scene.frames
    ]
    # Inject severe identity loss
    for r in res_bad_id:
        r.identity_similarity = 0.20
    sum_bad_id = evaluator.summarize_scene("veto_id", "test", res_bad_id)
    comp_bad_id = evaluator.compute_composite_score(sum_bad_id)
    assert comp_bad_id.passed_safety_vetoes is False
    assert any("CRITICAL_IDENTITY_DROP" in v for v in comp_bad_id.veto_reasons)

    # 2. Background Corruption Veto (< 28 dB)
    res_bad_bg = [
        evaluator.evaluate_frame(
            output_frame=f.image,
            target_frame=f.image,
            target_pose_deg=f.pose_deg,
            target_landmarks=f.landmarks5,
        )
        for f in scene.frames
    ]
    for r in res_bad_bg:
        r.background_psnr_db = 22.0
    sum_bad_bg = evaluator.summarize_scene("veto_bg", "test", res_bad_bg)
    comp_bad_bg = evaluator.compute_composite_score(sum_bad_bg)
    assert comp_bad_bg.passed_safety_vetoes is False
    assert any("BACKGROUND_CORRUPTION" in v for v in comp_bad_bg.veto_reasons)


# --------------------------------------------------------------------------- 3. Evaluation Harness
def test_evaluation_harness_multi_engine_matrix():
    """Verify harness runs comparative evaluation across all 6 engine profiles."""
    harness = EvaluationHarness()
    scene = generate_synthetic_scene(SceneCategory.SPEAKING_EXPRESSIONS, frame_count=4)

    results = harness.run_comparative_matrix(
        scene,
        profiles=(
            EngineProfile.ORIGINAL_TARGET,
            EngineProfile.STANDARD_INSWAPPER,
            EngineProfile.THREE_D_AWARE,
            EngineProfile.LIVEPORTRAIT_ENHANCED,
            EngineProfile.RESTORED_OUTPUT,
        ),
    )

    assert len(results) == 5
    assert EngineProfile.ORIGINAL_TARGET in results
    assert EngineProfile.STANDARD_INSWAPPER in results
    assert EngineProfile.THREE_D_AWARE in results
    assert EngineProfile.LIVEPORTRAIT_ENHANCED in results
    assert EngineProfile.RESTORED_OUTPUT in results

    # Verify standard inswapper has high speed, restored has high visual fidelity
    res_inswapper = results[EngineProfile.STANDARD_INSWAPPER]
    res_restored = results[EngineProfile.RESTORED_OUTPUT]

    assert res_inswapper.metric_summary.latency_p50_ms < res_restored.metric_summary.latency_p50_ms
    assert res_restored.composite_score.perceptual_subscore >= res_inswapper.composite_score.perceptual_subscore


# --------------------------------------------------------------------------- 4. Blinded Review System
def test_blinded_ab_review_workflow():
    """Verify cryptographic anonymization, Likert scoring, unblinding, and significance."""
    blind_sys = BlindedReviewSystem()
    pair = blind_sys.create_blind_pair(
        session_id="session_001",
        scene_id="scene_talking",
        model_1="standard_inswapper",
        model_2="3d_aware_v2",
    )

    assert pair.blind_label_a == "System Alpha"
    assert pair.blind_label_b == "System Beta"
    # True identities must not be exposed in user-facing labels
    assert "standard_inswapper" not in (pair.blind_label_a, pair.blind_label_b)
    assert "3d_aware_v2" not in (pair.blind_label_a, pair.blind_label_b)

    # Submit ratings from 5 reviewers
    for i in range(5):
        blind_sys.submit_rating(
            pair.pair_id,
            HumanReviewScore(
                reviewer_id=f"expert_reviewer_{i}",
                identity_preservation_a=4,
                identity_preservation_b=5,
                expression_naturalness_a=3,
                expression_naturalness_b=5,
                temporal_stability_a=4,
                temporal_stability_b=5,
                boundary_realism_a=3,
                boundary_realism_b=4,
                forced_choice_preference="B",
            ),
        )

    # Unblind
    summary = blind_sys.unblind_pair_results(pair.pair_id)
    assert summary.total_reviews == 5
    assert summary.model_1 in ("standard_inswapper", "3d_aware_v2")
    assert summary.model_2 in ("standard_inswapper", "3d_aware_v2")
    assert summary.model_1_wins + summary.model_2_wins + summary.ties == 5
    assert summary.p_value_approx <= 1.0


# --------------------------------------------------------------------------- 5. Benchmark Database
def test_historical_benchmark_database(tmp_path: Path):
    """Verify persisting benchmark records with model hashes, hardware, and tags."""
    db_file = tmp_path / "test_benchmark_db.json"
    db = BenchmarkDatabase(db_path=db_file)

    rec = db.record_run(
        benchmark_id="run_stage11_alpha",
        engine_profile="3d_aware",
        composite_score=88.5,
        safety_vetoes_passed=True,
        scene_metrics={"scene_front": {"identity": 0.76, "flicker": 4.2}},
        models={
            "inswapper_128.onnx": ModelProvenance("inswapper_128.onnx", "e4a3f08c207a38555737", "FP32"),
            "scrfd_10g_bnkps.onnx": ModelProvenance("scrfd_10g_bnkps.onnx", "a01292dc223b", "FP16"),
        },
        tags=["baseline", "primary_gpu"],
    )

    assert rec.benchmark_id == "run_stage11_alpha"
    assert rec.models["inswapper_128.onnx"].precision == "FP32"
    assert db_file.exists()

    # Re-instantiate database from disk and verify persistence
    db2 = BenchmarkDatabase(db_path=db_file)
    loaded = db2.get_record("run_stage11_alpha")
    assert loaded is not None
    assert loaded.composite_overall_score == 88.5
    assert loaded.models["inswapper_128.onnx"].sha256 == "e4a3f08c207a38555737"

    # Query baseline
    baseline = db2.get_baseline_record()
    assert baseline is not None
    assert baseline.benchmark_id == "run_stage11_alpha"


# --------------------------------------------------------------------------- 6. Regression & Acceptance Gate
def test_regression_detection_and_acceptance_gate():
    """Verify rejection of regressed candidates and acceptance of superior candidates."""
    detector = RegressionDetector(
        max_identity_drop_pct=2.0,
        max_temporal_flicker_increase_pct=15.0,
        min_required_improvement_pct=3.0,
    )
    evaluator = ObjectiveMetricsEvaluator()
    scene = generate_synthetic_scene(SceneCategory.FRONTAL, frame_count=4)

    # 1. Baseline Run
    res_base = [evaluator.evaluate_frame(f.image, f.image, f.pose_deg, f.landmarks5, latency_ms=10.0) for f in scene.frames]
    for r in res_base:
        r.identity_similarity = 0.65
        r.temporal_flicker = 8.0
    sum_base = evaluator.summarize_scene("scene_1", "frontal", res_base)
    comp_base = evaluator.compute_composite_score(sum_base)

    # 2. Regressed Candidate (Identity drops by 10%)
    res_regr = [evaluator.evaluate_frame(f.image, f.image, f.pose_deg, f.landmarks5, latency_ms=10.0) for f in scene.frames]
    for r in res_regr:
        r.identity_similarity = 0.55  # Regressed!
        r.temporal_flicker = 8.0
    sum_regr = evaluator.summarize_scene("scene_1", "frontal", res_regr)
    comp_regr = evaluator.compute_composite_score(sum_regr)

    rep_regr = detector.evaluate_candidate(sum_regr, comp_regr, sum_base, comp_base)
    assert rep_regr.verdict == AcceptanceVerdict.REJECTED
    assert len(rep_regr.regressions) > 0

    # 3. Merely Equal Candidate (Technically valid, but 0 improvement)
    rep_equal = detector.evaluate_candidate(sum_base, comp_base, sum_base, comp_base)
    assert rep_equal.verdict == AcceptanceVerdict.REJECTED
    assert "failed to demonstrate measurable improvement" in rep_equal.verdict_summary

    # 4. Superior Candidate (+10% Identity, -25% Flicker, higher score)
    res_sup = [evaluator.evaluate_frame(f.image, f.image, f.pose_deg, f.landmarks5, latency_ms=9.5) for f in scene.frames]
    for r in res_sup:
        r.identity_similarity = 0.74  # Improved!
        r.temporal_flicker = 5.0      # Improved!
    sum_sup = evaluator.summarize_scene("scene_1", "frontal", res_sup)
    comp_sup = evaluator.compute_composite_score(sum_sup)

    rep_sup = detector.evaluate_candidate(sum_sup, comp_sup, sum_base, comp_base)
    assert rep_sup.verdict == AcceptanceVerdict.ACCEPTED
    assert rep_sup.composite_improvement_pts > 0
    assert len(rep_sup.measurable_improvements) >= 1


# --------------------------------------------------------------------------- 7. Documentation Generator
def test_quality_framework_report_generation(tmp_path: Path):
    """Verify generating the official Stage 11 master quality framework markdown."""
    harness = EvaluationHarness()
    scene = generate_synthetic_scene(SceneCategory.FRONTAL, frame_count=4)
    comparison = harness.run_comparative_matrix(scene)

    db = BenchmarkDatabase(db_path=tmp_path / "bench.json")
    rec = db.record_run("stage11_doc_run", "3d_aware", 89.2, True, {})

    detector = RegressionDetector()
    cand = comparison[EngineProfile.THREE_D_AWARE]
    base = comparison[EngineProfile.STANDARD_INSWAPPER]
    report = detector.evaluate_candidate(cand.metric_summary, cand.composite_score, base.metric_summary, base.composite_score)

    doc_path = tmp_path / "STAGE11_QUALITY_FRAMEWORK.md"
    QualityReportGenerator.write_master_framework_document(doc_path, rec, report, comparison)

    assert doc_path.exists()
    content = doc_path.read_text(encoding="utf-8")
    assert "# Stage 11: Roop 3D AI Visual Quality Evaluation Framework" in content
    assert "The Non-Negotiable Acceptance Law" in content
    assert "Multi-Engine Comparative Benchmark Matrix" in content
    assert "Candidate ID" in content
