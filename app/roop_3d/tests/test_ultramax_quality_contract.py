"""UltraMax's public catalog/spec, real CPU adapter and observed-outcome contracts.

Arithmetic FP16 ONNX graphs have known outputs and explicit fixture governance.
They are not CodeFormer quality, a GPEN/ESRGAN cascade or GPU performance evidence.
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from roop_3d.compositor.restoration import UltraMaxPipelineRestorer
from roop_3d.compositor.stream_restoration import (
    CODEFORMER_REFERENCE, DecoupledRestorer, OnnxFaceRestorer, RestorationTelemetry,
)
from roop_3d.core.errors import DomainError
from roop_3d.engine.runtime import RuntimeConfig
from roop_3d.tests.quality_world import restorer_catalog


@pytest.fixture
def catalog(tmp_path):
    return restorer_catalog(tmp_path)


def test_ultramax_alias_resolves_the_verified_fp16_weight_without_claiming_inference(catalog):
    spec = catalog.ultramax_spec()
    assert spec.name == "ultramax" and spec.model_name == "codeformer.fp16.onnx"
    assert spec.size == 512 and spec.normalization == "minus_one_one"
    np.testing.assert_array_equal(spec.template, CODEFORMER_REFERENCE)
    assert catalog.weight_available("ultramax", spec.model_name)
    assert not catalog.weight_available("ultramax", "CodeFormerv0.1.onnx")
    assert not catalog.weight_available("not_registered", spec.model_name)
    row = next(row for row in catalog.public() if row["id"] == "ultramax")
    assert row["installed"] and row["verified"] and row["state"] == "VERIFIED_LOCAL"
    assert row["sha256"] == catalog.registry.get("quality-fixture").expected_sha256
    restorer = UltraMaxPipelineRestorer(catalog, RuntimeConfig())
    try:
        capability = restorer.capability()
        assert capability["model"] == spec.model_name and capability["inference"] == "NOT_RUN"
        assert capability["telemetry"]["applied"] == 0 and capability["telemetry"]["last"] is None
        assert capability["telemetry"]["mean_latency_ms"] is None
    finally:
        restorer.release()


def test_ultramax_availability_rejects_a_mutated_pinned_artifact(catalog):
    name = catalog.ultramax_spec().model_name
    assert catalog.weight_available("ultramax", name)
    path = catalog.provider.resolve("quality-fixture")
    path.write_bytes(path.read_bytes() + b"changed fixture bytes")
    assert not catalog.weight_available("ultramax", name)
    row = next(row for row in catalog.public() if row["id"] == "ultramax")
    assert not row["verified"] and row["state"] != "VERIFIED_LOCAL"
    with pytest.raises(DomainError):
        catalog.resolve_for_inference("quality-fixture")


def write_spec(catalog, **changes):
    info = {"model_name": "codeformer.fp16.onnx", "template": CODEFORMER_REFERENCE.tolist(),
            "size": 512, "normalization": "minus_one_one", "evidence": "Generated analytic test contract"}
    info.update(changes)
    path = catalog.pin_root / "roop_3d/.runtime/restorers/ultramax.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info), encoding="utf-8")
    return info


def test_runtime_ultramax_contract_is_consumed_instead_of_the_legacy_codeformer_default(catalog):
    template = CODEFORMER_REFERENCE + (4.0, 7.0)
    write_spec(catalog, template=template.tolist(), normalization="zero_one")
    restorer = UltraMaxPipelineRestorer(catalog, RuntimeConfig())
    runner = OnnxFaceRestorer(catalog, RuntimeConfig(), catalog.ultramax_spec(), fidelity_weight=1)
    try:
        assert restorer.spec.model_name == "codeformer.fp16.onnx" and restorer.spec.normalization == "zero_one"
        np.testing.assert_array_equal(restorer.spec.template, template)
        # Negating [0,1] gray clamps to black; incorrectly using [-1,1]
        # would instead produce gray 35 for this scalar image at x=0.
        image = np.full((512, 512, 3), 220, np.uint8)
        result = runner.restore(image)
        np.testing.assert_array_equal(result[:, 0], np.zeros((512, 3), np.uint8))
    finally:
        restorer.release()
        runner.close()


def test_invalid_runtime_contract_fails_closed_instead_of_using_a_default(catalog):
    write_spec(catalog, evidence="")
    with pytest.raises(ValueError, match="evidence"):
        catalog.ultramax_spec()
    with pytest.raises(ValueError, match="evidence"):
        UltraMaxPipelineRestorer(catalog, RuntimeConfig())


def test_fp16_image_and_scalar_double_fidelity_change_actual_runtime_pixels(catalog):
    runner = OnnxFaceRestorer(catalog, RuntimeConfig(), catalog.ultramax_spec(), fidelity_weight=0)
    image = np.full((512, 512, 3), 220, np.uint8)
    try:
        low = runner.restore(image)
        runner.fidelity_weight = 1
        high = runner.restore(image)
        np.testing.assert_array_equal(low, np.full_like(image, 128))
        np.testing.assert_array_equal(high[:, 0], np.full((512, 3), 35, np.uint8))
        assert np.ptp(high[256, :, 0]) > 20
    finally:
        runner.close()


@pytest.mark.parametrize("side", [128, 768])
def test_hires_restoration_keeps_native_geometry_and_reports_only_completed_inference(catalog, side):
    restorer = UltraMaxPipelineRestorer(catalog, RuntimeConfig())
    image = np.full((side, side, 3), (70, 100, 160), np.uint8)
    points = CODEFORMER_REFERENCE * side / 512
    try:
        result = restorer.restore_hires(image, points, native_face_size_px=32)
        size = max(side, 512)
        assert result.shape == (size, size, 3) and result.dtype == np.uint8
        base = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
        assert np.abs(result.astype(float) - base).mean() > 1
        assert np.max(np.abs(result.mean((0, 1)) - base.mean((0, 1)))) < 4
        capability = restorer.capability()
        telemetry = capability["telemetry"]
        assert capability["inference"] == "VERIFIED_THIS_SESSION"
        assert telemetry["applied"] == 1 and telemetry["bypassed"] == telemetry["failed"] == 0
        assert telemetry["last"]["status"] == "applied" and telemetry["last"]["strength"] == 1
        assert telemetry["mean_latency_ms"] is not None and telemetry["mean_latency_ms"] >= 0
        assert telemetry["mean_inference_ms"] is not None and telemetry["mean_inference_ms"] >= 0
        assert capability["provider"] == "CPUExecutionProvider"
        assert capability["model_sha256"] == catalog.registry.get("quality-fixture").expected_sha256
    finally:
        restorer.release()


def test_explicit_small_face_bypass_and_invalid_landmarks_have_distinct_observed_outcomes(catalog):
    restorer = UltraMaxPipelineRestorer(catalog, RuntimeConfig())
    restorer.small_face_mode = "bypass"
    image = np.full((128, 128, 3), 100, np.uint8)
    points = CODEFORMER_REFERENCE / 4
    try:
        aligned = restorer.restore_aligned(image, points, native_face_size_px=320)
        np.testing.assert_array_equal(aligned, image)
        assert aligned is not image
        assert restorer.restore_hires(image, points, native_face_size_px=32) is None
        telemetry = restorer.capability()["telemetry"]
        assert telemetry["applied"] == 0 and telemetry["bypassed"] == 2 and telemetry["failed"] == 0
        assert telemetry["bypass_reasons"] == {"small_face_bypass": 2}
        assert telemetry["mean_latency_ms"] is None and telemetry["mean_inference_ms"] is None
        assert telemetry["last"]["reason"] == "small_face_bypass"
        assert restorer.capability()["inference"] == "NOT_RUN"
        restorer.small_face_mode = "off"
        with pytest.raises(DomainError) as failure:
            restorer.restore_hires(image)
        assert failure.value.code == "RESTORER_LANDMARKS_REQUIRED"
        telemetry = restorer.capability()["telemetry"]
        assert telemetry["failed"] == 1 and telemetry["applied"] == 0
        assert telemetry["last"] == {"status": "failed", "error": "DomainError"}
    finally:
        restorer.release()


def test_adaptive_small_face_detail_is_measured_as_applied_not_bypassed(catalog):
    restorer = UltraMaxPipelineRestorer(catalog, RuntimeConfig())
    image = np.full((512, 512, 3), 110, np.uint8)
    try:
        full = restorer.restore_aligned(image, CODEFORMER_REFERENCE, native_face_size_px=160)
        restorer.small_face_mode = "adaptive"
        adaptive = restorer.restore_aligned(image, CODEFORMER_REFERENCE, native_face_size_px=160)
        delta = lambda value: np.abs(value.astype(float) - image).mean()
        assert 0.3 * delta(full) < delta(adaptive) < 0.7 * delta(full)
        telemetry = restorer.capability()["telemetry"]
        assert telemetry["applied"] == 2 and telemetry["bypassed"] == 0
        assert telemetry["last"]["status"] == "applied" and telemetry["last"]["strength"] == 0.5
    finally:
        restorer.release()


def test_nonfinite_model_output_is_reported_as_failed_and_still_raises(tmp_path):
    catalog = restorer_catalog(tmp_path, nonfinite=True)
    restorer = UltraMaxPipelineRestorer(catalog, RuntimeConfig())
    try:
        with pytest.raises(ValueError, match="output contract"):
            restorer.restore_aligned(np.full((512, 512, 3), 180, np.uint8), CODEFORMER_REFERENCE)
        telemetry = restorer.capability()["telemetry"]
        assert telemetry["failed"] == 1 and telemetry["applied"] == telemetry["bypassed"] == 0
        assert telemetry["last"] == {"status": "failed", "error": "ValueError"}
        assert telemetry["mean_latency_ms"] is None and restorer.capability()["inference"] == "NOT_RUN"
    finally:
        restorer.release()


def test_telemetry_means_exclude_bypassed_failed_and_unmeasured_inference_calls():
    telemetry = RestorationTelemetry("analytic.onnx")
    telemetry.record_applied(12, 8, strength=1)
    telemetry.record_bypassed("small_face_bypass", native_face_px=32)
    telemetry.record_failed(ValueError("invalid output"))
    telemetry.record_applied(20)
    snapshot = telemetry.snapshot()
    assert snapshot["applied"] == 2 and snapshot["bypassed"] == snapshot["failed"] == 1
    assert snapshot["mean_latency_ms"] == 16 and snapshot["mean_inference_ms"] == 8
    assert snapshot["last"] == {"status": "applied", "latency_ms": 20, "inference_ms": None}
    snapshot["bypass_reasons"].clear()
    snapshot["last"]["status"] = "modified caller copy"
    assert telemetry.snapshot()["bypass_reasons"] == {"small_face_bypass": 1}
    assert telemetry.snapshot()["last"]["status"] == "applied"


def test_worker_pool_records_host_completion_bypass_and_failure_without_gpu_latency_claims(catalog):
    spec = catalog.ultramax_spec()
    pool = DecoupledRestorer(lambda: OnnxFaceRestorer(catalog, RuntimeConfig(), spec), spec,
                             workers=1, ultramax_native_bypass=True)
    image = np.full((128, 128, 3), 180, np.uint8)
    points = CODEFORMER_REFERENCE / 4
    try:
        skipped = pool.submit(image, points, (0, 0, 32, 32), 7).result()
        np.testing.assert_array_equal(skipped, np.full((512, 512, 3), 180, np.uint8))
        restored = pool.submit(image, points, (0, 0, 400, 400), 7).result()
        assert restored.shape == (512, 512, 3) and np.abs(restored.astype(float) - 180).mean() > 10
        with pytest.raises((ValueError, DomainError)):
            pool.submit(image, np.zeros((2, 2)), (0, 0, 400, 400), 7).result()
        report = pool.telemetry_report()
        assert report["applied"] == report["bypassed"] == report["failed"] == 1
        assert report["mean_latency_ms"] is not None and report["mean_inference_ms"] is None
        assert report["device_dispatched"] == 0 and report["device_latency"] is None
        assert report["last"]["status"] == "failed"
    finally:
        pool.close()
