"""Unit tests for pipeline_audit diagnostic module."""
import pytest
import numpy as np

from diagnostics.pipeline_audit import (
    AuditConfig,
    PipelineAuditor,
    StageQualityDegradationError,
    ExecutionProviderFallbackError,
    EnhancerBypassError,
)


def test_audit_config_defaults():
    config = AuditConfig()
    assert isinstance(config.debug_mode, bool)
    assert isinstance(config.fail_loud, bool)
    assert config.min_enhanced_resolution == 512
    assert config.max_downsampled_target == 128


def test_stage3_downsampling_detection_non_debug():
    config = AuditConfig(debug_mode=False, fail_loud=False)
    auditor = PipelineAuditor(config)

    # Simulate entering at 128x128, enhancing at 512x512, downsampling to 128x128
    entering = (1, 3, 128, 128)
    enhanced = (1, 3, 512, 512)
    exiting = (1, 3, 128, 128)

    auditor.shapes.check_stage3_downsampling(
        entering_shape=entering,
        enhanced_shape=enhanced,
        exiting_shape=exiting,
        frame_index=0,
        enhancer_name="GPEN-BFR-512",
    )

    report = auditor.generate_report()
    assert report["total_downsampling_events"] == 1
    event = report["stage3_downsampling_events"][0]
    assert event["enhancer"] == "GPEN-BFR-512"
    assert "downsampled back to 128x128" in event["detail"]


def test_stage3_downsampling_fails_loud_in_debug():
    config = AuditConfig(debug_mode=True, fail_loud=True)
    auditor = PipelineAuditor(config)

    with pytest.raises(StageQualityDegradationError) as exc_info:
        auditor.shapes.check_stage3_downsampling(
            entering_shape=(1, 3, 128, 128),
            enhanced_shape=(1, 3, 1024, 1024),
            exiting_shape=(1, 3, 128, 128),
            frame_index=1,
            enhancer_name="UltraMax",
        )
    assert "downsampled back" in str(exc_info.value)


def test_enhancer_bypass_detection():
    config = AuditConfig(debug_mode=False, fail_loud=False)
    auditor = PipelineAuditor(config)

    # Test <=320px bypass rule
    event = auditor.enhancers.audit_invocation(
        enhancer_name="UltraMax",
        crop_shape=(128, 128, 3),
        native_face_pixels=240,
        restoration_blend=0.8,
    )
    assert "reasons" in event
    assert any("<=320px" in r for r in event["reasons"])

    # Test restoration_blend == 0 bypass
    event2 = auditor.enhancers.audit_invocation(
        enhancer_name="CodeFormer",
        crop_shape=(512, 512, 3),
        native_face_pixels=512,
        restoration_blend=0.0,
    )
    assert any("restoration_blend is 0.0" in r for r in event2["reasons"])

    # Test GFPGAN unhandled
    event3 = auditor.enhancers.audit_invocation(
        enhancer_name="GFPGAN",
        crop_shape=(512, 512, 3),
        native_face_pixels=512,
        restoration_blend=0.8,
    )
    assert any("GFPGAN" in r for r in event3["reasons"])

    report = auditor.generate_report()
    assert report["total_enhancer_bypass_events"] == 3


def test_enhancer_bypass_fails_loud_in_debug():
    config = AuditConfig(debug_mode=True, fail_loud=True)
    auditor = PipelineAuditor(config)

    with pytest.raises(EnhancerBypassError) as exc_info:
        auditor.enhancers.audit_invocation(
            enhancer_name="UltraMax",
            crop_shape=(128, 128, 3),
            native_face_pixels=200,
        )
    assert "<=320px" in str(exc_info.value)


def test_execution_provider_fallback_detection():
    class DummySession:
        def get_providers(self):
            return ["CPUExecutionProvider"]

        def get_provider_options(self):
            return {"CPUExecutionProvider": {}}

    config = AuditConfig(debug_mode=False, fail_loud=False)
    auditor = PipelineAuditor(config)

    rep = auditor.providers.inspect_session(
        name="inswapper_128",
        session=DummySession(),
        expected_provider="TensorrtExecutionProvider",
    )
    assert rep["has_fallback"] is True
    assert "Expected primary provider 'TensorrtExecutionProvider'" in rep["fallback_details"][0]


def test_copy_counter():
    config = AuditConfig()
    auditor = PipelineAuditor(config)

    auditor.copies.record_h2d("frame_upload", shape=(1, 3, 1080, 1920), bytes_count=1080 * 1920 * 3 * 4)
    auditor.copies.record_d2h("swap_output", shape=(1, 3, 128, 128), bytes_count=128 * 128 * 3 * 4)
    auditor.copies.record_cpu_op("cv2.warpAffine", "full frame inverse composite")

    summary = auditor.copies.get_summary()
    assert summary["h2d_copies"] == 1
    assert summary["d2h_copies"] == 1
    assert summary["total_sync_copies"] == 2
    assert summary["cpu_opencv_ops"] == 1
