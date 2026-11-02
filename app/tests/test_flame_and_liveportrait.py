"""Mandatory Stage 16 acceptance. Missing hardware/assets FAIL, never skip.

Run with the dedicated interpreter. The normal software suite lives under
roop_3d/tests. This gate intentionally requires actual models and output-image
gaze measurements, and cannot pass using synthetic parameter round-trips.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

import numpy as np
import pytest

from roop_3d.audit.project_paths import application_root
from roop_3d.core.storage import atomic_json
from roop_3d.models.owned_files import owned_path
from roop_3d.models.stage16 import CheckedSession, prepare_generator


def test_generator_warping_cuda_and_tensorrt_native():
    path, digest, size = prepare_generator()
    rng = np.random.default_rng(16)
    points = rng.uniform(-.3, .3, (1, 21, 3)).astype(np.float32)
    feeds = {"feature_3d": rng.normal(0, .1, (1, 32, 16, 64, 64)).astype(np.float32),
             "kp_source": points, "kp_driving": points + rng.normal(0, .03, points.shape).astype(np.float32)}
    outputs, evidence = {}, {}
    for provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
        session = CheckedSession(path, digest, size, provider=provider, profile="12gb")
        result = session.run(feeds)["out"]
        assert result.shape == (1, 3, 512, 512)
        assert np.isfinite(result).all()
        outputs[provider] = result
        evidence[provider] = session.finish_evidence()
        del session
    np.testing.assert_allclose(outputs["CUDAExecutionProvider"], outputs["TensorrtExecutionProvider"],
                               atol=2e-4, rtol=2e-4)
    atomic_json(application_root(), "roop_3d/.runtime/stage16/generator_acceptance.json",
                {"artifact_sha256": digest, "profile": "12gb", "input": "synthetic execution fixture",
                 "providers": evidence, "max_absolute_difference": float(np.max(np.abs(
                     outputs["CUDAExecutionProvider"] - outputs["TensorrtExecutionProvider"])))})


def test_hififace_256_with_exact_arcface_converter():
    from roop_3d.swappers.hififace import HifiFace256
    swapper = HifiFace256.from_registry(provider="CUDAExecutionProvider", profile="12gb")
    y, x = np.mgrid[0:256, 0:256]
    crop = np.stack((x, y, (x+y)//2), axis=-1).astype(np.uint8)
    rng = np.random.default_rng(16)
    result = swapper.swap(crop, rng.normal(size=512).astype(np.float32))
    other = swapper.swap(crop, rng.normal(size=512).astype(np.float32))
    assert result.rgb.shape == (256, 256, 3) and result.mask.shape == (256, 256)
    assert np.isfinite(result.mask).all() and not np.array_equal(result.rgb, crop)
    assert not np.array_equal(result.rgb, other.rgb), "Changing identity must affect actual network output"
    atomic_json(application_root(), "roop_3d/.runtime/stage16/hififace_acceptance.json", {
        "profile": "12gb", "input": "synthetic crop and embeddings; no likeness/fidelity claim",
        "swapper": swapper.swapper.finish_evidence(), "converter": swapper.converter.finish_evidence()})


def test_independent_eyeball_gaze_on_rendered_swap():
    from roop_3d.engine.stage16_pipeline import Stage16Options, Stage16Pipeline
    from roop_3d.flame_3d_engine import (Flame3DEngine, MediaPipe478Estimator, OnnxFlameEncoder,
                                        OnnxGazeEstimator)
    from roop_3d.liveportrait_trt import LivePortraitTRT, OnnxGazeRetargeter
    from roop_3d.swappers.hififace import HifiFace256

    filename = os.environ.get("ROOP3D_STAGE16_GAZE_CASE")
    assert filename, ("Stage 16 remains incomplete: configure ROOP3D_STAGE16_GAZE_CASE with local calibrated "
                      "SMIRK/EMOCA, MediaPipe478, independent gaze estimator/retargeter weights, and labelled media. "
                      "LivePortrait eye-open ratios and SMIRK eyelids cannot establish eyeball gaze accuracy.")
    case = json.loads(owned_path(Path(filename)).read_text("utf-8"))

    def session(key):
        record = case[key]
        return CheckedSession(owned_path(Path(record["path"])), record["sha256"], record["size"],
                              provider="CUDAExecutionProvider", profile="12gb")

    estimator = OnnxGazeEstimator(session("gaze_estimator"), calibration_evidence=case["gaze_calibration_evidence"])
    retargeter = OnnxGazeRetargeter(session("gaze_retargeter"), calibration_evidence=case["gaze_calibration_evidence"])
    flame = OnnxFlameEncoder(session("flame_encoder"), family=case.get("flame_family", "smirk"),
                             output_names=case.get("flame_output_names"))
    task = case["mediapipe_task"]
    mesh = MediaPipe478Estimator(owned_path(Path(task["path"])), task["sha256"], task["size"])
    engine = Flame3DEngine(flame, mesh, estimator)
    pipeline = Stage16Pipeline(engine, HifiFace256.from_registry(provider="CUDAExecutionProvider", profile="12gb"),
                               LivePortraitTRT.from_registry(provider="CUDAExecutionProvider", profile="12gb", gaze=retargeter))
    tolerance = float(case["tolerance_radians"])
    assert 0 < tolerance <= .1, "Declare a gaze error ceiling of at most 0.1 radians"
    assert len(case["examples"]) >= 2, "Use at least two independently labelled gaze directions"
    measured = []
    expected_all = []
    try:
        for example in case["examples"]:
            def array(key):
                return np.load(owned_path(Path(example[key])), allow_pickle=False)
            source, target, embedding = array("source_rgb256"), array("target_rgb256"), array("source_embedding")
            expected = np.asarray(example["expected_target_gaze_radians"], np.float32)
            assert expected.shape == (4,) and np.isfinite(expected).all()
            expected_all.append(expected)
            target_gaze = engine.extract(target, require_gaze=True).flame.gaze.angles()
            np.testing.assert_allclose(target_gaze, expected, atol=tolerance, rtol=0)
            swapped = pipeline.process(source, target, embedding, options=Stage16Options(require_independent_gaze=True))
            output_gaze = engine.extract(swapped.rgb, require_gaze=True).flame.gaze.angles()
            np.testing.assert_allclose(output_gaze, expected, atol=tolerance, rtol=0)
            measured.append(float(np.max(np.abs(output_gaze-expected))))
        assert np.max(np.ptp(expected_all, axis=0)) > tolerance, "Constant labels cannot verify gaze transfer"
        assert any(not np.allclose(v[:2], v[2:], atol=tolerance) for v in expected_all), "Include asymmetric gaze"
    finally:
        mesh.close()
    atomic_json(application_root(), "roop_3d/.runtime/stage16/gaze_acceptance.json", {
        "examples": len(measured), "maximum_error_radians": max(measured), "tolerance_radians": tolerance,
        "evidence": case["gaze_calibration_evidence"]})
