from __future__ import annotations

import io
import json
import zipfile
import numpy as np
import pytest

from roop_3d.audit.isolation import sha256_file
from roop_3d.core.errors import DomainError
from roop_3d.core.storage import owned
from roop_3d.models.benchmark import benchmark, summarize
from roop_3d.models.export_validation import ParityReport, Tolerance, compare_outputs, fixture_digest
from roop_3d.models.golden import GoldenBank, ModelGolden
from roop_3d.models.precision_policy import policy_for, require_no_custom_ops
from roop_3d.models.trt_builder import workspace_budget, full_precision_build


def test_embedding_gate_detects_rotation_even_with_small_absolute_errors():
    left = np.zeros((1, 512), np.float32)
    right = left.copy()
    left[0, 0], right[0, 1] = 1, 1
    report = compare_outputs({"embedding": left}, {"embedding": right}, Tolerance(2, 2, 0.995))
    assert not report.passed and report.outputs[0].failures == ("embedding_cosine",)
    with pytest.raises(DomainError):
        report.require()
    with pytest.raises(ValueError):
        Tolerance(embedding_cosine=0.99)


@pytest.mark.parametrize("change", ["nan", "dtype", "shape", "names", "zero"])
def test_contract_errors_never_turn_into_passing_parity(change):
    left = np.ones((1, 512), np.float32)
    candidate = left.copy()
    output_name = "out"
    if change == "nan": candidate[0, 0] = np.nan
    if change == "dtype": candidate = candidate.astype(np.float64)
    if change == "shape": candidate = candidate[:, :-1]
    if change == "names": output_name = "other"
    if change == "zero": candidate[:] = 0
    with pytest.raises(DomainError):
        compare_outputs({"out": left}, {output_name: candidate}, Tolerance(2, 2, 0.995))


def test_image_metrics_and_border_gate_detect_small_local_warp():
    rng = np.random.default_rng(7)
    image = rng.random((1, 3, 32, 32), dtype=np.float32)
    policy = Tolerance(0.1, 0.01, image_psnr_db=35, image_ssim=0.99, border_absolute=0.005)
    exact = compare_outputs({"image": image}, {"image": image.copy()}, policy)
    assert exact.passed and exact.outputs[0].psnr_infinite and exact.outputs[0].psnr_db is None
    assert exact.outputs[0].ssim == pytest.approx(1)
    shifted = image.copy()
    shifted[:, :, :1] += 0.02
    report = compare_outputs({"image": image}, {"image": shifted}, policy)
    assert not report.passed and "border_absolute" in report.outputs[0].failures
    shifted = np.roll(image, 1, axis=3)
    assert not compare_outputs({"image": image}, {"image": shifted}, policy).passed


@pytest.mark.parametrize("name,role", [("2d106det.onnx", None), ("motion.onnx", "liveportrait_motion"),
                                      ("warp.onnx", "liveportrait_warp"), ("stitch.onnx", "liveportrait_stitch"),
                                      ("detector.onnx", "detector")])
def test_sensitive_models_keep_fp32(name, role):
    policy = policy_for(name, role)
    assert not policy.fp16_candidate and policy.sensitive_reason
    with pytest.raises(DomainError):
        policy.tolerance("fp16")


def test_fp16_requires_every_case_at_the_exact_model_tolerance():
    policy = policy_for("recognizer.onnx", "recognizer")
    assert policy.fp16_candidate
    image = np.ones((1, 512), np.float32)
    good = compare_outputs({"out": image}, {"out": image.copy()}, policy.fp16)
    assert policy.admit("fp16", (good, good)) == "fp16"
    with pytest.raises(DomainError): policy.admit("fp16", ())
    with pytest.raises(DomainError): policy.admit("fp16", (ParityReport(True, (), policy.fp16),))
    altered = compare_outputs({"out": image}, {"out": image + 0.1}, policy.fp16)
    with pytest.raises(DomainError): policy.admit("fp16", (good, altered))
    with pytest.raises(DomainError):
        policy.admit("fp16", (compare_outputs({"out": image}, {"out": image}, policy.fp32),))
    require_no_custom_ops()
    with pytest.raises(DomainError): require_no_custom_ops(True)


def test_golden_cannot_hide_missing_expected_cases_via_zip():
    case = {"x": np.ones((1, 512), np.float32)}
    with pytest.raises(ValueError): ModelGolden("fixture", "a" * 64, (case, case), (case,))
    with pytest.raises(ValueError): ModelGolden("fixture", "a" * 64, (), ())


def test_golden_digest_and_private_roundtrip_detect_mutation(tmp_path):
    bank = GoldenBank()
    record = {"name": "fixture.onnx", "sha256": "a" * 64}
    inputs = {"input": np.arange(512, dtype=np.float32).reshape(1, 512)}
    expected = {"out": inputs["input"] * 0.5}
    bank.record(record, inputs, expected)
    original = bank.models["fixture.onnx"].digest()
    inputs["input"][0, 0] = 123
    assert bank.models["fixture.onnx"].digest() == original
    assert not bank.models["fixture.onnx"].cases[0]["input"].flags.writeable
    bank.save(tmp_path)
    loaded = GoldenBank.load(tmp_path)
    assert loaded.models["fixture.onnx"].digest() == original
    with pytest.raises(ValueError): bank.save(tmp_path, "roop_3d/outputs/public-golden")
    archive = owned(tmp_path, "roop_3d/.runtime/golden/stage07/" + original + ".npz")
    with archive.open("ab") as stream: stream.write(b"drift")
    with pytest.raises(DomainError): GoldenBank.load(tmp_path)


def test_npy_header_cannot_request_a_zip_bomb_allocation(tmp_path):
    bank = GoldenBank()
    bank.record({"name": "fixture.onnx", "sha256": "b" * 64},
                {"x": np.ones((1, 512), np.float32)}, {"y": np.ones((1, 512), np.float32)})
    bank.save(tmp_path)
    manifest = owned(tmp_path, "roop_3d/.runtime/golden/stage07/manifest.json")
    document = json.loads(manifest.read_text())
    archive = manifest.parent / document["models"][0]["file"]
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {"descr": "<f4", "fortran_order": False, "shape": (1, 2**40)})
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("i0_0.npy", header.getvalue() + b"0000")
    document["models"][0]["sha256"] = sha256_file(archive)[1]
    manifest.write_text(json.dumps(document))
    with pytest.raises(DomainError) as error:
        GoldenBank.load(tmp_path)
    assert error.value.code == "GOLDEN_ARCHIVE_INVALID"


def test_percentiles_are_exact_and_real_calls_are_measured():
    result = summarize([1, 2, 3, 4, 5])
    assert result.p50_ms == 3 and result.p95_ms == pytest.approx(4.8)
    calls = []
    measured = benchmark(lambda: calls.append(1), warmups=2, iterations=5)
    assert len(calls) == 7 and measured.iterations == 5
    assert measured.minimum_ms <= measured.p50_ms <= measured.p95_ms <= measured.maximum_ms
    with pytest.raises(ValueError): summarize([float("nan"), 1])


def test_workspace_caps_observed_free_and_profile_headroom():
    mib = 1048576
    assert workspace_budget(2048*mib, "6gb", 768*mib, 1024*mib) == 1024*mib
    assert workspace_budget(1024*mib, "6gb", 768*mib, 1024*mib) == 256*mib
    assert workspace_budget(16000*mib, "6gb", 768*mib, 16000*mib) == 5376*mib
    assert workspace_budget(16000*mib, "12gb", 768*mib, 16000*mib) == 11520*mib
    with pytest.raises(DomainError): workspace_budget(770*mib, "6gb", 768*mib, 1024*mib)


def test_tf32_build_override_is_restored_on_failure():
    import os
    previous = os.environ.get("NVIDIA_TF32_OVERRIDE")
    with pytest.raises(RuntimeError):
        with full_precision_build():
            assert os.environ["NVIDIA_TF32_OVERRIDE"] == "0"
            raise RuntimeError("build failed")
    assert os.environ.get("NVIDIA_TF32_OVERRIDE") == previous
