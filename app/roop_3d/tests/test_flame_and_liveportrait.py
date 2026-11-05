"""Stage 16 software checks; hardware/media acceptance is a separate fail-closed gate."""
from __future__ import annotations

import numpy as np
import pytest


def volume_graph(align=0, padding="zeros", depth=3, height=4, width=5):
    import onnx
    h = onnx.helper
    return h.make_model(h.make_graph([
        h.make_node("GridSample", ["volume", "grid"], ["out"], name="/dense_motion_network/GridSample",
                    mode="bilinear", align_corners=align, padding_mode=padding)
    ], "volumetric_reference", [
        h.make_tensor_value_info("volume", onnx.TensorProto.FLOAT, ["N", 2, depth, height, width]),
        h.make_tensor_value_info("grid", onnx.TensorProto.FLOAT, ["N", 2, 3, 4, 3])
    ], [h.make_tensor_value_info("out", onnx.TensorProto.FLOAT, ["N", 2, 2, 3, 4])]),
        opset_imports=[h.make_opsetid("", 16)], ir_version=9)


def trilinear_reference(volume, grid, align, padding):
    # Independent scalar interpolation, including boundaries and out-of-volume samples.
    n, channels, depth, height, width = volume.shape
    out = np.zeros((n, channels, *grid.shape[1:4]), np.float64)
    for b, od, oh, ow in np.ndindex(n, *grid.shape[1:4]):
        coordinates = []
        for g, size in zip(grid[b, od, oh, ow][::-1], (depth, height, width)):
            p = (float(g) + 1) * (size - 1) / 2 if align else ((float(g) + 1) * size - 1) / 2
            coordinates.append(np.clip(p, 0, size - 1) if padding == "border" else p)
        lo = np.floor(coordinates).astype(int)
        fraction = np.asarray(coordinates) - lo
        for offsets in np.ndindex(2, 2, 2):
            z, y, x = lo + offsets
            weight = np.prod([f if o else 1-f for f, o in zip(fraction, offsets)])
            if 0 <= z < depth and 0 <= y < height and 0 <= x < width:
                out[b, :, od, oh, ow] += volume[b, :, z, y, x] * weight
    return out


@pytest.mark.parametrize("align", [0, 1])
@pytest.mark.parametrize("padding", ["zeros", "border"])
@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("spatial", [(4, 5), (1, 1)])
def test_grid_sample_4d_matches_independent_trilinear(align, padding, depth, spatial):
    import onnxruntime as ort
    from roop_3d.models.grid_sample import lower_grid_sample_3d
    model, replaced = lower_grid_sample_3d(volume_graph(align, padding, depth, *spatial))
    assert replaced == ("/dense_motion_network/GridSample",)
    assert len([n for n in model.graph.node if n.op_type == "GridSample"]) == 2
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(16)
    for batch in (1, 2):
        volume = rng.normal(size=(batch, 2, depth, *spatial)).astype(np.float32)
        grid = rng.uniform(-3, 3, (batch, 2, 3, 4, 3)).astype(np.float32)
        grid[:, 0, 0, :, :] = np.array([-1, 0, 1, 4], np.float32)[:, None]
        expected = trilinear_reference(volume, grid, align, padding)
        actual = session.run(None, {"volume": volume, "grid": grid})[0]
        np.testing.assert_allclose(actual, expected, atol=4e-6, rtol=4e-6)


def test_grid_sample_lowering_is_idempotent_and_rejects_reflection():
    from roop_3d.models.grid_sample import lower_grid_sample_3d
    model, _ = lower_grid_sample_3d(volume_graph())
    again, changed = lower_grid_sample_3d(model)
    assert not changed
    assert again.SerializeToString() == model.SerializeToString()
    with pytest.raises(ValueError, match="zeros/border"):
        lower_grid_sample_3d(volume_graph(padding="reflection"))


def test_shape_blending_preserves_target_expression_pose_and_independent_gaze():
    from roop_3d.flame_3d_engine import FlameParameters, EyeGaze, blend_shape
    gaze = EyeGaze(.2, -.1, -.3, .15)
    source = FlameParameters(np.ones(100), np.zeros(50), np.zeros(3), np.eye(3))
    target = FlameParameters(np.full(100, 3), np.full(50, .4), np.array([.2, 0, 0]), np.eye(3), gaze=gaze)
    for weight, beta in ((0, 1), (.25, 1.5), (1, 3)):
        mixed = blend_shape(source, target, weight)
        np.testing.assert_array_equal(mixed.beta, np.full(100, beta))
        np.testing.assert_array_equal(mixed.psi, target.psi)
        np.testing.assert_array_equal(mixed.jaw_rotation, target.jaw_rotation)
        assert mixed.gaze == gaze and mixed.translation is None
        assert not mixed.beta.flags.writeable
    np.testing.assert_array_equal(source.beta, np.ones(100))
    np.testing.assert_allclose(np.linalg.norm(gaze.vectors(), axis=1), 1, atol=1e-6)
    assert np.sign(gaze.vectors()[0, 0]) != np.sign(gaze.vectors()[1, 0])
    for weight in (-.1, 1.1, np.nan, np.inf):
        with pytest.raises(ValueError, match="shape_retention_weight"):
            blend_shape(source, target, weight)
    with pytest.raises(ValueError, match=r"SO\(3\)"):
        FlameParameters(np.ones(100), np.zeros(50), np.zeros(3), np.diag([-1, 1, 1]))


def tiny_session(tmp_path, inputs, outputs, nodes=(), initializers=()):
    import onnx
    from roop_3d.models.stage16 import CheckedSession
    import hashlib
    model = onnx.helper.make_model(onnx.helper.make_graph(list(nodes), "stage16_test_export", inputs, outputs,
                                                         list(initializers)),
                                   opset_imports=[onnx.helper.make_opsetid("", 16)], ir_version=9)
    path = tmp_path / "export.onnx"
    onnx.save(model, path)
    return CheckedSession(path, hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size,
                          evidence_dir=tmp_path / "profiles")


def test_smirk_named_export_runs_without_inventing_metric_translation_or_gaze(tmp_path):
    import onnx
    from roop_3d.flame_3d_engine import OnnxFlameEncoder, Flame3DEngine
    h = onnx.helper
    fields = {"shape_params": np.arange(300, dtype=np.float32)[None] / 300,
              "expression_params": np.full((1, 50), .2, np.float32),
              "pose_params": np.array([[.1, -.2, .05]], np.float32),
              "jaw_params": np.array([[.2, .01, -.01]], np.float32),
              "cam": np.array([[7, .1, -.1]], np.float32),
              "eyelid_params": np.array([[.4, .6]], np.float32)}
    outputs = [h.make_tensor_value_info(k, onnx.TensorProto.FLOAT, v.shape) for k, v in fields.items()]
    constants = [onnx.numpy_helper.from_array(v, k) for k, v in fields.items()]
    session = tiny_session(tmp_path, [h.make_tensor_value_info("img", onnx.TensorProto.FLOAT, [1, 3, 224, 224])],
                           outputs, initializers=constants)
    encoder = OnnxFlameEncoder(session)
    engine = Flame3DEngine(encoder)
    crop = np.zeros((256, 256, 3), np.uint8)
    face = engine.extract(crop)
    np.testing.assert_array_equal(face.flame.beta, fields["shape_params"][0, :100])
    np.testing.assert_array_equal(face.flame.psi, fields["expression_params"][0])
    assert face.mesh478 is None and face.flame.gaze is None and face.flame.translation is None
    np.testing.assert_array_equal(face.flame.weak_camera, fields["cam"][0])
    with pytest.raises(RuntimeError, match="eyelids are not gaze"):
        engine.extract(crop, require_gaze=True)
    with pytest.raises(ValueError, match="cannot be decoded as FLAME"):
        OnnxFlameEncoder(session, family="3ddfa_v2")
    with pytest.raises(ValueError, match="inspected"):
        OnnxFlameEncoder(session, family="emoca")


def test_native_session_checks_hash_feeds_and_actual_profile(tmp_path):
    import onnx
    from roop_3d.models.stage16 import CheckedSession
    h = onnx.helper
    node = h.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 4])
    output = h.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 4])
    session = tiny_session(tmp_path, [node], [output], [h.make_node("Abs", ["x"], ["y"])])
    with pytest.raises(RuntimeError, match="Representative inference"):
        session.finish_evidence()
    for feeds in ({"wrong": np.zeros((1, 4), np.float32)}, {"x": np.zeros((1, 4), np.float64)},
                  {"x": np.zeros((4,), np.float32)}, {"x": np.full((1, 4), np.nan, np.float32)}):
        with pytest.raises(ValueError):
            session.run(feeds)
    np.testing.assert_array_equal(session.run({"x": -np.ones((1, 4), np.float32)})["y"], np.ones((1, 4)))
    assert session.finish_evidence()["native"]
    assert session.finish_evidence()["runs"] == 1
    with pytest.raises(RuntimeError, match="finalized"):
        session.run({"x": np.ones((1, 4), np.float32)})
    from roop_3d.core.errors import DomainError
    with pytest.raises(DomainError):
        CheckedSession(tmp_path / "export.onnx", "0" * 64, (tmp_path / "export.onnx").stat().st_size)


def test_stage16_official_metadata_and_rights_are_separate():
    from roop_3d.models.registry import ModelRegistry, authorize
    from roop_3d.models.stage16 import STAGE16_IDS, FACEFUSION_REPOSITORIES
    from roop_3d.core.errors import DomainError
    registry = ModelRegistry()
    assert len(FACEFUSION_REPOSITORIES) == 4
    for model_id in STAGE16_IDS:
        record = registry.get(model_id)
        assert record.license_id == "UNKNOWN" and record.commercial_use == "unknown"
        if record.status == "candidate":
            assert record.upstream_url is None and record.expected_sha256 is None
        else:
            assert record.status == "verified_remote" and len(record.expected_sha256) == 64
            assert len(record.upstream_url.split("/resolve/")[1].split("/")[0]) == 40
            assert record.input_output_contract["state"] == "UNKNOWN"
        # Unpublished rights: a verified source may download behind an explicit acknowledgement.
        # Candidates have no source or digest and stay refused.
        with pytest.raises(DomainError):
            authorize(record, action="download")
        if record.status == "candidate":
            with pytest.raises(DomainError):
                authorize(record, action="download", accepted_license=True)
        else:
            authorize(record, action="download", accepted_license=True)
            with pytest.raises(DomainError):
                authorize(record, action="download", accepted_license=True, commercial=True)
        with pytest.raises(DomainError):
            authorize(record, action="activate", accepted_license=True)


def test_age_years_require_measured_calibration_and_bounds():
    from roop_3d.enhancers.styleganex_age import AgeCalibration, StyleGANEXAge
    from roop_3d.engine.stage16_pipeline import Stage16Options
    from pydantic import ValidationError
    with pytest.raises(RuntimeError, match="measured"):
        StyleGANEXAge(None).transform(None, None, age_offset=10)
    # This is an explicitly synthetic interpolation fixture, not real age calibration.
    calibration = AgeCalibration((-30, 0, 40), (1, 0, -2), "Synthetic interpolation fixture only")
    assert calibration.direction(-30) == 1 and calibration.direction(40) == -2
    assert calibration.direction(20) == -1 and calibration.direction(0) == 0
    for offset in (-31, 41, np.inf, np.nan):
        with pytest.raises(ValueError):
            calibration.direction(offset)
        with pytest.raises(ValidationError):
            Stage16Options(age_offset=offset)
    for weight in (-.1, 1.1, np.nan):
        with pytest.raises(ValidationError):
            Stage16Options(shape_retention_weight=weight)


def test_gaze_cannot_silently_use_eye_open_retargeter():
    from roop_3d.liveportrait_trt import LivePortraitTRT
    from roop_3d.flame_3d_engine import EyeGaze
    portrait = LivePortraitTRT(None, None, None, None, eye=object())
    with pytest.raises(RuntimeError, match="calibrated gaze"):
        portrait.retarget(None, None, source_gaze=EyeGaze(0, 0, 0, 0), driving_gaze=EyeGaze(.2, 0, -.2, 0))


def test_converter_keeps_source_immutable_and_rejects_existing_and_external(tmp_path):
    import onnx
    import hashlib
    from roop_3d.models.grid_sample import convert_grid_sample
    source, target = tmp_path / "reference.onnx", tmp_path / "converted.onnx"
    onnx.save(volume_graph(), source)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    assert len(convert_grid_sample(source, target)) == 1
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    with pytest.raises(ValueError, match="new derivative"):
        convert_grid_sample(source, target)
    with pytest.raises(ValueError, match="new derivative"):
        convert_grid_sample(source, source)
    external = volume_graph()
    tensor = onnx.numpy_helper.from_array(np.ones(1, np.float32), "external_fixture")
    onnx.external_data_helper.set_external_data(tensor, location="../unverified_weights.bin")
    tensor.ClearField("raw_data")
    external.graph.initializer.append(tensor)
    source.write_bytes(external.SerializeToString())
    with pytest.raises(ValueError, match="External tensor data"):
        convert_grid_sample(source, tmp_path / "unverified.onnx")
