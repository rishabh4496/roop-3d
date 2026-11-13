"""Defect #1: enhancer detail must survive the Stage 2 -> Stage 3 handoff, and UltraMax must never be bypassed by default."""
from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from roop_3d import hires_handoff as hh
from roop_3d.compositor.pipeline import CompositeOptions, FrameCompositor
from roop_3d.compositor.restoration import UltraMaxPipelineRestorer
from roop_3d.compositor.stream_restoration import (CODEFORMER_REFERENCE, DecoupledRestorer, RestorerSpec, gfpgan_spec,
                                                   restore_in_crop_space)
from roop_3d.compositor.temporal import TemporalSmoother
from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Settings
from roop_3d.core.workstation import WorkstationParameters
from roop_3d.vectorized_pipeline import composite_cpu, crop_matrix

lap_var = hh.laplacian_variance


def _texture(side: int, seed: int = 3, sigma: float = 0.9) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.integers(40, 215, (side, side, 3), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (0, 0), sigma)


# ------------------------------------------------------------------------------------------------ geometry
def test_rebased_matrices_are_centre_aligned_and_invertible():
    rng = np.random.default_rng(1)
    base = cv2.GaussianBlur(rng.random((128, 128, 3), dtype=np.float32), (0, 0), 2)
    scale = 4.0
    hr = cv2.resize(base, (512, 512), interpolation=cv2.INTER_CUBIC)
    angle = np.deg2rad(7)
    linear = 2.2 * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    inverse = np.concatenate((linear, [[300.0], [210.0]]), axis=1)           # crop -> frame
    valid = cv2.warpAffine(np.ones((128, 128), np.float32), inverse, (640, 480), flags=cv2.INTER_NEAREST) > 0
    direct = cv2.warpAffine(base, inverse, (640, 480), flags=cv2.INTER_CUBIC)
    rebased = cv2.warpAffine(hr, hh.scale_matrix_input(inverse, scale), (640, 480), flags=cv2.INTER_CUBIC)
    naive = inverse.copy()
    naive[:, :2] /= scale
    uncorrected = cv2.warpAffine(hr, naive, (640, 480), flags=cv2.INTER_CUBIC)
    error = float(np.abs(direct - rebased)[valid].mean())
    assert error < 0.004 and error * 2 < float(np.abs(direct - uncorrected)[valid].mean())
    # frame->crop and crop->frame re-basings are exact inverses of each other
    forward = hh.invert_affine_np(inverse)
    np.testing.assert_allclose(hh.invert_affine_np(hh.scale_matrix_output(forward, scale)),
                               hh.scale_matrix_input(inverse, scale), atol=1e-9)
    assert hh.hires_scale(1024, 128) == 8.0


def test_destination_face_side_is_measured_in_frame_pixels_not_crop_pixels():
    alpha = np.zeros((128, 128), np.float32)
    alpha[20:100, 30:110] = 1                       # an 80x80 face inside a 128 px crop
    assert hh.destination_face_side(((2.5, 0, 0), (0, 2.5, 0)), alpha) == pytest.approx(200.0)
    assert hh.destination_face_side(((0.5, 0, 0), (0, 0.5, 0)), alpha) == pytest.approx(40.0)
    assert hh.destination_face_side(((1, 0, 0), (0, 1, 0)), np.zeros((8, 8))) is None


# ------------------------------------------------------------------------------------------ GPU primitives
# Only the @cuda tests below need PyTorch. A module-level importorskip would also skip the CPU compositor sharpness
# tests (and hide them) in any environment without torch, so the guard is per test.
try:
    import torch
except ImportError:
    torch = None
cuda = pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="Real CUDA hardware (PyTorch with CUDA) required")


@cuda
def test_bicubic_roi_warp_matches_opencv_and_antialiases_shrinks():
    hr = _texture(512, sigma=5.0).astype(np.float32) / 255      # smooth: registration, not antialiasing, is under test
    matrix = hh.scale_matrix_input(((2.2 * np.cos(.1), -2.2 * np.sin(.1), 300.0), (2.2 * np.sin(.1), 2.2 * np.cos(.1), 210.0)), 4.0)
    tensor = torch.from_numpy(hr.transpose(2, 0, 1)[None]).cuda()
    roi, (x0, y0, x1, y1) = hh.warp_to_frame_roi(tensor, matrix, (480, 640))
    reference = cv2.warpAffine(hr, matrix, (640, 480), flags=cv2.INTER_CUBIC)[y0:y1, x0:x1]
    covered = reference.sum(-1) > 0.02
    got = roi[0].permute(1, 2, 0).cpu().numpy()
    assert np.abs(got - reference)[covered].mean() < 2e-3
    # A 5x shrink must be area-filtered, not point sampled: stripes average to mid-grey instead of aliasing.
    stripes = np.zeros((512, 512, 3), np.float32)
    stripes[:, ::2] = 1
    small, _ = hh.warp_to_frame_roi(torch.from_numpy(stripes.transpose(2, 0, 1)[None]).cuda(),
                                    ((0.2, 0, 100.0), (0, 0.2, 100.0)), (480, 640))
    assert float(small[..., 8:-8, 8:-8].std()) < 0.2 and abs(float(small[..., 8:-8, 8:-8].mean()) - 0.5) < 0.1


@cuda
@pytest.mark.parametrize("method", hh.COLOR_METHODS)
def test_skin_colour_transfer_matches_target_inside_skin_only(method):
    gen = torch.Generator(device="cuda").manual_seed(2)
    source = torch.rand((1, 3, 96, 96), device="cuda", generator=gen) * 0.35 + 0.25
    target = torch.rand((1, 3, 96, 96), device="cuda", generator=gen) * 0.2 + 0.55
    skin = torch.zeros((1, 1, 96, 96), device="cuda")
    skin[..., 20:76, 20:76] = 1
    out = hh.match_skin_colour(source, target, skin, method=method, strength=1.0)
    inside = skin.bool().expand_as(out)
    assert torch.equal(out[~inside], source[~inside]), "eyes / hair / background must not be recoloured"
    before = abs(float(source[inside].mean()) - float(target[inside].mean()))
    after = abs(float(out[inside].mean()) - float(target[inside].mean()))
    assert after < before * 0.1
    assert torch.equal(hh.match_skin_colour(source, target, skin * 0, method=method), source)   # empty class: identity


@cuda
def test_multiband_blend_keeps_fine_detail_and_zero_alpha_exact():
    gen = torch.Generator(device="cuda").manual_seed(5)
    detail = torch.rand((1, 3, 256, 256), device="cuda", generator=gen)
    background = torch.full_like(detail, 0.3)
    alpha = torch.zeros((1, 1, 256, 256), device="cuda")
    alpha[..., 64:192, 64:192] = 1
    alpha = torch.nn.functional.avg_pool2d(alpha, 9, 1, 4)
    out = hh.multiband_blend(detail, background, alpha, levels=4)
    zero = (alpha == 0).expand_as(out)
    assert torch.equal(out[zero], background[zero])
    core = torch.zeros_like(alpha, dtype=torch.bool)
    core[..., 90:166, 90:166] = True
    core = core.expand_as(out)
    high = lambda t: t - torch.nn.functional.avg_pool2d(t, 7, 1, 3)
    kept = (high(out) - high(detail))[core].abs().mean() / high(detail)[core].abs().mean()
    assert float(kept) < 0.02, "pores / eyelash-scale contrast must survive the blend untouched"


# ------------------------------------------------------------------- reference compositor (CPU) end to end
class _Hires512:
    """Restorer returning a 512 px face whose detail only exists above the 128 px crop's Nyquist."""
    def __init__(self):
        self.texture = _texture(512, seed=11, sigma=0.8)

    def restore_hires(self, crop, landmarks5=None, **_):
        return self.texture


class _ShrinksToCrop:
    """The pre-fix behaviour: the same 512 px result forced back to the 128 px swap crop."""
    def __init__(self):
        self.texture = _texture(512, seed=11, sigma=0.8)

    def restore_aligned(self, crop, landmarks5=None):
        return cv2.resize(self.texture, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_AREA)


def _compose(restorer, mode="standard", options=None, side=128, scale=2.0):
    frame = np.full((340, 340, 3), 96, np.uint8)
    inverse = np.array(((scale, 0, 40), (0, scale, 40)), np.float32)
    points = np.array(((70, 90), (230, 90), (150, 150), (95, 210), (205, 210)), np.float32)
    replacement = SimpleNamespace(track_id=1, image=np.full((side, side, 3), 130, np.uint8), inverse=inverse,
                                  landmarks5=((points - 40) / scale).astype(np.float32))
    robust = options is not None and options.occlusion_mode == "robust"
    if robust:   # same analyser doubles the occlusion suite uses; parser trust is earned over three frames
        from roop_3d.tests.test_occlusion_compositor import _StubNet, _StubParser, _analyser, _visible
        compositor = FrameCompositor(parser=_StubParser(), temporal=TemporalSmoother())
        compositor._analyzer = _analyser(compositor.parser, _StubNet("xseg_3.onnx", _visible()),
                                         _StubNet("face_occluder.onnx", _visible()))
        points = np.array(((80, 80), (256, 80), (256, 256), (80, 256)), np.float32)
    else:
        compositor = FrameCompositor(parser=SimpleNamespace(predict=lambda crop: None), temporal=TemporalSmoother())
    compositor.restorer = restorer
    opts = options or CompositeOptions(restoration_blend=1.0, color_strength=0, feather_px=2, occlusion_mode=mode)
    for index in range(3 if robust else 1):
        out = compositor.compose(frame, (replacement,), {1: points}, frame_index=index, options=opts)
    return frame, out


def _inner_sharpness(frame, out):
    return lap_var(out[110:250, 110:250])


def test_reference_compositor_warps_the_512_face_straight_into_the_frame():
    _, hires = _compose(_Hires512())
    _, legacy = _compose(_ShrinksToCrop())
    ratio = _inner_sharpness(None, hires) / max(_inner_sharpness(None, legacy), 1e-6)
    assert ratio > 3.0, f"expected >=3x more high-frequency energy than the shrunk-to-crop path, got {ratio:.2f}x"


def test_reference_compositor_robust_path_keeps_detail_too():
    from roop_3d.compositor.pipeline import RobustOptions
    options = CompositeOptions(restoration_blend=1.0, color_strength=0, feather_px=2, occlusion_mode="robust",
                               robust=RobustOptions(local_illumination=0, texture_strength=0, sharpness_matching=False))
    _, hires = _compose(_Hires512(), options=options)
    _, legacy = _compose(_ShrinksToCrop(), options=options)
    assert _inner_sharpness(None, hires) > 3.0 * _inner_sharpness(None, legacy)


def test_unenhanced_swap_has_far_less_detail_than_either_enhancer_path():
    _, plain = _compose(_Hires512(), options=CompositeOptions(restoration_blend=0.0, color_strength=0, feather_px=2))
    _, hires = _compose(_Hires512())
    assert _inner_sharpness(None, hires) > 3.0 * max(_inner_sharpness(None, plain), 1e-6)


def test_restorer_failure_is_loud_never_a_silent_unenhanced_frame(caplog):
    class Broken:
        def restore_hires(self, crop, landmarks5=None, **_):
            raise RuntimeError("provider mismatch: expected float16 got float32")
    with pytest.raises(DomainError) as error, caplog.at_level("CRITICAL"):
        _compose(Broken())
    assert error.value.code == "RESTORER_FAILED"
    assert any("provider mismatch" in record.getMessage() for record in caplog.records)


# -------------------------------------------------------------------------------------- bypass is opt-in only
def test_no_defaults_bypass_ultramax():
    assert CompositeOptions().ultramax_small_face_mode == "off"
    assert WorkstationParameters().ultramax_small_face_mode == "off"
    assert Settings().ultramax_native_bypass is False
    from roop_3d.render_policy import RenderPolicy
    assert RenderPolicy().ultramax_native_bypass is False
    assert DecoupledRestorer(lambda: None, RestorerSpec("ultramax", "x", CODEFORMER_REFERENCE)).bypass is False


def test_ultramax_runs_on_a_tiny_face_by_default_and_skips_only_when_asked(monkeypatch):
    from roop_3d.compositor.restoration import CodeFormerRestorer
    # The real lazy constructor accepts the optional spec and initializes timing/telemetry state without loading a model.
    calls = []
    monkeypatch.setattr(CodeFormerRestorer, "restore_hires",
                        lambda self, image, landmarks5=None, **_: calls.append(1) or cv2.resize(image, (512, 512)))
    ultra = UltraMaxPipelineRestorer(None, None)
    image = _texture(128)
    assert ultra.small_face_mode == "off"
    assert ultra.restore_hires(image, None, native_face_size_px=40).shape == (512, 512, 3) and calls == [1]
    ultra.small_face_mode = "bypass"
    assert ultra.restore_hires(image, None, native_face_size_px=40) is None
    assert ultra.restore_hires(image, None, native_face_size_px=900) is not None


# ------------------------------------------------------------------------- host streaming restorer + composite
def test_host_restorer_returns_native_resolution_and_composite_cpu_keeps_it():
    spec = RestorerSpec("codeformer", "fixture", CODEFORMER_REFERENCE.copy(), 512)
    detail = _texture(512, seed=21, sigma=0.8)
    pool = DecoupledRestorer(lambda: SimpleNamespace(restore=lambda crop: detail), spec, workers=1, enhancer_blend=1.0)
    points = (CODEFORMER_REFERENCE / 4).astype(np.float32)
    crop = _texture(128, seed=4, sigma=2.0)
    try:
        restored = pool.submit(crop, points, np.array([0, 0, 900, 900]), 1).result()
        assert restored.shape == (512, 512, 3) and pool.hires_scale(128) == 4.0
        half = DecoupledRestorer(lambda: SimpleNamespace(restore=lambda c: detail), spec, enhancer_blend=0.0)
        np.testing.assert_array_equal(half.submit(crop, points, np.array([0, 0, 900, 900]), 1).result(),
                                      cv2.resize(crop, (512, 512), interpolation=cv2.INTER_CUBIC))
        half.close()
    finally:
        pool.close()
    frame = np.full((300, 300, 3), 90, np.uint8)
    matrix = crop_matrix(np.array(((100, 110), (180, 110), (140, 150), (110, 190), (170, 190)), np.float32), 128)
    hi = np.ascontiguousarray(restored[:, :, ::-1].transpose(2, 0, 1)[None]).astype(np.float32) / 255
    lo_img = cv2.resize(restored, (128, 128), interpolation=cv2.INTER_AREA)
    lo = np.ascontiguousarray(lo_img[:, :, ::-1].transpose(2, 0, 1)[None]).astype(np.float32) / 255
    out_hi, out_lo = composite_cpu(frame, hi, [matrix]), composite_cpu(frame, lo, [matrix])
    assert lap_var(out_hi[80:220, 80:220]) > 3.0 * lap_var(out_lo[80:220, 80:220])


def test_gfpgan_has_its_own_engine_and_rgb_minus_one_one_contract():
    spec = gfpgan_spec()
    assert (spec.name, spec.model_name, spec.size, spec.normalization) == ("gfpgan", "GFPGANv1.4.onnx", 512, "minus_one_one")
    assert gfpgan_spec().model_name != "gpen_bfr_512.onnx" and spec.model_name != "CodeFormerv0.1.onnx"
    # restore_in_crop_space puts the restored 512 face back in crop space at 512 (not 128)
    out, mask = restore_in_crop_space(lambda crop: crop, _texture(128), (CODEFORMER_REFERENCE / 4).astype(np.float64), spec)
    assert out.shape == (512, 512, 3) and mask.shape == (512, 512)


# ------------------------------------------------------------------------ resident GPU compositor and adapter
@cuda
def test_cuda_adapter_identity_restorer_is_geometrically_exact_at_512_and_1024(monkeypatch):
    import roop_3d.engine.runtime as runtime
    import roop_3d.gpu_compositor as gc
    from roop_3d.compositor.stream_restoration import OnnxFaceRestorer
    from roop_3d.faceset.recognizer import ARCFACE_REFERENCE_5PTS

    current = {"size": 512}

    class FakeSession:
        def __init__(self):
            n = current["size"]
            self.inputs = [SimpleNamespace(name="x", shape=[1, 3, n, n], type="tensor(float)"),
                           SimpleNamespace(name="w", shape=[1], type="tensor(float)")]
            self.outputs = [SimpleNamespace(name="y", shape=[1, 3, n, n], type="tensor(float)")]
        get_inputs = lambda self: self.inputs
        get_outputs = lambda self: self.outputs

    class FakeModel:
        def __init__(self, *_a, **_k):
            self.session, self._identity, self.path = FakeSession(), 1, "fake"
        _stat = lambda self: 1
        release = lambda self: None

    class FakeRunner:
        seen_w = []
        def __init__(self, *_a): pass
        def run(self, feeds, outputs):
            (name, (shape, dtype)), = outputs.items()
            FakeRunner.seen_w.append(float(feeds["w"]))
            value = feeds["x"]
            if tuple(shape[2:]) != tuple(value.shape[2:]):
                value = torch.nn.functional.interpolate(value, shape[2:], mode="bicubic", align_corners=False)
            return {name: value.to(dtype).contiguous()}

    monkeypatch.setattr(runtime, "OrtModel", FakeModel)
    monkeypatch.setattr(gc, "TorchIOBinding", FakeRunner)
    monkeypatch.setattr(gc, "stream_session", lambda session, stream: session)
    face = SimpleNamespace(box=np.array((0, 0, 512, 512)), landmarks5=((ARCFACE_REFERENCE_5PTS + [8, 0]) * 4).astype(np.float32))
    image = torch.from_numpy(cv2.GaussianBlur(np.random.default_rng(8).random((512, 512, 3), dtype=np.float32), (0, 0), 3)
                             .transpose(2, 0, 1)[None]).cuda()
    stream = torch.cuda.Stream()
    for size in (512, 1024):
        current["size"] = size
        spec = RestorerSpec("codeformer", "fixture", CODEFORMER_REFERENCE.copy(), size)
        pool = DecoupledRestorer(lambda spec=spec: OnnxFaceRestorer(None, None, spec), spec, enhancer_blend=1.0,
                                 fidelity_weight=0.25)
        with torch.cuda.stream(stream):
            restore = pool.cuda_adapter(stream)
            out = restore(image, face)
            pool.fidelity_weight = 0.9
            restore(image, face)
        stream.synchronize()
        assert out.shape == (1, 3, size, size), "restorer output keeps the enhancer's native resolution"
        expected = image if size == 512 else torch.nn.functional.interpolate(image, (size, size), mode="bicubic", align_corners=False)
        centre = slice(size // 4, 3 * size // 4)
        assert float((out - expected)[..., centre, centre].abs().mean()) < 4e-3
        assert FakeRunner.seen_w[-2:] == pytest.approx([0.25, 0.9]), "fidelity slider is bound per call"
        pool.close()


@cuda
def test_resident_compositor_keeps_1024_px_detail_and_color_matches_skin(monkeypatch):
    from roop_3d.faceset.recognizer import ARCFACE_REFERENCE_5PTS
    from roop_3d.gpu_compositor import CompositorOptions, GpuCompositor, TripleBufferedCudaPipeline
    from roop_3d.optimized_prepass import LockedFace
    detail = torch.from_numpy(_texture(1024, seed=13, sigma=0.8).astype(np.float32)[:, :, ::-1].transpose(2, 0, 1)[None] / 255).cuda()
    def build(restore):
        def factory(stream):
            def mask(image, face):
                ones = torch.ones_like(image[:, :1])
                return ones, ones.clone(), ones.clone()
            return GpuCompositor(lambda image, face: image, restorer=restore, masker=mask,
                                 options=CompositorOptions(color_strength=0.8, color_method="reinhard_ycbcr",
                                                           laplacian_blend_weight=1.0, blend_levels=4))
        return factory
    frame = np.full((512, 512, 3), 100, np.uint8)
    frame[:, :, 2] = 150                                                      # a warm target skin tone
    face = LockedFace(0, 0, np.array((0, 0, 512, 512)), ((ARCFACE_REFERENCE_5PTS + [8, 0]) * 4).astype(np.float32))
    native = TripleBufferedCudaPipeline(build(lambda image, f: detail)).process(frame, [face])
    shrunk = TripleBufferedCudaPipeline(build(lambda image, f: torch.nn.functional.interpolate(
        detail, (128, 128), mode="area"))).process(frame, [face])
    hi, lo = lap_var(native[150:350, 150:350]), lap_var(shrunk[150:350, 150:350])
    assert hi > 3.0 * lo, f"1024 px detail must reach the frame (got {hi / max(lo, 1e-9):.1f}x)"
    skin = native[200:300, 200:300].reshape(-1, 3).mean(0)
    assert abs(skin[2] - 150) < 25 and abs(skin[1] - 100) < 25, "enhanced skin tone follows the target skin"


def test_aligned_restorers_fail_closed_without_landmarks_instead_of_running_unaligned():
    from roop_3d.compositor.restoration import CodeFormerRestorer, GfpganRestorer
    for cls in (CodeFormerRestorer, GfpganRestorer):
        restorer = object.__new__(cls)
        restorer.spec = gfpgan_spec()
        restorer.runner = SimpleNamespace(restore=lambda *_: pytest.fail("model must not run on an unaligned crop"))
        with pytest.raises(DomainError) as error:
            restorer.restore_hires(_texture(128), None)
        assert error.value.code == "RESTORER_LANDMARKS_REQUIRED" and error.value.status == 422
