"""Regression tests for the Nov 12 2026 fixes: concurrent-stage CUDA graphs, trim-window work, warm preview sessions,
dense-3D matte at turned heads, GPU geometry selection and the terminal scan line.

Everything here is hermetic: no model, GPU or media file is needed.
"""
from __future__ import annotations

import io
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from roop_3d.compositor.masks import (JAW_CUTOFF_MIN_TURN_RATIO, confident_forbidden, fill_mask_holes, turn_ratio)
from roop_3d.core import ort_session
from roop_3d.engine.pipeline_runner import windowed_frames
from roop_3d.telemetry.pinokio_terminal import PinokioTerminalHUD, ResourceReading, ResourceSampler


# ------------------------------------------------------------------------------------------------ CUDA graphs
def test_cuda_graph_replay_is_off_unless_explicitly_enabled(monkeypatch):
    """Capture on one stage thread while another syncs a stream is CUDA 906 -> a failed swap -> a rebuild mid-flight."""
    monkeypatch.delenv("ROOP3D_CUDA_GRAPH", raising=False)
    assert ort_session.cuda_graphs_enabled() is False
    monkeypatch.setenv("ROOP3D_CUDA_GRAPH", "1")
    assert ort_session.cuda_graphs_enabled() is True


# ------------------------------------------------------------------------------------------------ trim window
class _Frames:
    def __init__(self, seconds):
        self.items = [SimpleNamespace(index=i, timestamp=SimpleNamespace(seconds=Fraction(s))) for i, s in enumerate(seconds)]
        self.closed = False

    def __iter__(self):
        return iter(self.items)

    def close(self):
        self.closed = True


def test_windowed_frames_keeps_the_writers_half_open_interval_and_closes_the_decoder():
    source = _Frames([Fraction(i, 30) for i in range(90)])          # 3 s at 30 fps
    kept = list(windowed_frames(source, Fraction(1), Fraction(2)))
    assert [f.index for f in kept] == list(range(30, 60)), "start inclusive, end exclusive, indices stay the source's own"
    assert source.closed, "stopping at the end must release the decoder"


def test_windowed_frames_is_a_noop_without_a_trim():
    source = _Frames([0, 1, 2])
    assert windowed_frames(source, None, None) is source


def test_windowed_frames_open_ended_start_only():
    source = _Frames([Fraction(i, 10) for i in range(10)])
    assert [f.index for f in windowed_frames(source, Fraction(1, 2), None)] == [5, 6, 7, 8, 9]


# ------------------------------------------------------------------------------------------------ matte helpers
def test_turn_ratio_separates_frontal_from_profile_landmarks():
    frontal = [[100, 100], [160, 100], [130, 130], [105, 170], [155, 170]]
    profile = [[100, 100], [104, 100], [130, 130], [110, 170], [125, 170]]
    assert turn_ratio(frontal) > 0.8 and turn_ratio(frontal) >= JAW_CUTOFF_MIN_TURN_RATIO
    assert turn_ratio(profile) < JAW_CUTOFF_MIN_TURN_RATIO
    assert turn_ratio(None) is None and turn_ratio([[0, 0]]) is None
    assert turn_ratio([[1, 1]] * 5) is None, "coincident points have no scale"


def test_fill_mask_holes_closes_eye_sized_gaps_but_not_borders_or_large_voids():
    mask = np.zeros((100, 100), np.float32)
    mask[10:90, 20:80] = 1.0
    mask[30:36, 40:46] = 0.0                      # an eye socket
    filled = fill_mask_holes(mask)
    assert filled[32, 42] > 0.9, "enclosed gap must be closed"
    big = np.zeros((100, 100), np.float32)
    big[5:95, 5:95] = 1.0
    big[25:75, 25:75] = 0.0                       # a 25 % void: a real gap (hand, microphone), not a mesh hole
    assert fill_mask_holes(big)[50, 50] < 0.1
    open_gap = np.ones((100, 100), np.float32)
    open_gap[40:60, 0:30] = 0.0                    # touches the border
    assert fill_mask_holes(open_gap)[50, 5] < 0.1


def test_confident_forbidden_drops_the_weak_fringe_and_keeps_strong_evidence():
    f = np.array([[0.0, 0.3, 0.49, 0.7, 0.86, 1.0]], np.float32)
    g = confident_forbidden(f)
    assert np.all(g[0, :3] == 0.0), "a diffuse 0.2-0.5 parser opinion on a synthesised face must not erode the swap"
    assert g[0, 3] > 0.0 and g[0, 4] == 1.0 and g[0, 5] == 1.0 and np.all(np.diff(g[0]) >= 0)


# ------------------------------------------------------------------------------------------------ preview session pool
class _FakeManager(ort_session.SessionManager):
    created = 0

    def _create(self, model_path, **kwargs):          # no ORT: count how often a session would really be built
        type(self).created += 1
        return ort_session.SessionInfo(object(), "TensorrtExecutionProvider", ("TensorrtExecutionProvider",))


@pytest.fixture()
def model_file(tmp_path):
    path = tmp_path / "m.onnx"
    path.write_bytes(b"x" * 64)
    ort_session.evict_pooled_sessions()
    _FakeManager.created = 0
    yield path
    ort_session.evict_pooled_sessions()


def test_sessions_outside_a_preview_are_never_pooled(model_file):
    manager = _FakeManager(cache_path=model_file.parent)
    first, second = manager.create(model_file, model_name="m"), manager.create(model_file, model_name="m")
    assert first is not second and _FakeManager.created == 2 and ort_session.pooled_session_count() == 0


def test_a_preview_reuses_the_session_it_built_last_time_and_eviction_empties_the_pool(model_file):
    manager = _FakeManager(cache_path=model_file.parent)
    with ort_session.pooled_sessions():
        first = manager.create(model_file, model_name="m")
    with ort_session.pooled_sessions():
        second = _FakeManager(cache_path=model_file.parent).create(model_file, model_name="m")
    assert first is second and _FakeManager.created == 1
    assert ort_session.evict_pooled_sessions() == 1 and ort_session.pooled_session_count() == 0
    with ort_session.pooled_sessions():
        manager.create(model_file, model_name="m")
    assert _FakeManager.created == 2


def test_a_changed_model_file_is_never_served_from_the_pool(model_file):
    manager = _FakeManager(cache_path=model_file.parent)
    with ort_session.pooled_sessions():
        manager.create(model_file, model_name="m")
        model_file.write_bytes(b"y" * 80)            # a different size and mtime
        manager.create(model_file, model_name="m")
    assert _FakeManager.created == 2


def test_the_pool_can_be_switched_off(model_file, monkeypatch):
    monkeypatch.setenv("ROOP3D_PREVIEW_SESSION_POOL", "0")
    manager = _FakeManager(cache_path=model_file.parent)
    with ort_session.pooled_sessions():
        manager.create(model_file, model_name="m")
        manager.create(model_file, model_name="m")
    assert _FakeManager.created == 2


def test_a_job_taking_the_gpu_releases_the_warm_sessions_and_the_warm_head_engine(model_file):
    from roop_3d.reconstruction.neural_face import WARM_ENGINE
    from roop_3d.server.engine_gate import EngineGate
    released = []
    engine = SimpleNamespace(release=lambda: released.append(True))
    WARM_ENGINE.put(("k",), engine)
    manager = _FakeManager(cache_path=model_file.parent)
    with ort_session.pooled_sessions():
        manager.create(model_file, model_name="m")
    assert ort_session.pooled_session_count() == 1
    gate = EngineGate()
    with gate.owner("RENDER_JOB"):
        assert ort_session.pooled_session_count() == 0, "a render must not share VRAM with idle preview sessions"
        assert released == [True] and WARM_ENGINE.take(("k",)) is None


# ------------------------------------------------------------------------------------------------ 3D geometry provider
def test_geometry_graphs_use_cuda_on_a_12gb_card_and_the_cpu_otherwise(monkeypatch):
    from roop_3d.reconstruction import morphable_backbone as backbone
    from roop_3d.engine import runtime as runtime_module
    seen = []

    def fake_load(catalog, model_id, path, config):
        seen.append(config.provider)
        if config.provider == "CUDAExecutionProvider" and getattr(config, "fail_cuda", False):
            raise RuntimeError("no cuda")
        return object()
    monkeypatch.setattr("roop_3d.engine.volumetric.swapper_engine.load_local_session", fake_load)
    monkeypatch.delenv("ROOP3D_GEOMETRY_PROVIDER", raising=False)

    def config(profile, provider="TensorrtExecutionProvider"):
        return SimpleNamespace(provider=provider, hardware_profile=profile, precision_policy="fp16")
    # dataclasses.replace needs a real dataclass: use RuntimeConfig for the cases that reach replace()
    from roop_3d.engine.runtime import RuntimeConfig
    twelve = RuntimeConfig(provider="TensorrtExecutionProvider", hardware_profile="12gb")
    six = RuntimeConfig(provider="TensorrtExecutionProvider", hardware_profile="6gb")
    assert backbone.load_geometry_session(None, "x", "p", twelve)[1] == "CUDAExecutionProvider"
    assert backbone.load_geometry_session(None, "x", "p", six)[1] == "CPUExecutionProvider"
    monkeypatch.setenv("ROOP3D_GEOMETRY_PROVIDER", "cpu")
    assert backbone.load_geometry_session(None, "x", "p", twelve)[1] == "CPUExecutionProvider"
    monkeypatch.setenv("ROOP3D_GEOMETRY_PROVIDER", "cuda")
    assert backbone.load_geometry_session(None, "x", "p", six)[1] == "CUDAExecutionProvider"
    assert seen[0] == "CUDAExecutionProvider"


def test_a_failing_gpu_geometry_session_falls_back_to_the_cpu(monkeypatch):
    from roop_3d.reconstruction import morphable_backbone as backbone
    from roop_3d.engine.runtime import RuntimeConfig

    def fake_load(catalog, model_id, path, config):
        if config.provider == "CUDAExecutionProvider":
            raise RuntimeError("cuda init failed")
        return "cpu-session"
    monkeypatch.setattr("roop_3d.engine.volumetric.swapper_engine.load_local_session", fake_load)
    monkeypatch.delenv("ROOP3D_GEOMETRY_PROVIDER", raising=False)
    session, provider = backbone.load_geometry_session(None, "x", "p", RuntimeConfig(provider="TensorrtExecutionProvider", hardware_profile="12gb"))
    assert (session, provider) == ("cpu-session", "CPUExecutionProvider")


# ------------------------------------------------------------------------------------------------ auto 3D limits
def test_automatic_3d_switch_limits_default_and_can_be_changed(monkeypatch):
    from roop_3d.engine.volumetric.session import auto_3d_limits
    for name in ("ROOP3D_AUTO_3D_YAW", "ROOP3D_AUTO_3D_PITCH"):
        monkeypatch.delenv(name, raising=False)
    assert auto_3d_limits() == (30.0, 25.0)
    monkeypatch.setenv("ROOP3D_AUTO_3D_YAW", "180")
    monkeypatch.setenv("ROOP3D_AUTO_3D_PITCH", "junk")
    assert auto_3d_limits() == (180.0, 25.0)


# ------------------------------------------------------------------------------------------------ terminal
class _Clock:
    now = 100.0

    def __call__(self):
        return self.now


def _hud(out):
    sampler = ResourceSampler(gpu=lambda: None)
    sampler.sample_once = lambda **_kwargs: ResourceReading()  # type: ignore[method-assign]
    return PinokioTerminalHUD(out, color=False, in_place=False, clock=_Clock(), sampler=sampler, width=lambda: 160,
                              min_redraw=0.0, plain_interval=0.0).arm()


def test_the_terminal_shows_the_angle_scan_phase_with_its_own_rate_and_the_render_after_it():
    out, clock = io.StringIO(), _Clock()
    h = _hud(out)
    h.clock = clock
    h.job_started("e" * 32, swap_mode="full_craniofacial", total_frames=180, report=False)
    for frame in (30, 60, 90, 120, 150, 180):
        clock.now += 1.0
        h.frame("e" * 32, frame=frame, total=180, stage="angle_scan")
    text = out.getvalue()
    assert "[SCAN]" in text, "the pre-pass announces itself"
    assert "SCAN" in text.split("[SCAN]")[1] and "100.0%" in text
    out.truncate(0), out.seek(0)
    clock.now += 1.0
    h.frame("e" * 32, frame=1, total=180, stage="render")
    clock.now += 1.0
    h.frame("e" * 32, frame=11, total=180, stage="render")
    render_text = out.getvalue()
    assert render_text and " SCAN" not in render_text, "the render line must not keep the scan label"
    assert "1/180" in render_text or "11/180" in render_text


# ------------------------------------------------------------------------------------------------ parser cache / prefetch
def _fake_parser():
    from roop_3d.compositor.masks import BisenetFaceParser
    calls = {"n": 0}

    class Model:
        def run(self, inputs):
            calls["n"] += 1
            rng = np.random.default_rng(int(inputs["input"].sum() * 1000) % (2 ** 32))
            return {"output": rng.standard_normal((1, 19, 512, 512)).astype(np.float32)}

    parser = BisenetFaceParser(catalog=object())
    parser.model = Model()
    return parser, calls


def test_parallel_softmax_is_bit_identical_to_the_single_thread_formula():
    from roop_3d.compositor.masks import softmax_last_axis
    logits = np.random.default_rng(1).standard_normal((300, 300, 19)).astype(np.float32) * 4
    exp = np.exp(np.clip(logits - logits.max(-1, keepdims=True), -80, 0))
    expected = (exp / np.maximum(exp.sum(-1, keepdims=True), 1e-12)).astype(np.float32)
    assert np.array_equal(softmax_last_axis(logits), expected)
    small = logits[:10]
    assert np.allclose(softmax_last_axis(small).sum(-1), 1.0, atol=1e-5), "below the band threshold it runs inline"


def test_the_parser_runs_once_per_distinct_crop_and_shares_a_read_only_result():
    parser, calls = _fake_parser()
    a = np.full((96, 96, 3), 40, np.uint8)
    b = np.full((96, 96, 3), 90, np.uint8)
    first, again = parser.predict(a), parser.predict(a.copy())
    assert first is again and calls["n"] == 1, "the compositor asks for the same target crop twice per frame"
    assert not first.flags.writeable, "a shared result must not be editable in place"
    assert parser.predict(b) is not first and calls["n"] == 2


def test_a_prefetched_parse_is_consumed_by_the_composite_stage_and_the_lead_is_bounded():
    parser, calls = _fake_parser()
    crops = [np.full((64, 64, 3), 10 * i + 5, np.uint8) for i in range(parser.PREFETCH_LEAD_CROPS + 3)]
    for crop in crops[:parser.PREFETCH_LEAD_CROPS]:
        parser.predict(crop, prefetch=True)
    assert parser.prefetch_pressure() == parser.PREFETCH_LEAD_CROPS
    before = calls["n"]
    parser.predict(crops[0])                      # the composite stage reads it
    assert calls["n"] == before, "a prefetch hit costs no inference"
    assert parser.prefetch_pressure() == parser.PREFETCH_LEAD_CROPS - 1, "consuming frees prefetch headroom"


def test_the_parser_cache_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("ROOP3D_PARSER_CACHE", "0")
    parser, calls = _fake_parser()
    image = np.full((64, 64, 3), 77, np.uint8)
    parser.predict(image), parser.predict(image)
    assert calls["n"] == 2


def test_the_compositor_prefetch_only_runs_for_the_standard_occlusion_path():
    from roop_3d.compositor.pipeline import CompositeOptions, FrameCompositor
    parser, calls = _fake_parser()
    compositor = FrameCompositor.__new__(FrameCompositor)
    compositor.parser = parser
    crop = np.full((64, 64, 3), 120, np.uint8)
    replacement = SimpleNamespace(image=crop, inverse=np.array([[1, 0, 0], [0, 1, 0]], np.float32))
    frame = np.full((64, 64, 3), 130, np.uint8)
    assert compositor.prefetch_parser(frame, [replacement], CompositeOptions(occlusion_mode="robust")) == 0
    assert calls["n"] == 0
    assert compositor.prefetch_parser(frame, [replacement], CompositeOptions()) == 2 and calls["n"] == 2


# ------------------------------------------------------------------------------------------------ chin clamp
def test_chin_line_clamp_cuts_below_the_chin_at_any_yaw_and_leaves_the_face_alone():
    from roop_3d.compositor.masks import chin_line_clamp
    mask = np.ones((200, 200), np.float32)
    frontal = [[70, 70], [130, 70], [100, 100], [78, 130], [122, 130]]       # eye-mouth distance 60, upright: cut from y~177
    out = chin_line_clamp(mask, frontal)
    assert out[100, 100] == 1.0 and out[125, 100] == 1.0, "the face and the mouth region are untouched"
    assert out[190, 100] == 0.0 and out[150, 100] == 1.0, "the neck below the chin is removed, the jaw is kept"
    profile = [[100, 70], [104, 70], [112, 100], [105, 130], [120, 130]]
    assert chin_line_clamp(mask, profile)[195, 110] == 0.0, "the five-point line still works when the eyes collapse"
    assert np.array_equal(chin_line_clamp(mask, None), mask) and np.array_equal(chin_line_clamp(mask, [[1, 1]]), mask)


# ------------------------------------------------------------------------------------------------ trimmed audio sync
class _FakeAudioContainer:
    """Just enough of an av container for ``decode_window``: frames with chosen pts over one known signal."""

    def __init__(self, frames):
        self.streams = [SimpleNamespace(), SimpleNamespace()]
        self._frames = frames

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def decode(self, stream):
        return iter(self._frames)


def _frames_of(signal, starts, rate=44100):
    av = pytest.importorskip("av")
    frames = []
    for start in starts:
        frame = av.AudioFrame.from_ndarray(signal[start:start + 1024][None, :], format="flt", layout="mono")
        frame.sample_rate, frame.pts, frame.time_base = rate, start, Fraction(1, rate)
        frames.append(frame)
    return frames


def test_decode_window_follows_each_frames_own_pts_when_the_first_aac_frame_reports_an_odd_time(monkeypatch):
    """Regression: a source whose first decoded AAC frame arrives at pts 512 (its next frame at 1024) was laid out contiguously
    from that frame, shifting everything after it by 512 samples (11.6 ms) so every trimmed job failed OUTPUT_AV_DESYNC."""
    from roop_3d.media import av_sync
    rng = np.random.default_rng(3)
    signal = (rng.standard_normal(44100 * 4) * 0.2).astype(np.float32)
    regular = _frames_of(signal, range(0, 44100 * 3, 1024))
    quirky = _frames_of(signal, [512, *range(1024, 44100 * 3, 1024)])
    containers = iter([_FakeAudioContainer(regular), _FakeAudioContainer(quirky)])
    monkeypatch.setattr(av_sync, "_open_audio_source", lambda path: next(containers))
    reference = av_sync.decode_window("regular", 1, Fraction(0), 2.5)
    odd = av_sync.decode_window("quirky", 1, Fraction(0), 2.5)
    lag = av_sync.measure_lag(reference, odd)
    assert lag.measurable and abs(lag.lag_seconds) <= 0.0005, lag


# ------------------------------------------------------------------------------------------------ admission / probes / hashing
def _fake_hardware(free_budget):
    from roop_3d.core.hardware import GPU, Hardware
    return Hardware(timestamp="t", os={"system": "Windows", "release": "11", "machine": "AMD64"}, logical_cpu_count=8,
                    advertised_providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
                    gpus=[GPU(index=0, name="RTX", total_vram_mib=12282, free_vram_mib=free_budget + 768,
                              driver_version="1", effective_vram_budget_mib=free_budget)])


def test_admission_counts_idle_warm_sessions_as_free_vram_instead_of_falling_back_to_the_cpu(monkeypatch):
    """Regression: with ~7 GB held by warm preview sessions, free VRAM looked < 4 GB and the whole preview ran on the CPU."""
    from roop_3d.core.settings import Settings
    from roop_3d.engine.vram_governor import VRAMGovernor
    settings = Settings(hardware_profile="12gb", preferred_provider="auto")
    governor = VRAMGovernor(hardware_probe=lambda _s: _fake_hardware(3000))
    monkeypatch.setattr(ort_session, "reclaimable_vram_mib", lambda: 0)
    lease = governor.try_acquire("a", settings)
    assert lease.admission.provider == "CPUExecutionProvider" and "below" in lease.admission.reason
    lease.release()
    monkeypatch.setattr(ort_session, "reclaimable_vram_mib", lambda: 4000)
    lease = governor.try_acquire("b", settings)
    assert lease.admission.provider == "CUDAExecutionProvider", lease.admission.reason


def test_reclaimable_vram_sums_the_pooled_sessions_and_clears_on_eviction(model_file):
    manager = _FakeManager(cache_path=model_file.parent)
    with ort_session.pooled_sessions():
        manager.create(model_file, model_name="m")
    key = next(iter(ort_session._POOL))
    ort_session._POOL_VRAM[key] = 1234
    assert ort_session.reclaimable_vram_mib() == 1234
    ort_session.evict_pooled_sessions()
    assert ort_session.reclaimable_vram_mib() == 0


def test_hardware_probe_is_cached_but_free_vram_is_always_read_live(monkeypatch):
    from roop_3d.core import hardware
    from roop_3d.core.settings import Settings
    calls = {"n": 0}

    def fake_uncached(settings):
        calls["n"] += 1
        return _fake_hardware(5000)

    class FakeNvml:
        available = True
        used = 4000

        def __init__(self, *_):
            pass

        def sample(self):
            return {"vram_total_mib": 12282, "vram_used_mib": FakeNvml.used}

    monkeypatch.setattr(hardware, "_probe_hardware_uncached", fake_uncached)
    monkeypatch.setattr("roop_3d.async_pipeline.hwmon.Nvml", FakeNvml)
    monkeypatch.setattr(hardware, "_PROBE_CACHE", {})
    settings = Settings()
    first = hardware.probe_hardware(settings)
    FakeNvml.used = 9000
    second = hardware.probe_hardware(settings)
    assert calls["n"] == 1, "the second call must not spawn another dozen subprocesses"
    assert first.gpus[0].free_vram_mib != second.gpus[0].free_vram_mib == 12282 - 9000, "free VRAM is re-read, never stale"
    monkeypatch.setenv("ROOP3D_HARDWARE_PROBE_TTL", "0")
    hardware.probe_hardware(settings)
    assert calls["n"] == 2, "TTL 0 always probes"


def test_unchanged_files_keep_their_verified_hash_and_changed_files_do_not(tmp_path, monkeypatch):
    import hashlib
    from roop_3d.audit import isolation
    from roop_3d.core.errors import DomainError
    from roop_3d.models import verifier
    monkeypatch.setattr(isolation, "_HASH_MEMO", {})
    monkeypatch.setattr(verifier, "_VERIFIED", {})
    path = tmp_path / "w.onnx"
    path.write_bytes(b"a" * 5000)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    reads = {"n": 0}
    original = isolation._sha256_file_uncached
    monkeypatch.setattr(isolation, "_sha256_file_uncached", lambda p: (reads.__setitem__("n", reads["n"] + 1), original(p))[1])
    assert isolation.sha256_file(path)[1] == expected and isolation.sha256_file(path)[1] == expected
    assert reads["n"] == 1, "an unchanged file is hashed once"
    with verifier.verified_stream(path, expected, 5000):
        pass
    with verifier.verified_stream(path, expected, 5000):
        pass
    path.write_bytes(b"b" * 5000)                 # same size, new bytes and mtime
    with pytest.raises(DomainError):
        with verifier.verified_stream(path, expected, 5000):
            pass
    assert isolation.sha256_file(path)[1] != expected and reads["n"] == 2
    monkeypatch.setenv("ROOP3D_VERIFY_MEMO_SECONDS", "0")
    isolation.sha256_file(path)
    isolation.sha256_file(path)
    assert reads["n"] == 4, "memo seconds 0 hashes every time"


def test_the_dense_3d_path_hands_strongly_rolled_heads_to_the_standard_swap(monkeypatch):
    from roop_3d.engine.volumetric.session import dense_3d_max_roll
    monkeypatch.delenv("ROOP3D_3D_MAX_ROLL", raising=False)
    assert dense_3d_max_roll() == 40.0
    monkeypatch.setenv("ROOP3D_3D_MAX_ROLL", "60")
    assert dense_3d_max_roll() == 60.0
    monkeypatch.setenv("ROOP3D_3D_MAX_ROLL", "x")
    assert dense_3d_max_roll() == 40.0
