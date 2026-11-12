"""Inspection / target-lock models (detector, ArcFace, 1k3d68) run on CUDA on big cards, on the CPU otherwise.

Measured on the real models next to a running job: RetinaFace 204 -> 27 ms, ArcFace 199 -> 17 ms, 1k3d68 43 -> 10 ms per
face-frame, identical output.  These tests pin the decision logic and the CPU fallback, not the GPU itself.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from roop_3d.engine import runtime
from roop_3d.engine.runtime import InspectionOnCpu, build_with_cpu_fallback, inspection_runtime, inspection_session_factory
from roop_3d.faceset_loader import LandmarkEngine


def _hardware(total_mib: int, free_mib: int):
    return SimpleNamespace(gpus=[SimpleNamespace(total_vram_mib=total_mib, free_vram_mib=free_mib)])


@pytest.fixture()
def gpu_box(monkeypatch):
    """A machine whose ORT advertises CUDA; the probe result is set per test."""
    import onnxruntime as ort
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.delenv("ROOP3D_INSPECTION_PROVIDER", raising=False)

    def install(total, free):
        from roop_3d.core import hardware
        monkeypatch.setattr(hardware, "probe_hardware", lambda settings=None: _hardware(total, free))
    return install


def test_big_card_uses_cuda_with_the_resolved_profile(gpu_box):
    gpu_box(12282, 8000)
    config = inspection_runtime()
    assert config is not None and config.provider == "CUDAExecutionProvider" and config.hardware_profile == "12gb"


def test_small_card_stays_on_the_cpu_so_job_admission_keeps_its_vram(gpu_box):
    gpu_box(6144, 5000)
    assert inspection_runtime() is None


def test_card_with_little_free_vram_stays_on_the_cpu(gpu_box):
    gpu_box(12282, 1500)
    assert inspection_runtime() is None


def test_no_gpu_means_cpu(monkeypatch, gpu_box):
    from roop_3d.core import hardware
    monkeypatch.setattr(hardware, "probe_hardware", lambda settings=None: SimpleNamespace(gpus=[]))
    assert inspection_runtime() is None


def test_environment_overrides(monkeypatch, gpu_box):
    gpu_box(12282, 8000)
    monkeypatch.setenv("ROOP3D_INSPECTION_PROVIDER", "cpu")
    assert inspection_runtime() is None
    gpu_box(6144, 5000)
    monkeypatch.setenv("ROOP3D_INSPECTION_PROVIDER", "cuda")
    assert inspection_runtime() is not None


def test_cpu_settings_never_touch_the_gpu(gpu_box):
    from roop_3d.core.settings import Settings
    gpu_box(12282, 8000)
    assert inspection_runtime(Settings(preferred_provider="CPUExecutionProvider")) is None
    assert inspection_runtime(Settings(hardware_profile="cpu")) is None


def test_factory_with_cpu_config_declines(gpu_box):
    cpu_config = runtime.RuntimeConfig()
    factory = inspection_session_factory(object(), config=cpu_config)
    with pytest.raises(InspectionOnCpu):
        factory({"name": "w600k_r50.onnx"})


def test_build_with_cpu_fallback_contract(caplog):
    assert build_with_cpu_fallback(None, {"name": "x"}, "x") is None

    def declines(record):
        raise InspectionOnCpu("cpu")
    declines.cpu_fallback = True
    assert build_with_cpu_fallback(declines, {"name": "x"}, "x") is None

    def explodes(record):
        raise RuntimeError("cuda init failed")
    explodes.cpu_fallback = True
    with caplog.at_level("WARNING"):
        assert build_with_cpu_fallback(explodes, {"name": "x"}, "x") is None
    assert "using the CPU" in caplog.text

    # a render tracker's factory is not an inspection factory: its failures stay visible
    def tracker_factory(record):
        raise RuntimeError("tensorrt failed")
    with pytest.raises(RuntimeError):
        build_with_cpu_fallback(tracker_factory, {"name": "x"}, "x")

    sentinel = object()
    assert build_with_cpu_fallback(lambda record: sentinel, {"name": "x"}, "x") is sentinel


class _FakeSession:
    def __init__(self, label):
        self.label = label
        self.calls = 0

    def get_inputs(self):
        return [SimpleNamespace(name="data")]

    def get_outputs(self):
        return [SimpleNamespace(name="fc1")]

    def get_providers(self):
        return ["CUDAExecutionProvider"]

    def run(self, names, feeds):
        self.calls += 1
        return [np.zeros((1, 3309), np.float32)]


def test_landmark_engine_routes_only_the_68_point_model_to_the_factory(tmp_path):
    (tmp_path / "1k3d68.onnx").write_bytes(b"not a real model")
    (tmp_path / "2d106det.onnx").write_bytes(b"not a real model")
    seen = []

    def factory(record):
        seen.append(record["name"])
        return _FakeSession(record["name"])
    factory.cpu_fallback = True
    engine = LandmarkEngine(model_dirs=[tmp_path], session_factory=factory)
    assert engine.has_68 and engine.provider("1k3d68.onnx") == "CUDAExecutionProvider"
    points = engine.landmarks68(np.zeros((200, 200, 3), np.uint8), [50, 50, 150, 150])
    assert points.shape == (68, 3) and np.isfinite(points).all()
    assert engine._sessions["1k3d68.onnx"].calls == 1
    engine.has_106                                     # the 106-point model never goes through the GPU factory
    assert seen == ["1k3d68.onnx"]


def test_landmark_engine_falls_back_when_the_factory_declines(tmp_path):
    (tmp_path / "1k3d68.onnx").write_bytes(b"not a real model")

    def factory(record):
        raise InspectionOnCpu("cpu")
    factory.cpu_fallback = True
    engine = LandmarkEngine(model_dirs=[tmp_path], session_factory=factory)
    # the CPU path is tried next; an unreadable file is reported as unavailable, never fabricated
    assert engine.has_68 is False
    assert engine.landmarks68(np.zeros((64, 64, 3), np.uint8), [0, 0, 32, 32]) is None
