"""The stub recovery harness must never read installed neural weights."""
from roop_3d.core.errors import DomainError
from roop_3d.jobs.schema import JobState
from roop_3d.models.local_provider import LocalWeightProvider
from roop_3d.reliability.harness import build_rig


def test_stub_recovery_uses_only_owned_fixture_models(tmp_path, monkeypatch):
    from roop_3d.core import hardware
    probes = []
    def no_machine_probe(settings):
        probes.append(settings)
        raise AssertionError("Stub recovery cannot probe machine hardware")
    monkeypatch.setattr(hardware, "probe_hardware", no_machine_probe)
    resolved = []
    original = LocalWeightProvider.resolve

    def local_only(provider, model):
        record = provider.registry.get(model) if isinstance(model, str) else model
        resolved.append(record.model_id)
        if not record.model_id.startswith('reliability_fixture_'):
            raise DomainError('MODEL_NOT_INSTALLED', 'Stub recovery cannot access installed neural weights', 503)
        path = original(provider, record)
        assert path.is_relative_to(tmp_path)
        return path

    monkeypatch.setattr(LocalWeightProvider, 'resolve', local_only)
    rig = build_rig(tmp_path, frames=4, tracks=1)
    try:
        job = rig.wait(rig.submit().job_id, timeout=5)
        assert job.state == JobState.succeeded, job.failure
        assert resolved and all(key.startswith('reliability_fixture_') for key in resolved)
        assert probes == []
    finally:
        rig.worker.stop()


def test_fixture_models_are_still_hash_checked_on_resume(tmp_path):
    import pytest
    from roop_3d.reliability.harness import fixture_catalog
    catalog = fixture_catalog(tmp_path)
    model = catalog.detector()
    path = catalog.provider.resolve(model['model_id'])
    payload = path.read_bytes()
    path.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
    resumed = fixture_catalog(tmp_path)
    assert path.read_bytes() != payload  # re-opening the rig must not repair drift silently
    with pytest.raises(DomainError) as error:
        resumed.verify(model)
    assert error.value.code == 'MODEL_DRIFT'


def test_zero_age_recovery_includes_future_dated_crash_artifacts(tmp_path):
    import os
    import time
    from roop_3d.core.settings import Settings
    from roop_3d.jobs.store import JobStore
    from roop_3d.reliability.recovery import ReliabilityAuditor
    settings = Settings(hardware_profile='cpu')
    temporary = tmp_path / 'roop_3d/outputs' / ('a'*32 + '.video.mp4')
    output = tmp_path / settings.output_root / ('roop3d_' + 'b'*32 + '.mp4')
    for path in (temporary, output):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'crash leftover')
        future = time.time() + 60
        os.utime(path, (future, future))
    auditor = ReliabilityAuditor(tmp_path, JobStore(tmp_path), lambda: settings)
    codes = {finding.code for finding in auditor.audit(min_age_s=0).findings}
    assert {'ORPHAN_ENCODER_TEMP', 'UNOWNED_OUTPUT'} <= codes
    assert auditor.audit(min_age_s=3600).findings == []


def test_cpu_job_completion_does_not_wait_for_gpu_telemetry(tmp_path, monkeypatch):
    import io
    import threading
    from roop_3d.jobs import worker as worker_module
    from roop_3d.reliability import probe
    from roop_3d.telemetry.pinokio_terminal import PinokioTerminalHUD, ResourceSampler
    gpu_samples = []
    cpu_sampled = threading.Event()

    class CpuMeter:
        def read(self):
            cpu_sampled.set()
            return 0.0

    def gpu_query():
        gpu_samples.append(True)
        return probe.GpuMemory()

    class SynchronizedSampler(ResourceSampler):
        def acquire(self, *, with_gpu=True):
            super().acquire(with_gpu=with_gpu)
            assert cpu_sampled.wait(2), "the HUD sampler did not publish its first CPU reading"

    sampler = SynchronizedSampler(gpu=gpu_query, cpu=CpuMeter())
    hud = PinokioTerminalHUD(io.StringIO(), color=False, in_place=False, sampler=sampler)
    hud.print_report = lambda *_args, **_kwargs: None
    monkeypatch.delenv("ROOP3D_HUD", raising=False)
    monkeypatch.setattr(worker_module, "HUD", hud.arm())
    rig = build_rig(tmp_path, frames=4, tracks=1)
    try:
        result = rig.wait(rig.submit().job_id, timeout=5)
        assert result.state == JobState.succeeded, result.failure
        assert cpu_sampled.is_set()
    finally:
        rig.worker.stop()
        hud.disarm()
    assert gpu_samples == []
