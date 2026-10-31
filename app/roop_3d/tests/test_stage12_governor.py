from __future__ import annotations

from roop_3d.core.hardware import GPU, Hardware
from roop_3d.core.settings import Settings
from roop_3d.engine.vram_governor import VRAMGovernor, is_cuda_oom
from roop_3d.engine.scheduler import RuntimeScheduler


def hardware(*, total: int, free: int, effective: int) -> Hardware:
    return Hardware(timestamp="2026-10-31T00:00:00Z", os={"system": "Windows"},
        logical_cpu_count=16, gpus=[GPU(index=0, name="fixture", total_vram_mib=total,
        free_vram_mib=free, effective_vram_budget_mib=effective, driver_version="fixture")],
        gpu_probe_state="VERIFIED_LOCAL", advertised_providers=["CUDAExecutionProvider"],
        provider_probe_state="VERIFIED_LOCAL")


def test_six_gigabyte_profile_is_single_stream_and_releases_admission():
    snapshot = hardware(total=6144, free=5600, effective=4800)
    governor = VRAMGovernor(lambda settings: snapshot)
    scheduler = RuntimeScheduler(governor)
    settings = Settings(hardware_profile="auto", preferred_provider="CUDAExecutionProvider",
                        max_concurrent_jobs=4)

    first = scheduler.try_admit("one", settings)
    assert first is not None and first.profile == "6gb"
    assert scheduler.try_admit("two", settings) is None
    scheduler.release("one")
    second = scheduler.try_admit("two", settings)
    assert second is not None
    scheduler.release("two")
    assert scheduler.active_count == governor.active_count == 0


def test_twelve_gigabyte_profile_uses_live_free_memory_and_configured_bound():
    snapshot = hardware(total=12288, free=12000, effective=11200)
    scheduler = RuntimeScheduler(VRAMGovernor(lambda settings: snapshot))
    settings = Settings(hardware_profile="auto", preferred_provider="CUDAExecutionProvider",
                        max_concurrent_jobs=3)

    first = scheduler.try_admit("one", settings)
    second = scheduler.try_admit("two", settings)
    assert first is not None and second is not None
    assert first.profile == second.profile == "12gb"
    assert scheduler.try_admit("three", settings) is None
    scheduler.release("one")
    scheduler.release("two")


def test_low_live_budget_selects_cpu_and_oom_detection_is_specific():
    low = hardware(total=12288, free=3200, effective=2500)
    governor = VRAMGovernor(lambda settings: low)
    lease = governor.try_acquire("low", Settings(hardware_profile="12gb",
        preferred_provider="CUDAExecutionProvider"))
    assert lease is not None
    assert lease.admission.profile == "cpu"
    assert lease.admission.provider == "CPUExecutionProvider"
    lease.release()

    assert is_cuda_oom(RuntimeError("CUDA out of memory while allocating tensor"))
    assert is_cuda_oom(MemoryError("allocator exhausted"))
    assert not is_cuda_oom(RuntimeError("invalid CUDA tensor shape"))


def test_cache_cleanup_path_cannot_be_redirected(tmp_path):
    from roop_3d.tools.cache_cleanup import find_candidates

    try:
        find_candidates(tmp_path)
    except ValueError as error:
        assert "restricted" in str(error)
    else:
        raise AssertionError("External cache root was accepted")
