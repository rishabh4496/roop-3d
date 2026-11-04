"""Stage 19: a real "12 GB" card reports slightly under 12288 MiB and must still get the 12gb profile.

Found while writing the hardware-profile guidance: ``resolve_hardware_profile`` (Auto) said ``12gb`` for an RTX 4070
(12282 MiB), the Settings option list refused to let the user pick ``12gb`` and the render admission silently
demoted it to the clamped ``6gb`` profile. All three gates now share ``MIN_12GB_PROFILE_VRAM_MIB`` (the documented
10,240 MiB minimum).
"""
from __future__ import annotations

from roop_3d.core.hardware import GPU, MIN_12GB_PROFILE_VRAM_MIB, Hardware, resolve_hardware_profile
from roop_3d.core.settings import Settings
from roop_3d.engine.scheduler import RuntimeScheduler
from roop_3d.engine.vram_governor import VRAMGovernor


def _hardware(total: int, free: int) -> Hardware:
    return Hardware(timestamp="2026-11-05T00:00:00Z", os={"system": "Windows"}, logical_cpu_count=16,
                    gpus=[GPU(index=0, name="RTX 4070 (reported 12282 MiB)", total_vram_mib=total, free_vram_mib=free,
                              effective_vram_budget_mib=free - 768, driver_version="fixture")],
                    gpu_probe_state="VERIFIED_LOCAL", advertised_providers=["CUDAExecutionProvider"], provider_probe_state="VERIFIED_LOCAL")


def test_documented_minimum_matches_the_compatibility_matrix():
    assert MIN_12GB_PROFILE_VRAM_MIB == 10240


def test_auto_and_admission_agree_for_a_card_that_reports_12282_mib():
    assert resolve_hardware_profile(12282, 9966, "auto") == "12gb"
    scheduler = RuntimeScheduler(VRAMGovernor(lambda settings: _hardware(12282, 9966)))
    lease = scheduler.try_admit("job", Settings(hardware_profile="auto", preferred_provider="CUDAExecutionProvider"))
    assert lease is not None and lease.profile == "12gb"
    scheduler.release("job")


def test_a_card_below_the_documented_minimum_is_still_clamped_to_6gb():
    scheduler = RuntimeScheduler(VRAMGovernor(lambda settings: _hardware(8192, 7000)))
    lease = scheduler.try_admit("job", Settings(hardware_profile="12gb", preferred_provider="CUDAExecutionProvider"))
    assert lease is not None and lease.profile == "6gb"
    scheduler.release("job")
