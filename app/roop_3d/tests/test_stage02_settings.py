from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError
import pytest

from roop_3d.core.hardware import driver_cuda_version, effective_vram_budget, probe_hardware
from roop_3d.core.settings import ConfigStore, Settings, migrate_config


@pytest.mark.parametrize("change", [
    {"vram_limit_mb": float("nan")}, {"vram_limit_mb": float("inf")},
    {"max_concurrent_jobs": 0}, {"max_concurrent_jobs": True},
    {"execution_mode": "imaginary"}, {"precision_policy": "int4"},
    {"schema_version": 99}, {"api_key": "DO_NOT_EXPOSE"},
    {"output_root": "../roop-ultimate"}, {"cache_root": "roop_3d/core"},
    {"model_search_paths": ["../../roop-ultimate"]},
    {"output_root": "roop_3d/.runtime/../core"},
])
def test_invalid_catalog(change: dict) -> None:
    with pytest.raises(ValidationError):
        Settings(**change)


def test_catalog_has_all_requested_keys() -> None:
    required = "schema_version execution_mode hardware_profile preferred_provider precision_policy vram_limit_mb memory_safety_reserve_mb max_concurrent_jobs face_restoration_policy video_timing_policy audio_policy lip_sync_mode model_search_paths output_root cache_root remote_processing_enabled synthetic_media_notice diagnostics_level".split()
    assert set(required) <= Settings.model_fields.keys()
    assert Settings.model_json_schema()["additionalProperties"] is False
    assert Settings(preferred_provider="TensorrtExecutionProvider").preferred_provider == "TensorrtExecutionProvider"


def test_config_atomic_backup_and_explicit_migration(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path)
    first = Settings()
    store.save(first)
    second = Settings(max_concurrent_jobs=2)
    store.save(second)
    assert store.load() == second
    backup = tmp_path / (store.relative + ".bak")
    assert Settings.model_validate_json(backup.read_text()) == first
    path = tmp_path / store.relative
    path.write_text(json.dumps({"port": 8031}))
    with pytest.raises(ValueError, match="migration"):
        store.load()
    # A missing version never gets a silent default while reading a saved file.
    assert migrate_config({"port": 8031})["schema_version"] == 2
    assert store.load(migrate=True).port == 8031
    with pytest.raises(ValueError):
        migrate_config({"schema_version": 1})


def test_dynamic_budget_profiles() -> None:
    config = Settings(hardware_profile="6gb", memory_safety_reserve_mb=512)
    assert effective_vram_budget(12288, 10000, config) == 5632
    assert effective_vram_budget(12288, 100, config) == 0
    assert effective_vram_budget(12288, 4000, config) == 3488
    assert effective_vram_budget(12288, 10000, Settings(hardware_profile="cpu")) == 0


def test_corrupt_configs_and_direct_validation_hide_secret_values(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path)
    path = tmp_path / store.relative
    path.parent.mkdir(parents=True)
    path.write_text("[]")
    with pytest.raises(ValueError, match="object"):
        store.load()
    with pytest.raises(ValidationError) as error:
        Settings(api_key="DO_NOT_EXPOSE")
    assert "DO_NOT_EXPOSE" not in str(error.value)


def test_real_probe_reports_ram_and_honest_codec_inventory() -> None:
    value = probe_hardware()
    assert value.logical_cpu_count and value.logical_cpu_count > 0
    assert value.system_ram_total_mib and value.system_ram_total_mib > 0
    assert value.system_ram_available_mib is not None
    assert value.system_ram_available_mib <= value.system_ram_total_mib
    assert value.provider_execution_verified is False
    assert value.ffmpeg_execution_verified is False


def test_old_and_new_nvidia_driver_cuda_output() -> None:
    assert driver_cuda_version("| CUDA Version: 12.4 |") == "12.4"
    assert driver_cuda_version("| KMD Version: 621.01 CUDA UMD Version: 13.5 |") == "13.5"
    assert driver_cuda_version("CUDA Version: N/A") is None
