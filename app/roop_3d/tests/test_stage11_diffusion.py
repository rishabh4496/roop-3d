from __future__ import annotations

import sys

import numpy as np
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Settings
from roop_3d.engine.diffusion.base import BackendState, check_memory_budget
from roop_3d.engine.diffusion.flux_backend import FluxBackend
from roop_3d.engine.diffusion.identity_adapter import validate_adapter
from roop_3d.engine.diffusion.sdxl_backend import SDXLBackend
from roop_3d.engine.diffusion.tile_manager import TileManager
from roop_3d.models.catalog import ModelCatalog


def test_diffusion_is_optional_and_local_candidates_report_unavailable() -> None:
    assert "diffusers" not in sys.modules
    catalog = ModelCatalog()
    result = SDXLBackend(catalog=catalog, profile="12gb", base_model_id="sdxl-turbo",
                         adapter_id="instantid", available_vram_bytes=8 * 1024**3).capability()
    assert result.state == BackendState.UNAVAILABLE
    assert dict(result.artifact_states) == {"sdxl-turbo": "CANDIDATE", "instantid": "CANDIDATE"}
    assert result.license_state == "UNKNOWN"
    assert "diffusers" not in sys.modules


@pytest.mark.parametrize("backend,kwargs", [
    (SDXLBackend, {"base_model_id": "sdxl-turbo", "adapter_id": "pulid-flux"}),
    (FluxBackend, {"base_model_id": "flux-1-schnell", "adapter_id": "instantid"}),
])
def test_cross_family_base_or_adapter_is_rejected_before_loading(backend, kwargs) -> None:
    with pytest.raises(DomainError) as error:
        backend(catalog=ModelCatalog(), **kwargs)
    assert error.value.code == "DIFFUSION_FAMILY_MISMATCH"


def test_loader_result_is_checked_for_family_before_use() -> None:
    sdxl = SDXLBackend(catalog=ModelCatalog(), base_model_id="sdxl-turbo", adapter_id="instantid")
    flux = FluxBackend(catalog=ModelCatalog(), base_model_id="flux-1-schnell", adapter_id="pulid-flux")

    class StableDiffusionXLPipeline:
        pass

    class FluxPipeline:
        pass

    sdxl.validate_pipeline(StableDiffusionXLPipeline())
    flux.validate_pipeline(FluxPipeline())
    with pytest.raises(DomainError) as error:
        sdxl.validate_pipeline(FluxPipeline())
    assert error.value.code == "DIFFUSION_FAMILY_MISMATCH"


def test_ambiguous_pulid_is_not_guessed_and_sdxl_embedding_contract_is_checked() -> None:
    with pytest.raises(DomainError, match="must be registered"):
        validate_adapter("sdxl", "pulid")
    validate_adapter("sdxl", "instantid", np.ones(512, dtype=np.float32))
    with pytest.raises(DomainError) as error:
        validate_adapter("sdxl", "instantid", np.ones(511, dtype=np.float32))
    assert error.value.code == "DIFFUSION_IDENTITY_EMBEDDING_INVALID"


def test_flux_6gb_unquantized_and_quantized_without_evidence_are_cleanly_unavailable() -> None:
    plain = FluxBackend(catalog=ModelCatalog(), profile="6gb", base_model_id="flux-1-schnell",
                        adapter_id="pulid-flux", available_vram_bytes=5 * 1024**3).capability()
    assert plain.state == BackendState.UNSUPPORTED_VRAM
    quantized = FluxBackend(catalog=ModelCatalog(), profile="6gb", base_model_id="flux-1-schnell",
                            adapter_id="pulid-flux", available_vram_bytes=5 * 1024**3,
                            quantization="nf4").capability()
    assert quantized.state == BackendState.UNAVAILABLE
    assert "VERIFIED_LOCAL" in quantized.reason


def test_memory_gate_reports_insufficient_or_unknown_vram() -> None:
    assert check_memory_budget(100, 101)[0] == BackendState.UNSUPPORTED_VRAM
    assert check_memory_budget(None, 101)[0] == BackendState.UNKNOWN
    assert check_memory_budget(200, 100)[0] == BackendState.AVAILABLE


def test_sdxl_six_gb_policy_calls_offload_and_slicing_methods() -> None:
    calls = []

    class FakePipeline:
        def enable_model_cpu_offload(self): calls.append("offload")
        def enable_vae_slicing(self): calls.append("vae")
        def enable_attention_slicing(self): calls.append("attention")

    backend = SDXLBackend(catalog=ModelCatalog(), profile="6gb", base_model_id="sdxl-turbo",
                          adapter_id="instantid")
    plan = backend.configure_pipeline_memory(FakePipeline())
    assert calls == ["offload", "vae", "attention"]
    assert plan.model_cpu_offload and plan.vae_slicing and plan.attention_slicing


def test_overlapping_tiles_reassemble_unchanged_image_and_have_stable_seeds() -> None:
    image = np.arange(719 * 911 * 3, dtype=np.uint32).reshape(719, 911, 3).astype(np.uint8)
    manager = TileManager((256, 320), overlap=48, job_seed=8675309)
    tiles = manager.split(image)
    rendered = [manager.extract(image, tile) for tile in tiles]
    merged = manager.reassemble(image.shape, tiles, rendered)
    assert np.array_equal(merged, image)
    assert [tile.seed for tile in tiles] == [tile.seed for tile in manager.split(image)]
    assert manager.denoising_strength == 0.25
    assert all(tile.seed < 2**63 for tile in tiles)
    points = np.array([[10, 12], [900, 700], [500, 300]], dtype=np.float32)
    local = manager.landmarks_for_tile(points, tiles[0])
    assert local.ndim == 2 and local.shape[1] == 2


def test_tile_validation_and_execution_mode() -> None:
    with pytest.raises(ValueError, match="0.15..0.45"):
        TileManager(tile_size=32, overlap=4, job_seed=1, denoising_strength=0.5)
    manager = TileManager(tile_size=32, overlap=8, job_seed=1)
    with pytest.raises(DomainError):
        manager.split(np.full((32, 32), np.nan, dtype=np.float32))
    assert Settings(execution_mode="diffusion_vfx").execution_mode == "diffusion_vfx"
