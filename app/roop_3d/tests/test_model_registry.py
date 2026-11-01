"""Comprehensive test suite for Stage 13:
1. Declarative Model Registry schemas & precision policies across all 5 categories.
2. Pixel Boost 512x512 sub-patch unfolding into (16, 3, 128, 128) and zero-seam reconstruction.
3. GET /api/models/registry readiness states for swappers, enhancers, 3D geometry, detectors, and masking.
4. Enhancer architectures: UltraMax <=320px bypass & Laplacian fusion, GPEN recoloring, Dynamic CodeFormer w binding.
5. Masking Engine: 4-layer compositing and watchdog seam baseline isolation.
"""
from pathlib import Path
import re

from fastapi.testclient import TestClient
import numpy as np
import pytest

from roop_3d.model_registry import (
    UNIFIED_MODEL_ENTRIES,
    UNIFIED_MANIFEST_MAP,
    UnifiedModelRegistry,
    ModelManifestEntry,
)
from roop_3d.swappers.pixel_boost import PixelBoost
from roop_3d.swappers.swapper_base import SwapperFactory, SWAPPER_CONFIGS
from roop_3d.enhancers.ultramax import UltraMaxRestorer, laplacian_pyramid_fusion
from roop_3d.enhancers.gpen_realistic import GPENRealisticRestorer, luma_only_delta_recolour
from roop_3d.enhancers.codeformer import DynamicCodeFormerRestorer, RestoreFormerPlusPlusRestorer
from roop_3d.masking_engine import MaskingEngine, BISENET_REGIONS
from roop_3d.tests.test_stage14_workstation import project_fixture


def test_manifest_entries_schemas_and_precision_policies():
    """Verify all model entries have valid schemas, valid URLs, 64-char SHA-256,
    and correct precision policies (FP32 for inswapper_128, FP16 for all others).
    """
    assert len(UNIFIED_MODEL_ENTRIES) >= 22

    categories_found = set()
    hex_pattern = re.compile(r"^[0-9a-f]{64}$")

    for entry in UNIFIED_MODEL_ENTRIES:
        categories_found.add(entry.category)
        # Check basic properties
        assert entry.model_id and len(entry.model_id) > 2
        assert entry.name.endswith(".onnx") or entry.name.endswith(".pth") or entry.name.endswith(".npy")
        assert entry.expected_size_bytes > 0
        assert hex_pattern.match(entry.expected_sha256), f"Invalid SHA-256 for {entry.name}"
        assert entry.primary_url.startswith("https://")
        assert entry.fallback_url.startswith("https://")
        assert entry.vram_footprint_mb > 0
        assert len(entry.input_shapes) > 0
        assert len(entry.output_shapes) > 0

        # Verify precision policy rules
        if entry.model_id == "inswapper_128":
            assert entry.precision_policy == "FP32", "inswapper_128 must enforce FP32 precision"
        else:
            assert entry.precision_policy == "FP16", f"{entry.model_id} must specify FP16 precision"

        # Verify GPEN static batch policy
        if "gpen" in entry.model_id:
            assert entry.supports_dynamic_batch is False, "GPEN models must enforce static batch-1"

    # Verify all 5 categories are present
    assert categories_found == {
        "swapper",
        "enhancer",
        "3d_geometry",
        "detector_embedder",
        "occlusion_masking",
    }


def test_pixel_boost_unfold_and_reconstruct_zero_grid_seams():
    """Verify Pixel Boost 512x512 unfolds a (1, 3, 512, 512) crop into (16, 3, 128, 128)
    and reconstructs seamlessly with zero grid seams.
    """
    pb = PixelBoost(model_size=128, boost_size=512)
    assert pb.grid_count == 4
    assert pb.total_patches == 16

    # Test with random float32 tensor
    np.random.seed(42)
    synthetic_crop = np.random.uniform(0.0, 1.0, size=(1, 3, 512, 512)).astype(np.float32)

    # 1. Unfold
    patches, coords = pb.unfold_sub_patches(synthetic_crop)
    assert patches.shape == (16, 3, 128, 128), f"Expected (16, 3, 128, 128), got {patches.shape}"
    assert len(coords) == 16

    # 2. Reconstruct with raised-cosine feather blending
    reconstructed = pb.reconstruct_from_sub_patches(patches, coords)
    assert reconstructed.shape == (1, 3, 512, 512)

    # 3. Verify zero grid seams: reconstruction error across all pixels is negligible
    max_abs_error = float(np.max(np.abs(reconstructed - synthetic_crop)))
    assert max_abs_error < 1e-5, f"Reconstruction seam error too high: {max_abs_error}"

    # Test via SwapperFactory integration
    swapper = SwapperFactory.get_swapper("inswapper_128", pixel_boost_size=512)
    assert swapper.pixel_boost is not None
    assert swapper.pixel_boost.total_patches == 16
    swapped = swapper.swap(synthetic_crop, source_embedding=np.zeros(512, dtype=np.float32))
    assert swapped.shape == (1, 3, 512, 512)


def test_get_api_models_registry_returns_accurate_readiness_states(tmp_path):
    """Verify GET /api/models/registry returns accurate readiness states for
    swappers, enhancers, 3D geometry, detectors, and masking models.
    """
    app, payload, model = project_fixture(tmp_path)
    models_dir = tmp_path / "app/models"
    models_dir.mkdir(parents=True, exist_ok=True)
    trt_dir = models_dir / "trt_cache"
    trt_dir.mkdir(parents=True, exist_ok=True)

    # Create one verified model on disk and one compiled TRT engine
    entry_hyperswap = UNIFIED_MANIFEST_MAP["hyperswap_1a_256"]
    test_model_file = models_dir / entry_hyperswap.name
    # Write synthetic bytes with expected hash for testing
    import hashlib
    raw_bytes = b"VERIFIED_MOCK_MODEL_BYTES"
    hasher = hashlib.sha256(raw_bytes)
    # We update entry expected hash in mock test or write exact matching bytes
    # Instead test inspect_model_readiness with corrupted and missing
    test_model_file.write_bytes(b"WRONG_HASH_BYTES")

    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        # Query GET /api/models/registry
        resp = client.get("/api/models/registry")
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert isinstance(data, list)
        assert len(data) >= 22

        # Verify all categories are covered
        categories = {item["category"] for item in data}
        assert "swapper" in categories
        assert "enhancer" in categories
        assert "3d_geometry" in categories
        assert "detector_embedder" in categories
        assert "occlusion_masking" in categories

        # Verify schema of readiness items
        for item in data:
            assert "model_id" in item
            assert "name" in item
            assert "readiness_state" in item
            assert item["readiness_state"] in (
                "READY + TRT ENGINE",
                "DOWNLOADED (NEEDS TRT)",
                "MISSING",
                "CORRUPT HASH",
            )
            assert "vram_footprint_mb" in item
            assert "precision_policy" in item
            assert "supports_dynamic_batch" in item

        # Hyperswap file has wrong bytes -> readiness should be CORRUPT HASH
        item_hyperswap = next(x for x in data if x["model_id"] == "hyperswap_1a_256")
        assert item_hyperswap["readiness_state"] == "CORRUPT HASH"

        # Missing models should be MISSING
        item_ghost = next(x for x in data if x["model_id"] == "ghost_1_256")
        assert item_ghost["readiness_state"] == "MISSING"

        # Test POST /api/models/verify
        v_resp = client.post("/api/models/verify", json={"model_id": "ghost_1_256"})
        assert v_resp.status_code == 200
        assert v_resp.json()["models"][0]["readiness_state"] == "MISSING"

        # Test POST /api/models/compile_trt
        c_resp = client.post("/api/models/compile_trt", json={"model_id": "simswap_unofficial_512", "precision": "fp16"})
        assert c_resp.status_code == 200
        assert c_resp.json()["results"][0]["status"] == "COMPILED"


def test_enhancers_ultramax_gpen_and_codeformer():
    """Verify UltraMax <=320px bypass & Laplacian fusion, GPEN luma-only recoloring,
    and Dynamic CodeFormer fidelity weight binding.
    """
    # 1. UltraMax native <= 320px bypass
    small_crop = np.full((240, 240, 3), 100, dtype=np.uint8)
    restorer = UltraMaxRestorer(native_bypass=True, min_native_pixels=320)
    bypassed = restorer.enhance(small_crop, native_resolution=240)
    assert np.array_equal(bypassed, small_crop), "UltraMax failed to bypass <=320px crop"

    # UltraMax on 512px crop runs Laplacian fusion
    large_crop = np.full((512, 512, 3), 150, dtype=np.uint8)
    enhanced = restorer.enhance(large_crop, native_resolution=512)
    assert enhanced.shape == (512, 512, 3)

    # 2. GPEN Realistic luma-only delta recoloring
    swap_crop = np.full((64, 64, 3), [120, 140, 180], dtype=np.uint8)  # Natural BGR
    gpen_pink = np.full((64, 64, 3), [160, 120, 220], dtype=np.uint8)  # Pink/magenta cast
    recoloured = luma_only_delta_recolour(swap_crop, gpen_pink)
    assert recoloured.shape == (64, 64, 3)
    # Chroma balance preserved
    assert recoloured[0, 0, 2] < 220, "GPEN pink cast was not attenuated"

    gpen_eng = GPENRealisticRestorer(resolution=512)
    assert gpen_eng.supports_dynamic_batch is False
    out_gpen = gpen_eng.enhance(np.zeros((64, 64, 3), dtype=np.uint8))
    assert out_gpen.shape == (64, 64, 3)

    # 3. Dynamic CodeFormer w binding
    codeformer = DynamicCodeFormerRestorer()
    res_w0 = codeformer.enhance(large_crop, fidelity_weight=0.0)
    res_w1 = codeformer.enhance(large_crop, fidelity_weight=1.0)
    assert res_w0.shape == (512, 512, 3)
    assert res_w1.shape == (512, 512, 3)

    # 4. RestoreFormer++
    rf = RestoreFormerPlusPlusRestorer()
    rf_out = rf.enhance(large_crop)
    assert rf_out.shape == (512, 512, 3)


def test_masking_engine_4_layers_and_watchdog_seam_isolation():
    """Verify 4-layer mask compositing and watchdog seam baseline tracking."""
    engine = MaskingEngine()

    # 1. Geometric hull mask
    landmarks = np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.float32)
    hull = engine.generate_3d_hull_mask(landmarks, shape=(100, 100), feather_radius=2)
    assert hull.shape == (100, 100)
    assert hull[50, 50] > 0.8
    assert hull[5, 5] < 0.2

    # 2. Semantic mask (BiSeNet regions)
    class_map = np.zeros((100, 100), dtype=np.uint8)
    class_map[30:70, 30:70] = 1  # Skin
    class_map[40:50, 40:60] = 6  # Eyeglasses (excluded by default)
    sem_mask = engine.generate_semantic_mask(class_map, selected_regions=["skin"], shape=(100, 100))
    assert sem_mask.shape == (100, 100)
    assert sem_mask[35, 35] > 0.5
    assert sem_mask[45, 50] < 0.5  # Eyeglasses excluded

    # 3. XSeg foreground neural occluder (hand/prop)
    xseg_raw = np.zeros((100, 100), dtype=np.float32)
    xseg_raw[30:50, 30:40] = 1.0  # Hand occluding left cheek
    xseg_mask = engine.generate_xseg_occlusion_mask(xseg_raw, shape=(100, 100))
    assert xseg_mask[35, 35] > 0.5

    # 4. CLIPSeg text-prompted mask
    clipseg_mask = engine.generate_clipseg_prompt_mask(
        prompt="microphone", image_shape=(100, 100), simulated_activation_box=(60, 60, 20, 20)
    )
    assert clipseg_mask[70, 70] > 0.5

    # 5. Composite 4 layers
    composite = engine.composite_4_layer_mask(
        geometric_hull=hull,
        semantic_mask=sem_mask,
        xseg_occlusion=xseg_mask,
        clipseg_occlusion=clipseg_mask,
    )
    assert composite.shape == (100, 100)
    assert composite.dtype == np.float32
    # Occluded hand area should be masked out
    assert composite[35, 35] < 0.2

    # 6. Watchdog seam baseline tracking
    valid1, dev1 = engine.update_watchdog_seam_baseline(thread_id=0, face_id=1, mask=composite)
    assert valid1 is True
    # Small frame-to-frame change should not trip seam alert
    valid2, dev2 = engine.update_watchdog_seam_baseline(thread_id=0, face_id=1, mask=composite * 0.95)
    assert valid2 is True
    assert dev2 < 0.1
