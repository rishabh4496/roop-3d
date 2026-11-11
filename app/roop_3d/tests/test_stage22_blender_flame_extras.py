"""Stage 22 geometry, restoration, grading, proxy and timeline utilities."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from roop_3d.io.proxy_and_edl import (CalibratedRenderTimeEstimator, VideoProxyAndEDLManager,
                                      frame_to_timecode, parse_edl, timecode_to_frame)
from roop_3d.reconstruction.subdivision_and_masks import (DEFAULT_REGION_FEATHER_PX, REGION_NAMES, REGION_PBR,
                                                           feather_region_swap_masks, segment_head_vertices,
                                                           subdivide_silhouette)
from roop_3d.rendering.color_and_flow import CubeLUT, CubeLUTColorMatcher, FlowMapAnisotropicShader
from roop_3d.rendering.super_res_tiles import SuperResTileEnhancer, split_image_tiles


def _head_grid() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    x = np.linspace(-1.0, 1.0, 35)
    y = np.linspace(-1.2, 1.0, 39)
    xx, yy = np.meshgrid(x, y)
    # An ellipsoid-like, front/back-varying head surface; all measurements are
    # deterministic and exercise bounds-only segmentation as well as anchors.
    zz = 0.35 * (xx * xx + yy * yy)
    vertices = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())).astype(np.float32)
    def nearest(px: float, py: float) -> int:
        return int(np.argmin((vertices[:, 0] - px) ** 2 + (vertices[:, 1] - py) ** 2))
    landmarks = {"5pt": np.array([nearest(-0.3, -0.2), nearest(0.3, -0.2), nearest(0, 0.0),
                                  nearest(-0.16, 0.28), nearest(0.16, 0.28)])}
    return vertices, landmarks


@pytest.mark.parametrize("topology_id", ["makehuman_cc0", "flame2023_open"])
def test_segmentation_defines_nine_regions_and_t_zone_pbr(topology_id):
    vertices, landmarks = _head_grid()
    groups = segment_head_vertices(vertices, topology_id=topology_id, landmarks=landmarks,
                                   feather_px={"nose": 0.5, "submental_neck": 8.0})
    assert tuple(groups.groups) == REGION_NAMES
    assert sorted(np.concatenate(list(groups.groups.values())).tolist()) == list(range(len(vertices)))
    assert all(len(groups.groups[name]) > 0 for name in REGION_NAMES)
    assert groups.weights.shape == (len(vertices), 9)
    np.testing.assert_allclose(groups.weights.sum(axis=1), 1.0, atol=1e-6)
    assert REGION_PBR["nose"]["roughness"] < REGION_PBR["malar_cheeks"]["roughness"]
    assert REGION_PBR["forehead"]["specular"] > REGION_PBR["submental_neck"]["specular"]
    assert groups.feather_px["nose"] == 0.5 and groups.feather_px["submental_neck"] == 8.0
    assert DEFAULT_REGION_FEATHER_PX["mandible_jaw_chin"] > DEFAULT_REGION_FEATHER_PX["nose"]


def test_per_region_feathering_is_independent_and_bounded():
    masks = {"nose": np.pad(np.ones((3, 3), np.float32), 5),
             "malar_cheeks": np.pad(np.ones((3, 3), np.float32), 5)}
    output = feather_region_swap_masks(masks, {"nose": 0, "malar_cheeks": 4})
    assert np.array_equal(output["nose"], masks["nose"])
    assert 0 < output["malar_cheeks"][6, 4] < 1
    with pytest.raises(ValueError):
        feather_region_swap_masks(masks, {"nose": 70})


def test_one_level_subdivision_smooths_contour_and_interpolates_vertex_weights():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int64)
    weights = np.eye(4, dtype=np.float32)
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    result = subdivide_silhouette(vertices, faces, silhouette_edges=edges, vertex_weights=weights, device="cpu")
    assert result.vertices.shape == (9, 3)
    assert result.faces.shape == (8, 3)
    assert result.device == "cpu"
    np.testing.assert_allclose(result.vertices[1], [0.875, 0.125, 0])
    np.testing.assert_allclose(result.vertex_weights.sum(axis=1), 1.0)
    assert set(map(tuple, result.silhouette_edges)) == set(map(tuple, edges))


def test_tile_plan_covers_frame_and_vram_pressure_triggers_cosine_tiles():
    tiles = split_image_tiles(173, 131, tile_size=64, overlap=12)
    cover = np.zeros((131, 173), np.uint8)
    for tile in tiles:
        cover[tile.y0:tile.y1, tile.x0:tile.x1] += 1
    assert cover.min() >= 1 and cover.max() > 1

    image = np.full((131, 173, 3), 20, np.uint8)
    enhancer = SuperResTileEnhancer(tile_size=64, overlap=12, memory_probe=lambda: {"pressure_pct": 0.91})
    result = enhancer.enhance(image, lambda crop: crop.astype(np.uint8) + 5, model_name="UltraMax",
                              frame_resolution=(3840, 2160), budget_mb=6144)
    assert enhancer.last_run["tiled"] and enhancer.last_run["tile_count"] > 1
    np.testing.assert_array_equal(result, np.full_like(image, 25))


def test_tile_enhancer_retries_a_supported_full_head_after_oom():
    image = np.full((150, 130, 3), 7, np.uint8)
    calls = []
    def enhance(crop):
        calls.append(crop.shape[:2])
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory")
        return crop
    tiler = SuperResTileEnhancer(tile_size=64, overlap=8)
    output = tiler.enhance(image, enhance, model_name="GPEN Realistic", frame_resolution=(3840, 2160))
    assert tiler.last_run["tiled"] and len(calls) > 1
    np.testing.assert_array_equal(output, image)


def _identity_cube(size: int = 5) -> CubeLUT:
    axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
    b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")
    return CubeLUT(np.stack((r, g, b), axis=-1), title="Identity")


def test_cube_lut_round_trip_and_rgb_order(tmp_path):
    lut = _identity_cube()
    text = lut.to_text()
    parsed = CubeLUT.parse(text)
    image = np.array([[[0, 64, 255], [12, 180, 90]]], np.uint8)
    np.testing.assert_allclose(parsed.apply(image), image, atol=1)
    path = CubeLUTColorMatcher.export_cube(parsed, tmp_path / "identity.cube")
    np.testing.assert_allclose(CubeLUTColorMatcher.import_cube(path).apply(image), image, atol=1)
    with pytest.raises(ValueError, match="1D"):
        CubeLUT.parse("LUT_1D_SIZE 2\n")


def test_histogram_match_lut_maps_source_distribution_toward_target():
    levels = np.linspace(0.1, 0.7, 64, dtype=np.float32)
    source = np.tile(np.rint(levels[:, None, None] * 255).astype(np.uint8), (1, 16, 3))
    target = np.tile(np.rint(np.linspace(0.25, 0.95, 64, dtype=np.float32)[:, None, None] * 255).astype(np.uint8), (1, 16, 3))
    lut = CubeLUTColorMatcher.match_histograms(source, target, size=9)
    matched = lut.apply(source)
    assert float(matched.mean()) > float(source.mean())
    assert lut.size == 9


def test_tangent_flow_tracks_strand_orientation_and_shading_changes_pixels():
    image = np.zeros((96, 96, 3), np.uint8)
    image[:, ::4] = 255  # vertical strands: intensity gradient is horizontal, flow is vertical
    flow = FlowMapAnisotropicShader.tangent_flow(image, smooth_sigma=1.0)
    assert np.median(np.abs(flow[20:-20, 20:-20, 1])) > np.median(np.abs(flow[20:-20, 20:-20, 0]))
    assert flow[..., 2].max() > 0.1
    shaded = FlowMapAnisotropicShader.shade(image, flow, light_direction=(0.0, 0.02, 0.9998), strength=1.0)
    assert shaded.shape == image.shape and np.any(shaded.astype(np.int16) > image.astype(np.int16))
    masks = {name: np.ones(image.shape[:2], np.float32) for name in FlowMapAnisotropicShader.REGIONS}
    assert set(FlowMapAnisotropicShader.region_flows(image, masks)) == set(masks)


def test_proxy_and_filmstrip_generation_uses_local_ffmpeg_commands(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"local fixture")
    commands = []
    def fake_run(command, **_kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"generated fixture")
        return SimpleNamespace(returncode=0, stderr="")
    manager = VideoProxyAndEDLManager(ffmpeg="ffmpeg-local", runner=fake_run)
    assets = manager.prepare_import(source, tmp_path / "assets", duration_seconds=10.0, asset_key="shot_1")
    assert assets["state"] == "ready"
    assert Path(assets["proxy"]).is_file() and Path(assets["filmstrip"]).is_file()
    assert len(commands) == 2
    filmstrip_filter = commands[1][commands[1].index("-vf") + 1]
    assert "libx264" in commands[0] and "tile=4x4" in filmstrip_filter


def test_edl_export_and_import_preserve_swap_ranges_and_timecode():
    plan = {"name": "Shot", "frame_rate": "25/1", "frame_count": 100, "width": 1920, "height": 1080,
            "source_url": "source.mov", "swapped_url": "swap.mov", "swapped_ranges": [[25, 50]],
            "scene_cuts": [0, 40, 100]}
    files = VideoProxyAndEDLManager.export_edl(plan)
    imported = parse_edl(files["V2.edl"])
    assert imported.title == "Shot V2 Swapped Clips"
    assert imported.swapped_ranges == ((25, 50),)
    assert 25 in imported.scene_cuts

    assert frame_to_timecode(1800, "30000/1001", drop_frame=True) == "00:01:00;02"
    assert timecode_to_frame("00:01:00;02", "30000/1001") == 1800
    with pytest.raises(ValueError, match="skipped"):
        timecode_to_frame("00:01:00;00", "30000/1001")


def test_render_eta_becomes_available_after_eight_calibration_frames():
    eta = CalibratedRenderTimeEstimator(total_frames=100)
    for _ in range(7):
        snapshot = eta.observe_frame(0.5)
        assert snapshot["eta_seconds"] is None
    snapshot = eta.observe_frame(0.5)
    assert snapshot["calibrated"] and snapshot["warmup_remaining"] == 0
    assert snapshot["completed_frames"] == 8
    assert snapshot["eta_seconds"] == pytest.approx(46.0)
    assert snapshot["estimated_total_seconds"] >= snapshot["eta_seconds"]
