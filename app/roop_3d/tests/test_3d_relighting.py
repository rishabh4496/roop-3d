"""Stage 17 Verification Gate: 3D Relighting & Normal Map Estimation.

Verifies:
(1) Spherical Harmonics estimator accurately recovers a synthetic right-side directional light source from a test face normal map.
(2) Relighting transfers directional shading across the nose bridge without altering underlying skin hue.
(3) POST /api/preview supports solo_node="normal_map" and solo_node="sh_lighting".
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from roop_3d.relighting_3d import (
    NeuralRelighter3D,
    SphericalHarmonicsLighting,
    SurfaceNormalEstimator,
)
from roop_3d.server.app import create_app
from roop_3d.tests.test_remediation_stage4 import BASE, live_payload
from roop_3d.tests.test_workstation_preview import preview_fixture


def test_spherical_harmonics_recovers_synthetic_right_side_directional_light() -> None:
    """Gate 1: Accurately recover synthetic right-side directional light source."""
    estimator = SurfaceNormalEstimator()
    sh_lighting = SphericalHarmonicsLighting()

    # Generate 128x128 test face normal map with realistic 3D facial curvature
    h, w = 128, 128
    normals = estimator.estimate_geometric_normals(h, w)
    assert normals.shape == (h, w, 3)
    assert np.all(np.isfinite(normals))

    # True synthetic key light source positioned on the RIGHT side (+X direction)
    true_light = np.array([0.85, 0.0, 0.52], dtype=np.float32)
    true_light /= np.linalg.norm(true_light)

    # Generate synthetic diffuse face irradiance (ambient + directional)
    # n . l irradiance
    ambient = 0.25
    intensity = 0.75
    diffuse = np.maximum(normals @ true_light, 0.0)
    irradiance = ambient + intensity * diffuse
    test_face_bgr = np.clip(irradiance[..., None] * np.array([210.0, 220.0, 240.0]), 0, 255).astype(np.uint8)

    # Face mask covering visible face area
    y, x = np.mgrid[-1.0:1.0:complex(h), -1.0:1.0:complex(w)]
    mask = (x * x + y * y) <= 0.80

    # Estimate 9x3 SH lighting environment
    sh_coeffs = sh_lighting.estimate_lighting(normals, test_face_bgr, mask=mask)
    assert sh_coeffs.shape == (9, 3)

    # Extract dominant key light direction vector
    recovered_dir = sh_lighting.extract_key_light_direction(sh_coeffs)
    assert recovered_dir.shape == (3,)
    assert np.isclose(np.linalg.norm(recovered_dir), 1.0, atol=1e-4)

    # Assert that recovered direction is clearly oriented to the RIGHT (+X)
    assert recovered_dir[0] > 0.6, f"Expected positive right-side X component, got {recovered_dir[0]}"
    assert recovered_dir[0] > abs(recovered_dir[1]), "X component must dominate over vertical Y"

    # Cosine similarity between true light and recovered light
    cos_sim = float(np.dot(recovered_dir, true_light))
    assert cos_sim > 0.85, f"Expected high cosine similarity (>0.85), got {cos_sim:.3f}"


def test_relighting_transfers_directional_shading_across_nose_bridge_without_altering_skin_hue() -> None:
    """Gate 2: Directional shading across the nose bridge while preserving underlying skin hue."""
    relighter = NeuralRelighter3D()
    h, w = 128, 128

    # Create face geometry with a distinct nose bridge at center (x = 64)
    y, x = np.mgrid[-1.0:1.0:complex(h), -1.0:1.0:complex(w)]
    r2 = x * x + y * y
    head_z = np.sqrt(np.maximum(1.0 - np.minimum(r2, 0.95), 0.0))
    nose_z = 0.25 * np.exp(-(x * x) / (2.0 * 0.07 * 0.07)) * np.maximum(0.0, 1.0 - np.abs(y) / 0.4)
    total_z = head_z + nose_z

    dz_dy, dz_dx = np.gradient(total_z)
    normals = np.stack([-dz_dx, -dz_dy, np.ones_like(dz_dx) * 0.06], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

    # Verify nose slope geometry: left slope has nx < 0, right slope has nx > 0
    center_y = h // 2
    left_x = w // 2 - 5
    right_x = w // 2 + 5
    assert normals[center_y, left_x, 0] < -0.15, "Left slope must have negative nx"
    assert normals[center_y, right_x, 0] > 0.15, "Right slope must have positive nx"

    # Swapped face with uniform natural skin color (BGR: 110, 140, 195)
    swapped_face = np.full((h, w, 3), [110, 140, 195], dtype=np.uint8)
    orig_hsv = cv2.cvtColor(swapped_face, cv2.COLOR_BGR2HSV)
    orig_hue = orig_hsv[center_y, left_x, 0]

    # Target face with right-side directional light (+X)
    target_face = swapped_face.copy()

    # Relight swapped face with right-side directional key light
    relit, normal_vis, sh_vis, meta = relighter.relight(
        swapped_bgr=swapped_face,
        target_bgr=target_face,
        swap_normals=normals,
        target_normals=normals,
        relight_strength=1.0,
        specular_sheen=0.0,  # Test diffuse hue preservation
        shadow_preservation=0.0,
        lighting_mode="manual",
        manual_light_yaw=55.0,  # 55 deg right-side key light
        manual_light_pitch=0.0,
    )

    # 1. Check directional shading across the nose bridge:
    # Right slope facing key light must be significantly brighter than left slope (in self-shadow)
    left_val = float(np.mean(relit[center_y - 10:center_y + 10, left_x - 3:left_x + 1]))
    right_val = float(np.mean(relit[center_y - 10:center_y + 10, right_x - 1:right_x + 3]))
    assert right_val > left_val * 1.3, (
        f"Right slope ({right_val:.1f}) must be noticeably brighter than left slope ({left_val:.1f})"
    )

    # 2. Check skin hue invariance:
    # Converting relit skin to HSV must preserve the original skin hue
    relit_hsv = cv2.cvtColor(relit, cv2.COLOR_BGR2HSV)
    left_hue = relit_hsv[center_y, left_x, 0]
    right_hue = relit_hsv[center_y, right_x, 0]

    assert abs(int(left_hue) - int(orig_hue)) <= 1, f"Left slope hue shifted: {left_hue} vs {orig_hue}"
    assert abs(int(right_hue) - int(orig_hue)) <= 1, f"Right slope hue shifted: {right_hue} vs {orig_hue}"

    # Also check normal map visualization
    assert normal_vis.shape == (h, w, 3)
    assert sh_vis.shape == (h, w, 3)


def test_preview_supports_solo_normal_map_and_sh_lighting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Gate 3: POST /api/preview supports solo_node='normal_map' and solo_node='sh_lighting'."""
    test_app, fixture = preview_fixture(tmp_path, monkeypatch)

    with TestClient(test_app, base_url=BASE) as client:
        # 1. Solo normal_map
        payload_normal = live_payload(fixture, solo_node="normal_map")
        resp_normal = client.post("/api/preview", json=payload_normal)
        assert resp_normal.status_code == 200, resp_normal.text
        assert resp_normal.headers["content-type"] == "image/png"
        meta_normal = json.loads(resp_normal.headers["x-preview-meta"])
        assert meta_normal["solo_node"] == "normal_map"

        img_normal = cv2.imdecode(np.frombuffer(resp_normal.content, np.uint8), cv2.IMREAD_COLOR)
        assert img_normal is not None
        assert img_normal.ndim == 3
        assert img_normal.shape[0] > 0 and img_normal.shape[1] > 0

        # 2. Solo sh_lighting
        payload_sh = live_payload(fixture, solo_node="sh_lighting")
        resp_sh = client.post("/api/preview", json=payload_sh)
        assert resp_sh.status_code == 200, resp_sh.text
        assert resp_sh.headers["content-type"] == "image/png"
        meta_sh = json.loads(resp_sh.headers["x-preview-meta"])
        assert meta_sh["solo_node"] == "sh_lighting"

        img_sh = cv2.imdecode(np.frombuffer(resp_sh.content, np.uint8), cv2.IMREAD_COLOR)
        assert img_sh is not None
        assert img_sh.ndim == 3
        assert img_sh.shape[0] > 0 and img_sh.shape[1] > 0

        # Normal map and SH lighting buffers should differ from each other and final composite
        payload_final = live_payload(fixture, solo_node="final")
        resp_final = client.post("/api/preview", json=payload_final)
        assert resp_final.status_code == 200
        assert resp_normal.content != resp_sh.content
