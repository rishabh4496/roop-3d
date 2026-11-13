"""G4: the identity-asset mesh response carries a ``parametric`` block built only from what the builder fitted."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import identity_world as W  # noqa: E402

from roop_3d.core.pose import rotation_from_euler  # noqa: E402
from roop_3d.reconstruction.geometry_types import ConfidenceState  # noqa: E402
from roop_3d.reconstruction.identity_asset import IdentityAssetSettings, build_identity_asset, parametric_block  # noqa: E402


@pytest.fixture(scope="module")
def asset():
    subject = W.make_subject("p68", W.LAYOUT_68, 91)
    views = [W.make_view(subject, i, y) for i, y in enumerate((0, -40, 40, 12))]
    return build_identity_asset(W.make_faceset("param68", views), IdentityAssetSettings(uv_size=64, subdivisions=1))


def test_parametric_contract_is_json_serialisable_and_honest(asset):
    import json
    block = parametric_block(asset)
    json.dumps(block)
    assert set(block) >= {"identity_shape_alpha", "expression_beta", "camera_pose", "albedo_uv"}
    # canonical template: no parametric basis was fitted -> null + reason, never zeros
    shape = block["identity_shape_alpha"]
    assert shape["value"] is None and shape["reason"]
    beta = block["expression_beta"]
    assert beta["value"] is None and "FLAME" in beta["reason"] and beta["measurement_fields"]
    pose = block["camera_pose"]
    assert pose["t"] is None and "weak-perspective" in pose["reason"]
    yaw, pitch, roll = (pose["euler_deg"][k] for k in ("yaw", "pitch", "roll"))
    np.testing.assert_allclose(pose["R"], rotation_from_euler(yaw, pitch, roll), atol=1e-5)
    assert abs(np.linalg.det(np.array(pose["R"])) - 1.0) < 1e-5


def test_albedo_summary_matches_the_atlas(asset):
    albedo = parametric_block(asset)["albedo_uv"]
    h, w = asset.texture.shape[:2]
    assert albedo["size"] == [w, h]
    observed = asset.texel_state == int(ConfidenceState.OBSERVED)
    assert albedo["observed_texel_fraction"] == pytest.approx(observed.mean(), abs=1e-5)
    assert observed.any() and albedo["mean_rgb"] is not None
    expected = asset.texture[..., :3][observed].mean(axis=0)[::-1]            # BGR -> RGB
    np.testing.assert_allclose(albedo["mean_rgb"], expected, atol=1e-2)


def test_unobserved_atlas_reports_null_mean_with_reason(asset):
    import copy
    blank = copy.copy(asset)
    blank.texel_state = np.zeros_like(asset.texel_state)
    albedo = parametric_block(blank)["albedo_uv"]
    assert albedo["mean_rgb"] is None and albedo["mean_rgb_reason"] and albedo["observed_texel_fraction"] == 0.0
