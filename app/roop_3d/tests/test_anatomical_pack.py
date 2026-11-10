"""Unit tests for the CC0 full-head synthesis and renderer-neutral dynamics."""
from __future__ import annotations

import json

import numpy as np
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.reconstruction import anatomical_pack as ap


@pytest.fixture(scope="module")
def asset() -> ap.AnatomicalFullHeadAsset:
    vertices, faces = ap._analytic_fallback_head()
    return ap.AnatomicalFullHeadAsset.from_template(vertices, faces, template_source="test_cc0")


def test_pack_has_60_orthogonal_named_craniofacial_targets(asset):
    assert asset.num_morphs == 60
    assert len(set(asset.morph_names)) == 60
    assert set(ap.MORPH_TARGET_NAMES) == set(asset.morph_names)
    flattened = asset.morph_targets.reshape(60, -1).astype(np.float64)
    norms = np.linalg.norm(flattened, axis=1)
    correlation = (flattened @ flattened.T) / np.outer(norms, norms)
    np.fill_diagonal(correlation, 0.0)
    assert np.max(np.abs(correlation)) < 2e-5
    assert set(ap.MORPH_GROUPS) == {"forehead", "temples", "zygoma", "nose", "mandible", "ears", "neck", "craniofacial"}


def test_head_morphs_and_anatomical_rig_respond_to_controls(asset):
    neutral = asset.morph()
    changed = asset.morph({"forehead_height": 0.8, "nose_tip_projection": 0.5})
    assert not np.allclose(neutral, changed)
    closed = asset.articulate(theta_jaw=0.0, psi_lip=0.0)
    open = asset.articulate(theta_jaw=np.radians(35), psi_lip=0.8)
    assert np.allclose(closed["upper_teeth"], open["upper_teeth"])
    assert not np.allclose(closed["lower_teeth"], open["lower_teeth"])
    assert not np.allclose(closed["tongue"], open["tongue"])
    assert {"eyeballs", "corneas", "eyelashes", "eyebrows", "hair_cards"}.issubset(asset.submeshes)
    assert asset.submeshes["eyeballs"].uvs.shape == (len(asset.submeshes["eyeballs"].vertices), 2)
    assert "ior_1.376" in asset.submeshes["corneas"].material
    assert asset.submeshes["corneas"].uvs.shape == (len(asset.submeshes["corneas"].vertices), 2)
    assert asset.submeshes["eyelashes"].alpha is not None
    assert asset.submeshes["eyebrows"].alpha is not None
    assert asset.submeshes["hair_cards"].spring_weights is not None


def test_cornea_parallax_uses_refraction_and_is_centered_on_axis(asset):
    uv = np.array([[0.5, 0.5], [0.62, 0.42]], dtype=np.float32)
    frontal = asset.cornea_parallax_uv(uv, np.array([0.0, 0.0, -1.0]))
    oblique = asset.cornea_parallax_uv(uv, np.array([0.4, 0.1, -0.91]))
    assert np.allclose(frontal[0], uv[0], atol=1e-7)
    assert np.linalg.norm(oblique[0] - uv[0]) > 1e-4
    assert np.linalg.norm(oblique[1] - uv[1]) > 1e-5
    assert np.all((oblique >= 0.0) & (oblique <= 1.0))


def test_analytical_cavity_ao_opens_with_jaw_and_keeps_nostril_shadow(asset):
    ao = asset.cavity_ao
    mouth_samples = np.array([ao.mouth_center])
    nostril_samples = ao.nostril_centers.copy()
    closed = ao.evaluate(0.0, mouth_samples, cavity="oral")[0]
    half_open = ao.evaluate(np.radians(35), mouth_samples, cavity="oral")[0]
    wide_open = ao.evaluate(np.radians(70), mouth_samples, cavity="oral")[0]
    nose_closed = ao.evaluate(0.0, nostril_samples, cavity="nostrils")
    nose_open = ao.evaluate(np.radians(70), nostril_samples, cavity="nostrils")
    assert closed > half_open > wide_open
    assert np.all(nose_closed > 0.6)
    assert np.allclose(nose_closed, nose_open)


def test_soft_tissue_filter_is_critically_damped_and_returns_to_rest():
    points = np.array([[0.0, 0.0, 0.0], [0.08, 0.0, 0.0], [-0.08, 0.0, 0.0]])
    weights = np.array([1.0, 0.5, 0.25])
    spring = ap.SoftTissueSpringFilter(natural_frequency_hz=4.0)
    acceleration = np.array([0.0, 9.81, 0.0, 0.0, 0.0, 0.0])
    samples = [spring.update(points, acceleration, 1 / 60, weights) for _ in range(90)]
    final = samples[-1]
    magnitude = np.linalg.norm(final, axis=1)
    assert np.all(magnitude > 0)
    assert np.linalg.norm(samples[0]) < np.linalg.norm(final)
    assert np.linalg.norm(final - samples[-2]) < 1e-4
    rest = np.zeros_like(final)
    for _ in range(90):
        rest = spring.update(points, np.zeros(6), 1 / 60, weights)
    assert np.linalg.norm(rest) < 1e-4


def test_asset_roundtrip_verifies_digest_and_is_pickle_free(asset, tmp_path):
    path = asset.save(tmp_path / ap.ASSET_NAME)
    loaded = ap.AnatomicalFullHeadAsset.load(path, auto_build=False)
    assert loaded.license_id == "CC0-1.0"
    assert loaded.template_source == "test_cc0"
    assert loaded.morph_targets.shape == asset.morph_targets.shape
    assert set(loaded.submeshes) == set(asset.submeshes)
    with np.load(path, allow_pickle=False) as arrays:
        arrays = {key: arrays[key] for key in arrays.files}
    arrays["vertices"] = arrays["vertices"].copy()
    arrays["vertices"][0, 0] += 0.01
    np.savez_compressed(path, **arrays)
    with pytest.raises(DomainError, match="content digest mismatch"):
        ap.AnatomicalFullHeadAsset.load(path, auto_build=False)


def test_pack_builder_uses_local_analytic_fallback_without_network(monkeypatch, tmp_path):
    monkeypatch.setattr(ap, "_local_makehuman_template", lambda: (*ap._analytic_fallback_head(), None, None,
                                                                     "analytic_cc0_offline_fallback"))
    path = ap.build_anatomical_pack(tmp_path / ap.ASSET_NAME)
    result = ap.AnatomicalFullHeadAsset.load(path, auto_build=False)
    meta = json.loads(str(np.load(path, allow_pickle=False)["metadata"].item()))
    assert result.template_source == "analytic_cc0_offline_fallback"
    assert meta["license"] == "CC0-1.0"
    assert result.num_morphs == 60
