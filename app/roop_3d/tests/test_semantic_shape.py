"""Semantic shape sliders: every claim is re-measured from the *mesh*, not from the Jacobian the controller solved with.

Runs on the CC0 MakeHuman basis (skipped until it is built) and, when installed, on FLAME 2023 Open through the bridge.
"""
from __future__ import annotations

import numpy as np
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.reconstruction.makehuman_build import ASSET_NAME, DEFAULT_ASSET_DIR
from roop_3d.reconstruction.semantic_shape import (
    NUISANCE_KEYS,
    SLIDER_KEYS,
    SLIDERS,
    SemanticShapeController,
    slider_definitions,
)
from roop_3d.reconstruction.topology import MorphableTopologyRegistry

REGISTRY = MorphableTopologyRegistry()
needs_makehuman = pytest.mark.skipif(not (DEFAULT_ASSET_DIR / ASSET_NAME).exists(), reason="build the MakeHuman asset first")
needs_flame = pytest.mark.skipif(
    not ((DEFAULT_ASSET_DIR / ASSET_NAME).exists() and REGISTRY.availability("flame2023_open")["available"]),
    reason="needs the MakeHuman asset and FLAME 2023 Open")
UNIT = {s.key: s.unit_mm for s in SLIDERS}


@pytest.fixture(scope="module")
def mh():
    return SemanticShapeController.for_topology(REGISTRY, "makehuman_cc0")


@pytest.fixture(scope="module")
def flame():
    return SemanticShapeController.for_topology(REGISTRY, "flame2023_open")


def measured_change(ctl, values):
    new, result = ctl.apply(ctl.rest, values)
    before, after = ctl.measure(ctl.rest), ctl.measure(new)
    return {k: after[k] - before[k] for k in ctl.keys}, result, new


def test_slider_catalogue_matches_the_requested_six():
    assert SLIDER_KEYS == ("jaw_width", "cheekbone_prominence", "chin_projection", "nose_bridge", "brow_arch", "face_fullness")
    defs = {d["key"]: d for d in slider_definitions()}
    assert set(defs) == set(SLIDER_KEYS) and all(d["unit_mm"] > 0 and d["minimum"] < 0 < d["maximum"] for d in defs.values())
    assert set(NUISANCE_KEYS) == {"head_width", "face_height", "face_depth"}


# --------------------------------------------------------------------------------------------------------- MakeHuman
@needs_makehuman
def test_rest_measures_are_anatomically_plausible(mh):
    m = mh.measure(mh.rest)
    assert 60 < m["jaw_width"] < 130 and 80 < m["head_width"] < 160          # millimetres, adult head
    assert 60 < m["face_height"] < 130 and 25 < m["face_depth"] < 80
    assert m["nose_bridge"] > 8, "the bridge stands proud of the inner eye corners (+z toward the viewer)"
    assert m["brow_arch"] > 0, "brow peaks sit above the line through their ends (+y is down, so 'up' is -y)"
    assert m["chin_projection"] < 5, "the chin tip does not stand in front of the lower lip on the neutral head"


@needs_makehuman
@pytest.mark.parametrize("key", SLIDER_KEYS)
def test_each_slider_moves_only_its_own_measure(mh, key):
    change, result, _ = measured_change(mh, {key: 1.5})
    assert change[key] == pytest.approx(1.5 * UNIT[key], abs=0.05), "re-measured on the mesh, not on the Jacobian"
    for other in mh.keys:
        if other != key:
            assert abs(change[other]) < 0.05, f"{key} leaked {change[other]:.3f} mm into {other}"
    assert not result.clamped and result.limit_scale == 1.0


@needs_makehuman
@pytest.mark.parametrize("key", SLIDER_KEYS)
def test_slider_is_signed_and_symmetric(mh, key):
    up, _, _ = measured_change(mh, {key: 2.0})
    down, _, _ = measured_change(mh, {key: -2.0})
    assert up[key] == pytest.approx(-down[key], abs=0.05) and up[key] > 0


@needs_makehuman
def test_sliders_superpose_and_never_resize_the_head(mh):
    values = {"jaw_width": 1.0, "chin_projection": -1.5, "nose_bridge": 2.0, "brow_arch": 1.0, "face_fullness": -1.0, "cheekbone_prominence": 0.5}
    combined, _, _ = measured_change(mh, values)
    for key, v in values.items():
        assert combined[key] == pytest.approx(v * UNIT[key], abs=0.1)
    for key in NUISANCE_KEYS:
        assert abs(combined[key]) < 0.1, "head size is held fixed while a slider moves"


@needs_makehuman
def test_zero_is_a_no_op(mh):
    new, result = mh.apply(mh.rest, {})
    assert np.array_equal(new, mh.rest) and not np.any(result.delta_beta) and not result.clamped


@needs_makehuman
def test_coefficient_limit_clamps_the_whole_move_and_says_so():
    probe = SemanticShapeController.for_topology(REGISTRY, "makehuman_cc0")
    tight = SemanticShapeController(probe.rest, REGISTRY.prepared("makehuman_cc0").faces, probe.basis, _landmarks(probe),
                                    coeff_limit=0.05, topology_id="makehuman_cc0")
    result = tight.solve({"jaw_width": 3.0, "chin_projection": 3.0})
    assert result.clamped and result.limit_scale < 1.0
    assert np.abs(result.delta_beta).max() <= 0.05 + 1e-9, "the model's coefficient range is never exceeded"
    assert 0 < result.achieved_mm["jaw_width"] < 3.0 * UNIT["jaw_width"]
    assert result.achieved_mm["jaw_width"] / (3 * UNIT["jaw_width"]) == pytest.approx(result.limit_scale, rel=1e-4)
    assert result.public()["clamped"] is True


def _landmarks(ctl):
    from scipy import sparse
    lm = REGISTRY.prepared("makehuman_cc0").landmarks["68pt"]
    return sparse.csr_matrix((np.ones(68), (np.arange(68), lm)), shape=(68, len(ctl.rest)))


@needs_makehuman
@pytest.mark.parametrize("values,code", [({"jaw_width": 9.0}, "SEMANTIC_SLIDER_RANGE"), ({"nose_bridge": float("nan")}, "SEMANTIC_SLIDER_RANGE"),
                                         ({"jaw_width": float("inf")}, "SEMANTIC_SLIDER_RANGE"), ({"cheek_size": 1.0}, "SEMANTIC_SLIDER_UNKNOWN")])
def test_invalid_sliders_are_domain_errors(mh, values, code):
    with pytest.raises(DomainError) as error:
        mh.solve(values)
    assert error.value.code == code and error.value.status == 422


@needs_makehuman
def test_diagnostics_report_decoupling_and_reachability(mh):
    d = mh.diagnostics()
    assert d["topology"] == "makehuman_cc0" and d["basis_axes"] >= 40
    assert all(v > 0.99 for v in d["reachable_fraction"].values())
    assert all(v < 0.01 for v in d["max_leakage_mm_per_unit"].values())
    assert 1.0 <= d["jacobian_condition"] < 1e4 and 0.0 <= d["max_offdiag_cosine"] <= 1.0
    assert set(d["rest_measures_mm"]) == set(SLIDER_KEYS) | set(NUISANCE_KEYS)


@needs_makehuman
def test_displacement_is_what_the_basis_says(mh):
    result = mh.solve({"jaw_width": 1.0})
    expected = np.einsum("vck,k->vc", mh.basis.astype(np.float64), result.delta_beta.astype(np.float32).astype(np.float64))
    assert np.allclose(mh.displacement(result.delta_beta), expected, atol=1e-6)
    assert 0 < np.linalg.norm(mh.displacement(result.delta_beta), axis=1).max() < 0.03, "a 4 mm slider unit moves vertices by millimetres"


def test_controller_rejects_a_missing_basis_or_landmarks():
    from scipy import sparse
    v = np.zeros((10, 3))
    with pytest.raises(DomainError) as error:
        SemanticShapeController(v, np.zeros((1, 3), int), None, sparse.csr_matrix((68, 10)), coeff_limit=1.0)
    assert error.value.code == "SEMANTIC_BASIS_MISSING"
    with pytest.raises(DomainError) as error:
        SemanticShapeController(v, np.zeros((1, 3), int), np.zeros((10, 3, 4), np.float32), sparse.csr_matrix((5, 10)), coeff_limit=1.0)
    assert error.value.code == "SEMANTIC_LANDMARKS_MISSING"


# ------------------------------------------------------------------------------------------------------------ FLAME
@needs_flame
@pytest.mark.parametrize("key", SLIDER_KEYS)
def test_flame_sliders_are_decoupled_on_the_mesh(flame, key):
    change, result, _ = measured_change(flame, {key: 1.0})
    assert change[key] == pytest.approx(result.achieved_mm[key], abs=0.05)
    assert change[key] == pytest.approx(UNIT[key], abs=0.05) or result.clamped
    for other in flame.keys:
        if other != key:
            assert abs(change[other]) < 0.05, f"{key} leaked {change[other]:.3f} mm into {other}"


@needs_flame
def test_flame_uses_its_own_basis_with_a_three_sigma_limit(flame):
    assert flame.topology_id == "flame2023_open" and flame.coeff_limit == 3.0 and flame.basis.shape[2] == 300
    d = flame.diagnostics()
    assert d["basis_axes"] == 300 and all(v > 0.98 for v in d["reachable_fraction"].values())
    result = flame.solve({"brow_arch": 3.0})
    assert np.abs(result.delta_beta).max() <= 3.0 + 1e-9


@needs_flame
def test_flame_landmarks_are_carried_across_the_bridge(flame):
    m = flame.measure(flame.rest)
    assert 60 < m["jaw_width"] < 140 and m["nose_bridge"] > 8 and m["brow_arch"] > 0
    assert 80 < m["head_width"] < 160 and 60 < m["face_height"] < 130
