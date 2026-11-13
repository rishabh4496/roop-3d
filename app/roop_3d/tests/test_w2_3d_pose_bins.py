"""G2: one spherical bin grid, great-circle reference weights, shared legacy coverage grid."""
from __future__ import annotations

import math

import pytest

from roop_3d.core.errors import DomainError
from roop_3d.reconstruction import pose_bins as pb


def test_grid_shape_and_names():
    assert pb.TOTAL_BINS == 65 == len(pb.ALL_BINS) == len({b.name for b in pb.ALL_BINS})
    assert pb.classify(0, 0).name == "yaw0/pitch0"
    assert pb.classify(45, 0).name == "yaw+45/pitch0"
    assert pb.classify(-60, -15).name == "yaw-60/pitch-15"


@pytest.mark.parametrize("yaw,pitch,expected", [
    (7.4, 7.4, (0, 0)), (7.5, 7.5, (15, 15)), (-7.5, -7.5, (-15, -15)),     # ties go away from zero, symmetric
    (52.0, -22.4, (45, -15)), (80.0, 0, (75, 0)), (82.6, 0, (90, 0)),
    (140.0, 80.0, (90, 30)), (-140.0, -80.0, (-90, -30)),                    # clamped to the outermost bin
])
def test_classification_is_nearest_lattice_centre(yaw, pitch, expected):
    found = pb.classify(yaw, pitch)
    assert (found.yaw_deg, found.pitch_deg) == expected
    assert pb.classify(yaw, pitch) == found                                  # deterministic


def test_non_finite_pose_is_refused():
    with pytest.raises(DomainError):
        pb.classify(float("nan"), 0)


def test_weights_sum_to_one_and_peak_on_the_nearest_bin():
    available = [pb.classify(y, 0) for y in (0, 15, 30, 45, 60, 75)]
    weights = pb.reference_weights(33.0, 0.0, available)
    assert len(weights) == 3 and sum(weights.values()) == pytest.approx(1.0)
    assert next(iter(weights)) == "yaw+30/pitch0"
    assert set(weights) == {"yaw+15/pitch0", "yaw+30/pitch0", "yaw+45/pitch0"}
    # exactly on a bin with a single available bin -> weight 1
    assert pb.reference_weights(45, 0, ["yaw+45/pitch0"]) == {"yaw+45/pitch0": 1.0}


def test_weight_values_follow_the_gaussian_of_the_great_circle_angle():
    a, b = pb.classify(0, 0), pb.classify(30, 0)
    w = pb.reference_weights(10.0, 0.0, [a, b], top_k=2)
    ta, tb = 10.0, 20.0
    ea, eb = math.exp(-ta ** 2 / (2 * 18 ** 2)), math.exp(-tb ** 2 / (2 * 18 ** 2))
    assert w[a.name] == pytest.approx(ea / (ea + eb), rel=1e-6)


def test_great_circle_angle_is_the_angle_between_view_directions():
    assert pb.great_circle_deg(0, 0, 60, 0) == pytest.approx(60.0)
    assert pb.great_circle_deg(0, 0, 45, 45) == pytest.approx(60.0)     # n = (sin45, -sin45 cos45, cos45 cos45) -> cos = 1/2
    assert pb.great_circle_deg(0, 30, 0, -30) == pytest.approx(60.0)


def test_empty_availability_is_an_explicit_error():
    with pytest.raises(DomainError) as error:
        pb.reference_weights(0, 0, [])
    assert error.value.code == "POSE_BINS_EMPTY"


def test_legacy_coverage_grid_is_unchanged_and_shared():
    from roop_3d import pose_matrix
    from roop_3d.reconstruction import faceset_3d
    assert (faceset_3d.YAW_BINS, faceset_3d.PITCH_BINS, faceset_3d.TOTAL_BINS) == (12, 8, 96)
    assert (pose_matrix.YAW_BINS, pose_matrix.PITCH_BINS) == (12, 8)
    # original formula values: centre cell, extremes, clamping
    assert faceset_3d.pose_bin(0, 0) == 4 * 12 + 6
    assert faceset_3d.pose_bin(-90, -60) == 0 and faceset_3d.pose_bin(200, 200) == 96 - 1
    assert pose_matrix._sphere_bin(30, -20) == faceset_3d.pose_bin(30, -20)
