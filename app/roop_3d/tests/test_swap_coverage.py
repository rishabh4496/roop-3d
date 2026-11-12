"""Coverage of the dense swap: resolution-independent feather, landmark-hull bound, no nearest-neighbour stairs."""
import cv2
import numpy as np
import pytest

from roop_3d.reconstruction.swap_coverage import feathered_coverage, hull_mask, hull_points


def disc(side, radius_fraction=0.36):
    surface = np.zeros((side, side), np.uint8)
    cv2.circle(surface, (side // 2, side // 2), int(side * radius_fraction), 1, -1)
    return surface.astype(bool)


def ramp_width_fraction(alpha):
    """Width of the 0.1 -> 0.9 transition along the centre row, as a fraction of the image."""
    row = alpha[alpha.shape[0] // 2, alpha.shape[1] // 2:]
    return (np.argmax(row < 0.1) - np.argmax(row < 0.9)) / alpha.shape[1]


def test_feather_width_follows_face_extent_at_any_resolution():
    small, large = feathered_coverage(disc(256)), feathered_coverage(disc(1024))
    assert ramp_width_fraction(small) == pytest.approx(ramp_width_fraction(large), rel=0.15)
    assert ramp_width_fraction(large) > 0.06            # ~12 % of the face, not the old 1 %-of-width ramp
    smaller_face = feathered_coverage(disc(1024, 0.18))
    assert ramp_width_fraction(smaller_face) == pytest.approx(ramp_width_fraction(large) / 2, rel=0.2)


def test_silhouette_is_not_a_nearest_neighbour_staircase():
    # A silhouette rasterised at 64 px and blown up 8x with INTER_NEAREST moves its edge in 8 px steps.
    stairs = cv2.resize(disc(64).astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST).astype(bool)
    alpha = feathered_coverage(stairs)
    assert np.abs(np.diff(alpha, axis=1)).max() < 0.05
    assert np.abs(np.diff(alpha, axis=0)).max() < 0.05


def head_landmarks():
    """A 106-point-like outline (ellipse) and the five points of a 512 px crop."""
    t = np.linspace(0, 2 * np.pi, 106, endpoint=False)
    outline = np.column_stack((256 + 120 * np.cos(t), 280 + 150 * np.sin(t)))
    five = np.array(((200, 230), (312, 230), (256, 290), (210, 350), (302, 350)), np.float64)
    return outline, five


def test_mesh_overshoot_beyond_the_landmark_hull_never_paints_background():
    outline, five = head_landmarks()
    surface = np.zeros((512, 512), bool)
    surface[40:480, 40:472] = True                       # the mesh is far larger than the head outline
    alpha = feathered_coverage(surface, landmarks106=outline, landmarks5=five)
    assert alpha[280, 256] > 0.95                        # inside the hull
    assert alpha[280, 60] < 0.02 and alpha[470, 256] < 0.02    # mesh overshoot beside / below the head
    unbounded = feathered_coverage(surface)
    assert unbounded[280, 60] > 0.5


def test_forehead_extension_lies_above_the_brows_along_the_eye_to_mouth_axis():
    outline, five = head_landmarks()
    extra = hull_points(outline, five)[len(outline):]
    assert len(extra) == 3 and (extra[:, 1] < outline[:, 1].min()).all()
    # a rolled head pushes the extension along its own up axis, not straight up the image
    angle = np.radians(40)
    rotate = np.array(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
    centre = np.array((256.0, 280.0))
    rolled_extra = hull_points((outline - centre) @ rotate.T + centre, (five - centre) @ rotate.T + centre)[len(outline):]
    expected = (extra - centre) @ rotate.T + centre
    np.testing.assert_allclose(rolled_extra, expected, atol=1e-6)


def test_collapsed_landmark_hull_is_ignored_not_trusted():
    outline, five = head_landmarks()
    surface = disc(512, 0.3)
    collapsed = np.tile(np.array((256.0, 280.0)), (106, 1)) + np.random.default_rng(1).normal(0, 3, (106, 2))
    alpha = feathered_coverage(surface, landmarks106=collapsed, landmarks5=five)
    np.testing.assert_array_equal(alpha, feathered_coverage(surface))


def test_empty_surface_and_hull_mask_edge_cases():
    assert not feathered_coverage(np.zeros((64, 64), bool)).any()
    assert hull_mask((64, 64), np.array(((1.0, 1.0), (5.0, 5.0)))) is None
