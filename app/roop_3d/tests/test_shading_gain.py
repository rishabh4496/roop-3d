"""Per-pixel lightness gain: the swapped face follows the target's low-frequency shading, rim included."""
import cv2
import numpy as np
import pytest

from roop_3d.compositor.quality import shading_gain_match

SIDE = 384
YY, XX = np.mgrid[:SIDE, :SIDE].astype(np.float32)
FACE = ((XX - 192) / 120) ** 2 + ((YY - 200) / 150) ** 2 < 1
ALPHA = cv2.GaussianBlur(FACE.astype(np.float32), (0, 0), 6)


def luma(image):
    return image.astype(np.float32) @ np.array((0.114, 0.587, 0.299), np.float32)


def scene():
    """Target: lit from the right (dark left cheek), with dark hair wrapped around the face. Swap: flat studio skin."""
    light = 0.55 + 0.45 * (XX / SIDE)                                  # left 0.55 .. right 1.0 across the frame
    target = np.zeros((SIDE, SIDE, 3), np.float32)
    skin = np.array((120, 150, 200), np.float32)
    target[:] = skin * light[..., None]
    hair = ~FACE & (((XX - 192) / 150) ** 2 + ((YY - 200) / 185) ** 2 < 1)
    target[hair] = (25, 30, 40)
    target = cv2.GaussianBlur(target, (0, 0), 3)
    rng = np.random.default_rng(0)
    texture = rng.normal(0, 6, (SIDE, SIDE, 1)).astype(np.float32)
    swap = np.clip(skin * 0.95 + cv2.GaussianBlur(texture, (0, 0), 1.2)[..., None], 0, 255)
    return swap.astype(np.uint8), np.clip(target, 0, 255).astype(np.uint8)


def side_ratio(image):
    inner = cv2.erode(FACE.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
    left, right = inner & (XX < 192 - 25), inner & (XX > 192 + 25)
    return luma(image)[left].mean() / luma(image)[right].mean()


def low_pass_correlation(a, b):
    inner = cv2.erode(FACE.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)
    la, lb = (cv2.GaussianBlur(luma(x), (0, 0), 12)[inner] for x in (a, b))
    return float(np.corrcoef(la, lb)[0, 1])


def test_left_right_lighting_and_low_frequency_shape_follow_the_target():
    swap, target = scene()
    out = shading_gain_match(swap, target, ALPHA)
    assert side_ratio(swap) == pytest.approx(1.0, abs=0.03)
    assert side_ratio(out) == pytest.approx(side_ratio(target), abs=0.04)          # was ~1.0 against ~0.75
    assert low_pass_correlation(out, target) > low_pass_correlation(swap, target) + 0.2
    assert low_pass_correlation(out, target) > 0.95


def test_the_rim_continues_into_the_surrounding_head_luma():
    swap, target = scene()
    out = shading_gain_match(swap, target, ALPHA)
    rim = cv2.erode(FACE.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~cv2.erode(
        FACE.astype(np.uint8), np.ones((29, 29), np.uint8)).astype(bool)
    before, after, goal = (luma(x)[rim].mean() for x in (swap, out, target))
    assert abs(after - goal) < abs(before - goal) * 0.6


def test_only_low_frequencies_change_and_the_mean_is_left_to_the_colour_step():
    swap, target = scene()
    out = shading_gain_match(swap, target, ALPHA)
    inner = ALPHA > 0.9
    assert luma(out)[inner].mean() == pytest.approx(luma(swap)[inner].mean(), rel=0.03)
    high = lambda x: luma(x) - cv2.GaussianBlur(luma(x), (0, 0), 4)
    assert high(out)[inner].std() == pytest.approx(high(swap)[inner].std(), rel=0.2)   # multiplicative, so detail scales with local light


def test_matching_light_is_a_no_op_and_strength_zero_is_exact():
    swap, target = scene()
    same = shading_gain_match(target, target, ALPHA)
    assert np.abs(same.astype(int) - target).max() <= 2
    np.testing.assert_array_equal(shading_gain_match(swap, target, ALPHA, strength=0.0), swap)
    half = shading_gain_match(swap, target, ALPHA, strength=0.5)
    assert abs(side_ratio(swap)) > side_ratio(half) > side_ratio(shading_gain_match(swap, target, ALPHA))


def test_gain_is_bounded_so_a_dark_hair_patch_cannot_black_out_the_face():
    swap, target = scene()
    target = target.copy()
    target[:150] = 0                                                              # an occluder over the whole forehead
    out = shading_gain_match(swap, target, ALPHA)
    assert luma(out)[ALPHA > 0.9].min() > 0.55 * luma(swap)[ALPHA > 0.9].min() * 0.6
