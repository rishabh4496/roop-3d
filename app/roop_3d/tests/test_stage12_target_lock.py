"""Stage 12: rank-preserved person clusters, pose-bridging bank, face analysis validity rules."""
from __future__ import annotations

import numpy as np

from roop_3d.identity.target_lock import ClusterRegistry, FaceRef, TargetLockStore
from roop_3d.media.face_analysis import agrees_with_106, five_point_plausible, thumbnail_b64


def unit(v) -> np.ndarray:
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def person(seed: int) -> np.ndarray:
    return unit(np.random.default_rng(seed).normal(size=512))


def toward(base: np.ndarray, other: np.ndarray, cosine: float) -> np.ndarray:
    """A unit vector with the given cosine to `base`."""
    ortho = unit(other - (other @ base) * base)
    return unit(cosine * base + np.sqrt(1 - cosine ** 2) * ortho)


def ref(embedding, yaw=0.0, x=0.0) -> FaceRef:
    return FaceRef(embedding, yaw, 0.0, 0.0, [x, 0.0, x + 100.0, 100.0])


def test_cluster_ids_follow_the_person_not_the_left_to_right_order():
    a, b = person(1), person(2)
    reg = ClusterRegistry()
    first = reg.associate([ref(a, x=0), ref(b, x=300)], 0)
    swapped = reg.associate([ref(toward(b, a, 0.9), x=0), ref(toward(a, b, 0.9), x=300)], 1)
    assert [x.cluster_id for x in first] == [1, 2]
    assert [x.cluster_id for x in swapped] == [2, 1]                      # order swapped, ids did not


def test_profile_is_bridged_by_margin_when_only_weakly_similar():
    a, b = person(3), person(4)
    reg = ClusterRegistry()
    reg.associate([ref(a)], 0)
    profile_a = toward(a, person(5), 0.47)                                # weak, below STRONG
    out = reg.associate([ref(profile_a, yaw=70, x=0), ref(b, x=300)], 1)
    assert out[0].cluster_id == 1 and out[0].reason == "pose_bridge"
    assert out[1].cluster_id == 2                                         # the other person starts their own cluster


def test_ambiguous_weak_match_is_not_merged():
    a = person(6)
    reg = ClusterRegistry()
    reg.associate([ref(a)], 0)
    c2 = reg.associate([ref(person(7))], 1)[0].cluster_id
    # a face that is equally close to both clusters must not be assigned to either (fragment, don't merge)
    c = reg.clusters[c2].bank[0].embedding
    both = unit(0.45 * a + 0.45 * c + 0.77 * unit(np.random.default_rng(99).normal(size=512)))
    assert 0.40 <= float(both @ a) and 0.40 <= float(both @ c)
    out = reg.associate([ref(both)], 2)[0]
    assert out.reason == "new_cluster" and out.cluster_id not in (1, c2)


def test_bank_grows_by_chain_linking_and_skips_redundant_poses():
    a = person(8)
    reg = ClusterRegistry()
    reg.associate([ref(a, yaw=0)], 0)
    mid = toward(a, person(9), 0.70)
    assert reg.associate([ref(mid, yaw=35)], 1)[0].cluster_id == 1
    far = toward(mid, person(10), 0.62)                                   # only similar to the 35 deg entry, not to the anchor
    cos_to_anchor = float(far @ a)
    out = reg.associate([ref(far, yaw=70)], 2)[0]
    assert out.cluster_id == 1, f"chain link failed (cosine to anchor {cos_to_anchor:.2f})"
    before = len(reg.clusters[1].bank)
    reg.associate([ref(toward(a, person(11), 0.95), yaw=2)], 3)           # same pose as the anchor: no new entry
    assert len(reg.clusters[1].bank) == before


def test_lock_creation_is_idempotent_for_one_person_and_numbers_targets():
    a, b = person(12), person(13)
    store = TargetLockStore()
    l1 = store.create("v.mp4", 0, ref(a))
    again = store.create("v.mp4", 5, ref(toward(a, person(99), 0.9)))
    l2 = store.create("v.mp4", 0, ref(b, x=300))
    assert again.lock_id == l1.lock_id and (l1.target_number, l2.target_number) == (1, 2)
    assert l1.public(store.registry("v.mp4"))["bank_size"] >= 1


def test_five_point_validity_and_106_agreement():
    box = np.array([100.0, 100.0, 200.0, 220.0])
    good = np.array([[125, 140], [175, 140], [150, 165], [130, 190], [170, 190]], float)
    assert five_point_plausible(box, good) == (True, "ok")
    assert five_point_plausible(box, good[::-1])[0] is False              # vertical order broken
    assert five_point_plausible(np.array([0, 0, 10, 10.0]), good)[1] == "face_too_small"
    dense = np.vstack([good + 1.0, np.random.default_rng(0).uniform(100, 200, (101, 2))])
    assert agrees_with_106(box, good, dense)[0] is True
    far = np.tile([[900.0, 900.0]], (106, 1))
    assert agrees_with_106(box, good, far)[0] is False                     # 106-point model disagrees: a hallucinated profile is rejected
    assert agrees_with_106(box, good, None)[0] is True                     # no independent model: the geometry check decides


def test_thumbnail_is_a_small_jpeg_inside_the_frame():
    image = np.full((200, 300, 3), 90, np.uint8)
    data = thumbnail_b64(image, np.array([100.0, 40.0, 180.0, 140.0]))
    assert data and len(data) < 6000
    assert thumbnail_b64(image, np.array([299.0, 199.0, 300.0, 200.0])) is None
