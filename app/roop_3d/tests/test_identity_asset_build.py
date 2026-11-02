"""Stage 3 Phases D+E: the identity asset (contents, honesty, GLB inspection) and its persistence / reuse rules."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import identity_world as W  # noqa: E402

from roop_3d.core.errors import DomainError  # noqa: E402
from roop_3d.reconstruction import identity_asset as ia  # noqa: E402
from roop_3d.reconstruction.geometry_types import ConfidenceState  # noqa: E402
from roop_3d.reconstruction.identity_asset import (  # noqa: E402
    IdentityAssetSettings,
    IdentityAssetStore,
    build_identity_asset,
    compute_build_key,
    hash_array,
)
from roop_3d.reconstruction.mesh_io import read_glb  # noqa: E402

OBS, INTERP, UNSEEN = (int(ConfidenceState.OBSERVED), int(ConfidenceState.INTERPOLATED), int(ConfidenceState.UNSEEN))
SETTINGS = IdentityAssetSettings(uv_size=W.ATLAS)
SMALL = IdentityAssetSettings(uv_size=64, subdivisions=1)


@pytest.fixture(scope="module")
def s68():
    return W.make_subject("b68", W.LAYOUT_68, 71)


@pytest.fixture(scope="module")
def s5():
    return W.make_subject("b5", W.LAYOUT_5, 72)


@pytest.fixture(scope="module")
def full_doc(s68):
    views = [W.make_view(s68, i, y, expression=("open" if i == 3 else None)) for i, y in enumerate((0, -40, 40, 12, -72, 72))]
    return W.make_faceset("full68", views)


@pytest.fixture(scope="module")
def full_asset(full_doc):
    return build_identity_asset(full_doc, SETTINGS)


# ----------------------------------------------------------------------------- front-only honesty
@pytest.mark.parametrize("layout", [W.LAYOUT_5, W.LAYOUT_68])
def test_front_only_does_not_claim_3d_recovery(layout):
    subject = W.make_subject("fo", layout, 73)
    asset = build_identity_asset(W.make_faceset("front_only", [W.make_view(subject, 0, 0.0)]), SETTINGS)
    m = asset.manifest
    assert m.reconstruction_level == "single_view_prior"
    assert m.claims["complete_3d_recovery"] is False and m.claims["depth_triangulated"] is False
    assert m.claims["back_of_head_recovered"] is False and m.claims["full_head_geometry"] is False
    assert any("Depth is not triangulated" in text for text in m.missing_information)
    assert any("three-quarter" in text for text in m.missing_information)
    assert (asset.vertex_state == OBS).sum() == 0, "no vertex may be OBSERVED when depth is unconstrained"
    assert float(asset.landmark_confidence.max()) < 0.15
    assert np.abs(asset.landmarks_3d[:, 2] - asset.template_landmarks[:, 2]).max() < 0.003
    assert float(np.abs(asset.shape_params[:, 2]).max()) < 0.003


def test_three_views_triangulate_depth_only_when_landmarks_allow_it(s68, s5):
    a68 = build_identity_asset(W.full_set(s68, (0, -40, 40)), SETTINGS)
    a5 = build_identity_asset(W.full_set(s5, (0, -40, 40)), SETTINGS)
    assert a68.manifest.claims["depth_triangulated"] is True and a68.manifest.reconstruction_level == "multiview"
    assert a5.manifest.claims["depth_triangulated"] is False and a5.manifest.reconstruction_level == "limited_multiview"
    assert any("Only 5 landmarks" in text for text in a5.manifest.limitations)
    assert a68.manifest.claims["complete_3d_recovery"] is False and a5.manifest.claims["complete_3d_recovery"] is False


def test_geometry_corresponds_to_the_supplied_identity(s68):
    other = W.make_subject("b68_other", W.LAYOUT_68, 74, width=-0.10, nose_depth=-0.008, jaw_drop=-0.008)
    mine = build_identity_asset(W.full_set(s68, (0, -40, 40, -72, 72)), SETTINGS)
    theirs = build_identity_asset(W.full_set(other, (0, -40, 40, -72, 72)), SETTINGS)
    err = lambda est, truth: float(np.linalg.norm(W.sim_align(est, truth) - truth, axis=1).mean() * 1000)
    template_err = err(mine.template_landmarks, s68.true_lm)
    assert err(mine.landmarks_3d, s68.true_lm) < 0.25 * template_err
    assert err(mine.landmarks_3d, s68.true_lm) < 0.3 * err(mine.landmarks_3d, other.true_lm)       # not a generic head
    assert err(theirs.landmarks_3d, other.true_lm) < 0.3 * err(theirs.landmarks_3d, s68.true_lm)
    # the full mesh follows the landmarks: its vertices differ between identities and from the template
    assert np.abs(mine.vertices - theirs.vertices).max() > 0.004 and np.abs(mine.vertices - W.identity_template(2).vertices).max() > 0.004


# ----------------------------------------------------------------------------- contents
def test_asset_contains_everything_the_spec_requires(full_asset, full_doc, s68):
    a, m = full_asset, full_asset.manifest
    n_v, n_f = len(a.vertices), len(a.faces)
    assert a.vertices.shape == (n_v, 3) and a.faces.shape == (n_f, 3) and a.faces.min() >= 0 and a.faces.max() < n_v
    assert a.normals.shape == (n_v, 3) and np.allclose(np.linalg.norm(a.normals, axis=1), 1.0, atol=1e-3)
    assert (a.normals[:, 2] > 0).mean() > 0.95                                  # outward = toward the viewer
    assert a.uvs.shape == (n_v, 2) and a.uvs.min() >= 0.0 and a.uvs.max() <= 1.0
    assert a.vertex_confidence.shape == (n_v,) and a.vertex_state.shape == (n_v,)
    assert a.texture.shape == (W.ATLAS, W.ATLAS, 3) and a.texel_state.shape == (W.ATLAS, W.ATLAS)
    assert a.texel_confidence.shape == a.texel_state.shape and a.texel_confidence.dtype == np.float32
    assert ((a.vertex_state == OBS) <= (a.vertex_confidence >= ia.OBSERVED_VERTEX_CONFIDENCE)).all()
    assert (a.vertex_confidence[a.vertex_state == UNSEEN] == 0).all()
    assert a.shape_params.shape == (68, 3) and a.landmarks_3d.shape == (68, 3) and a.landmark_sigma.shape == (68, 3)
    assert m.shape["flame_shape_coefficients"] is None and m.expression["flame_expression_coefficients"] is None
    assert m.flame["used"] is False and m.flame["available"] is False and m.model_hashes["flame_weights_sha256"] is None
    assert m.canonical_head_pose["yaw"] == 0.0 and m.canonical_head_pose["reference_view"] in (0, 3)
    assert m.model_version["asset_schema"] == ia.SCHEMA and m.model_version["algorithm"] == ia.ALGORITHM_VERSION
    assert m.model_hashes["template_geometry_sha256"] == ia._template_hash(SETTINGS.subdivisions)
    for name, array in a.array_fields.items():
        assert m.arrays_sha256[name] == hash_array(array)


def test_source_references_are_hashed_and_explain_every_exclusion(s68):
    views = [W.make_view(s68, i, y) for i, y in enumerate((0, -40, 40, -72))]
    views.append(views[0].model_copy(update={"index": 4}))                                   # exact duplicate
    views.append(W.make_view(s68, 5, 0.0, stored_pose=(55.0, 0.0, 0.0)))                      # pose contradicts landmarks
    asset = build_identity_asset(W.make_faceset("refs", views), SETTINGS)
    refs = {s.index: s for s in asset.manifest.sources}
    assert len(refs) == 6
    for view in views:
        assert refs[view.index].image_sha256 == hashlib.sha256(view.image_bytes).hexdigest()
    assert refs[4].used is False and refs[4].exclusion_reasons == ["DUPLICATE_IMAGE"]
    assert refs[5].used is False and "POSE_LANDMARK_MISMATCH" in refs[5].exclusion_reasons
    assert all(refs[i].used and refs[i].refined_yaw is not None and refs[i].reprojection_rms_mm is not None for i in range(4))


def test_expression_is_measured_per_view(full_asset):
    per_view = full_asset.manifest.expression["per_view"]
    opened, neutral = per_view["3"]["measurements"]["mouth_open_over_iod"], per_view["0"]["measurements"]["mouth_open_over_iod"]
    assert opened is not None and neutral is not None and opened > neutral + 0.05
    assert per_view["4"]["measurements"]["mouth_open_over_iod"] is None                       # |yaw| 72: too turned to measure
    assert np.isfinite(full_asset.expression_params[~np.isnan(full_asset.expression_params)]).all()


def test_missing_profile_images_are_reported(s68):
    asset = build_identity_asset(W.full_set(s68, (0, -40, 40)), SETTINGS)
    m = asset.manifest
    assert {"profile_image_left", "profile_image_right"} <= set(m.coverage["missing"])
    assert any("profile" in text for text in m.missing_information)
    assert m.coverage["readiness"] == "full_coverage" and m.coverage["yaw_coverage"]["profile_image_left"] == []


def test_unbuildable_facesets_fail_with_actionable_requests(s68):
    other = W.make_subject("b68_mix", W.LAYOUT_68, 75, width=-0.1)
    mixed = W.make_faceset("mixed", [W.make_view(s68, 0, 0.0), W.make_view(s68, 1, -40.0),
                                      W.make_view(other, 2, 0.0), W.make_view(other, 3, 40.0)])
    with pytest.raises(DomainError) as info:
        build_identity_asset(mixed, SETTINGS)
    assert info.value.code == "IDENTITY_ASSET_NOT_BUILDABLE" and "more than one person" in info.value.message
    blurry = W.make_view(s68, 0, 0.0)
    blurred = cv2.GaussianBlur(cv2.imdecode(np.frombuffer(blurry.image_bytes, np.uint8), 1), (0, 0), 30)
    ok, png = cv2.imencode(".png", blurred)
    with pytest.raises(DomainError) as info:
        build_identity_asset(W.make_faceset("none", [blurry.model_copy(update={"image_bytes": png.tobytes()})]), SETTINGS)
    assert info.value.code == "IDENTITY_ASSET_NOT_BUILDABLE" and "TOO_BLURRY" in info.value.message


def test_inconsistent_view_is_dropped_and_geometry_stays_accurate(s68):
    views = [W.make_view(s68, i, y) for i, y in enumerate((0, -40, 40, -72, 72))]
    rng = np.random.default_rng(3)
    # the dense-landmark detector failed on this image while the 5-point detector was fine (so the quality analysis
    # accepts the view and only the multi-view consistency check can catch it)
    wrecked = np.asarray(views[1].landmarks_68) + rng.normal(0, 28.0, (68, 2))
    views[1] = views[1].model_copy(update={"landmarks_68": tuple(map(tuple, wrecked))})
    asset = build_identity_asset(W.make_faceset("wrecked", views), SETTINGS)
    refs = {s.index: s for s in asset.manifest.sources}
    assert asset.manifest.fit["dropped_inconsistent_views"] == [1]
    assert refs[1].used is False and "MULTIVIEW_INCONSISTENT" in refs[1].exclusion_reasons
    err = float(np.linalg.norm(W.sim_align(asset.landmarks_3d, s68.true_lm) - s68.true_lm, axis=1).mean() * 1000)
    assert err < 0.4 * float(np.linalg.norm(W.sim_align(asset.template_landmarks, s68.true_lm) - s68.true_lm, axis=1).mean() * 1000)


def test_build_is_deterministic(full_doc, full_asset):
    again = build_identity_asset(full_doc, SETTINGS)
    assert again.manifest.arrays_sha256 == full_asset.manifest.arrays_sha256
    assert again.manifest.build_key == full_asset.manifest.build_key


# ----------------------------------------------------------------------------- GLB inspection
def test_glb_is_inspectable_and_orientation_is_correct(full_asset, tmp_path, s68):
    path = ia.export_asset_glb(full_asset, tmp_path)
    glb = read_glb(path)
    attrs, indices = glb["attributes"], glb["indices"]
    n_v = len(full_asset.vertices)
    assert attrs["POSITION"].shape == (n_v, 3) and attrs["NORMAL"].shape == (n_v, 3) and attrs["TEXCOORD_0"].shape == (n_v, 2)
    assert attrs["_CONFIDENCE"].shape == (n_v,) and attrs["_STATE"].shape == (n_v,)
    assert np.allclose(attrs["_CONFIDENCE"], full_asset.vertex_confidence) and set(np.unique(attrs["_STATE"])) <= {0.0, 1.0, 2.0}
    assert indices.size == len(full_asset.faces) * 3
    extras = glb["gltf"]["extras"]["roop3d_identity_asset"]
    assert extras["build_key"] == full_asset.build_key and extras["claims"]["complete_3d_recovery"] is False
    assert extras["arrays_sha256"] == full_asset.manifest.arrays_sha256 and extras["missing_information"] is not None
    pos = attrs["POSITION"]
    assert pos[36:48, 1].mean() > pos[48:60, 1].mean(), "glTF is y-up: the eyes must be above the mouth"
    tri = indices.reshape(-1, 3)
    geometric = np.cross(pos[tri[:, 1]] - pos[tri[:, 0]], pos[tri[:, 2]] - pos[tri[:, 0]])
    stored = attrs["NORMAL"][tri].mean(axis=1)
    assert (np.einsum("ij,ij->i", geometric, stored) > 0).mean() > 0.97, "triangle winding must agree with the normals"
    assert pos[:, 2].max() > pos[:, 2].min() and (stored[:, 2] > 0).mean() > 0.9
    # the embedded texture is the atlas, and TEXCOORD_0 addresses it with the glTF (top-left) convention
    texture = cv2.imdecode(np.frombuffer(glb["texture_png"], np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(texture, full_asset.texture)
    uv = attrs["TEXCOORD_0"]
    px = np.clip(np.round(uv[:, 0] * (W.ATLAS - 1)).astype(int), 0, W.ATLAS - 1)
    py = np.clip(np.round(uv[:, 1] * (W.ATLAS - 1)).astype(int), 0, W.ATLAS - 1)
    seen = full_asset.vertex_state == OBS
    sampled = texture[py[seen], px[seen]].astype(np.float32)
    truth = s68.texture[py[seen], px[seen]].astype(np.float32)
    assert float(np.abs(sampled - truth).mean()) < 25.0, "texture would appear flipped / misregistered in a glTF viewer"


def test_glb_reader_rejects_corrupt_files(tmp_path):
    bad = tmp_path / "bad.glb"
    bad.write_bytes(b"not a glb at all, but long enough to pass the size check....")
    with pytest.raises(DomainError):
        read_glb(bad)


# ----------------------------------------------------------------------------- persistence / reuse
def test_asset_is_persisted_independently_and_reused(tmp_path, s68, monkeypatch):
    doc = W.full_set(s68, (0, -40, 40))
    before = (doc.faceset_id, doc.source_sha256, [hashlib.sha256(v.image_bytes).hexdigest() for v in doc.views])
    store = IdentityAssetStore(tmp_path / "store")
    first, reused = store.get_or_build(doc, SETTINGS)
    assert reused is False
    directory = store.base / doc.faceset_id / first.build_key[:24]
    assert {p.name for p in directory.iterdir()} == {"arrays.npz", "manifest.json", "texture.png", "head.glb"}
    assert read_glb(store.glb_path(doc.faceset_id, first.build_key))["attributes"]["POSITION"].shape[0] == len(first.vertices)
    source_images = {v.image_bytes for v in doc.views}
    assert not any(p.read_bytes() in source_images for p in directory.iterdir())            # independent of the FaceSet
    assert before == (doc.faceset_id, doc.source_sha256, [hashlib.sha256(v.image_bytes).hexdigest() for v in doc.views])

    monkeypatch.setattr(ia, "build_identity_asset", lambda *a, **k: (_ for _ in ()).throw(AssertionError("rebuilt")))
    second, reused = store.get_or_build(doc, SETTINGS)
    assert reused is True and second.manifest.arrays_sha256 == first.manifest.arrays_sha256
    assert all(np.array_equal(first.array_fields[k], second.array_fields[k]) for k in first.array_fields)
    renamed, reused = store.get_or_build(doc.model_copy(update={"name": "A different display name"}), SETTINGS)
    assert reused is True and renamed.build_key == first.build_key                            # cosmetic FaceSet edits do not rebuild


def test_rebuild_only_when_sources_settings_or_model_hashes_change(tmp_path, s68):
    store = IdentityAssetStore(tmp_path / "store")
    views = [W.make_view(s68, i, y) for i, y in enumerate((0, -40, 40))]
    doc = W.make_faceset("rebuild", views)
    base_key = compute_build_key(doc, SMALL)
    asset, reused = store.get_or_build(doc, SMALL)
    assert not reused and asset.build_key == base_key
    assert store.get_or_build(doc, SMALL)[1] is True

    changed_image = W.make_view(s68, 2, 40.0, light=0.9)                                      # different pixels, same pose
    changed_doc = W.make_faceset("rebuild", [views[0], views[1], changed_image])
    assert compute_build_key(changed_doc, SMALL) != base_key and store.get_or_build(changed_doc, SMALL)[1] is False
    for variant in (SMALL.model_copy(update={"uv_size": 96}), SMALL.model_copy(update={"max_interpolation_fraction": 0.05}),
                    SMALL.model_copy(update={"model_hashes": {"detector": "ab" * 32}}),
                    SMALL.model_copy(update={"model_hashes": {"detector": "cd" * 32}})):
        assert compute_build_key(doc, variant) != base_key, variant
    moved = views[1].model_copy(update={"landmarks": tuple((x + 3.0, y) for x, y in views[1].landmarks)})
    assert compute_build_key(W.make_faceset("rebuild", [views[0], moved, views[2]]), SMALL) != base_key
    assert store.get_or_build(doc, SMALL)[1] is True                                          # the original is still cached


def test_tampered_or_corrupt_storage_is_never_trusted(tmp_path, s68):
    store = IdentityAssetStore(tmp_path / "store")
    doc = W.make_faceset("tamper", [W.make_view(s68, i, y) for i, y in enumerate((0, -40, 40))])
    asset, _ = store.get_or_build(doc, SMALL)
    directory = store.base / doc.faceset_id / asset.build_key[:24]
    arrays = dict(np.load(directory / "arrays.npz"))
    arrays["vertices"] = arrays["vertices"] + 0.001                                           # valid file, altered content
    np.savez_compressed(directory / "arrays.npz", **arrays)
    assert store.load(doc.faceset_id, asset.build_key) is None
    rebuilt, reused = store.get_or_build(doc, SMALL)
    assert reused is False and np.array_equal(rebuilt.vertices, asset.vertices)
    (directory / "arrays.npz").write_bytes(b"garbage")
    assert store.load(doc.faceset_id, asset.build_key) is None
    (directory / "manifest.json").write_text("{not json", encoding="utf-8")
    assert store.load(doc.faceset_id, asset.build_key) is None


def test_store_confines_paths_and_prunes_old_builds(tmp_path, s68):
    store = IdentityAssetStore(tmp_path / "store")
    for bad in ("../escape", "a/b", "", "x" + chr(92) + "y"):
        with pytest.raises(DomainError) as info:
            store.load(bad, "0" * 64)
        assert info.value.code == "INVALID_FACESET_ID"
    doc = W.make_faceset("prune", [W.make_view(s68, i, y) for i, y in enumerate((0, -40, 40))])
    for max_views in (1, 2, 3, 4, 5):
        store.get_or_build(doc, SMALL.model_copy(update={"max_views": max_views}))
    kept = [p for p in (store.base / "prune").iterdir() if p.is_dir()]
    assert len(kept) == IdentityAssetStore.KEEP_PER_FACESET
    assert not any(p.name.startswith(".tmp-") for p in kept)


def test_default_store_lives_inside_the_runtime_tree():
    store = IdentityAssetStore()
    assert store.base.parts[-2:] == (".runtime", "identity_assets")
    assert "roop_3d" in store.base.parts and "roop-ultimate" not in str(store.base).lower()
