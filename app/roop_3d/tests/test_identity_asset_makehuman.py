"""Identity asset built on the open MakeHuman (CC0) head template: template contract, build, honesty, persistence, API.

Uses the ground-truth world from ``identity_world`` (real rendered images, real landmarks, real reconstruction code); only
the head template under test differs from the canonical tests. Skipped when the MakeHuman asset has not been built.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Settings
from roop_3d.reconstruction import identity_asset as ia
from roop_3d.reconstruction import makehuman_template as mt
from roop_3d.reconstruction.geometry_types import ConfidenceState, Mesh3D
from roop_3d.reconstruction.identity_asset import (
    IdentityAssetSettings,
    IdentityAssetStore,
    build_identity_asset,
    compute_build_key,
)
from roop_3d.reconstruction.makehuman_build import ASSET_NAME, DEFAULT_ASSET_DIR
from roop_3d.reconstruction.mesh_io import read_glb
from roop_3d.reconstruction.multiview_fit import LAYOUT_5, LAYOUT_68, deform_template, landmark_normals, template_landmarks
from roop_3d.server import identity_assets
from roop_3d.server.app import create_app
from roop_3d.tests import identity_world as W

pytestmark = pytest.mark.skipif(not (DEFAULT_ASSET_DIR / ASSET_NAME).exists(),
                                reason="run `python -m roop_3d.reconstruction.makehuman_build` first")

MH = IdentityAssetSettings(template="makehuman", uv_size=W.ATLAS)
YAWS = (0, -40, 40, -72, 72)
UNSEEN = int(ConfidenceState.UNSEEN)


@pytest.fixture(scope="module", autouse=True)
def _head_sized_camera():
    """The MakeHuman head+neck is taller than the canonical facial mesh: use a smaller render scale so it stays in frame."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(W, "SCALE", 1700.0)
        yield


def make_mh_subject(seed: int = 91, *, width: float = 0.12, nose_depth: float = 0.016, jaw_drop: float = 0.010) -> W.Subject:
    template = mt.makehuman_template()
    t_lm = template_landmarks(LAYOUT_5, template)
    d = np.zeros_like(t_lm)
    d[:, 0] = t_lm[:, 0] * width
    d[:, 1] = np.where(t_lm[:, 1] > 0.02, jaw_drop, 0.0)
    d[2, 2] = nose_depth
    true_lm = t_lm + d
    verts, _, _ = deform_template(template, t_lm, true_lm, np.ones(len(t_lm)))
    mesh = Mesh3D(verts, template.faces, uvs=template.uvs)
    rng = np.random.default_rng(seed)
    uv_raster = np.column_stack([template.uvs[:, 0] * (W.ATLAS - 1), (1.0 - template.uvs[:, 1]) * (W.ATLAS - 1)])
    return W.Subject("mh", LAYOUT_5, seed, template, t_lm, true_lm, mesh, W.ground_truth_texture(W.ATLAS, seed),
                     t_lm.mean(axis=0), W._unit(rng.normal(size=512)), uv_raster)


@pytest.fixture(scope="module")
def subject():
    return make_mh_subject()


@pytest.fixture(scope="module")
def doc(subject):
    return W.full_set(subject, YAWS)


@pytest.fixture(scope="module")
def asset(doc):
    return build_identity_asset(doc, MH)


# ----------------------------------------------------------------------------- template contract
def test_template_contract():
    t = mt.makehuman_template()
    assert t.uvs.shape == (len(t.vertices), 2), "per-vertex UVs: vertices are split at UV seams"
    assert t.uvs.min() >= 0.0 and t.uvs.max() <= 1.0 and 0.9 < t.uvs.max()
    tri = t.uvs[t.faces]
    area = 0.5 * np.abs(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]))
    assert area.min() > 0 and 0.3 < area.sum() <= 1.0, "islands are repacked to use the atlas, without overlap (area <= 1)"
    lm, normals = template_landmarks(LAYOUT_5, t), landmark_normals(LAYOUT_5, t)
    assert lm[0, 0] < 0 < lm[1, 0] and lm[3, 0] < 0 < lm[4, 0]            # image-left first, +x image-right
    assert lm[:2, 1].mean() < lm[3:, 1].mean()                              # +y down: eyes above the mouth
    assert lm[2, 2] > lm[:2, 2].max()                                       # +z toward the viewer: the nose is nearest
    assert np.all(normals[:, 2] > 0.5), "outward normals at the landmarks face the viewer"
    front = t.vertices[t.faces].mean(axis=1)[:, 2] > t.vertices[:, 2].max() - 0.03
    assert (t.compute_face_normals()[front, 2] > 0).mean() > 0.75
    assert mt.template_hash() == mt.template_hash()
    lm68 = template_landmarks(LAYOUT_68, t)           # derived correspondence (makehuman_landmarks)
    assert lm68.shape == (68, 3)
    assert lm68[36:42, 1].mean() < lm68[48:60, 1].mean()                     # eyes above the mouth (+y down)
    assert lm68[36:42, 0].mean() < lm68[42:48, 0].mean()                     # image-left eye has the smaller x
    assert len(set(t.landmark_vertices["68pt"].tolist())) > 55                # mostly distinct vertices


def test_island_repack_is_deterministic_and_disjoint():
    a, b = mt.makehuman_template(), mt.makehuman_template()
    assert np.array_equal(a.uvs, b.uvs) and np.array_equal(a.faces, b.faces)
    ids = mt._uv_islands(a.faces, len(a.uvs))
    boxes = [(a.uvs[ids == r].min(0), a.uvs[ids == r].max(0)) for r in np.unique(ids)]
    for i, (lo1, hi1) in enumerate(boxes):
        for lo2, hi2 in boxes[i + 1:]:
            assert np.any(hi1 <= lo2) or np.any(hi2 <= lo1), "island bounding boxes must not overlap"


def test_missing_template_asset_is_actionable(tmp_path, monkeypatch):
    with pytest.raises(DomainError) as err:
        mt.makehuman_template(tmp_path / "absent.npz")
    assert err.value.code == "MAKEHUMAN_ASSET_MISSING" and "makehuman_build" in str(err.value)


# ----------------------------------------------------------------------------- build
def test_build_follows_the_supplied_identity_and_keeps_template_topology(asset, subject):
    other = make_mh_subject(92, width=-0.10, nose_depth=-0.008, jaw_drop=-0.008)
    theirs = build_identity_asset(W.full_set(other, YAWS), MH)
    err = lambda est, truth: float(np.linalg.norm(W.sim_align(est, truth) - truth, axis=1).mean() * 1000)
    assert asset.manifest.settings["template"] == "makehuman" and asset.manifest.landmark_layout == "5pt"
    assert len(asset.vertices) == len(mt.makehuman_template().vertices) and asset.uvs.shape == (len(asset.vertices), 2)
    assert err(asset.landmarks_3d, subject.true_lm) < 0.5 * err(asset.template_landmarks, subject.true_lm)
    assert err(asset.landmarks_3d, subject.true_lm) < 0.6 * err(asset.landmarks_3d, other.true_lm)          # not a generic head
    assert err(theirs.landmarks_3d, other.true_lm) < 0.6 * err(theirs.landmarks_3d, subject.true_lm)
    assert np.abs(asset.vertices - theirs.vertices).max() > 0.003


def test_manifest_is_honest_about_what_was_measured(asset):
    m = asset.manifest
    assert m.model_version["template"].startswith("makehuman-cc0-head") and m.model_hashes["template_geometry_sha256"] == mt.template_hash()
    assert m.claims["complete_3d_recovery"] is False and m.claims["back_of_head_recovered"] is False and m.claims["full_head_geometry"] is False
    assert any("MakeHuman" in lim and "never measured" in lim for lim in m.limitations)
    assert not any("FACIAL REGION only" in lim for lim in m.limitations)
    v = asset.vertices
    back = v[:, 2] < v[:, 2].min() + 0.04                                     # the rearmost 4 cm of the skull: no view reaches it
    assert back.sum() > 100 and np.all(asset.vertex_state[back] == UNSEEN)
    assert not np.any(asset.vertex_state == int(ConfidenceState.OBSERVED)) or np.all(asset.vertex_confidence[asset.vertex_state == int(ConfidenceState.OBSERVED)] >= 0.6)
    texture = asset.texture
    assert texture.shape[0] == W.ATLAS and (asset.texel_state == UNSEEN).mean() > 0.2, "the unseen back/neck stays unseen, not invented"


def test_dense_landmarks_use_the_derived_68_point_correspondence():
    s68 = W.make_subject("mh68", W.LAYOUT_68, 93)
    built = build_identity_asset(W.full_set(s68, (0, -40, 40)), MH)
    assert built.manifest.landmark_layout == "68pt"
    assert any("derived from the anthropometric model" in lim for lim in built.manifest.limitations)


def test_build_is_deterministic(doc, asset):
    again = build_identity_asset(doc, MH)
    assert again.manifest.arrays_sha256 == asset.manifest.arrays_sha256 and again.build_key == asset.build_key


# ----------------------------------------------------------------------------- keys
def test_build_key_separates_templates_and_leaves_canonical_keys_untouched(doc):
    canonical, makehuman = IdentityAssetSettings(uv_size=W.ATLAS), MH
    assert compute_build_key(doc, canonical) != compute_build_key(doc, makehuman)
    assert "template" not in ia._key_settings(canonical) and ia._key_settings(makehuman)["template"] == "makehuman"
    assert compute_build_key(doc, canonical) == compute_build_key(doc, IdentityAssetSettings(uv_size=W.ATLAS, template="canonical"))
    assert ia._settings_template_hash(canonical) == ia._template_hash(canonical.subdivisions)       # canonical hash is the old one


# ----------------------------------------------------------------------------- persistence / export
def test_store_roundtrip_reuse_and_glb(tmp_path, doc, subject):
    store = IdentityAssetStore(tmp_path / "store")
    first, reused = store.get_or_build(doc, MH)
    again, reused_again = store.get_or_build(doc, MH)
    assert reused is False and reused_again is True and again.manifest.arrays_sha256 == first.manifest.arrays_sha256
    glb = read_glb(store.glb_path(doc.faceset_id, first.build_key))
    pos = glb["attributes"]["POSITION"]
    assert pos.shape == (len(first.vertices), 3) and glb["indices"].size == len(first.faces) * 3
    lm = mt.makehuman_template().landmark_vertices["5pt"]
    assert pos[lm[:2], 1].mean() > pos[lm[3:], 1].mean(), "glTF is y-up: the eyes must be above the mouth"
    tri = glb["indices"].reshape(-1, 3)
    geometric = np.cross(pos[tri[:, 1]] - pos[tri[:, 0]], pos[tri[:, 2]] - pos[tri[:, 0]])
    stored = glb["attributes"]["NORMAL"][tri].mean(axis=1)
    assert (np.einsum("ij,ij->i", geometric, stored) > 0).mean() > 0.97, "winding agrees with the normals"


# ----------------------------------------------------------------------------- downstream consumers
def test_expression_rig_and_pose_blend_refuse_a_non_canonical_asset(asset):
    from roop_3d.reconstruction.pose_blend import PoseBlendProjector
    from roop_3d.reconstruction.reenactment import IdentityRig
    with pytest.raises(DomainError) as err:
        IdentityRig(asset)
    assert err.value.code == "IDENTITY_TEMPLATE_UNSUPPORTED"
    with pytest.raises(DomainError) as err:
        PoseBlendProjector.from_identity_asset(asset)
    assert err.value.code == "IDENTITY_TEMPLATE_UNSUPPORTED"


# ----------------------------------------------------------------------------- API
@pytest.fixture()
def api(tmp_path, monkeypatch, doc):
    class Reader:
        def __init__(self, *args, **kwargs):
            pass

        def read(self, archive, faceset_id):
            return doc

    root = tmp_path / "roop_3d/.runtime/facesets"
    root.mkdir(parents=True)
    (root / f"{doc.faceset_id}.json").write_text(json.dumps({"faceset_id": doc.faceset_id, "name": doc.name,
                                                             "item_count": len(doc.views), "schema_version": 1}))
    (root / f"{doc.faceset_id}.fsz").write_bytes(b"placeholder")
    store_root = tmp_path / "store"
    monkeypatch.setattr(identity_assets, "FaceSetReader", Reader)
    monkeypatch.setattr(identity_assets, "_store", lambda: IdentityAssetStore(root=store_root))
    return TestClient(create_app(Settings(), repo_root=tmp_path), base_url="http://127.0.0.1:8030")


def test_api_template_parameter_builds_a_separate_asset(api, doc):
    base = f"/api/v1/facesets/{doc.faceset_id}/identity-asset"
    with api as client:
        assert client.get(base, params={"template": "makehuman"}).json()["state"] == "not_built"
        built = client.post(base, params={"template": "makehuman"})
        assert built.status_code == 200, built.text
        body = built.json()
        assert body["state"] == "built" and body["template"] == "makehuman" and body["asset"]["vertex_count"] > 4000
        assert client.get(base).json()["state"] == "not_built", "the default (canonical) asset is a different, unbuilt asset"
        mesh = client.get(base + "/mesh", params={"template": "makehuman"})
        assert mesh.status_code == 200 and len(mesh.json()["vertices"]) == body["asset"]["vertex_count"]
        assert client.get(base + "/mesh").status_code == 404
        assert client.get(base, params={"template": "bogus"}).status_code == 422
