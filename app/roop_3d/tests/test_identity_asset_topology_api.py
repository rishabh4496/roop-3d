"""``/identity-asset/mesh`` topology selection and semantic sliders, plus ``/morphable-topologies``.

The cached identity asset is replaced by a stand-in that carries the *real* MakeHuman template geometry and UV atlas (asset
building is covered by ``test_identity_asset_makehuman``), so every assertion here is about the new route logic: it re-measures the
returned glTF-frame mesh to prove the sliders did what the response claims.
"""
from __future__ import annotations

import base64
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from roop_3d.core.settings import Settings
from roop_3d.reconstruction import identity_asset as ia
from roop_3d.reconstruction import topology as topo
from roop_3d.reconstruction.geometry_types import Mesh3D
from roop_3d.reconstruction.makehuman_build import ASSET_NAME, DEFAULT_ASSET_DIR
from roop_3d.reconstruction.semantic_shape import SLIDER_KEYS, SLIDERS, SemanticShapeController
from roop_3d.server import identity_assets
from roop_3d.server.app import create_app

pytestmark = pytest.mark.skipif(not (DEFAULT_ASSET_DIR / ASSET_NAME).exists(), reason="build the MakeHuman asset first")
UNIT = {s.key: s.unit_mm for s in SLIDERS}
FAKE_ID = "stand_in"
TEX = 256


def _flame_installed() -> bool:
    return topo.MorphableTopologyRegistry().availability("flame2023_open")["available"]


needs_flame = pytest.mark.skipif(not _flame_installed(), reason="FLAME 2023 Open is not installed")


@pytest.fixture(scope="module")
def registry():
    return topo.MorphableTopologyRegistry()


@pytest.fixture()
def api(tmp_path, monkeypatch, registry):
    base = registry.load("makehuman_cc0")
    yy, xx = np.mgrid[0:TEX, 0:TEX]
    texture = np.stack([xx * 255 // TEX, yy * 255 // TEX, np.full_like(xx, 140)], -1).astype(np.uint8)
    state = np.full((TEX, TEX), 2, np.uint8)
    state[:, : TEX // 8] = 0                                                  # a strip nobody saw
    mesh = Mesh3D(base.vertices, base.faces, uvs=base.uvs)
    asset = SimpleNamespace(vertices=base.vertices.astype(np.float32), faces=base.faces, texture=texture, texel_state=state,
                            manifest=SimpleNamespace(build_key="k" * 64, reconstruction_level="template_prior", landmark_layout="5pt",
                                                     shape={}, expression={}, canonical_head_pose={}, statistics={}, settings={}),
                            mesh=lambda: mesh)
    monkeypatch.setattr(identity_assets, "load_document", lambda request, faceset_id: SimpleNamespace(faceset_id=faceset_id))
    monkeypatch.setattr(identity_assets, "_store", lambda: SimpleNamespace(load=lambda faceset_id, key: asset))
    monkeypatch.setattr(ia, "compute_build_key", lambda document, settings: "k" * 64)
    monkeypatch.setattr(topo, "_DEFAULT", registry)                            # the route's process-wide registry
    identity_assets._CONTROLLERS.clear()
    client = TestClient(create_app(Settings(), repo_root=tmp_path), base_url="http://127.0.0.1:8030")
    client.asset = asset
    return client


URL = f"/api/v1/facesets/{FAKE_ID}/identity-asset/mesh"


def identity_vertices(body: dict) -> np.ndarray:
    """Response vertices (glTF frame, y up) back in the identity-asset frame (y down) the controllers measure in."""
    return np.asarray(body["vertices"], np.float64) * np.array([1.0, -1.0, 1.0])


def error_code(response) -> str:
    return response.json()["error"]["code"]


# ----------------------------------------------------------------------------------------------------------- defaults
def test_makehuman_default_topology_is_unchanged_and_labelled(api):
    with api as client:
        response = client.get(URL, params={"template": "makehuman"})
        body = response.json()
    assert response.status_code == 200
    assert len(body["vertices"]) == len(client.asset.vertices) and body["semantic"] is None
    assert body["topology"]["id"] == "makehuman_cc0" and body["topology"]["default"] is True
    assert body["topology"]["license_id"] == "CC0-1.0" and body["topology"]["bridge"] is None
    assert np.allclose(identity_vertices(body), client.asset.vertices, atol=2e-6)


def test_canonical_response_is_exactly_what_it_was(api):
    with api as client:
        body = client.get(URL).json()
    assert set(body) == {"method", "coordinate_system", "vertices", "faces", "uvs", "texture", "provenance", "parametric"}


def test_canonical_template_cannot_take_a_topology_or_sliders(api):
    with api as client:
        assert error_code(client.get(URL, params={"topology": "makehuman_cc0"})) == "TOPOLOGY_REQUIRES_MAKEHUMAN_TEMPLATE"
        assert error_code(client.get(URL, params={"jaw_width": 1})) == "TOPOLOGY_REQUIRES_MAKEHUMAN_TEMPLATE"
        assert client.get(URL, params={"jaw_width": 1}).status_code == 422
        assert client.get(URL, params={"jaw_width": 0}).status_code == 200, "an all-zero slider set is not a request"


def test_unknown_topology_and_bad_sliders_are_422(api):
    with api as client:
        r = client.get(URL, params={"template": "makehuman", "topology": "deca"})
        assert r.status_code == 422 and error_code(r) == "TOPOLOGY_UNKNOWN"
        r = client.get(URL, params={"template": "makehuman", "jaw_width": 99})
        assert r.status_code == 422 and error_code(r) == "SEMANTIC_SLIDER_RANGE"
        r = client.get(URL, params={"template": "makehuman", "jaw_width": "wide"})
        assert r.status_code == 422


def test_asset_from_another_template_version_is_a_conflict(api):
    api.asset.vertices = api.asset.vertices[:-5]
    with api as client:
        r = client.get(URL, params={"template": "makehuman"})
    assert r.status_code == 409 and error_code(r) == "TOPOLOGY_ASSET_MISMATCH"


# ------------------------------------------------------------------------------------------------------------ sliders
@pytest.mark.parametrize("key", SLIDER_KEYS)
def test_slider_changes_the_returned_mesh_by_the_documented_amount(api, registry, key):
    ctl = SemanticShapeController.for_topology(registry, "makehuman_cc0")
    with api as client:
        base = identity_vertices(client.get(URL, params={"template": "makehuman"}).json())
        body = client.get(URL, params={"template": "makehuman", key: 1.0}).json()
    moved = identity_vertices(body)
    before, after = ctl.measure(base), ctl.measure(moved)
    assert after[key] - before[key] == pytest.approx(UNIT[key], abs=0.05)
    assert max(abs(after[k] - before[k]) for k in ctl.keys if k != key) < 0.05
    sem = body["semantic"]
    assert sem["achieved_mm"][key] == pytest.approx(UNIT[key], abs=0.01) and sem["clamped"] is False
    assert sem["requested"][key] == 1.0 and "semantic sliders applied" in body["provenance"]


def test_several_sliders_combine(api, registry):
    ctl = SemanticShapeController.for_topology(registry, "makehuman_cc0")
    values = {"jaw_width": -1.0, "chin_projection": 2.0, "brow_arch": 1.0}
    with api as client:
        base = identity_vertices(client.get(URL, params={"template": "makehuman"}).json())
        moved = identity_vertices(client.get(URL, params={"template": "makehuman", **values}).json())
    before, after = ctl.measure(base), ctl.measure(moved)
    for key, v in values.items():
        assert after[key] - before[key] == pytest.approx(v * UNIT[key], abs=0.1)
    assert abs(after["head_width"] - before["head_width"]) < 0.1


def test_extreme_sliders_report_clamping_instead_of_exceeding_the_basis(api):
    params = {"template": "makehuman", **{k: 3.0 for k in SLIDER_KEYS}}
    with api as client:
        body = client.get(URL, params=params).json()
    sem = body["semantic"]
    assert sem["max_abs_delta_beta"] <= 1.0 + 1e-6
    assert sem["clamped"] is False or sem["limit_scale"] < 1.0


def test_sliders_do_not_touch_the_atlas(api):
    with api as client:
        plain = client.get(URL, params={"template": "makehuman"}).json()["texture"]
        slid = client.get(URL, params={"template": "makehuman", "jaw_width": 1.0}).json()["texture"]
    assert plain == slid


# --------------------------------------------------------------------------------------------------------- FLAME
def decode_png(data_url: str) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(base64.b64decode(data_url.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)


@needs_flame
def test_flame_topology_returns_a_consistent_attributed_mesh(api, registry):
    bridge = registry.bridge("makehuman_cc0", "flame2023_open")
    with api as client:
        r = client.get(URL, params={"template": "makehuman", "topology": "flame2023_open"})
        assert r.status_code == 200, r.text
        body = r.json()
    n = bridge.target.vertex_count
    assert len(body["vertices"]) == n and len(body["uvs"]) == n and len(body["faces"]) == len(bridge.target.faces)
    assert max(max(f) for f in body["faces"]) < n and min(min(f) for f in body["faces"]) >= 0
    t = body["topology"]
    assert t["id"] == "flame2023_open" and t["default"] is False and t["license_id"] == "CC-BY-4.0"
    assert "SIGGRAPH Asia" in t["attribution"] and "derived" in t["uv_provenance"] and t["vertex_count"] == n
    assert t["bridge"]["to_target"]["matched_fraction"] > 0.9 and t["bridge"]["uv"]["uv_overlap_fraction"] < 0.03
    assert "flame2023_open" in body["provenance"] and "CC-BY-4.0" in body["provenance"]
    uvs = np.asarray(body["uvs"])
    assert uvs.min() >= 0 and uvs.max() <= 1.0 + 1e-6
    image = decode_png(body["texture"])
    assert image.shape == (TEX, TEX, 3)


@needs_flame
def test_flame_mesh_is_the_asset_surface_re_expressed(api, registry):
    bridge = registry.bridge("makehuman_cc0", "flame2023_open")
    with api as client:
        body = client.get(URL, params={"template": "makehuman", "topology": "flame2023_open"}).json()
    expected = bridge.convert_positions(client.asset.vertices.astype(np.float64), "target")
    assert np.allclose(identity_vertices(body), expected, atol=2e-6)


@needs_flame
def test_flame_unseen_texels_stay_grey(api):
    with api as client:
        body = client.get(URL, params={"template": "makehuman", "topology": "flame2023_open"}).json()
    image = decode_png(body["texture"])
    grey = np.all(image == 96, axis=-1)
    assert grey.any(), "texels with no counterpart or that the source never saw are shown grey, not invented"
    assert not grey.all()


@needs_flame
@pytest.mark.parametrize("key", SLIDER_KEYS)
def test_flame_sliders_use_the_flame_basis(api, registry, key):
    ctl = SemanticShapeController.for_topology(registry, "flame2023_open")
    params = {"template": "makehuman", "topology": "flame2023_open"}
    with api as client:
        base = identity_vertices(client.get(URL, params=params).json())
        body = client.get(URL, params={**params, key: 1.0}).json()
    before, after = ctl.measure(base), ctl.measure(identity_vertices(body))
    sem = body["semantic"]
    assert after[key] - before[key] == pytest.approx(sem["achieved_mm"][key], abs=0.06)
    assert sem["achieved_mm"][key] == pytest.approx(UNIT[key], abs=0.05) or sem["clamped"]
    assert max(abs(after[k] - before[k]) for k in ctl.keys if k != key) < 0.06


def test_missing_flame_is_a_clear_404_and_does_not_affect_makehuman(api, registry, tmp_path, monkeypatch):
    absent = topo.MorphableTopologyRegistry(flame_path=tmp_path / "flame2023_Open.pkl")
    monkeypatch.setattr(topo, "_DEFAULT", absent)
    with api as client:
        r = client.get(URL, params={"template": "makehuman", "topology": "flame2023_open"})
        assert r.status_code == 404 and error_code(r) == "TOPOLOGY_ASSET_MISSING"
        assert client.get(URL, params={"template": "makehuman", "topology": "makehuman_cc0"}).status_code == 200


def test_tampered_flame_file_is_refused(api, tmp_path, monkeypatch):
    bad = tmp_path / "flame2023_Open.pkl"
    bad.write_bytes(b"x" * 1024)
    monkeypatch.setattr(topo, "_DEFAULT", topo.MorphableTopologyRegistry(flame_path=bad))
    with api as client:
        r = client.get(URL, params={"template": "makehuman", "topology": "flame2023_open"})
    assert r.status_code == 422 and error_code(r) == "TOPOLOGY_ASSET_HASH_MISMATCH"


# ----------------------------------------------------------------------------------------------- topology catalogue
def test_morphable_topologies_endpoint(api):
    with api as client:
        body = client.get("/api/v1/morphable-topologies").json()
    rows = {t["id"]: t for t in body["topologies"]}
    assert body["default"] == "makehuman_cc0" and rows["makehuman_cc0"]["default"] is True
    assert rows["makehuman_cc0"]["license_id"] == "CC0-1.0" and rows["makehuman_cc0"]["available"] is True
    assert rows["flame2023_open"]["license_id"] == "CC-BY-4.0" and isinstance(rows["flame2023_open"]["available"], bool)
    assert [s["key"] for s in body["semantic_sliders"]] == list(SLIDER_KEYS)


# -------------------------------------------------------------------------------------------------- albedo prior flag
def _skin_texture() -> np.ndarray:
    from roop_3d.reconstruction.albedo_prior import lab_to_bgr
    yy, xx = np.mgrid[0:TEX, 0:TEX]
    lab = np.stack([72 - 18 * yy / TEX, 16 + 3 * xx / TEX, 20 + 0 * xx], -1).astype(np.float32)
    return lab_to_bgr(lab)


def _footprint(mesh_uvs, faces):
    from roop_3d.reconstruction.mesh_bridge import rasterize_uv
    return rasterize_uv(np.asarray(mesh_uvs, np.float64), faces, TEX)[0] >= 0


def test_albedo_prior_is_opt_in_and_off_by_default(api):
    api.asset.texture = _skin_texture()
    with api as client:
        body = client.get(URL, params={"template": "makehuman"}).json()
    assert "albedo_prior" not in body and "unseen texels grey" in body["provenance"]


def test_albedo_prior_fills_unseen_texels_with_skin_and_reports_its_provenance(api):
    api.asset.texture = _skin_texture()
    base = api.asset
    with api as client:
        plain = decode_png(client.get(URL, params={"template": "makehuman"}).json()["texture"])
        r = client.get(URL, params={"template": "makehuman", "albedo_prior": "true"})
        assert r.status_code == 200, r.text
        body = r.json()
    image = decode_png(body["texture"])
    footprint = _footprint(np.asarray(body["uvs"]), np.asarray(identity_assets_faces(base)))
    unseen = footprint & (base.texel_state == 0)
    assert unseen.any()
    assert (plain[unseen] == 96).all(axis=-1).all(), "without the flag the same texels are grey"
    assert not np.all(image[unseen] == 96, axis=-1).any(), "with the flag no unseen footprint texel stays grey"
    from roop_3d.reconstruction.albedo_prior import bgr_to_lab, delta_e
    skin_lab = bgr_to_lab(_skin_texture()[footprint & (base.texel_state == 2)]).mean(axis=0)
    assert delta_e(bgr_to_lab(image[unseen]), skin_lab).mean() < 12.0, "the fill is skin-coloured"
    block = body["albedo_prior"]
    assert block["third_party_data"] is False and block["colour_space"] == "CIELAB" and "not observations" in block["limits"]
    assert block["observed_ratio"] + block["symmetry_ratio"] + block["harmonic_ratio"] == pytest.approx(1.0, abs=0.01)
    assert block["skin_tone"]["in_gamut"] is True and "data-free albedo prior" in body["provenance"]


def identity_assets_faces(asset):
    return asset.faces


def test_albedo_prior_leaves_texels_with_data_nearby_alone(api):
    import cv2
    api.asset.texture = _skin_texture()
    with api as client:
        plain = decode_png(client.get(URL, params={"template": "makehuman"}).json()["texture"])
        body = client.get(URL, params={"template": "makehuman", "albedo_prior": "true"}).json()
    image = decode_png(body["texture"])
    footprint = _footprint(body["uvs"], api.asset.faces)
    unseen = footprint & (api.asset.texel_state == 0)
    near = cv2.dilate(unseen.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
    safe = footprint & (api.asset.texel_state == 2) & ~near
    assert safe.any() and np.array_equal(image[safe], _skin_texture()[safe])
    assert np.array_equal(plain[safe], image[safe])


def test_albedo_prior_refuses_when_there_is_no_skin_to_learn_from(api):
    api.asset.texture = np.full((TEX, TEX, 3), (210, 40, 40), np.uint8)
    with api as client:
        r = client.get(URL, params={"template": "makehuman", "albedo_prior": "true"})
        ok = client.get(URL, params={"template": "makehuman"})
    assert r.status_code == 422 and error_code(r) == "ALBEDO_PRIOR_NO_SKIN"
    assert ok.status_code == 200, "the default route never needs skin"


@needs_flame
def test_albedo_prior_works_on_the_flame_topology_too(api):
    api.asset.texture = _skin_texture()
    with api as client:
        r = client.get(URL, params={"template": "makehuman", "topology": "flame2023_open", "albedo_prior": "true", "jaw_width": 1.0})
        assert r.status_code == 200, r.text
        body = r.json()
    assert body["albedo_prior"]["third_party_data"] is False and body["topology"]["id"] == "flame2023_open"
    assert body["semantic"]["achieved_mm"]["jaw_width"] == pytest.approx(UNIT["jaw_width"], abs=0.05)
