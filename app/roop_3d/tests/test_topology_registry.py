"""Morphable topology registry: defaults, licence gating, SHA-256 pinning, safe parsing, and the FLAME 2023 Open frame.

The FLAME file is user-supplied and git-ignored, so everything that needs its real bytes is skipped when it is absent; the
governance behaviour (hash gate, allow-listed unpickler, licence refusal) is tested without it.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from roop_3d.audit.isolation import AuditError
from roop_3d.audit.local_models import OPEN_ARTIFACTS, verify_open_artifact, verify_open_artifacts
from roop_3d.core.errors import DomainError
from roop_3d.reconstruction import topology as topo
from roop_3d.reconstruction.makehuman_build import ASSET_NAME, DEFAULT_ASSET_DIR

HAS_MAKEHUMAN = (DEFAULT_ASSET_DIR / ASSET_NAME).exists()
needs_makehuman = pytest.mark.skipif(not HAS_MAKEHUMAN, reason="run `python -m roop_3d.reconstruction.makehuman_build` first")


def _flame_available() -> bool:
    return topo.MorphableTopologyRegistry().availability(topo.FLAME_OPEN_ID)["available"]


needs_flame = pytest.mark.skipif(not _flame_available(), reason="FLAME 2023 Open is not installed (user-supplied, git-ignored)")


# ------------------------------------------------------------------------------------------------ governance (no assets)
def test_default_is_the_cc0_makehuman_topology():
    registry = topo.MorphableTopologyRegistry()
    assert registry.default_id == topo.MAKEHUMAN_ID == "makehuman_cc0"
    spec = registry.spec("makehuman_cc0")
    assert spec.builtin and spec.license_id == "CC0-1.0" and spec.commercial_use == "allow"
    assert registry.ids()[0] == "makehuman_cc0"


def test_flame_open_is_registered_with_pinned_hash_and_attribution():
    spec = topo.MorphableTopologyRegistry().spec("flame2023_open")
    record = OPEN_ARTIFACTS["flame2023_open"]
    assert spec.license_id == record["license_id"] == "CC-BY-4.0" and spec.commercial_use == "allow" and not spec.builtin
    assert len(record["sha256"]) == 64 and record["size_bytes"] > 50_000_000
    assert "Li" in spec.attribution and "SIGGRAPH Asia" in spec.attribution and "CC BY 4.0" in spec.attribution
    assert spec.artifact_id == "flame2023_open"


@pytest.mark.parametrize("license_id", ["LicenseRef-FLAME-Academic", "LicenseRef-MPG-NonCommercial", "UNKNOWN"])
def test_non_commercial_topologies_cannot_be_registered(license_id):
    registry = topo.MorphableTopologyRegistry()
    spec = topo.TopologySpec("flame2020", "FLAME 2020", license_id, "https://example.invalid", "allow", "n/a", "n/a")
    with pytest.raises(DomainError) as error:
        registry.register(spec, lambda *_: None)
    assert error.value.code == "TOPOLOGY_LICENSE_REFUSED"
    denied = topo.TopologySpec("x", "x", "CC-BY-4.0", "https://example.invalid", "deny", "n/a", "n/a")
    with pytest.raises(DomainError):
        registry.register(denied, lambda *_: None)
    assert "flame2020" not in registry.ids()


def test_unpinned_artifact_and_duplicates_are_refused():
    registry = topo.MorphableTopologyRegistry()
    spec = topo.TopologySpec("mystery", "Mystery", "CC-BY-4.0", "https://example.invalid", "allow", "n/a", "n/a", artifact_id="no_such_pin")
    with pytest.raises(DomainError) as error:
        registry.register(spec, lambda *_: None)
    assert error.value.code == "TOPOLOGY_NOT_PINNED"
    with pytest.raises(DomainError) as dup:
        registry.register(registry.spec("makehuman_cc0"), lambda *_: None)
    assert dup.value.code == "TOPOLOGY_DUPLICATE"


def test_unknown_topology_is_a_domain_error():
    with pytest.raises(DomainError) as error:
        topo.MorphableTopologyRegistry().spec("deca")
    assert error.value.code == "TOPOLOGY_UNKNOWN" and error.value.status == 422


def test_wrong_bytes_fail_the_sha256_gate_before_parsing(tmp_path):
    junk = tmp_path / "flame2023_Open.pkl"
    junk.write_bytes(b"not the pinned file")
    registry = topo.MorphableTopologyRegistry(flame_path=junk)
    state = registry.availability("flame2023_open")
    assert state["available"] is False and state["state"] == "hash_mismatch"
    with pytest.raises(DomainError) as error:
        registry.load("flame2023_open")
    assert error.value.code == "TOPOLOGY_ASSET_HASH_MISMATCH"
    with pytest.raises(AuditError) as audit:
        verify_open_artifact("flame2023_open", junk)
    assert audit.value.code == "MODEL_DRIFT"


def test_missing_file_is_reported_not_raised_by_availability(tmp_path):
    registry = topo.MorphableTopologyRegistry(flame_path=tmp_path / "absent.pkl")
    state = registry.availability("flame2023_open")
    assert state["available"] is False and state["state"] == "asset_missing"
    with pytest.raises(DomainError) as error:
        registry.load("flame2023_open")
    assert error.value.code == "TOPOLOGY_ASSET_MISSING" and error.value.status == 404


def test_flame_path_must_stay_inside_the_checkout(tmp_path):
    outside = Path(tmp_path.anchor) / "definitely_outside_the_checkout" / "flame2023_Open.pkl"
    registry = topo.MorphableTopologyRegistry(flame_path=outside)
    with pytest.raises(DomainError) as error:
        registry.availability("flame2023_open")
    assert error.value.code == "TOPOLOGY_PATH_REJECTED"


def test_unpickler_refuses_code_execution(tmp_path):
    marker = tmp_path / "executed.txt"

    class Evil:
        def __reduce__(self):
            return (open, (str(marker), "w"))

    payload = tmp_path / "evil.pkl"
    payload.write_bytes(pickle.dumps({"v_template": Evil()}))
    with pytest.raises(DomainError) as error:
        topo.read_flame_open(payload)
    assert error.value.code == "TOPOLOGY_ASSET_INVALID"
    assert not marker.exists(), "a refused global must never be called"


def test_structurally_wrong_files_are_rejected(tmp_path):
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(pickle.dumps({"v_template": np.zeros((4, 3)), "f": np.array([[0, 1, 9]]), "shapedirs": np.zeros((4, 3, 400))}))
    with pytest.raises(DomainError) as error:
        topo.read_flame_open(bad)
    assert error.value.code == "TOPOLOGY_ASSET_INVALID"


def test_audit_reports_absent_optional_artifact_without_failing(tmp_path):
    result = verify_open_artifacts(tmp_path)
    assert result["ok"] is True and result["artifacts"][0]["state"] == "NOT_PRESENT"


def test_audit_flags_a_present_file_that_is_not_the_pinned_one(tmp_path):
    target = tmp_path / "flame" / "flame2023"
    target.mkdir(parents=True)
    (target / "flame2023_Open.pkl").write_bytes(b"tampered")
    result = verify_open_artifacts(tmp_path)
    assert result["ok"] is False and result["violations"][0]["artifact_id"] == "flame2023_open"


def test_frame_conversion_is_a_mirror_with_flipped_winding():
    v = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    f = np.array([[0, 1, 2]])
    basis = np.ones((3, 3, 2))
    v2, f2, b2 = topo.flame_to_identity_frame(v, f, basis)
    assert np.allclose(v2[:, 1], -v[:, 1]) and np.allclose(v2[:, [0, 2]], v[:, [0, 2]])
    assert f2.tolist() == [[0, 2, 1]] and np.allclose(b2[:, 1, :], -1.0)
    n1 = np.cross(v[1] - v[0], v[2] - v[0])
    n2 = np.cross(v2[f2[0, 1]] - v2[f2[0, 0]], v2[f2[0, 2]] - v2[f2[0, 0]])
    assert np.allclose(n2, n1 * np.array([1.0, -1.0, 1.0])), "the mirrored surface keeps its outward normal (mirrored)"


# ------------------------------------------------------------------------------------------------------- real assets
@needs_makehuman
def test_makehuman_topology_loads_with_basis_and_landmarks():
    mesh = topo.default_registry().load("makehuman_cc0")
    assert mesh.vertices.shape[1] == 3 and mesh.uvs.shape == (len(mesh.vertices), 2)
    assert mesh.shape_basis.shape[:2] == mesh.vertices.shape and mesh.shape_axes >= 40
    assert len(mesh.landmarks["68pt"]) == 68 and len(mesh.landmarks["5pt"]) == 5
    assert mesh.uv_provenance == "makehuman_atlas"
    assert topo.default_registry().prepared("makehuman_cc0") is mesh


@needs_makehuman
def test_describe_lists_licences_and_availability():
    rows = {row["id"]: row for row in topo.default_registry().describe()}
    assert rows["makehuman_cc0"]["default"] is True and rows["makehuman_cc0"]["available"] is True
    assert rows["makehuman_cc0"]["license_id"] == "CC0-1.0"
    assert rows["flame2023_open"]["license_id"] == "CC-BY-4.0" and "attribution" in rows["flame2023_open"]


@needs_flame
def test_flame_open_loads_in_the_identity_asset_frame():
    registry = topo.MorphableTopologyRegistry()
    mesh = registry.load("flame2023_open")
    assert mesh.vertices.shape == (5023, 3) and mesh.faces.shape == (9976, 3) and mesh.shape_basis.shape == (5023, 3, 300)
    assert mesh.uvs is None and mesh.uv_provenance == "none", "FLAME 2023 Open ships no UV layout; none is claimed"
    assert mesh.meta["sha256"] == OPEN_ARTIFACTS["flame2023_open"]["sha256"]
    v = mesh.vertices
    assert -0.03 < v[np.argmax(v[:, 2]), 1] < 0.05, "the nose tip (max z, toward the viewer) sits at face height"
    top, bottom = v[:, 1].argmin(), v[:, 1].argmax()
    assert v[top, 1] < 0 < v[bottom, 1], "+y is down: the crown is at the most negative y, the neck at the most positive"
    tri = v[mesh.faces]
    centre = np.array([0.0, float(v[:, 1].mean()), float(v[:, 2].mean())])
    outward = np.einsum("fc,fc->f", np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), tri.mean(axis=1) - centre)
    assert (outward > 0).mean() > 0.7, "triangles are wound outward after the mirror"
    assert np.abs(mesh.shape_basis).max() < 0.2


@needs_flame
def test_prepared_flame_has_a_derived_labelled_uv_layout():
    registry = topo.MorphableTopologyRegistry()
    raw, prepared = registry.load("flame2023_open"), registry.prepared("flame2023_open")
    assert prepared.uvs is not None and len(prepared.uvs) == prepared.vertex_count >= raw.vertex_count
    assert "derived" in prepared.uv_provenance and "makehuman_cc0" in prepared.uv_provenance
    assert prepared.shape_basis.shape[:2] == prepared.vertices.shape
    assert np.array_equal(prepared.vertices, raw.vertices[prepared.source_index])
