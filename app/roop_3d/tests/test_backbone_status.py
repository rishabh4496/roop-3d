"""The read-only backbone report behind the Inspector's "3D Head Model Backbone" dropdown."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from roop_3d.reconstruction.backbone_status import BACKBONES, DEFAULT_BACKBONE, backbone_status
from roop_3d.server import faceset_3d


def _bundle(root: Path, topology: str = "bfm35709", *, with_assets: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    assets = {"topology": {"file": "topology.npz", "sha256": "0" * 64}, "coarse": {"file": "coarse.onnx", "sha256": "0" * 64}}
    if with_assets:
        for record in assets.values():
            (root / record["file"]).write_bytes(b"x")
    path = root / "bundle.json"
    path.write_text(json.dumps({"schema": "roop3d.native_dense_face.v1", "topology": topology, "assets": assets}), encoding="utf-8")
    return path


def _by_id(report: dict) -> dict:
    return {row["id"]: row for row in report["backbones"]}


def test_nothing_installed_marks_every_family_unavailable_with_a_reason(tmp_path: Path) -> None:
    report = backbone_status(tmp_path / "missing" / "bundle.json", tmp_path / "no_head.npz")
    rows = _by_id(report)
    assert [row["id"] for row in report["backbones"]] == [item["id"] for item in BACKBONES]
    assert report["active"] is None and report["default"] == DEFAULT_BACKBONE
    assert not any(row["available"] for row in rows.values())
    assert all(row["reason"] for row in rows.values())
    assert "No dense-face bundle" in rows["3ddfa_v3_hrn"]["reason"]


@pytest.mark.parametrize("topology, active, other", [("bfm35709", "3ddfa_v3_hrn", "flame_smirk"), ("flame2023open", "flame_smirk", "3ddfa_v3_hrn")])
def test_the_installed_topology_is_the_only_available_dense_family(tmp_path: Path, topology: str, active: str, other: str) -> None:
    report = backbone_status(_bundle(tmp_path / "dense", topology), tmp_path / "no_head.npz")
    rows = _by_id(report)
    assert report["active"] == active and rows[active]["available"] and rows[active]["active"] and not rows[active]["reason"]
    assert not rows[other]["available"] and "graph families are never mixed" in rows[other]["reason"]


def test_a_bundle_with_missing_asset_files_is_not_reported_available(tmp_path: Path) -> None:
    report = backbone_status(_bundle(tmp_path / "dense", with_assets=False), tmp_path / "no_head.npz")
    row = _by_id(report)["3ddfa_v3_hrn"]
    assert not row["available"] and "missing asset files" in row["reason"] and "coarse" in row["reason"]


def test_an_unreadable_manifest_is_reported_not_raised(tmp_path: Path) -> None:
    broken = tmp_path / "dense" / "bundle.json"
    broken.parent.mkdir()
    broken.write_text("{not json", encoding="utf-8")
    assert "unreadable" in _by_id(backbone_status(broken, tmp_path / "no_head.npz"))["3ddfa_v3_hrn"]["reason"]


def test_faceverse_is_never_available_and_makehuman_only_when_built(tmp_path: Path) -> None:
    head = tmp_path / "head.npz"
    assert not _by_id(backbone_status(tmp_path / "x.json", head))["makehuman_legacy"]["available"]
    head.write_bytes(b"x")
    rows = _by_id(backbone_status(tmp_path / "x.json", head))
    assert rows["makehuman_legacy"]["available"] and "never uses it" in rows["makehuman_legacy"]["reason"]
    assert not rows["faceverse_v4"]["available"] and "no FaceVerse" in rows["faceverse_v4"]["reason"]


def test_the_route_serves_the_report() -> None:
    app = FastAPI()
    app.include_router(faceset_3d.router)
    body = TestClient(app).get("/api/v1/reconstruction/backbones").json()
    assert set(body) == {"default", "active", "backbones"} and len(body["backbones"]) == 4
