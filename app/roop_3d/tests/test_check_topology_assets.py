"""The asset check tells a new user exactly what is missing, never fails an install, and never touches the network."""
from __future__ import annotations

import json

from roop_3d.reconstruction.topology import MorphableTopologyRegistry
from roop_3d.tools import check_topology_assets as tool


class NoMakeHuman(MorphableTopologyRegistry):
    def availability(self, topology_id):
        if topology_id == "makehuman_cc0":
            return {"available": False, "state": "asset_missing", "detail": "not built"}
        return super().availability(topology_id)


def test_missing_flame_gives_actionable_steps(tmp_path):
    report = tool.check(MorphableTopologyRegistry(flame_path=tmp_path / "flame2023_Open.pkl"))
    flame = next(r for r in report["topologies"] if r["id"] == "flame2023_open")
    assert flame["available"] is False and flame["state"] == "asset_missing"
    text = tool.render(report)
    assert "flame2023_Open.pkl" in text and "https://flame.is.tue.mpg.de/" in text and "ROOP3D_FLAME2023_OPEN" in text
    assert "non-commercial" in text and "never loaded" in text
    assert flame["expected_sha256"] and "optional" in text.lower()


def test_wrong_file_is_called_out_as_wrong_not_missing(tmp_path):
    junk = tmp_path / "flame2023_Open.pkl"
    junk.write_bytes(b"not the pinned file")
    report = tool.check(MorphableTopologyRegistry(flame_path=junk))
    flame = next(r for r in report["topologies"] if r["id"] == "flame2023_open")
    assert flame["state"] == "hash_mismatch" and "[WRONG  ] flame2023_open" in tool.render(report)


def test_path_outside_the_checkout_is_reported_not_raised(tmp_path):
    outside = type(tmp_path)(tmp_path.anchor) / "definitely_outside_the_checkout" / "flame2023_Open.pkl"
    report = tool.check(MorphableTopologyRegistry(flame_path=outside))
    flame = next(r for r in report["topologies"] if r["id"] == "flame2023_open")
    assert flame["state"] == "path_rejected" and flame["available"] is False


def test_install_never_fails_on_missing_optional_assets(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tool, "MorphableTopologyRegistry", lambda: MorphableTopologyRegistry(flame_path=tmp_path / "x.pkl"))
    assert tool.main([]) == 0
    assert "MISSING" in capsys.readouterr().out


def test_strict_fails_only_when_the_default_topology_is_missing(monkeypatch, capsys):
    monkeypatch.setattr(tool, "MorphableTopologyRegistry", NoMakeHuman)
    assert tool.main(["--strict"]) == 1
    out = capsys.readouterr().out
    assert "makehuman_build" in out and "default topology is NOT ready" in out
    assert tool.main([]) == 0


def test_json_output_is_machine_readable(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tool, "MorphableTopologyRegistry", lambda: MorphableTopologyRegistry(flame_path=tmp_path / "x.pkl"))
    tool.main(["--json"])
    data = json.loads(capsys.readouterr().out)
    assert {r["id"] for r in data["topologies"]} == {"makehuman_cc0", "flame2023_open"} and "default_ready" in data
