"""Which 3D head-model backbones this install can really use, for the Inspector's "3D Head Model Backbone" dropdown.

A cheap, read-only report: it never loads a graph or downloads anything. The dense-face path loads exactly one bundle from
``roop_3d/.runtime/models/dense_face/bundle.json`` (docs/dense-face-backbone.md); its declared ``topology`` is the backbone family that is
installed. Families the code has no loader for are reported unavailable with the reason, never as selectable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from roop_3d.audit.project_paths import application_root

BUNDLE_RELATIVE = "roop_3d/.runtime/models/dense_face/bundle.json"

#: Stable ids the UI stores; labels are the user-facing names.
BACKBONES: tuple[dict[str, str], ...] = (
    {"id": "3ddfa_v3_hrn", "label": "3DDFA-V3 Dense 35k + HRN Detail (Photorealistic Default)", "topology": "bfm35709"},
    {"id": "flame_smirk", "label": "FLAME 2023 Open + SMIRK Micro-Expressions", "topology": "flame2023open"},
    {"id": "faceverse_v4", "label": "FaceVerse V4 (52 ARKit Blendshapes)", "topology": ""},
    {"id": "makehuman_legacy", "label": "Legacy MakeHuman CC0 (Debug Only)", "topology": ""},
)
DEFAULT_BACKBONE = "3ddfa_v3_hrn"


def _installed_topology(bundle: Path) -> tuple[str | None, str]:
    """(topology of the installed bundle or None, plain-language reason when there is none). Checks the manifest and that every
    declared asset file exists; digests are verified by the loader when a render actually uses the bundle."""
    if not bundle.is_file():
        return None, ("No dense-face bundle is installed. Prepare one with the tools in docs/native-dense-face-setup.md "
                      f"({BUNDLE_RELATIVE}).")
    try:
        spec = json.loads(bundle.read_text(encoding="utf-8"))
        topology = spec["topology"]
        missing = [role for role, record in spec["assets"].items() if not (bundle.parent / record["file"]).is_file()]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None, "The installed dense-face bundle.json is unreadable or incomplete."
    if missing:
        return None, f"The installed dense-face bundle is missing asset files: {', '.join(sorted(missing))}."
    return str(topology), ""


def backbone_status(bundle_path: Path | str | None = None, makehuman_path: Path | str | None = None) -> dict[str, Any]:
    bundle = Path(bundle_path) if bundle_path else application_root() / BUNDLE_RELATIVE
    topology, bundle_reason = _installed_topology(bundle)
    if makehuman_path is None:
        from roop_3d.reconstruction.makehuman_build import ASSET_NAME, DEFAULT_ASSET_DIR
        makehuman = Path(DEFAULT_ASSET_DIR) / ASSET_NAME
    else:
        makehuman = Path(makehuman_path)

    rows: list[dict[str, Any]] = []
    for item in BACKBONES:
        row: dict[str, Any] = {"id": item["id"], "label": item["label"], "available": False, "active": False, "reason": ""}
        if item["id"] in ("3ddfa_v3_hrn", "flame_smirk"):
            if topology is None:
                row["reason"] = bundle_reason
            elif topology == item["topology"]:
                row["available"] = row["active"] = True
            else:
                row["reason"] = (f"The installed bundle is the {topology} family; this backbone needs its own bundle "
                                 "(only one family is loaded per session, graph families are never mixed).")
        elif item["id"] == "faceverse_v4":
            row["reason"] = "Not supported: this build has no FaceVerse topology, exporter or bundle schema."
        else:
            if makehuman.is_file():
                row["available"] = True
                row["reason"] = "Inspection and export only: the live swap never uses it."
            else:
                row["reason"] = "The MakeHuman head is not built. Build it once with python -m roop_3d.reconstruction.makehuman_build."
        rows.append(row)
    active = next((row["id"] for row in rows if row["active"]), None)
    return {"default": DEFAULT_BACKBONE, "active": active, "backbones": rows}
