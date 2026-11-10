#!/usr/bin/env python3
"""Report which morphable-topology assets are present and say exactly how to get the missing ones.

Run from ``app/``:  python -B -m roop_3d.tools.check_topology_assets [--strict] [--json]

Advisory by default (exit 0) so an install never fails because an *optional* model is absent. ``--strict`` exits 1 when the
default topology (``makehuman_cc0``) is not ready. Nothing is downloaded and no model licence is accepted on anyone's behalf.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from roop_3d.audit.local_models import OPEN_ARTIFACTS
from roop_3d.core.errors import DomainError
from roop_3d.reconstruction.topology import FLAME_PATH_ENV, MorphableTopologyRegistry

HOWTO = {
    "makehuman_cc0": [
        "Build the CC0 MakeHuman head model once (downloads ~130 small files from a pinned commit, SHA-256 verified):",
        "    python -B -m roop_3d.reconstruction.makehuman_build        (run from the app/ folder)",
    ],
    "flame2023_open": [
        "Optional. Needed only for topology=flame2023_open. You must download it yourself (licence acceptance is manual):",
        "  1. Register and accept the licence at https://flame.is.tue.mpg.de/ (download the 'FLAME 2023 Open' package).",
        "  2. Use ONLY the file named flame2023_Open.pkl (CC-BY-4.0). The other FLAME files (FLAME 2020, flame2023.pkl,",
        "     flame2023_no_jaw.pkl) are non-commercial and are never loaded.",
        "  3. Put it at  flame/flame2023/flame2023_Open.pkl  in this checkout (or app/roop_3d/.runtime/models/flame2023_open/),",
        f"     or set {FLAME_PATH_ENV} to a path inside the checkout.",
        "  4. Re-run this check. The file is verified by SHA-256; a different file is rejected.",
    ],
}


def check(registry: MorphableTopologyRegistry | None = None) -> dict[str, Any]:
    registry = registry or MorphableTopologyRegistry()
    rows = []
    for topology_id in registry.ids():
        try:
            state = registry.availability(topology_id)
        except DomainError as error:        # e.g. a configured path outside the checkout
            state = {"available": False, "state": "path_rejected", "detail": error.message}
        spec = registry.spec(topology_id)
        rows.append({"id": topology_id, "title": spec.title, "license_id": spec.license_id, "default": topology_id == registry.default_id,
                     "available": state["available"], "state": state["state"], "detail": state["detail"],
                     "expected_sha256": OPEN_ARTIFACTS[spec.artifact_id]["sha256"] if spec.artifact_id else None,
                     "how_to_fix": [] if state["available"] else HOWTO.get(topology_id, [])})
    default_ready = next(r["available"] for r in rows if r["default"])
    return {"topologies": rows, "default_ready": default_ready, "all_ready": all(r["available"] for r in rows)}


def render(report: dict[str, Any]) -> str:
    out = ["Morphable topology assets"]
    for row in report["topologies"]:
        mark = "OK     " if row["available"] else ("MISSING" if row["state"] != "hash_mismatch" else "WRONG  ")
        out.append(f"  [{mark}] {row['id']}  ({row['title']}, {row['license_id']}){'  [default]' if row['default'] else ''}")
        if not row["available"]:
            if row["detail"]:
                out.append(f"      {row['detail']}")
            out += [f"      {line}" for line in row["how_to_fix"]]
    if report["all_ready"]:
        out.append("All topology assets are ready.")
    elif report["default_ready"]:
        out.append("The default topology is ready; the missing ones are optional and the app works without them.")
    else:
        out.append("The default topology is NOT ready: mesh requests with template=makehuman will return 404 until it is built.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strict", action="store_true", help="exit 1 if the default topology is not ready")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)
    report = check()
    print(json.dumps(report, indent=2) if args.json else render(report))
    return 1 if args.strict and not report["default_ready"] else 0


if __name__ == "__main__":
    sys.exit(main())
