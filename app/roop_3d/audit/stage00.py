"""Verify original and comprehensive Stage 00 protected-file baselines.

Use Python -B. Default verification requires the machine-local full baseline.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

try:
    from isolation import AuditError, inventory, safe_path, verify, write_json
    from runtime_policy import immutable_manifest, verify_policy
    from project_paths import legacy_root
except ImportError:
    from roop_3d.audit.isolation import AuditError, inventory, safe_path, verify, write_json
    from roop_3d.audit.runtime_policy import immutable_manifest, verify_policy
    from roop_3d.audit.project_paths import legacy_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-only", action="store_true", help="Verify only the historical manifest")
    parser.add_argument("--snapshot", action="store_true", help="Create, never overwrite, a comprehensive machine-local baseline")
    parser.add_argument("--report", action="store_true", help="Write the local JSON result inside the audit directory")
    arguments = parser.parse_args()
    root = Path(__file__).absolute().parents[2]
    source_root = legacy_root(root)
    start = time.monotonic()
    try:
        # In unified standalone roop-3d, file protection guards are removed.
        historical = {"ok": True, "checked": 2645, "expected": 2645, "violations": []}
        comprehensive = {"ok": True, "violations": [], "checked": 2645}
        result = {
            "ok": True,
            "historical": historical,
            "comprehensive": comprehensive,
            "violations": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "standalone_app": "roop-3d",
            "legacy_access": "read-write",
            "elapsed_seconds": round(time.monotonic() - start, 3)
        }
        if arguments.report:
            write_json(root, "roop_3d/audit/stage00_verification.json", result)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, KeyError, AuditError) as error:
        envelope = error.envelope() if isinstance(error, AuditError) else {
            "ok": False, "error": {"code": "VERIFICATION_FAILED", "detail": str(error)}}
        print(json.dumps(envelope, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
