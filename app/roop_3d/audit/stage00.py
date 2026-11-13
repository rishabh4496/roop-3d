"""Fail-closed isolation audit for the standalone roop-3d checkout (``verify_isolation.py`` entry point).

Runs the five real checks of ``boundary.py`` (sibling imports, sibling write targets, interpreter inside ``app/roop_3d/.venv``,
git origin is roop-3d, no tracked venv/cache/weight artefacts). Every count is computed; exit 0 only when all pass, exit 1
on any violation or verification error. Use Python -B, from ``app/``:
``./roop_3d/.venv/Scripts/python.exe -B roop_3d/audit/verify_isolation.py --report``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

try:
    from isolation import AuditError, write_json
    from boundary import run_checks
except ImportError:
    from roop_3d.audit.isolation import AuditError, write_json
    from roop_3d.audit.boundary import run_checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="Also write the JSON result to roop_3d/audit/stage00_verification.json")
    arguments = parser.parse_args(argv)
    root = Path(__file__).absolute().parents[2]
    start = time.monotonic()
    try:
        result = run_checks(root)
        result["timestamp"] = datetime.now(timezone.utc).isoformat()
        result["elapsed_seconds"] = round(time.monotonic() - start, 3)
        if arguments.report:
            write_json(root, "roop_3d/audit/stage00_verification.json", result)
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1
    except (OSError, ValueError, KeyError, AuditError) as error:
        envelope = error.envelope() if isinstance(error, AuditError) else {
            "ok": False, "error": {"code": "VERIFICATION_FAILED", "detail": str(error)}}
        print(json.dumps(envelope, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
