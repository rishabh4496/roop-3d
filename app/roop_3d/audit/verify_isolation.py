#!/usr/bin/env python3
"""Entry point for the fail-closed isolation audit (see stage00.py / boundary.py).

Exit 0 means every check passed (sibling imports, sibling write targets, interpreter inside app/roop_3d/.venv, git origin
roop-3d, no tracked venv/cache/weight artefacts); exit 1 means a violation or a verification error. --report also writes
roop_3d/audit/stage00_verification.json.
"""
try:
    from stage00 import main
except ImportError:
    from roop_3d.audit.stage00 import main


if __name__ == "__main__":
    raise SystemExit(main())
