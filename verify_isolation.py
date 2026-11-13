#!/usr/bin/env python3
"""Root forwarder for app/roop_3d/audit/verify_isolation.py."""
from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root / "app"))

from roop_3d.audit.stage00 import main

if __name__ == "__main__":
    raise SystemExit(main())
