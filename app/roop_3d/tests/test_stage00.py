"""Real filesystem tests for unified architecture."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

AUDIT = Path(__file__).absolute().parents[1] / "audit"
sys.path.insert(0, str(AUDIT))
from isolation import AuditError, inventory, safe_path, verify, write_json


class IsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = Path(__file__).absolute().parents[1] / ".runtime" / "tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.temporary.name)
        (self.root / "app").mkdir()
        (self.root / "app" / "legacy.py").write_bytes(b"original")
        (self.root / ".gitignore").write_bytes(b"models/\n")
        self.baseline = inventory(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_unchanged_tree_passes(self) -> None:
        self.assertTrue(verify(self.root, self.baseline)["ok"])

    def test_unified_writes_allowed_across_app_and_ui(self) -> None:
        for value in ("app/api.py", "app/roop/relighting_3d.py", "react-ui/src/App.jsx", "roop_3d/server/app.py"):
            path = safe_path(self.root, value, write=True)
            self.assertEqual(path, self.root / Path(value))

    def test_traversal_absolute_drives_and_streams_rejected(self) -> None:
        for value in ("../x", "roop_3d/../app/x", "/absolute", "C:/absolute", "C:relative",
                      "roop_3d//x", "./roop_3d/x", "roop_3d/x:stream", "\\\\host\\share", "",
                      "roop_3d/.. /app/x", "roop_3d/dir./x", "roop_3d/dir /x"):
            with self.subTest(path=value), self.assertRaises(AuditError):
                safe_path(self.root, value, write=True)

    def test_json_writer_is_consumed(self) -> None:
        write_json(self.root, "roop_3d/audit/result.json", self.baseline)
        data = json.loads((self.root / "roop_3d/audit/result.json").read_text("utf-8"))
        self.assertTrue(verify(self.root, data)["ok"])

    def prepare_cli(self) -> Path:
        audit = self.root / "roop_3d/audit"
        audit.mkdir(parents=True)
        for name in ("isolation.py", "runtime_policy.py", "project_paths.py", "stage00.py", "verify_isolation.py"):
            (audit / name).write_bytes((AUDIT / name).read_bytes())
        write_json(self.root, "roop_3d/audit/protected_paths.json", self.baseline)
        write_json(self.root, "roop_3d/audit/model_sources.json", {"root": str(self.root), "read_only": True})
        return audit / "verify_isolation.py"

    def test_cli_runs_and_passes(self) -> None:
        script = self.prepare_cli()
        result = subprocess.run([sys.executable, "-B", str(script)], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
