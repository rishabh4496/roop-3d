"""Real filesystem tests for unified architecture."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

AUDIT = Path(__file__).absolute().parents[1] / "audit"
sys.path.insert(0, str(AUDIT))
from boundary import scan_sources
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

    def test_real_boundary_gate_fails_on_injected_sibling_import_and_write(self) -> None:
        app_dir = self.root / "app"
        with tempfile.TemporaryDirectory() as scratch:
            app_dir = Path(scratch) / "app"
            app_dir.mkdir()
            (app_dir / "bad.py").write_text(
                "from roop_ultimate.core import model\n"
                "from pathlib import Path\n"
                "Path('../roop-ultimate/output').write_text('forbidden')\n",
                encoding="utf-8")
            result = scan_sources(app_dir, siblings=("roop-ultimate",))
            self.assertTrue(result["imports"], "sibling source imports must fail the static gate")
            self.assertTrue(result["writes"], "writes targeting the sibling checkout must fail the static gate")


if __name__ == "__main__":
    unittest.main()
