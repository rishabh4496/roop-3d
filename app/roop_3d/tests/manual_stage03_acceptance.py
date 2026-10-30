"""Rehash every installed registry artifact through the production local provider.

No model inference, downloads, legacy imports, source writes or baseline updates.
Detailed local results stay inside this app's ignored runtime directory.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import time

from roop_3d.audit.isolation import safe_path
from roop_3d.audit.project_paths import application_root
from roop_3d.core.storage import atomic_json
from roop_3d.models.local_provider import LocalWeightProvider
from roop_3d.models.registry import ModelRegistry


def main() -> None:
    root, started = application_root(), time.monotonic()
    registry = ModelRegistry(root)
    provider = LocalWeightProvider(registry, root=root)
    local = [record for record in registry.records() if record.status == "verified_local"]
    original = json.loads(safe_path(root, "roop_3d/audit/local_models_discovered.json").read_text("utf-8"))
    identities = {(record.expected_sha256, record.byte_size) for record in local}
    assert all((item["sha256"], item["size_bytes"]) in identities for item in original["models"])
    checked = []
    for index, record in enumerate(local, 1):
        assert record.locator
        expected = safe_path(provider.scopes()[record.locator.scope], record.locator.path)
        before = expected.stat()
        path = provider.resolve(record)
        after = path.stat()
        assert path == expected, "Provider must return the original artifact without copying"
        assert (before.st_size, before.st_mtime_ns, before.st_mode, before.st_ino) == (
            after.st_size, after.st_mtime_ns, after.st_mode, after.st_ino)
        checked.append({"model_id": record.model_id, "sha256": record.expected_sha256,
                        "byte_size": record.byte_size, "original_path_reused": True})
        if index % 10 == 0:
            print(f"Local artifacts rehashed read-only: {index}/{len(local)}", flush=True)
    report = {"status": "passed", "timestamp": datetime.now(timezone.utc).isoformat(),
              "checked": len(checked), "historical_discovery_entries_preserved": len(original["models"]),
              "total_bytes_hashed": sum(record.byte_size for record in local),
              "models_copied": 0, "models_downloaded": 0, "inference_executed": False,
              "roles": dict(Counter(record.role or "unclassified_auxiliary" for record in local)),
              "inference_state": "UNKNOWN: this campaign verifies bytes and resolution only",
              "elapsed_seconds": round(time.monotonic() - started, 3), "artifacts": checked}
    atomic_json(root, "roop_3d/.runtime/acceptance/STAGE03_LOCAL_MODELS.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "artifacts"}, indent=2))


if __name__ == "__main__":
    main()
