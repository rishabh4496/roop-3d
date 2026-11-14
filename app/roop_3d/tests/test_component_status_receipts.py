"""Standalone forward receipts must not become provider residency or render readiness."""
from pathlib import Path

from roop_3d.server.model_status import _forward, _identity, _report


def test_forward_receipt_is_bound_to_resolved_file_identity(tmp_path: Path):
    source = tmp_path / "implementation.py"
    source.write_text("value = 1\n", encoding="utf-8")
    identity = _identity(str(source))
    row = {"component": "uv_synthesis"}
    receipt = {**identity, "component": "uv_synthesis", "provider": "CPU", "execution_verified": True, "error": None}
    assert _forward(row, identity, [receipt]) == {"state": "verified", "provider": "CPU", "error": None}
    source.write_text("value = 12\n", encoding="utf-8")
    assert _forward(row, _identity(str(source)), [receipt])["state"] == "stale"


def test_report_cannot_substitute_an_unrelated_file(tmp_path: Path):
    source = tmp_path / "implementation.py"
    source.write_text("value = 1\n", encoding="utf-8")
    identity = _identity(str(source))
    receipt = {**identity, "component": "uv_synthesis", "path": str(tmp_path / "unrelated.py"),
               "execution_verified": True, "provider": "CPU", "error": None}
    assert _forward({"component": "uv_synthesis"}, identity, [receipt])["state"] == "not_run"


def test_incomplete_or_conflicting_receipts_do_not_prove_a_forward(tmp_path: Path):
    source = tmp_path / "implementation.py"
    source.write_text("value = 1\n", encoding="utf-8")
    identity = _identity(str(source))
    row = {"component": "uv_synthesis"}
    receipt = {**identity, "component": "uv_synthesis", "execution_verified": True, "provider": "CPU", "error": "Forward failed"}
    assert _forward(row, identity, [receipt])["state"] == "failed"
    del receipt["mtime_ns"]
    assert _forward(row, identity, [receipt])["state"] == "stale"


def test_absent_report_is_not_a_forward_proof(tmp_path: Path):
    report, results = _report(tmp_path)
    assert report["state"] == "not_run"
    assert results == []


def test_malformed_report_is_reported_without_returning_its_contents(tmp_path: Path):
    report_path = tmp_path / "roop_3d/.runtime/stage1_verification.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text('{"schema_version": 99, "unrelated_path": "private"}', encoding="utf-8")
    report, results = _report(tmp_path)
    assert report["state"] == "invalid"
    assert results == []
    assert "private" not in str(report)


def test_missing_asset_preserves_the_resolver_error(tmp_path: Path):
    from roop_3d.server.model_status import _decorate
    error = {"code": "MAKEHUMAN_ASSET_MISSING", "message": "Build the pinned CC0 head asset", "status": 404}
    row = {"component": "head_reconstructor", "kind": "asset", "state": "MISSING",
           "path": str(tmp_path / "absent.npz"), "error": error, "assets": []}
    decorated = _decorate(row, [])
    assert decorated["error"] == error
    assert decorated["state"] == "MISSING"
    assert decorated["runtime"]["active_provider"] is None
    assert decorated["last_forward"]["state"] == "not_run"
