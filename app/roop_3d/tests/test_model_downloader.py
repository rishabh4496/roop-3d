"""Universal model download manager: resumable Range downloads, mirror failover, checksum recovery, licence gating,
offline procedural assets, and the REST / WebSocket surface. No test touches the network: every HTTP exchange is served by
``httpx.MockTransport`` from a small in-memory web."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from roop_3d.audit import local_models as lm
from roop_3d.audit import procedural_assets as pa
from roop_3d.audit.isolation import AuditError
from roop_3d.core.settings import Settings
from roop_3d.server.app import create_app
from roop_3d.server.ws import EVENT_ADAPTER

BASE = "https://huggingface.co/test"
STAGE = "STAGE_3_MAIN_SWAP"


def payload(seed: str, size: int = 300_000) -> bytes:
    block = hashlib.sha256(seed.encode()).digest()
    return (block * (size // len(block) + 1))[:size]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Resource:
    """One URL of the fake web. Knobs model the failures a real mirror shows."""

    def __init__(self, data: bytes = b"", *, status: int = 200, honor_range: bool = True, etag: str = '"v1"',
                 drop_after: int | None = None, drops: int = 1, redirect: str | None = None, omit_length: bool = False,
                 bad_content_range: bool = False, range_status: int | None = None, on_chunk=None):
        self.data, self.status, self.honor_range, self.etag = data, status, honor_range, etag
        self.drop_after, self.drops_left, self.redirect = drop_after, drops, redirect
        self.omit_length, self.bad_content_range, self.range_status, self.on_chunk = omit_length, bad_content_range, range_status, on_chunk


class _Stream(httpx.SyncByteStream):
    def __init__(self, body: bytes, error: Exception | None, on_chunk, step: int = 65_536):
        self.body, self.error, self.on_chunk, self.step = body, error, on_chunk, step

    def __iter__(self):
        for index in range(0, len(self.body), self.step):
            if self.on_chunk:
                self.on_chunk(index)
            yield self.body[index:index + self.step]
        if self.error:
            raise self.error


class FakeWeb:
    def __init__(self, resources: dict[str, Resource] | None = None):
        self.resources = dict(resources or {})
        self.log: list[tuple[str, str | None, str | None]] = []

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def requests_to(self, url: str) -> list[tuple[str | None, str | None]]:
        return [(r, i) for u, r, i in self.log if u == url]

    def handle(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        wanted = request.headers.get("range")
        self.log.append((url, wanted, request.headers.get("if-range")))
        res = self.resources.get(url)
        if res is None:
            return httpx.Response(404)
        if res.redirect:
            return httpx.Response(302, headers={"location": res.redirect})
        if res.status != 200:
            return httpx.Response(res.status)
        start, status = 0, 200
        if wanted:
            if res.range_status:
                return httpx.Response(res.range_status)
            if res.honor_range:
                start, status = int(re.fullmatch(r"bytes=(\d+)-", wanted)[1]), 206
        body = res.data[start:]
        headers = {"etag": res.etag}
        if not res.omit_length:
            headers["content-length"] = str(len(body))
        if status == 206:
            headers["content-range"] = (f"bytes 0-{len(res.data) - 1}/{len(res.data)}" if res.bad_content_range
                                        else f"bytes {start}-{len(res.data) - 1}/{len(res.data)}")
        error = None
        if res.drop_after is not None and res.drops_left > 0:
            res.drops_left -= 1
            body, error = body[:res.drop_after], httpx.ReadError("connection reset")
        return httpx.Response(status, headers=headers, stream=_Stream(body, error, res.on_chunk))


def entry_dict(entry_id: str, data: bytes, mirrors: list[str], **over) -> dict:
    raw = {"id": entry_id, "name": entry_id, "kind": "weight", "category": "test", "install_root": "app",
           "install_path": f"models/{entry_id}.bin", "mirrors": mirrors, "sha256": sha(data), "size_bytes": len(data),
           "license": "MIT", "license_url": None, "commercial_use": "allow", "requires_acceptance": False,
           "required_for_stages": [STAGE], "optional": False, "acquisition": "zero_auth"}
    raw.update(over)
    return raw


def make_manager(root: Path, web: FakeWeb, *raw: dict, **kwargs) -> tuple[lm.UniversalModelDownloadManager, list[dict]]:
    events: list[dict] = []
    manager = lm.UniversalModelDownloadManager(
        root, manifest={r["id"]: lm.ManifestEntry.from_dict(r) for r in raw}, transport=web.transport(),
        sink=events.append, handlers=kwargs.pop("handlers", {}), sleep=lambda seconds: None, **kwargs)
    return manager, events


def run(manager: lm.UniversalModelDownloadManager, ids=None, **kwargs) -> tuple[dict, dict]:
    batch = manager.start_batch(ids, **kwargs)
    return batch, manager.wait(batch["batch_id"], timeout=30)


def leftovers(root: Path) -> list[str]:
    return sorted(p.name for p in root.rglob("*") if p.is_file() and (p.suffix == ".part" or p.name.endswith(".part.json")))


# ------------------------------------------------------------------------------------------- resumable downloading
def test_download_goes_through_part_and_appears_atomically_complete(tmp_path):
    data = payload("a")
    final = tmp_path / "models/a.bin"
    seen: list[tuple[bool, bool]] = []
    web = FakeWeb({f"{BASE}/a": Resource(data, on_chunk=lambda i: seen.append((final.exists(), final.with_name("a.bin.part").exists())))})
    manager, events = make_manager(tmp_path, web, entry_dict("a", data, [f"{BASE}/a"]))
    _, outcome = run(manager)
    assert outcome["entries"]["a"]["status"] == "ready"
    assert final.read_bytes() == data and leftovers(tmp_path) == []
    assert seen and all(not complete and has_part for complete, has_part in seen[1:])   # never visible under its final name while in flight
    assert [e["status"] for e in events if e["model_id"] == "a"][-1] == "ready"


def test_interrupted_transfer_resumes_from_the_part_with_a_range_request(tmp_path):
    data = payload("b")
    url = f"{BASE}/b"
    web = FakeWeb({url: Resource(data, drop_after=100_000)})
    manager, _ = make_manager(tmp_path, web, entry_dict("b", data, [url]))
    _, outcome = run(manager)
    assert outcome["entries"]["b"]["status"] == "ready"
    assert (tmp_path / "models/b.bin").read_bytes() == data
    first, second = web.requests_to(url)
    assert first == (None, None)
    assert second == ("bytes=100000-", '"v1"')            # resumed exactly where it stopped, guarded by the validator
    assert leftovers(tmp_path) == []


def test_part_from_an_earlier_run_is_resumed(tmp_path):
    data = payload("c")
    part = tmp_path / "models/c.bin.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(data[:70_000])
    web = FakeWeb({f"{BASE}/c": Resource(data)})
    manager, _ = make_manager(tmp_path, web, entry_dict("c", data, [f"{BASE}/c"]))
    assert manager.inspect("c")["state"] == "PARTIAL" and manager.inspect("c")["partial_bytes"] == 70_000
    run(manager)
    assert web.requests_to(f"{BASE}/c") == [("bytes=70000-", None)]
    assert (tmp_path / "models/c.bin").read_bytes() == data


def test_server_that_ignores_range_restarts_from_zero(tmp_path):
    data = payload("d")
    part = tmp_path / "models/d.bin.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(data[:60_000])
    web = FakeWeb({f"{BASE}/d": Resource(data, honor_range=False)})
    manager, _ = make_manager(tmp_path, web, entry_dict("d", data, [f"{BASE}/d"]))
    _, outcome = run(manager)
    assert outcome["entries"]["d"]["status"] == "ready"
    assert (tmp_path / "models/d.bin").read_bytes() == data      # not 60 000 duplicated bytes followed by a full copy


def test_range_not_satisfiable_discards_the_part_and_restarts(tmp_path):
    data = payload("e")
    part = tmp_path / "models/e.bin.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(data[:60_000])
    web = FakeWeb({f"{BASE}/e": Resource(data, range_status=416)})
    manager, _ = make_manager(tmp_path, web, entry_dict("e", data, [f"{BASE}/e"]))
    _, outcome = run(manager)
    assert outcome["entries"]["e"]["status"] == "ready"
    assert web.requests_to(f"{BASE}/e") == [("bytes=60000-", None), (None, None)]


def test_complete_part_left_by_a_crash_is_verified_and_finished_without_any_request(tmp_path):
    data = payload("f")
    part = tmp_path / "models/f.bin.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(data)
    web = FakeWeb()
    manager, _ = make_manager(tmp_path, web, entry_dict("f", data, [f"{BASE}/f"]))
    _, outcome = run(manager)
    assert outcome["entries"]["f"]["status"] == "ready" and web.log == []
    assert (tmp_path / "models/f.bin").read_bytes() == data and not part.exists()


# ------------------------------------------------------------------------------------------- checksum failure & mirrors
def test_wrong_bytes_are_dropped_and_the_next_mirror_supplies_the_file(tmp_path):
    data = payload("g")
    bad = payload("evil", len(data))
    a, b = f"{BASE}/g-a", f"{BASE}/g-b"
    web = FakeWeb({a: Resource(bad), b: Resource(data)})
    final = tmp_path / "models/g.bin"
    manager, events = make_manager(tmp_path, web, entry_dict("g", data, [a, b]))
    _, outcome = run(manager)
    assert outcome["entries"]["g"] == {"status": "ready", "error": None, "mirror_used": b}
    assert final.read_bytes() == data and sha(final.read_bytes()) == sha(data)
    assert leftovers(tmp_path) == []                                 # the corrupt .part was not kept
    assert len(web.requests_to(a)) == 1                               # a clean transfer that mismatched is not retried
    assert [e["mirror_used"] for e in events if e["status"] == "ready"] == [b]


def test_corrupt_bytes_resumed_from_a_part_are_refetched_clean_from_the_same_mirror(tmp_path):
    data = payload("h")
    part = tmp_path / "models/h.bin.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(payload("garbage", 50_000))                      # a prefix that is not this file's prefix
    url = f"{BASE}/h"
    web = FakeWeb({url: Resource(data)})
    manager, _ = make_manager(tmp_path, web, entry_dict("h", data, [url]))
    _, outcome = run(manager)
    assert outcome["entries"]["h"]["status"] == "ready"
    assert web.requests_to(url) == [("bytes=50000-", None), (None, None)]      # resumed, failed the hash, started over
    assert (tmp_path / "models/h.bin").read_bytes() == data


def test_missing_and_locked_mirrors_fall_through_without_retrying(tmp_path):
    data = payload("i")
    a, b, c = f"{BASE}/i-a", f"{BASE}/i-b", f"{BASE}/i-c"
    web = FakeWeb({a: Resource(status=404), b: Resource(status=403), c: Resource(data)})
    manager, _ = make_manager(tmp_path, web, entry_dict("i", data, [a, b, c]))
    _, outcome = run(manager)
    assert outcome["entries"]["i"]["mirror_used"] == c
    assert len(web.requests_to(a)) == 1 and len(web.requests_to(b)) == 1


def test_every_failure_reason_is_reported_when_all_mirrors_fail(tmp_path):
    data = payload("j")
    a, b = f"{BASE}/j-a", f"{BASE}/j-b"
    web = FakeWeb({a: Resource(status=404), b: Resource(status=503)})
    manager, events = make_manager(tmp_path, web, entry_dict("j", data, [a, b]))
    _, outcome = run(manager)
    assert outcome["entries"]["j"]["status"] == "failed" and outcome["entries"]["j"]["error"] == "ALL_MIRRORS_FAILED"
    detail = [e for e in events if e["model_id"] == "j"][-1]["detail"]
    assert "HTTP_404" in detail and "HTTP_503" in detail
    assert len(web.requests_to(b)) == 2                                # transient errors are retried once before moving on
    assert not (tmp_path / "models/j.bin").exists()


def test_redirects_may_only_land_on_storage_hosts(tmp_path):
    data = payload("k")
    a, b = f"{BASE}/k-a", f"{BASE}/k-b"
    web = FakeWeb({a: Resource(redirect="https://evil.example/k"), "https://evil.example/k": Resource(data),
                   b: Resource(redirect="https://cas-bridge.xethub.hf.co/k"), "https://cas-bridge.xethub.hf.co/k": Resource(data)})
    manager, _ = make_manager(tmp_path, web, entry_dict("k", data, [a, b]))
    _, outcome = run(manager)
    assert outcome["entries"]["k"]["mirror_used"] == b
    assert all(url != "https://evil.example/k" for url, _, _ in web.log)       # never even contacted


def test_response_longer_than_the_pinned_size_is_refused_and_not_kept(tmp_path):
    data = payload("l")
    a, b = f"{BASE}/l-a", f"{BASE}/l-b"
    web = FakeWeb({a: Resource(data + b"extra", omit_length=True), b: Resource(data)})
    manager, _ = make_manager(tmp_path, web, entry_dict("l", data, [a, b]))
    _, outcome = run(manager)
    assert outcome["entries"]["l"]["mirror_used"] == b
    assert web.requests_to(b) == [(None, None)]                        # the oversize source left no .part to resume from


def test_a_content_range_that_does_not_match_the_request_is_refused(tmp_path):
    data = payload("m")
    part = tmp_path / "models/m.bin.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(data[:40_000])
    a, b = f"{BASE}/m-a", f"{BASE}/m-b"
    web = FakeWeb({a: Resource(data, bad_content_range=True), b: Resource(data)})
    manager, _ = make_manager(tmp_path, web, entry_dict("m", data, [a, b]))
    _, outcome = run(manager)
    assert outcome["entries"]["m"]["mirror_used"] == b and (tmp_path / "models/m.bin").read_bytes() == data


def test_cancel_stops_the_transfer_and_keeps_the_part_for_resume(tmp_path):
    data = payload("n", 600_000)
    holder: dict = {}
    started = threading.Event()

    def on_chunk(index: int) -> None:
        if index >= 131_072:
            started.wait(5)
            holder["manager"].cancel(holder["batch"])

    web = FakeWeb({f"{BASE}/n": Resource(data, on_chunk=on_chunk)})
    manager, _ = make_manager(tmp_path, web, entry_dict("n", data, [f"{BASE}/n"]))
    holder["manager"] = manager
    batch = manager.start_batch()
    holder["batch"] = batch["batch_id"]
    started.set()
    outcome = manager.wait(batch["batch_id"], timeout=30)
    assert outcome["entries"]["n"]["status"] == "cancelled" and outcome["state"] == "cancelled"
    assert not (tmp_path / "models/n.bin").exists()
    assert 0 < (tmp_path / "models/n.bin.part").stat().st_size < len(data)


def test_not_enough_disk_space_fails_before_any_request(tmp_path, monkeypatch):
    data = payload("o")
    web = FakeWeb({f"{BASE}/o": Resource(data)})
    manager, events = make_manager(tmp_path, web, entry_dict("o", data, [f"{BASE}/o"]))
    from collections import namedtuple
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(lm.shutil, "disk_usage", lambda path: Usage(10**12, 10**12 - 1000, 1000))
    _, outcome = run(manager)
    assert outcome["entries"]["o"]["error"] == "INSUFFICIENT_DISK_SPACE" and web.log == []


# ------------------------------------------------------------------------------------------- progress events
def test_progress_events_carry_the_required_fields_and_validate_as_broker_events(tmp_path):
    data = payload("p", 400_000)
    web = FakeWeb({f"{BASE}/p": Resource(data)})
    ticks = iter(range(1, 10_000))
    manager, events = make_manager(tmp_path, web, entry_dict("p", data, [f"{BASE}/p"]), clock=lambda: next(ticks) * 0.3)
    run(manager)
    frames = [e for e in events if e["model_id"] == "p"]
    for field in ("model_id", "downloaded_bytes", "total_bytes", "speed_mbps", "eta_seconds", "mirror_used", "status"):
        assert all(field in f for f in frames), field
    assert [f["status"] for f in frames][0] == "queued" and frames[-1]["status"] == "ready"
    assert {"downloading", "verifying", "ready"} <= {f["status"] for f in frames}
    assert any(f["speed_mbps"] > 0 and isinstance(f["eta_seconds"], float) for f in frames if f["status"] == "downloading")
    assert frames[-1]["downloaded_bytes"] == frames[-1]["total_bytes"] == len(data) and frames[-1]["percent"] == 100.0
    assert frames[-1]["mirror_used"] == f"{BASE}/p"
    for frame in frames:
        event = EVENT_ADAPTER.validate_python(frame)
        assert event.type == "model_download_progress" and event.downloaded_bytes == event.bytes_downloaded


# ------------------------------------------------------------------------------------------- asset sets
def set_entry(files: dict[str, bytes], mirrors: list[str], entry_id: str = "pack") -> dict:
    return {"id": entry_id, "name": entry_id, "kind": "asset_set", "category": "3d_asset", "install_root": "app",
            "install_path": f"assets/{entry_id}", "mirrors": mirrors, "license": "CC0-1.0", "commercial_use": "allow",
            "requires_acceptance": False, "required_for_stages": [STAGE], "optional": False, "acquisition": "zero_auth",
            "files": [{"path": p, "size_bytes": len(d), "sha256": sha(d)} for p, d in files.items()]}


def test_asset_set_fetches_members_concurrently_under_one_steady_status(tmp_path):
    files = {f"targets/part{i}.target": payload(f"m{i}", 20_000 + i) for i in range(12)}
    base = "https://cdn.jsdelivr.net/gh/o/r@abc/data/"
    web = FakeWeb({base + p: Resource(d) for p, d in files.items()})
    manager, events = make_manager(tmp_path, web, set_entry(files, [base]))
    _, outcome = run(manager)
    assert outcome["entries"]["pack"]["status"] == "ready"
    assert all((tmp_path / "assets/pack" / p).read_bytes() == d for p, d in files.items()) and leftovers(tmp_path) == []
    frames = [e for e in events if e["model_id"] == "pack"]
    assert {f["status"] for f in frames} == {"queued", "downloading", "ready"}        # no per-file flapping
    assert frames[-1]["downloaded_bytes"] == frames[-1]["total_bytes"] == sum(map(len, files.values()))
    assert [f["downloaded_bytes"] for f in frames if f["status"] == "downloading"] == sorted(
        f["downloaded_bytes"] for f in frames if f["status"] == "downloading")        # bytes only ever go up
    row = manager.inspect("pack")
    assert row["state"] == "READY" and (row["files_ready"], row["files_total"]) == (12, 12)


def test_asset_set_resumes_by_fetching_only_what_is_missing_or_wrong(tmp_path):
    files = {f"f{i}.obj": payload(f"s{i}", 15_000) for i in range(6)}
    base = "https://raw.githubusercontent.com/o/r/abc/data/"
    web = FakeWeb({base + p: Resource(d) for p, d in files.items()})
    manager, _ = make_manager(tmp_path, web, set_entry(files, [base]))
    run(manager)
    folder = tmp_path / "assets/pack"
    (folder / "f2.obj").unlink()
    (folder / "f4.obj").write_bytes(payload("tampered", 15_000))
    row = manager.inspect("pack")
    assert row["state"] == "CORRUPT" and row["files_ready"] == 4
    before = len(web.log)
    _, outcome = run(manager)
    assert outcome["entries"]["pack"]["status"] == "ready"
    assert sorted(url.rsplit("/", 1)[1] for url, _, _ in web.log[before:]) == ["f2.obj", "f4.obj"]
    assert (folder / "f4.obj").read_bytes() == files["f4.obj"]
    assert any(".quarantine" in str(p) for p in folder.rglob("*"))                  # the tampered file was kept aside


def test_asset_set_members_fail_over_between_mirrors_independently(tmp_path):
    files = {"a.obj": payload("A", 12_000), "b.obj": payload("B", 12_000)}
    down, up = "https://raw.githubusercontent.com/o/r/abc/", "https://cdn.jsdelivr.net/gh/o/r@abc/"
    web = FakeWeb({down + "a.obj": Resource(files["a.obj"]), up + "a.obj": Resource(files["a.obj"]),
                   up + "b.obj": Resource(files["b.obj"])})                       # the first mirror lacks b.obj
    manager, events = make_manager(tmp_path, web, set_entry(files, [down, up]))
    _, outcome = run(manager)
    assert outcome["entries"]["pack"]["status"] == "ready" and outcome["entries"]["pack"]["mirror_used"] in {down + "a.obj", up + "b.obj"}
    assert web.requests_to(down + "b.obj") and web.requests_to(up + "b.obj")


def test_one_missing_member_fails_the_set_and_keeps_the_others(tmp_path):
    files = {"ok1.obj": payload("1", 12_000), "ok2.obj": payload("2", 12_000), "gone.obj": payload("3", 12_000)}
    base = "https://raw.githubusercontent.com/o/r/abc/"
    web = FakeWeb({base + "ok1.obj": Resource(files["ok1.obj"]), base + "ok2.obj": Resource(files["ok2.obj"])})
    manager, events = make_manager(tmp_path, web, set_entry(files, [base]))
    _, outcome = run(manager)
    assert outcome["entries"]["pack"]["error"] == "ALL_MIRRORS_FAILED" and outcome["state"] == "failed"
    assert [e for e in events if e["model_id"] == "pack"][-1]["status"] == "failed"
    assert manager.inspect("pack")["state"] in {"PARTIAL", "MISSING"} and not (tmp_path / "assets/pack/gone.obj").exists()


# ------------------------------------------------------------------------------------------- licence gate
def test_restricted_licences_need_acknowledgement_and_nothing_is_fetched_without_it(tmp_path):
    data = payload("q")
    web = FakeWeb({f"{BASE}/q": Resource(data)})
    raw = entry_dict("q", data, [f"{BASE}/q"], license="UNVERIFIED", commercial_use="deny", requires_acceptance=True,
                     license_note="no licence verified")
    manager, _ = make_manager(tmp_path, web, raw)
    batch, _ = run(manager)
    assert batch["queued"] == [] and batch["skipped"][0]["reason"] == "LICENSE_ACCEPTANCE_REQUIRED" and web.log == []
    _, outcome = run(manager, accepted_license=True)
    assert outcome["entries"]["q"]["status"] == "ready"


def test_commercial_use_is_refused_for_models_that_do_not_allow_it_even_when_acknowledged(tmp_path):
    data = payload("r")
    web = FakeWeb({f"{BASE}/r": Resource(data)})
    raw = entry_dict("r", data, [f"{BASE}/r"], license="UNVERIFIED", commercial_use="deny", requires_acceptance=True)
    manager, _ = make_manager(tmp_path, web, raw)
    batch, _ = run(manager, accepted_license=True, commercial=True)
    assert batch["skipped"][0]["reason"] == "COMMERCIAL_USE_NOT_PERMITTED" and web.log == []


def test_entries_without_a_public_source_are_reported_not_invented(tmp_path):
    data = payload("s")
    raw = entry_dict("s", data, [], acquisition="registration_required", license="CC-BY-4.0", note="needs a free account")
    manager, _ = make_manager(tmp_path, FakeWeb(), raw)
    batch, _ = run(manager)
    assert batch["queued"] == [] and batch["skipped"] == [
        {"id": "s", "reason": "NO_PUBLIC_SOURCE", "detail": "needs a free account"}]


def test_acknowledgement_is_required_by_construction_for_non_open_licences():
    raw = entry_dict("t", payload("t"), [], license="UNVERIFIED", commercial_use="deny", requires_acceptance=False)
    with pytest.raises(AuditError, match="requires_acceptance"):
        lm.ManifestEntry.from_dict(raw)


# ------------------------------------------------------------------------------------------- inspection & repair
def test_inspect_reports_missing_partial_corrupt_and_ready_and_finds_renamed_files(tmp_path):
    data = payload("u")
    manager, _ = make_manager(tmp_path, FakeWeb(), entry_dict("u", data, [f"{BASE}/u"]))
    assert manager.inspect("u")["state"] == "MISSING"
    models = tmp_path / "models"
    models.mkdir()
    (models / "u.bin").write_bytes(payload("other", len(data)))
    assert manager.inspect("u")["state"] == "CORRUPT"
    (models / "u.bin").unlink()
    (models / "somebody_renamed_it.onnx").write_bytes(data)
    row = manager.inspect("u")
    assert row["state"] == "READY" and row["location"] == "models/somebody_renamed_it.onnx" and row["hash_status"] == "verified"


def test_manifest_listing_has_licence_badges_hash_status_and_disk_usage(tmp_path):
    good, other = payload("v"), payload("w")
    (tmp_path / "models").mkdir()
    (tmp_path / "models/v.bin").write_bytes(good)
    manager, _ = make_manager(
        tmp_path, FakeWeb(), entry_dict("v", good, [f"{BASE}/v"]),
        entry_dict("w", other, [f"{BASE}/w"], license="UNVERIFIED", commercial_use="deny", requires_acceptance=True))
    listing = manager.manifest()
    rows = {r["id"]: r for r in listing["models"]}
    assert rows["v"]["license_badge"] == "open" and rows["v"]["hash_status"] == "verified" and rows["v"]["installed"]
    assert rows["w"]["license_badge"] == "unverified" and rows["w"]["hash_status"] == "not_installed" and not rows["w"]["installed"]
    assert listing["disk"]["installed_bytes"] == len(good) and listing["disk"]["download_needed_bytes"] == len(other)
    assert listing["disk"]["free_bytes"] > 0 and listing["summary"]["by_state"] == {"READY": 1, "MISSING": 1}
    assert str(tmp_path) not in json.dumps(listing)                    # no absolute local paths leave the process


def test_verify_and_repair_sets_corrupt_files_aside_and_refetches_missing_and_bad_ones(tmp_path):
    ok, bad, gone = payload("x1"), payload("x2"), payload("x3")
    models = tmp_path / "models"
    models.mkdir()
    (models / "x1.bin").write_bytes(ok)
    wrong = payload("not x2", len(bad))
    (models / "x2.bin").write_bytes(wrong)
    web = FakeWeb({f"{BASE}/x2": Resource(bad), f"{BASE}/x3": Resource(gone)})
    manager, _ = make_manager(tmp_path, web, entry_dict("x1", ok, [f"{BASE}/x1"]), entry_dict("x2", bad, [f"{BASE}/x2"]),
                              entry_dict("x3", gone, [f"{BASE}/x3"]))
    before = (models / "x1.bin").stat().st_mtime_ns
    report = manager.verify_and_repair()
    assert report["ok"] is True and report["corrupt"] == ["x2"] and sorted(report["missing"]) == ["x2", "x3"]
    moved = tmp_path / report["quarantined"][0]["moved_to"]
    assert moved.read_bytes() == wrong and moved.parent.name == ".quarantine"       # set aside, not destroyed
    outcome = manager.wait(report["repair"]["batch_id"], timeout=30)
    assert {k: v["status"] for k, v in outcome["entries"].items()} == {"x2": "ready", "x3": "ready"}
    assert (models / "x2.bin").read_bytes() == bad and (models / "x3.bin").read_bytes() == gone
    assert (models / "x1.bin").stat().st_mtime_ns == before and web.requests_to(f"{BASE}/x1") == []
    assert manager.verify_and_repair(repair=False)["missing"] == []


def test_an_active_entry_is_not_started_twice(tmp_path):
    data = payload("y", 500_000)
    gate = threading.Event()
    web = FakeWeb({f"{BASE}/y": Resource(data, on_chunk=lambda i: gate.wait(5))})
    manager, _ = make_manager(tmp_path, web, entry_dict("y", data, [f"{BASE}/y"]))
    first = manager.start_batch()
    second = manager.start_batch()
    assert first["queued"] == ["y"] and second["queued"] == [] and second["skipped"][0]["reason"] == "ALREADY_DOWNLOADING"
    gate.set()
    manager.wait(first["batch_id"], timeout=30)


# ------------------------------------------------------------------------------------------- URL policy & manifest
@pytest.mark.parametrize("url,redirect,allowed", [
    ("https://huggingface.co/a/b/resolve/rev/f.onnx", False, True),
    ("https://github.com/o/r/releases/download/t/f.onnx", False, True),
    ("https://cas-bridge.xethub.hf.co/x", True, True),
    ("https://cas-bridge.xethub.hf.co/x", False, False),           # storage hosts are redirect targets only
    ("https://release-assets.githubusercontent.com/x", True, True),
    ("http://huggingface.co/a", False, False),
    ("https://huggingface.co.evil.example/a", False, False),
    ("https://evil.example/huggingface.co/a", False, False),
    ("https://user:pw@huggingface.co/a", False, False),
    ("https://huggingface.co:8443/a", False, False),
    ("https://huggingface.co/a/../b", False, False),
    ("https://huggingface.co/a/%2e%2e/b", False, False),
])
def test_url_policy(url, redirect, allowed):
    assert lm._url_allowed(url, redirect=redirect) is allowed


@pytest.mark.parametrize("override,message", [
    ({"mirrors": ["http://huggingface.co/x"]}, "approved host"),
    ({"mirrors": ["https://example.com/x"]}, "approved host"),
    ({"install_path": "../escape.bin"}, "escape.bin"),
    ({"install_path": "/abs/escape.bin"}, "escape.bin"),
    ({"sha256": "abc"}, "sha256"),
    ({"kind": "teapot"}, "kind"),
    ({"install_root": "system"}, "install_root"),
])
def test_manifest_entries_are_validated(override, message):
    raw = entry_dict("z", payload("z"), [f"{BASE}/z"])
    raw.update(override)
    with pytest.raises(AuditError, match=message):
        lm.ManifestEntry.from_dict(raw)


def test_two_entries_cannot_install_to_the_same_file(tmp_path):
    document = {"schema": lm.MANIFEST_SCHEMA, "models": [entry_dict("a", payload("a"), []), entry_dict("b", payload("b"), [], install_path="models/A.bin")]}
    path = tmp_path / "m.json"
    path.write_text(json.dumps(document))
    with pytest.raises(AuditError, match="install to"):
        lm.load_manifest(path)


class TestCommittedManifest:
    manifest = lm.load_manifest()

    def test_covers_every_family_the_pipeline_needs(self):
        ids = set(self.manifest)
        by_category = {}
        for entry in self.manifest.values():
            by_category.setdefault(entry.category, []).append(entry.id)
        for category in ("detection", "recognition", "swapper_2d", "restoration", "occlusion", "face_parsing", "landmarks_3d"):
            assert by_category.get(category), category
        assert any(i.startswith("arcface") for i in ids)
        assert {"makehuman_cc0_sources", "makehuman_cc0_head", "flame2023_open", "smirk_em1_checkpoint", "cc0_probe_bundle",
                "cc0_studio_hdri_sources", "correspondence_maps", "anatomical_pack_cc0"} <= ids

    def test_every_downloadable_entry_is_fully_pinned_and_zero_auth(self):
        for entry in self.manifest.values():
            if entry.kind in {"weight", "asset_set"} and entry.acquisition == "zero_auth":
                assert entry.mirrors, entry.id
                assert len(set(entry.mirrors)) == len(entry.mirrors), entry.id
                assert all(m.startswith("https://") for m in entry.mirrors)
            if entry.kind == "weight":
                assert lm._hex64(entry.sha256) and entry.size_bytes > 0, entry.id
            if entry.kind == "asset_set":
                assert all(lm._hex64(f["sha256"]) and f["size_bytes"] > 0 for f in entry.files), entry.id
            assert entry.license not in {"", "UNKNOWN"} and entry.commercial_use in {"allow", "deny", "unknown"}
            assert set(entry.required_for_stages) <= {"STAGE_1_SPARSE_PRESCAN", "STAGE_2_ACTIVE_PREPASS", STAGE}

    def test_a_missing_public_source_is_stated_rather_than_hidden(self):
        flame = self.manifest["flame2023_open"]
        assert flame.mirrors == () and flame.acquisition == "registration_required" and flame.license == "CC-BY-4.0"
        assert flame.sha256 == "e75a0990728ba038c7da2a420ae4396f7ddd7781366c13026066c5b24f127623"

    def test_licence_labels_are_never_more_permissive_than_the_registry_says(self):
        from roop_3d.models.registry import ModelRegistry
        registry = ModelRegistry()
        for entry in self.manifest.values():
            for registry_id in entry.registry_ids:
                record = registry.get(registry_id)
                assert (record.expected_sha256, record.byte_size) == (entry.sha256, entry.size_bytes), entry.id
                if record.commercial_use != "allow":
                    assert entry.commercial_use != "allow" and entry.requires_acceptance, entry.id
        assert self.manifest["detector_det_10g"].license_badge == "restricted"
        assert self.manifest["swapper_inswapper_128"].license_badge == "unverified"

    def test_open_licences_are_exactly_the_ones_that_skip_acknowledgement(self):
        for entry in self.manifest.values():
            assert (entry.license in lm.OPEN_LICENSES) == (entry.license_badge == "open")
            if entry.license in lm.OPEN_LICENSES:
                assert entry.commercial_use == "allow" and not entry.requires_acceptance, entry.id
            else:
                assert entry.requires_acceptance, entry.id


# ------------------------------------------------------------------------------------------- procedural assets
def test_makehuman_head_restores_from_the_bundled_build_with_the_pinned_digest(tmp_path):
    assert pa.BUNDLED_MAKEHUMAN.is_file(), "the compact CC0 build must ship with the app"
    expected = lm.load_manifest()["makehuman_cc0_head"].content_digest
    target = tmp_path / "makehuman/makehuman_head_v1.npz"
    pa.build_makehuman_cc0_head(target)
    assert pa.verify_makehuman_cc0_head(target, expected)["ok"] is True
    target.write_bytes(target.read_bytes()[:-50])
    assert pa.verify_makehuman_cc0_head(target, expected)["ok"] is False


def test_makehuman_head_compiles_from_cached_sources_with_no_network_and_matches_the_pin(tmp_path, monkeypatch):
    from roop_3d.reconstruction import makehuman_build as builder
    cache = builder.DEFAULT_ASSET_DIR / "sources"
    if not pa._makehuman_cache_complete(cache):
        pytest.skip("the 130 pinned MakeHuman sources are not cached on this machine")
    out = tmp_path / "makehuman"
    shutil.copytree(cache, out / "sources")
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network used")))
    target = pa.build_makehuman_cc0_head(out / builder.ASSET_NAME, bundled=tmp_path / "does-not-exist.npz")
    assert pa.verify_makehuman_cc0_head(target, lm.load_manifest()["makehuman_cc0_head"].content_digest)["ok"] is True


def test_makehuman_head_without_sources_or_bundle_says_so(tmp_path):
    with pytest.raises(AuditError, match="PROCEDURAL_SOURCE_MISSING|not fully cached"):
        pa.build_makehuman_cc0_head(tmp_path / "x/makehuman_head_v1.npz", bundled=tmp_path / "none.npz")


def test_procedural_probe_bundle_is_deterministic_cc0_and_loads_like_the_real_one(tmp_path):
    from roop_3d.rendering.physical_camera import load_cc0_preview_assets
    a = pa.build_cc0_probe_bundle_offline(tmp_path / "a/probes.npz")
    b = pa.build_cc0_probe_bundle_offline(tmp_path / "b/probes.npz")
    result = pa.verify_cc0_probe_bundle(a)
    assert result["ok"] and result["kind"] == "procedural" and result["digest"] == pa.verify_cc0_probe_bundle(b)["digest"]
    assets = load_cc0_preview_assets(a)
    assert len(assets.probes) == 5 and assets.metadata["license"] == "CC0-1.0"
    assert all(p.sh_coefficients.shape == (9, 3) and (p.environment_bgr > 0).all() for p in assets.probes)
    assert "not Poly Haven" in assets.metadata["environment_source"]               # labelled as the stand-in it is
    assert len({p.sh_coefficients.tobytes() for p in assets.probes}) == 5            # five different lighting setups


def test_shipped_probe_bundle_matches_its_pinned_digest():
    entry = lm.load_manifest()["cc0_probe_bundle"]
    result = pa.verify_cc0_probe_bundle(Path(pa.__file__).resolve().parents[1] / "rendering/assets/stage10_cc0_probes.npz", entry.content_digest)
    assert result["ok"] and result["kind"] == "pinned"


def test_correspondence_maps_compile_deterministically_offline_from_two_local_meshes(tmp_path, monkeypatch):
    import numpy as np
    from roop_3d.reconstruction.topology import TopologyMesh

    def sphere(rings: int, segments: int, wobble: float) -> TopologyMesh:
        vertices = [[0.0, 0.1, 0.0]]
        for r in range(1, rings):
            theta = np.pi * r / rings
            for s in range(segments):
                phi = 2 * np.pi * s / segments
                vertices.append([0.08 * np.sin(theta) * np.cos(phi) * (1 + wobble * np.cos(3 * phi)), 0.1 * np.cos(theta),
                                 0.09 * np.sin(theta) * np.sin(phi)])
        vertices.append([0.0, -0.1, 0.0])
        faces = [[0, 1 + (s + 1) % segments, 1 + s] for s in range(segments)]
        for r in range(rings - 2):
            a, b = 1 + r * segments, 1 + (r + 1) * segments
            for s in range(segments):
                n = (s + 1) % segments
                faces += [[a + s, a + n, b + s], [a + n, b + n, b + s]]
        last, bottom = 1 + (rings - 2) * segments, len(vertices) - 1
        faces += [[last + s, last + (s + 1) % segments, bottom] for s in range(segments)]
        return TopologyMesh("synthetic", np.asarray(vertices, np.float64), np.asarray(faces, np.int64))

    import socket
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network used")))
    source, target = sphere(14, 20, 0.0), sphere(11, 15, 0.02)
    first = pa.compile_correspondence_maps(source, target)
    second = pa.compile_correspondence_maps(source, target)
    assert first.keys() == second.keys() and all(np.array_equal(first[k], second[k]) for k in first)
    path = tmp_path / "maps.npz"
    np.savez_compressed(path, **first)
    assert pa.verify_correspondence_maps(path)["ok"] is True
    broken = dict(first, target_data=first["target_data"] * 1.5)
    np.savez_compressed(tmp_path / "broken.npz", **broken)
    assert pa.verify_correspondence_maps(tmp_path / "broken.npz")["ok"] is False


def test_manager_builds_the_offline_assets_with_every_request_forbidden(tmp_path):
    def forbidden(request):
        raise AssertionError(f"network used: {request.url}")

    events: list[dict] = []
    manager = lm.UniversalModelDownloadManager(tmp_path, transport=httpx.MockTransport(forbidden), sink=events.append, sleep=lambda s: None)
    try:
        batch, outcome = run(manager, ["makehuman_cc0_head", "cc0_probe_bundle"])
    finally:
        manager.close()
    assert batch["queued"] == ["makehuman_cc0_head", "cc0_probe_bundle"]
    assert {k: v["status"] for k, v in outcome["entries"].items()} == {"makehuman_cc0_head": "ready", "cc0_probe_bundle": "ready"}
    states = {r["id"]: r["state"] for r in manager.manifest()["models"]}
    assert states["makehuman_cc0_head"] == "READY" and states["cc0_probe_bundle"] == "PROCEDURAL_FALLBACK"
    assert "compiling" in {e["status"] for e in events}


def test_a_damaged_compiled_asset_is_reported_corrupt_and_rebuilt_not_a_crash(tmp_path):
    entry = lm.load_manifest()["cc0_probe_bundle"]
    target = tmp_path / entry.install_path
    target.parent.mkdir(parents=True)
    target.write_bytes(b"PK\x03\x04 this is not an npz")
    manager = lm.UniversalModelDownloadManager(tmp_path, sink=None, sleep=lambda s: None)
    try:
        row = manager.inspect("cc0_probe_bundle")
        assert row["state"] == "CORRUPT" and row["detail"] and row["hash_status"] == "mismatch"
        report = manager.verify_and_repair(["cc0_probe_bundle"])
        outcome = manager.wait(report["repair"]["batch_id"], timeout=60)
    finally:
        manager.close()
    assert outcome["entries"]["cc0_probe_bundle"]["status"] == "ready"
    assert report["corrupt"] == ["cc0_probe_bundle"] and (target.parent / ".quarantine").is_dir()
    assert pa.verify_cc0_probe_bundle(target)["ok"]


def test_compiled_asset_verification_is_cached_until_the_file_changes_or_a_refresh_is_forced(tmp_path):
    calls: list[int] = []

    class Handler:
        def verify(self, target, expected):
            calls.append(1)
            return {"ok": True, "detail": "fine", "kind": "pinned"}

    raw = {"id": "c1", "name": "c1", "kind": "compiled", "category": "3d_asset", "install_root": "app", "install_path": "assets/c1.bin",
           "mirrors": [], "license": "CC0-1.0", "commercial_use": "allow", "requires_acceptance": False, "procedural_fallback": "h"}
    manager = lm.UniversalModelDownloadManager(tmp_path, manifest={"c1": lm.ManifestEntry.from_dict(raw)}, handlers={"h": Handler()}, sink=None)
    (tmp_path / "assets").mkdir()
    asset = tmp_path / "assets/c1.bin"
    asset.write_bytes(b"x")
    for _ in range(3):
        assert manager.inspect("c1")["state"] == "READY"
    assert len(calls) == 1                                   # a UI polling /models/manifest does not reload the asset
    manager.inspect("c1", force=True)
    assert len(calls) == 2                                   # ?refresh=true really re-verifies
    asset.write_bytes(b"xy")
    manager.inspect("c1")
    assert len(calls) == 3                                   # and a changed file is never served from the cache
    manager.close()


def test_correspondence_maps_wait_for_flame_instead_of_pretending(tmp_path):
    manager = lm.UniversalModelDownloadManager(tmp_path, sink=None, sleep=lambda s: None)
    try:
        row = next(r for r in manager.manifest()["models"] if r["id"] == "correspondence_maps")
        batch = manager.start_batch(["correspondence_maps"])
    finally:
        manager.close()
    assert row["state"] == "NOT_APPLICABLE" and "flame2023_open" in row["detail"]
    assert batch["skipped"][0]["reason"] == "DEPENDENCY_MISSING"


def test_verify_report_flags_corruption_and_optionally_missing_models(tmp_path):
    clean = lm.universal_manifest_report(tmp_path)
    assert clean["ok"] and clean["corrupt"] == [] and "makehuman_cc0_sources" not in clean["corrupt"]
    assert lm.universal_manifest_report(tmp_path, require_installed=True)["ok"] is False
    models = tmp_path / "models"
    models.mkdir()
    size = lm.load_manifest()["landmarks_2d106det"].size_bytes
    (models / "2d106det.onnx").write_bytes(b"\0" * size)                        # right size, wrong bytes
    assert lm.universal_manifest_report(tmp_path)["corrupt"] == ["landmarks_2d106det"]


# ------------------------------------------------------------------------------------------- REST & WebSocket
def client_for(tmp_path, web: FakeWeb, *raw: dict):
    app = create_app(Settings(), repo_root=tmp_path, worker_enabled=False)
    manager = lm.UniversalModelDownloadManager(
        tmp_path, manifest={r["id"]: lm.ManifestEntry.from_dict(r) for r in raw}, transport=web.transport(),
        sink=app.state.events.publish, handlers={}, sleep=lambda s: None)
    app.state.universal_models = manager
    return app, manager, TestClient(app, base_url="http://127.0.0.1:8030")


def test_manifest_endpoint_lists_models_with_licence_state_hash_and_disk(tmp_path):
    good = payload("e1")
    (tmp_path / "models").mkdir()
    (tmp_path / "models/e1.bin").write_bytes(good)
    app, _, client = client_for(tmp_path, FakeWeb(), entry_dict("e1", good, [f"{BASE}/e1"]),
                                entry_dict("e2", payload("e2"), [f"{BASE}/e2"], license="UNVERIFIED", commercial_use="deny", requires_acceptance=True))
    with client:
        body = client.get("/api/v1/models/manifest").json()
        assert client.get("/api/models/manifest").status_code == 200                  # the /api alias works too
    rows = {r["id"]: r for r in body["models"]}
    assert rows["e1"]["installed"] and rows["e1"]["hash_status"] == "verified" and rows["e1"]["license_badge"] == "open"
    assert rows["e2"]["state"] == "MISSING" and rows["e2"]["license_badge"] == "unverified" and rows["e2"]["requires_acceptance"]
    assert body["disk"]["installed_bytes"] == len(good) and body["summary"]["required_missing"] == ["e2"]


def test_download_endpoint_queues_missing_models_and_reports_what_needs_acknowledgement(tmp_path):
    free, locked = payload("d1"), payload("d2")
    web = FakeWeb({f"{BASE}/d1": Resource(free), f"{BASE}/d2": Resource(locked)})
    app, manager, client = client_for(tmp_path, web, entry_dict("d1", free, [f"{BASE}/d1"]),
                                      entry_dict("d2", locked, [f"{BASE}/d2"], license="UNVERIFIED", commercial_use="deny", requires_acceptance=True))
    with client:
        response = client.post("/api/v1/models/download", json={})
        assert response.status_code == 202, response.text
        body = response.json()
        assert body["status"] == "accepted" and body["batch"]["queued"] == ["d1"]
        assert body["batch"]["skipped"][0]["id"] == "d2" and body["batch"]["skipped"][0]["reason"] == "LICENSE_ACCEPTANCE_REQUIRED"
        manager.wait(body["batch"]["batch_id"], timeout=30)
        polled = client.get(f"/api/v1/models/batches/{body['batch']['batch_id']}").json()
        assert polled["entries"]["d1"]["status"] == "ready" and polled["state"] == "completed"
        again = client.post("/api/v1/models/download", json={"models": ["d2"], "accepted_license": True})
        assert again.status_code == 202 and again.json()["batch"]["queued"] == ["d2"]
        manager.wait(again.json()["batch"]["batch_id"], timeout=30)
        assert client.post("/api/v1/models/download", json={}).status_code == 200      # nothing left to do
        assert client.get("/api/v1/models/batches/nope").status_code == 404
    assert (tmp_path / "models/d1.bin").read_bytes() == free and (tmp_path / "models/d2.bin").read_bytes() == locked


def test_verify_and_repair_endpoint(tmp_path):
    data = payload("r1")
    (tmp_path / "models").mkdir()
    (tmp_path / "models/r1.bin").write_bytes(payload("wrong", len(data)))
    web = FakeWeb({f"{BASE}/r1": Resource(data)})
    app, manager, client = client_for(tmp_path, web, entry_dict("r1", data, [f"{BASE}/r1"]))
    with client:
        body = client.post("/api/v1/models/verify-and-repair", json={}).json()
        assert body["corrupt"] == ["r1"] and body["quarantined"] and body["repair"]["queued"] == ["r1"]
        manager.wait(body["repair"]["batch_id"], timeout=30)
        assert client.post("/api/v1/models/verify-and-repair", json={"repair": False}).json()["verified"] == 1
        assert client.post("/api/v1/models/verify-and-repair", json={"models": ["ghost"]}).status_code == 404
    assert (tmp_path / "models/r1.bin").read_bytes() == data


def test_download_progress_is_broadcast_on_the_pipeline_status_websocket(tmp_path):
    data = payload("w1", 400_000)
    web = FakeWeb({f"{BASE}/w1": Resource(data)})
    app, manager, client = client_for(tmp_path, web, entry_dict("w1", data, [f"{BASE}/w1"]))
    with client, client.websocket_connect("ws://127.0.0.1:8030/ws/v1/pipeline/status") as socket_:
        batch = client.post("/api/v1/models/download", json={"models": ["w1"]}).json()["batch"]
        frames = []
        while True:
            frame = socket_.receive_json()
            if frame["type"] != "model_download_progress":
                continue
            frames.append(frame)
            if frame["status"] in {"ready", "failed", "cancelled"}:
                break
        manager.wait(batch["batch_id"], timeout=30)
    assert frames[-1]["status"] == "ready" and frames[-1]["batch_id"] == batch["batch_id"]
    for field in ("model_id", "downloaded_bytes", "total_bytes", "speed_mbps", "eta_seconds", "mirror_used", "status"):
        assert all(field in f for f in frames), field
    assert frames[-1]["downloaded_bytes"] == frames[-1]["total_bytes"] == len(data)
    assert frames[-1]["mirror_used"] == f"{BASE}/w1" and frames[-1]["model_id"] == "w1"


def test_a_client_that_connects_mid_download_is_brought_up_to_date(tmp_path):
    data = payload("w2", 500_000)
    gate, running = threading.Event(), threading.Event()

    def hold(index: int) -> None:
        running.set()
        gate.wait(10)

    web = FakeWeb({f"{BASE}/w2": Resource(data, on_chunk=hold)})
    app, manager, client = client_for(tmp_path, web, entry_dict("w2", data, [f"{BASE}/w2"]))
    with client:
        batch = client.post("/api/v1/models/download", json={"models": ["w2"]}).json()["batch"]
        assert running.wait(10)
        with client.websocket_connect("ws://127.0.0.1:8030/ws/v1/pipeline/status") as socket_:
            seen = []
            while not any(f["type"] == "model_download_progress" for f in seen):
                seen.append(socket_.receive_json())
            gate.set()
        manager.wait(batch["batch_id"], timeout=30)
    live = next(f for f in seen if f["type"] == "model_download_progress")
    assert live["model_id"] == "w2" and live["status"] in {"queued", "downloading"}


def test_endpoints_report_an_unavailable_manifest_instead_of_failing_the_app(tmp_path):
    app = create_app(Settings(), repo_root=tmp_path, worker_enabled=False)
    app.state.universal_models = None
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        assert client.get("/api/v1/models/manifest").json()["error"]["code"] == "MODEL_MANIFEST_UNAVAILABLE"
        assert client.post("/api/v1/models/download", json={}).status_code == 503
