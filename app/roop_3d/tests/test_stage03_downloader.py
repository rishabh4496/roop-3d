from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import httpx
import pytest

from roop_3d.audit.isolation import AuditError
from roop_3d.core.errors import DomainError
from roop_3d.core.storage import atomic_json
from roop_3d.models.downloader import Downloader, validate_url
from roop_3d.tests.test_stage03_registry import artifact, reviewed_artifact

DATA = b"reviewed fixture bytes" * 8192
URL = "https://github.com/fixture/model.onnx"


def remote(data=DATA, **changes):
    values = {"status": "verified_remote", "upstream_url": URL, "provenance_state": "VERIFIED_REMOTE",
              "license_id": "LicenseRef-TestFixture", "license_url": "https://example.org/license",
              "commercial_use": "allow", "redistribution_permission": "allow",
              "attribution_requirements": "Retain fixture notice", "original_repository": "https://github.com/fixture",
              "source_revision": "a" * 40, "architecture": "FixtureNet", "required_base_model": "NONE",
              "hardware_requirements": {"state": "VERIFIED"}, "precision_requirements": ["FP32"],
              "dependency_compatibility": {"state": "COMPATIBLE", "requirements": []},
              "security_status": "VERIFIED", "serialization_format": "ONNX", **changes}
    return reviewed_artifact(data, **values)


def prepare_partial(root, record, data=DATA[:65536], identity=None):
    downloader = Downloader(root)
    _, part, meta = downloader.paths(record)
    path = root / part
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    atomic_json(root, meta, {"url": URL, "sha256": record.expected_sha256, "size": len(DATA),
                            "validator": identity or {"etag": '"v1"'}})
    return downloader, path


class Interrupted(httpx.SyncByteStream):
    def __iter__(self):
        yield DATA[:65536]
        raise httpx.ReadError("Fixture connection interrupted")


def test_atomic_download_digest_and_reuse(tmp_path):
    calls = []
    states = []
    record = remote()
    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=DATA, headers={"ETag": '"v1"', "Content-Length": str(len(DATA))})
    downloader = Downloader(tmp_path, transport=httpx.MockTransport(handler))
    result = downloader.download(record, state_callback=states.append)
    assert result.downloaded and not result.resumed
    assert result.path.read_bytes() == DATA
    assert hashlib.sha256(result.path.read_bytes()).hexdigest() == record.expected_sha256
    assert not downloader.download(record).downloaded
    assert len(calls) == 1
    assert not list(result.path.parent.glob("*.part"))
    assert not list(result.path.parent.glob("*.part.json"))
    assert states == ["DOWNLOADING", "VERIFYING", "READY"]


def test_interruption_is_fsynced_and_next_call_resumes(tmp_path):
    calls = []
    record = remote()
    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(200, stream=Interrupted(), headers={"ETag": '"v1"'})
        assert request.headers["Range"] == "bytes=65536-"
        assert request.headers["If-Range"] == '"v1"'
        return httpx.Response(206, content=DATA[65536:], headers={"ETag": '"v1"',
            "Content-Range": f"bytes 65536-{len(DATA)-1}/{len(DATA)}"})
    downloader = Downloader(tmp_path, transport=httpx.MockTransport(handler))
    with pytest.raises(DomainError) as failure:
        downloader.download(record)
    assert failure.value.code == "DOWNLOAD_INTERRUPTED"
    assert (tmp_path / downloader.paths(record)[1]).read_bytes() == DATA[:65536]
    result = downloader.download(record)
    assert result.resumed and result.path.read_bytes() == DATA


@pytest.mark.parametrize("identity", [{"etag": '"v1"'}, {"last-modified": "Wed, 21 Oct 2015 07:28:00 GMT"}])
def test_range_resume_with_matching_identity(tmp_path, identity):
    record = remote()
    downloader, path = prepare_partial(tmp_path, record, identity=identity)
    def handler(request):
        assert request.headers["range"] == f"bytes={path.stat().st_size}-"
        assert request.headers["if-range"] == next(iter(identity.values()))
        return httpx.Response(206, content=DATA[65536:], headers={**identity,
            "Content-Range": f"bytes 65536-{len(DATA)-1}/{len(DATA)}"})
    downloader.transport = httpx.MockTransport(handler)
    result = downloader.download(record)
    assert result.resumed and result.path.read_bytes() == DATA


@pytest.mark.parametrize("status", [200, 206, 416])
def test_identity_changes_or_ignored_range_restart_cleanly(tmp_path, status):
    record = remote()
    downloader, _ = prepare_partial(tmp_path, record)
    calls = []
    def handler(request):
        calls.append(request)
        if len(calls) == 1 and status != 200:
            return httpx.Response(status, content=b"ignored", headers={"ETag": '"v2"'})
        if len(calls) > 1:
            assert "range" not in request.headers
        return httpx.Response(200, content=DATA, headers={"ETag": '"v2"'})
    downloader.transport = httpx.MockTransport(handler)
    result = downloader.download(record)
    assert not result.resumed and result.path.read_bytes() == DATA
    assert len(calls) == (1 if status == 200 else 2)


@pytest.mark.parametrize("metadata", [{"validator": {"etag": 'W/"v1"'}},
                                      {"validator": {}}, {"url": URL + ".changed"},
                                      {"sha256": "0" * 64}, {"validator": "bad"}])
def test_untrusted_or_stale_resume_metadata_restarts(tmp_path, metadata):
    record = remote()
    downloader, _ = prepare_partial(tmp_path, record)
    meta = tmp_path / downloader.paths(record)[2]
    state = {**json.loads(meta.read_text()), **metadata}
    meta.write_text(json.dumps(state))
    def handler(request):
        assert "range" not in request.headers
        return httpx.Response(200, content=DATA)
    downloader.transport = httpx.MockTransport(handler)
    assert not downloader.download(record).resumed


def test_corrupt_download_quarantined_without_final_publication(tmp_path):
    record = remote()
    downloader = Downloader(tmp_path, transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=b"X" * len(DATA))))
    with pytest.raises(DomainError) as failure:
        downloader.download(record)
    assert failure.value.code == "DOWNLOAD_DIGEST_MISMATCH"
    final, part, meta = downloader.paths(record)
    assert not (tmp_path / final).exists() and not (tmp_path / part).exists()
    assert not (tmp_path / meta).exists()
    assert len(list((tmp_path / part).parent.glob("*.corrupt"))) == 1


@pytest.mark.parametrize("headers,body,code", [
    ({"Content-Length": str(len(DATA) + 1)}, DATA, "DOWNLOAD_SIZE_MISMATCH"),
    ({}, DATA + b"overflow", "DOWNLOAD_SIZE_LIMIT"),
    ({"Content-Encoding": "gzip"}, b"invalid gzip", "DOWNLOAD_ENCODING_BLOCKED"),
    ({}, DATA[:20], "DOWNLOAD_INCOMPLETE")], ids=["length", "overflow", "encoding", "incomplete"])
def test_size_and_encoding_enforced(tmp_path, headers, body, code):
    # Stream fixture keeps HTTPX from pre-decoding the deliberately invalid encoding.
    downloader = Downloader(tmp_path, transport=httpx.MockTransport(
        lambda request: httpx.Response(200, stream=httpx.ByteStream(body), headers=headers)))
    with pytest.raises(DomainError) as failure:
        downloader.download(remote())
    assert failure.value.code == code
    assert not (tmp_path / downloader.paths(remote())[0]).exists()


@pytest.mark.parametrize("content_range", ["bytes 0-10/11", "bytes 65535-180223/180224", "garbage"])
def test_wrong_ranges_rejected(tmp_path, content_range):
    record = remote()
    downloader, path = prepare_partial(tmp_path, record)
    downloader.transport = httpx.MockTransport(lambda request: httpx.Response(206, content=DATA[65536:],
        headers={"ETag": '"v1"', "Content-Range": content_range}))
    with pytest.raises(DomainError) as failure:
        downloader.download(record)
    assert failure.value.code == "DOWNLOAD_RANGE_INVALID"
    assert path.read_bytes() == DATA[:65536]


@pytest.mark.parametrize("url", ["http://github.com/test", "https://evil.example/test",
    "https://github.com.evil.example/test", "https://github.com@evil.example/test",
    "https://user:secret@github.com/test", "https://github.com:444/test",
    "https://github.com/a/../test", "https://github.com/a/%252e%252e/test",
    "https://github.com/a%5ctest", "https://github.com/test?token=secret",
    "https://github.com/test#fragment", "https://127.0.0.1/test"])
def test_url_policy_blocks_insecure_or_unreviewed_targets(url):
    with pytest.raises(DomainError) as failure:
        validate_url(url)
    assert failure.value.code == "DOWNLOAD_URL_BLOCKED"


def test_redirects_are_never_followed_and_policy_fails_before_network(tmp_path):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(302, headers={"Location": "https://raw.githubusercontent.com/test"})
    downloader = Downloader(tmp_path, transport=httpx.MockTransport(handler))
    with pytest.raises(DomainError) as failure:
        downloader.download(remote())
    assert failure.value.code == "DOWNLOAD_REDIRECT_BLOCKED" and len(calls) == 1
    for change in ({"status": "blocked"}, {"license_id": "UNKNOWN"},
                   {"manual_acceptance_required": True}, {"provenance_state": "UNKNOWN"}):
        record = remote().model_copy(update=change)
        with pytest.raises(DomainError):
            downloader.download(record)
    assert len(calls) == 1


def test_only_observed_huggingface_storage_redirect_is_followed(tmp_path):
    calls = []
    record = remote(upstream_url="https://huggingface.co/org/repo/resolve/rev/fixture.onnx")
    def handler(request):
        calls.append(request.url.host)
        if request.url.host == "huggingface.co":
            return httpx.Response(302, headers={"Location": "https://us.aws.cdn.hf.co/path/object?signature=fixture"})
        assert request.url.host == "us.aws.cdn.hf.co"
        return httpx.Response(200, content=DATA, headers={"ETag": '"hf-v1"'})
    result = Downloader(tmp_path, transport=httpx.MockTransport(handler)).download(record)
    assert result.path.read_bytes() == DATA
    assert calls == ["huggingface.co", "us.aws.cdn.hf.co"]


def test_huggingface_redirect_to_unreviewed_host_is_rejected(tmp_path):
    record = remote(upstream_url="https://huggingface.co/org/repo/resolve/rev/fixture.onnx")
    downloader = Downloader(tmp_path, transport=httpx.MockTransport(
        lambda request: httpx.Response(302, headers={"Location": "https://evil.example/model"})))
    with pytest.raises(DomainError) as failure:
        downloader.download(record)
    assert failure.value.code == "DOWNLOAD_REDIRECT_BLOCKED"


def test_download_cancellation_keeps_only_resumable_partial_bytes(tmp_path):
    record = remote()
    cancel = threading.Event()
    class TwoChunks(httpx.SyncByteStream):
        def __iter__(self):
            yield DATA[:65536]
            yield DATA[65536:]
    def first(request):
        return httpx.Response(200, stream=TwoChunks(), headers={"ETag": '"cancel-v1"'})
    downloader = Downloader(tmp_path, transport=httpx.MockTransport(first))
    with pytest.raises(DomainError) as failure:
        downloader.download(record, cancel_event=cancel,
                            progress_callback=lambda downloaded, total: cancel.set() if downloaded else None)
    assert failure.value.code == "DOWNLOAD_CANCELLED"
    final, part, metadata = downloader.paths(record)
    assert not (tmp_path / final).exists()
    assert (tmp_path / part).read_bytes() == DATA[:65536]
    assert (tmp_path / metadata).is_file()

    def resume(request):
        assert request.headers["range"] == "bytes=65536-"
        return httpx.Response(206, content=DATA[65536:], headers={
            "ETag": '"cancel-v1"', "Content-Range": f"bytes 65536-{len(DATA)-1}/{len(DATA)}"})
    downloader.transport = httpx.MockTransport(resume)
    result = downloader.download(record)
    assert result.resumed and result.path.read_bytes() == DATA


def test_completed_part_crash_recovery_and_max_artifact_limit(tmp_path):
    record = remote()
    downloader, _ = prepare_partial(tmp_path, record, DATA)
    result = downloader.download(record)
    assert result.path.read_bytes() == DATA
    with pytest.raises(DomainError) as failure:
        Downloader(tmp_path, max_bytes=1).download(record)
    assert failure.value.code == "DOWNLOAD_SIZE_LIMIT"


def test_hardlinked_partial_is_rejected_without_modifying_other_file(tmp_path):
    record = remote()
    downloader = Downloader(tmp_path)
    _, partial, _ = downloader.paths(record)
    path = tmp_path / partial
    path.parent.mkdir(parents=True)
    original = tmp_path / "original.onnx"
    original.write_bytes(b"immutable fixture")
    os.link(original, path)
    with pytest.raises(DomainError) as failure:
        downloader.download(record)
    assert failure.value.code == "CACHE_FILE_UNSAFE"
    assert original.read_bytes() == b"immutable fixture"


def test_two_threads_serialize_one_download(tmp_path):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=DATA)
    downloader = Downloader(tmp_path, transport=httpx.MockTransport(handler))
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: downloader.download(remote()), range(2)))
    assert sum(r.downloaded for r in results) == 1 and len(calls) == 1


def test_separate_processes_serialize_real_cache_lock(tmp_path):
    script = '''
import json, sys, time
from pathlib import Path
import httpx
from roop_3d.models.downloader import Downloader
from roop_3d.tests.test_stage03_downloader import remote, DATA
root=Path(sys.argv[1])
def handler(request):
    with (root/'requests.txt').open('ab') as log: log.write(b'GET\\n')
    time.sleep(0.15)
    return httpx.Response(200, content=DATA)
result=Downloader(root,transport=httpx.MockTransport(handler)).download(remote())
print(json.dumps({'downloaded':result.downloaded}))
'''
    processes = [subprocess.Popen([sys.executable, "-B", "-c", script, str(tmp_path)],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
    outputs = [process.communicate(timeout=20) for process in processes]
    assert all(process.returncode == 0 for process in processes), outputs
    assert sum(json.loads(out)["downloaded"] for out, _ in outputs) == 1
    assert (tmp_path / "requests.txt").read_bytes() == b"GET\n"


def test_model_archives_are_rejected_before_network_and_never_extracted(tmp_path):
    import io
    import zipfile
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../../should-never-be-created.py", "raise RuntimeError('never execute')")
    payload = buffer.getvalue()
    record = remote(payload, filenames=["fixture.zip"], serialization_format="ZIP")
    calls = []
    downloader = Downloader(tmp_path, transport=httpx.MockTransport(
        lambda request: (calls.append(request), httpx.Response(200, content=payload))[1]))
    with pytest.raises(DomainError) as failure:
        downloader.download(record)
    assert failure.value.code == "DOWNLOAD_FORMAT_BLOCKED"
    assert not calls
    assert not list(tmp_path.rglob("*.py"))


@pytest.mark.parametrize("operation", ["download", "resolve"])
def test_real_junction_or_symlink_rejected_before_cache_access(tmp_path, operation):
    from roop_3d.models.local_provider import LocalWeightProvider
    from roop_3d.tests.test_stage03_registry import registry_for
    target = tmp_path / "roop_3d/.runtime/link-target"
    target.mkdir(parents=True)
    link = tmp_path / "roop_3d/models/cache"
    link.parent.mkdir(parents=True)
    if os.name == "nt":
        env = dict(os.environ, ROOP3D_TEST_LINK=str(link), ROOP3D_TEST_TARGET=str(target))
        process = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
            "New-Item -ItemType Junction -Path $env:ROOP3D_TEST_LINK -Target $env:ROOP3D_TEST_TARGET | Out-Null"],
            env=env, capture_output=True, text=True, timeout=10)
        assert process.returncode == 0, process.stderr
    else:
        os.symlink(target, link, target_is_directory=True)
    try:
        with pytest.raises(AuditError) as failure:
            if operation == "download":
                Downloader(tmp_path).download(remote())
            else:
                LocalWeightProvider(registry_for(remote()), root=tmp_path).resolve("fixture")
        assert failure.value.code == "LINK_TRAVERSAL"
        assert list(target.iterdir()) == []
    finally:
        if os.name == "nt":
            os.rmdir(link)
        else:
            link.unlink()
