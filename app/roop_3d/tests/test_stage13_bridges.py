from __future__ import annotations

import asyncio
import json
import logging
import struct
from dataclasses import replace
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

from roop_3d.bridges.comfyui_bridge import BridgeConfig, ComfyUIBridge, LoopbackConnect, OutputRef, decode_image
from roop_3d.bridges.cloud_stub import CloudStub
from roop_3d.bridges.provider_protocol import (ConsentAuthority, ConsentGatedProvider, MediaDescriptor,
    ProviderCapabilities, ProviderStatus, RemoteJob, SecretRedactionFilter, redact_secrets)
from roop_3d.core.errors import DomainError


WORKFLOW = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "local.safetensors"}}}
NODES = {"CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["local.safetensors"]]}}}}
PNG = cv2.imencode(".png", np.full((16, 16, 3), 128, np.uint8))[1].tobytes()


class Server:
    def __init__(self):
        self.calls = []
        self.prompt_id = None
        self.complete = False
        self.content_type = "image/png"
        self.image = PNG
        self.nodes = NODES
        self.ambiguous = False
        self.version = "0.43.0"

    def request(self, request):
        path = request.url.path
        payload = json.loads(request.content) if request.content else None
        self.calls.append((request.method, path, payload))
        if path == "/system_stats":
            return httpx.Response(200, json={"system": {"comfyui_version": self.version}})
        if path == "/object_info":
            return httpx.Response(200, json=self.nodes)
        if path == "/prompt":
            self.prompt_id = payload["prompt_id"]
            if self.ambiguous:
                raise httpx.ReadTimeout("secret server detail", request=request)
            return httpx.Response(200, json={"prompt_id": self.prompt_id, "node_errors": {}})
        if path.startswith("/history/"):
            return httpx.Response(200, json={self.prompt_id: {"status": {"completed": True, "status_str": "success"},
                "outputs": {"1": {"images": [{"filename": "face.png", "subfolder": "", "type": "output"}]}}}} if self.complete else {})
        if path == "/view":
            return httpx.Response(200, content=self.image, headers={"content-type": self.content_type})
        if path in {"/interrupt", "/queue"}:
            return httpx.Response(200)
        raise AssertionError(path)


class Socket:
    def __init__(self, server, messages):
        self.server, self.messages = server, list(messages)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    async def recv(self):
        if not self.messages:
            await asyncio.sleep(60)
        item = self.messages.pop(0)
        if isinstance(item, Exception):
            raise item
        if item == "done":
            self.server.complete = True
            return json.dumps({"type": "executing", "data": {"prompt_id": self.server.prompt_id, "node": None}})
        if isinstance(item, bytes):
            return item
        kind, prompt, node = item
        return json.dumps({"type": kind, "data": {"prompt_id": self.server.prompt_id if prompt == "self" else prompt,
            "node": node, "value": 1, "max": 2}})


def bridge(tmp_path, server, ws=None, **limits):
    return ComfyUIBridge(BridgeConfig(poll_interval=.01, execution_timeout=.15, backoff_initial=.001, **limits),
        root=tmp_path, http_transport=httpx.MockTransport(server.request), ws_connect=ws or (lambda *a, **k: Socket(server, [])))


def test_ready_idempotence_and_conflict(tmp_path):
    async def run():
        server = Server()
        async with bridge(tmp_path, server) as local:
            handle = await local.submit("job", WORKFLOW)
            assert await local.submit("job", WORKFLOW) == handle
            assert len([call for call in server.calls if call[1] == "/prompt"]) == 1
            changed = {"2": WORKFLOW["1"]}
            with pytest.raises(DomainError, match="another workflow"):
                await local.submit("job", changed)
        # Correlation survives reopening the bridge and never re-POSTs.
        async with bridge(tmp_path, server) as local:
            assert await local.submit("job", WORKFLOW) == handle
        assert len([call for call in server.calls if call[1] == "/prompt"]) == 1
    asyncio.run(run())


@pytest.mark.parametrize("workflow", [{"1": {"class_type": "Missing", "inputs": {}}},
    {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "absent.safetensors"}}},
    {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {}}}])
def test_missing_node_checkpoint_or_input_rejected_before_post(tmp_path, workflow):
    async def run():
        server = Server()
        async with bridge(tmp_path, server) as local:
            with pytest.raises(DomainError):
                await local.submit("job", workflow)
        assert not any(call[1] == "/prompt" for call in server.calls)
    asyncio.run(run())


def test_progress_preview_correlation_and_reconnect(tmp_path):
    async def run():
        server = Server()
        connections = []
        def ws(url, **kwargs):
            connections.append(url)
            if len(connections) == 1:
                return Socket(server, [("executing", "other", "9"), struct.pack(">II", 1, 2) + PNG, OSError("disconnect")])
            return Socket(server, [("executing", "self", "1"), ("progress", "self", "1"),
                struct.pack(">II", 1, 2) + PNG, "done"])
        events = []
        async with bridge(tmp_path, server, ws) as local:
            handle = await local.submit("job", WORKFLOW)
            history = await local.wait(handle, on_event=events.append)
            assert history["status"]["completed"]
            image = await local.fetch_output(handle, OutputRef("face.png"))
            assert image.shape == (16, 16, 3)
            with pytest.raises(DomainError):
                await local.fetch_output(handle, OutputRef("unrelated.png"))
        assert len(connections) == 2 and connections[0] == connections[1]
        assert all(event.prompt_id == handle.prompt_id for event in events)
        assert len([event for event in events if event.kind == "preview"]) == 1
        assert any(event.kind == "progress" for event in events)
        assert len([call for call in server.calls if call[1] == "/prompt"]) == 1
    asyncio.run(run())


def test_ambiguous_post_preserves_receipt_without_retry(tmp_path):
    async def run():
        server = Server()
        server.ambiguous = True
        async with bridge(tmp_path, server) as local:
            with pytest.raises(DomainError) as error:
                await local.submit("job", WORKFLOW)
            assert "secret server detail" not in str(error.value)
            handle = await local.submit("job", WORKFLOW)
            assert handle.prompt_id == server.prompt_id
            server.complete = True
            assert await local.poll(handle)
        assert len([call for call in server.calls if call[1] == "/prompt"]) == 1
    asyncio.run(run())


def test_timeout_and_cancellation_are_bounded_and_targeted(tmp_path):
    async def run():
        server = Server()
        async with bridge(tmp_path, server) as local:
            handle = await local.submit("job", WORKFLOW)
            with pytest.raises(DomainError) as error:
                await local.wait(handle)
            assert error.value.code == "COMFY_TIMEOUT"
            cancel = asyncio.Event()
            cancel.set()
            with pytest.raises(DomainError) as error:
                await local.wait(handle, cancel=cancel)
            assert error.value.code == "COMFY_CANCELLED"
            assert ("POST", "/queue", {"delete": [handle.prompt_id]}) in server.calls
            assert ("POST", "/interrupt", {"prompt_id": handle.prompt_id}) in server.calls
    asyncio.run(run())


@pytest.mark.parametrize("endpoint", ["http://example.com:8188", "http://192.168.1.1:8188", "file:///x", "http://user:secret@127.0.0.1:8188", "http://127.0.0.1:8188/?token=x"])
def test_nonlocal_or_credential_endpoint_rejected(endpoint):
    with pytest.raises(ValueError):
        BridgeConfig(endpoint=endpoint)


@pytest.mark.parametrize("filename,folder", [("../x.png", ""), ("C:evil.png", ""), ("x.png", "../x"), ("x.png", "a\\b"), ("NUL.png", "")])
def test_unsafe_output_names_rejected(filename, folder):
    with pytest.raises(ValueError):
        OutputRef(filename, folder)


def test_fetch_rejects_type_size_and_pixel_bombs(tmp_path):
    async def run():
        server = Server()
        async with bridge(tmp_path, server) as local:
            handle = await local.submit("job", WORKFLOW)
            server.complete = True
            server.content_type = "text/html"
            with pytest.raises(DomainError):
                await local.fetch_output(handle, OutputRef("face.png"))
            server.content_type = "image/png"
            server.image = PNG * 2
            local.config = replace(local.config, max_image_bytes=len(PNG))
            with pytest.raises(DomainError):
                await local.fetch_output(handle, OutputRef("face.png"))
        bomb = PNG[:16] + struct.pack(">II", 100000, 100000) + PNG[24:]
        with pytest.raises(DomainError):
            decode_image(bomb, "image/png", BridgeConfig())
    asyncio.run(run())


def remote_job():
    return RemoteJob(job_id="job", provider_id="fixture", destination_host="provider.invalid",
                     files=(MediaDescriptor(filename="face.png", size_bytes=len(PNG), sha256="a" * 64),))


class Adapter:
    def __init__(self):
        self.calls = []
    def capabilities(self):
        return ProviderCapabilities("VERIFIED_REMOTE", "fixture", "provider.invalid")
    def validate_consent(self, job):
        pass
    async def submit(self, job):
        self.calls.append(job.job_id)
        return ProviderStatus(job.job_id, "queued")
    poll = submit
    cancel = submit
    async def fetch_output(self, job):
        self.calls.append(job.job_id)
        return b"output"


def test_cloud_disabled_and_every_remote_io_requires_exact_signed_consent():
    async def run():
        now = [1000.0]
        authority = ConsentAuthority(b"a" * 32, clock=lambda: now[0])
        adapter = Adapter()
        gated = ConsentGatedProvider(adapter, authority)
        job = remote_job()
        token = authority.issue(job, acknowledged=True)
        signed = job.model_copy(update={"consent_token": token})
        for method in (gated.submit, gated.poll, gated.cancel, gated.fetch_output):
            with pytest.raises(DomainError):
                await method(signed)
        gated.remote_processing_enabled = True
        with pytest.raises(DomainError):
            await gated.submit(job)
        for update in ({"destination_host": "other.invalid"}, {"job_id": "other"},
                       {"files": (job.files[0].model_copy(update={"size_bytes": 1}),)},
                       {"files": (job.files[0].model_copy(update={"filename": "other.png"}),)},
                       {"consent_token": token[:-1] + "!"}):
            with pytest.raises(DomainError):
                await gated.submit(signed.model_copy(update=update))
        assert adapter.calls == []
        assert (await gated.submit(signed)).state == "queued"
        now[0] += 301
        with pytest.raises(DomainError):
            await gated.submit(signed)
        disabled = ConsentGatedProvider(CloudStub(), authority, remote_processing_enabled=True)
        assert disabled.capabilities().state == "NOT_CONFIGURED"
        fresh = job.model_copy(update={"consent_token": authority.issue(job, acknowledged=True)})
        with pytest.raises(DomainError) as error:
            await disabled.submit(fresh)
        assert error.value.code == "NOT_CONFIGURED"
        assert token not in repr(signed)
    asyncio.run(run())


def test_secret_redaction_including_logging_exceptions():
    value = "Authorization: Bearer verysecret url=https://host.invalid/file?sig=secret token=hidden"
    redacted = redact_secrets(value)
    assert all(secret not in redacted for secret in ("verysecret", "sig=secret", "hidden"))
    record = logging.LogRecord("bridge", logging.ERROR, "", 1, "%s", (value,), (ValueError, ValueError(value), None))
    assert SecretRedactionFilter().filter(record)
    assert record.exc_info is None and "verysecret" not in record.getMessage()
    assert "jsonsecret" not in redact_secrets('{"token": "jsonsecret"}')


def test_real_stage09_compositor_handoff(tmp_path):
    from roop_3d.compositor.pipeline import CompositeOptions, FrameCompositor
    class Parser:
        def predict(self, image):
            return None
    async def run():
        server = Server()
        async with bridge(tmp_path, server) as local:
            handle = await local.submit("job", WORKFLOW)
            server.complete = True
            frame = np.full((32, 32, 3), 32, np.uint8)
            inverse = np.array([[1, 0, 8], [0, 1, 8]], np.float32)
            result = await local.composite_output(handle, OutputRef("face.png"),
                compositor=FrameCompositor(parser=Parser()), frame=frame, inverse=inverse, track_id=0,
                landmarks=np.array([[10, 10], [21, 10], [21, 21], [10, 21]], np.float32),
                options=CompositeOptions(color_strength=0))
            assert result.shape == frame.shape and result.dtype == np.uint8
            assert np.any(result != frame)
            assert np.array_equal(result[0], frame[0])
    asyncio.run(run())


def test_reconnect_exhaustion_keeps_prompt_receipt(tmp_path):
    async def run():
        server = Server()
        count = []
        def fail(url, **kwargs):
            count.append(url)
            return Socket(server, [OSError("lost")])
        async with bridge(tmp_path, server, fail, reconnect_attempts=1) as local:
            handle = await local.submit("job", WORKFLOW)
            with pytest.raises(DomainError) as error:
                await local.wait(handle)
            assert error.value.code == "COMFY_DISCONNECTED"
            assert len(count) == 2
            assert local._load(handle)["state"] == "submitted"
    asyncio.run(run())


@pytest.mark.parametrize("workflow", [[], {"1": {"class_type": [], "inputs": {}}},
    {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "local.safetensors", "unknown": 1}}}])
def test_malformed_workflow_fails_with_safe_error(tmp_path, workflow):
    async def run():
        server = Server()
        async with bridge(tmp_path, server) as local:
            with pytest.raises(DomainError):
                await local.submit("job", workflow)
        assert not any(call[1] == "/prompt" for call in server.calls)
    asyncio.run(run())


def test_request_redirect_and_json_limits(tmp_path):
    async def run():
        def redirect(request):
            return httpx.Response(302, headers={"location": "https://remote.invalid/secret"})
        async with ComfyUIBridge(root=tmp_path, http_transport=httpx.MockTransport(redirect)) as local:
            with pytest.raises(DomainError):
                await local.probe()
        def oversized(request):
            return httpx.Response(200, json={"system": {"long": "x" * 1000}})
        async with ComfyUIBridge(BridgeConfig(max_json_bytes=100), root=tmp_path,
                http_transport=httpx.MockTransport(oversized)) as local:
            with pytest.raises(DomainError) as error:
                await local.probe()
            assert error.value.code == "COMFY_RESPONSE_SIZE"
    asyncio.run(run())


def test_real_websocket_handshake_cannot_follow_remote_redirect():
    from websockets.asyncio.server import serve
    from websockets.datastructures import Headers
    from websockets.http11 import Response
    from websockets.exceptions import InvalidStatus
    async def run():
        calls = []
        def redirect(connection, request):
            calls.append(request.path)
            return Response(302, "Found", Headers({"Location": "ws://remote.invalid/ws"}))
        async def handler(socket):
            raise AssertionError("Redirected handshake must not upgrade")
        async with serve(handler, "127.0.0.1", 0, process_request=redirect) as server:
            port = server.sockets[0].getsockname()[1]
            with pytest.raises(InvalidStatus):
                async with LoopbackConnect(f"ws://127.0.0.1:{port}/ws", proxy=None):
                    raise AssertionError("Unexpected upgrade")
        assert calls == ["/ws"]
    asyncio.run(run())


def test_total_wait_deadline_includes_slow_http(tmp_path):
    async def run():
        server = Server()
        async with bridge(tmp_path, server) as local:
            handle = await local.submit("job", WORKFLOW)
        async def slow(request):
            await asyncio.sleep(1)
            return httpx.Response(200, json={})
        async with ComfyUIBridge(BridgeConfig(execution_timeout=.02), root=tmp_path,
                http_transport=httpx.MockTransport(slow)) as local:
            with pytest.raises(DomainError) as error:
                await asyncio.wait_for(local.wait(handle), .25)
            assert error.value.code == "COMFY_TIMEOUT"
    asyncio.run(run())


def test_corrupt_receipt_is_preserved_and_rejected(tmp_path):
    async def run():
        server = Server()
        async with bridge(tmp_path, server) as local:
            handle = await local.submit("job", WORKFLOW)
            path = tmp_path / "roop_3d/.runtime/bridges/comfyui/job.json"
            path.write_text("[]", encoding="utf-8")
            with pytest.raises(DomainError) as error:
                await local.poll(handle)
            assert error.value.code == "COMFY_RECEIPT_INVALID"
            assert path.read_text("utf-8") == "[]"
    asyncio.run(run())


def test_unreviewed_protocol_is_rejected_before_submission_or_interrupt(tmp_path):
    async def run():
        server = Server()
        async with bridge(tmp_path, server) as local:
            handle = await local.submit("job", WORKFLOW)
            server.version = "0.3.0"
            with pytest.raises(DomainError) as error:
                await local.submit("other", WORKFLOW)
            assert error.value.code == "COMFY_PROTOCOL_UNSUPPORTED"
            with pytest.raises(DomainError):
                await local.cancel(handle)
        assert len([call for call in server.calls if call[1] == "/prompt"]) == 1
        assert not any(call[1] in {"/interrupt", "/queue"} for call in server.calls)
    asyncio.run(run())
