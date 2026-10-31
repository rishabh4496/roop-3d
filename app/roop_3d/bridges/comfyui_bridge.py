"""Bounded loopback-only ComfyUI API bridge; no model imports or media writes.

Protocol evidence: Comfy-Org/ComfyUI server.py and
script_examples/websockets_api_example.py (reviewed 2026-10-31).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import struct
import time
from typing import Callable
from urllib.parse import urlsplit, urlunsplit
import uuid

import cv2
import httpx
import numpy as np
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake

from roop_3d.audit.project_paths import application_root
from roop_3d.bridges.provider_protocol import safe_filename
from roop_3d.core.errors import DomainError
from roop_3d.core.storage import atomic_json, exclusive, owned


class LoopbackConnect(connect):
    """Pinned websockets 15 hook: no handshake redirect can leave the origin."""
    def process_redirect(self, exc: Exception) -> Exception | str:
        return exc


@dataclass(frozen=True)
class BridgeConfig:
    endpoint: str = "http://127.0.0.1:8188"
    request_timeout: float = 10.0
    execution_timeout: float = 300.0
    poll_interval: float = 0.5
    reconnect_attempts: int = 5
    backoff_initial: float = 0.1
    max_json_bytes: int = 8 * 1024 * 1024
    max_image_bytes: int = 16 * 1024 * 1024
    max_pixels: int = 16_777_216

    def __post_init__(self):
        try:
            parsed = urlsplit(self.endpoint)
            host = parsed.hostname
            if host == "localhost":
                host = "127.0.0.1"
            if (parsed.scheme not in {"http", "https"} or not host or not ipaddress.ip_address(host).is_loopback
                    or parsed.username or parsed.password or parsed.path not in {"", "/"}
                    or parsed.query or parsed.fragment or not 1 <= (parsed.port if parsed.port is not None else 80) <= 65535):
                raise ValueError()
            netloc = f"[{host}]" if ":" in host else host
            if parsed.port:
                netloc += f":{parsed.port}"
            object.__setattr__(self, "endpoint", urlunsplit((parsed.scheme, netloc, "", "", "")))
        except ValueError:
            raise ValueError("ComfyUI endpoint must be a credential-free loopback HTTP origin") from None
        if (not all(0 < value <= 3600 for value in (self.request_timeout, self.execution_timeout, self.poll_interval))
                or not 0 <= self.reconnect_attempts <= 20 or not 0 < self.backoff_initial <= 10
                or not 1 <= self.max_json_bytes <= 32 * 1024 * 1024
                or not 1 <= self.max_image_bytes <= 64 * 1024 * 1024 or not 1 <= self.max_pixels <= 16_777_216):
            raise ValueError("Bridge resource limits are invalid")


@dataclass(frozen=True)
class PromptHandle:
    job_id: str
    prompt_id: str
    client_id: str
    workflow_sha256: str

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", self.job_id) or not re.fullmatch(r"[a-f0-9]{64}", self.workflow_sha256):
            raise ValueError("Invalid prompt correlation identity")
        for value in (self.prompt_id, self.client_id):
            if str(uuid.UUID(value)) != value:
                raise ValueError("Prompt and client IDs must be canonical UUIDs")


@dataclass(frozen=True)
class OutputRef:
    filename: str
    subfolder: str = ""
    type: str = "output"

    def __post_init__(self):
        safe_filename(self.filename)
        if self.type not in {"output", "temp"} or len(self.subfolder) > 512:
            raise ValueError("Invalid ComfyUI output location")
        if self.subfolder:
            for part in self.subfolder.split("/"):
                safe_filename(part)


@dataclass(frozen=True)
class BridgeEvent:
    prompt_id: str
    kind: str
    node: str | None = None
    value: int | float | None = None
    maximum: int | float | None = None
    preview: bytes | None = field(default=None, repr=False)


def decode_image(data: bytes, content_type: str, config: BridgeConfig) -> np.ndarray:
    """Inspect dimensions before allocating decoded pixel storage."""
    if not data or len(data) > config.max_image_bytes:
        raise DomainError("COMFY_IMAGE_SIZE", "Image exceeds the configured byte limit", 422)
    width = height = 0
    if content_type == "image/png" and data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 33 and data[12:16] == b"IHDR":
        width, height = struct.unpack(">II", data[16:24])
    elif content_type == "image/jpeg" and data.startswith(b"\xff\xd8"):
        at = 2
        while at + 4 <= len(data):
            if data[at] != 255:
                break
            marker = data[at + 1]
            if marker == 255:
                at += 1
                continue
            length = int.from_bytes(data[at + 2:at + 4], "big")
            if length < 2 or at + 2 + length > len(data):
                break
            if marker in {0xC0, 0xC1, 0xC2} and length >= 7:
                height, width = struct.unpack(">HH", data[at + 5:at + 9])
                break
            at += 2 + length
    if width <= 0 or height <= 0 or width * height > config.max_pixels:
        raise DomainError("COMFY_IMAGE_FORMAT", "Image format or dimensions are unsupported", 422)
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape != (height, width, 3):
        raise DomainError("COMFY_IMAGE_INVALID", "Image decode failed validation", 422)
    return image


class ComfyUIBridge:
    def __init__(self, config: BridgeConfig = BridgeConfig(), *, root: Path | None = None,
                 http_transport=None, ws_connect=LoopbackConnect):
        self.config, self.root = config, root or application_root()
        self.http = httpx.AsyncClient(base_url=config.endpoint, timeout=config.request_timeout,
                                     follow_redirects=False, trust_env=False, transport=http_transport,
                                     headers={"Accept-Encoding": "identity"})
        self._ws_connect = ws_connect
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.http.aclose()

    async def _request(self, method: str, path: str, *, payload=None, params=None, image=False):
        limit = self.config.max_image_bytes if image else self.config.max_json_bytes
        try:
            async with self.http.stream(method, path, json=payload, params=params,
                                        timeout=self.config.request_timeout) as response:
                if not 200 <= response.status_code < 300:
                    code = "COMFY_TRANSIENT" if response.status_code == 429 or response.status_code >= 500 else "COMFY_HTTP_ERROR"
                    raise DomainError(code, "Local ComfyUI rejected the request", 503)
                if response.headers.get("content-encoding", "identity") != "identity":
                    raise DomainError("COMFY_RESPONSE_INVALID", "Compressed responses are not accepted", 422)
                content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                if content_type not in ({"image/png", "image/jpeg"} if image else {"application/json", ""}):
                    raise DomainError("COMFY_CONTENT_TYPE", "Unexpected ComfyUI content type", 422)
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    data.extend(chunk)
                    if len(data) > limit:
                        raise DomainError("COMFY_RESPONSE_SIZE", "ComfyUI response exceeds the configured byte limit", 422)
                if image:
                    return bytes(data), content_type
                result = json.loads(data) if data else {}
                if not isinstance(result, dict):
                    raise ValueError()
                return result
        except httpx.HTTPError:
            raise DomainError("COMFY_UNAVAILABLE", "Local ComfyUI is unavailable", 503) from None
        except (ValueError, RecursionError):
            raise DomainError("COMFY_RESPONSE_INVALID", "Local ComfyUI returned an invalid response", 422) from None

    async def probe(self) -> dict:
        stats = await self._request("GET", "/system_stats")
        nodes = await self._request("GET", "/object_info")
        if not isinstance(stats.get("system"), dict) or not nodes:
            raise DomainError("COMFY_NOT_READY", "ComfyUI system or node metadata is unavailable", 503)
        self._verify_protocol(stats)
        return nodes

    @staticmethod
    def _verify_protocol(stats: dict) -> None:
        # Conservative reviewed floor, not a claim about the first supporting
        # release. v0.43.0/server.py verifies UUID prompt_id and targeted interrupt.
        system = stats.get("system", {})
        version = system.get("comfyui_version") if isinstance(system, dict) else None
        if (not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", version)
                or tuple(map(int, version.split("."))) < (0, 43, 0)):
            raise DomainError("COMFY_PROTOCOL_UNSUPPORTED", "ComfyUI 0.43.0 or newer is required for the reviewed prompt and cancellation protocol", 503)

    async def validate_workflow(self, workflow: dict) -> dict:
        try:
            encoded = json.dumps(workflow, allow_nan=False).encode()
            if len(encoded) > 1024 * 1024 or not workflow or len(workflow) > 1024:
                raise ValueError()
            workflow = json.loads(encoded)
            if not isinstance(workflow, dict) or any(not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", key) for key in workflow):
                raise ValueError()
        except (TypeError, ValueError, RecursionError):
            raise DomainError("COMFY_WORKFLOW_INVALID", "A bounded API-format workflow object is required", 422) from None
        nodes = await self.probe()
        edges = {}
        for key, node in workflow.items():
            if (not isinstance(node, dict) or not isinstance(node.get("inputs"), dict)
                    or not isinstance(node.get("class_type"), str) or node["class_type"] not in nodes):
                raise DomainError("COMFY_NODE_MISSING", "A required workflow node is unavailable", 422)
            metadata = nodes[node["class_type"]]
            if not isinstance(metadata, dict) or not isinstance(metadata.get("input"), dict):
                raise DomainError("COMFY_NODE_METADATA_INVALID", "Node metadata is invalid", 422)
            spec = metadata["input"]
            required, optional = spec.get("required", {}), spec.get("optional", {})
            if not isinstance(required, dict) or not isinstance(optional, dict):
                raise DomainError("COMFY_NODE_METADATA_INVALID", "Node input metadata is invalid", 422)
            if not set(required).issubset(node["inputs"]) or not set(node["inputs"]).issubset(set(required) | set(optional)):
                raise DomainError("COMFY_INPUT_INVALID", "Workflow node inputs do not match runtime metadata", 422)
            edges[key] = []
            for name, value in node["inputs"].items():
                declared = (required | optional)[name]
                kind = declared[0] if isinstance(declared, list) and declared else None
                if isinstance(kind, list):
                    if value not in kind:
                        raise DomainError("COMFY_MODEL_OR_OPTION_MISSING", "A checkpoint or node option is unavailable", 422)
                elif isinstance(value, list):
                    if (len(value) != 2 or not isinstance(value[0], str) or value[0] not in workflow
                            or type(value[1]) is not int or value[1] < 0):
                        raise DomainError("COMFY_LINK_INVALID", "Workflow contains an invalid node link", 422)
                    upstream = workflow[value[0]]
                    upstream_class = upstream.get("class_type") if isinstance(upstream, dict) else None
                    upstream_info = nodes.get(upstream_class, {}) if isinstance(upstream_class, str) else {}
                    outputs = upstream_info.get("output") if isinstance(upstream_info, dict) else None
                    if isinstance(outputs, list) and value[1] >= len(outputs):
                        raise DomainError("COMFY_LINK_INVALID", "Node output slot is unavailable", 422)
                    edges[key].append(value[0])
                elif name == "ckpt_name":
                    raise DomainError("COMFY_CHECKPOINT_UNVERIFIED", "Checkpoint list is absent from runtime node metadata", 422)
                elif isinstance(kind, str) and kind in {"INT", "FLOAT", "STRING", "BOOLEAN"}:
                    expected = {"INT": (int,), "FLOAT": (int, float), "STRING": (str,), "BOOLEAN": (bool,)}[kind]
                    if type(value) not in expected:
                        raise DomainError("COMFY_INPUT_INVALID", "Node literal has the wrong type", 422)
        visited, active = set(), set()
        def visit(key):
            if key in active:
                raise DomainError("COMFY_LINK_INVALID", "Workflow contains a cycle", 422)
            if key not in visited:
                active.add(key)
                for dependency in edges[key]:
                    visit(dependency)
                active.remove(key)
                visited.add(key)
        try:
            for key in edges:
                visit(key)
        except RecursionError:
            raise DomainError("COMFY_LINK_INVALID", "Workflow graph is too deep", 422) from None
        return workflow

    @staticmethod
    def _receipt(job_id: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", job_id):
            raise ValueError("Invalid bridge job ID")
        return f"roop_3d/.runtime/bridges/comfyui/{job_id}.json"

    def _load(self, handle: PromptHandle) -> dict:
        try:
            value = json.loads(owned(self.root, self._receipt(handle.job_id)).read_text("utf-8"))
            if (value["handle"] != vars(handle) or value["endpoint"] != self.config.endpoint):
                raise ValueError()
            return value
        except (OSError, ValueError, KeyError, TypeError):
            raise DomainError("COMFY_RECEIPT_INVALID", "Prompt correlation receipt is invalid", 409) from None

    async def submit(self, job_id: str, workflow: dict) -> PromptHandle:
        workflow = await self.validate_workflow(workflow)
        digest = hashlib.sha256(json.dumps(workflow, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        relative = self._receipt(job_id)
        async with self._lock:
            with exclusive(self.root, relative + ".lock"):
                path = owned(self.root, relative)
                if path.exists():
                    try:
                        saved = json.loads(path.read_text("utf-8"))
                        handle = PromptHandle(**saved["handle"])
                    except (ValueError, TypeError, KeyError):
                        raise DomainError("COMFY_RECEIPT_INVALID", "Saved prompt receipt is invalid", 409) from None
                    if handle.job_id != job_id:
                        raise DomainError("COMFY_RECEIPT_INVALID", "Saved prompt job identity is invalid", 409)
                    self._load(handle)
                    if handle.workflow_sha256 != digest:
                        raise DomainError("COMFY_IDEMPOTENCY_CONFLICT", "Job ID already belongs to another workflow", 409)
                    return handle
                handle = PromptHandle(job_id, str(uuid.uuid4()), str(uuid.uuid4()), digest)
                saved = {"handle": vars(handle), "endpoint": self.config.endpoint, "state": "submission_unknown"}
                atomic_json(self.root, relative, saved)
            # Never retry POST: timeout may occur after ComfyUI has enqueued it.
            result = await self._request("POST", "/prompt", payload={"prompt": workflow,
                "client_id": handle.client_id, "prompt_id": handle.prompt_id})
            if result.get("prompt_id") != handle.prompt_id or result.get("node_errors"):
                raise DomainError("COMFY_SUBMISSION_UNVERIFIED", "Server did not confirm the reserved prompt ID", 409)
            atomic_json(self.root, relative, {**saved, "state": "submitted"})
            return handle

    async def poll(self, handle: PromptHandle) -> dict | None:
        if self._load(handle)["state"] == "cancelled":
            raise DomainError("COMFY_CANCELLED", "Prompt was cancelled", 409)
        history = await self._request("GET", f"/history/{handle.prompt_id}")
        entry = history.get(handle.prompt_id)
        if entry is None:
            return None
        if not isinstance(entry, dict):
            raise DomainError("COMFY_HISTORY_INVALID", "Prompt history has an invalid structure", 422)
        status = entry.get("status", {})
        if not isinstance(status, dict):
            raise DomainError("COMFY_HISTORY_INVALID", "Prompt history status has an invalid structure", 422)
        if status.get("status_str") == "error":
            raise DomainError("COMFY_EXECUTION_FAILED", "ComfyUI prompt execution failed", 422)
        return entry if status.get("completed") is True else None

    async def cancel(self, handle: PromptHandle) -> None:
        saved = self._load(handle)
        if saved["state"] == "cancelled":
            return
        self._verify_protocol(await self._request("GET", "/system_stats"))
        await self._request("POST", "/queue", payload={"delete": [handle.prompt_id]})
        await self._request("POST", "/interrupt", payload={"prompt_id": handle.prompt_id})
        atomic_json(self.root, self._receipt(handle.job_id), {**saved, "state": "cancelled"})

    async def wait(self, handle: PromptHandle, *, on_event: Callable[[BridgeEvent], None] | None = None,
                   cancel: asyncio.Event | None = None) -> dict:
        try:
            return await asyncio.wait_for(self._wait(handle, on_event=on_event, cancel=cancel), self.config.execution_timeout)
        except asyncio.TimeoutError:
            raise DomainError("COMFY_TIMEOUT", "Execution deadline elapsed; prompt ID is preserved for polling or cancellation", 504) from None

    async def _wait(self, handle: PromptHandle, *, on_event=None, cancel=None) -> dict:
        self._load(handle)
        deadline = time.monotonic() + self.config.execution_timeout
        attempts = 0
        emit = on_event or (lambda event: None)
        url = self.config.endpoint.replace("http", "ws", 1) + "/ws?clientId=" + handle.client_id
        while time.monotonic() < deadline:
            if cancel is not None and cancel.is_set():
                await self.cancel(handle)
                raise DomainError("COMFY_CANCELLED", "Prompt cancellation was requested", 409)
            if self._load(handle)["state"] == "cancelled":
                raise DomainError("COMFY_CANCELLED", "Prompt was cancelled", 409)
            try:
                completed = await self.poll(handle)
                if completed is not None:
                    return completed
                async with self._ws_connect(url, open_timeout=min(self.config.request_timeout, max(.001, deadline - time.monotonic())),
                        close_timeout=1, max_size=self.config.max_image_bytes + 8, proxy=None) as socket:
                    correlated_node = None
                    while time.monotonic() < deadline:
                        if cancel is not None and cancel.is_set():
                            await self.cancel(handle)
                            raise DomainError("COMFY_CANCELLED", "Prompt cancellation was requested", 409)
                        try:
                            raw = await asyncio.wait_for(socket.recv(), min(self.config.poll_interval, max(.001, deadline - time.monotonic())))
                        except asyncio.TimeoutError:
                            completed = await self.poll(handle)
                            if completed is not None:
                                return completed
                            continue
                        if isinstance(raw, bytes):
                            if correlated_node is not None and len(raw) > 8:
                                event_type, format_type = struct.unpack(">II", raw[:8])
                                if event_type == 1 and format_type in {1, 2}:
                                    decode_image(raw[8:], "image/jpeg" if format_type == 1 else "image/png", self.config)
                                    emit(BridgeEvent(handle.prompt_id, "preview", correlated_node, preview=raw[8:]))
                            continue
                        if len(raw.encode()) > self.config.max_json_bytes:
                            raise DomainError("COMFY_RESPONSE_SIZE", "WebSocket message exceeds the byte limit", 422)
                        try:
                            message = json.loads(raw)
                            data, kind = message["data"], message["type"]
                            if not isinstance(data, dict):
                                raise ValueError()
                        except (ValueError, TypeError, KeyError, RecursionError):
                            raise DomainError("COMFY_EVENT_INVALID", "ComfyUI event has invalid structure", 422) from None
                        if data.get("prompt_id") != handle.prompt_id:
                            correlated_node = None
                            continue
                        if kind in {"execution_error", "execution_interrupted"}:
                            raise DomainError("COMFY_EXECUTION_FAILED", "ComfyUI execution failed or was interrupted", 422)
                        node = data.get("node")
                        correlated_node = str(node) if node is not None else None
                        if kind in {"executing", "progress", "executed", "execution_start", "execution_success"}:
                            emit(BridgeEvent(handle.prompt_id, kind, correlated_node, data.get("value"), data.get("max")))
                        if kind == "execution_success" or kind == "executing" and node is None:
                            completed = await self.poll(handle)
                            if completed is not None:
                                return completed
            except (ConnectionClosed, InvalidHandshake, OSError, asyncio.TimeoutError, DomainError) as error:
                if isinstance(error, DomainError) and error.code not in {"COMFY_TRANSIENT", "COMFY_UNAVAILABLE"}:
                    raise
                attempts += 1
                if attempts > self.config.reconnect_attempts:
                    raise DomainError("COMFY_DISCONNECTED", "ComfyUI reconnect budget was exhausted; prompt ID is preserved", 503) from None
                await asyncio.sleep(min(self.config.backoff_initial * 2 ** (attempts - 1), 2, max(0, deadline - time.monotonic())))
        raise DomainError("COMFY_TIMEOUT", "Execution deadline elapsed; prompt ID is preserved for polling or cancellation", 504)

    @staticmethod
    def outputs(history: dict) -> tuple[OutputRef, ...]:
        try:
            return tuple(OutputRef(**image) for node in history.get("outputs", {}).values() for image in node.get("images", []))
        except (TypeError, ValueError, AttributeError):
            raise DomainError("COMFY_OUTPUT_INVALID", "History contains an unsafe output descriptor", 422) from None

    async def fetch_output(self, handle: PromptHandle, output: OutputRef) -> np.ndarray:
        history = await self.poll(handle)
        if history is None or output not in self.outputs(history):
            raise DomainError("COMFY_OUTPUT_UNCORRELATED", "Output is not part of the completed prompt history", 409)
        data, content_type = await self._request("GET", "/view", params=vars(output), image=True)
        return decode_image(data, content_type, self.config)

    async def composite_output(self, handle: PromptHandle, output: OutputRef, *, compositor,
                               frame: np.ndarray, inverse: np.ndarray, track_id: int, landmarks=None, **options) -> np.ndarray:
        from roop_3d.engine.volumetric.session import Replacement
        crop = await self.fetch_output(handle, output)
        inverse = np.asarray(inverse, np.float32)
        if inverse.shape != (2, 3) or not np.isfinite(inverse).all() or type(track_id) is not int or track_id < 0:
            raise DomainError("COMFY_COMPOSITE_INPUT_INVALID", "A valid crop transform and track ID are required", 422)
        replacement = Replacement(track_id, crop, inverse, 0, "local_comfyui")
        return compositor.compose(frame, (replacement,), {track_id: landmarks} if landmarks is not None else {}, **options)
