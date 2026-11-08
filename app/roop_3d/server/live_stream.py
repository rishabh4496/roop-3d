"""Binary live-frame WebSocket and pipeline saturation telemetry.

    WS   /ws/v1/preview/stream      server -> client: binary frames (16-byte header + JPEG), JSON control messages
    GET  /api/v1/system/pipeline    queue depths, per-stage latency, GPU / CPU / PCIe utilisation, encoder backend

Frames reach the browser as raw ``ArrayBuffer`` messages -- no base64, no JSON, no PNG -- and are decoded there with
``createImageBitmap``.  Two kinds of frame share the socket:

* **job egress** -- every finished frame of a running render is offered to the hub (latest wins, never blocks the render);
* **interactive replies** -- the client sends ``{"op": "render", "id": n, "config": <hash>, "frame_index": i}`` for a
  configuration it POSTed to ``/api/preview`` earlier; the reply is a binary frame for that subscriber only.  A newer
  request supersedes an unfinished one, so scrubbing never queues stale work.

Text messages are control: ``ready``, ``rendered``, ``busy``, ``error``, ``pong``.
"""
from __future__ import annotations

import asyncio
import json
from threading import Event
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from roop_3d.async_pipeline import monitor, snapshot_all
from roop_3d.core.transfer_ledger import TRANSFERS
from roop_3d.async_pipeline.live import HEADER_SIZE, SOURCE_PREVIEW, LiveFrameHub, Subscription
from roop_3d.core.errors import DomainError
from roop_3d.engine.resources import fence_snapshot
from roop_3d.server.live_preview import PreviewCancelled

router = APIRouter(prefix="/api/v1")
MAX_MESSAGE_BYTES = 4096


def make_hub() -> LiveFrameHub:
    """One hub per app.  The encoder (nvJPEG, else libjpeg-turbo) is chosen lazily on the first watched frame."""
    return LiveFrameHub(None, max_fps=60.0)


def _stage_ms(meta: dict[str, Any]) -> tuple[float, float, float]:
    latency = meta.get("latency") or {}
    detection = float(latency.get("detection_ms") or 0.0)
    swap = float(latency.get("swap_ms") or 0.0) + float(latency.get("retarget_ms") or 0.0) + float(latency.get("projection_ms") or 0.0)
    return detection, swap, float(latency.get("composite_ms") or 0.0)


@router.get("/system/pipeline")
def pipeline_metrics(request: Request) -> dict[str, Any]:
    hardware = monitor().keepalive(30.0)
    hub: LiveFrameHub = request.app.state.live_hub
    runner = getattr(getattr(request.app.state, "worker", None), "runner", None)
    finished = getattr(runner, "last_pipeline_snapshot", None)
    live = snapshot_all()
    if finished is not None and all(item["id"] != finished["id"] for item in live):
        live.append(finished)
    return {"pipelines": live, "hardware": hardware.snapshot(), "live_stream": hub.stats(), "fence": fence_snapshot(),
            "transfers": TRANSFERS.snapshot(),
            "pcie_note": "NVML reports PCIe throughput over a ~20 ms window; values are instantaneous, not averaged."}


async def preview_stream(websocket: WebSocket) -> None:
    app = websocket.app
    hub: LiveFrameHub = app.state.live_hub
    await websocket.accept()
    subscription: Subscription = hub.subscribe(asyncio.get_running_loop())
    current: dict[str, Any] = {"task": None, "cancel": None}
    send_lock = asyncio.Lock()

    async def send_json(value: dict[str, Any]) -> None:
        async with send_lock:
            await websocket.send_text(json.dumps(value, separators=(",", ":")))

    async def pump() -> None:
        while True:
            message = await subscription.next()
            async with send_lock:
                await websocket.send_bytes(message)

    async def render(request_id: Any, config: str, frame_index: int, cancel: Event) -> None:
        service = app.state.live_previews
        payload = service.configs.get(config)
        if payload is None:
            await send_json({"type": "error", "id": request_id, "code": "PREVIEW_CONFIG_UNKNOWN",
                             "message": "Unknown swap configuration; POST /api/preview first"})
            return
        payload = payload.model_copy(update={"frame_index": frame_index, "timestamp": None})
        try:
            _, (outcome,) = await service.frames(payload, [frame_index], cancel)
        except PreviewCancelled:
            return                                   # superseded by a newer request or the client left
        except DomainError as error:
            if error.code != "PREVIEW_CANCELLED":
                await send_json({"type": "error", "id": request_id, "code": error.code, "message": error.message})
            return
        except Exception:                            # anything unexpected must not kill the socket
            await send_json({"type": "error", "id": request_id, "code": "INTERNAL_ERROR", "message": "Preview failed"})
            return
        if outcome.status == "busy":
            await send_json({"type": "busy", "id": request_id, "stage": outcome.stage, "retry_after_ms": 1000})
            return
        entry = outcome.entry
        hub.publish(entry.bgr(), frame_index=int(entry.meta.get("frame_index", frame_index)),
                    timestamp_s=float(entry.meta.get("timestamp") or 0.0), stage_ms=_stage_ms(entry.meta),
                    cached=outcome.cached, source=SOURCE_PREVIEW, target=subscription, force=True)
        await send_json({"type": "rendered", "id": request_id, "frame_index": entry.meta.get("frame_index", frame_index),
                         "cached": outcome.cached, "width": entry.meta.get("width"), "height": entry.meta.get("height")})

    pump_task = asyncio.create_task(pump())
    try:
        encoder = hub.encoder
        await send_json({"type": "ready", "protocol": 1, "header_bytes": HEADER_SIZE,
                         "encoder": encoder.backend if encoder else "lazy", "max_fps": 1 / hub.min_interval if hub.min_interval else None})
        while True:
            raw = await websocket.receive_text()
            if len(raw) > MAX_MESSAGE_BYTES:
                await websocket.close(code=1009, reason="Control message too large")
                return
            try:
                message = json.loads(raw)
                op = message["op"]
            except (ValueError, KeyError, TypeError):
                await send_json({"type": "error", "code": "BAD_MESSAGE", "message": "Expected a JSON object with an op"})
                continue
            if op == "ping":
                await send_json({"type": "pong", "t": message.get("t")})
            elif op == "render":
                config, index = message.get("config"), message.get("frame_index")
                if not isinstance(config, str) or not 8 <= len(config) <= 64 or type(index) is not int or not 0 <= index <= 10_000_000:
                    await send_json({"type": "error", "id": message.get("id"), "code": "BAD_MESSAGE",
                                     "message": "render needs a config hash and a frame_index"})
                    continue
                if current["cancel"] is not None:
                    current["cancel"].set()                  # a newer request supersedes the unfinished one
                cancel = current["cancel"] = Event()
                current["task"] = asyncio.create_task(render(message.get("id"), config, index, cancel))
            else:
                await send_json({"type": "error", "code": "BAD_MESSAGE", "message": f"Unknown op {str(op)[:32]}"})
    except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
        return
    finally:
        if current["cancel"] is not None:
            current["cancel"].set()
        for task in (pump_task, current["task"]):
            if task is not None:
                task.cancel()
        await asyncio.gather(*(t for t in (pump_task, current["task"]) if t is not None), return_exceptions=True)
        subscription.close()
