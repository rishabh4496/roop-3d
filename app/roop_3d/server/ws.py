"""Bounded process-local event fanout; workers publish after durable commits."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import threading
from typing import Annotated, Iterator, Literal, Union

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import Field, TypeAdapter

from roop_3d.core.hardware import Hardware
from roop_3d.core.settings import Contract
from roop_3d.jobs.schema import JobProgress, JobState


class EventBase(Contract):
    schema_version: Literal[1] = 1
    sequence: int = Field(default=0, ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class JobStateEvent(EventBase):
    type: Literal["job.state"]
    job_id: str
    previous_state: JobState | None
    state: JobState
    revision: int = Field(ge=0)


class JobProgressEvent(EventBase, JobProgress):
    type: Literal["job.progress"]
    job_id: str
    revision: int = Field(ge=0)


class TelemetryEvent(EventBase):
    type: Literal["system.telemetry"]
    hardware: Hardware


class StreamGapEvent(EventBase):
    type: Literal["stream.gap"] = "stream.gap"
    action: Literal["refresh_jobs"] = "refresh_jobs"


class StreamReadyEvent(EventBase):
    type: Literal["stream.ready"] = "stream.ready"
    action: Literal["refresh_jobs"] = "refresh_jobs"


Event = Annotated[Union[JobStateEvent, JobProgressEvent, TelemetryEvent, StreamGapEvent, StreamReadyEvent], Field(discriminator="type")]
EVENT_ADAPTER = TypeAdapter(Event)


class EventBroker:
    def __init__(self, capacity: int = 128):
        self.capacity = capacity
        self._subscribers: dict[asyncio.Queue, asyncio.AbstractEventLoop] = {}
        self._lock = threading.Lock()
        self.sequence = 0

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    @contextmanager
    def subscribe(self) -> Iterator[asyncio.Queue]:
        queue: asyncio.Queue = asyncio.Queue(maxsize=self.capacity)
        with self._lock:
            self._subscribers[queue] = asyncio.get_running_loop()
        try:
            yield queue
        finally:
            with self._lock:
                self._subscribers.pop(queue, None)

    def publish(self, payload: dict) -> None:
        event = EVENT_ADAPTER.validate_python(payload)
        with self._lock:
            self.sequence += 1
            event = event.model_copy(update={"sequence": self.sequence})
            for queue, loop in self._subscribers.items():
                if not loop.is_closed():
                    loop.call_soon_threadsafe(self._deliver, queue, event)

    @staticmethod
    def _deliver(queue: asyncio.Queue, event: EventBase) -> None:
        if queue.full():
            while not queue.empty():
                queue.get_nowait()
            queue.put_nowait(StreamGapEvent(sequence=event.sequence))
        else:
            queue.put_nowait(event)


async def event_stream(websocket: WebSocket) -> None:
    broker: EventBroker = websocket.app.state.events
    with broker.subscribe() as queue:
        await websocket.accept()
        await websocket.send_json(StreamReadyEvent(sequence=broker.sequence).model_dump(mode="json"))
        receive = asyncio.create_task(websocket.receive())
        pending = asyncio.create_task(queue.get())
        try:
            while True:
                done, _ = await asyncio.wait({receive, pending}, return_when=asyncio.FIRST_COMPLETED)
                if receive in done:
                    message = receive.result()
                    if message["type"] == "websocket.disconnect":
                        return
                    # This stream is server-to-client; reject all application payloads.
                    await websocket.close(code=1008, reason="Event stream is receive-only")
                    return
                if pending in done:
                    await websocket.send_json(pending.result().model_dump(mode="json"))
                    pending = asyncio.create_task(queue.get())
        except (WebSocketDisconnect, RuntimeError):
            return
        finally:
            receive.cancel()
            pending.cancel()
            await asyncio.gather(receive, pending, return_exceptions=True)
