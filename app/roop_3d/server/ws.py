"""Bounded process-local event fanout; workers publish after durable commits."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import threading
from typing import Annotated, Any, Callable, Iterator, Literal, Union

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


StageId = Literal["STAGE_1_SPARSE_PRESCAN", "STAGE_2_ACTIVE_PREPASS", "STAGE_3_MAIN_SWAP"]
StageStatus = Literal["PENDING", "RUNNING", "DONE", "SKIPPED", "FAILED", "RESTORED"]
EngineState = Literal["IDLE", "COMPILING_TRT", "RUNNING_CANARY"]


class PipelineStatusEvent(EventBase):
    """One telemetry frame. None means "not measured", never a placeholder number."""
    type: Literal["pipeline.status"] = "pipeline.status"
    job_id: str | None = None
    stage: StageId | None = None
    stages: dict[StageId, StageStatus]
    stage_progress_pct: float | None = Field(default=None, ge=0, le=100)
    active_ranges_count: int | None = Field(default=None, ge=0)
    verified_target_frames: int | None = Field(default=None, ge=0)
    skipped_frames_pct: float | None = Field(default=None, ge=0, le=100)
    eta_seconds: float | None = Field(default=None, ge=0)
    fps: float = Field(default=0, ge=0)
    engine_state: EngineState = "IDLE"
    engine_label: str | None = None
    engine_model: str | None = None
    engine_precision: str | None = None
    engine_since: float | None = None


class StreamGapEvent(EventBase):
    type: Literal["stream.gap"] = "stream.gap"
    action: Literal["refresh_jobs"] = "refresh_jobs"


class StreamReadyEvent(EventBase):
    type: Literal["stream.ready"] = "stream.ready"
    action: Literal["refresh_jobs"] = "refresh_jobs"


class QueueEvent(EventBase):
    type: Literal["queue.event"] = "queue.event"
    action: str = ""
    job_id: str | None = None
    state: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


QueueEvent.model_rebuild()

Event = Annotated[Union[JobStateEvent, JobProgressEvent, TelemetryEvent, PipelineStatusEvent, StreamGapEvent, StreamReadyEvent, QueueEvent],
                  Field(discriminator="type")]
EVENT_ADAPTER = TypeAdapter(Event)
EVENT_ADAPTER.rebuild()


class EventBroker:
    def __init__(self, capacity: int = 128):
        self.capacity = capacity
        self._subscribers: dict[asyncio.Queue, asyncio.AbstractEventLoop] = {}
        self._observers: list[Callable[[EventBase], None]] = []
        self._lock = threading.Lock()
        self.sequence = 0

    def add_observer(self, observer: Callable[[EventBase], None]) -> None:
        """Synchronous tap called after fanout, outside the broker lock (it may publish)."""
        self._observers.append(observer)

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
        for observer in self._observers:
            observer(event)

    @staticmethod
    def _deliver(queue: asyncio.Queue, event: EventBase) -> None:
        if queue.full():
            while not queue.empty():
                queue.get_nowait()
            queue.put_nowait(StreamGapEvent(sequence=event.sequence))
        else:
            queue.put_nowait(event)


async def event_stream(websocket: WebSocket) -> None:
    """Job, hardware and pipeline events (`/ws/telemetry`, `/ws/events`)."""
    await _stream(websocket, lambda event: event)


async def pipeline_status_stream(websocket: WebSocket) -> None:
    """Only `pipeline.status` frames (`/ws/v1/pipeline/status`); a snapshot opens and follows every gap."""
    tracker = websocket.app.state.pipeline

    def select(event: EventBase) -> EventBase | None:
        if isinstance(event, PipelineStatusEvent):
            return event
        return tracker.snapshot() if isinstance(event, StreamGapEvent) else None

    await _stream(websocket, select, greeting=lambda: tracker.snapshot())


async def _stream(websocket: WebSocket, select: Callable[[EventBase], EventBase | None],
                  greeting: Callable[[], EventBase] | None = None) -> None:
    broker: EventBroker = websocket.app.state.events
    with broker.subscribe() as queue:
        await websocket.accept()
        await websocket.send_json(StreamReadyEvent(sequence=broker.sequence).model_dump(mode="json"))
        if greeting is not None:
            await websocket.send_json(greeting().model_dump(mode="json"))
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
                    selected = select(pending.result())
                    if selected is not None:
                        await websocket.send_json(selected.model_dump(mode="json"))
                    pending = asyncio.create_task(queue.get())
        except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
            return
        finally:
            receive.cancel()
            pending.cancel()
            try:
                await asyncio.gather(receive, pending, return_exceptions=True)
            except asyncio.CancelledError:
                pass  # ASGI cancellation may arrive again during disconnect cleanup.
