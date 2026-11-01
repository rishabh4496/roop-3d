"""Mutual exclusion between live-preview inference and model-lifecycle owners.

Two classes of participant:

* **previews** run inference on the shared model/session pool. They never wait: if an owner
  holds (or is waiting for) the gate they are refused immediately with `EngineBusy`.
* **owners** (render jobs, Start-button model re-binds, TensorRT compiles) tear down and
  re-bind models. Several owners may coexist (the job scheduler already bounds them), but an
  owner waits for in-flight previews to drain first. While an owner waits, new previews are
  refused so continuous slider-driven previews can never starve the Start button.

Stage 15 heterogeneous extension:
* Supports device-domain isolation (`gpu` vs `openvino`). When pre-scanning or semantic
  parsing runs on the Intel OpenVINO co-processor (`domain="openvino"`), it does not hold
  or wait for the NVIDIA RTX GPU lock. Thus, `POST /api/preview` calls targeting `gpu`
  execute instantaneously with zero lock contention.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
import itertools
import threading
import time
from typing import AsyncIterator, Iterator

from roop_3d.core.errors import DomainError

REBIND = "MODEL_REBIND"


class EngineBusy(DomainError):
    """The GPU/model pool is owned elsewhere; carries the stage name for the 503 body."""

    def __init__(self, stage: str):
        super().__init__("ENGINE_BUSY", f"The inference engine is busy ({stage}); retry shortly", 503)
        self.stage = stage


class EngineGate:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._previews = 0
        self._waiting = 0
        self._owners: dict[int, str] = {}
        self._domains: dict[int, str] = {}
        self._tokens = itertools.count(1)

    # -- preview side (never blocks) -----------------------------------------------
    def try_enter_preview(self, domain: str = "gpu") -> bool:
        with self._cond:
            if domain == "gpu":
                has_gpu_owner = any(d == "gpu" for d in self._domains.values())
                if has_gpu_owner or self._waiting:
                    return False
            self._previews += 1
            return True

    def leave_preview(self) -> None:
        with self._cond:
            if self._previews <= 0:
                raise RuntimeError("leave_preview without matching enter")
            self._previews -= 1
            if self._previews == 0:
                self._cond.notify_all()

    @contextmanager
    def preview_section(self, domain: str = "gpu") -> Iterator[None]:
        if not self.try_enter_preview(domain=domain):
            raise EngineBusy(self.busy_reason(domain=domain) or REBIND)
        try:
            yield
        finally:
            self.leave_preview()

    # -- owner side (waits for previews to drain) ----------------------------------
    def acquire_owner(self, label: str, timeout: float | None = None, domain: str = "gpu") -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            if domain == "gpu":
                self._waiting += 1
            try:
                if domain == "gpu":
                    while self._previews:
                        remaining = None if deadline is None else deadline - time.monotonic()
                        if remaining is not None and remaining <= 0:
                            raise TimeoutError("In-flight previews did not drain in time")
                        self._cond.wait(remaining)
                token = next(self._tokens)
                self._owners[token] = label
                self._domains[token] = domain
                return token
            finally:
                if domain == "gpu":
                    self._waiting -= 1

    def release_owner(self, token: int) -> None:
        with self._cond:
            self._owners.pop(token, None)
            self._domains.pop(token, None)
            self._cond.notify_all()

    @contextmanager
    def owner(self, label: str, timeout: float | None = None, domain: str = "gpu") -> Iterator[int]:
        token = self.acquire_owner(label, timeout, domain=domain)
        try:
            yield token
        finally:
            self.release_owner(token)

    @asynccontextmanager
    async def aowner(self, label: str, timeout: float | None = None, domain: str = "gpu") -> AsyncIterator[int]:
        """Async variant: the event loop is never blocked while previews drain."""
        token = await asyncio.to_thread(self.acquire_owner, label, timeout, domain)
        try:
            yield token
        finally:
            self.release_owner(token)

    # -- introspection -------------------------------------------------------------
    def busy_reason(self, domain: str = "gpu") -> str | None:
        with self._cond:
            for tok, label in self._owners.items():
                if self._domains.get(tok, "gpu") == domain:
                    return label
            return REBIND if self._waiting and domain == "gpu" else None

    @property
    def previews_in_flight(self) -> int:
        with self._cond:
            return self._previews

    @property
    def owner_count(self) -> int:
        with self._cond:
            return len(self._owners)

    def owner_count_for_domain(self, domain: str) -> int:
        with self._cond:
            return sum(1 for d in self._domains.values() if d == domain)
