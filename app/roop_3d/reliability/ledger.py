"""Lifecycle ledger: every long-lived native resource is opened and closed through a counted token.

A leak shows up as a kind whose ``live`` count does not return to its idle baseline, or as a token that was garbage
collected without ever being closed (``abandoned``) - the latter is a definite defect, not a statistic.

The ledger has no dependency on the rest of the application so that low-level modules (decoder, encoder, ffmpeg
spawners, the SQLite queue) can use it without import cycles. Tokens are cheap (one lock, one dict entry).
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import itertools
import threading
import time
import weakref


class Token:
    __slots__ = ("kind", "label", "serial", "opened", "_ledger", "closed", "__weakref__")

    def __init__(self, ledger: "ResourceLedger", kind: str, label: str, serial: int):
        self._ledger, self.kind, self.label, self.serial = ledger, kind, label, serial
        self.opened, self.closed = time.monotonic(), False

    def close(self) -> None:
        self._ledger._close(self)

    def __enter__(self) -> "Token":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class _Stat:
    __slots__ = ("live", "high_water", "opened", "closed", "abandoned")

    def __init__(self) -> None:
        self.live = self.high_water = self.opened = self.closed = self.abandoned = 0


class ResourceLedger:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats: dict[str, _Stat] = {}
        self._open: dict[int, tuple[str, str, float]] = {}
        self._serial = itertools.count(1)
        self._gauges: dict[str, Callable[[], int]] = {}

    # ---- lifecycle
    def open(self, kind: str, label: str = "") -> Token:
        token = Token(self, kind, label, next(self._serial))
        with self._lock:
            stat = self._stats.setdefault(kind, _Stat())
            stat.live += 1
            stat.opened += 1
            stat.high_water = max(stat.high_water, stat.live)
            self._open[token.serial] = (kind, label, token.opened)
        # A token dropped without close() is an abandoned resource; settle the books so counts stay truthful.
        weakref.finalize(token, self._abandon, token.serial, kind)
        return token

    def _close(self, token: Token) -> None:
        with self._lock:
            if token.closed:
                return
            token.closed = True
            if self._open.pop(token.serial, None) is not None:
                stat = self._stats[token.kind]
                stat.live -= 1
                stat.closed += 1

    def _abandon(self, serial: int, kind: str) -> None:
        with self._lock:
            if self._open.pop(serial, None) is not None:
                stat = self._stats[kind]
                stat.live -= 1
                stat.abandoned += 1

    @contextmanager
    def hold(self, kind: str, label: str = "") -> Iterator[Token]:
        token = self.open(kind, label)
        try:
            yield token
        finally:
            token.close()

    # ---- externally owned counts (queues, subscribers, leases): reported next to the tokens
    def gauge(self, name: str, read: Callable[[], int]) -> None:
        with self._lock:
            self._gauges[name] = read

    def drop_gauge(self, name: str) -> None:
        with self._lock:
            self._gauges.pop(name, None)

    # ---- reporting
    def live(self, kind: str) -> int:
        with self._lock:
            stat = self._stats.get(kind)
            return stat.live if stat else 0

    def counts(self) -> dict[str, int]:
        """Flat ``kind -> live`` mapping, gauges included (gauge failures read as 0, never raise)."""
        with self._lock:
            values = {kind: stat.live for kind, stat in self._stats.items()}
            gauges = dict(self._gauges)
        for name, read in gauges.items():
            try:
                values[name] = int(read())
            except Exception:
                values[name] = 0
        return values

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        now = time.monotonic()
        with self._lock:
            oldest: dict[str, float] = {}
            for kind, _, opened in self._open.values():
                oldest[kind] = max(oldest.get(kind, 0.0), now - opened)
            report = {kind: {"live": stat.live, "high_water": stat.high_water, "opened": stat.opened,
                             "closed": stat.closed, "abandoned": stat.abandoned,
                             "oldest_open_s": round(oldest.get(kind, 0.0), 3)}
                      for kind, stat in sorted(self._stats.items())}
            gauges = dict(self._gauges)
        for name, read in gauges.items():
            try:
                report[name] = {"live": int(read()), "gauge": True}
            except Exception:
                report[name] = {"live": 0, "gauge": True, "error": True}
        return report

    def open_older_than(self, seconds: float) -> list[dict[str, float | str]]:
        """Tokens that have been open longer than a job could legitimately hold them (suspected leaks)."""
        now = time.monotonic()
        with self._lock:
            return [{"kind": kind, "label": label, "age_s": round(now - opened, 3)}
                    for kind, label, opened in self._open.values() if now - opened > seconds]

    def abandoned(self) -> int:
        with self._lock:
            return sum(stat.abandoned for stat in self._stats.values())

    def reset(self) -> None:
        """Tests only: forget every counter (tokens still held become inert)."""
        with self._lock:
            self._stats.clear()
            self._open.clear()
            self._gauges.clear()


LEDGER = ResourceLedger()
