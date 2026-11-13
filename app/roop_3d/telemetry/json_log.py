"""Machine-readable log lines for ``roop_3d.cli --json-logs``: one JSON object per line, versioned schema, never ANSI.

Every record has ``schema`` (``roop3d.cli.log/1``), ``ts`` (UTC ISO-8601), ``level`` and ``event``; extra fields are event
specific. Strings are stripped of terminal escape sequences and the line is ASCII-only JSON, so the stream is safe to pipe
into ``jq`` or a log shipper even when the producing process saw colour-capable output.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import re
import sys
import threading
from typing import Any, TextIO

SCHEMA = "roop3d.cli.log/1"
LEVELS = {"debug": 10, "info": 20, "warning": 30, "error": 40}
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")


def clean(value: Any) -> Any:
    """Recursively strip ANSI escape sequences from strings; convert unserialisable leaves to ``repr``."""
    if isinstance(value, str):
        return _ANSI.sub("", value)
    if isinstance(value, dict):
        return {str(_ANSI.sub("", str(k))): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None   # NaN/inf are not JSON
    return clean(repr(value))


class JsonLogger:
    """Thread-safe JSONL emitter honouring a minimum ``level``."""

    def __init__(self, stream: TextIO | None = None, level: str = "info"):
        if level not in LEVELS:
            raise ValueError(f"unknown log level {level!r}")
        self.stream, self.level, self._lock = stream or sys.stdout, level, threading.Lock()

    def emit(self, event: str, level: str = "info", **fields: Any) -> None:
        if LEVELS[level] < LEVELS[self.level]:
            return
        record = {"schema": SCHEMA, "ts": datetime.now(timezone.utc).isoformat(), "level": level, "event": event, **clean(fields)}
        line = json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        with self._lock:
            self.stream.write(line + "\n")
            self.stream.flush()
