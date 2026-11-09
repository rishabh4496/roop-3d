"""Which model file each pipeline stage has loaded right now.

Every session holder (detector, swapper, enhancer) reports here when it builds a session. The registry is the
single answer to "what is actually in VRAM" for the UI read-back badge, and it is where a model switch is
detected: sessions are keyed by ``(stage, model_id, model_path, execution_provider, precision_mode)`` and a
request whose key differs from the active one must replace the session instead of reusing it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

STAGES = ("detector", "swapper", "enhancer")
CacheKey = tuple[str, str, str, str, str]


def cache_key(stage: str, model_id: str, model_path: Path | str | None, provider: str | None,
              precision: str | None) -> CacheKey:
    return (stage, str(model_id), str(model_path or ""), str(provider or "auto"), str(precision or "auto"))


@dataclass(frozen=True)
class ActiveModel:
    stage: str
    model_id: str
    model_path: str
    filename: str
    provider: str
    precision: str
    sha256: str | None
    loaded_at: float
    switches: int

    @property
    def key(self) -> CacheKey:
        return cache_key(self.stage, self.model_id, self.model_path, self.provider, self.precision)


_lock = threading.RLock()
_active: dict[str, ActiveModel] = {}
_switches: dict[str, int] = {}


def current_key(stage: str) -> CacheKey | None:
    with _lock:
        value = _active.get(stage)
        return value.key if value else None


def activate(stage: str, model_id: str, model_path: Path | str | None, provider: str | None,
             precision: str | None, sha256: str | None = None) -> ActiveModel:
    """Record that ``stage`` now runs ``model_path``; logs a switch when the cache key changed."""
    path = str(model_path or "")
    with _lock:
        previous = _active.get(stage)
        key = cache_key(stage, model_id, path, provider, precision)
        _failures.pop(stage, None)          # a working session of this stage supersedes an earlier failed switch
        if previous is not None and previous.key == key:
            return previous
        switches = _switches.get(stage, 0) + (1 if previous is not None else 0)
        _switches[stage] = switches
        entry = ActiveModel(stage, str(model_id), path, Path(path).name if path else str(model_id),
                            str(provider or "auto"), str(precision or "auto"), sha256, time.time(), switches)
        _active[stage] = entry
    if previous is not None:
        log.info("Switched %s model -> %s (%s)", stage, model_id, path or "no file")
    else:
        log.info("Loaded %s model -> %s (%s)", stage, model_id, path or "no file")
    return entry


@dataclass(frozen=True)
class SwitchFailure:
    """A model that could not be loaded for ``stage``: the exact backend code and reason, never a paraphrase."""
    id: int
    stage: str
    model_id: str
    code: str
    reason: str
    at: float


_failures: dict[str, SwitchFailure] = {}
_failure_seq = 0


def report_failure(stage: str, model_id: str, error: BaseException) -> SwitchFailure:
    """Record that loading ``model_id`` for ``stage`` failed (latest failure per stage wins)."""
    global _failure_seq
    code = str(getattr(error, "code", None) or type(error).__name__)
    reason = str(getattr(error, "message", None) or error) or code
    with _lock:
        _failure_seq += 1
        failure = SwitchFailure(_failure_seq, stage, str(model_id), code, reason[:500], time.time())
        _failures[stage] = failure
    log.warning("Could not switch %s model -> %s: %s (%s)", stage, model_id, reason, code)
    return failure


def clear_failure(stage: str) -> None:
    """A request for ``stage`` was served by a working session, so an older failed switch no longer applies."""
    with _lock:
        _failures.pop(stage, None)


def failures() -> list[dict[str, Any]]:
    with _lock:
        return [asdict(item) for item in sorted(_failures.values(), key=lambda item: item.id)]


class switch_guard:
    """``with switch_guard(stage, model_id):`` records the failure of a session build (then re-raises it).

    ``on_fail`` runs first, e.g. to forget a session that was dropped before the replacement failed to build."""

    def __init__(self, stage: str, model_id: str, on_fail=None):
        self.stage, self.model_id, self.on_fail = stage, model_id, on_fail

    def __enter__(self):
        return self

    def __exit__(self, kind, error, trace):
        if error is not None and isinstance(error, Exception):
            if self.on_fail is not None:
                self.on_fail()
            report_failure(self.stage, self.model_id, error)
        return False


def release(stage: str, key: CacheKey | None = None) -> None:
    """Forget ``stage`` once its session is released. With ``key`` only that exact session is forgotten, so a
    late release of an old session cannot erase the model that replaced it."""
    with _lock:
        value = _active.get(stage)
        if value is not None and (key is None or value.key == key):
            del _active[stage]


def snapshot() -> dict[str, Any]:
    with _lock:
        return {stage: (asdict(_active[stage]) if stage in _active else None) for stage in STAGES}


def reset() -> None:
    global _failure_seq
    with _lock:
        _active.clear()
        _switches.clear()
        _failures.clear()
        _failure_seq = 0
