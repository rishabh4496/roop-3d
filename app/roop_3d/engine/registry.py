"""Local job-engine dispatch; adapters reuse the existing worker and job contract."""
from __future__ import annotations

from typing import Any
import re

from roop_3d.core.errors import DomainError

REFERENCE_ENGINE = "reference_onnx"
UNAVAILABLE_ENGINES = {
    "minimax_h3_ref2va_experimental": "No verified local H3 video input/output adapter is installed.",
    "bfs_qwen_image21_experimental": "No verified local Qwen image/keyframe adapter is installed.",
}


class EngineRegistry:
    """Resolve explicit job choices to real runners without fallback or model loading.

    The reference adapter delegates to the already configured MasterPipelineRunner
    (including its optimized stream implementation). Experimental descriptions
    are capability gates, never executable adapters.
    """

    def __init__(self) -> None:
        self._runners: dict[str, Any] = {}

    def validate(self, engine_id: str) -> None:
        if engine_id in UNAVAILABLE_ENGINES:
            raise DomainError("ENGINE_UNAVAILABLE", UNAVAILABLE_ENGINES[engine_id], 409)
        if engine_id != REFERENCE_ENGINE and engine_id not in self._runners:
            raise DomainError("ENGINE_NOT_FOUND", "Unknown local engine ID", 422)

    def bind_reference(self, runner: Any) -> None:
        self.register(REFERENCE_ENGINE, runner)

    def register(self, engine_id: str, runner: Any, *, job_contract: str = "roop3d.local-job.v1") -> None:
        """Register application-owned adapters; never accept uploaded code or workflows here."""
        if not re.fullmatch(r"[a-z0-9][a-z0-9_]{0,95}", engine_id) or job_contract != "roop3d.local-job.v1":
            raise ValueError("Engine must support the existing local FaceSet/track job contract")
        if engine_id in UNAVAILABLE_ENGINES:
            raise DomainError("ENGINE_UNAVAILABLE", UNAVAILABLE_ENGINES[engine_id], 409)
        if not all(callable(getattr(runner, name, None)) for name in ("prepare", "run_prepared", "retry_profile")):
            raise ValueError("A local job runner must implement prepare, run_prepared and retry_profile")
        if engine_id in self._runners:
            raise ValueError("An engine adapter is already registered")
        self._runners[engine_id] = runner

    def resolve(self, engine_id: str) -> Any:
        self.validate(engine_id)
        runner = self._runners.get(engine_id)
        if runner is None:
            raise DomainError("ENGINE_UNAVAILABLE", "The local job worker is not configured", 409)
        return runner

    def selectable(self, engine_id: str) -> bool:
        """True only when the engine is registered AND its render path can run: with the true-3D renderer unwired no render
        engine is selectable (there is no 2D / 2.5D fallback to offer instead)."""
        try:
            self.validate(engine_id)
        except DomainError:
            return False
        from roop_3d.core.true3d_guard import renderer_is_wired
        return renderer_is_wired()

    def require_renderer(self, engine_id: str) -> None:
        """Validate the engine and raise ``True3DRendererUnavailable`` ("3D renderer not wired") while the renderer is not wired."""
        self.validate(engine_id)
        from roop_3d.core.true3d_guard import require_true3d_renderer
        require_true3d_renderer(f"engine {engine_id}")
