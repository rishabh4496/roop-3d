"""SDXL-only diffusion VFX backend policy."""
from __future__ import annotations

from typing import Any

from roop_3d.core.errors import DomainError

from .base import BackendState, DiffusionBackend, MemoryPlan, check_memory_budget, configure_sdxl_memory
from .identity_adapter import ModelFamily


class SDXLBackend(DiffusionBackend):
    family: ModelFamily = "sdxl"
    base_model_ids = frozenset({"sdxl", "sdxl-turbo"})
    adapter_model_ids = frozenset({"instantid", "ip-adapter-faceid-sdxl", "pulid-sdxl"})

    def validate_pipeline(self, pipeline: Any) -> None:
        declared = getattr(getattr(pipeline, "config", None), "_class_name", None)
        name = str(declared or type(pipeline).__name__).casefold()
        if "flux" in name or not any(token in name for token in ("xl", "sdxl")):
            raise DomainError("DIFFUSION_FAMILY_MISMATCH", "Pipeline loader returned a non-SDXL pipeline", 422)

    def _family_memory_plan(self, required_vram_bytes: int | None) -> MemoryPlan:
        state, reason = check_memory_budget(self.available_vram_bytes, required_vram_bytes)
        if self.profile == "6gb":
            if state == BackendState.UNSUPPORTED_VRAM:
                return MemoryPlan(self.profile, self.available_vram_bytes, required_vram_bytes, state,
                                  True, True, True, reason)
            return MemoryPlan(self.profile, self.available_vram_bytes, required_vram_bytes,
                              state,
                              True, True, True,
                              reason or ("6 GB offload/slicing enabled" if state == BackendState.AVAILABLE
                                         else "6 GB offload/slicing enabled; model VRAM requirement is not registered"))
        return MemoryPlan(self.profile, self.available_vram_bytes, required_vram_bytes, state,
                          reason=reason)

    def configure_pipeline_memory(self, pipeline: Any) -> MemoryPlan:
        plan = configure_sdxl_memory(pipeline, self.profile)
        if plan.state == BackendState.UNAVAILABLE:
            return plan
        required = self._registered_required_vram((self._record(self.base_model_id),
                                                   self._record(self.adapter_id)))
        state, reason = check_memory_budget(self.available_vram_bytes, required)
        return MemoryPlan(plan.profile, self.available_vram_bytes, required,
                          state, plan.model_cpu_offload, plan.vae_slicing,
                          plan.attention_slicing, reason)

    def generate(self, **kwargs: Any) -> Any:
        pipeline = self.require_loaded_pipeline()
        prompt = kwargs.get("prompt")
        image = kwargs.get("image")
        strength = kwargs.get("strength", 0.25)
        if not isinstance(prompt, str) or not prompt.strip() or image is None or not 0.15 <= strength <= 0.45:
            raise DomainError("DIFFUSION_REQUEST_INVALID", "SDXL ROI generation requires prompt, image, and strength 0.15..0.45", 422)
        return pipeline(**self.deterministic_generation_kwargs(kwargs))
