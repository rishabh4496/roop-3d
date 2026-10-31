"""FLUX-only diffusion VFX backend; SDXL components are never accepted."""
from __future__ import annotations

from typing import Any

from roop_3d.core.errors import DomainError

from .base import BackendState, DiffusionBackend, MemoryPlan, check_memory_budget
from .identity_adapter import ModelFamily


class FluxBackend(DiffusionBackend):
    family: ModelFamily = "flux"
    base_model_ids = frozenset({"flux-1-schnell"})
    adapter_model_ids = frozenset({"pulid-flux"})

    def validate_pipeline(self, pipeline: Any) -> None:
        declared = getattr(getattr(pipeline, "config", None), "_class_name", None)
        name = str(declared or type(pipeline).__name__).casefold()
        if "flux" not in name:
            raise DomainError("DIFFUSION_FAMILY_MISMATCH", "Pipeline loader returned a non-FLUX pipeline", 422)

    def _family_memory_plan(self, required_vram_bytes: int | None) -> MemoryPlan:
        if self.profile == "6gb" and self.quantization not in {"nf4", "gguf"}:
            return MemoryPlan(self.profile, self.available_vram_bytes, required_vram_bytes,
                              BackendState.UNSUPPORTED_VRAM,
                              reason="Unquantized FLUX is unsupported on the 6 GB profile")
        if self.profile == "6gb" and self.quantization in {"nf4", "gguf"}:
            records = (self._record(self.base_model_id), self._record(self.adapter_id))
            proven = all(record is not None and record.status == "verified_local"
                         and record.provenance_state == "VERIFIED_LOCAL"
                         and record.input_output_contract.get("quantization") == self.quantization
                         for record in records)
            if not proven:
                return MemoryPlan(self.profile, self.available_vram_bytes, required_vram_bytes,
                                  BackendState.UNAVAILABLE,
                                  reason="Quantized FLUX requires VERIFIED_LOCAL matching quantization metadata")
        state, reason = check_memory_budget(self.available_vram_bytes, required_vram_bytes)
        return MemoryPlan(self.profile, self.available_vram_bytes, required_vram_bytes,
                          state, reason=reason)

    def configure_pipeline_memory(self, pipeline: Any) -> MemoryPlan:
        # Offload is not a proof that FLUX fits; a quantized registered artifact
        # and a known free-memory requirement are still required on the 6 GB tier.
        if self.profile == "6gb":
            plan = self._family_memory_plan(self._registered_required_vram(
                (self._record(self.base_model_id), self._record(self.adapter_id))))
            if plan.state != BackendState.AVAILABLE:
                return plan
        return self._family_memory_plan(self._registered_required_vram(
            (self._record(self.base_model_id), self._record(self.adapter_id))))

    def generate(self, **kwargs: Any) -> Any:
        pipeline = self.require_loaded_pipeline()
        image = kwargs.get("image")
        prompt = kwargs.get("prompt")
        strength = kwargs.get("strength", 0.25)
        if not isinstance(prompt, str) or not prompt.strip() or image is None or not 0.15 <= strength <= 0.45:
            raise DomainError("DIFFUSION_REQUEST_INVALID", "FLUX ROI generation requires prompt, image, and strength 0.15..0.45", 422)
        return pipeline(**self.deterministic_generation_kwargs(kwargs))
