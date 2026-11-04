"""Stage 14: capability-based multi-engine orchestrator (planning and reporting; no new execution path)."""
from roop_3d.orchestrator.compile import CompiledPlan, compile_plan
from roop_3d.orchestrator.evidence import EvidenceMatch, EvidenceStore
from roop_3d.orchestrator.plan import Plan
from roop_3d.orchestrator.registry import CapabilityRegistry, OrchestratorRegistryError
from roop_3d.orchestrator.report import build_report, render_text
from roop_3d.orchestrator.scene import SceneSegment, segments_from_tracks
from roop_3d.orchestrator.selector import ManualPick, MediaFacts, Selection, SelectionRequest, Selector

__all__ = ["CapabilityRegistry", "CompiledPlan", "EvidenceMatch", "EvidenceStore", "ManualPick", "MediaFacts",
           "OrchestratorRegistryError", "Plan", "SceneSegment", "Selection", "SelectionRequest", "Selector",
           "build_report", "compile_plan", "render_text", "segments_from_tracks"]
