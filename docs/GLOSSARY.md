# Roop 3D Orchestration

Language for choosing and coordinating the AI components that make up a face-swap processing run.

## Language

**Model**:
A governed, hash-verified neural-network artifact with recorded provenance, license, and hardware/precision requirements.
_Avoid_: Weights, checkpoint, network

**Engine**:
One orchestrator-selectable provider of a single capability category, composed of one or more Models, an adapter, and a declared input/output contract.
_Avoid_: Backend, provider, plugin

**Pipeline plan**:
The set of Engines chosen to process one scene segment of one track.
_Avoid_: Workflow, configuration

**Render profile**:
A named set of allowed Engines per category, required or optional, from which Pipeline plans are formed. The baseline profile allows exactly one Engine per category.
_Avoid_: Mode (reserved for the execution mode)

**Job runner**:
The existing whole-job executor that a job is dispatched to by identifier.
_Avoid_: Engine (reserved for the capability provider)

**Shot**:
One continuous stretch of the target between hard cuts. All tracks in a shot share one Pipeline plan.
_Avoid_: Segment, scene (when meaning the plan unit)

**Evidence record**:
A measurement of one concrete Pipeline plan on one hardware tier, made with real Models. Only an exact match counts for a plan.
_Avoid_: Benchmark, score

**Baseline plan**:
The pinned plan equal to today's standard reference swap; it is the fallback and the quality floor.
_Avoid_: Default plan

**Unavailable Engine**:
A well-formed Engine that cannot be used on a tier because its adapter, governance state or tier support is missing.
_Avoid_: Disabled engine
