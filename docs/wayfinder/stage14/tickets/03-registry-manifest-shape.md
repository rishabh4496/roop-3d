---
title: Registry manifest shape
label: wayfinder:grilling
status: closed
assignee: rishabh4496
blocked_by: []
---

## Question

What is the capability registry's source of truth and shape? Declarative manifest (like `stage6_models.json`) versus code-registered Engines, how Engines bind to code adapters and reference `ModelRecord` ids, where it lives, how it is versioned and hashed into the selection snapshot, where the scene-rule table and per-Engine declared fields (contracts, peak VRAM, `exclusive_with`, declared fallbacks) are stored, and how validation fails closed on a malformed or incomplete entry.

## Resolution

Resolved by grilling (user confirmed every point).

- **Source of truth**: declarative JSON manifest validated by a frozen pydantic `Contract` (extra=forbid, like `ModelRecord`). Each Engine names an `adapter_id`; adapters are registered in code by id only. No capability facts in adapter classes.
- **Location**: new package `app/roop_3d/orchestrator/` containing `engines_manifest.json`, `scene_rules.json` and the loader.
- **Models**: Engines reference Models by `model_id` only. Supported providers/precisions are derived as the intersection across the Engine's Models; the validator fails closed if an Engine declares anything a Model contradicts.
- **Evidence**: not in the manifest. Joined from the benchmark database at selection time by the exact-match key; the registry view shows joined evidence or "none".
- **Fail-closed**: malformed manifest / duplicate id / hash mismatch / Engine contradicting its Models -> whole registry invalid (503 `ORCHESTRATOR_REGISTRY_INVALID`). A well-formed Engine with an unregistered adapter or non-READY Models stays listed as `unavailable` with a reason code and appears as a rejected alternative in the report.
- **Versioning/hashing**: each file has `schema_version` and `content_version`; the selection snapshot records each file's sha256 (canonical JSON, sorted keys, compact separators) and the sha256 of every referenced `ModelRecord`. No `source_documents` pin.
- **Scene rules**: no capability tags (this supersedes the "capability tags" wording in the charting summary). Rules map a segment's traits to Stage 11 `SceneCategory`; a candidate plan is scored on that category's recorded scene metrics, else the overall composite with the report saying so. Engines declare only hard constraints, no soft suitability preferences.
- **Validation errors**: collected in full, ordered by Engine id then field; stable codes with Engine id and field path, no raw input echoed.
- **Residency**: Engine declares `heavy|light|none`; validator fails closed if it contradicts legacy `HEAVY_MODELS`/`LIGHT_MODELS` for any shared Model filename. Legacy sets untouched.
- **v1 Engine entry fields**: `engine_id`, `category`, `adapter_id`, `models[{model_id, role}]`, `inputs`, `outputs` (data kinds, optional shape constraints), `resolution{min,max}` (both-null forbidden for image-processing Engines), `precisions`, `providers`, `hardware_tiers` (subset of cpu/6gb/12gb), `declared_peak_vram_mib` (int|null), `residency`, `exclusive_with` (must be symmetric), `description` (never read by selection). `fallback` is deferred to ticket 05; no placeholder in schema v1. Unknowns are explicit `"UNKNOWN"`/`null`; a missing field is a schema error.
- **Identifiers**: Engine id `^[a-z0-9][a-z0-9_]{0,95}$`; must not equal any job-runner id (e.g. `reference_onnx`); the ten categories are a closed enum, adding one is a schema change.
- **Left open (new ticket 07)**: contents of the closed data-kind vocabulary and its shape constraints.
