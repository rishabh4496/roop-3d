---
label: wayfinder:map
title: Stage 14 - Intelligent Multi-Engine Orchestrator
tracker: local-markdown
---

# Map: Stage 14 - Intelligent Multi-Engine Orchestrator

Tracker layout: this file is the map. Tickets are `tickets/NN-slug.md` (child issues). Frontmatter `status` is open|closed, `assignee` is the claim, `blocked_by` lists ticket files. Research output lands in `research/`. Vocabulary: `docs/GLOSSARY.md`.

## Destination

A locked Stage 14 design spec at `docs/designs/ORCHESTRATOR_STAGE14.md` (plus `docs/GLOSSARY.md`) that a normal implementation prompt can build from with nothing left to decide. Charted as planning only; the user then delegated the rest ("complete stage 14, do what you think best"), so the spec was written alongside an implementation in `app/roop_3d/orchestrator/` (35 tests). Status: destination reached; remaining work is the prerequisites listed in the spec.

## Notes

- Domain: roop-3d backend `app/roop_3d/` (paths in the original Stage 14 prompt map onto this layout; the prompt named none).
- Consult `docs/GLOSSARY.md` for Model / Engine / Pipeline plan / Render profile / Job runner. "Engine" never means job runner or TensorRT plan here.
- Standing rules settled while charting (see "Settled while charting" below): they bind every ticket; do not re-litigate them.
- Honest-scope precedent (Stage 4): anything a plan requests that the runner cannot do goes in an `unsupported` list, never faked or dropped.
- Do not commit unless the user asks.

## Settled while charting

- Registry holds only Engines with a real adapter; capability registry sits on top of `ModelRecord` (models/registry.py), it does not replace `ModelRecord`, `UnifiedModelRegistry` or `EngineRegistry`.
- Balanced/Quality differ only where a category has more than one validated candidate; the report says so when they do not. No placeholder Engines.
- Validated (auto modes) = Stage 12 lifecycle READY + precision admitted by precision policy + exact-match benchmark on the target profile (all Model sha256, precision, provider, hardware tier, render-profile definition). Manual may use unmeasured Engines if READY and within budget, reported `unmeasured`. No qualifying Engine in a category -> declared baseline render path, reported.
- Engine choice fixed per (track, scene segment); fallbacks declared inside the plan; selection never re-runs mid-segment; changes only at segment boundaries (hard cut or minimum dwell + hysteresis), logged.
- Selection is a pure function of a frozen, hashed snapshot (registry, mode + manual picks, hardware tier + configured budget not live VRAM, inspection summary, scene-rule-table version); fixed tie-break order; live VRAM may refuse a plan, never change it; no automatic downgrade.
- Quality = highest Stage 11 composite among admissible plans (vetoes pass; tie -> lower latency). Balanced = lowest latency among admissible plans with composite >= baseline and vetoes pass.
- Admission wraps `VRAMGovernor`/`AdaptiveModelPool`; 6 GB one heavy Engine resident at a time; 12 GB co-residence within configured budget; each Engine declares peak VRAM (measured, or for Manual only declared `recommended_vram_gib` labeled so, else refused); `exclusive_with` supported; legacy jobs unchanged.
- (Refined by ticket 03: no capability tags; traits map to Stage 11 `SceneCategory` and plans are scored on that category's recorded metrics.) Scene traits from a fixed versioned rule table in registry config; yaw labeled `candidate estimate`; occlusion `not assessed` -> occlusion-aware segmentation manual-only.
- Decode/encode are Engines; strict NVDEC/NVENC flags become Manual constraints; CPU fallback only if declared in the plan, use recorded.
- Engine contracts typed over a closed vocabulary of data kinds; plan validated as a typed chain; no auto-conversion.
- Report: structured JSON + rendered view; one decision per (track, scene segment); closed-enum rejection codes; Manual lists all failures; no timestamps; hashed into job provenance.
- Plans compile to the existing execution path (`render_mode`, enhancers, swapper) with an `unsupported` list; dry-run endpoint; job submission may request orchestration.

## Decisions so far

<!-- one line per closed ticket: [title](tickets/file.md): gist -->

- [Registry manifest shape](tickets/03-registry-manifest-shape.md): declarative JSON manifest + code-registered adapters in new `orchestrator/` package; Engines reference Models by id only; evidence joined not stored; malformed manifest = whole registry invalid, non-READY Engine = listed `unavailable`; no capability tags (category-scored evidence instead); v1 field list and id rules fixed; `fallback` deferred to ticket 05.
- [Evidence audit](tickets/02-evidence-audit.md): real evidence is only per-Model FP32 receipts (10 Models, "6gb" = budget on a 12 GB card, no cpu); the Stage 11 harness is simulated and its DB records invalid; 0 of 121 Models reach READY, so the settled Validated gate fails for everything today (see ticket 08).
- [Engine inventory](tickets/01-engine-inventory.md): no per-capability Engine registry exists today (only the job-runner one); every neural Engine is wired but refused at construction by the READY gate, non-neural ones (pose blend, TPS, reenactment, PyAV/FFmpeg codecs) are not gated; no measured per-Engine peak VRAM; `ModelRecord` lacks VRAM/precision/provider data and roles for dense landmarks, codecs and 3D; candidate 19-kind data vocabulary and "not a real adapter" lists in `research/01-engine-inventory.md`.

- [Validated gate versus reality](tickets/08-validated-gate-vs-reality.md): zero-READY is the intended fail-closed state; only real-model evidence with valid hashes counts; Manual cannot override the gate; if even the baseline cannot run the status is INFEASIBLE with a full report; acceptance proven on fixture registries plus one real-registry fail-closed test.
- [Plan-to-runner mapping](tickets/04-plan-to-runner-mapping.md) (delegated): plans compile to `render_backend`, restoration policy and `WorkstationParameters`; requests the runner would silently alter are `unsupported` and unselectable; dry-run API exists, job submission not wired.
- [Segment parameters and fallbacks](tickets/05-segment-parameters-and-fallbacks.md) (delegated): a shot is the segment, no dwell or hysteresis; all tracks in a shot share a plan; low confidence = existing tracker withhold; fallback modes none / withhold / degrade_to; large_yaw at 30 degrees.
- [Data-kind vocabulary](tickets/07-data-kind-vocabulary.md) (delegated): 13 closed kinds, typed chain in category order, built-in compositor derives `composited_frame`.
- [Selector prototype](tickets/06-selector-prototype.md): superseded by the implementation; on shipped data every mode is INFEASIBLE.

## Open tickets (index; the ticket files are the truth)

None. The six prerequisites in the spec's "Current state and prerequisites" section are follow-on work outside this map.

## Not yet specified

- 6 GB evidence strategy: only 12 GB-card budget runs exist; physical 6 GB evidence is a prerequisite.
- Robust yaw statistic (for example p95 of |yaw|) from media inspection, which currently stores only min and max.
- Evidence-refresh workflow when a Model hash, provider or Engine entry changes (the digest invalidates old records; producing new ones needs a real-run harness).
- Wiring orchestrated job submission to the runner (one `WorkstationParameters` per job vs plans that differ between shots).
- ADR candidate: "selection is pure; live VRAM refuses but never changes the plan" (hard to reverse, surprising, a real trade-off). Not written; offer it when asked.

## Out of scope

- A UI panel for the decision report (closed by user at charting).
- Adding new Engines or Models.
- Community or uploaded Engines (bridges/cloud).
- Automatic benchmarking infrastructure.
