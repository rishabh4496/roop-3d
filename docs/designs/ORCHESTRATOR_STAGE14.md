# Stage 14: Intelligent Multi-Engine Orchestrator

Status: implemented as a planning and reporting layer. It adds no execution path and does not run jobs.
Code: `app/roop_3d/orchestrator/` · API: `app/roop_3d/server/orchestrator.py` · Tests: `app/roop_3d/tests/test_stage14_orchestrator.py` (35).
Vocabulary: `docs/GLOSSARY.md`. Decision history: `docs/wayfinder/stage14/` (map, tickets, two research reports).

## What it does

Given a media target, a hardware tier and a mode, it chooses one **Engine** per capability category, validates the
combination, admits it against a VRAM budget, and explains the choice. It never loads a Model, reserves VRAM or
starts work. The result is a **Pipeline plan** that compiles onto the existing runner parameters.

The prompt named no file paths, so there is no path mapping to apply. Everything lives in a new package.

## Vocabulary

| Term | Meaning |
|---|---|
| Model | A governed, hash-verified artifact (`ModelRecord`, Stage 12). |
| Engine | One selectable provider for one category: Models + adapter + declared contract. |
| Pipeline plan | One Engine per category for one scene. |
| Render profile | A named search space of allowed Engines per category (`render_profiles.json`). |
| Job runner | The existing whole-job executor (`engine/registry.py`). Not an Engine. |

The ten categories, in pipeline order: video decoding, face detection, face recognition, dense landmarks, identity
swapping, 3D reconstruction, expression transfer, restoration, occlusion segmentation, video encoding.

## Files

| File | Role |
|---|---|
| `engines_manifest.json` | 21 Engines with real adapters. Declarative facts only (contracts, tiers, providers, precisions, resolution, declared peak, residency, exclusions, fallback). |
| `render_profiles.json` | `baseline` (today's reference swap, pinned: one Engine per required category) and `adaptive` (the search space for Balanced/Quality). |
| `scene_rules.json` | Versioned trait thresholds (large yaw at 30 degrees) and the trait to Stage 11 `SceneCategory` map. |
| `schema.py` | Frozen pydantic contracts and the closed vocabularies (categories, data kinds, tiers, providers, reason codes). |
| `registry.py` | Loads and validates the manifests against `ModelRegistry`; adapter presence; per-Engine eligibility. |
| `plan.py` | Plans, the typed-chain/exclusivity/resolution checks, VRAM admission. |
| `scene.py` | Shots and traits from media-inspection tracks. |
| `evidence.py` | Evidence store and the exact-match rule. |
| `selector.py` | Manual, Balanced and Quality selection. |
| `compile.py` | Plan to `Settings`/`WorkstationParameters`, with the `unsupported` list. |
| `report.py` | Structured decision report, its digest, and a text rendering. |

## Registry rules

* Adapters are registered in code by id (`ADAPTERS`); capability facts are never in adapter classes. An adapter counts
  as present if its module defines the named class or function (checked without importing or loading anything).
* Engines reference Models by `model_id` only. The validator fails closed if an Engine contradicts a Model: a provider
  outside the Model's declared providers, a precision outside its declared set, FP16 on a model the precision policy keeps
  at FP32 (including INSwapper), or a residency class that contradicts `HEAVY_MODELS`/`LIGHT_MODELS`.
* Two failure levels. A malformed manifest, duplicate id, unknown Model, asymmetric `exclusive_with`, invalid fallback
  target, unpinned baseline or contradiction makes the **whole registry** invalid (`ORCHESTRATOR_REGISTRY_INVALID`, 503),
  with every issue collected. A well-formed Engine whose adapter is missing, whose Models are not Stage 12 `READY`, or
  which does not support the tier stays registered but **unavailable**; the reasons appear in the report.
* Engine ids use `^[a-z0-9][a-z0-9_]{0,95}$` and may not equal a job-runner id.
* Evidence is **not** in the manifest. It is joined at selection time.

## Selection

Selection is a pure function of a frozen snapshot, hashed into the report: the three manifest digests, the sha256 of
every referenced Model, the evidence set, mode, tier, profile, manual picks, the configured budget and reserve, the media
size and the scene segments. Live free VRAM is never an input. At run time it may refuse an admitted plan; it cannot
change which Engines a plan contains. There is no automatic downgrade. Ties break by a fixed order (the tuple of Engine ids).

**Manual.** The picks become the plan exactly. Every problem is reported together: unknown or out-of-profile picks,
duplicate categories, Models not `READY`, unsupported provider or precision, missing required categories, broken contract
chain, exclusivity, resolution, a plan the runner cannot express, and VRAM admission from **declared** peaks (labelled
`declared`). There is no force flag and no bypass of the Stage 12 gate. Nothing is substituted.

**Balanced / Quality.** Candidates are all complete plans in the `adaptive` profile whose Engines are eligible. Each is
checked structurally (typed chain, exclusivity, resolution, runner expressibility), matched to evidence, and admitted
using the evidence's **measured** peak.

* Quality picks the highest score, then lowest measured latency, then the fixed order.
* Balanced needs real evidence for the baseline plan to set a quality floor. It then picks the lowest latency among plans
  scoring at least the floor. Without baseline evidence it does not invent a floor.
* If no plan has applicable real evidence, the result is the **declared baseline** (`baseline_unscored`) only if the
  baseline is itself eligible, expressible and admitted from declared peaks. Otherwise the status is `INFEASIBLE`.
* The score of a plan for a shot is the worst of its recorded scores over the shot's Stage 11 categories, falling back to
  the overall composite with a note when the category score is missing.

### Evidence

An evidence record applies to a plan only if it matches **exactly**: tier, the plan digest (which covers every Engine's
manifest entry digest, Model sha256s, precision and provider), `measured_with == "real_models"`, and the safety vetoes
passed. A near-miss is `EVIDENCE_STALE`; a simulated record is `EVIDENCE_NOT_REAL`; none is `NO_EVIDENCE_FOR_PROFILE`; a
failed veto is `SAFETY_VETO`. Hashes must be 64-hex, which is why the two Stage 11 database records cannot be loaded as
evidence. Evidence lives in `app/roop_3d/.runtime/orchestrator/evidence.json` (git-ignored).

### Scenes

A shot is one `scene_id`. All tracks in a shot share one plan, because they share frames, residency and one runner
invocation. Plans can differ only between shots. Traits, v1: `large_yaw` when the inspection yaw extent reaches 30
degrees (labelled a candidate estimate; inspection stores only min and max, so one noisy sample can trigger it),
`frontal`, or `pose_unassessed`. Low-confidence frames use the existing tracker withhold (the original frame passes through),
identical for every plan. Occlusion difficulty is `not_assessed` because no signal exists; occlusion-aware segmentation
is therefore a manual or evidence choice, never an inferred one.

### Admission

* `cpu`: every Engine must resolve to a CPU provider; peak 0.
* `6gb`: sequential residency, peak = largest heavy Engine + all light Engines.
* `12gb`: co-residence, peak = sum of all.
* Peak plus the safety reserve (768 MiB default) must fit the configured budget (cpu 0, 6gb 4600, 12gb 12288 MiB unless
  overridden). An unknown peak is refused (`PEAK_VRAM_UNKNOWN`). Engines marked `exclusive_with` can never be resident together.
* Hardware codec VRAM is not accounted by the governor today; the report says so.

## Decision report

`roop3d.orchestrator.report.v1`: status, snapshot, per-Engine eligibility, and one decision per shot with the selected
Engine per category (provider, precision, reason code, declared fallback), score, evidence status, admission, the compiled
runner parameters and `unsupported`, problems, and every rejected alternative with a closed-enum reason code and detail.
No timestamps, so identical inputs give a byte-identical report; `report_digest` hashes it. `reason_vocabulary` lists the codes.

## Runner compilation

A plan compiles to `render_backend`, `face_restoration_policy` and `WorkstationParameters` fields (`render_mode`,
`pose_mode`, `occlusion_mode`, `restorer_model`, `tps_expression_weight`, `restoration_blend`). Plans the runner would
alter silently are marked `unsupported` and are treated as structurally invalid, so they are never selected:

* the reference runner always attaches the BiSeNet parser, so a reference plan needs an occlusion Engine;
* it retargets with LivePortrait by default, so omitting `liveportrait_retarget` requires a 3D projection Engine;
* the stream backend has no 3D, expression or occlusion Engines;
* TensorRT is a stream-backend provider only.

Strengths (`tps_expression_weight`, `restoration_blend`) default to 1.0 and are reported as notes. `HardwareCodec` is a
preference: the runner may fall back to software and records the state in its manifest; `ROOP_STRICT_NVDEC/NVENC` are
environment settings a plan cannot set.

## API

* `GET /api/v1/orchestrator/engines?tier=` the registry with per-tier availability.
* `POST /api/v1/orchestrator/plan` mode, optional tier (else the resolved hardware profile), manual picks, budget, media
  size and tracks; returns the report and its text rendering. Read-only.

## Deviations from the charting decisions

These were made while building; they should be reviewed.

1. **One plan per shot, not per track.** Tracks in one shot share frames and residency, so a per-track plan would need
   the union of Engines resident at once. The report lists the tracks in each decision.
2. **No capability tags.** Traits map to `SceneCategory`; plans are scored on that category's evidence (ticket 03).
3. **Evidence is keyed to the plan digest** (entry digests, Model hashes, precision, provider, tier) instead of a separate
   render-profile definition hash. It is stricter, and any manifest change invalidates it.
4. **The baseline includes `liveportrait_retarget` and the BiSeNet parser**, because the reference runner attaches both by
   default. It is today's behavior, not a minimal swap.
5. **Providers are checked strictly against `ModelRecord.supported_providers`.** That field lists CPU only for SCRFD, so the
   detector is declared CPU-only until provider evidence is recorded, even though TensorRT receipts exist.
6. **Fallback is declared as `none`, `withhold` or `degrade_to`.** Degradations to non-Engine code paths (the geometric hull
   mask when the parser fails) are described in the Engine text, not declared.
7. **Fallback target ordering:** a `degrade_to` target must be the Engine itself or an Engine in the same or an earlier category.
8. **Delegated decisions.** The user delegated the remaining open tickets to this implementation. The defaults recommended
   during grilling were adopted without further confirmation (tickets 04, 05, 07, 08).

## Current state and prerequisites

Against the shipped registry the orchestrator reports **`INFEASIBLE`**: no Model is Stage 12 `READY` (113 `LICENSE_RESTRICTED`,
8 `FAILED`), which was confirmed independently. This is the intended fail-closed behavior, and the report names the cause
per Engine. Making Balanced and Quality useful needs work that is **outside Stage 14**:

1. Complete the governance fields on `ModelRecord` (dependency compatibility, hardware, precision, provenance are derivable
   from the qualification receipts; license and security review are human attestations).
2. A harness that runs real Engines and writes `EvidenceRecord`s. The Stage 11 harness is simulated (pixel filters and
   `time.sleep`) and is not evidence.
3. Measured per-plan peak VRAM. None exists today; only legacy hand-written constants.
4. Physical 6 GB hardware evidence. The `6gb` receipts are a budget policy on a 12 GB card.
5. Declared peaks for `liveportrait_retarget`, `xseg_visible_face` and the stream restorers (currently `null`), which also
   blocks Manual admission of plans that contain them.
6. Wiring job submission to request orchestration. The compile step and the dry-run API exist; submitting a job from a plan
   does not, because the runner takes one `WorkstationParameters` per job and a multi-shot plan with differing plans is
   not expressible there yet.

## Acceptance

| Criterion | Where it is enforced and tested |
|---|---|
| Deterministic engine selection | frozen hashed snapshot, fixed tie-break, no live inputs; `test_selection_is_deterministic_*`, `test_ties_break_*`, `test_live_vram_never_changes_the_plan`, real-registry byte-identical report |
| No unsupported configuration | typed chain, exclusivity, resolution, tier/provider/precision checks, runner expressibility; manual and auto both; `test_manual_rejects_*`, `test_plans_the_runner_would_silently_alter_*` |
| No silent model substitution | no fallback plan other than the declared baseline, reported as `baseline_unscored`; infeasible requests return `selected: null`; `test_manual_reports_every_problem_and_never_substitutes`, `test_manual_cannot_override_the_governance_gate` |
| No uncontrolled VRAM allocation | per-tier residency schedule, declared or measured peak required, budget plus reserve enforced; the module never loads a Model; `test_vram_admission_*`, `test_unknown_or_oversized_*`, `test_a_plan_with_an_undeclared_peak_*` |
