# Remediation Stage 8: temporal curves and Face Banking

Implementation uses the standalone checkout's `app/roop_3d` and `app/roop_3d_ui` paths. The legacy `app/roop` and `react-ui` paths in the request are not used.

## Temporal filtering

`temporal_stabilizer.py` implements array-valued velocity-adaptive One-Euro filtering, Euler unwrap, trajectory/scene resets, and 2.5 degree pose-cell hysteresis. Standard pitch direction is preserved when indexing the Stage 0 matrix. Dense Stage 2 checkpoints contain filtered five-point landmarks; cache identity includes the filter configuration. Stream tracking supplies raw observations to avoid filtering twice. Reference render and preview supply decoded timestamps; direct callers without timestamps use the existing nominal 30 Hz fallback.

Settings expose `temporal_min_cutoff` (default 0.15), `temporal_beta` (0.04), `temporal_d_cutoff` (1.0), and `pose_hysteresis_degrees` (2.5). Per-track state is discarded when trajectories end. The Colour transfer selector enables normalized LAB histogram filtering before CDF transfer; the existing LAB-statistics path remains selectable.

## Stage 25 curve editor

The timeline edits TPS expression, pose frontalization, CodeFormer fidelity, swap opacity, mask feather, and identity blend, alongside existing restoration/geometry controls. Linear, Step, and cubic Bezier interpolation use exact source frame indices. Custom Bezier controls edit normalized value handles; time handles are fixed at one third and two thirds of each segment, so time is linear. Curve keys and controls survive project save/load.

The shared backend `ParameterCurves` compiles keys once into a fixed four-level radix predecessor index over bounded integer frame indices. Production frame evaluation performs bounded lookup and a cubic evaluation, without scanning keys or allocating an array proportional to video length. Preparation costs scale with key count.

Choose **Restoration model > CodeFormer**, enable restoration blend, then animate **CodeFormer fidelity**. The reference compositor maps measured crop landmarks to the CodeFormer training template and restores to the original crop coordinates. The existing lazy ONNX runner supplies `w` using the registered graph dtype/shape. Artifacts without a runtime fidelity input are rejected explicitly. No weights load merely to start the API. The local registry currently records `CodeFormerv0.1.onnx` with FLOAT image input `x` and scalar DOUBLE input `w`; artifact availability, activation rights and inference evidence remain separate.

Choose the normal source as A and **Identity B for animated Slerp**, then keyframe `identity_blend_t`. This explicit global recipe applies to selected swap tracks. Both real source vectors are retained for unit-length Slerp; endpoints are exact. Preview and render use the same recipe and evaluator. Projects retain the recipe and checkpoints fingerprint both source archives; changing B invalidates resume just as changing A does. Unsupported or antipodal endpoint recipes fail explicitly.

## Face Banking and archive studio

Opening `#/facemgr` discovers identities from observed inspection tracks. The bank samples original frames, computes local embeddings, sorts cosine matches, and reuses the closest qualifying identity with deterministic rank tie-breaking. Re-entry tracks can join an existing person across scene boundaries. Unresolved tracks remain reported rather than receiving fabricated embeddings.

Banks are stored atomically under ignored `.runtime/face_banks`, keyed by target SHA-256. Names, source assignments, explicit merges, ranks and retired-ID aliases survive reopening. Merge keeps the earliest discovery rank, never renumbers other people, and preserves measured observations. Conflicting source assignments must be cleared before merging. Revision checks and file locks prevent stale edits from overwriting newer ones. Source assignment applies to every original track in the person cluster. Imported source photos remain separate editable groups beside the target bank.

The source selector loads the saved `.fsz` archive into a nine-cell Yaw x Pitch matrix. Every cell shows actual support and its best view's measured quality/sharpness; empty cells remain empty. Positive standard pitch is up. **Fill Missing Angles from Current Target Video** scans decoded video frames, adds only observed views matching the selected source identity, preserves existing views, and reloads the archive matrix after completion. It stops at full coverage, EOF, or a reported resource budget (default 100,000 frames / 120 seconds; bounded API overrides are available). No synthetic unseen angles are represented as observations.

Source-photo analysis and archive building validate PNG/JPEG dimensions before native decoding. The builder accepts measured geometry/embeddings or performs local inference; it no longer invents image bytes, embeddings, landmarks or perfect quality scores. Export uses an atomic archive replacement and the existing validated archive reader/import pipeline. Archive import traversal/expansion controls remain in force.

## Verification and evidence

The required numerical pytest gate verifies more than 80 percent Gaussian landmark-noise standard-deviation attenuation at 120 Hz, an 80-unit turn over half a second with less than five units endpoint lag, both yaw boundaries with hysteresis, and exact arbitrary-frame Bezier values. Additional checks cover standard pitch direction, durable banking/re-entry/merges, archive fill and refresh, CodeFormer training alignment and scalar fidelity feeds, Slerp unit endpoints, preview/render parity, source-B drift, image budgets, and frontend request/refresh behavior.

Neural adapters in studio integration tests use explicit numerical fixtures; those tests establish contracts, not a neural fidelity or hardware performance result. The existing real golden-video campaign remains in the full suite. No new speed, fidelity, 6 GB or 12 GB acceptance claim is made.

## Completion record

Stage 8 software implementation and the requested verification gate are complete.

- Final full Python suite: **645 passed**, **18 subtests passed**, exit **0**, using checkout-local `pytest_stage8_release` temporary storage. This includes the existing real golden-video test.
- Stage 8 numerical/integration campaign: **26 passed**, exit **0** (16 numerical/pose checks plus 10 studio integration cases).
- Frontend: **224 tests passed**; typecheck, lint and production build exited **0**. Existing lint warnings remain (70 warnings, no errors).
- Read-only integrity audit: **2026-11-01T17:40:28.431638Z**, exit **0**. Historical baseline: **2,645 files**; comprehensive immutable baseline: **68,390 files**. Both matched with **zero violations**. Runtime activity was classified separately under the existing explicit policy.
- Verified local CodeFormer registry graph metadata exposes scalar DOUBLE `w`; studio tests exercise the real runner feed contract with a numerical model fixture. New hardware-specific output-quality/performance campaigns are not asserted by this completion record.

No legacy baseline was replaced, no sibling application state was modified, and no model weights, environments, biometric artifacts or caches were staged. Commit and push are handled separately on explicit user request.
