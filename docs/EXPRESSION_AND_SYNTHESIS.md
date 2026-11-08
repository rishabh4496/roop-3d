# Micro-expression restoration, face editing and 36-view synthesis

The Face Swap inspector now contains **Micro-expression & 3D Face Editor**.
`expression_restore_factor` runs Stage 2.5 after the core swap/projection and before
UltraMax (or the selected enhancer), in both the standard and robust compositors.
It defaults to zero, which is an exact bypass with no additional model loads.
The same parameters reach live preview, frame preview, queued renders and exports.

The processor extracts motion from the current target and current swapped crop.
It applies the expression residual while preserving the swapped identity's canonical
21-point 3D geometry, scale, translation and head pose. It does not wait for a neutral
frame, so motion on the first frame is supported. Motion stitching and the existing
LivePortrait appearance/warping decoder generate the replacement.

The editor stores overrides by inspection track label, resolves those labels to the
persistent tracker identity, and resets only the selected face. Its controls are:

| Control | Range |
| --- | --- |
| Pitch / yaw / roll offsets | ±30° / ±45° / ±30° |
| Gaze X/Y, eyebrow tilt/raise, smile | −1 to +1 |
| Independent eye blinks, pout, mouth opening | 0 to 1 |
| Jaw width and chin length | −1 to +1 |

These are edits to implicit 3D portrait keypoints. Jaw/chin controls explicitly
deform the source canonical keypoints; ordinary expression restoration does not.
Gaze edits are artistic controls, not a calibrated iris tracker. The target's
implicit expression output carries fine motion, but exact pupil saccade recovery
and anatomical mesh measurements are not established by these models.

Six-point 68-landmark EAR is evaluated independently for each target eye. Below
0.16, its closure request is set to full strength even at a small positive restore
factor. An explicit aperture guard replaces the remaining eye opening with adjacent
identity skin and a lid seam. The guard also runs after enhancement and temporal
detail stabilization, preventing an enhancer from reopening that aperture. Closed
lips use a corresponding guard below a 0.02 aperture/width ratio unless mouth opening
is explicitly requested. Left/right refer to the image's left/right eyes. This skin
sampling can show artifacts on difficult occlusions or profiles; inspect the result.

## Local models and runtime

Required expression models are the registered local LivePortrait appearance feature
extractor, motion extractor, stitching, eye retargeter, warping/SPADE graph, and
`1k3d68.onnx`. The lip retargeter is optional: when unavailable, extracted expression
and the explicit lip guard still drive the mouth. Synthesis additionally needs the
existing face detector, 106-point landmark model, ArcFace recognizer and UltraMax's
CodeFormer model. No weights are downloaded by these actions.

Models use the existing verified catalog and ONNX runtime. TensorRT uses the existing
FP16 session builder and the selected 6 GB or 12 GB budget; CPU ONNX inference remains
available. An already prepared portrait session is shared with Stage 2.5 to avoid
loading a second set of LivePortrait weights. Synthesis runs one angle at a time,
uses the engine admission gate, releases its sessions, and renders 512px on the
6 GB profile or 1024px on the 12 GB profile. The mesh projector requires the existing
hardware OpenGL renderer even when neural inference uses CPU.

TensorRT's portrait generator uses a separate batch-one, shape-inferred derivative
after volumetric GridSample lowering. The reference model stays unchanged. Initial
TensorRT compilation can take several minutes; synthesis progress identifies model
preparation, generation, enhancement and archive verification separately.

The reference CUDA path allows CPU shape/control partitions and reports its provider
chain; whole-session runtime fallback stays disabled. This fixes the incompatible
combination of explicitly adding CPU while disabling CPU graph assignment. Strict
TensorRT builders retain their GPU-only policy. Profile allocation failures can
still select the existing reported CPU reference fallback; a CUDA request alone is
not evidence that every graph ran entirely on the GPU.

## One-click faceset synthesis

In **Face Manager**, import one to three source photos, select their person and use
**Synthesize Full 36-Angle 3D Faceset (.fsz)**. The same action is available for one
to three selected reference crops in the multi-angle builder. Progress is polled
through the existing builder job endpoint; successful output is saved and activated
automatically. An error is shown without publishing a partial archive.

The requested yaw lattice −75° through +75° at 15° intervals contains eleven angles.
At pitch −30°, 0° and +30° this gives **33 views**. The 36-view preset preserves every
one of these views and adds (yaw, pitch) (0°, −15°), (0°, +15°), and (+7.5°, 0°).
These intermediate frontal poses stay within the requested bounds. The supplemental poses are
recorded in the UI and archive rather than silently changing the requested spacing.
These are 36 distinct camera commands, not 36 independent measured observations.

The original photographs fit the head mesh. LivePortrait generates each posed
portrait from the closest original, UltraMax enhances it, and the posed mesh's UV
field is registered to that portrait. Each output includes its newly extracted
512-dimensional ArcFace embedding, landmarks and a generated UV texture with an
alpha support mask. Close crops receive extra detector context; all coordinates
are then mapped back into the unpadded crop.

The archive is reopened and validated before publication. It includes images,
per-view embeddings and UVs, the original reference crops, source mesh and albedo,
and `synthesis.json` with camera
commands, source selection, model evidence and identity confidence. `is_synthetic`
survives export, reload and pose selection; generated texture support is classified
as an appearance prior, and generated view count cannot establish triangulated
source depth. UV bytes and synthesis provenance survive a subsequent export.

Frontal identity drift, invalid embeddings, missing faces, model errors or invalid
archives fail the job. Extreme self-occluded views can have poor ArcFace alignment:
their actual similarity is retained, lowers their quality score, and is labelled
`low_confidence_generated_prior` below 0.45. Such views do **not** prove identity
fidelity or recover genuinely observed hidden detail from a single photograph.

## API

- `POST /api/v1/faceset/synthesize-36`: builder `job_id`, `cluster_rank`, one to three
  `selected_ids`, and `name`.
- `POST /api/v1/faceset/synthesize-photos`: one to three base64 `images` and `name`.
- `GET /api/v1/faceset/extraction/{job_id}`: progress, error or saved result.
- Workstation/live preview: `expression_restore_factor` and
  `face_expression_overrides: { "track_0000": { "head_yaw": 12, "blink_left": 1 } }`.
- Live preview `solo_node: "expression"` exposes the Stage 2.5 buffer when enabled.

The inference architecture builds on the existing adapter for
[LivePortrait](https://github.com/KwaiVGI/LivePortrait) and its
[stitching/retargeting wrapper](https://github.com/KwaiVGI/LivePortrait/blob/main/src/live_portrait_wrapper.py).

## Verification

`test_expression_and_synthesis.py` covers identity/pose invariants, first-frame winks,
zero-factor bypass, per-face routing, enhancement order, post-enhancement closure,
validation, the exact pose lattice, UV support and complete 36-view archive export.
`expressionEditor.test.tsx` covers face scoping, reset, live payload parity and the
one-click import/poll/activation flow. Real-model acceptance results depend on the
installed models, execution provider and source photograph; deterministic tests do
not establish visual accuracy or TensorRT performance.

### Local acceptance, 2026-11-09

- Production UI build, TypeScript and lint passed; all **546 frontend tests** passed.
- **116 focused backend regressions** passed. The additional model/contract group
  passed **40 tests**, overlapping the new processor tests.
- A real **36-view archive** completed under the **6 GB CUDA profile** on an
  RTX 4070, including UltraMax, per-view embeddings/UVs and archive readback. This
  tests the profile on a 12 GB physical card, not a separate 6 GB device.
- A real **12 GB profile TensorRT FP16 expression run** completed with TensorRT
  selected for appearance, motion, generator, stitching and eye networks, with no
  provider fallback. TensorRT/CUDA partitioning is permitted inside those sessions.
- The initial broad backend run finished with **1,689 passed, 30 skipped and 10
  failed**. The old synthetic-only-false assertion was updated and passes. Startup,
  import-order and CUDA resource tests passed isolated reruns. Five legacy
  reenactment failures reproduced against the unchanged identity-asset implementation;
  the existing job-progress assertion still fails. The full legacy suite is not
  being represented as green.

The acceptance archive remained in ignored local runtime storage and was not
activated in the user's library. Generated profile identity confidence was low in
some views; the recorded scores and quality weighting retain that limitation.
