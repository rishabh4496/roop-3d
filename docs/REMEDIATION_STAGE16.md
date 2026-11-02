Stage 16 is **incomplete**. The required acceptance command currently reports
two passed tests and one failed test, exit 1. Independent eyeball gaze on the
rendered swap has not been established. No Stage 16 completion, likeness,
performance, 6 GB acceptance, or Linux hardware-execution claim is made.

All implementation is under the standalone `app/roop_3d/` package. The requested
`app/roop/` paths would introduce a different package outside the owned Python
layout. The requested hardware gate is at `app/tests/test_flame_and_liveportrait.py`;
software tests are at `app/roop_3d/tests/test_flame_and_liveportrait.py`.

| Component | Implemented behavior | Current evidence / limitation |
| --- | --- | --- |
| FLAME extraction | Named-output ONNX SMIRK adapter, explicit EMOCA output mapping, beta100/psi50/jaw/SO(3) validation | Tested with an executable synthetic export; real SMIRK/EMOCA ONNX weights are absent |
| Dense mesh | Optional, lazy MediaPipe FaceLandmarker 478 adapter with checked task identity | MediaPipe and its reviewed task are absent; 3DDFA's different topology/basis is not relabelled as FLAME or 478 points |
| Skull retention | Bounded `Stage16Options.shape_retention_weight`, `(1-w)*beta_source + w*beta_target` | 0 retains source, 1 retains target. A licensed geometry conditioner is required to change pixels. No active UI slider is connected to the reference renderer |
| LivePortrait | Actual appearance, motion, stitching, eye-open and converted generator session orchestration | All five modules executed on CUDA with native-only profiles on this Windows RTX 4070 |
| Volumetric sampling | Offline ONNX lowering, two 4D spatial samplers plus depth interpolation per original 5D node | Both real generator nodes lowered; actual full generator executed on CUDA and TensorRT 10 without CPU fallback |
| HifiFace256 | Exact paired ArcFace converter, converter-then-normalize identity input, RGB normalization, image and mask output | Actual 256x256 inference and identity-dependent output executed on CUDA using synthetic input fixtures; no likeness assessment |
| StyleGANEX | Separate 256 face/384 context inputs, continuous direction, preflight re-embedding or post-swap operation | Adapter only; weights, measured year calibration and output acceptance absent |
| Independent gaze | Four separate head-local angles, calibrated estimator/retargeter export interfaces, output-image gate | Trained calibrated exports and labelled examples absent; eye-open weights do not meet this requirement |
| HF catalogue | Observed SHA-256/size and revision-pinned URLs for the requested eight artifacts and six LivePortrait modules | Byte provenance only. Graph, weight rights, inference, VRAM and performance remain separate evidence states |

SMIRK's default shape vector has 300 ordered components; this adapter retains
its first 100. Its weak-perspective camera is preserved separately. It does not
become metric translation, and eyelid coefficients do not become eyeball angles.
The expression vector retains 50 coefficients; jaw and gaze are separate fields.
The existing FLAME license gate and preserved reference inference are unchanged.

HifiFace's exported runtime inputs are an image and identity embedding. Geometric
supervision during training does not expose a runtime FLAME-beta input. The
optional crop pipeline therefore refuses nonzero skull preservation unless an
actual geometry conditioner is supplied. Likewise, upstream StyleGANEX's
direction slider is not a calibrated years scale: `age_offset` in -30..+40 years
requires a supplied measured calibration. No conversion to years is fabricated.

The optional crop pipeline is an explicit API for prepared adapters; the existing
preview/render session remains on its preserved reference path. Activating this
pipeline in the workstation UI and production scheduler remains outstanding.

The graph transformation tiles input depth planes along height, with a zero
guard row above and below every plane. It samples the adjacent depth planes with
ordinary 4D `GridSample`, masks invalid depth indices and linearly interpolates.
It preserves `align_corners` and zeros/border padding. It rejects unsupported
interpolation, reflection, unknown ranks, custom operators, control-flow graphs
and external tensor bundles. It preserves already-4D samplers. Synthetic
parity checks cover batch 1/2, singleton depth/height/width, both corner modes,
both supported padding modes, boundaries and out-of-volume coordinates.

The conversion CLI is `tools/convert_dynamic_shapes.py MODEL --output DERIVATIVE
--grid-sample-4d`. It preserves the source and refuses to overwrite an existing
output. Conversion alone prints `provider_parity=UNVERIFIED`. Native acceptance
requires actual execution profiles; merely listing an available provider or
finding an engine file is insufficient. No custom DLL/plugin is loaded.
The optional runtime prepares a content-addressed derivative with source-digest
provenance under `app/roop_3d/.runtime/stage16/`.

Observed Windows RTX 4070, 12 GB profile evidence:

- Full converted generator: CUDA 396 kernels, TensorRT one native subgraph.
  Maximum output difference for the gate's synthetic inputs was
  `3.454089164733887e-05`, below its declared `2e-4` absolute/relative limits.
- Comparison with the independently preserved decomposed `-trt.onnx` generator
  on the same inputs: maximum absolute difference `2.1636486053466797e-05`.
- HifiFace and the exact `arcface_converter_hififace.onnx`: both native CUDA;
  changing the input identity embedding changed actual output pixels.
- Full modular CUDA smoke: appearance 46 kernels, motion 926 across two calls,
  generator 396, stitching 7, eye-open 22 across two calls.
- The 6 GB and 12 GB workspace caps are separate policies (256 and 512 MiB);
  this is not a measurement of total model VRAM or a 6 GB acceptance campaign.

The exact converter was fetched for this explicit task (20,993,559 bytes,
SHA-256 `34255826c8da689858a7d3c1615cde2df0e115674a0136e05296a1291b31cf66`).
The existing HifiFace checkpoint and LivePortrait modules were read through the
preserved hashed registry. The StyleGANEX 1.34 GB checkpoint was not downloaded.
Supplemental weight rights remain `UNKNOWN`; automatic application downloading
and commercial activation remain gated by the existing authoritative policy.
No code-license assumption is used to grant checkpoint rights.

To run the requested gate from the checkout root:

```powershell
& app/roop_3d/.venv/Scripts/python.exe -B -m pytest app/tests/test_flame_and_liveportrait.py -c app/roop_3d/pyproject.toml --basetemp=app/roop_3d/.runtime/stage16-gate -q
```

The gaze test deliberately fails when assets are unavailable; it never skips or
substitutes a parameter-copy test. `ROOP3D_STAGE16_GAZE_CASE` must name an owned
JSON case containing:

- `flame_encoder`, `gaze_estimator`, `gaze_retargeter`, `mediapipe_task`: artifact
  objects with `path`, observed `sha256` and `size`.
- `flame_family` (`smirk` or `emoca`), optional `flame_output_names` for a reviewed
  named-output export, and `gaze_calibration_evidence`.
- `tolerance_radians` greater than zero and at most 0.1, an engineering acceptance
  ceiling, not an observed accuracy claim.
- At least two `examples`, each with owned NPY paths for `source_rgb256`,
  `target_rgb256`, `source_embedding`, and four independently annotated
  `expected_target_gaze_radians`. Images must be RGB uint8; embeddings must have
  512 components. Include varied and asymmetric left/right gaze.

The gaze estimator's explicit export contract is RGB `[1,3,224,224]` in [0,1],
normalized `mesh478` `[1,478,3]`, head `rotation` `[1,3,3]`, and `angles` `[1,4]`
in head-local radians, left yaw/pitch then right yaw/pitch. The calibrated
retargeter takes `source_gaze`, `target_gaze` `[1,4]` and `points` `[1,21,3]`,
returning a `delta` `[1,21,3]` in LivePortrait canonical units. These are adapter
contracts for reviewed exports, not claims that such pretrained models ship
upstream. The gate re-estimates the target and rendered swapped image, comparing
both against the independent labels. Identity/vector round-tripping alone cannot
pass it. MediaPipe remains optional; install it only into the dedicated venv
when activating the reviewed task, not merely to boot the API.

Software verification uses the full AGENTS.md command from `app/`:

```powershell
& roop_3d/.venv/Scripts/python.exe -B -m pytest -c roop_3d/pyproject.toml --confcutdir=roop_3d roop_3d/tests -q
```

The final read-only legacy integrity audit passed at
`2026-11-02T08:41:37.672426+00:00`, exit 0: 2,645 historical and 68,390
comprehensive immutable files matched, zero violations. Its existing policy
classified runtime activity separately. The baseline was not replaced. The
24 new software checks passed; the focused model/API/Stage 16 run passed all
32 tests, and the four Git-boundary tests passed. The Git guard admits only the
exact requested Stage 16 acceptance filename under `app/tests/`, leaving other
unapproved files blocked.

The final complete local suite run reported **708 passed, 5 failed, 7 skipped,
18 subtests passed**, exit 1 (393.79 seconds). Four failures in the unchanged
`test_nv_cvcuda_pipeline.py` require the absent optional `torch` package. The
unchanged Stage 15 far-cheek raycaster timing assertion observed **1.1702 ms**
against its **<0.8 ms** limit. Those failures were not skipped or relaxed. The
requested Stage 16 hardware gate separately reported **2 passed, 1 failed**,
exit 1 (13.81 seconds): only the unavailable calibrated output-gaze case failed.
Both native generator and actual HifiFace/converter tests passed. The last source
policy adjustment was additionally verified with all four Git-boundary tests.

Machine-local generator and HifiFace acceptance JSON files are under
`app/roop_3d/.runtime/stage16/`; the referenced native profiles are under
`cache/stage16/providers/`. These ignored runtime artifacts, model weights,
environments and media are not source changes. Committing this partial work does
not establish Stage 16 completion; the missing-gaze acceptance failure remains.

Upstream contracts consulted:

- [SMIRK encoder](https://github.com/georgeretsi/smirk/blob/main/src/smirk_encoder.py)
- [FaceFusion 3.1 identity preprocessing](https://github.com/facefusion/facefusion/blob/3.1.0/facefusion/processors/modules/face_swapper.py)
- [FaceFusion 3.1 StyleGANEX direction and alignment](https://github.com/facefusion/facefusion/blob/3.1.0/facefusion/processors/modules/age_modifier.py)
- [LivePortrait reference orchestration](https://github.com/KlingAIResearch/LivePortrait/blob/main/src/live_portrait_wrapper.py)
- [Official models 3.0.0 revision](https://huggingface.co/facefusion/models-3.0.0/tree/728b9659bd9691bf32cbf7f61af478d94b7ba81e)
- [Official models 3.1.0 revision](https://huggingface.co/facefusion/models-3.1.0/tree/c9e3a503d8e84e91c5cd89ee2d510fe5e793e570)
- [Official models 3.2.0 revision](https://huggingface.co/facefusion/models-3.2.0/tree/f9cbb5a090256d304f662b81b85880853a9d62cc)
- [Official models 3.3.0 revision](https://huggingface.co/facefusion/models-3.3.0/tree/6271ab1c4619ceb2e534d65e0ec973f22a9aec38)
