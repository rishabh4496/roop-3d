# Stage 5: Advanced Occlusion and High-Fidelity Compositor

**Document Path:** `app/roop_3d/docs/OCCLUSION_STAGE5.md`
**Scope:** keep everything that is physically in front of a replaced face (hands, hair, glasses, props, a nearer face) in
front of it, stay temporally stable while doing so, and fuse the swapped face without halos, over-smoothing or damaged
eyes and teeth. Delivered as an opt-in compositor strategy (`occlusion_mode = "robust"`) inside the **existing**
`FrameCompositor`; the standard path is byte-for-byte unchanged.

> **Read this first.** There is no hand, microphone or object *class* model installed locally. Foreground objects are found
> by two "visible face" networks (XSeg, FaceFusion face occluder) that answer *"is this pixel visible face?"*, so anything that
> hides the face is foreground, but the system cannot say *what* it is. Teeth are **not** separable by the parser; a labelled
> heuristic only protects them from restoration. Face depth ordering is a face-size heuristic, not a depth model. Every
> number below comes from real local video and real local inference; where there was no footage, the scenario is reported
> `NOT_COVERED` rather than simulated. The acceptance criteria are **partly** met; see section 7.

## 0. Path mapping (the prompt names the real modules, but two claims needed correcting)

| Prompt | Reality |
| :--- | :--- |
| `compositor/{pipeline,masks,occlusion,boundary_fusion,color_match,lighting,temporal,stream_restoration}.py` | exist and were inspected; extended, not duplicated (`fuse_warped` is shared by both paths) |
| `masking_engine.py` | a mask-*combination helper* that is **not on the compositor call path** (only `tests/test_model_registry.py` imports it). It loads no model; its "CLIPSeg" layer draws a caller-supplied rectangle (`simulated_activation_box`) and there is no CLIPSeg runtime in the repo. Documented in its docstring; behaviour untouched (a test pins it). |
| "validated local segmentation system" | BiSeNet 19-class, `xseg_3` / `xseg` (same weights, renamed I/O) and `face_occluder`, all hash-registered and now **run and measured** (section 2) |

## 1. What was wrong with the existing path (measured, not assumed)

> The numbers in this section come from exploratory samples of face crops from the local videos (40 to ~285 crops each, not retained). The
> direction of each finding is re-checked by `tests/test_occlusion_real_models.py`; the exact figures are not reproducible from the repository.

1. **The parser sees a 128 px crop.** The replacement crop is the inswapper's 128 px aligned crop; BiSeNet was fed that
   upsampled to 512, and every mask was authored at 128 px, so occluder edges were blocky before they were ever warped.
2. **BiSeNet preprocessing was an unverified "candidate".** On 40 real crops, agreement (IoU) with the independent XSeg face
   region was **0.478 mean / 0.622 median** for the shipped `zero_one` normalisation versus **0.692 / 0.901** for `imagenet`.
   The standard path keeps `zero_one` (compatibility); the robust path uses `imagenet`.
3. **The parser fails exactly where compositing is hardest.** On side profiles and heavily blurred faces BiSeNet returned
   (almost) no face class while XSeg segmented them correctly. The standard path then intersects the hull with that empty
   face, so the swap **vanishes and re-appears** frame to frame (profile clip, flicker 4.5 to 6.2 versus 0.05 now).
4. **Collapsed landmarks.** On a head rolled 90 degrees the 106-point hull collapses to a small cluster and only a patch of the
   face is painted.
5. **XSeg and the face occluder are equivalent**, not complementary: on 285 real crops their masks had median IoU 0.92, equal
   fragmentation (1.2 components, 0.13 holes) and equal agreement with BiSeNet (0.81). Disagreement lives in the hard tail
   (10th percentile IoU 0.69), which is exactly where "keep the original pixels" is the safe choice.

## 2. Architecture

```
frame + replacement crops (128 px) + 106 landmarks + manual masks
   │ for every face (analysis pass, 256 px crop through the SAME geometry; out-of-frame pixels get validity 0)
   ▼
OcclusionAnalyzer ── BiSeNet(imagenet) ─► named regions (skin, brows, eyes, nose, mouth interior, lips, ears, hair, neck, cloth, glasses, hat)
   │              └─ XSeg ∧ face_occluder  ► "visible face" = min(a, b)      (disagreement ⇒ keep the target's pixels)
   │   trust checks: parser vs networks, hull vs networks, per-track hysteresis (3 frames to regain trust)
   ▼ FaceAnalysis{ face_alpha, occluder, skin, regions, protect, tier, confidence, reasons }
OcclusionTracker ── DIS optical flow (current→previous), flow-warped history only where the model is UNCERTAIN,
   │                scene-cut / frame-gap / appearance-cut reset, bounded hold, bounded release
   ▼
depth order (back→front) + nearer-face heads added as occluders of farther faces
   ▼ per face (composite pass)
restoration (protect map keeps eyes, brows, lips, mouth, teeth) → mouth → LAB colour → local illumination → TemporalSmoother(face alpha)
   ▼
fuse_robust: warp → guided matte refinement on the FULL-RES frame → edge decontamination (geometric edge only)
             → defocus / motion-blur matching → target fine-texture transfer (skin only) → fuse_warped (linear-light feather | multi-band | Poisson)
```

### Tiers (never reported stronger than what ran)

| Tier | Meaning | Occlusion evidence |
| :--- | :--- | :--- |
| `full` | parser trusted + at least one network | parser classes ∪ networks (interior-weighted) ∪ manual mask |
| `semantic` | parser only | parser classes ∪ manual mask |
| `neural` | parser distrusted or still warming up | networks define shape and occlusion (or, if the landmark hull collapsed, the network face region is the shape) |
| `geometric` | nothing usable | **none** — `OCCLUSION_UNVERIFIED`, confidence ≤ 0.3, hull only; never an invented foreground |

Reasons recorded per face: `PARSER_DISAGREES_WITH_NETWORKS`, `PARSER_LOW_CONFIDENCE`, `PARSER_TRUST_WARMUP`,
`HULL_IMPLAUSIBLE_USING_NETWORKS`, `NETWORKS_UNAVAILABLE`, `OCCLUSION_UNVERIFIED`, `FACE_MOSTLY_HIDDEN`,
`PARTIALLY_OUTSIDE_FRAME`, `OCCLUDED_BY_NEARER_FACE`.

### New and changed files

| File | Role |
| :--- | :--- |
| `compositor/occlusion_model.py` | `VisibleFaceNet`: lazy, hash-checked XSeg / face-occluder adapter; capability reports weights / inference as separate states |
| `compositor/face_regions.py` | named regions, restoration protection map, heuristic teeth estimate |
| `compositor/occlusion_analysis.py` | `OcclusionAnalyzer` → `FaceAnalysis` (unified face + occlusion mask, tiers, confidence) |
| `compositor/occlusion_tracker.py` | flow-aware temporal matte, resets, hold / release |
| `compositor/matte.py` | guided-filter matte refinement, edge decontamination |
| `compositor/quality.py` | local illumination, sharpness / motion-blur matching, fine-texture transfer, linear-light blend |
| `compositor/robust_fusion.py` | frame-space fusion for the robust path |
| `compositor/pipeline.py` | `occlusion_mode`, `RobustOptions`, shared `_enhance`, depth-ordered robust path, per-track `last_reports`, capability |
| `compositor/boundary_fusion.py` | `fuse_crop` split into warp + `fuse_warped` (identical output; `linear_light` option) |
| `compositor/restoration.py` | `blend_restored_roi(..., protect=)` |
| `core/workstation.py` | `occlusion_mode` (`standard` default, omitted from stored requests unless set) |
| `app/roop_3d_ui/...` | "Occlusion handling" selector in the Masking panel + type |
| `tools/occlusion_acceptance.py` | real-footage harness (identity-neutral probe) |
| `masking_engine.py` | docstring only: states what it is not |

## 3. Quality features: what each does and does not do

| Feature | Behaviour | Limits |
| :--- | :--- | :--- |
| Matte refinement | guided filter on the full-resolution frame snaps the occluder edge to real edges; then plateau tightening (`matte_contrast`) removes the faint halo the filter keeps | thin low-contrast fingers on skin have weak edges; the filter cannot create foreground that was not detected |
| Edge decontamination | outer feather (alpha ≤ 0.3 fully, 0.8 → 0 ramp) takes the colour propagated from the opaque interior | defined on the *geometric* edge only (a regression test pins this: using the post-occlusion alpha flattened partially occluded interiors) |
| Local illumination | spatial part of the target/source lightness ratio, L channel only, global gain left to the LAB step | low-order (σ ≈ 16 % of crop); no cast-shadow model |
| Sharpness / motion blur | blurs the swap only as much as the target is blurred (isotropic σ search + directional kernel from the **face's frame-pixel velocity**); never sharpens | needs ≥ 2 consecutive frames for the directional part |
| Skin texture, pores, grain | adds the **target's own** fine luminance band (σ 0.8–2.2 px) on skin, only up to the energy deficit; a smooth or motion-blurred target contributes nothing | skin map needs the parser to be trusted; no synthesis of detail that the target does not have |
| Restoration guard | eyes, brows, lips, mouth interior and the teeth estimate are excluded from the restorer's blend | verified with the real GPEN weights; CodeFormer / GFPGAN paths use the same `blend_restored_roi` |
| Linear-light blend | feather blend of the boundary in linear light | multi-band and Poisson keep their own maths; LAB colour matching is still 8-bit (not changed) |

## 4. Capability states

| Component | Weights | Inference | Provenance / licence |
| :--- | :--- | :--- | :--- |
| `bisenet_resnet_34.onnx` (imagenet RGB) | VERIFIED_LOCAL | VERIFIED_LOCAL (this stage) | UNKNOWN |
| `xseg_3.onnx`, `face_occluder.onnx` (256² NHWC BGR/255, 1 = visible face) | VERIFIED_LOCAL | VERIFIED_LOCAL (this stage) | UNKNOWN |
| teeth | n/a | heuristic only | n/a |
| object / hand / microphone classes | none installed | n/a | n/a |
| CLIPSeg | only `rd64-uni-refined.pth` weights, no runtime | none | UNKNOWN |

Nothing is downloaded. Missing networks lower the tier and are listed in `capability()["robust_occlusion"]["unavailable"]`.
The BGR channel order of the networks follows their documented consumer; on the 25 real crops tried, BGR and RGB gave the same
agreement with BiSeNet, so the order is a convention, not a measured preference.

## 5. Evidence on real local video

Harness: `python -m roop_3d.tools.occlusion_acceptance --out roop_3d/.runtime/acceptance/occlusion_stage5` (CPU provider, 10 frames per
clip, real SCRFD tracker + real BiSeNet / XSeg / face occluder). The replacement is an **identity-neutral probe** (the face's own aligned
crop, green tint, colour matching off), so every tinted output pixel is a pixel the compositor chose to paint. No identity is transferred.
Only non-explicit local clips are used; the local media folder also holds adult videos that were deliberately excluded.

Metrics (lower is better; arrows read *standard → robust*). They are **proxies, not ground truth**: the face-region reference uses the same
model families, so `leak` and `occluded px painted` measure end-to-end self-consistency (matte refinement, temporal logic, fusion, depth
order). `flicker / face area` is flow-compensated change of the painted mask normalised by a mode-independent area (a mode that paints less is
not penalised); `flicker / painted` normalises by what each mode painted and is dominated by pop-in of the standard path.

| Scenario | Clip (local media) | Leak outside face | Occluded px painted | Flicker / face area | Flicker / painted | Robust tiers | Temporal resets |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `hand_over_face` | `5256…` f10800+10 | 0.064 → **0.067** | 0.941 → **0.000** | 0.108 → **0.075** | 0.215 → 0.386 | neural:2, full:8 | - |
| `hair_in_front_of_face` | `5256…` f13200+10 | 0.060 → **0.003** | 0.378 → **0.257** | 0.144 → **0.149** | 0.658 → 0.705 | neural:12, full:2 | - |
| `glasses` | `5256…` f7800+10 | 0.013 → **0.000** | 0.159 → **0.005** | 0.095 → **0.074** | 0.179 → 0.220 | neural:4, full:10 | SCENE_CUT:2 |
| `fast_movement` | `5256…` f3900+10 | 0.036 → **0.008** | 0.120 → **0.064** | 0.181 → **0.170** | 0.465 → 0.443 | neural:8, full:5 | - |
| `microphone_crossing_mouth` | NOT_COVERED | no real footage of this event was found in the local media; synthetic unit tests only | | | | | |
| `multiple_interacting_subjects` | `5256…` f14400+10 | 0.027 → **0.005** | 0.264 → **0.009** | 0.247 → **0.043** | 1.145 → 0.077 | neural:3, full:8 | - |
| `dark_lighting` | `5256…` f24450+10 | 0.023 → **0.000** | 0.000 → **0.001** | 0.027 → **0.045** | 0.051 → 0.072 | neural:12, full:8 | - |
| `side_profile` | `03de…` f6750+10 | 0.035 → **0.000** | n/a → **n/a** | 0.089 → **0.029** | 4.548 → 0.046 | neural:2, full:8 | - |
| `motion_blur` | `03de…` f10650+10 | 0.553 → **0.010** | 0.022 → **0.010** | 0.047 → **0.120** | 3.663 → 0.258 | neural:9, full:10 | SCENE_CUT:2 |
| `rapid_scene_transition` | `03de…` f11812+10 | 0.041 → **0.012** | n/a → **n/a** | 0.184 → **0.189** | 0.207 → 0.285 | neural:4, full:6 | - |

Reading it honestly:

* **Hands, glasses, a nearer face, profiles, motion blur:** painting over occluded interior pixels collapses (hand 0.94 → 0.00, glasses
  0.16 → 0.005, two faces 0.26 → 0.009) and leakage outside the face drops (motion blur 0.55 → 0.01, where the standard path paints a blob
  over the background). Profiles stop popping (flicker per painted pixel 4.5 → 0.05).
* **Hair across the face is not solved.** `hair_in_front_of_face` is a head rolled ~90° on a pillow; the landmark hull collapses
  (`HULL_IMPLAUSIBLE_USING_NETWORKS`) and the parser is distrusted, so the robust path paints *less* (conservatively) but still paints 26 % of
  the pixels both networks call occluded. Thin strands over cheeks are below what the 256 px networks resolve.
* **Temporal stability is not uniformly better.** It improves for hands, glasses, fast movement, two faces and profiles, is level for the
  rolled-head hair clip and the shot change, and is **worse** for motion blur (0.047 → 0.120) and dark lighting (0.027 → 0.045) when normalised by face area: the networks
  are noisy there and the matte follows them. The release term (`occlusion_release` 0.75) was tuned on these clips and removed roughly a fifth of
  the excess on those two clips, not all of it.
* **The shot change:** the real tracker gave the face a new id across the cut, so there was no history to reset in the compositor; the pixel-based
  cut detector is verified separately on the real cut (`test_occlusion_tracker_detects_a_real_shot_change_from_pixels_alone`), where ordinary
  consecutive frames never reset and the cut frame resets with `APPEARANCE_CUT`.
* **Pore-level texture is not demonstrated.** On 720p film where a face is ~200 px, the target's fine band is almost empty, so texture transfer
  changed the detail ratio by less than 0.01 in the ablation; the mechanism is unit-tested on synthetic textures only.
* Ablation on three clips (luminance shift and detail energy against the original face box): no stage darkens or smooths systematically
  (|Δ luminance| ≤ 2.2 levels, detail ratio 0.91–1.05 in every variant).

`NOT_COVERED`: **microphone crossing the mouth** (no real footage; only the manual-mask path is unit-tested). Foreground *objects* in general
are covered only to the extent a hand / glasses / a face is one.

Timing (CPU, this machine, measured while a hashing job ran, per face per frame): standard ≈ 170-250 ms, robust ≈ 440-720 ms (three
networks). A CUDA run on the RTX 4070 executed correctly but its timing was not isolated; no performance claim is made for either GPU profile.

## 6. Tests actually executed

* Backend, full suite (`roop_3d/tests`) after the last code change: **897 passed, 10 skipped, 1 failed**. The one failure was my own new GPEN
  guard test (a missing `import cv2` and thresholds that were too strict for soft parser probabilities); it was fixed and the five files that touch
  this stage were re-run: **67 passed** (32 compositor tests with test doubles, 7 real-inference tests, `test_stage09_compositor`,
  `test_model_registry`, `test_workstation_preview`). The previous baseline was 859 passed / 10 skipped.
* Real-inference tests (`test_occlusion_real_models.py`, no mocks: real SCRFD, BiSeNet, XSeg, face occluder, GPEN, real video): network agreement
  and inference state, anatomical order of the parser regions, `imagenet` >= `zero_one` agreement with the networks, hands protected on real
  footage (standard paints > 50 % of occluded pixels, robust < 10 %), a real profile is still composited, the real shot change is detected from
  pixels alone, and the restoration guard with the real GPEN weights (protected eyes/mouth untouched, skin still restored).
* Frontend: `npm run build` (tsc, eslint, vite) exit 0; Vitest 36 files / **302 tests** passed (one new: the opt-in selector).
* Legacy integrity / isolation audit (`audit/verify_isolation.py --report`, read-only): **ok** (2,645 historical + 68,390 comprehensive files
  matched, zero violations, zero unauthorised additions).
* The ten real-footage runs are in section 5; raw JSON: `app/roop_3d/audit/OCCLUSION_STAGE5_VALIDATION.json`, montages (face regions only) in the
  git-ignored `app/roop_3d/.runtime/acceptance/occlusion_stage5/`.

## 7. Acceptance criteria

| Criterion | Status |
| :--- | :--- |
| No background contamination | **Met on 8 of 9 clips** (leak ≤ 0.012). Not met on the hand clip (0.067: the face is almost entirely hidden and the visible remainder is thin; leak is *not* lower than standard there) and unproven where the reference is the same models |
| No visible mask boundaries | **Partly.** A hard rectangular seam at the crop border was found on real footage (a chin leaving the 128 px crop; standard mode shows it plainly). A border fade removes most of it and is regression-tested, but a faint edge is still visible at the chin in the fast-movement clip, and a faint green fringe remains on some thin fingers in the probe montages |
| No floating facial edges | **Partly.** Matte is snapped to frame edges; low-contrast skin-on-skin fingers keep a soft edge |
| No loss of legitimate foreground | **Met for hands, glasses and a nearer face** (occluded-pixel painting ≤ 0.01); **not met for hair strands** (0.26) |
| No temporal flicker | **Not met.** Better on 5 clips, level on 2, worse on motion blur and dark lighting (see above) |

## 8. Known limitations

1. Foreground is "whatever hides the face" to two 256 px networks; there is no hand / microphone / object class model. Thin or low-contrast
   occluders (hair strands, fingers on skin) are partly missed. Adding a promptable segmenter (SAM-class) would need explicit user-approved
   downloads with licence and hash verification; none was added.
2. Teeth are a heuristic used only to protect pixels from restoration.
3. Depth ordering of faces is a size / position heuristic; there is no depth estimate, and a farther hand cannot be ordered against a nearer face.
4. Landmarks (106-point) collapse on strongly rolled heads; the robust path substitutes the network face region, which is a partial face on a
   90° roll. Fixing roll needs a roll-aware aligner (not in scope).
5. Cost: three networks per face per frame (CPU ≈ 0.5 s). Not benchmarked for the 4070 / 3060 profiles; the new models use the default
   arena divisor in `OrtModel`.
6. The standard path keeps its known weaknesses (128 px parsing, `zero_one` BiSeNet preprocessing, parser-failure pop-in); they are documented
   here, not changed, to preserve backward compatibility. Migrating the default is a separate, tested decision.
7. Colour matching is still 8-bit LAB; only the boundary blend is linear-light.
8. Provenance and licence of the three networks are `UNKNOWN` in the registry; this stage verified weights and inference only.

## 9. Reproduce

```
# from app/
roop_3d/.venv/Scripts/python.exe -B -m pytest -c roop_3d/pyproject.toml --confcutdir=roop_3d roop_3d/tests/test_occlusion_compositor.py -q
roop_3d/.venv/Scripts/python.exe -B -m pytest -c roop_3d/pyproject.toml --confcutdir=roop_3d roop_3d/tests/test_occlusion_real_models.py -q   # real models + video, skips if absent
roop_3d/.venv/Scripts/python.exe -B -m roop_3d.tools.occlusion_acceptance --out roop_3d/.runtime/acceptance/occlusion_stage5
```

API: `WorkstationParameters(occlusion_mode="robust")` (or the "Occlusion handling" selector in the Masking panel); per-face evidence is in
`FrameCompositor.last_reports[track_id]` (tier, confidence, reasons, networks, temporal resets, face motion, fusion metrics) and
`capability()["robust_occlusion"]`.
