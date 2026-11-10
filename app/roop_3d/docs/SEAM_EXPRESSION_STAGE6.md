# Stage 6 - boundary seams and continuous expression direction

Path mapping: the prompt names `app/roop_3d/rendering/` and `app/roop_3d/editing/`. `rendering/` exists; `editing/` is new (and
registered in `pyproject.toml`). Tests are in `app/roop_3d/tests/`.

| Task | Module | Inspired by | What it actually is |
|---|---|---|---|
| 1 | `rendering/swap_region_3d.py` | ROME | Per-vertex 3D swap mask (inner oval, opt-in jawline / chin underside / ears) and a biharmonic Laplacian displacement field that carries the jaw onto a different jaw width |
| 1 | `rendering/vertex_raster.py` | - | Exact CPU z-buffer for per-vertex attributes (mask weights, normals); the bridge from the 3D mask to pixels |
| 2 | `rendering/strand_occlusion.py` | NeuralHaircut | `StrandAwareOcclusionCompositor`: depth z-test, N.V grazing falloff, parser + object occluders, and a directional Gabor strand matte |
| 3 | `editing/expression_audio_director.py` | NED, FaceFormer | `ExpressionAndAudioDirector`: fixed linear emotion offsets in psi-space, and an opt-in audio lock of lip/jaw when the mouth is occluded |

## What was NOT there, and what that changed

* There was **no existing "3D swap mask"**. The swap region was the render's texture coverage plus `IdentityRig` weight fields.
  `SwapMask3D` is new. The only existing "Laplacian" is the umbrella smoother in `reconstruction/rbf_deform.py`; it is reused
  (weld ids, welded normals, fold-relief smoothing), and the displacement field itself is new.
* **No FaceFormer / NED / ROME / NeuralHaircut weights or code are used.** The names describe the idea each task borrowed.
* No object-class model (hands, microphones) is installed (`OCCLUSION_STAGE5.md`), so the `object` cue is a caller-supplied map.

## Task 1 - jaw, chin, ears

`build_swap_mask_3d(vertices, faces, landmarks68, SwapRegionConfig)` -> `SwapMask3D.weights` (max over enabled components).
Default config = inner oval only, unchanged behaviour. `adapt_jaw(...)` with no extension enabled returns the mesh bit-identical.

* **Field**: minimise `|L d|^2` on the position-welded graph, `d` = handle displacement at the jaw landmarks (and chin / ear
  anchors when enabled), `d = 0` outside a geodesic zone (`jaw_band_m + inner_blend_m`). Biharmonic, not harmonic: a harmonic field
  has a log spike at every point handle. UV-seam copies move identically. Front-facing triangles that would flip are relieved
  locally (`rbf_deform` pattern); the report states folds, residual, and strength rather than claiming success.
* **Where the target jaw comes from is the caller's job**: `jaw_target_from_ratios` (a dial: width, chin drop, ears follow by
  `ear_follow`) or `jaw_target_from_landmarks` (a fitted target head, registered on the *non-jaw* landmarks so pose/scale offsets
  are not mistaken for a jaw change). Nothing estimates it from video pixels.
* **Priors, not annotations**: the 68-point correspondence on the MakeHuman head is derived (`makehuman_landmarks`); ear anchors are
  the lateral extremum of the eye-to-mouth band. Both are noted in `SwapMask3D.notes`.
* Measured on the real head, 0.86x jaw: swap-region/target-jaw disagreement fell by >75 % (`test_seam_free_swap_pipeline`).
  That is a geometric mismatch area, **not** a pixel-seam measurement on real footage.

## Task 2 - occlusion

`alpha = swap_alpha * grazing * (1 - occluder)`, `occluder` = noisy-OR of depth, semantic (glasses/hat/hair weights), object, strand.

* **depth**: occluder depth nearer than the rendered face depth by `depth_tolerance_m`; unknown (inf/NaN) = no occlusion.
* **grazing**: smoothstep of `N.V` between `grazing_cos_low` and `grazing_cos_high`. Not an occluder; the target's pixels take over.
* **strand**: even-Gabor bank (12 orientations) for thin ridges x structure-tensor coherence x hair-flow alignment x skin-to-hair
  colour line, and only where the ridge is within `strand_reach_px` of hair **or connected to such a ridge** (hysteresis, up to
  `strand_max_length_px`). Isolated skin creases, isotropic grain/speckle, and bright (non-hair-coloured) threads are rejected;
  each of those has a test whose positive control proves the cue alone *would* have fired. Facial hair uses the same matte via a
  `facial_hair` prior.
* **Boundary snap** (guided filter) acts only where alpha changes **and** the target has local contrast. On a flat guide a guided
  filter is a box blur of radius ~2r that erases thin occluders; this was a real defect found by the tests and fixed.
* Wiring: depth/normals come from `PhysicalBuffers` or `vertex_raster`; `OcclusionResult.occlusion_in(region)` gives the director
  its `mouth_occlusion`. `rendering/__init__` exports the new names lazily (eager import would be circular).

## Task 3 - director

psi-space = the tracker's eight channels (`mouth_open, mouth_width, smile_a/b, blink_a/b, brow_a/b`) in `ExpressionParams` units.

* Offsets: `smile`, `frown`, `surprise`, `eye_openness` in [-1, 1], fractions of channel full scale, eased with `ramp_s`, clamped
  to the rig limits (clamped channels are listed). The jaw follows the *final, clamped* mouth opening.
* **Eyes cannot open wider than neutral** in this rig (blink is closure only): `eye_openness > 0` suppresses blinks and adds a
  small brow cue, and says so in `limits`; `< 0` squints (never reduces a real blink).
* **Audio lock** (`audio_lock_enabled`, default off): weight = smoothstep of mouth occlusion between `lock_start` and `lock_full`,
  eased by `blend_s`. Audio -> mouth via DSP proxies on the decoded 16 kHz signal (`engine/audio_sync.decode_audio`): loudness
  above the noise floor x F1 centroid (jaw height), F2 centroid (spread/round as the mouth closes). Amplitude is calibrated online
  against the actor's own *visible* mouth motion, else a default gain (reported). **Not lip-sync**: /a/ vs /u/ are only partly
  separable from spectral centroids, bilabial closure is only a silent gap, and noise-only audio is refused (`speech_detected`).
* A FLAME 50-d psi needs a caller-supplied `EmotionBasis`; FLAME weights are not installed, so none is shipped.

## Wiring into `ReferencePipeline` (follow-up)

Both are **opt-in and off by default**; with the flags unset every code path and every request fingerprint is unchanged
(`WorkstationParameters` omits unset fields from serialisation, as for its other opt-in knobs).

| What | Where it plugs in | Switch |
|---|---|---|
| Strand matte | `compositor/robust_fusion.fuse_robust` (frame resolution), fed the parser's hair map by `FrameCompositor._compose_robust` | `RobustOptions.strand_matte` / `WorkstationParameters.strand_matte` (needs `occlusion_mode='robust'` and a trusted parser) |
| Jaw-width fit | `PoseBlendProjector.render(jaw_contour=...)`, fed the 106-point face contour by `ReferencePipeline.infer_frame` | `infer_frame(jaw_adaptation=True)` / `WorkstationParameters.jaw_adaptation` |

What the investigation found, because it changed the design:

* **The 3D swap mask / `adapt_jaw` do not belong in the dense (`neural_face`) route.** That route builds its replacement boundary
  from the *target's own* fitted surface and synthesises the face neurally, so the target's jaw is already the boundary and no
  source geometry is rendered onto it.
* **`ReferencePipeline.rigs` is never populated** (commit 9459784 replaced the `IdentityRig` route), so `reenactment_3d`
  always reports `IDENTITY_ASSET_UNAVAILABLE` and `_pose_3d_projector` is unreachable (`mode` is reset to `standard` when the
  dense bundle is missing). Nothing was wired into those paths.
* **The one live place a 3D proxy meets the target is `PoseBlendProjector`** (`pose_mode='mesh_blend'`, method
  `identity_swap_tps_mesh_blend`): a generic 76-vertex head (vertices 0-67 = iBUG landmarks) fitted to the target by five
  points, whose generic jaw outline is the footprint. That is where the jaw fit goes.
* The strand matte math moved to `compositor/strand_matte.py` (matting; also breaks an import cycle). `rendering.strand_occlusion`
  re-exports it, so nothing that imported the old names changed.

### Jaw fit (`reconstruction/jaw_fit.py`, `pose_blend.py`)

* **Measurement is reference-free and pose-aware**: the proxy jaw is projected through the SAME PnP camera as the rest of the proxy
  (foreshortening cancels), the target outline is the detector's 106 contour (BLOCKS `contour`, gated by `layout_is_plausible`),
  and the two are compared as lateral width at matching heights of the lower face (20-85 % of the way from the ears to the chin);
  the median is the width ratio. No population mean is assumed.
* **Guards**: pose gate `cos(yaw) >= 0.75` (from the PnP rotation, ~41 degrees), dead band +-3 %, clamp 0.85-1.18, at least 4
  comparable heights, fold check (falls back to the generic proxy). Every outcome is on `replacement.projection.jaw`
  (`JawFitReport`: applied, reason, measured/used ratio); an untrustworthy layout reports `contour_invalid` rather than skipping silently.
* **Geometry**: `adapt_jaw` with the inner-face landmarks 17-67 *pinned* (new `pinned=` option). Without it the coarse mesh's
  long edges carried the field into the eye corners (up to 6.5 mm, ~0.1 inter-ocular) and off the PnP anchors; pinned, the
  anchors move exactly 0. UVs of the moved jaw vertices are re-derived in canonical reference space; every other UV is unchanged.
* **Not verified**: the 106 contour and the iBUG jaw are two annotations of one outline, not identical definitions, so a small
  systematic bias is possible (the dead band exists for it). Validation is synthetic (known ratios) plus the canonical mesh; no
  measured real-footage seam is claimed. Try it on your own clip and read `projection.jaw.measured_ratio` before trusting it.

## Not done / follow-ups

* Not exposed in the UI (`roop_3d_ui`), and not committed.
* No pixel-level seam evaluation on real swapped footage; the explicit local videos must not be used for that (see Stage 5 notes).
* The strand matte's depth and grazing cues are not fed by the live compositor: it has no occluder depth and no per-pixel
  normals, so only the strand cue is live there. They remain available through `StrandAwareOcclusionCompositor` directly.
* Packaging: `roop_3d.tracking` and `roop_3d.editing` are now listed. **Eight further packages are still missing from
  `[tool.setuptools] packages`** (`color`, `enhancers`, `evaluation`, `identity`, `installer`, `orchestrator`, `security`,
  `swappers`); `color` is imported by `core.workstation`, so an installed wheel would fail at import. They are pinned as a
  documented gap in `test_packaging.py::KNOWN_UNLISTED`; adding them to `pyproject.toml` and emptying that set closes it.

## Tests

`test_swap_region_3d.py` (18), `test_strand_occlusion.py` (21), `test_expression_audio_director.py` (20),
`test_seam_free_swap_pipeline.py` (5, real head; skipped when the MakeHuman asset is not built), `test_jaw_fit.py` (24),
`test_packaging.py` (4; one needs setuptools and skips in the project venv), plus 4 strand-matte tests appended to
`test_occlusion_compositor.py`. Deliberate defects were injected to check the suites fail: 9 of 10 (rendering/director),
4 of 5 (compositor wiring) and 11 of 11 (jaw wiring) killed. The two survivors are equivalent mutants (the grazing multiply is
also enforced by a final clamp; the trust gate duplicates the analyser's `regions=None` for an untrusted parser).

```
cd app && roop_3d/.venv/Scripts/python.exe -m pytest roop_3d/tests/test_swap_region_3d.py roop_3d/tests/test_strand_occlusion.py \
  roop_3d/tests/test_expression_audio_director.py roop_3d/tests/test_seam_free_swap_pipeline.py roop_3d/tests/test_jaw_fit.py \
  roop_3d/tests/test_packaging.py roop_3d/tests/test_occlusion_compositor.py
```
