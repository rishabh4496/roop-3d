# Stage 6B - silhouette replacement (overhang inpaint, perimeter cage, swap scope)

Path mapping: the prompt names `app/roop_3d/rendering/compositor.py`; the compositor is `app/roop_3d/compositor/pipeline.py` (it
already had a one-tier `_remove_old_silhouette`, which this replaces). UI is `app/roop_3d_ui/src/components/studio/`. Tests:
`app/roop_3d/tests/test_silhouette_replacement.py` (the prompt's `roop_3d/tests/` is that folder), `src/swapScope.test.tsx`, and one
Playwright spec in `e2e/workstation3d.e2e.ts`.

| Task | Module | What it is |
|---|---|---|
| 1 | `rendering/silhouette_inpaint.py` | `SilhouetteDeltaInpainter`: `M_overhang = dilate(M_tgt) & ~M_src`, filled from temporal donors first, a multi-scale spatial fill second |
| 2 | `rendering/perimeter_cage.py` | `PerimeterCageWarper`: Gaussian-RBF field from `DeltaP`, exponential falloff, bisection to a fold-free field; mesh and silhouette correspondences |
| 1+2 | `rendering/silhouette_replacement.py` | `SilhouetteReplacer` (warp hair/ears/neck, then fill what they vacate), `resolve_scope`, `DonorHistory` |
| 3 | `WorkstationParameters.swap_scope` / `skull_blend`, `SwapScopeSelector.tsx` | the 3-mode selector; default `full_craniofacial` |
| 4 | `tests/test_silhouette_replacement.py` | wide-target to slim-source and slim-target to wide-source, through the real `FrameCompositor` |

## What it does, in order (all BEFORE the source face is composited)

1. `DeltaP = Project(V_source_perimeter) - Project(V_target_perimeter)` drives the cage. Correspondences come from
   `mesh_perimeter_pairs` (same vertex ids of two topology-identical heads; the ring is `perimeter_vertex_ids(SwapMask3D.weights, faces)`,
   so it follows the full craniofacial mask) or, when the engine supplies none, `perimeter_from_masks` (ray-matched outermost pixels of the two silhouettes).
2. The structure layer (BiSeNet hair + ears + neck) is resampled through the field and laid back over everything outside the new head.
3. What it vacated and the overhang ring are filled with background. Donors never include old skin, the structure layer, the new head or a foreground occluder.
4. Occluders are never moved, filled or used.

Inpaint tiers: (a) neighbouring frames, each registered with a RANSAC homography fitted to Farneback flow sampled only outside every head mask,
gated on a photometric residual over the band around the hole, corrected for exposure drift, rejected where the donor itself had the head, median over donors;
(b) multi-scale Navier-Stokes (isophote propagation) with a thin-band refinement per scale and matched grain. `neural_inpainter=` plugs a LaMa-style callable in; **none is bundled**, the report names the method that ran.

## Scope

| scope | `source_identity_lock` (-> `ArticulatedHeadLayer`) | silhouette step |
|---|---|---|
| `full_craniofacial` (default) | 1.0 | warp + fill |
| `adaptive_skull` | `skull_blend` | warp + fill, on the silhouette the engine rendered for that blend |
| `inner_face` (legacy) | 0.0 | none; the swap stays inside the target's own face matte |

The compositor never rescales the silhouettes it receives (they already are the geometry for that lock), so in the compositor `adaptive_skull` and
`full_craniofacial` run the same step; the lock itself is the engine's.

## Honest limits

* **Synthetic scenes only.** Every measurement (overhang equals the true background to 6 grey levels with a donor, hair contracts/expands to hug the
  new skull) is on constructed ellipse scenes with a known background. There is no real-footage evaluation.
* **The production trigger is narrow.** The step runs for `Replacement.full_head=True` items, and `capability()` reports the full-head renderer as
  `disabled_legacy_cg`: nothing in the live engine produces those items today. Wiring is complete up to the compositor (params, preview payload,
  job fingerprint, UI), but a user flipping the selector gets no visible change until an engine produces full-head replacements. The UI says "takes effect in a 3D render mode" - that is the weakest honest claim; it is not a promise that the current engine exercises it.
* `target_silhouette` from `FullHeadEngine` is the 17-point jaw polygon only; the compositor widens it with the parser's skin map when the parser is trusted.
* Donors are causal (past `silhouette_history`=2 frames) plus whatever the caller passes in `compose(..., background_donors=)`; an offline driver can pass t+1..t+k.
* Work happens at the analysis crop resolution (`work`), then is fused back like the old ribbon fill.
* Neither tier recovers background nobody saw; tier (b) is plausible continuation.
* The dilation of `M_tgt` also rewrites a thin rim of real background around the old head (and a rim at the poles of two equally tall heads).

## Pitfalls found

* OpenCV Telea drifts by up to 4 grey levels on a perfectly flat background; Navier-Stokes is exact on flat and linear fields. Spatial tier uses NS throughout.
* A pyramid that area-averages skin into the background estimate bleeds colour: downsampling is normalised over KNOWN pixels only.
* Grain measured next to an old-skin edge reads the edge as texture: measured only where the blur cannot reach unknown pixels.
* The dilated ring covers hair: structure pixels are `keep`-protected, otherwise an identity warp erases the hair.
* The warped structure can land on an occluder: it is masked by the occluder.
* Pre-existing failing test: `test_extreme_pose_3d_swap.py::test_exact_view_does_not_import_other_hemisphere_texture`.
