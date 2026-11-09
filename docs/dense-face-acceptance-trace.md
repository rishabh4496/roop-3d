# Dense-face requirement and public-interface acceptance trace

Date: November 9, 2026. Implementation commit: `9459784`.

## Rechecked intent and interpretations

The requested outcome is a runnable, visibly more photorealistic face swap, not merely native encoders, shape-valid graphs or a convincing mesh preview. **That outcome is still unproven and not delivered by the available asset set.** No source/target comparison establishes that this change improves real swapped-face appearance. Engineering checks below must not be presented as such a comparison.

The checkout already contained substantial uncommitted adapters, export preparation and live-route changes when this session began. They were reviewed against the settled request and integrated, rather than discarded or represented as newly trained networks. The following choices are interpretations, not facts supplied by the user:

1. **Replace the active template, preserve historical utilities.** MakeHuman assets and offline builders remain for existing projects and regression fixtures. Live 3D swapping does not select them. The user might instead have intended deletion of every legacy asset; that destructive interpretation was not taken.
2. **Use genuine SMIRK coefficients, not a fabricated ARKit conversion.** The native route uses expression-50, jaw-3 and independent eyelid-2 from the public SMIRK checkpoint. These are not 52 named ARKit controls. The custom graph route can accept genuine ARKit52 exports, but none has been installed or accepted here.
3. **Use registered BFM identity offsets on either output surface.** FLAME support means a genuine 5,023-native-vertex FLAME Open rig and pruned surface coupled by measured registration, not a separately trained FLAME coarse identity regressor. Monocular camera scale uses an explicit metric gauge, not measured absolute head size.
4. **Honor target anatomy, use source identity for appearance.** Corresponding source photographs are projected into target-fitted bones and expression; source identity embeddings condition the neural model. Whether that distribution preserves identity and expression perceptually still needs real evaluation.
5. **Bound photographic caching.** Original `.fsz` images are sampled into 512px/1,024px aligned caches depending on profile; original-coordinate diagnostic maps are exposed separately. This is not unlimited-resolution source texture storage. Only selected trusted real views, not every stored view, are fitted and cached.
6. **Allow a labelled inspection-only proxy.** The FaceSet viewer may show a canonical limited reconstruction when dense assets are absent. This is not the live swapping fallback and is not labelled as a three-tier fit.

## Actual project interfaces exercised

A production `create_app(worker_enabled=False)` instance used the checkout's real settings, model catalog, FaceSet parser, landmark engines, reconstruction and middleware. Requests went through FastAPI's actual ASGI routes with a loopback Host accepted by the normal middleware. There were no substituted model sessions, source photographs, response objects or inference services. Lifespan recovery and the queue worker were not started, to avoid reconciling existing jobs or sweeping processes. Consequently these are real route/integration observations, **not a full Pinokio-launched or browser acceptance run**.

| Actual interface | Observed result |
| --- | --- |
| `GET /api/v1/health` | HTTP 200, `application=roop_3d`, `standalone=true` |
| `GET /api/v1/models` | HTTP 200 using the installed model catalog |
| `GET /api/v1/facesets` | HTTP 200, existing local FaceSet list |
| `GET /api/v1/faceset/3d-state` before reconstruction | HTTP 404, `RECONSTRUCTION_NOT_FOUND` |
| `POST /api/v1/faceset/reconstruct-3d` with an existing loaded local `.fsz` | HTTP 200 in 8.0s; `template=canonical`, `reconstruction_level=limited_multiview`, `backbone.available=false`, `raw_mesh_compositing=false`; limitation explicitly names the missing dense bundle |
| `GET /api/v1/faceset/3d-state` for that source afterward | HTTP 200; same canonical limited reconstruction and unavailable-dense-backbone metadata, not a falsely reported dense fit |
| Actual `MorphableBackbone.load` under `6gb` and `12gb` runtime configurations | Both reject with `DENSE_FACE_ASSETS_MISSING`, status 503. These are profile configuration checks on the same 12 GB device, not physical 6 GB measurements |
| Existing `test_reenactment_session_real.py` with real installed models, FaceSet and video | **3 passed, 2 skipped in 24.22s**. Actual standard neural frame inference passed, invalid render modes were rejected, and missing dense assets refused 3D preparation without a CG fallback. Both complete three-tier photo/video tests skipped specifically because `bundle.json` is absent |

The last real-media test invocation was:

```powershell
# From app/
roop_3d/.venv/Scripts/python.exe -u -B -m pytest -c roop_3d/pyproject.toml --confcutdir=roop_3d --basetemp=roop_3d/.runtime/pytest-real-acceptance-audit roop_3d/tests/test_reenactment_session_real.py -rs -q --tb=short
```

Local route evidence is retained in ignored `cache/dense-face-public-acceptance.json`; the one-shot driver is `cache/dense-face-acceptance-boundary.py`. It can be run from `app/` with the dedicated Python using `-c "import runpy; runpy.run_path('../cache/dense-face-acceptance-boundary.py', run_name='__main__')"`. No photographs or model data are included in the committed evidence.

## Requirement-to-observation matrix

Named tests below passed in the final 128-case focused run unless explicitly labelled otherwise. Fixtures are synthetic where noted. A passed mechanical check does not certify anatomical or perceptual accuracy.

| Explicit requirement / changed public output | Concrete check and observation | Remaining acceptance constraint |
| --- | --- | --- |
| Replace MakeHuman in the active swapping route | Real `test_missing_dense_bundle_never_falls_back_to_a_cg_head` passed; real preparation rejects absent assets instead of producing the former CG face. Session-route regressions assert the dense neural replacement method and `full_head=false` | Actual BFM/FLAME face output cannot be activated yet |
| BFM 35,709 vertices and eight semantic parts | `test_semantic_cut_physically_removes_excluded_vertices_and_boundary_triangles[bfm35709]` and full-count native fit fixture passed | Genuine anatomy archive and actual facial cut are missing. Eight upstream masks alone do not classify scalp/neck |
| FLAME 2023 Open 5,023-vertex support | Same semantic-cut test for `flame2023open`, full-count native decoder fixture and `test_flame_topology_publication` passed | Actual Open rig, basis conversion and registrations are missing |
| Exclude scalp/hair, ears and neck permanently from the active surface | Semantic-cut fixture verifies exclusion vetoes inclusion, drops boundary-crossing triangles and physically compacts indices. Refinement test verifies no new anatomy or boundary holes are added | Not a real-person hairline/ear/neck annotation validation |
| Include forehead below hairline, brows, periorbital region, nose, cheeks, lips, chin and mandibular border | Requires genuine anatomical inclusion masks and real projected boundary review; current published readiness reports missing topology/masks | **Not verified** for these individual anatomical regions; no inferred broad mask was installed |
| Tier 1 coarse identity and camera `(R,t,s)` on source and target | `test_native_camera_matches_independent_upstream_projection_including_tz`, exact crop-transform test and native decoder fixtures passed. Existing genuine coarse export/report was inventoried | Real full-chain source and target fits are blocked; native `s=1` is a fixed gauge, not measured scale |
| Tier 2 genuine asymmetric SMIRK expression basis | `test_flame_decoder_articulates_jaw_without_mirroring_smirk_or_eyelids` and asymmetric surface-transfer test passed, preserving separate left/right controls | No real squint, smirk, nostril flare or lip-tension accuracy measured after Open conversion |
| 52-ARKit alternative | Custom schema validates a declared 52-value range; native schema explicitly declares SMIRK50 instead | No genuine ARKit52 export or semantic mapping accepted here; SMIRK50 is not relabelled ARKit52 |
| Tier 3 HRN mid deformation and high UV displacement | Native UV lookup and `test_mid_uv_sampling_matches_upstream_hrn_formula_at_noninteger_uvs` passed. `test_high_frequency_displacement_between_native_vertices_is_not_lost` recovered a supplied **3 mm** midpoint peak that native vertices sampled as **0 mm** | Component improvement is analytic, not recovered human microgeometry or measured nasolabial/crease fidelity |
| Predict all three tiers for `.fsz` sources and target frames | Both native-family fit fixtures exercise the real geometry decoders with stub encoders; refined-engine fixture connects source and target correspondence | Genuine multi-network inference cannot complete without rig/anatomy/registration assets; selected source-view fitting only |
| Dense 2D-to-3D-to-2D field, not affine UV pasting | `test_perspective_correspondence_matches_independent_camera_ray_solution` passed; actual hardware interpolation agrees with CPU within the test's 0.2 mm tolerance | No real-pose warp error or temporal stability measured |
| Do not sample self-occluded, padded or foreign-surface texture | Bilinear-tap tests reject all nine non-nearest obstruction cases. Rasterized foreground at a non-nearest tap produces confidence 0. Slanted visible interior keeps confidence 1 with projection agreement within 2e-6 px | Real hair, glasses, hands and mouth occlusion distributions need video evaluation |
| Real high-resolution, multi-angle source photography | Refined-engine fixture verifies photograph sampling and preserved source-view correspondence. Public real `.fsz` reconstruction accepted actual photographs, but reported canonical fallback | Real dense photo warping not executed; cache sizes and per-frame view limits are explicit, not a full-resolution guarantee |
| Target pose and bones, source embedding and identity | Native target-fit and photographic-warp contracts pass; full-neural-output test checks 512-value source embedding and target-native crop | Bone preservation, identity retention and expression accuracy need matched real outputs |
| HyperSwap 1A/1B/1C 256, RealSwap 512 or InSwapper 128 integration | `test_full_neural_output_and_ultramax_own_all_replacement_pixels` passed at 128, 256, 512 ArcFace and 512 FFHQ using model doubles. Existing real standard InSwapper path passed | No actual HyperSwap variant, RealSwap or InSwapper was run with the **new dense conditioning**. Resolution fixtures are not substitute pretrained-model acceptance |
| UltraMax and no final raw Lambertian/Phong face image | Full-neural-output fixture verifies enhancer receives only full neural output and final replacement equals enhancer output, including source holes; real missing-bundle preparation supplies no CG replacement | Actual dense-to-neural-to-UltraMax image chain remains blocked, so 100% neural replacement ownership is contract-tested, not visually demonstrated |
| Profile-specific refinement/public capability fields | `test_geometry_detail_budget_is_profile_specific` passed: requested level 2 becomes 1 on 6 GB/CPU and remains 2 on 12 GB. Native vertex count stays fixed while warp vertex count changes. Invalid levels fail publication | No peak VRAM/latency or physical 6 GB acceptance measured |
| Original-coordinate warp diagnostics | Refined-engine test verifies inverse affine maps fields into original photograph coordinates, paired source-view indices, invalid -1 coordinates and stale-state clearing | Diagnostics are not unlimited-resolution sampling or a recovered texture quality measure |
| Changed FaceSet preview/cache/UI output | Actual POST then cached GET return the same canonical limited fallback, unavailable backbone and missing-bundle reason. Cache-signature invalidation test and 39 viewer/inspector/studio tests passed. Production build passed | No screenshot of an activated dense BFM/FLAME preview or browser workflow with genuine complete assets |
| Safe local model preparation and readiness | Bundle tests reject mismatched digests, missing assets, invalid geometry/graph contracts, external weights, hardlinks and publication races. Actual CLI finds four encoder files but five missing geometry roles and no manifest | Evidence strings do not certify registrations or data licenses; genuine prepared assets still required |

## Conclusion and stopping condition

**Improved with concrete component evidence:** the refined surface retains a supplied inter-vertex detail feature that was previously lost, and bilinear-footprint visibility rejects non-nearest occluder contamination while retaining valid slanted correspondence. Real application routes expose an honest unavailable-dense/canonical-preview state. Existing standard neural video inference and explicit missing-bundle refusal were observed with actual models and media.

**Not established:** better photorealism, correct real facial anatomy cuts, recovered real asymmetric expressions/skin creases, actual new-conditioning performance for the named swapper models, or safe realtime 6 GB/12 GB operation. Those outcomes require the missing genuine assets and real before/after photo/video measurements. The goal remains **integrated implementation with acceptance blocked**, not a completed photorealistic upgrade. No further safe local test can manufacture the absent rig, registration quality or ground-truth appearance evidence.
