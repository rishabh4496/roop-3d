# Data-free albedo prior (Stage 1 follow-up)

Code: `app/roop_3d/reconstruction/albedo_prior.py`; opt-in hook `far_field` in `uv_completion.complete_texture`; opt-in query
flag `albedo_prior=true` on `GET /api/v1/facesets/{id}/identity-asset/mesh`; tests `tests/test_albedo_prior.py` and the `albedo`
tests in `tests/test_identity_asset_topology_api.py`.

## Why this exists, and why it carries no licence

The statistical albedo model we would have liked (AlbedoMM) is *academic research only* and its FLAME-topology version also depends on
FLAME data. MakeHuman's repository says its textures are CC0, but the three skin litsphere PNGs are stamped "Lic. AGPL" in the image
itself, and the real UV skin textures live in separate asset packs with their own licences. None of that could be verified as licence-free,
so **nothing from any of them is used**. The prior uses only the subject's own OBSERVED texels and ordinary statistics: no scans, no
images, no learned weights. `report.third_party_data` is `false` and is asserted in the tests.

## What it does

1. Skin gate (YCrCb range, then a Lab radius around its own median): a *fixed support*, so hair, brows, lips, eyes and specular
   highlights can never contribute.
2. A constant plus 96 Gaussian bumps over 3D surface position, fitted in CIELAB by ridge-penalised, Huber-weighted least squares. Only
   the bumps are penalised, so far from observed texels the prediction relaxes to the subject's robust skin tone.
3. Used as `far_field` of the existing symmetry + harmonic completion, or as a standalone fill of UNSEEN texels (`fill_unseen_with_prior`).
4. It **refuses** (`ALBEDO_PRIOR_NO_SKIN`, 422) when too few observed texels look like skin, rather than inventing a colour. An implausible
   tone (strong colour cast) is flagged `in_gamut: false`, never silently clamped.

## Measured result on real photos: no benefit

Hold-out study on six real FaceSets (asset built with the MakeHuman template, 512 px atlas): hide a patch of *observed* skin, and
a mirrored patch so symmetry cannot help, complete it, and compare with what the photos actually show (skin-like texels, CIE76 dE,
lower is better). Parameters were tuned on three FaceSets and validated on the other three.

| Hole radius | existing completion | + prior as far field | gain | better in |
|---|---|---|---|---|
| 3.5 cm | 15.79 | 15.60 | +0.19 | 10/15 |
| 5.5 cm | 17.02 | 16.98 | +0.04 | 7/15 |
| 7.5 cm | 17.96 | 18.96 | -1.00 | 4/15 |
| all | | | **-0.26** | 21/45 |

* As a standalone fill the prior is clearly better than one flat tone, but clearly worse than the existing completion (about 26 vs 15 dE), because the
  harmonic fill reproduces the colours at the hole's border exactly and a smooth global fit cannot.
* An earlier version only *seeded* the skin gate; dark hair texels re-entered the fit and the numbers looked better (about 1.5 dE gain) while the picture
  showed a visibly greyed far field. Gating properly removed the "gain", which was contamination.
* Conclusion: **leave `albedo_prior` off.** It is shipped opt-in because it is correct, tested, licence-clean and reports its own limits, and
  because the flag also produces a hole-free display atlas; it does not improve texture quality over what the pipeline already does.

## What would actually help

A real statistical albedo model needs data we do not have the rights to: a commercial AlbedoMM licence (contact the authors), or your own
consented, CC0-licensed scans. The back and sides of the head are mostly hair in reality, which no skin prior can predict.

## Limits

Smooth tone only: no lip, brow, eyelid, freckle or identity detail for texels nobody saw. Filled texels are priors, not observations;
the completion also softens observed texels within a few pixels of filled areas to hide the seam. Tuned and validated on six FaceSets
(five-view, 512 px atlas); other capture setups are unmeasured.
