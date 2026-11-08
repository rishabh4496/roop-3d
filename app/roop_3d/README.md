# Roop 3D Python package

This package belongs to the independent Pinokio app at the repository root.
Read the root README.md, CONTRIBUTING.md and AGENTS.md for all later stages.
The embedded Roop Ultimate location is historical source material only.

From this project app/ directory:

    roop_3d/.venv/Scripts/python.exe -B -m pytest -c roop_3d/pyproject.toml --confcutdir=roop_3d roop_3d/tests -q

External model source locations are read-only and separately hashed. No legacy
application code or environment is imported.

Full-head GPU SH relighting, optional iris tracking, oral detail and neck seam
handling are documented in [Photometric head synthesis](../../docs/photometric-head.md),
including local model setup and the real ArcFace acceptance fixture.


## Open head model (MakeHuman, CC0) - FLAME alternative

`reconstruction/makehuman_head.py` is a licence-clean parametric head (4,289 vertices, 50 identity + 10 expression axes) built from
MakeHuman CC0 data. Build once (downloads ~130 small files from a pinned commit, SHA-256 verified against
`models/makehuman_sources.json`):

    python -m roop_3d.reconstruction.makehuman_build

    model = MakeHumanHeadModel.load()
    result = fit_multiview(model, [ViewObservation(landmarks5, width, height), ...])   # shared identity, per-view pose

Frame: OpenCV head frame in metres (R = I is a frontal face). Scope: 5 landmarks per view constrain pose and coarse proportions, not
fine identity; views whose landmarks disagree with the rest are rejected from the shared identity (`ViewResult.rejected`).

### MakeHuman as the identity-asset template

    settings = IdentityAssetSettings(template="makehuman")      # default stays "canonical"
    asset = build_identity_asset(document, settings)            # or IdentityAssetStore().get_or_build(document, settings)

API: `GET|POST /api/v1/facesets/{id}/identity-asset[?template=makehuman]` and `.../identity-asset/mesh?template=makehuman`.

The MakeHuman head+neck (4,481 vertices after seam splitting, UV islands repacked to ~46 % atlas use) replaces the canonical
facial mesh as the template the fitted landmarks are carried onto. 5-point landmarks only (dense 68-point sets are ignored with
a note). Everything outside the landmarks' influence (ears, skull, back of head, neck) is template prior and reported
UNSEEN / INTERPOLATED, never measured. The expression rig (`IdentityRig`) and pose blend refuse these assets
(`IDENTITY_TEMPLATE_UNSUPPORTED`) because they index the canonical 68-point topology; build with `template="canonical"` for those.
