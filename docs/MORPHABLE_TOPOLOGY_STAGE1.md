# Morphable topologies, surface bridge and semantic shape sliders (Stage 1)

Path map (the stage prompt named `app/roop_3d/`, which is the real backend): code in `app/roop_3d/reconstruction/`
(`topology.py`, `mesh_bridge.py`, `semantic_shape.py`), route in `app/roop_3d/server/identity_assets.py`, licence pin in
`app/roop_3d/audit/local_models.py`, tests in `app/roop_3d/tests/test_{topology_registry,mesh_bridge,semantic_shape,identity_asset_topology_api}.py`
plus one end-to-end test in `test_identity_asset_makehuman.py`.

## What it does

| Piece | Behaviour |
|---|---|
| `MorphableTopologyRegistry` | `makehuman_cc0` (CC0-1.0) is built in and the default. `flame2023_open` (CC-BY-4.0) is pluggable: user-supplied, git-ignored, pinned by SHA-256 in `local_models.OPEN_ARTIFACTS`, parsed with an allow-listed unpickler (numpy + `scipy.sparse` only). The registry refuses any spec whose licence is not commercial-use (FLAME 2020, DECA, EMOCA, MICA, VOCA can never be registered). |
| `BarycentricMeshBridge` | Registers the two rest surfaces once (trimmed similarity ICP, then a TPS refinement kept only if it helps and does not fold), expresses every vertex of one topology as a surface point of the other, and converts **positions, normals and UV albedo (plus texel state) in both directions**. Conversion is a sparse matrix product (about 2 ms for positions). |
| `SemanticShapeController` | Six sliders (jaw width, cheekbone prominence, chin projection, nose bridge, brow arch, face fullness) become shape-coefficient offsets `delta_beta` for the topology's own linear basis. |
| API | `GET /api/v1/facesets/{id}/identity-asset/mesh?template=makehuman&topology=flame2023_open&jaw_width=1.5&...` and `GET /api/v1/morphable-topologies`. |

```bash
curl "http://127.0.0.1:8030/api/v1/facesets/$ID/identity-asset/mesh?template=makehuman&topology=flame2023_open&chin_projection=-1&brow_arch=2"
curl  http://127.0.0.1:8030/api/v1/morphable-topologies
```
```python
import requests
r = requests.get(f"{base}/facesets/{fid}/identity-asset/mesh",
                 params={"template": "makehuman", "topology": "flame2023_open", "jaw_width": 1.5}).json()
r["topology"]["attribution"], r["semantic"]["achieved_mm"], r["topology"]["bridge"]["to_target"]
```
`topology` and the sliders apply to `template=makehuman` assets only (the canonical template has no basis or landmark set); passing them with
`canonical` is a 422 (`TOPOLOGY_REQUIRES_MAKEHUMAN_TEMPLATE`). Omitting both leaves every existing response unchanged
for `canonical`; `makehuman` responses gain `topology` and `semantic` blocks.

## Attribution (CC-BY-4.0)

FLAME 2023 Open requires attribution; it is returned in `topology.attribution` of every response that uses it, in `/morphable-topologies`, and
in `python roop_3d/audit/local_models.py --verify`. The file lives at `app/roop_3d/.runtime/models/flame2023_open/flame2023_Open.pkl` or
`flame/flame2023/flame2023_Open.pkl` (both ignored by git), or any path inside the checkout named by `ROOP3D_FLAME2023_OPEN`.

## Measured behaviour (this machine, real assets)

* Registration FLAME -> MakeHuman: similarity scale 0.944, rigid p95 6.7 mm, trimmed mean 1.41 mm rigid and 1.12 mm after the TPS (0 folded triangles).
* Coverage: 94.6 % of FLAME vertices and 92.3 % of MakeHuman vertices have a counterpart within 6 mm; the rest (eyeballs, mouth and nostril interiors, some neck) are clamped and flagged, never invented.
* Albedo transfer, checked geometrically (texture colour = 3D position): 1.5 mm mean error in both directions, the same as the closest-point distance.
* Semantic sliders, re-measured **on the output mesh**: the moved measure changes by `value x unit` (to 0.05 mm) and every other measure, including head width / face height / face depth, by under 0.05 mm, on both bases.

## Honest limits

* **"Orthogonal" is implemented as decoupled in measure space, not orthogonal in coefficient space.** Each direction is the minimum-norm solution that moves one measure and holds the other five and the three head-size measures at zero. That makes the sliders independent *as the user perceives them* (verified above), but the six `delta_beta` vectors are not mutually orthogonal (largest pairwise cosine 0.51 MakeHuman, 0.48 FLAME; both are in `SemanticShapeController.diagnostics()`). Both properties at once are impossible unless the measures are statistically uncorrelated.
* `flame2023_open` ships **no UV layout and no landmark embedding**. The UV layout used here is *derived* from the CC0 MakeHuman atlas through the bridge (labelled `uv_provenance`, 1.4 % overlapped texels, vertices split 5023 -> 5664); it is not an official FLAME UV set. The 68-point landmarks FLAME's sliders measure are the MakeHuman set carried across the bridge, and the MakeHuman set is itself derived (about 5 mm, see `makehuman_landmarks`), not annotated.
* Correspondence is closest-point, not anatomical; lateral rest-pose differences between the two surfaces are dropped (only the normal offset is carried).
* Slider units are physical (millimetres of a defined measure: 4, 2, 3, 2, 1.5, 1.5), not population standard deviations. FLAME's PCA basis moves the brow arch poorly (about 5 coefficient units per 1.5 mm), so large brow requests hit the |beta| <= 3 limit; the response then reports `clamped: true`, `limit_scale` and the *achieved* millimetres. MakeHuman is limited to |beta| <= 1 (its slider range).
* Sliders edit the geometry only; the atlas is untouched, so a strongly edited face keeps the texture of the original surface positions.
* The identity-asset fit is still 5-point (or derived 68-point) landmark based: this stage changes how the fitted surface is expressed and edited, not how well the identity is measured.
