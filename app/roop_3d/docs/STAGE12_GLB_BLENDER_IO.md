# Stage 12: binary glTF export and the optional Blender bridge

Path mapping: the prompt's `app/roop_3d/io/` is the new package `roop_3d/io/` (it did not exist), the "existing JSON `/mesh`
endpoint" is `server/identity_assets.py`, and tests live in `roop_3d/tests/` (run from `app/`).
(Not to be confused with `STAGE12_MODEL_GOVERNANCE.md`, a different series.)

## What was added

| Piece | Where |
|---|---|
| `GLBBinarySerializer`, `MorphTarget`, `HeadTrack`, `parse_glb`, PNG encoder, normal-map + tangent helpers | `io/glb_export.py` |
| `BlenderMCPBridge`, `EXPORTER_SCRIPT` | `io/blender_bridge.py` |
| `GET /api/v1/facesets/{id}/identity-asset/mesh.glb` | `server/identity_assets.py` |
| `GET /api/v1/blender/exporter-script`, `GET /api/v1/blender/status`, `POST /api/v1/facesets/{id}/identity-asset/send-to-blender` | `server/blender_routes.py` |
| Tests (56 + 1 real-asset) | `tests/test_glb_export.py` |

`/mesh` and `/mesh.glb` now share `_prepare_mesh`, so both serve the same topology, sliders, albedo prior and atlas. The JSON route's
response is unchanged (existing tests pass untouched).

## `/mesh.glb`

Same query as `/mesh` (`template`, `topology`, six slider values, `albedo_prior`) plus `shape_keys`, `normal_map` (default true) and
`download`. Content type `model/gltf-binary`.

* Geometry is converted from the y-down identity-asset frame to glTF (y up, right-handed, winding swapped), metres.
* Albedo is the display atlas of `/mesh` (unseen texels grey, or prior-filled with `albedo_prior=true`).
* `shape_keys=true` (template `makehuman` only): the base mesh is the **un-slid** mesh and the six semantic sliders are morph targets
  named after the slider keys (`mesh.extras.targetNames`, which three.js and Blender read). The mesh `weights` reproduce the requested
  sliders (times the controller's one whole-move limit scale), so `base + sum(weight * target)` equals what the JSON route returns for the
  same sliders (tested to 2e-5 m). Without `shape_keys`, sliders are baked into the vertices as in `/mesh`.
* `_CONFIDENCE` / `_STATE` per-vertex attributes and the manifest evidence are carried in the file (`extras.roop3d_identity_asset`).
* Normal map: baked from the asset's `displacement_uv_m` when it holds measured relief, with `TANGENT` vertices. **Real canonical assets
  currently have a flat displacement field, so today no normal map is emitted for them**; `extras.roop3d_identity_asset.normal_map` says why
  (and it is omitted for non-default topologies, whose UV layout differs).
* Not included: morph *normal* deltas (the serializer supports them; the route does not compute them, so shape-key shading uses base normals).

## Head track (library level)

`HeadTrack.from_tracked_frames(frames)` turns `tracking.video_tracker.TrackedHeadFrame`s into a glTF animation on the head node
(translation, rotation and optionally morph weights). The tracker's camera is weak-perspective, so rotation (3 DoF) and in-plane position
(2 DoF) are exact, but **depth is not metric**: translation Z is 0 and the per-key `scale` (px/m) is stored in the animation `extras`.
The track is not served by any route yet because nothing persists tracker output (see the Stage 4 head-tracking note: not wired into the job runner).

## Blender bridge and GPL isolation

* `roop_3d` never imports `bpy`, `bmesh` or `mathutils`; `test_roop_3d_never_imports_bpy` AST-scans every source file in the package.
* The bridge is a loopback-only TCP client (non-loopback hosts are refused; the HTTP routes take a port, never a host). It sends text to a
  Blender-side MCP server the user starts. The import snippet it sends is a string with the path embedded via `json.dumps`.
* `EXPORTER_SCRIPT` is the standalone script for the user to run inside Blender (imports a `.glb` path or loopback URL; `export_glb()` writes
  back with shape keys and animation). It carries a `GPL-3.0-or-later` header, is served as a download, and is never imported or run here.
* **Unverified:** the wire framing of the default `blendermcp` dialect (port 9876, `{"type","params"}` -> `{"status","result"|"message"}`) is
  from a summary of the public `ahujasid/blender-mcp` add-on, not checked against a running Blender; the tests use a fake server that
  implements that assumption. The reader accepts any JSON reply (newline-terminated or not) and also JSON-RPC 2.0 (`dialect="jsonrpc"`).
  The Blender-side snippet and script were also never run in Blender (it is not installed here); the script is only compile-checked.

## Measured performance (4,481-vertex MakeHuman head, this Windows box)

* Geometry only: 0.25 ms median; with 6 morph targets ~0.4 ms.
* With a 1024x1024 atlas already in the PNG cache: ~3.6 ms.
* **First** PNG encode of a 1024x1024 atlas (worst case: random noise): ~58 ms. The 15 ms budget therefore holds for geometry plus a cached
  atlas, not for a cold 1024 atlas. Re-requests of an unchanged atlas hit a content-hash LRU cache.

## Not done

* The React viewport still loads the JSON `/mesh`; switching it to `GLTFLoader`, and the "Export .glb / Send to Blender" buttons, are UI work
  that was out of scope for this backend-only prompt. The routes are ready for them.
* No sculpt round-trip back into the `.fsz` (the script can export a `.glb` from Blender; nothing re-imports it).
