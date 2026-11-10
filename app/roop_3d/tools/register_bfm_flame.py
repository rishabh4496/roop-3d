"""Register 3DDFA-V3 BFM-35709 and FLAME-2023-Open surfaces and derive BFM anatomical masks.

Run from app/:
  python -B -m roop_3d.tools.register_bfm_flame --bfm .../face_model.npy --flame-open .../flame2023_Open.pkl \
      --masks .../FLAME_masks.pkl --flame-anatomy .../flame_anatomy.npz --eyelids-left l_eyelid.npy \
      --eyelids-right r_eyelid.npy --hrn-uvs .../bfm_uvs2.npy --output-dir .../prepared/registration

Method (geometric, no learning): similarity from semantic centroids, trimmed similarity ICP on the FLAME facial
surface, then thin-plate-spline non-rigid refinement driven by semantic anchors (mouth corners, lip borders, nose tip,
eye-opening centroids) plus closest-point pairs. Surface transfer is exact closest-point barycentric projection in
both directions. The report records measured residuals; it is NOT evidence of expression-recovery quality.
All inputs must be inspected official bytes inside this checkout; FLAME/BFM pickles are trusted one-time imports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from roop_3d.models.owned_files import owned_path


def _sha(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def closest_on_triangles(points, vertices, triangles, k=12):
    """Exact nearest surface point (Ericson): returns (triangle ids, convex barycentric weights, distances)."""
    from scipy.spatial import cKDTree
    incident = [[] for _ in range(len(vertices))]
    for index, triangle in enumerate(triangles):
        for vertex in triangle:
            incident[vertex].append(index)
    _, nearest = cKDTree(vertices).query(points, k=k)
    ids = np.full(len(points), -1)
    weights = np.zeros((len(points), 3))
    best = np.full(len(points), np.inf)
    for i, p in enumerate(points):
        candidates = np.unique(np.concatenate([incident[v] for v in nearest[i]]))
        a, b, c = (vertices[triangles[candidates, j]] for j in range(3))
        ab, ac = b - a, c - a
        d1, d2 = ((ab * (p - a)).sum(1), (ac * (p - a)).sum(1))
        d3, d4 = ((ab * (p - b)).sum(1), (ac * (p - b)).sum(1))
        d5, d6 = ((ab * (p - c)).sum(1), (ac * (p - c)).sum(1))
        vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
        uvw = np.zeros((len(candidates), 3))
        for n in range(len(candidates)):
            if d1[n] <= 0 and d2[n] <= 0: uvw[n] = (1, 0, 0)
            elif d3[n] >= 0 and d4[n] <= d3[n]: uvw[n] = (0, 1, 0)
            elif vc[n] <= 0 and d1[n] >= 0 and d3[n] <= 0:
                t = d1[n] / (d1[n] - d3[n]); uvw[n] = (1 - t, t, 0)
            elif d6[n] >= 0 and d5[n] <= d6[n]: uvw[n] = (0, 0, 1)
            elif vb[n] <= 0 and d2[n] >= 0 and d6[n] <= 0:
                t = d2[n] / (d2[n] - d6[n]); uvw[n] = (1 - t, 0, t)
            elif va[n] <= 0 and d4[n] - d3[n] >= 0 and d5[n] - d6[n] >= 0:
                t = (d4[n] - d3[n]) / ((d4[n] - d3[n]) + (d5[n] - d6[n])); uvw[n] = (0, 1 - t, t)
            else:
                den = 1 / (va[n] + vb[n] + vc[n]); uvw[n] = (1 - vb[n] * den - vc[n] * den, vb[n] * den, vc[n] * den)
        q = uvw[:, :1] * a + uvw[:, 1:2] * b + uvw[:, 2:] * c
        distance = np.linalg.norm(q - p, axis=1)
        j = int(distance.argmin())
        ids[i], weights[i], best[i] = candidates[j], uvw[j], distance[j]
    return ids, weights, best


def similarity(source, target):
    """Least-squares s, R, t with target ~= s R source + t (Umeyama)."""
    ms, mt = source.mean(0), target.mean(0)
    xs, xt = source - ms, target - mt
    u, sv, vt = np.linalg.svd(xt.T @ xs / len(source))
    e = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
    rotation = u @ e @ vt
    scale = (sv * np.diag(e)).sum() / xs.var(0).sum()
    return rotation, scale, mt - scale * rotation @ ms


def register(args):
    from scipy.interpolate import RBFInterpolator
    bfm = np.load(owned_path(Path(args.bfm)), allow_pickle=True).item()
    verts = np.asarray(bfm["u"]).reshape(-1, 3).astype(np.float64) * .1      # native axes, metres
    tris = np.asarray(bfm["tri"]).astype(np.int64)
    parts = [np.asarray(a) for a in bfm["annotation"]]
    ldm = np.asarray(bfm["ldm68"])
    with owned_path(Path(args.flame_open)).open("rb") as stream:
        flame = pickle.load(stream, encoding="latin1")
    template = np.asarray(flame["v_template"], np.float64)
    flame_tris = np.asarray(flame["f"]).astype(np.int64)
    with owned_path(Path(args.masks)).open("rb") as stream:
        regions = pickle.load(stream, encoding="latin1")
    with np.load(owned_path(Path(args.flame_anatomy)), allow_pickle=False) as data:
        anatomy = {k: data[k].astype(bool) for k in ("face_surface", "scalp_hair", "ears", "neck")}
    eyelids = {side: np.load(owned_path(Path(path)), allow_pickle=False) for side, path in
               (("left", args.eyelids_left), ("right", args.eyelids_right))}
    uv = np.load(owned_path(Path(args.hrn_uvs)), allow_pickle=False)
    if uv.shape != (35709, 2) or template.shape != (5023, 3) or verts.shape != (35709, 3):
        raise ValueError("Unexpected BFM/FLAME topology sizes")

    centroid = lambda points, ids: points[np.asarray(ids)].mean(0)
    bfm_eyes = (centroid(verts, parts[0]), centroid(verts, parts[1]))             # right, left
    flame_eyes = (centroid(template, regions["right_eyeball"]), centroid(template, regions["left_eyeball"]))
    anchors_b = np.stack([*bfm_eyes, centroid(verts, parts[4]), centroid(verts, np.concatenate([parts[5], parts[6]]))])
    anchors_f = np.stack([*flame_eyes, centroid(template, regions["nose"]), centroid(template, regions["lips"])])
    rotation, scale, translation = similarity(anchors_f, anchors_b)
    face = anatomy["face_surface"]
    for _ in range(30):                                  # trimmed similarity ICP on the FLAME facial surface
        moved = (scale * (rotation @ template.T)).T + translation
        ids, weights, distance = closest_on_triangles(moved[face], verts, tris)
        nearest = (weights[:, :, None] * verts[tris[ids]]).sum(1)
        keep = distance <= np.quantile(distance, .9)
        r2, s2, t2 = similarity(moved[face][keep], nearest[keep])
        rotation, scale, translation = r2 @ rotation, s2 * scale, s2 * (r2 @ translation) + t2
    placed = (scale * (rotation @ template.T)).T + translation

    # Non-rigid refinement anchors.
    lips = np.asarray(regions["lips"])
    centre = lips[np.abs(placed[lips, 0]) < .003]
    flame_anchor = {"mouth_pos": lips[placed[lips, 0].argmax()], "mouth_neg": lips[placed[lips, 0].argmin()],
                    "lip_top": centre[placed[centre, 1].argmax()], "lip_bottom": centre[placed[centre, 1].argmin()],
                    "nose_tip": np.asarray(regions["nose"])[placed[np.asarray(regions["nose"]), 2].argmax()]}
    pos, neg = (ldm[54], ldm[48]) if verts[ldm[54], 0] > verts[ldm[48], 0] else (ldm[48], ldm[54])
    bfm_anchor = {"mouth_pos": pos, "mouth_neg": neg, "lip_top": ldm[51], "lip_bottom": ldm[57], "nose_tip": ldm[30]}
    src = [placed[flame_anchor[k]] for k in bfm_anchor]
    dst = [verts[bfm_anchor[k]] for k in bfm_anchor]
    for side, key, part in (("left", "left_eye_region", 1), ("right", "right_eye_region", 0)):
        active = np.where(np.linalg.norm(eyelids[side], axis=1) > 1e-6)[0]
        active = active[np.isin(active, regions[key])]
        src.append(placed[active].mean(0)); dst.append(verts[parts[part]].mean(0))
    src, dst = np.asarray(src), np.asarray(dst)
    anchor_error_before = np.linalg.norm(src - dst, axis=1) * 1000
    warped = placed.copy()
    for iteration in range(8):
        ids, weights, distance = closest_on_triangles(warped[face], verts, tris)
        nearest = (weights[:, :, None] * verts[tris[ids]]).sum(1)
        keep = distance <= np.quantile(distance, .85)
        chosen = np.where(face)[0][keep][::3]
        x = np.vstack([src, placed[chosen]]); y = np.vstack([dst, nearest[keep][::3]])
        field = RBFInterpolator(x, y - x, kernel="thin_plate_spline", smoothing=1e-5 * (1 + 3 * (8 - iteration)), degree=1)
        warped = placed + field(placed)

    # Exact barycentric transfer in both directions on the registered surfaces.
    f_ids, f_w, f_d = closest_on_triangles(warped, verts, tris)                  # FLAME vertex -> BFM triangle
    b_ids, b_w, b_d = closest_on_triangles(verts, warped, flame_tris)            # BFM vertex -> FLAME triangle
    bfm_to_flame_indices, bfm_to_flame_weights = tris[f_ids], f_w
    flame_to_bfm_indices, flame_to_bfm_weights = flame_tris[b_ids], b_w
    flame_hrn_uvs = (f_w[:, :, None] * uv[tris[f_ids]]).sum(1)
    if np.any((flame_hrn_uvs < 0) | (flame_hrn_uvs > 1)):
        raise ValueError("Registered UVs left [0,1]")
    # Triangles of a single BFM surface never straddle a UV seam (per-vertex chart), so this is a single-chart lookup.

    nearest_flame = flame_tris[b_ids][np.arange(len(b_ids)), b_w.argmax(1)]
    bfm_anatomy = {name: mask[nearest_flame] for name, mask in anatomy.items()}
    allowed = bfm_anatomy["face_surface"] & ~(bfm_anatomy["scalp_hair"] | bfm_anatomy["ears"] | bfm_anatomy["neck"])
    names = ["right_eye", "left_eye", "right_eyebrow", "left_eyebrow", "nose", "up_lip", "down_lip", "skin"]
    membership = np.zeros((8, 35709), bool)
    for index, ids_ in enumerate(parts):
        membership[index, ids_] = True
    allowed &= membership.any(0)

    from scipy.spatial import cKDTree
    _, nearest_vertex = cKDTree(warped).query(verts)
    lips_mask = np.zeros(5023, bool); lips_mask[regions["lips"]] = True
    nose_mask = np.zeros(5023, bool); nose_mask[regions["nose"]] = True
    bfm_lips = np.zeros(35709, bool); bfm_lips[parts[5]] = bfm_lips[parts[6]] = True
    bfm_nose = np.zeros(35709, bool); bfm_nose[parts[4]] = True
    landmark_mm = {str(i): float(closest_on_triangles(verts[[ldm[i]]], warped, flame_tris)[2][0] * 1000)
                   for i in (8, 30, 36, 39, 42, 45, 48, 51, 54, 57)}
    report = {
        "method": "similarity ICP + thin-plate non-rigid refinement + closest-point barycentric transfer",
        "similarity": {"scale": float(scale), "rotation": rotation.tolist(), "translation_m": translation.tolist()},
        "anchor_error_before_nonrigid_mm": anchor_error_before.tolist(),
        "flame_face_to_bfm_surface_mm": {"mean": float(f_d[face].mean() * 1000),
                                           "p95": float(np.percentile(f_d[face], 95) * 1000), "max": float(f_d[face].max() * 1000)},
        "bfm_to_flame_surface_mm": {"mean": float(b_d.mean() * 1000), "p95": float(np.percentile(b_d, 95) * 1000)},
        "bfm_landmark_to_flame_surface_mm": landmark_mm,
        "bfm_lips_vertices_landing_on_flame_lips": float(lips_mask[nearest_vertex][bfm_lips].mean()),
        "bfm_nose_vertices_landing_on_flame_nose": float(nose_mask[nearest_vertex][bfm_nose].mean()),
        "bfm_active_face_vertices": int(allowed.sum()),
        "inputs_sha256": {"bfm": _sha(args.bfm), "flame_open": _sha(args.flame_open), "masks": _sha(args.masks),
                          "flame_anatomy": _sha(args.flame_anatomy)},
        "caveats": ["geometric registration only; no learned or measured expression-recovery validation",
                    "eyelid/inner-mouth correspondence is limited by the template geometry and has no landmark embedding"]}
    output = owned_path(Path(args.output_dir))
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / "registration.npz", output / "bfm_anatomy.npz", output / "registration.report.json"]
    if any(path.exists() for path in targets):
        raise ValueError("Registration outputs already exist; choose an empty output directory")
    np.savez_compressed(targets[0], bfm_to_flame_indices=bfm_to_flame_indices.astype(np.int32),
                        bfm_to_flame_weights=bfm_to_flame_weights.astype(np.float32),
                        flame_to_bfm_indices=flame_to_bfm_indices.astype(np.int32),
                        flame_to_bfm_weights=flame_to_bfm_weights.astype(np.float32),
                        flame_to_bfm_rotation=rotation.astype(np.float32), flame_to_bfm_scale=np.float32(scale),
                        flame_to_bfm_translation_m=translation.astype(np.float32),
                        flame_hrn_uvs=flame_hrn_uvs.astype(np.float32))
    np.savez_compressed(targets[1], **bfm_anatomy)
    targets[2].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**report, "files": {p.name: _sha(p) for p in targets[:2]}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("bfm", "flame-open", "masks", "flame-anatomy", "eyelids-left", "eyelids-right", "hrn-uvs", "output-dir"):
        parser.add_argument("--" + name, required=True)
    register(parser.parse_args())


if __name__ == "__main__":
    main()
