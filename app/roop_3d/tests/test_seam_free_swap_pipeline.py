"""Stage 6 end to end on the real MakeHuman head: jaw-width adaptation -> rasterised swap mask -> z-buffer / normals into the
occlusion compositor -> mouth occlusion into the audio-locked director.

What each test establishes (and what it does not):

* The jaw-width seam is GEOMETRIC here: the area where the source's swap region and the target's jaw outline disagree. The test
  measures that area before and after the Laplacian adaptation. It does not measure pixel seams on a real photograph; no real
  swapped footage is evaluated in this suite.
* The compositor is fed the depth, normals and mask the rasteriser produces for the same mesh, so the cues are exercised on
  consistent real geometry rather than hand-written arrays.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from audio_world import speech  # noqa: E402

from roop_3d.editing import AudioVisemeTrack, DirectorSettings, ExpressionAndAudioDirector  # noqa: E402
from roop_3d.reconstruction.makehuman_build import ASSET_NAME, DEFAULT_ASSET_DIR  # noqa: E402
from roop_3d.reconstruction.rbf_deform import welded_vertex_normals  # noqa: E402
from roop_3d.rendering.strand_occlusion import OcclusionInputs, StrandAwareOcclusionCompositor  # noqa: E402
from roop_3d.rendering.swap_region_3d import (  # noqa: E402
    MOUTH_LANDMARKS, SwapRegionConfig, WeldedGraph, adapt_jaw, build_swap_mask_3d, derive_ear_anchors,
    jaw_target_from_landmarks, jaw_target_from_ratios)
from roop_3d.rendering.vertex_raster import orthographic_pixels, rasterize_vertex_attributes  # noqa: E402
from roop_3d.tracking.video_tracker import EXPRESSION_CHANNELS  # noqa: E402

pytestmark = pytest.mark.skipif(not (DEFAULT_ASSET_DIR / ASSET_NAME).exists(), reason="MakeHuman head asset not built")

SIDE = 384
SCALE, CENTER = 1500.0, (192.0, 215.0)
CFG = SwapRegionConfig(include_jawline=True, include_ears=True)
MOUTH_OPEN = EXPRESSION_CHANNELS.index("mouth_open")


@pytest.fixture(scope="module")
def head():
    from roop_3d.reconstruction.makehuman_template import makehuman_template
    mesh = makehuman_template()
    v = mesh.vertices.astype(np.float64)
    return v, np.asarray(mesh.faces, np.int64), np.asarray(mesh.landmark_vertices["68pt"], np.int64), WeldedGraph(v, mesh.faces)


def render(vertices, faces, attributes):
    pixels, depth = orthographic_pixels(vertices, scale=SCALE, center_px=CENTER)
    return rasterize_vertex_attributes(pixels, depth, faces, attributes, (SIDE, SIDE)), pixels


# =========================================================================== jaw-width seam
def test_laplacian_jaw_adaptation_removes_the_jaw_width_mismatch(head):
    v, f, lm, graph = head
    # The "target head" has a 14 % narrower jaw (its own geometry); the source is the wide-jawed original.
    target_mesh = adapt_jaw(v, f, lm, jaw_target_from_ratios(v, lm, width_ratio=0.86, config=CFG), CFG, graph=graph).vertices.astype(np.float64)
    ears = derive_ear_anchors(v, lm)
    swap_target = jaw_target_from_landmarks(v[lm], target_mesh[lm], target_ears3d=target_mesh[ears])
    adapted = adapt_jaw(v, f, lm, swap_target, CFG, graph=graph)

    def region(vertices, config):
        mask = build_swap_mask_3d(vertices, f, lm, config, graph=graph)
        raster, _ = render(vertices, f, mask.weights)
        return raster.attributes[..., 0] > 0.5

    def disagreement(a, b):
        return float((a ^ b).sum() / b.sum()), float((a & ~b).sum() / b.sum())

    target_region = region(target_mesh, CFG)
    before, spill_before = disagreement(region(v, CFG), target_region)
    after, spill_after = disagreement(region(adapted.vertices.astype(np.float64), CFG), target_region)
    assert before > 0.04 and spill_before > 0.02              # the unadapted swap region really does spill past the narrower jaw
    assert after < 0.25 * before and spill_after < 0.25 * spill_before
    r = adapted.report
    assert r.jaw_width_after_m == pytest.approx(r.jaw_width_target_m, rel=0.01) and r.folded_triangles == 0
    # the oval-only mask (the old behaviour) has no jaw band to align at all: it never reaches the contour the seam sits on
    oval = SwapRegionConfig()
    assert region(v, oval).sum() < region(v, CFG).sum()


def test_inner_oval_only_swap_is_untouched_by_the_adaptation_machinery(head):
    v, f, lm, graph = head
    result = adapt_jaw(v, f, lm, jaw_target_from_ratios(v, lm, width_ratio=0.8), SwapRegionConfig(), graph=graph)
    assert np.array_equal(result.vertices, v.astype(np.float32))


# =========================================================================== compositor on the rasterised geometry
@pytest.fixture(scope="module")
def buffers(head):
    v, f, lm, graph = head
    mask = build_swap_mask_3d(v, f, lm, CFG, graph=graph)
    normals = welded_vertex_normals(v, f)
    raster, pixels = render(v, f, np.column_stack([normals, mask.weights]))
    coverage = raster.coverage
    swap_alpha = np.clip(raster.attributes[..., 3], 0, 1) * coverage
    rng = np.random.default_rng(0)
    frame = np.clip(np.full((SIDE, SIDE, 3), (150, 170, 205), np.float32) + rng.normal(0, 3, (SIDE, SIDE, 3)), 0, 255).astype(np.uint8)
    return {"raster": raster, "normals": raster.attributes[..., :3], "swap": swap_alpha, "frame": frame, "mask": mask,
            "pixels": pixels, "graph": graph, "lm": lm, "v": v, "f": f}


def test_depth_and_normals_from_the_same_mesh_gate_the_swap(buffers):
    comp = StrandAwareOcclusionCompositor()
    b = buffers
    clear = comp.evaluate(OcclusionInputs(b["frame"], b["swap"], face_depth=b["raster"].depth, face_normals=b["normals"]))
    assert b["swap"].sum() > 5000 and np.all(clear.alpha <= b["swap"] + 1e-6)
    nz = b["normals"][..., 2]
    face_on = (b["swap"] > 0.9) & (nz > 0.8)
    grazing = (b["swap"] > 0.9) & (nz < 0.2)
    assert face_on.sum() > 500 and grazing.sum() > 50
    assert clear.alpha[face_on].mean() > 0.95 and clear.alpha[grazing].mean() < 0.4          # the swap is faded where the skin turns away
    assert clear.report.grazing_share > 0.02 and clear.report.occluded_share == 0

    # a hand-like proxy 5 cm nearer than any face point, covering the right-hand side of the image
    hand = np.full(b["frame"].shape[:2], np.inf, np.float32)
    hand[:, 220:] = float(b["raster"].depth[np.isfinite(b["raster"].depth)].min()) - 0.05
    blocked = comp.evaluate(OcclusionInputs(b["frame"], b["swap"], face_depth=b["raster"].depth, face_normals=b["normals"],
                                            occluder_depth=hand))
    assert blocked.alpha[:, 228:].max() == 0 and blocked.components["depth"][:, 228:][b["swap"][:, 228:] > 0].min() == 1.0
    left = slice(0, 210)
    assert np.abs(blocked.alpha[:, left] - clear.alpha[:, left]).max() < 1e-6              # and nothing left of it changed
    assert blocked.report.removed_share["depth"] > 0.05


def test_the_z_buffer_resolves_self_occlusion_before_the_compositor_sees_it(buffers):
    """A face point hidden behind a nearer surface of the same head must not be painted: the rasteriser's depth owns that."""
    b = buffers
    raster = b["raster"]
    covered = raster.face_id >= 0
    # every covered pixel's depth is the nearest of all triangles there: no pixel carries a farther surface's weight
    assert np.isfinite(raster.depth[covered]).all() and np.isinf(raster.depth[~covered]).all()
    pixels, depth = b["pixels"], -b["v"][:, 2]
    nose = b["lm"][30]
    px, py = pixels[nose].astype(int)
    assert raster.depth[py, px] <= depth[nose] + 5e-3                                       # the nose is the nearest surface at its pixel
    back = int(np.argmin(b["v"][:, 2]))                                                      # farthest vertex: back of the head
    bx, by = pixels[back].astype(int)
    assert raster.depth[by, bx] < depth[back] - 0.05                                         # hidden behind the face from the front


# =========================================================================== occlusion -> audio lock
def test_a_hand_over_the_mouth_measured_by_the_compositor_engages_the_audio_lock(buffers):
    b = buffers
    comp = StrandAwareOcclusionCompositor()
    near_mouth = b["graph"].geodesic(b["lm"][list(MOUTH_LANDMARKS)], limit=0.02)
    mouth_vertices = np.isfinite(near_mouth).astype(np.float64)
    mouth_img, _ = render(b["v"], b["f"], mouth_vertices)
    mouth = mouth_img.attributes[..., 0]
    assert mouth.sum() > 300
    hand = (cv2.dilate((mouth > 0).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0).astype(np.float32)   # covers the mouth, 7 px margin
    bare = comp.evaluate(OcclusionInputs(b["frame"], b["swap"], face_normals=b["normals"]))
    covered = comp.evaluate(OcclusionInputs(b["frame"], b["swap"], face_normals=b["normals"], object_occluder=hand))
    assert bare.occlusion_in(mouth) < 0.05 and covered.occlusion_in(mouth) > 0.9

    track = AudioVisemeTrack.from_signal(speech())
    director = ExpressionAndAudioDirector(DirectorSettings(audio_lock_enabled=True), audio=track)
    dt = 1 / 30
    frames = []
    for i in range(int(4.0 * 30)):
        t = i * dt
        psi = np.zeros(len(EXPRESSION_CHANNELS))
        psi[MOUTH_OPEN] = 0.3 * track.at(t).opening if t < 2.0 else 0.1                   # tracking until the hand arrives, then held
        share = bare.occlusion_in(mouth) if t < 2.0 else covered.occlusion_in(mouth)
        frames.append(director.direct(t, psi, None, mouth_occlusion=share))
    assert all(o.audio_weight == 0 for o in frames[:60]) and frames[-1].audio_weight > 0.99
    locked = frames[int(2.6 * 30):]
    mouth_open = np.array([o.psi[MOUTH_OPEN] for o in locked])
    heard = np.array([track.at(2.6 + i * dt).opening for i in range(len(locked))])
    assert np.corrcoef(mouth_open, heard)[0, 1] > 0.95                                     # follows the speech, not the held 0.1
