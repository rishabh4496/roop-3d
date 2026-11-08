"""Independent directional-light math and real hardware shader regressions."""
from dataclasses import replace
from types import SimpleNamespace
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from roop_3d.compositor.pipeline import CompositeOptions
from roop_3d.reconstruction.photometry import HeadPhotometry, decompose_albedo, fit_sh, linearize, extend_neck
from roop_3d.reconstruction.photometric_gpu import renderer, release_renderer
from roop_3d.reconstruction.face_detail import extract_detail


@pytest.fixture(autouse=True)
def release_gl():
    yield
    release_renderer()


def encode(linear):
    linear = np.maximum(linear, 0)
    return np.clip(np.where(linear <= .0031308, linear*12.92,
                           1.055*linear**(1/2.4)-.055)*255, 0, 255).round().astype(np.uint8)


def basis(n):
    # Independent scalar polynomial reference; never call production evaluator.
    x, y, z = np.moveaxis(n, -1, 0)
    return np.stack([np.ones_like(x)*.886227, 1.023328*x, 1.023328*y, 1.023328*z,
                     .858086*x*y, .858086*x*z, .858086*y*z,
                     .429043*(x*x-y*y), .743125*(3*z*z-1)], -1)


def geometry(side=128):
    y, x = np.mgrid[-1:1:complex(side), -1:1:complex(side)]
    # Metric depth with a protruding nose, brow and recessed oral cavity.
    depth = (1 - .035*np.exp(-x*x/.015-y*y/.18)
             - .012*np.exp(-(y+.35)**2/.012)*np.exp(-x*x/.5)
             + .009*np.exp(-x*x/.06-(y-.35)**2/.012)).astype('f4')
    dy, dx = np.gradient(depth, .24/(side-1), .24/(side-1))
    n = np.stack((dx, dy, np.ones_like(dx)), -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    source = np.zeros((9, 3), np.float32)
    source[0] = 1/.886227
    photo = HeadPhotometry(n, n, n, depth, np.ones_like(depth), source,
        np.array([[1, 0, 0], [0, 1, 0]], np.float32),
        np.array([[side/.24, 0, (side-1)/2], [0, side/.24, (side-1)/2], [0, 0, 1]], np.float32))
    return photo


def lights():
    # BGR: blue right rim + red left rim, and a neutral overhead key.
    split = np.zeros((9, 3), np.float32)
    split[0] = np.array([.55, .18, .55])/.886227
    split[1] = [.5/1.023328, 0, -.5/1.023328]
    top = np.zeros_like(split)
    top[0] = .48/.886227
    top[2] = -.45/1.023328
    top[3] = .08/1.023328
    return split, top


@pytest.mark.parametrize('light', range(2), ids=['red-blue-split', 'overhead-shadow'])
def test_gpu_ratio_matches_independent_normal_shading_and_preserves_texture(light):
    photo = geometry()
    sh = lights()[light]
    yy, xx = np.indices(photo.depth.shape)
    albedo = np.ones((*photo.depth.shape, 3), np.float32)*.28
    albedo *= (1 + .06*((xx+yy) % 2*2-1))[..., None]  # identity-scale high frequencies
    source = encode(albedo)
    expected = np.maximum(basis(photo.normals) @ sh, .01)
    target = encode(expected*.5)
    mask = np.ones_like(photo.depth)
    mask[:8] = 0  # foreground veto must be exact
    result, _ = renderer().render(photo, source, target, mask, sh,
        strength=1, specular=0, shadows=0)
    reference = encode(linearize(source)*expected)
    np.testing.assert_allclose(result[mask>0].astype(float), reference[mask>0], atol=1)
    np.testing.assert_array_equal(result[mask==0], source[mask==0])
    recovered = linearize(result)/expected
    active = mask>.9
    assert np.corrcoef(recovered[active].ravel(), linearize(source)[active].ravel())[0, 1] > .98
    # Shadow/chromatic gradients must follow the actual normal direction.
    if light == 0:
        left = photo.normals[..., 0] < -.3
        right = photo.normals[..., 0] > .3
        assert result[right, 0].mean() > result[left, 0].mean()*1.2
        assert result[left, 2].mean() > result[right, 2].mean()*1.2
    else:
        up = photo.normals[..., 1] < -.2
        down = photo.normals[..., 1] > .2
        assert result[up].mean() > result[down].mean()*1.15


def test_robust_skin_fit_ignores_occluders_and_recovers_albedo():
    photo = geometry()
    sh = lights()[0]
    expected = np.maximum(basis(photo.normals) @ sh, .01)
    image = encode(.5*expected)
    mask = np.ones_like(photo.depth)
    image[20:45, 15:55] = [0, 255, 0]
    mask[20:45, 15:55] = 0
    recovered = fit_sh(photo.normals, linearize(image), mask)
    predicted = np.maximum(basis(photo.normals) @ recovered, .01)
    assert np.mean(np.abs(predicted[mask>0] - expected[mask>0])) < .006
    decomposition = decompose_albedo(image, photo.normals, mask)
    # Colored illumination/reflectance has an unavoidable global ambiguity;
    # decomposition must still recompose the observed linear image.
    reconstructed = decomposition.albedo * np.maximum(basis(photo.normals) @ decomposition.sh_source,.01)
    assert np.mean(np.abs(reconstructed[mask>0] - linearize(image)[mask>0])) < .003
    # No usable visible skin means neutral light, finite outputs, no NaNs.
    neutral = fit_sh(photo.normals, linearize(image), mask*0)
    np.testing.assert_allclose(basis(photo.normals) @ neutral, 1, atol=1e-6)


def test_gpu_ao_and_key_shadow_are_geometric_and_no_light_behind_surface_specular():
    photo = geometry()
    source = np.full((128, 128, 3), 155, np.uint8)
    sh = lights()[1]
    target = encode(np.maximum(basis(photo.normals) @ sh, .01)*.5)
    mask = np.ones_like(photo.depth)
    plain, _ = renderer().render(photo, source, target, mask, sh, strength=1, specular=0, shadows=0)
    shadowed, _ = renderer().render(photo, source, target, mask, sh, strength=1, specular=0, shadows=1)
    assert np.all(shadowed.astype(int) <= plain.astype(int)+1)
    assert (plain.astype(float)-shadowed).max() > 8
    # Uniform plane: AO should not manufacture a shadow from skin texture.
    n = np.zeros_like(photo.normals)
    n[..., 2] = 1
    flat = replace(photo, normals=n, rest_normals=n, depth=np.ones_like(photo.depth))
    plane, _ = renderer().render(flat, source, np.full_like(source, 188), mask,
        flat.sh_source, strength=1, specular=0, shadows=1)
    np.testing.assert_allclose(plane, source, atol=1)
    behind = flat.sh_source.copy()
    behind[3] = -.1/1.023328
    no_spec, _ = renderer().render(flat, source, np.full_like(source, 188), mask,
        behind, strength=1, specular=0, shadows=0)
    with_spec, _ = renderer().render(flat, source, np.full_like(source, 188), mask,
        behind, strength=1, specular=1, shadows=0)
    np.testing.assert_array_equal(with_spec, no_spec)


def dense_fixture():
    mesh = np.full((478, 3), [64., 64., 0.], np.float32)
    for corners, lids, contour, iris_ids, center in (
        ((33,133),(159,145),(33,160,158,133,153,144),(468,469,470,471,472),(42.25,44.5)),
        ((362,263),(386,374),(362,385,387,263,373,380),(473,474,475,476,477),(86.75,44.25)),
    ):
        cx, cy = center
        mesh[list(contour), :2] = np.array([[-12,0],[-6,-5],[6,-5],[12,0],[6,5],[-6,5]]) + [cx,cy]
        mesh[list(lids), :2] = [[cx, cy-5], [cx, cy+5]]
        mesh[list(iris_ids), :2] = np.array([[1.25,.25],[4.25,.25],[1.25,-2.75],[-1.75,.25],[1.25,3.25]])+[cx,cy]
    aperture = [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]
    a = np.linspace(-np.pi, np.pi, len(aperture), endpoint=False)
    mesh[aperture, :2] = np.column_stack((64+18*np.cos(a), 88+12*np.sin(a)))
    mesh[[78,308,13,14], :2] = [[46,88],[82,88],[64,76],[64,100]]
    return mesh


def test_subpixel_independent_gaze_blinks_and_468_has_no_invented_iris():
    mesh = dense_fixture()
    detail = extract_detail(mesh, None, np.eye(2,3), 128)
    assert len(detail.eyes) == 2 and all(e.visible for e in detail.eyes)
    np.testing.assert_allclose(detail.eyes[0].center, [43.5,44.75])
    assert detail.iris.max() == 1 and ((detail.iris>0) & (detail.iris<1)).any()
    mesh[468:473, 0] += 2.5
    moved = extract_detail(mesh, None, np.eye(2,3), 128)
    assert moved.eyes[0].yaw > detail.eyes[0].yaw
    assert moved.eyes[1].yaw == detail.eyes[1].yaw
    mesh[[159,145], 1] = 44.5
    blink = extract_detail(mesh, None, np.eye(2,3), 128)
    assert not blink.eyes[0].visible
    assert blink.iris[:, :64].max() == 0
    no_iris = extract_detail(mesh[:468], None, np.eye(2,3), 128)
    assert no_iris.eyes == () and no_iris.iris.max() == 0 and no_iris.mouth.max() == 1


def test_native_resolution_mouth_preserves_tooth_gaps_and_iris_lock():
    detail = extract_detail(dense_fixture(), None, np.eye(2,3), 128)
    source = np.full((512,512,3), 235, np.uint8) # enhancer's white-block failure
    target = np.full_like(source, 60)
    target[320:365,210:310] = 220
    target[320:365,247:251] = 8 # narrow interdental gap at high resolution
    restored = detail.condition(source, target)
    assert restored[340,249].mean() < 30
    assert restored[340,240].mean() > 190
    iris, _, _ = detail.masks(512)
    np.testing.assert_array_equal(restored[iris>.999], target[iris>.999])
    closed = replace(detail, openness=.02, iris=detail.iris*0, eye_surface=None)
    np.testing.assert_array_equal(closed.condition(source, target), source)
    # A new iris mask must not leave a second pupil at the old gaze location.
    small_source = np.full((128,128,3),200,np.uint8)
    small_source[44,37] = 0
    small_target = np.full_like(small_source,220)
    locked = detail.condition(small_source,small_target,mouth=False)
    np.testing.assert_array_equal(locked[44,37],small_target[44,37])


def test_neck_extrudes_only_open_lower_boundary_with_soft_weights():
    vertices = np.array([[-.04,0,0],[.04,0,0],[-.04,.1,0],[.04,.1,0]], np.float32)
    faces = np.array([[0,2,1],[1,2,3]])
    uv = np.array([[0,1],[1,1],[0,0],[1,0]], np.float32)
    expanded, triangles, atlas, feather = extend_neck(vertices,faces,uv)
    np.testing.assert_array_equal(expanded[:4], vertices)
    assert len(expanded)==6 and len(triangles)==4
    assert np.max(expanded[:,1])-np.max(vertices[:,1]) == pytest.approx(.008)
    np.testing.assert_array_equal(feather, [1,1,1,1,0,0])
    np.testing.assert_array_equal(atlas[4:], uv[2:])
    moving = vertices.copy()
    moving[0,1] += .2
    # Extreme jaw motion must not change the ribbon's vertex count/topology.
    animated = extend_neck(moving,faces,uv,reference_vertices=vertices)
    np.testing.assert_array_equal(animated[1],triangles)
    assert animated[0].shape == expanded.shape


@pytest.mark.parametrize('native_side',(512,1024))
def test_compositor_runs_photometry_after_native_resolution_restoration(native_side):
    from roop_3d.compositor.pipeline import FrameCompositor
    photo = geometry()
    source = np.full((128,128,3),155,np.uint8)
    target = np.full_like(source,155)
    restored = np.full((native_side,native_side,3),155,np.uint8)
    restored[200:230,200:230]=165
    class Restorer:
        def restore_hires(self,*args,**kwargs): return restored.copy()
        def release(self): pass
    item = SimpleNamespace(track_id=1,inverse=np.eye(2,3),landmarks5=None,photometry=photo)
    compositor = FrameCompositor(restorer=Restorer())
    options = CompositeOptions(relight=True,color_strength=.8,restoration_blend=1,
        photometric_identity_floor=0,
        lighting_mode='manual',manual_light_yaw=60,specular_sheen=0,shadow_preservation=0)
    seen = {}
    try:
        output,_,_ = compositor._enhance(item,source,target,np.ones((128,128),np.float32),
            probabilities=None,skin=np.ones((128,128),np.float32),options=options,
            observer=lambda name,image,*args: seen.update({name:image.shape}),lip_result=None,frame_index=0)
        assert output.shape == (native_side,native_side,3)
        assert seen['normal_map'] == seen['sh_lighting'] == seen['restorer'] == output.shape
        expected,_,_ = photo.apply(restored,cv2.resize(target,(native_side,native_side)),
                                   np.ones((native_side,native_side),np.float32),
                                   np.ones((native_side,native_side),np.float32),options)
        np.testing.assert_array_equal(output,expected)
    finally:
        compositor.release()


def test_fitted_mesh_normals_rotate_and_albedo_atlas_is_real_geometry():
    from roop_3d.tests.test_full_head import head_asset
    from roop_3d.reconstruction.full_head import GPUHeadRasterizer, HeadPose, head_roi
    from roop_3d.core.pose import rotation_from_euler
    asset = head_asset()
    atlas = renderer().uv_normals(asset.vertices, asset.faces, asset.uvs, 128)
    assert atlas[...,3].sum() > 5000
    np.testing.assert_allclose(np.linalg.norm(atlas[atlas[...,3]>.5,:3], axis=-1), 1, atol=1e-5)
    r = GPUHeadRasterizer()
    pose = HeadPose(rotation_from_euler(50,10,0), np.array([0,0,.9]), np.array([[800,0,320],[0,800,320],[0,0,1]]))
    try:
        buf = r.render_buffers(asset.vertices,asset.faces,asset.uvs,asset.texture,asset.texel_confidence,
                               pose,head_roi(pose.project(asset.vertices),128),128)
        active = buf.silhouette>.9
        expected = (buf.rest_normals[active]*(1,1,-1)) @ pose.rotation.T * (1,1,-1)
        np.testing.assert_allclose(buf.normals[active],expected,atol=2e-6)
    finally:
        r.release()


def test_real_arcface_similarity_above_point_78_for_neon_and_overhead():
    """Opt-in real-photo acceptance. Never substitute a fake ArcFace embedding.

    ROOP_3D_RELIGHT_CASE points to an owned JSON fixture: image, buffers (.npz
    with normals/depth/inverse/camera from fitted MakeHuman), landmarks5. Paths
    in the manifest are relative to it and must remain in this checkout.
    """
    fixture = os.environ.get('ROOP_3D_RELIGHT_CASE')
    if not fixture:
        pytest.skip('real relighting case absent: set ROOP_3D_RELIGHT_CASE to an owned fixture manifest')
    from roop_3d.models.owned_files import owned_path
    from roop_3d.models.catalog import ModelCatalog
    from roop_3d.faceset.recognizer import Recognizer
    manifest = owned_path(Path(fixture).absolute())
    case = json.loads(manifest.read_text('utf-8'))
    image = cv2.imread(str(owned_path(manifest.parent/case['image'])))
    assert image is not None
    with np.load(owned_path(manifest.parent/case['buffers']), allow_pickle=False) as arrays:
        n, depth, inverse, camera = (arrays[key].copy() for key in ('normals','depth','inverse','camera'))
        detail = (extract_detail(None,arrays['landmarks68'],np.eye(2,3),image.shape[0])
                  if 'landmarks68' in arrays else None)
    assert image.shape[:2] == n.shape[:2] == depth.shape
    assert np.std(n[...,0]) > .1 and np.std(n[...,1]) > .1
    mask = np.isfinite(depth).astype('f4')
    skin = mask * (1-detail.excluded_skin) if detail is not None else mask
    sh_source = decompose_albedo(image,n,skin).sh_source
    photo = HeadPhotometry(n,n,n,depth,mask,sh_source,inverse,camera,detail)
    recognizer = Recognizer(ModelCatalog())
    points = np.asarray(case['landmarks5'],np.float32)
    try:
        original = recognizer.embed(image, points)
        for name,sh in zip(('neon_split','overhead'),lights()):
            target = encode(np.maximum(basis(n) @ sh,.01)*.5)
            output,_ = renderer().render(photo,image,target,photo.lighting_mask(mask),sh,
                                        strength=1,specular=.1,shadows=.85)
            similarity = float(original @ recognizer.embed(output, points))
            from roop_3d.compositor.photometric_identity import constrain_identity
            output,constraint = constrain_identity(image,output,points,recognizer)
            similarity = float(original @ recognizer.embed(output, points))
            print(f'{name}: full_strength={constraint.requested_cosine:.6f}, '
                  f'accepted={similarity:.6f}, applied_fraction={constraint.applied_fraction:.5f}')
            assert similarity > .78, f'ArcFace cosine {similarity:.4f} must exceed .78'
    finally:
        recognizer.release_for_build()
