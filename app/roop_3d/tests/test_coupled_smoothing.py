"""Size-normalised, coupled smoothing of landmarks and the dense fit."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from roop_3d.engine.volumetric.preprocess import aligned_crop
from roop_3d.faceset.recognizer import ARCFACE_REFERENCE_5PTS
from roop_3d.reconstruction.dense_correspondence import ReferenceSurfaceRasterizer
from roop_3d.reconstruction.neural_face import NeuralFaceEngine
from roop_3d.temporal import CoupledTrackSmoother

FACE = np.array(((100, 100), (160, 100), (130, 130), (110, 160), (150, 160)), np.float64)


def rotation_y(degrees):
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)))


def second_difference_rms(series):
    series = np.asarray(series)
    return float(np.sqrt(np.mean((series[2:] - 2 * series[1:-1] + series[:-2]) ** 2)))


def run(smoother, points_by_frame, track=0):
    out = []
    for frame, points in enumerate(points_by_frame):
        out.append(smoother.advance(track, frame, points, timestamp=frame / 30).landmarks)
    return np.array(out)


def test_still_head_noise_drops_well_below_raw():
    rng = np.random.default_rng(1)
    raw = FACE * 3 + rng.normal(0, 1.0, (400, 5, 2))     # a 180 px inter-ocular face, 1 px of detector noise
    smooth = run(CoupledTrackSmoother(), raw)
    assert second_difference_rms(smooth[50:]) < 0.6 * second_difference_rms(raw[50:])
    assert smooth[50:].std(axis=0).mean() < 0.6 * raw[50:].std(axis=0).mean()


@pytest.mark.parametrize("speed", [0.5, 2.0, 15.0])
def test_constant_velocity_pan_lags_by_under_three_pixels(speed):
    track = [FACE + (speed * i, 0.0) for i in range(120)]
    smooth = run(CoupledTrackSmoother(), track)
    assert np.abs(smooth[60:] - np.array(track[60:])).max() < 3.0


def test_beta_is_smooth_bounded_and_monotonic_in_motion():
    betas = [CoupledTrackSmoother.beta_for(v) for v in np.linspace(0, .08, 200)]
    assert betas[0] == pytest.approx(.35) and betas[-1] == pytest.approx(.90)
    assert np.all(np.diff(betas) >= -1e-12) and np.max(np.abs(np.diff(betas))) < .02


def test_motion_is_normalised_by_face_extent_not_pixels():
    big = CoupledTrackSmoother()
    small = CoupledTrackSmoother()
    for smoother, scale in ((big, 4.0), (small, 1.0)):
        smoother.advance(0, 0, FACE * scale, timestamp=0.0)
    # The same relative motion (1.5 % of the face per frame) selects the same beta at any resolution.
    betas = [smoother.advance(0, 1, FACE * scale + (1.5 * scale, 0), timestamp=1 / 30).beta
             for smoother, scale in ((big, 4.0), (small, 1.0))]
    assert betas[0] == pytest.approx(betas[1])


def test_profile_extent_does_not_collapse_with_interocular_distance():
    profile = FACE.copy()
    profile[1] = profile[0] + (0.5, 0.0)          # both eyes on top of each other
    assert CoupledTrackSmoother.extent(profile) > 25


def test_one_beta_drives_every_channel_and_rotation_stays_on_so3():
    smoother = CoupledTrackSmoother()
    first = smoother.advance(7, 0, FACE, timestamp=0.0)
    assert first.reset and first.beta == 1.0
    first.filter("translation", (0.0, 0.0, 1.0))
    first.filter_rotation("rotation", rotation_y(0))
    second = smoother.advance(7, 1, FACE + (0.2, 0.0), timestamp=1 / 30)
    beta = second.beta
    assert .35 <= beta < .9
    np.testing.assert_allclose(second.filter("translation", (1.0, 0.0, 1.0)), (beta, 0.0, 1.0))
    rotated = second.filter_rotation("rotation", rotation_y(20))
    np.testing.assert_allclose(rotated.T @ rotated, np.eye(3), atol=1e-9)
    angle = np.degrees(np.arctan2(rotated[0, 2], rotated[0, 0]))
    assert angle == pytest.approx(20 * beta, abs=0.3)


@pytest.mark.parametrize("frame,timestamp,cut", [(12, 1.07, False), (10, 1.0, False), (11, 4.0, False), (11, 1.033, True)])
def test_gap_rewind_time_jump_and_scene_cut_restart_the_trajectory(frame, timestamp, cut):
    smoother = CoupledTrackSmoother()
    smoother.advance(0, 10, FACE, timestamp=1.0)
    step = smoother.advance(0, frame, FACE + (40, 0), timestamp=timestamp, scene_cut=cut)
    assert step.reset
    np.testing.assert_allclose(step.landmarks, FACE + (40, 0))


def test_contiguous_frame_blends_and_tracks_are_independent():
    smoother = CoupledTrackSmoother()
    smoother.advance(0, 10, FACE, timestamp=1.0)
    follow = smoother.advance(0, 11, FACE + (1, 0), timestamp=1.033)
    assert not follow.reset and 0 < follow.landmarks[0, 0] - FACE[0, 0] < 1
    assert smoother.advance(1, 11, FACE, timestamp=1.033).reset
    smoother.forget_track(0)
    assert smoother.advance(0, 12, FACE, timestamp=1.066).reset


def test_nonfinite_landmarks_are_rejected():
    with pytest.raises(ValueError):
        CoupledTrackSmoother().advance(0, 0, np.full((5, 2), np.nan))


# -- the dense engine applies it to the fit ---------------------------------------------------------------------
def noisy_engine(noise):
    rng = np.random.default_rng(5)
    base = np.array([[.2, .1, 1], [.8, .1, 1], [.2, .8, 1], [.8, .8, 1]], np.float32)

    class Backbone:
        topology = SimpleNamespace(faces=np.array([[0, 2, 1], [1, 2, 3]], np.int32))

        def fit(self, image, points):
            shift = np.array((rng.normal(0, noise), rng.normal(0, noise), 0), np.float32)
            return SimpleNamespace(camera_vertices=base, intrinsics=np.diag([224, 224, 1]),
                                   rotation=rotation_y(rng.normal(0, 3 * noise / .02)).astype(np.float32),
                                   translation=shift, scale=1.0, identity=base.copy(), vertices=base.copy(),
                                   expression_vertices=base.copy(),
                                   alignment=aligned_crop(image, np.asarray(points), 128))

    class Enhancer:
        def restore_hires(self, crop, points):
            return np.full((128, 128, 3), 200, np.uint8)

    class Swapper:
        model = SimpleNamespace(resolution=128, spec={}, infer=lambda *a, **k: None)

        def synthesize_correspondence(self, conditioning, **kwargs):
            return conditioning

    image = np.full((200, 200, 3), 60, np.uint8)
    points = ARCFACE_REFERENCE_5PTS.astype(np.float64) + (8, 0)
    view = SimpleNamespace(index=0, image_bytes=cv2.imencode(".png", image)[1].tobytes(), landmarks=points,
                           is_synthetic=False, landmarks_trusted=True, yaw=0, pitch=0, quality=SimpleNamespace(score=1))
    document = SimpleNamespace(faceset_id="f", views=(view,))
    engine = NeuralFaceEngine(Backbone(), ReferenceSurfaceRasterizer(), Enhancer())
    engine.prepare(document)
    return engine, document, Swapper(), image, points


def silhouette_centroids(filtered):
    engine, document, swapper, image, points = noisy_engine(noise=.02)
    centroids = []
    for frame in range(60):
        kwargs = dict(frame_index=frame, timestamp=frame / 30) if filtered else {}
        result = engine.process(image, SimpleNamespace(landmarks5=points, pose=(0, 0, 0)), document, swapper,
                                track_id=0, **kwargs)
        ys, xs = np.nonzero(result.coverage > .5)
        centroids.append((xs.mean(), ys.mean()))
    return np.array(centroids)[10:]


def test_dense_fit_filtering_cuts_silhouette_jitter():
    raw, filtered = silhouette_centroids(False), silhouette_centroids(True)
    assert second_difference_rms(filtered) < 0.6 * second_difference_rms(raw)
