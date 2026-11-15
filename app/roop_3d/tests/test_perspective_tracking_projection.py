import numpy as np

from roop_3d.reconstruction.multiview_fit import _Problem, batch_rotations
from roop_3d.reconstruction.geometry_types import PerspectiveCamera


def test_tracker_five_point_anchors_follow_the_dense_landmark_definition():
    from roop_3d.reconstruction.multiview_fit import identity_template, template_landmarks, five_from_68
    from roop_3d.tracking.video_tracker import HeadAnchors
    dense = template_landmarks("68pt", identity_template(2))
    np.testing.assert_allclose(HeadAnchors.canonical().points, five_from_68(dense))


def test_landmark_fit_uses_the_same_perspective_camera_as_texture_projection():
    model = np.array([[-.04, -.03, .02], [.04, -.03, .02], [0, 0, .06],
                      [-.03, .04, 0], [.03, .04, 0]], dtype=float)
    problem = _Problem(model, np.tile([0, 0, 1], (5, 1)), np.zeros((1, 5, 2)),
                       np.ones(1), np.array([[25., 8., 2.]]))
    x = np.r_[np.zeros(15), [25., 8., 2., np.log(1600), 210., 170.]]
    expected = PerspectiveCamera.from_scaled_pose(batch_rotations(np.array([[25., 8., 2.]]))[0],
                                                  1600, 210, 170).project(model - problem.centroid)
    predicted, _, _ = problem.predict(x)
    np.testing.assert_allclose(predicted[0], expected, atol=1e-7)


def test_perspective_bundle_jacobian_agrees_with_finite_differences():
    model = np.array([[-.04, -.03, .02], [.04, -.03, .02], [0, 0, .06],
                      [-.03, .04, 0], [.03, .04, 0]], dtype=float)
    p = _Problem(model, np.tile([0, 0, 1], (5, 1)), np.zeros((1, 5, 2)),
                 np.ones(1), np.array([[25., 8., 2.]]))
    x = np.r_[np.zeros(15), [25., 8., 2., np.log(1600), 210., 170.]]
    # Visibility/IRLS weights are held fixed in the Gauss-Newton Jacobian.
    p.sqrt_weight = lambda rotation: np.ones((1, 5))
    numeric = []
    for i in range(len(x)):
        step = np.zeros_like(x); step[i] = 1e-6
        numeric.append((p.residual(x + step) - p.residual(x - step)) / 2e-6)
    np.testing.assert_allclose(p.jacobian(x), np.array(numeric).T, rtol=1e-4, atol=1e-4)
