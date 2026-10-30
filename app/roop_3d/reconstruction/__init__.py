"""Roop 3D Reconstruction subsystem: 3D geometry contracts, PnP pose estimation,
2.5D TPS fallback, license-gated FLAME parametric fitting, and UV synthesis.
"""
from __future__ import annotations

from roop_3d.reconstruction.flame_fitter import (
    FLAMEFitter,
    FLAMEFittingResult,
    FLAMEModel,
    SyntheticFLAMEModel,
    geman_mcclure_weights,
    huber_weights,
    verify_flame_license,
)
from roop_3d.reconstruction.geometry_types import (
    METRIC_SCALE_METERS,
    METRIC_SCALE_MILLIMETERS,
    CameraIntrinsics,
    ConfidenceState,
    Mesh3D,
    Pose3D,
    UVOrigin,
    euler_to_rotation_matrix,
    opencv_to_opengl_camera,
    opengl_to_opencv_camera,
    pixel_to_uv,
    quaternion_to_rotation_matrix,
    rodrigues_to_rotation_matrix,
    rotation_matrix_to_euler,
    rotation_matrix_to_quaternion,
    rotation_matrix_to_rodrigues,
    uv_to_pixel,
)
from roop_3d.reconstruction.mesh_io import (
    export_mesh_glb,
    export_mesh_obj,
    load_mesh_obj,
)
from roop_3d.reconstruction.tps_fallback import (
    ThinPlateSplineWarper,
    build_canonical_head_mesh,
    get_canonical_3d_landmarks,
    solve_pose_pnp,
    warp_delaunay_piecewise_affine,
)
from roop_3d.reconstruction.uv_synthesizer import (
    UVSynthesizer,
    UVSynthesisResult,
    ViewObservation,
)

__all__ = [
    "METRIC_SCALE_METERS",
    "METRIC_SCALE_MILLIMETERS",
    "CameraIntrinsics",
    "ConfidenceState",
    "FLAMEFitter",
    "FLAMEFittingResult",
    "FLAMEModel",
    "Mesh3D",
    "Pose3D",
    "SyntheticFLAMEModel",
    "ThinPlateSplineWarper",
    "UVSynthesisResult",
    "UVSynthesizer",
    "UVOrigin",
    "ViewObservation",
    "build_canonical_head_mesh",
    "euler_to_rotation_matrix",
    "export_mesh_glb",
    "export_mesh_obj",
    "geman_mcclure_weights",
    "get_canonical_3d_landmarks",
    "huber_weights",
    "load_mesh_obj",
    "opencv_to_opengl_camera",
    "opengl_to_opencv_camera",
    "pixel_to_uv",
    "quaternion_to_rotation_matrix",
    "rodrigues_to_rotation_matrix",
    "rotation_matrix_to_euler",
    "rotation_matrix_to_quaternion",
    "rotation_matrix_to_rodrigues",
    "solve_pose_pnp",
    "uv_to_pixel",
    "verify_flame_license",
    "warp_delaunay_piecewise_affine",
]
