"""Local, full-frame face replacement compositing primitives."""

from roop_3d.compositor.boundary_fusion import BlendMode, fuse_crop
from roop_3d.compositor.masks import FaceMask, build_face_mask, landmark_hull_mask
from roop_3d.compositor.mouth import composite_mouth_region, lower_face_soft_mask
from roop_3d.compositor.pipeline import CompositeOptions, FrameCompositor

__all__ = [
    "BlendMode",
    "CompositeOptions",
    "FaceMask",
    "FrameCompositor",
    "build_face_mask",
    "composite_mouth_region",
    "fuse_crop",
    "landmark_hull_mask",
    "lower_face_soft_mask",
]
