import cv2
import numpy as np

from roop_3d.compositor.boundary_fusion import BlendMode, fuse_crop
from roop_3d.pipeline.subpixel import warp_affine


def test_crop_fusion_uses_subpixel_lanczos_image_warp_and_linear_matte():
    y, x = np.indices((48, 48))
    pattern = (((x + y) % 2) * 255).astype(np.uint8)
    crop = np.repeat(pattern[..., None], 3, axis=2)
    frame = np.zeros((80, 80, 3), np.uint8)
    transform = np.array([[1.0, 0.0, 14.35], [0.0, 1.0, 12.65]], np.float64)

    # Isolate resampling from the default MULTI_BAND frequency mixing.
    output = fuse_crop(frame, crop, transform, np.ones((48, 48), np.float32), mode=BlendMode.FEATHER)
    expected = warp_affine(crop, transform, (80, 80), interpolation="lanczos4",
                           border="constant", border_value=(0, 0, 0))
    linear = cv2.warpAffine(crop, transform.astype(np.float32), (80, 80), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    np.testing.assert_array_equal(output[20:55, 20:55], expected[20:55, 20:55])
    assert not np.array_equal(output[20:55, 20:55], linear[20:55, 20:55])
    np.testing.assert_array_equal(output[0:8], frame[0:8])
