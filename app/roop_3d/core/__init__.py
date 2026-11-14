"""Runtime settings and local system contracts."""
from roop_3d.core.true3d_guard import (  # noqa: F401  (public export of the true-3D architecture guard)
    True3DArchitectureError,
    True3DArchitectureGuard,
    True3DRendererUnavailable,
    assert_true_3d,
    require_true3d_renderer,
    validate_camera,
    validate_head_mesh,
    validate_render_config,
)
