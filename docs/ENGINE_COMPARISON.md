# Engine and render options

Roop 3D has exactly one rendering architecture: a **complete 3D head** seen through a **perspective camera**. The former 2D swappers
(inswapper-class models), 5-point crop-and-paste, 2.5D frontal-strip meshes, weak-perspective cameras and screen-space face restorers were
removed in the Phase 0 purge. This page replaces the old engine comparison.

## Options (`/api/v1/true3d`)

| Option | Values | Notes |
|---|---|---|
| Head topology engine | `FLAME-FullHead-3D`, `MakeHuman-CC0-Craniofacial-3D` | Selectable only when its hash-verified artifact is installed (registry ids `flame2023_open`, `makehuman_cc0`). Default: MakeHuman CC0. |
| Shader mode | `3D-PBR-Subsurface-Scattering + Neural-Deferred-UV`, `3D-Gaussian-Head-Avatar` | PBR+SSS or Gaussian head shading. Neural detail may only act on UV texture maps. |
| Camera | `auto` (dynamic focal-length solver) or `manual` (focal length in pixels or horizontal FOV, optional principal point) | Perspective intrinsics only. There is no orthographic or weak-perspective value. |

## Guarantees enforced at runtime (`core/true3d_guard.py`)

* the head mesh is watertight and complete: cranium, forehead dome, temples, cheekbones, mandible, submental neck, eyeballs and oral cavity;
* the camera is a full 3x3 pin-hole K with a rigid SE(3) pose [R|t]; affine/orthographic projections and scaled or reflected rotations are rejected;
* the render configuration is perspective, Z-buffered and PBR-SSS or Gaussian shaded; swapper, restorer, crop-paste, mesh-blend and frontalization keys are rejected.

## Status

The live render path is **unwired** pending integration of the full-head renderer (`roop_3d.pipeline.true3d_renderer`). Until then render entry points
answer `RENDERER_NOT_WIRED` ("3D renderer not wired"); nothing falls back to a 2D or 2.5D path.

## Hardware and providers

Provider, precision and hardware profile selection are unchanged: see [Performance guide](PERFORMANCE_GUIDE.md) and the
[hardware compatibility matrix](HARDWARE_COMPATIBILITY_MATRIX.md).
