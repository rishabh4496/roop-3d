# Temporal stability implementation notes

## Integrated paths

- **Stage 1 tracking:** `TrackTemporalStabilizer.filter_pose` converts the legacy yaw/pitch/roll pose into SO(3), filters it as a quaternion with a velocity-adaptive One-Euro cutoff, then returns to the legacy angle convention. Rotation jumps at or above 900 degrees/second reset that track's rotational state to the measurement. The existing scene-cut detector resets the tracker state. Landmark, box, expression, and coefficient channels continue to use bounded track-local One-Euro/Kalman state.
- **Stage 2 full-head camera:** `FullHeadEngine` now filters its calibrated camera rotation on SO(3), camera translation as a separate Kalman channel, and expression coefficients in their existing track-local channel. This avoids averaging rotation-vector components as though they were Euclidean coordinates.
- **Stage 3 restoration detail:** `TemporalDetailStabilizer` extracts the high-pass difference between the enhanced crop and the unenhanced base, checks forward/backward consistency plus registered photometric agreement, and blends at most 0.25 history weight. It selects the existing OpenCV NVIDIA Optical Flow 2.0 bridge when CUDA and its custom OpenCV bindings are available, otherwise it uses DIS on CPU. In either case, the current implementation downloads flow to host memory and performs residual warping/fusion on CPU because the compositor/restorer boundary is NumPy. Reports name the backend actually used and expose GPU fallback reasons. The current valid-face mask gates history, so currently masked/occluded pixels get no previous-frame detail. Scene cuts, appearance cuts, frame gaps, and shape changes clear the affected history.
- **Mask diagnostics:** the per-face, per-thread `MaskSeamWatchdog` records combined-mask IoU, area change, boundary-gradient level/jump, and anomaly status. Reports are available in `FrameCompositor.last_reports` and compositor capability diagnostics, with warnings logged for anomalies.

## Benchmark

Run from `app/`:

```text
roop_3d/.venv/Scripts/python.exe -B -m roop_3d.tools.temporal_stability --frames 120
```

The deterministic synthetic 120-frame result (seed 7, side 128, cut at frame 60) measured:

| Metric | Raw | Stabilized | Reduction |
|---|---:|---:|---:|
| Flow-registered high-frequency temporal warp error | 6.3836 | 4.6642 | 26.94% |
| Laplacian high-frequency flicker | 61.6087 | 44.3449 | 28.02% |

The scene-cut frame was byte-identical to the independently enhanced input (maximum pixel delta **0**). The requested **>70%** reduction gate was **not met** by a 0.25 previous-frame weight. The benchmark reports this rather than turning the target into an unverified claim. It is synthetic and does not establish performance on an UltraMax model or real footage.

## Deliberate limitations / next gates

1. The Stage 3 compositor and restorer boundary is NumPy/OpenCV. The NVIDIA hardware-flow option accelerates only optical-flow calculation, and uploads grayscale guide crops then downloads flow vectors; residual warping/fusion remains CPU. This is not an end-to-end GPU-resident pipeline, and the CUDA path could not be exercised on the current machine (OpenCV 4.11.0, zero CUDA devices, no NVIDIA OF 2.0 binding). True GPU residency requires a coordinated device-tensor contract through decode/crop, restorer, residual history, mask fusion, and output staging, plus 6 GB and 12 GB profile measurements.
2. The mask watchdog detects and logs anomalies but does not repair them. Correct `[t-1,t+1]` interpolation needs future-frame lookahead and delayed output; the current compositor is causal and returns the current frame immediately. Applying a stale prior mask as a supposed recovery could smear a genuinely entering hand or hair strand. Add a bounded one-frame output delay and a three-frame mask/flow buffer before enabling automatic interpolation.
3. A universal 100% flicker-free guarantee is not a valid acceptance criterion. Real clips, pose estimation uncertainty, restoration randomness, motion blur, occlusion, cuts, and flow failures prevent such a guarantee. The next quality gate needs a provided, licensed 120-frame target clip, frozen model versions, a pinned GPU profile, and an agreed temporal-weight/ghosting tradeoff. The current 0.25 history weight is too weak to meet the requested 70% synthetic gate.
