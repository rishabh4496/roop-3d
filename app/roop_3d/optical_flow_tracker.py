"""Device flow propagation with explicit direction, visibility and scene resets."""
from __future__ import annotations


def _grid(like, height, width):
    import torch
    y, x = torch.meshgrid(torch.arange(height, device=like.device, dtype=like.dtype),
        torch.arange(width, device=like.device, dtype=like.dtype), indexing="ij")
    return torch.stack((x, y), -1)[None]


def _sample(image, xy):
    import torch.nn.functional as F
    height, width = image.shape[-2:]
    scale = image.new_tensor((max(width - 1, 1), max(height - 1, 1)))
    return F.grid_sample(image, xy * (2 / scale) - 1, align_corners=True, padding_mode="zeros")


class OpticalFlowTracker:
    """One instance per track/shot. Backward flow samples the previous matte.

    Forward flow updates vertex XY in screen pixels; Z is preserved, not inferred.
    Current visibility always wins over temporal history at new occlusions.
    """
    def __init__(self, history=.75):
        if not 0 <= history < 1:
            raise ValueError("History must be in [0, 1)")
        self.history, self.alpha = history, None

    def reset(self):
        self.alpha = None

    def propagate(self, alpha, backward_flow=None, *, vertices=None, forward_flow=None,
                  visibility=None, confidence=None, scene_cut=False):
        import torch
        if alpha.ndim != 4 or alpha.shape[1] != 1 or not alpha.is_floating_point():
            raise ValueError("Alpha must be N1HW floating point")
        if visibility is None:
            visibility = torch.ones_like(alpha)
        if visibility.shape != alpha.shape or visibility.device != alpha.device:
            raise ValueError("Visibility must match alpha")
        shape = (alpha.shape[0], *alpha.shape[-2:], 2)
        for flow in (backward_flow, forward_flow):
            if flow is not None and (tuple(flow.shape) != shape or flow.device != alpha.device or flow.dtype != alpha.dtype):
                raise ValueError("Flow must be NHW2 in full-resolution pixel units on alpha's device")
        if confidence is not None and (confidence.shape != alpha.shape or confidence.device != alpha.device):
            raise ValueError("Confidence must match alpha")
        if scene_cut or self.alpha is None or self.alpha.shape != alpha.shape or self.alpha.device != alpha.device:
            result = alpha.clamp(0, 1)
        else:
            if backward_flow is None:
                raise ValueError("Temporal propagation requires current-to-previous flow")
            xy = _grid(alpha, *alpha.shape[-2:]) + backward_flow
            valid = ((xy[..., 0] >= 0) & (xy[..., 0] <= alpha.shape[-1] - 1)
                     & (xy[..., 1] >= 0) & (xy[..., 1] <= alpha.shape[-2] - 1))[:, None]
            previous = _sample(self.alpha, xy)
            weight = valid * self.history
            if confidence is not None:
                weight = weight * confidence.clamp(0, 1)
            result = previous * weight + alpha.clamp(0, 1) * (1 - weight)
        result = result.clamp(0, 1) * visibility.clamp(0, 1)
        self.alpha = result.detach()
        updated = None
        if vertices is not None:
            if (forward_flow is None or vertices.ndim != 3 or vertices.shape[-1] != 3
                    or vertices.shape[0] != alpha.shape[0] or vertices.device != alpha.device):
                raise ValueError("Mesh propagation requires N,V,3 vertices and previous-to-current flow")
            displacement = _sample(forward_flow.permute(0, 3, 1, 2), vertices[:, :, None, :2])
            updated = vertices.clone()
            updated[..., :2] += displacement[:, :, :, 0].transpose(1, 2)
        return result, updated


class RAFTSmallFlow:
    """Injected local RAFT-Small Torch/TRT runner; no downloads or boot-time load.

    Runner accepts two N3HW RGB float tensors and returns N2HW forward pixel flow.
    Model preprocessing, padding and weight provenance belong to that runner.
    """
    backend = "RAFT_SMALL"

    def __init__(self, runner):
        self.runner = runner

    def pair(self, previous, current):
        def run(a, b):
            value = self.runner(a, b)
            if (not value.is_cuda or value.device != a.device
                    or tuple(value.shape) != (a.shape[0], 2, *a.shape[-2:])):
                raise ValueError("RAFT runner must return full-resolution CUDA N2HW flow")
            return value.permute(0, 2, 3, 1).to(dtype=a.dtype)
        if previous.shape != current.shape or not previous.is_cuda or previous.device != current.device:
            raise ValueError("RAFT inputs must share CUDA shape and device")
        return run(previous, current), run(current, previous)


class NvidiaHardwareFlow:
    """OpenCV OF 2.0 pointer interop, with no GpuMat upload/download.

    Requires a custom OpenCV CUDA/optflow build. Explicit synchronization bridges
    the OpenCV default stream; this is not advertised as asynchronous throughput.
    """
    backend = "NVIDIA_OF_2_0"

    def __init__(self, size, *, device=0):
        import cv2
        from roop_3d.nv_video_codec import GPUBackendUnavailable
        if (not hasattr(cv2, "cuda_NvidiaOpticalFlow_2_0")
                or not hasattr(cv2.cuda, "createGpuMatFromCudaMemory")):
            raise GPUBackendUnavailable("OpenCV CUDA OF 2.0 and external-device-memory bindings required")
        self.size, self.device = tuple(size), device
        self.engine = cv2.cuda_NvidiaOpticalFlow_2_0.create(
            self.size, outputGridSize=1, enableTemporalHints=False, gpuId=device)

    def pair(self, previous, current):
        import cv2
        import torch
        for value in (previous, current):
            if (not value.is_cuda or value.dtype != torch.uint8 or tuple(value.shape) != self.size[::-1]
                    or value.device.index != self.device or not value.is_contiguous()):
                raise ValueError("Hardware flow requires contiguous CUDA grayscale HW uint8")
        torch.cuda.current_stream(self.device).synchronize()
        mats = [cv2.cuda.createGpuMatFromCudaMemory(self.size, cv2.CV_8UC1,
                v.data_ptr(), v.stride(0)) for v in (previous, current)]
        def run(a, b):
            raw, _ = self.engine.calc(a, b, None)
            output = cv2.cuda_GpuMat()
            self.engine.convertToFloat(raw, output)
            cv2.cuda.Stream_Null().waitForCompletion()
            # CAI does not own the GpuMat: clone on device before it is released.
            class Buffer:
                __cuda_array_interface__ = {"shape": (*self.size[::-1], 2), "typestr": "<f4",
                    "data": (output.cudaPtr(), False), "strides": (output.step, 8, 4), "version": 3}
            result = torch.as_tensor(Buffer(), device=f"cuda:{self.device}").clone()[None]
            torch.cuda.current_stream(self.device).synchronize()
            return result
        return run(*mats), run(*reversed(mats))

    def close(self):
        self.engine.collectGarbage()


def select_flow(size, *, raft_runner=None, device=0):
    import cv2
    from roop_3d.nv_video_codec import GPUBackendUnavailable
    try:
        return NvidiaHardwareFlow(size, device=device)
    except (GPUBackendUnavailable, RuntimeError, cv2.error) as error:
        if raft_runner is None:
            raise GPUBackendUnavailable(f"Hardware optical flow unavailable; provide a local RAFT runner: {error}") from error
        return RAFTSmallFlow(raft_runner)
