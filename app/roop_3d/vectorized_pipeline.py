"""Batched identity transfer and GPU-resident affine/composite ONNX operators.

Host frames cross one upload and one final download. ORT device input/output
allocations are retained between equal-shape calls; no Torch/CuPy dependency.
"""
from __future__ import annotations

import threading
from typing import Mapping

import cv2
import numpy as np

from roop_3d.engine.volumetric.preprocess import aligned_crop, similarity
from roop_3d.faceset.recognizer import ARCFACE_REFERENCE_5PTS
from roop_3d.render_policy import CudaWorkerContext


def crop_matrix(points: np.ndarray, size=128):
    reference = ARCFACE_REFERENCE_5PTS.astype(np.float64).copy()
    reference[:, 0] += 8
    return similarity(points, reference * size / 128)


def feather_mask(size=128):
    y, x = np.mgrid[:size, :size].astype(np.float32)
    return np.clip((1 - ((x - (size - 1) / 2) / (size * 0.46))**2
                    - ((y - (size - 1) / 2) / (size * 0.48))**2) * 8, 0, 1).astype(np.float32)


def composite_cpu(frame, crops, matrices):
    result = frame.astype(np.float32)
    height, width = frame.shape[:2]
    mask = feather_mask(crops.shape[-1])
    for crop, matrix in zip(crops, matrices):
        inverse = cv2.invertAffineTransform(matrix)
        rgb = crop.transpose(1, 2, 0)[:, :, ::-1] * 255
        warped = cv2.warpAffine(rgb, inverse, (width, height), flags=cv2.INTER_LINEAR)
        alpha = cv2.warpAffine(mask, inverse, (width, height), flags=cv2.INTER_LINEAR)[:, :, None]
        result = warped * alpha + result * (1 - alpha)
    return np.clip(result, 0, 255).round().astype(np.uint8)


class CpuBatchPipeline:
    """Honest CPU/synthetic path using the same crop and composition geometry."""
    backend = "CPU"
    def __init__(self, inference, *, embeddings: Mapping[int, np.ndarray] | None = None, restorer=None):
        self.inference, self.embeddings, self.restorer = inference, embeddings or {}, restorer
        self.context = CudaWorkerContext(enabled=False)

    def fork_worker(self):
        return CpuBatchPipeline(self.inference, embeddings=self.embeddings, restorer=self.restorer)

    def process_batch(self, frames, faces_by_frame):
        if any(len(v) != 1 for v in faces_by_frame):
            return [self.process(frame, faces) for frame, faces in zip(frames, faces_by_frame)]
        aligned = [aligned_crop(frame, faces[0].landmarks5, 128) for frame, faces in zip(frames, faces_by_frame)]
        values = np.stack([v.image[:, :, ::-1].transpose(2, 0, 1) for v in aligned]).astype(np.float32) / 255
        sources = np.stack([self.embeddings.get(v[0].track_id, np.zeros(512, np.float32)) for v in faces_by_frame])
        with self.context.activate():
            output = np.asarray(self.inference(values, sources), np.float32)
        if output.shape != values.shape or not np.isfinite(output).all():
            raise ValueError("Multi-frame inference contract failed")
        if self.restorer:
            futures = []
            for index, faces in enumerate(faces_by_frame):
                face = faces[0]
                crop = np.clip(output[index].transpose(1, 2, 0)[:, :, ::-1] * 255, 0, 255).astype(np.uint8)
                points = cv2.transform(face.landmarks5[None], aligned[index].matrix)[0]
                futures.append(self.restorer.submit(crop, points, face.box, face.track_id, native_shape=frames[index].shape[:2]))
            output = np.stack([v.result()[:, :, ::-1].transpose(2, 0, 1) for v in futures]).astype(np.float32) / 255
        return [composite_cpu(frame, output[index:index + 1], [aligned[index].matrix]) for index, frame in enumerate(frames)]

    def process(self, frame, faces):
        if not faces:
            return frame
        with self.context.activate():
            aligned = [aligned_crop(frame, face.landmarks5, 128) for face in faces]
            batch = np.stack([v.image[:, :, ::-1].transpose(2, 0, 1) for v in aligned]).astype(np.float32) / 255
            source = np.stack([self.embeddings.get(v.track_id, np.zeros(512, np.float32)) for v in faces])
            output = np.asarray(self.inference(batch, source), np.float32)
            if output.shape != batch.shape or not np.isfinite(output).all():
                raise ValueError("Batched inference output contract failed")
            if self.restorer:
                futures = []
                for index, face in enumerate(faces):
                    image = np.clip(output[index].transpose(1, 2, 0)[:, :, ::-1] * 255, 0, 255).astype(np.uint8)
                    local_points = cv2.transform(face.landmarks5[None], aligned[index].matrix)[0]
                    futures.append(self.restorer.submit(image, local_points, face.box, face.track_id, native_shape=frame.shape[:2]))
                for index, future in enumerate(futures):
                    image = future.result()
                    output[index] = image[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255
            return composite_cpu(frame, output, [v.matrix for v in aligned])


def affine_graph(height, width, batch, *, compose=False, multi_frame=False):
    """Build ordinary ONNX GridSample/MatMul operators, executable on CPU or CUDA."""
    import onnx
    h, t = onnx.helper, onnx.TensorProto
    oh, ow = (height, width) if compose else (128, 128)
    ih, iw = (128, 128) if compose else (height, width)
    y, x = np.mgrid[:oh, :ow].astype(np.float32)
    grid = np.stack([x, y, np.ones_like(x)], -1).reshape(1, -1, 3)
    initializers = []
    def constant(name, value):
        initializers.append(onnx.numpy_helper.from_array(np.asarray(value), name))
        return name
    constant("grid", grid)
    constant("grid_shape", np.array([batch, oh, ow, 2], np.int64))
    nodes = [h.make_node("Transpose", ["matrix"], ["mt"], perm=[0, 2, 1]),
             h.make_node("MatMul", ["grid", "mt"], ["xy"]),
             h.make_node("Reshape", ["xy", "grid_shape"], ["sampling_grid"])]
    if compose:
        constant("mask", np.repeat(feather_mask()[None, None], batch, 0))
        nodes.append(h.make_node("Concat", ["crops", "mask"], ["rgba"], axis=1))
        nodes.append(h.make_node("GridSample", ["rgba", "sampling_grid"], ["warped"], align_corners=1,
                                 mode="bilinear", padding_mode="zeros"))
        constant("rgb_indices", np.array([0, 1, 2], np.int64))
        constant("alpha_index", np.array([3], np.int64))
        constant("one", np.array(1, np.float32))
        previous = "image"
        if multi_frame:
            nodes += [h.make_node("Gather", ["warped", "rgb_indices"], ["rgb"], axis=1),
                      h.make_node("Gather", ["warped", "alpha_index"], ["alpha"], axis=1),
                      h.make_node("Sub", ["one", "alpha"], ["inverse_alpha"]),
                      h.make_node("Mul", ["image", "inverse_alpha"], ["base"]),
                      h.make_node("Mul", ["rgb", "alpha"], ["foreground"]),
                      h.make_node("Add", ["base", "foreground"], ["multi_composite"])]
            previous = "multi_composite"
        for index in range(0 if multi_frame else batch):
            constant(f"index{index}", np.array([index], np.int64))
            nodes += [h.make_node("Gather", ["warped", f"index{index}"], [f"face{index}"], axis=0),
                      h.make_node("Gather", [f"face{index}", "rgb_indices"], [f"rgb{index}"], axis=1),
                      h.make_node("Gather", [f"face{index}", "alpha_index"], [f"alpha{index}"], axis=1),
                      h.make_node("Sub", ["one", f"alpha{index}"], [f"inv{index}"]),
                      h.make_node("Mul", [previous, f"inv{index}"], [f"base{index}"]),
                      h.make_node("Mul", [f"rgb{index}", f"alpha{index}"], [f"fg{index}"]),
                      h.make_node("Add", [f"base{index}", f"fg{index}"], [f"composite{index}"])]
            previous = f"composite{index}"
        nodes.append(h.make_node("Identity", [previous], ["output"]))
    else:
        if not multi_frame:
            constant("expand_shape", np.array([batch, 3, height, width], np.int64))
            nodes.append(h.make_node("Expand", ["image", "expand_shape"], ["expanded"]))
        nodes.append(h.make_node("GridSample", ["image" if multi_frame else "expanded", "sampling_grid"], ["output"],
                                align_corners=1, mode="bilinear", padding_mode="zeros"))
    inputs = [h.make_tensor_value_info("image", t.FLOAT, [batch if multi_frame else 1, 3, height, width]),
              h.make_tensor_value_info("matrix", t.FLOAT, [batch, 2, 3])]
    if compose:
        inputs.append(h.make_tensor_value_info("crops", t.FLOAT, [batch, 3, 128, 128]))
    outputs = [h.make_tensor_value_info("output", t.FLOAT, [batch if multi_frame else 1, 3, height, width] if compose else [batch, 3, 128, 128])]
    model = h.make_model(h.make_graph(nodes, "roop3d_gpu_affine", inputs, outputs, initializers),
                         opset_imports=[h.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


def normalized_matrix(matrix, height, width):
    result = np.asarray(matrix, np.float32).copy()
    result[0] *= 2 / max(1, width - 1)
    result[1] *= 2 / max(1, height - 1)
    result[:, 2] -= 1
    return result


class OrtValueRunner:
    """Persistent device allocations and I/O binding; caller serializes ownership."""
    def __init__(self, session, outputs: dict[str, tuple[int, ...]], *, device="cuda"):
        import onnxruntime as ort
        self.session, self.device = session, device
        self.outputs = {n: ort.OrtValue.ortvalue_from_shape_and_type(shape, np.float32, device, 0) for n, shape in outputs.items()}
        self.inputs = {}
        self.binding = session.io_binding()

    def run(self, feeds):
        import onnxruntime as ort
        self.binding.clear_binding_inputs()
        self.binding.clear_binding_outputs()
        for name, value in feeds.items():
            if isinstance(value, np.ndarray):
                old = self.inputs.get(name)
                if old is None or tuple(old.shape()) != value.shape or old.data_type() != "tensor(float)":
                    old = ort.OrtValue.ortvalue_from_shape_and_type(value.shape, value.dtype, self.device, 0)
                    self.inputs[name] = old
                old.update_inplace(np.ascontiguousarray(value))
                value = old
            self.binding.bind_ortvalue_input(name, value)
        for name, value in self.outputs.items():
            self.binding.bind_ortvalue_output(name, value)
        self.binding.synchronize_inputs()
        self.session.run_with_iobinding(self.binding)
        self.binding.synchronize_outputs()
        return self.outputs


class GpuBatchPipeline:
    backend = "CUDA_IOBINDING"
    def __init__(self, session, embeddings: Mapping[int, np.ndarray], *, max_batch=4, hardware_profile="6gb", restorer=None,
                 execution_threads=1, compositor_options=None, device_masker_factory=None):
        if not session.get_providers() or session.get_providers()[0] not in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
            raise ValueError("GPU pipeline requires an activated GPU provider")
        if not 1 <= max_batch <= 8 or hardware_profile not in ("6gb", "12gb") or not 1 <= execution_threads <= 8:
            raise ValueError("Invalid GPU batch/profile")
        self.session, self.embeddings, self.max_batch = session, embeddings, max_batch
        self.hardware_profile = hardware_profile
        self.restorer = restorer
        self.execution_threads = execution_threads
        self.compositor_options, self.device_masker_factory = compositor_options, device_masker_factory
        self.context = CudaWorkerContext()
        self._graphs = {}
        self._swap = {}
        self._lock = threading.RLock()
        self._frame = None
        self._resident = None
        import os
        if os.environ.get("ROOP_GPU_COMPOSITOR") == "1" or compositor_options is not None:
            from roop_3d.gpu_compositor import resident_swap_pipeline, CompositorOptions
            self._resident = resident_swap_pipeline(session, embeddings, hardware_profile=hardware_profile,
                execution_threads=execution_threads, restorer=restorer, masker_factory=device_masker_factory,
                options=compositor_options if compositor_options is not None else CompositorOptions())
            self.backend = self._resident.backend

    def fork_worker(self):
        return GpuBatchPipeline(self.session, self.embeddings, max_batch=self.max_batch,
                                hardware_profile=self.hardware_profile, restorer=self.restorer,
                                execution_threads=self.execution_threads, compositor_options=self.compositor_options,
                                device_masker_factory=self.device_masker_factory)

    def _graph(self, height, width, count, compose, multi_frame=False):
        key = height, width, count, compose, multi_frame
        if key not in self._graphs:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            session = ort.InferenceSession(affine_graph(height, width, count, compose=compose, multi_frame=multi_frame).SerializeToString(),
                sess_options=options, providers=[("CUDAExecutionProvider", {"use_tf32": False,
                    "gpu_mem_limit": (512 if self.hardware_profile == "6gb" else 1024) * 1048576})])
            session.disable_fallback()
            if session.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError("GPU affine operator session fell back to CPU")
            output_shape = (count if multi_frame else 1, 3, height, width) if compose else (count, 3, 128, 128)
            self._graphs[key] = OrtValueRunner(session, {"output": output_shape})
        return self._graphs[key]

    def process_batch(self, frames, faces_by_frame):
        if self._resident is not None:
            return self._resident.process_batch(frames, faces_by_frame)
        if any(len(v) != 1 for v in faces_by_frame) or len({v.shape for v in frames}) != 1:
            return [self.process(frame, faces) for frame, faces in zip(frames, faces_by_frame)]
        with self._lock, self.context.activate():
            height, width = frames[0].shape[:2]
            budget = (256 if self.hardware_profile == "6gb" else 768) * 1048576 // self.execution_threads
            limit = min(self.max_batch, budget // (height * width * 32))
            if limit < 1:
                raise ValueError("Multi-frame batch exceeds this GPU profile's worker budget")
            if len(frames) > limit:
                return [output for first in range(0, len(frames), limit) for output in self.process_batch(
                    frames[first:first + limit], faces_by_frame[first:first + limit])]
            count = len(frames)
            image = np.stack([frame[:, :, ::-1].transpose(2, 0, 1) for frame in frames]).astype(np.float32) / 255
            crop_runner = self._graph(height, width, count, False, True)
            matrices = np.stack([crop_matrix(faces[0].landmarks5) for faces in faces_by_frame])
            inverse = np.stack([normalized_matrix(cv2.invertAffineTransform(m), height, width) for m in matrices])
            crops = crop_runner.run({"image": image, "matrix": inverse})["output"]
            if count not in self._swap:
                self._swap[count] = OrtValueRunner(self.session, {"output": (count, 3, 128, 128)})
            sources = np.stack([self.embeddings[v[0].track_id] for v in faces_by_frame]).astype(np.float32)
            swapped = self._swap[count].run({"target": crops, "source": sources})["output"]
            if self.restorer:
                values, futures = swapped.numpy(), []
                for index, faces in enumerate(faces_by_frame):
                    face = faces[0]
                    crop = np.clip(values[index].transpose(1, 2, 0)[:, :, ::-1] * 255, 0, 255).astype(np.uint8)
                    points = cv2.transform(face.landmarks5[None], matrices[index])[0]
                    futures.append(self.restorer.submit(crop, points, face.box, face.track_id, native_shape=frames[index].shape[:2]))
                swapped = np.ascontiguousarray(np.stack([v.result()[:, :, ::-1].transpose(2, 0, 1) for v in futures]).astype(np.float32) / 255)
            forward = np.stack([normalized_matrix(m, 128, 128) for m in matrices])
            # Rebind the crop runner's existing device image buffer rather than upload it twice.
            result = self._graph(height, width, count, True, True).run(
                {"image": crop_runner.inputs["image"], "crops": swapped, "matrix": forward})["output"].numpy()
            if not np.isfinite(result).all():
                raise RuntimeError("Multi-frame GPU compositor produced non-finite pixels")
            return [np.clip(v.transpose(1, 2, 0)[:, :, ::-1] * 255, 0, 255).round().astype(np.uint8) for v in result]

    def process(self, frame, faces):
        if self._resident is not None:
            return self._resident.process(frame, faces)
        if not faces:
            return frame
        with self._lock, self.context.activate():
            height, width = frame.shape[:2]
            # Expand/warp intermediates are bounded independently for each GPU profile.
            budget = (256 if self.hardware_profile == "6gb" else 768) * 1048576 // self.execution_threads
            limit = min(self.max_batch, budget // (height * width * 32))
            if limit < 1:
                raise ValueError("Frame size and worker count exceed this GPU profile's affine budget; reduce execution_threads")
            image = frame[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
            import onnxruntime as ort
            if self._frame is None or tuple(self._frame.shape()) != image.shape:
                self._frame = ort.OrtValue.ortvalue_from_shape_and_type(image.shape, np.float32, "cuda", 0)
            self._frame.update_inplace(np.ascontiguousarray(image))
            device_frame = self._frame
            for start in range(0, len(faces), limit):
                group = faces[start:start + limit]
                count = len(group)
                matrices = np.stack([crop_matrix(v.landmarks5) for v in group])
                inverse = np.stack([normalized_matrix(cv2.invertAffineTransform(m), height, width) for m in matrices])
                crops = self._graph(height, width, count, False).run({"image": device_frame, "matrix": inverse})["output"]
                if count not in self._swap:
                    self._swap[count] = OrtValueRunner(self.session, {"output": (count, 3, 128, 128)})
                source = np.stack([self.embeddings[v.track_id] for v in group]).astype(np.float32)
                swapped = self._swap[count].run({"target": crops, "source": source})["output"]
                if self.restorer:
                    # Optional detached restorers use an explicit crop-sized host bridge;
                    # affine/composite tensors and the full frame remain device-resident.
                    values = swapped.numpy()
                    futures = []
                    for index, face in enumerate(group):
                        image = np.clip(values[index].transpose(1, 2, 0)[:, :, ::-1] * 255, 0, 255).astype(np.uint8)
                        points = cv2.transform(face.landmarks5[None], matrices[index])[0]
                        futures.append(self.restorer.submit(image, points, face.box, face.track_id, native_shape=frame.shape[:2]))
                    restored = np.stack([future.result()[:, :, ::-1].transpose(2, 0, 1) for future in futures]).astype(np.float32) / 255
                    runner = self._graph(height, width, count, True)
                    swapped = np.ascontiguousarray(restored)
                forward = np.stack([normalized_matrix(m, 128, 128) for m in matrices])
                device_frame = self._graph(height, width, count, True).run(
                    {"image": device_frame, "crops": swapped, "matrix": forward})["output"]
            output = device_frame.numpy()[0].transpose(1, 2, 0)[:, :, ::-1]
            if not np.isfinite(output).all():
                raise RuntimeError("GPU compositor produced non-finite pixels")
            return np.clip(output * 255, 0, 255).round().astype(np.uint8)
