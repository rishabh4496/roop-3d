"""Independent restorer sessions, explicit training alignment and native-size bypass."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import threading

import cv2
import numpy as np

from roop_3d.engine.volumetric.preprocess import similarity
from roop_3d.render_policy import CudaWorkerContext

# Reference preprocessing coordinates, not a claim of measured output parity.
# github.com/yangxy/GPEN/blob/main/{align_faces.py,face_enhancement.py}
GPEN_REFERENCE = (np.array([[30.29459953, 51.69630051], [65.53179932, 51.50139999],
    [48.02519989, 71.73660278], [33.54930115, 92.3655014], [62.72990036, 92.20410156]]) + [36, 28]) * (512 / 168)
# github.com/sczhou/CodeFormer/blob/master/facelib/utils/face_restoration_helper.py
CODEFORMER_REFERENCE = np.array([[192.98138, 239.94708], [318.90277, 240.1936],
    [256.63416, 314.01935], [201.26117, 371.41043], [313.08905, 371.15118]])


@dataclass(frozen=True)
class RestorerSpec:
    name: str
    model_name: str
    template: np.ndarray
    size: int = 512
    normalization: str = "minus_one_one"

    def __post_init__(self):
        if (np.asarray(self.template).shape != (5, 2) or not np.isfinite(self.template).all()
                or self.size not in (128, 256, 512, 1024) or self.normalization not in ("minus_one_one", "zero_one")):
            raise ValueError("Restorer requires an explicit finite training alignment/input contract")


def gpen_spec(model_name="gpen_bfr_512.onnx"):
    return RestorerSpec("gpen_realistic", model_name, GPEN_REFERENCE.copy())


def codeformer_spec(model_name="CodeFormerv0.1.onnx"):
    return RestorerSpec("codeformer", model_name, CODEFORMER_REFERENCE.copy())


def ultramax_spec():
    """Observed preserved processor: CodeFormer fp16, FFHQ-512, RGB [-1,1]."""
    return RestorerSpec("ultramax", "codeformer.fp16.onnx", CODEFORMER_REFERENCE.copy())


def load_ultramax_spec(root):
    import json
    from roop_3d.core.storage import owned
    path = owned(root, "roop_3d/.runtime/restorers/ultramax.json")
    if not path.exists():
        return ultramax_spec()
    if path.stat().st_size > 16384:
        raise ValueError("UltraMax preprocessing contract exceeds the metadata budget")
    info = json.loads(path.read_text())
    if set(info) != {"model_name", "template", "size", "normalization", "evidence"} or not info["evidence"]:
        raise ValueError("UltraMax alignment/normalization evidence is required")
    return RestorerSpec("ultramax", info["model_name"], np.asarray(info["template"], np.float64),
                        info["size"], info["normalization"])


class MaskSeamWatchdog:
    """Never compare occlusion coverage to a geometric-face-mask baseline."""
    def __init__(self, maximum=256):
        self.maximum = maximum
        self._baselines = {}
        self._lock = threading.Lock()

    def observe(self, face_id: int, mask: np.ndarray, *, kind: str):
        if kind not in ("geometric", "occluder", "restoration"):
            raise ValueError("Mask baseline type must be explicit")
        mask = np.asarray(mask, np.float32)
        if mask.ndim != 2 or not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
            raise ValueError("Invalid seam watchdog mask")
        key = (threading.get_ident(), face_id, kind)
        with self._lock:
            old = self._baselines.get(key)
            drift = float(np.abs(mask - old).mean()) if old is not None and old.shape == mask.shape else 0.0
            if key not in self._baselines and len(self._baselines) >= self.maximum:
                self._baselines.pop(next(iter(self._baselines)))
            self._baselines[key] = mask.copy()
        return drift

    def clear(self):
        with self._lock:
            self._baselines.clear()


class OnnxFaceRestorer:
    def __init__(self, catalog, config, spec: RestorerSpec, *, fidelity_weight=0.5):
        if not np.isfinite(fidelity_weight) or not 0 <= fidelity_weight <= 1:
            raise ValueError("CodeFormer fidelity must be in [0,1]")
        self.catalog, self.config, self.spec, self.fidelity_weight = catalog, config, spec, fidelity_weight
        self.model = None
        self._binding = self._output = self._lookup = None

    def restore(self, aligned):
        from roop_3d.engine.runtime import OrtModel
        if self.model is None:
            from dataclasses import replace
            config = (replace(self.config, graph_optimization="extended")
                      if self.spec.name == "ultramax" and self.config.provider == "CPUExecutionProvider" else self.config)
            self.model = OrtModel(self.catalog, self.spec.model_name, config)
        inputs = self.model.record["graph"]["inputs"]
        image_input = next((v for v in inputs if len(v["shape"]) == 4), None)
        if image_input is None or image_input["shape"][1:] != [3, self.spec.size, self.spec.size]:
            raise ValueError("Restorer graph does not match its registered training input contract")
        dtype = {"FLOAT": np.float32, "FLOAT16": np.float16}.get(image_input["dtype"])
        if dtype is None:
            raise ValueError("Unsupported restorer image dtype")
        if self.spec.name == "ultramax":
            if self._lookup is None:
                values = np.arange(256, dtype=np.float32) / 255
                self._lookup = (values * 2 - 1 if self.spec.normalization == "minus_one_one" else values).astype(dtype)
            values = self._lookup[aligned.transpose(2, 0, 1)[::-1]][None]
        else:
            values = aligned[:, :, ::-1].transpose(2, 0, 1)[None].astype(dtype) / 255
            if self.spec.normalization == "minus_one_one":
                values = values * 2 - 1
        feeds = {image_input["name"]: np.ascontiguousarray(values)}
        for item in inputs:
            if item["name"] == image_input["name"]:
                continue
            if item["name"] not in ("w", "fidelity_weight") or item["dtype"] not in ("FLOAT", "DOUBLE"):
                raise ValueError("Unknown restorer auxiliary input")
            if any(not isinstance(v, int) or v < 1 for v in item["shape"]):
                raise ValueError("Unknown fidelity input shape")
            feeds[item["name"]] = np.full(tuple(item["shape"]), self.fidelity_weight,
                                           np.float64 if item["dtype"] == "DOUBLE" else np.float32)
        if self.spec.name == "ultramax":
            if self.model._stat() != self.model._identity:
                raise ValueError("UltraMax model changed after session initialization")
            if self._binding is None:
                self._binding = self.model.session.io_binding()
            self._binding.clear_binding_inputs()
            self._binding.clear_binding_outputs()
            for name, value in feeds.items():
                if not np.isfinite(value).all():
                    raise ValueError("Non-finite UltraMax input")
                self._binding.bind_cpu_input(name, value)
            image_outputs = [v for v in self.model.record["graph"]["outputs"]
                             if len(v["shape"]) == 4 and v["shape"][1] == 3]
            if len(image_outputs) != 1:
                raise ValueError("UltraMax requires an unambiguous RGB image output")
            name = image_outputs[0]["name"]
            if self._output is None:
                self._binding.bind_output(name, "cuda" if self.model.provider == "CUDAExecutionProvider" else "cpu")
            else:
                self._binding.bind_ortvalue_output(name, self._output)
            self.model.session.run_with_iobinding(self._binding)
            self._binding.synchronize_outputs()
            self._output = self._binding.get_outputs()[0]
            outputs = {name: self._output.numpy()}
            self.model.executions += 1
        else:
            outputs = self.model.run(feeds)
        candidates = [v for v in outputs.values() if v.shape == (1, 3, self.spec.size, self.spec.size)]
        if len(candidates) != 1 or not np.isfinite(candidates[0]).all():
            raise ValueError("Invalid restorer output contract")
        output = candidates[0].astype(np.float32)
        if self.spec.normalization == "minus_one_one":
            output = (output + 1) * 0.5
        restored = np.clip(output[0].transpose(1, 2, 0)[:, :, ::-1] * 255, 0, 255).round().astype(np.uint8)
        if self.spec.name in ("ultramax", "gpen_realistic"):
            # Network luminance on the supplied crop's chrominance. No legacy
            # eye/detail filters or legacy performance measurements are inherited.
            delta = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY).astype(np.int16) - cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY).astype(np.int16)
            restored = np.clip(aligned.astype(np.int16) + delta[:, :, None], 0, 255).astype(np.uint8)
        return restored

    def close(self):
        if self.model:
            self.model.release()
            self.model = None
        self._binding = self._output = self._lookup = None


class DecoupledRestorer:
    def __init__(self, factory, spec: RestorerSpec, *, workers=1, pool_size=2, cuda=False,
                 ultramax_native_bypass=True, ultramax_min_pixels=320):
        if not 1 <= workers <= 8 or not 1 <= pool_size <= 32 or ultramax_min_pixels < 1:
            raise ValueError("Invalid restoration pool limits")
        self.factory, self.spec, self.cuda = factory, spec, cuda
        self.bypass, self.threshold = ultramax_native_bypass, ultramax_min_pixels
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="roop3d-restorer")
        self.slots, self.local = threading.BoundedSemaphore(pool_size), threading.local()
        self.instances, self.lock = [], threading.Lock()
        self.watchdog = MaskSeamWatchdog()
        self.closed = False

    def submit(self, image, landmarks5, native_box, face_id, *, native_shape=None):
        box = np.asarray(native_box, np.float64).copy()
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError("Native face box must be finite XYXY")
        if native_shape is not None:
            box = box.clip([0, 0, 0, 0], [native_shape[1], native_shape[0], native_shape[1], native_shape[0]])
        native = min(max(0, box[2] - box[0]), max(0, box[3] - box[1]))
        if self.spec.name.casefold() == "ultramax" and self.bypass and native <= self.threshold:
            future = Future()
            future.set_result(image.copy())
            return future  # No model construction, alignment or queue contention.
        if self.closed:
            raise RuntimeError("Restorer pool is closed")
        self.slots.acquire()
        try:
            future = self.executor.submit(self._restore, image.copy(), np.asarray(landmarks5).copy(), face_id)
            future.add_done_callback(lambda _: self.slots.release())
            return future
        except BaseException:
            self.slots.release()
            raise

    def _restore(self, image, points, face_id):
        if not hasattr(self.local, "model"):
            self.local.context = CudaWorkerContext(enabled=self.cuda)
            with self.local.context.activate():
                self.local.model = self.factory()
            with self.lock:
                self.instances.append(self.local.model)
        with self.local.context.activate():
            matrix = similarity(points, self.spec.template * self.spec.size / 512)
            crop = cv2.warpAffine(image, matrix, (self.spec.size, self.spec.size), borderMode=cv2.BORDER_CONSTANT)
            restored = self.local.model.restore(crop)
            if restored.shape != crop.shape or restored.dtype != np.uint8:
                raise ValueError("Restoration returned an invalid crop")
            inverse = cv2.invertAffineTransform(matrix)
            warped = cv2.warpAffine(restored, inverse, (image.shape[1], image.shape[0]))
            # Blend only where the aligned crop actually sampled valid input pixels.
            support = cv2.warpAffine(np.ones(image.shape[:2], np.float32), matrix, (self.spec.size, self.spec.size))
            mask = cv2.warpAffine(support, inverse, (image.shape[1], image.shape[0]))
            self.watchdog.observe(face_id, mask, kind="restoration")
            return np.clip(warped.astype(np.float32) * mask[:, :, None] + image * (1 - mask[:, :, None]), 0, 255).round().astype(np.uint8)

    def close(self):
        self.closed = True
        self.executor.shutdown(wait=True, cancel_futures=True)
        for model in self.instances:
            if hasattr(model, "close"):
                model.close()
        self.instances.clear()
        self.watchdog.clear()
