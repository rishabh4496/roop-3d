"""Architecture-aware Stage 2 adapter for registered local swapper models.

The model catalog is the architecture manifest. This adapter hides model-specific
image sizes, identity tensor names, layouts and normalization behind one crop-in,
native-resolution image-out interface. Graph inputs are always discovered from the
loaded session; no architecture is allowed to silently assume InSwapper's names.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.engine.runtime import RuntimeConfig
from roop_3d.engine.volumetric.preprocess import aligned_crop
from roop_3d.models.catalog import ModelCatalog


def _shape(node) -> tuple[Any, ...]:
    return tuple(getattr(node, "shape", ()) or ())


def _dtype(node) -> np.dtype:
    return np.dtype({"tensor(float16)": "float16", "tensor(float)": "float32",
                     "tensor(double)": "float64"}.get(getattr(node, "type", "tensor(float)"), "float32"))


def _static_dim(value: Any, fallback: int) -> int:
    return value if type(value) is int and value > 0 else fallback


def load_local_session(catalog, model_id: str, path: Path, config: RuntimeConfig):
    """Validate a local graph and load it within the selected hardware profile."""
    try:
        import onnx
        from onnx import TensorProto
        graph = onnx.load(str(path), load_external_data=False)
        if any(item.data_location == TensorProto.EXTERNAL or item.external_data
               for item in graph.graph.initializer):
            raise ValueError("external tensors are not supported")
        if any(node.domain not in ("", "ai.onnx", "ai.onnx.ml") for node in graph.graph.node):
            raise ValueError("custom operator domains are not supported")
        onnx.checker.check_model(graph)
    except Exception as error:
        raise DomainError("LOCAL_MODEL_GRAPH_UNSUPPORTED",
            f"Local ONNX model graph failed safety/format checks ({type(error).__name__})", 422) from None
    from roop_3d.core.ort_session import CUDA, CPU, TRT, SessionManager
    providers = ((TRT, CUDA) if config.provider == TRT else
                 (CUDA,) if config.provider == CUDA else (CPU,))
    if config.provider in (TRT, CUDA):
        # SessionManager disables CPU EP fallback and keeps unsupported nodes on CUDA.
        manager = SessionManager(hardware_profile=config.hardware_profile,
            fp16=config.provider == TRT and config.precision_policy != 'fp32', gpu_memory_limit=config.gpu_budget_bytes)
    else:
        manager = SessionManager(hardware_profile="6gb", fp16=False)
    try:
        info = manager.create(path, model_name=path.name, providers=providers,
                              intra_op_threads=config.cpu_threads)
    except Exception as error:
        raise DomainError("MODEL_SESSION_FAILED",
            f"Local ONNX model could not initialize on the selected provider ({type(error).__name__})", 503) from None
    catalog.set_vram_loaded(model_id, info.provider in (TRT, CUDA))
    return info.session


class SwapperEngineAdapter:
    """Unified inference interface: aligned BGR crop + identity -> native BGR crop.

    `runner` is an explicit test seam for contract tests. Production sessions use
    ORT I/O Binding and SessionManager, including TensorRT/CUDA provider selection.
    """

    def __init__(self, catalog: ModelCatalog, model_id: str, config: RuntimeConfig = RuntimeConfig(), *,
                 session=None, runner: Callable[[Mapping[str, np.ndarray]], Mapping[str, Any]] | None = None):
        spec = catalog.swapper_spec(model_id)
        if spec is None or spec.get("category") != "swapper":
            raise DomainError("SWAPPER_MODEL_UNSUPPORTED", "Selected model is not a registered swapper", 422)
        self.catalog, self.model_id, self.spec, self.config = catalog, model_id, spec, config
        self.resolution = int(spec["native_resolution"])
        self.path: Path | None = None
        self.model = None
        self.session = session
        self._runner = runner
        self._binding = None
        self._inputs = tuple(session.get_inputs()) if session is not None else ()
        self._outputs = tuple(session.get_outputs()) if session is not None else ()
        self.morph_factor = 0.75
        self.dfm_masks = None
        self._source_cache: dict[tuple[int, int], np.ndarray] = {}
        self.executions = 0
        if session is None:
            try:
                self.path = catalog.resolve_local_model(model_id)
            except DomainError as error:
                if error.code != "MODEL_NOT_INSTALLED":
                    raise
                record = self._registered_record()
                if record is None or not record.get("sha256") or not record.get("size_bytes"):
                    raise DomainError("MODEL_NOT_INSTALLED", "Selected swapper has no installed local ONNX weights", 404) from None
                from roop_3d.engine.runtime import OrtModel
                self.model = OrtModel(catalog, record["name"], config)
                self.session = self.model.session
                self._runner = self.model.run
            else:
                self._load_unblocked_local(self.path)
        self._inputs = tuple(self.session.get_inputs())
        self._outputs = tuple(self.session.get_outputs())
        if self.spec.get("identity_mode") == "dfm":
            self._inspect_dfm()
        if self._runner is None:
            from roop_3d.core.ort_session import GpuIoBinding
            self._binding = GpuIoBinding(self.session)
            self._runner = lambda feeds: self._binding.run(feeds)
        # Read-back of the file now in VRAM, keyed by (stage, model_id, path, provider, precision).
        self._active_key = None
        if session is None:
            from roop_3d.engine import active_models
            path = self.path or getattr(self.model, "path", None)
            provider = ((self.session.get_providers() or [config.provider])[0]
                        if hasattr(self.session, "get_providers") else config.provider)
            record = getattr(self.model, "record", None)
            self._active_key = active_models.activate(
                "swapper", model_id, path, provider, config.precision_policy,
                record.get("sha256") if isinstance(record, dict) else None).key

    def _inspect_dfm(self) -> None:
        names = {node.name for node in self._inputs}
        if names not in ({"in_face:0"}, {"in_face:0", "morph_value:0"}):
            raise DomainError("DFM_GRAPH_UNSUPPORTED", "Expected DeepFaceLive in_face:0 and optional morph_value:0", 422)
        node = next(node for node in self._inputs if node.name == "in_face:0")
        shape = _shape(node)
        if (len(shape) != 4 or shape[-1] != 3 or type(shape[1]) is not int
                or shape[1] != shape[2] or not 32 <= shape[1] <= 2048 or len(self._outputs) != 3):
            raise DomainError("DFM_GRAPH_UNSUPPORTED", "DFM requires square NHWC face input and three ordered outputs", 422)
        if "morph_value:0" in names:
            morph = next(node for node in self._inputs if node.name == "morph_value:0")
            if _shape(morph) != (1,):
                raise DomainError("DFM_GRAPH_UNSUPPORTED", "DFM morph factor must have shape [1]", 422)
        self.resolution = shape[1]

    def _infer_dfm(self, crop: np.ndarray, morph_factor: float | None) -> np.ndarray:
        factor = self.morph_factor if morph_factor is None else morph_factor
        if not np.isfinite(factor) or not 0 <= factor <= 1:
            raise DomainError("DFM_MORPH_INVALID", "DFM morph factor must be in 0..1", 422)
        face_node = next(node for node in self._inputs if node.name == "in_face:0")
        # DeepFaceLive uses BGR NHWC, unlike RGB ArcFace swappers.
        feed = {"in_face:0": np.ascontiguousarray(crop[None] / 255.0, dtype=_dtype(face_node))}
        for node in self._inputs:
            if node.name == "morph_value:0":
                feed[node.name] = np.array([factor], dtype=_dtype(node))
        values = self.run(feed)
        masks, face, celeb_mask = [values[node.name] for node in self._outputs]
        expected = (1, self.resolution, self.resolution)
        if (face.shape != (*expected, 3) or masks.shape != (*expected, 1)
                or celeb_mask.shape != (*expected, 1)
                or not all(np.isfinite(value).all() for value in (face, masks, celeb_mask))):
            raise DomainError("DFM_OUTPUT_INVALID", "DFM outputs do not match its face/mask contract", 503)
        self.dfm_masks = (np.clip(masks[0, :, :, 0], 0, 1), np.clip(celeb_mask[0, :, :, 0], 0, 1))
        self.executions += 1
        alpha = (self.dfm_masks[0] * self.dfm_masks[1])[:, :, None]
        blended = np.clip(face[0] * 255, 0, 255) * alpha + crop * (1 - alpha)
        return blended.round().astype(np.uint8)

    def _registered_record(self) -> dict[str, Any] | None:
        aliases = {str(value).casefold() for value in (self.model_id, *self.spec.get("aliases", ()))}
        for registered in self.catalog.registry.records():
            if registered.model_id == self.model_id or registered.name.casefold() in aliases or any(
                    Path(name).stem.casefold() in aliases for name in registered.filenames):
                return self.catalog._legacy(registered)
        return None

    def _load_unblocked_local(self, path: Path) -> None:
        self.session = load_local_session(self.catalog, self.model_id, path, self.config)

    def _image_input_specs(self):
        result = []
        for node in self._inputs:
            shape = _shape(node)
            folded = node.name.casefold()
            if len(shape) != 4:
                continue
            has_rgb_axis = ((type(shape[1]) is int and shape[1] == 3)
                            or (type(shape[-1]) is int and shape[-1] == 3))
            dynamic_rgb_axis = ((not isinstance(shape[1], int) or not isinstance(shape[-1], int))
                and any(word in folded for word in ("target", "image", "source", "src", "reference", "ref")))
            is_image = len(shape) == 4 and (has_rgb_axis or dynamic_rgb_axis)
            if is_image:
                role = "source" if any(word in folded for word in ("source", "src", "identity", "reference", "ref")) else "target"
                result.append((node, role))
        if len(result) > 1 and sum(role == "target" for _, role in result) > 1:
            # For anonymous two-image graphs, graph order is the only available contract.
            result[0] = (result[0][0], "target")
            result[1] = (result[1][0], "source")
            for index in range(2, len(result)):
                result[index] = (result[index][0], "unsupported")
        return result

    @staticmethod
    def _image_tensor(image: np.ndarray, node, value_range: str) -> np.ndarray:
        shape = _shape(node)
        if len(shape) != 4:
            raise DomainError("SWAPPER_GRAPH_UNSUPPORTED", "Image input must be rank four", 422)
        if type(shape[1]) is int and shape[1] == 3:
            height, width, channels_last = _static_dim(shape[2], image.shape[0]), _static_dim(shape[3], image.shape[1]), False
        elif type(shape[3]) is int and shape[3] == 3:
            height, width, channels_last = _static_dim(shape[1], image.shape[0]), _static_dim(shape[2], image.shape[1]), True
        else:
            raise DomainError("SWAPPER_GRAPH_UNSUPPORTED", "Swapper image input must have three channels", 422)
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA if height < image.shape[0] else cv2.INTER_LINEAR)
        if value_range == "minus1_1":
            value = resized.astype(np.float32) / 127.5 - 1.0
        else:
            value = resized.astype(np.float32) / 255.0
        value = value[:, :, ::-1]  # roop frames and FaceSet crops are BGR; model image tensors are RGB.
        value = value.transpose(2, 0, 1) if not channels_last else value
        return np.ascontiguousarray(value[None], dtype=_dtype(node))

    @staticmethod
    def _embedding_tensor(embedding: np.ndarray, node) -> np.ndarray:
        shape = _shape(node)
        vector = np.asarray(embedding, np.float32).reshape(-1)
        if vector.size != 512:
            raise DomainError("IDENTITY_EMBEDDING_INVALID", "Swapper identity input requires a 512-D ArcFace vector", 422)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(vector).all() or norm < 1e-8:
            raise DomainError("IDENTITY_EMBEDDING_INVALID", "Swapper identity vector must be finite and nonzero", 422)
        vector = vector / norm
        expected = int(np.prod([dim for dim in shape if type(dim) is int and dim > 0]))
        if expected not in (0, 512):
            raise DomainError("SWAPPER_IDENTITY_CONTRACT", "Inspected identity tensor does not match ArcFace 512-D input", 422)
        if len(shape) == 1:
            packed = vector
        else:
            dims = [1 if index == 0 and (type(dim) is not int or dim <= 0) else _static_dim(dim, 1)
                    for index, dim in enumerate(shape)]
            if int(np.prod(dims)) != 512:
                dims = [1, 512]
            packed = vector.reshape(dims)
        return np.ascontiguousarray(packed, dtype=_dtype(node))

    def source_crop(self, view) -> np.ndarray:
        cache_key = (id(view), self.resolution)
        cached = self._source_cache.get(cache_key)
        if cached is not None:
            return cached
        encoded = np.frombuffer(view.image_bytes, np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        points = np.asarray(view.landmarks, np.float64)
        if image is None or points.ndim != 2 or points.shape[0] < 5:
            raise DomainError("SOURCE_CROP_INVALID", "Selected FaceSet view has no decodable source crop", 422)
        crop = aligned_crop(image, points[:5], self.resolution,
                            template=self.spec.get("alignment_template", "arcface_112")).image
        self._source_cache[cache_key] = crop
        return crop

    def _gpu_bound_inputs(self, feed: dict[str, np.ndarray]):
        """Bind tensors from CUDA memory directly when the optional CUDA Torch bridge exists.

        The reference cropper currently emits host arrays. This bridge performs that
        one explicit upload, then I/O Binding passes the CUDA pointers to ORT without
        an additional ORT-side staging copy.
        """
        if self._binding is None or getattr(self._binding, "device_name", "cpu") != "cuda":
            return feed
        try:
            import torch
            if not torch.cuda.is_available():
                return feed
            return {name: torch.from_numpy(np.ascontiguousarray(value)).to("cuda", non_blocking=False)
                    for name, value in feed.items()}
        except (ImportError, RuntimeError, OSError):
            return feed

    def infer(self, crop_bgr: np.ndarray, embedding: np.ndarray, *, source_view=None, morph_factor: float | None = None) -> np.ndarray:
        if self.session is None or not self._inputs or not self._outputs:
            raise DomainError("SESSION_NOT_PREPARED", "Prepare the selected swapper before inference", 409)
        if crop_bgr.shape[:2] != (self.resolution, self.resolution):
            raise DomainError("SWAPPER_CROP_SHAPE", "Aligned crop does not match the model's native resolution", 422)
        if self.spec.get("identity_mode") == "dfm":
            return self._infer_dfm(crop_bgr, morph_factor)
        feed: dict[str, np.ndarray] = {}
        image_specs = self._image_input_specs()
        vector_specs = [node for node in self._inputs if len(_shape(node)) in (1, 2, 4)
                        and (int(np.prod([d for d in _shape(node) if type(d) is int and d > 0])) in (0, 512)
                             or any(word in node.name.casefold() for word in ("id", "embed", "identity", "source")))]
        source = self.source_crop(source_view) if source_view is not None and any(role == "source" for _, role in image_specs) else None
        for node, role in image_specs:
            if role == "unsupported":
                raise DomainError("SWAPPER_GRAPH_UNSUPPORTED", "Swapper graph exposes more than two image inputs", 422)
            image = crop_bgr if role == "target" else source
            if image is None:
                raise DomainError("SWAPPER_SOURCE_REQUIRED", "Selected architecture requires a source crop from the FaceSet", 422)
            feed[node.name] = self._image_tensor(image, node, self.spec.get("input_range", "minus1_1"))
        for node in vector_specs:
            feed[node.name] = self._embedding_tensor(embedding, node)
        if len(feed) != len(self._inputs):
            names = sorted(set(node.name for node in self._inputs) - set(feed))
            raise DomainError("SWAPPER_GRAPH_UNSUPPORTED", "Unmapped swapper graph inputs: " + ", ".join(names), 422)
        try:
            outputs = self._runner(self._gpu_bound_inputs(feed))
            self.executions += 1
            values = {}
            for name, value in outputs.items():
                if hasattr(value, "numpy"):
                    value = value.numpy()
                values[name] = np.asarray(value)
            image_value = next((value for value in values.values() if value.ndim == 4 and 3 in (value.shape[1], value.shape[-1])), None)
            if image_value is None:
                raise ValueError("no image output")
            if image_value.shape[1] == 3:
                image_value = image_value[0].transpose(1, 2, 0)
            else:
                image_value = image_value[0]
            image_value = image_value.astype(np.float32)
            if not np.isfinite(image_value).all():
                raise ValueError("non-finite output")
            if self.spec.get("output_range", "minus1_1") == "minus1_1":
                image_value = (image_value + 1.0) * 0.5
            image_value = np.clip(image_value, 0.0, 1.0)
            if image_value.shape[:2] != (self.resolution, self.resolution):
                image_value = cv2.resize(image_value, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
            return np.clip(image_value[:, :, ::-1] * 255.0, 0, 255).round().astype(np.uint8)
        except DomainError:
            raise
        except Exception as error:
            raise DomainError("MODEL_INFERENCE_FAILED", f"Swapper inference failed ({type(error).__name__})", 503) from None

    def run(self, values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Compatibility hook for existing diagnostic callers using raw graph tensors."""
        if self.model is not None:
            return self.model.run(dict(values))
        expected = {node.name for node in self._inputs}
        if set(values) != expected:
            raise DomainError("INFERENCE_INPUT_NAMES", "Input names do not match the inspected swapper graph", 422)
        for node in self._inputs:
            value, shape = values[node.name], _shape(node)
            if not isinstance(value, np.ndarray) or value.dtype != _dtype(node) or value.ndim != len(shape):
                raise DomainError("INFERENCE_TENSOR_INVALID", "Raw swapper tensor shape or dtype is invalid", 422)
            if any(type(dim) is int and dim > 0 and dim != actual for dim, actual in zip(shape, value.shape)):
                raise DomainError("INFERENCE_TENSOR_INVALID", "Raw swapper tensor does not match the inspected graph", 422)
        try:
            raw = self._runner(values)
            return {name: value.numpy() if hasattr(value, "numpy") else np.asarray(value)
                    for name, value in raw.items()}
        except DomainError:
            raise
        except Exception as error:
            raise DomainError("MODEL_INFERENCE_FAILED", f"Swapper inference failed ({type(error).__name__})", 503) from None

    def release(self) -> None:
        if self.model is not None:
            self.model.release()
        if getattr(self, "_active_key", None) is not None:
            from roop_3d.engine import active_models
            active_models.release("swapper", self._active_key)
            self._active_key = None
        if self.catalog:
            self.catalog.set_vram_loaded(self.model_id, False)
        self.session = None
        self.model = None
        self._binding = None
        self._source_cache.clear()

    def capability(self) -> dict[str, Any]:
        if self.model is not None:
            return {**self.model.capability(), "native_resolution": self.resolution,
                    "identity_mode": self.spec.get("identity_mode", "source_crop")}
        providers = tuple(self.session.get_providers()) if self.session is not None and hasattr(self.session, "get_providers") else ()
        provider = providers[0] if providers else self.config.provider
        return {"model_id": self.model_id, "sha256": None, "artifact_state": "LOCAL_UNVERIFIED" if self.path else "REGISTERED",
                "inference_state": "LOCAL_INFERENCE" if self.executions else "UNKNOWN", "provider": provider,
                "executions": self.executions, "requested_provider": self.config.provider,
                "fallback_reason": None, "native_resolution": self.resolution,
                "identity_mode": self.spec.get("identity_mode", "source_crop")}
