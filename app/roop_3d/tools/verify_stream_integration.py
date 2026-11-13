"""Real synthetic-clip gate; no production model/hardware claim is inferred."""
from __future__ import annotations

from fractions import Fraction
import json
import multiprocessing as mp
from pathlib import Path
import tempfile
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module=r"albumentations(?:\..*)?")

import av
import cv2
import numpy as np

from roop_3d.audit.project_paths import application_root
from roop_3d.core.storage import owned
from roop_3d.models.owned_files import owned_path
from roop_3d.hardware_streamer import SharedMemoryFrameRing, FrameMetadata
from roop_3d.models.emap_cache import cached_emap
from roop_3d.optimized_processor import StreamingProcessor
from roop_3d.render_policy import RenderPolicy
from roop_3d.vectorized_pipeline import CpuBatchPipeline, crop_matrix, affine_graph, normalized_matrix, composite_cpu
from roop_3d.faceset.recognizer import ARCFACE_REFERENCE_5PTS


def _ring_child(ring):
    # Any per-frame pickle use causes this child to fail rather than silently pass.
    import multiprocessing.reduction
    multiprocessing.reduction.ForkingPickler.dumps = staticmethod(lambda *_: (_ for _ in ()).throw(AssertionError("Frame pickle forbidden")))
    try:
        for index in range(7):
            ring.put(np.full(ring.shape, index, np.uint8), FrameMetadata(index, index * 3, 1, 75, index == 0))
        ring.finish()
    finally:
        ring.close()


def verify_ring():
    context = mp.get_context("spawn")
    with SharedMemoryFrameRing((16, 16, 3), 2, context=context) as ring:
        process = context.Process(target=_ring_child, args=(ring,))
        process.start()
        try:
            for index in range(7):
                with ring.lease(timeout=10) as item:
                    assert item is not None
                    frame, metadata = item
                    assert metadata.index == index and metadata.pts == index * 3
                    assert not frame.flags.writeable and np.all(frame == index)
                del item, frame
            with ring.lease(timeout=10) as end:
                assert end is None
            process.join(timeout=10)
            assert not process.is_alive() and process.exitcode == 0
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)


def create_clip(path: Path, size: int = 64, *, audio: bool = False):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=25)
        stream.width, stream.height, stream.pix_fmt = size, size, "yuv420p"
        stream.codec_context.max_b_frames = 0
        # A properly tagged BT.709 limited-range source: untagged matrix/range is routed to the colour-managed reference path.
        context = stream.codec_context
        context.colorspace = context.color_primaries = context.color_trc = 1
        context.color_range = 1
        audio_stream = container.add_stream("aac", rate=48000) if audio else None
        if audio_stream:
            audio_stream.layout = "mono"
        for index in range(10):
            image = np.full((size, size, 3), (40, 80, 20), np.uint8)
            if 2 <= index <= 6:
                image[8:44, :30] = (240, 240, 240)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24").reformat(format="yuv420p", dst_colorspace=1, dst_color_range=1)
            frame.pts, frame.time_base = index, Fraction(1, 25)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
        if audio_stream:
            samples = (np.sin(np.arange(19200) * (2 * np.pi * 440 / 48000)) * 0.2).astype(np.float32)[None]
            frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="mono")
            frame.sample_rate, frame.pts, frame.time_base = 48000, 0, Fraction(1, 48000)
            for packet in audio_stream.encode(frame):
                container.mux(packet)
            for packet in audio_stream.encode(None):
                container.mux(packet)


def verify_geometry():
    import onnxruntime as ort
    image = np.linspace(0, 1, 1 * 3 * 48 * 64, dtype=np.float32).reshape(1, 3, 48, 64)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    crop_session = ort.InferenceSession(affine_graph(48, 64, 1).SerializeToString(), sess_options=options,
                                       providers=["CPUExecutionProvider"])
    composite = ort.InferenceSession(affine_graph(48, 64, 1, compose=True).SerializeToString(), sess_options=options,
                                    providers=["CPUExecutionProvider"])
    for translation in ((-100, -80), (0, 0), (60, 45), (150, 120)):
        points = (ARCFACE_REFERENCE_5PTS + [8, 0]) * 0.4 + translation
        matrix = crop_matrix(points)
        inverse = normalized_matrix(cv2.invertAffineTransform(matrix), 48, 64)[None]
        crop = crop_session.run(None, {"image": image, "matrix": inverse})[0]
        actual = composite.run(None, {"image": image, "crops": crop,
            "matrix": normalized_matrix(matrix, 128, 128)[None]})[0]
        assert crop.shape == (1, 3, 128, 128) and np.isfinite(actual).all()
        assert 0 <= actual.min() <= actual.max() <= 1.001
        if translation == (150, 120) or translation == (-100, -80):
            np.testing.assert_allclose(actual, image, atol=1e-6)


def verify_emap(directory):
    source = directory / "synthetic_projection.onnx"
    import onnx
    data = np.eye(512, dtype=np.float32)
    h, t = onnx.helper, onnx.TensorProto
    model = h.make_model(h.make_graph([h.make_node("MatMul", ["input", "buff2fs"], ["output"])], "emap_fixture",
        [h.make_tensor_value_info("input", t.FLOAT, [1, 512])], [h.make_tensor_value_info("output", t.FLOAT, [1, 512])],
        [onnx.numpy_helper.from_array(data, "buff2fs")]), opset_imports=[h.make_opsetid("", 17)])
    onnx.save(model, source)
    from roop_3d.engine.volumetric.identity_swapper import embedding_map
    destination = directory / "inswapper_128_emap.npy"
    first = cached_emap(source, embedding_map, destination=destination)
    second = cached_emap(source, lambda _: (_ for _ in ()).throw(AssertionError("Model protobuf was reparsed")), destination=destination)
    np.testing.assert_array_equal(first, data)
    np.testing.assert_array_equal(second, data)


def verify_fp32_precision_rewrite(directory):
    import onnx
    from onnx import TensorProto, helper
    from roop_3d.audit.project_paths import application_root
    from roop_3d.core.ort_session import _fp32_safe_model_path

    path = directory / "gpen_precision_fixture.onnx"
    initializer = [
        onnx.numpy_helper.from_array(np.ones((3,), np.float16), "scale"),
        onnx.numpy_helper.from_array(np.zeros((3,), np.float16), "bias"),
        onnx.numpy_helper.from_array(np.ones((1, 3, 1, 1), np.float16), "gain"),
    ]
    graph = helper.make_graph([
        helper.make_node("InstanceNormalization", ["x", "scale", "bias"], ["normalized"]),
        helper.make_node("Softmax", ["normalized"], ["probabilities"], axis=1),
        helper.make_node("Mul", ["probabilities", "gain"], ["output"]),
    ], "gpen_fp32_policy_fixture",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 3, 8, 8])], initializer)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, path)
    derived, changed_model = _fp32_safe_model_path(
        path, application_root() / "models" / ".trt_cache", "gpen_bfr_512.onnx")
    assert derived != path and changed_model is not None and derived.is_file()
    rewritten = onnx.load(str(derived), load_external_data=False)
    cast_targets = [attribute.i for node in rewritten.graph.node if node.op_type == "Cast"
                    for attribute in node.attribute if attribute.name == "to"]
    assert TensorProto.FLOAT in cast_targets and TensorProto.FLOAT16 in cast_targets
    assert [value.name for value in rewritten.graph.output] == ["output"]


def verify_gpu_inference_benchmark(*, iterations: int = 5) -> dict:
    """Exercise installed production artifacts under strict GPU placement.

    The benchmark is opt-in because it compiles local TensorRT engines and needs
    the model weights plus an Ampere-or-newer NVIDIA GPU. CPU timings are a real
    per-model reference; no synthetic compute graph is presented as a pipeline
    speedup.
    """
    import gc
    import os
    import time
    import onnx
    import onnxruntime as ort
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("The GPU benchmark requires the optional gpu-compositor PyTorch CUDA extra") from error
    from roop_3d.core.ort_session import GpuIoBinding, _static_model_path
    from roop_3d.models.catalog import ModelCatalog
    from roop_3d.models.dynamic_shapes import sample_inputs
    from roop_3d.models.owned_files import cache_directory
    from roop_3d.trt_session_builder import (BatchProfile, TensorRTBuildError, TensorRTSessionBuilder,
                                             auto_shape_profiles)

    if "TensorrtExecutionProvider" not in ort.get_available_providers() or "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("The GPU benchmark requires TensorRT and CUDAExecutionProvider")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (8, 0):
        raise RuntimeError("The GPU benchmark requires an Ampere-or-newer CUDA device for FP16 Tensor Cores")
    catalog = ModelCatalog()
    installed = []
    for record in catalog.records():
        try:
            path = catalog.resolve_for_inference(record["model_id"])
        except Exception:
            continue
        installed.append((record, path))

    groups = (
        ("detector", ("scrfd", "retinaface")),
        ("pose_reconstruction", ("smirk", "emoca", "flame", "liveportrait_motion", "1k3d68")),
        ("swapper", ("inswapper_128",)),
        ("face_parser", ("bisenet", "xseg", "face_parser")),
        ("ultramax", ("ultramax", "codeformer.fp16")),
        ("gpen_realistic", ("gpen",)),
    )
    chosen = []
    used = set()
    missing = []
    for label, tokens in groups:
        match = next(((record, path) for record, path in installed
                      if record["name"].casefold() not in used
                      and any(token in record["name"].casefold() for token in tokens)), None)
        if match is None:
            missing.append(label)
            continue
        chosen.append((label, *match))
        used.add(match[0]["name"].casefold())
    if missing:
        raise RuntimeError("GPU benchmark needs verified installed models for: " + ", ".join(missing))

    capability = torch.cuda.get_device_capability(0)
    builder = TensorRTSessionBuilder(hardware_profile="12gb")
    results = []
    fp16_executed = False
    cpu_total_seconds = gpu_total_seconds = 0.0
    for label, record, path in chosen:
        model = onnx.load(str(path), load_external_data=False)
        feeds = sample_inputs(model, 1, auto_shape_profiles(model, record["name"])[1])
        batch = next((int(value.shape[0]) for value in feeds.values() if value.ndim == 4), 1)
        session = None
        cpu = None
        runner = None
        used_trt = False
        fallback_reason = None
        try:
            profile = BatchProfile((batch,), batch)
            builder.last_build = {}
            try:
                session = builder.build(path, profile=profile, canary_inputs=lambda _batch, f=feeds: f)
                used_trt = True
            except TensorRTBuildError as error:
                # Unsupported dynamic partitions can use CUDA EP, while CPU EP
                # remains disabled and is independently checked from profiling.
                fallback_reason = str(error)
                options = ort.SessionOptions()
                options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
                options.enable_profiling = True
                options.intra_op_num_threads = 2
                trace = owned_path(cache_directory() / f"gpu_benchmark_cuda_{os.getpid()}_{label}")
                options.profile_file_prefix = str(trace)
                gpu_options = {"device_id": 0,
                    "gpu_mem_limit": max(1, torch.cuda.get_device_properties(0).total_memory - 768 * 1048576),
                    "arena_extend_strategy": "kNextPowerOfTwo", "use_tf32": False,
                    "cudnn_conv_algo_search": "DEFAULT", "cudnn_conv_use_max_workspace": False}
                cuda_path, _ = _static_model_path(path, cache_directory(), record["name"])
                session = ort.InferenceSession(str(cuda_path), sess_options=options,
                    providers=[("CUDAExecutionProvider", gpu_options)])
                session.disable_fallback()
                session._model_path = str(cuda_path)
                if not session.get_providers() or session.get_providers()[0] != "CUDAExecutionProvider":
                    raise RuntimeError(f"{label} did not activate its CUDA-only recovery session")
            precision = builder.last_build["precision"] if used_trt else "CUDA"

            runner = GpuIoBinding(session)
            sample_outputs = runner.run(feeds)
            output_values = [sample_outputs[node.name].numpy() for node in session.get_outputs()]
            if any(not np.isfinite(value).all() for value in output_values):
                raise RuntimeError(f"{label} produced NaN or Inf under TensorRT")
            saturation = sum(int(np.count_nonzero(np.abs(value.astype(np.float32)) >= np.finfo(np.float16).max))
                             for value in output_values) if precision == "FP16" else 0
            if saturation:
                raise RuntimeError(f"{label} saturated FP16 outputs ({saturation} values)")

            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            cpu_path = Path(getattr(session, "_model_path", path))
            # ORT's transformer optimizer currently fails to rewrite the FP16
            # CodeFormer LayerNorm graph after the explicit FP32 cast surgery.
            # Disable graph rewrites for that CPU reference so both providers
            # execute the same derived ONNX graph.
            if "codeformer.fp16" in record["name"].casefold():
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            cpu = ort.InferenceSession(str(cpu_path), sess_options=options, providers=["CPUExecutionProvider"])
            for _ in range(2):
                cpu.run(None, feeds)
                runner.run(feeds)
            cpu_times, trt_times = [], []
            for _ in range(iterations):
                started = time.perf_counter()
                cpu.run(None, feeds)
                cpu_times.append(time.perf_counter() - started)
                started = time.perf_counter()
                runner.run(feeds)
                trt_times.append(time.perf_counter() - started)
            cpu_ms = float(np.median(cpu_times) * 1000)
            trt_ms = float(np.median(trt_times) * 1000)
            speedup = cpu_ms / max(trt_ms, 1e-9)
            cpu_total_seconds += sum(cpu_times)
            gpu_total_seconds += sum(trt_times)
            if used_trt:
                placement = list(builder.last_build["node_providers"])
            else:
                trace_path = Path(session.end_profiling())
                events = json.loads(trace_path.read_text("utf-8"))
                placement = sorted({event.get("args", {}).get("provider") for event in events
                                    if event.get("cat") == "Node" and event.get("args", {}).get("provider")})
            if ("CPUExecutionProvider" in placement
                    or not ({"CUDAExecutionProvider", "TensorrtExecutionProvider"} & set(placement))):
                raise RuntimeError(f"{label} has invalid GPU node placement: {placement}")
            if used_trt and "TensorrtExecutionProvider" not in placement:
                raise RuntimeError(f"{label} TensorRT profiling found no TensorRT nodes: {placement}")
            expected_precision = {"ultramax": "FP16", "gpen_realistic": "FP32"}.get(label)
            if expected_precision is not None and (not used_trt or precision != expected_precision):
                raise RuntimeError(
                    f"{label} did not meet its numerically safe TensorRT precision policy "
                    f"(expected {expected_precision}, got {precision})"
                )
            fp16_executed |= used_trt and precision == "FP16" and "TensorrtExecutionProvider" in placement
            results.append({"stage": label, "model": record["name"], "precision": precision,
                            "node_providers": placement, "cpu_ms_per_batch": round(cpu_ms, 3),
                            "trt_ms_per_batch": round(trt_ms, 3), "speedup": round(speedup, 2),
                            "frames_per_batch": batch, "fp16_saturation_values": saturation,
                            "cuda_fallback_reason": fallback_reason})
        finally:
            runner = None
            cpu = None
            session = None
            gc.collect()
            torch.cuda.empty_cache()
    if not fp16_executed:
        raise RuntimeError("No FP16 TensorRT model executed on the Tensor-Core-capable GPU")
    aggregate_speedup = cpu_total_seconds / max(gpu_total_seconds, 1e-9)
    if aggregate_speedup <= 4:
        raise RuntimeError(f"Aggregate measured CPU-to-GPU throughput was {aggregate_speedup:.2f}x; required >4x")
    return {"state": "PASS", "device": torch.cuda.get_device_name(0),
            "compute_capability": f"{capability[0]}.{capability[1]}",
            "fp16_tensor_core_eligible": True,
            "tensor_core_evidence": "FP16 TensorRT execution on Ampere-or-newer hardware; ORT node profiling reports EP placement, not per-kernel Tensor Core counters",
            "cpu_node_fallbacks": 0, "iterations": iterations, "models": results,
            "aggregate_speedup": round(aggregate_speedup, 2),
            "minimum_speedup": min(value["speedup"] for value in results)}

MIN_DETAIL_GAIN = 3.0        # required spatial-frequency (Laplacian variance) gain of an enhancer over the unenhanced swap


def _lap_var(image: np.ndarray) -> float:
    from roop_3d.hires_handoff import laplacian_variance
    return laplacian_variance(image)


def _smooth_swap_crop(size: int = 128) -> np.ndarray:
    """A soft, face-like aligned swap crop: what a 128 px swapper hands to Stage 3 (no detail above ~24 cycles)."""
    rng = np.random.default_rng(5)
    base = cv2.GaussianBlur(rng.integers(90, 170, (size, size, 3), dtype=np.uint8), (0, 0), 6)
    cv2.ellipse(base, (size // 2, size // 2), (size // 3, int(size * 0.42)), 0, 0, 360, (150, 170, 205), -1)
    return cv2.GaussianBlur(base, (0, 0), 2.0)


def _compose_enhanced(restorer, crop: np.ndarray, *, blend: float):
    """One face through the production FrameCompositor (standard path), scale 2.0 crop->frame."""
    from types import SimpleNamespace
    from roop_3d.compositor.pipeline import CompositeOptions, FrameCompositor
    from roop_3d.compositor.temporal import TemporalSmoother
    frame = np.full((340, 340, 3), 96, np.uint8)
    inverse = np.array(((2.0, 0, 40), (0, 2.0, 40)), np.float32)
    points = np.array(((70, 90), (230, 90), (150, 150), (95, 210), (205, 210)), np.float32)
    replacement = SimpleNamespace(track_id=1, image=crop, inverse=inverse, landmarks5=((points - 40) / 2.0).astype(np.float32))
    compositor = FrameCompositor(parser=SimpleNamespace(predict=lambda image: None), temporal=TemporalSmoother())
    compositor.restorer = restorer
    out = compositor.compose(frame, (replacement,), {1: points},
                             options=CompositeOptions(restoration_blend=blend, color_strength=0, feather_px=2))
    return out[110:250, 110:250]


class _CropSizedLegacy:
    """The behaviour this gate guards against: the enhanced face forced back to the 128 px swap crop."""
    def __init__(self, inner):
        self.inner = inner

    def restore_aligned(self, crop, landmarks5):
        restored = self.inner.restore_hires(crop, landmarks5)
        return cv2.resize(restored, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_AREA)


def verify_enhancer_detail():
    """UltraMax and GPEN Realistic must deliver distinct, high-resolution detail with >= 3x the Laplacian variance of the
    unenhanced swap, and the enhanced face must reach the frame at its native resolution (not via a 128 px crop).

    Scope: the two engines are explicit fixtures with the production adapters' real tensor contracts (BGR/RGB order,
    [-1, 1] range, 512 px, five-point alignment, UltraMax Laplacian fusion, GPEN luma handling). No production-weights
    claim is made here; ``--real-models`` runs the same measurement with the installed weights.
    """
    from roop_3d.compositor.restoration import GpenRestorer, UltraMaxPipelineRestorer

    def detail_fixture(seed: int, sigma: float):
        noise = cv2.GaussianBlur(np.random.default_rng(seed).normal(0, 1, (512, 512, 3)).astype(np.float32), (0, 0), sigma)
        return noise / max(float(noise.std()), 1e-6)

    ultramax_detail, gpen_detail = detail_fixture(101, 0.9), detail_fixture(202, 1.3)
    # Use the real lazy constructor: spec, timing and outcome telemetry are part of the adapter contract.
    ultramax = UltraMaxPipelineRestorer(None, None)
    ultramax.runner.restore = lambda aligned: np.clip(
        aligned.astype(np.float32) + 38.0 * ultramax_detail, 0, 255).astype(np.uint8)

    gpen = object.__new__(GpenRestorer)
    gpen.model_name, gpen.normalization, gpen.inference_verified = "gpen_bfr_512.onnx", "minus_one_one", False

    class GpenFixtureModel:
        def run(self, feeds):
            tensor = feeds["input"]                                         # RGB, [-1, 1], NCHW float32
            assert tensor.shape == (1, 3, 512, 512) and tensor.dtype == np.float32
            assert -1.0001 <= float(tensor.min()) and float(tensor.max()) <= 1.0001
            return {"output": np.clip(tensor + 0.30 * gpen_detail.transpose(2, 0, 1)[None], -1, 1).astype(np.float32)}
    gpen.model = GpenFixtureModel()

    crop = _smooth_swap_crop()
    baseline = _lap_var(_compose_enhanced(None, crop, blend=0.0))
    landmarks = (ultramax.spec.template / 4.0).astype(np.float32)
    report = {}
    outputs = {}
    for name, engine in (("ultramax", ultramax), ("gpen_realistic", gpen)):
        native = engine.restore_hires(crop, landmarks)
        assert native.shape == (512, 512, 3), f"{name} must hand Stage 3 its native 512 px face, got {native.shape}"
        enhanced = _compose_enhanced(engine, crop, blend=1.0)
        crop_sized = _compose_enhanced(_CropSizedLegacy(engine), crop, blend=1.0)
        gain, legacy_gain = _lap_var(enhanced) / baseline, _lap_var(crop_sized) / baseline
        assert gain >= MIN_DETAIL_GAIN, f"{name}: only {gain:.2f}x the unenhanced swap's Laplacian variance"
        assert gain >= 2.0 * legacy_gain, f"{name}: native warp must clearly beat the 128 px crop path ({gain:.1f}x vs {legacy_gain:.1f}x)"
        report[name] = {"laplacian_gain_vs_unenhanced": round(gain, 2), "crop_sized_path_gain": round(legacy_gain, 2),
                        "restorer_output_px": native.shape[0]}
        outputs[name] = enhanced
    difference = float(np.abs(outputs["ultramax"].astype(np.float32) - outputs["gpen_realistic"].astype(np.float32)).mean())
    assert difference > 1.0, "UltraMax and GPEN Realistic outputs must be measurably distinct"
    report["mean_abs_difference_ultramax_vs_gpen"] = round(difference, 2)
    # UltraMax is never bypassed by default, even for a small destination face.
    ultramax.small_face_mode = "off"
    assert ultramax.restore_hires(crop, landmarks, native_face_size_px=40) is not None
    report["ultramax_default_bypass"] = "off"
    return report


def verify_enhancer_detail_real_models(*, frames=(1200, 3000), require_gain: bool = False):
    """The same measurement on REAL inference: SCRFD + 2d106 + inswapper_128 + BiSeNet + installed enhancer weights, on a
    local video frame, with a local multi-view FaceSet. Reports NOT_RUN (with the reason) when anything is missing.

    Gain is the Laplacian variance of the swapped face interior relative to the unenhanced swap. ``require_gain`` makes a
    miss of ``MIN_DETAIL_GAIN`` fail instead of being reported as BELOW_TARGET.
    """
    import glob
    import shutil
    from roop_3d.audit.project_paths import legacy_root
    from roop_3d.compositor.pipeline import CompositeOptions
    from roop_3d.compositor.restoration import GpenRestorer, UltraMaxPipelineRestorer
    from roop_3d.core.errors import DomainError
    try:
        from roop_3d.engine.volumetric.session import ReferencePipeline
        from roop_3d.faceset.faceset_reader import FaceSetReader
        from roop_3d.reconstruction.identity_asset import IdentityAssetStore
        document = FaceSetReader().read(legacy_root() / "app/facesets/ashna.fsz")
        videos = glob.glob(str(application_root() / "roop_3d/.runtime/media/03de3642*.mp4"))
        if not videos:
            raise FileNotFoundError("local test video 03de3642*.mp4")
    except (DomainError, OSError, KeyError, ValueError, ImportError) as error:
        return {"state": "NOT_RUN", "reason": f"local models, FaceSet or video unavailable: {error}"}

    scratch = owned(application_root(), "roop_3d/.runtime/enhancer-detail-gate")
    scratch.mkdir(parents=True, exist_ok=True)
    session = ReferencePipeline()
    results = []
    try:
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            try:
                session.prepare_session(document, retarget="off", identity_store=IdentityAssetStore(Path(temporary) / "assets"))
            except (DomainError, OSError, ValueError, RuntimeError) as error:
                return {"state": "NOT_RUN", "reason": f"reference session unavailable: {error}"}
            capture = cv2.VideoCapture(videos[0])
            fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            for start in frames:
                capture.set(cv2.CAP_PROP_POS_FRAMES, start)
                ok, frame = capture.read()
                if not ok:
                    continue
                session.set_frame_timestamp(start / fps)
                result = session.infer_frame(frame)
                if not result.replacements:
                    continue
                inverse = np.asarray(result.replacements[0].inverse, np.float64)
                side = result.replacements[0].image.shape[0]
                corners = np.array(((.15, .15), (.85, .15), (.85, .85), (.15, .85))) * side
                mapped = corners @ inverse[:, :2].T + inverse[:, 2]
                x0, y0 = np.maximum(np.floor(mapped.min(0)).astype(int), 0)
                x1, y1 = np.ceil(mapped.max(0)).astype(int)
                interior = lambda image: image[y0:y1, x0:x1]
                options = lambda blend: CompositeOptions(restoration_blend=blend, color_strength=0.65, feather_px=5)
                session.compositor.restorer = None
                plain = interior(session.composite_frame(frame, result, options=options(0.0)))
                row = {"frame": start, "swap_crop_px": side}
                composites = {}
                for name, factory in (("ultramax", UltraMaxPipelineRestorer), ("gpen_realistic", GpenRestorer)):
                    restorer = factory(session.catalog if hasattr(session, "catalog") else None, session.config)
                    session.compositor.restorer = restorer
                    composites[name] = interior(session.composite_frame(frame, result, options=options(1.0)))
                    restorer.release()
                    row[name] = {"laplacian_gain": round(_lap_var(composites[name]) / _lap_var(plain), 2)}
                row["mean_abs_difference"] = round(float(np.abs(composites["ultramax"].astype(np.float32)
                                                          - composites["gpen_realistic"].astype(np.float32)).mean()), 2)
                assert row["mean_abs_difference"] > 0.3, f"frame {start}: UltraMax and GPEN outputs are not distinct"
                for name in ("ultramax", "gpen_realistic"):
                    assert row[name]["laplacian_gain"] > 1.0, f"frame {start}: {name} added no detail"
                    row[name]["meets_target"] = row[name]["laplacian_gain"] >= MIN_DETAIL_GAIN
                    if require_gain:
                        assert row[name]["meets_target"], f"frame {start}: {name} gain {row[name]['laplacian_gain']} < {MIN_DETAIL_GAIN}"
                results.append(row)
            capture.release()
    finally:
        session.release_session()
        shutil.rmtree(scratch, ignore_errors=True)
    if not results:
        return {"state": "NOT_RUN", "reason": "no face was detected in the selected local frames"}
    return {"state": "MEASURED", "target_gain": MIN_DETAIL_GAIN, "frames": results,
            "all_meet_target": all(row[name]["meets_target"] for row in results for name in ("ultramax", "gpen_realistic"))}


def run_verification():
    verify_ring()
    verify_geometry()
    enhancer_report = verify_enhancer_detail()
    base = owned(application_root(), "roop_3d/.runtime/stage3-integration")
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as temporary:
        directory = owned_path(Path(temporary))
        verify_emap(directory)
        verify_fp32_precision_rewrite(directory)
        source = directory / "synthetic.mp4"
        create_clip(source)
        points = ((ARCFACE_REFERENCE_5PTS + [8, 0]) * 0.4 + [-10, 0]).astype(np.float32)
        def detect(image):
            return [1] if image[10:40, :25].mean() > 180 else []
        def track(image):
            return [(0, np.array([-10, 0, 42, 52], np.float32), points)] if detect(image) else []
        events = []
        batches = []
        def inference(target, _):
            batches.append(target.shape[0])
            return target * np.array([0.6, 0.9, 0.7], np.float32)[None, :, None, None]
        pipeline = CpuBatchPipeline(inference)
        policy = RenderPolicy(hardware_profile="cpu", sparse_interval=2, range_padding=2, segment_frames=3, ring_slots=2)
        processor = StreamingProcessor(directory, "synthetic", source, detect=detect, track=track,
            pipeline=pipeline, policy=policy, identity={"fixture": "analytic_no_production_model"}, emit=events.append)
        first = processor.run(directory / "first.mp4", audio=False)
        assert first["frames"] == 10 and first["verified_target_frames"] == 5
        assert max(batches) > 1
        assert len(first["segments"]) == 4 and len(set(first["segments"])) == 4
        assert [e["stage"] for e in events if e["state"] == "DONE"] == [
            "STAGE_1_SPARSE_PRESCAN", "STAGE_2_ACTIVE_PREPASS", "STAGE_3_MAIN_SWAP"]
        events.clear()
        second = processor.run(directory / "resumed.mp4", audio=False)
        assert second["prepass_restored"]
        assert any(e["stage"] == "STAGE_2_ACTIVE_PREPASS" and e["state"] == "RESTORED" for e in events)
        assert set(first["segments"]).isdisjoint(second["segments"])
        return {"state": "PASS", "checks": ["out_of_frame_affine", "cross_process_zero_frame_pickle_ring", "emap_cache",
                "fp32_enhancer_precision_rewrite", "enhancer_laplacian_detail_3x", "three_stage_synthetic_clip",
                "stage2_checkpoint_restore", "unique_stream_copy_segments"],
                "enhancer_detail": enhancer_report,
                "frames": first["frames"], "verified_target_frames": first["verified_target_frames"],
                "max_synthetic_batch": max(batches),
                "scope": "synthetic_CPU_inference_and_real_FFmpeg_streaming; no production_GPU_parity_or_FPS_claim"}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", action="store_true", help="Require real CUDA, NVDEC and NVENC execution on a synthetic clip")
    parser.add_argument("--real-models", action="store_true",
                        help="Measure UltraMax / GPEN Laplacian detail gain on real video with the installed weights")
    parser.add_argument("--require-3x", action="store_true",
                        help="With --real-models: fail when an enhancer misses the 3x Laplacian-variance target")
    parser.add_argument("--benchmark-gpu", action="store_true",
                        help="Compile/benchmark installed production models; require zero CPU nodes and >4x CPU speedup")
    args = parser.parse_args()
    try:
        report = run_verification()
        if args.hardware:
            from roop_3d.tools.verify_stream_hardware import verify_hardware
            report["hardware"] = verify_hardware()
        if args.real_models or args.require_3x:
            report["enhancer_detail_real_models"] = verify_enhancer_detail_real_models(require_gain=args.require_3x)
        if args.benchmark_gpu:
            report["gpu_benchmark"] = verify_gpu_inference_benchmark()
        print(json.dumps(report, indent=2))
        return 0
    except Exception as error:
        import traceback
        traceback.print_exc()
        print(f"STAGE3_INTEGRATION_FAILED: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
