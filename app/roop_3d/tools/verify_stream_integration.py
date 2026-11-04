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


def run_verification():
    verify_ring()
    verify_geometry()
    base = owned(application_root(), "roop_3d/.runtime/stage3-integration")
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as temporary:
        directory = owned_path(Path(temporary))
        verify_emap(directory)
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
                "three_stage_synthetic_clip", "stage2_checkpoint_restore", "unique_stream_copy_segments"],
                "frames": first["frames"], "verified_target_frames": first["verified_target_frames"],
                "max_synthetic_batch": max(batches),
                "scope": "synthetic_CPU_inference_and_real_FFmpeg_streaming; no production_GPU_parity_or_FPS_claim"}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", action="store_true", help="Require real CUDA, NVDEC and NVENC execution on a synthetic clip")
    args = parser.parse_args()
    try:
        report = run_verification()
        if args.hardware:
            from roop_3d.tools.verify_stream_hardware import verify_hardware
            report["hardware"] = verify_hardware()
        print(json.dumps(report, indent=2))
        return 0
    except Exception as error:
        import traceback
        traceback.print_exc()
        print(f"STAGE3_INTEGRATION_FAILED: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
