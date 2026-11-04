from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.media.schema import InspectRequest
from roop_3d.media.service import MediaService
from roop_3d.models.catalog import ModelCatalog
from roop_3d.tests.model_governance_support import require_governed_model


@pytest.fixture
def video(tmp_path: Path) -> tuple[MediaService, str]:
    path = tmp_path / "roop_3d/.runtime/media/fixture.mp4"
    path.parent.mkdir(parents=True)
    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 160, 120, "yuv420p"
        stream.codec_context.gop_size = 2
        for index in range(8):
            image = np.full((120, 160, 3), index * 25, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return MediaService(tmp_path, ModelCatalog()), path.relative_to(tmp_path).as_posix()


def test_real_container_probe_and_precise_frame_preview(video) -> None:
    service, path = video
    require_governed_model(service.detector.catalog, service.detector.catalog.detector()["model_id"])
    result = service.probe(path)
    assert result.width == 160 and result.height == 120
    assert result.fps == 10 and result.duration_seconds == pytest.approx(0.8)
    assert result.timing_sample_complete and result.vfr is False
    assert len(result.audio_tracks) == 0
    zero = service.frame(path, None, 0)
    later = service.frame(path, 0.5, None)
    assert float(later.mean()) > float(zero.mean()) + 100
    assert np.abs(service.frame(path, None, 5).astype(float) - later).mean() < 3
    jpeg = service.preview(path, None, 3, path)
    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (120, 320, 3)

    # Test detect_frame_faces directly from preview frame
    res = service.detect_frame_faces(path, timestamp=0.2)
    assert "inspection_id" in res and "faces" in res
    assert isinstance(res["faces"], list)


@pytest.mark.parametrize("path", ["../secret.mp4", "https://example.com/a.mp4", "roop_3d/core/settings.py",
                                 "roop_3d/.runtime/config.json", "roop_3d/.runtime/media/a.m3u8"])
def test_media_read_boundary(video, path: str) -> None:
    service, _ = video
    with pytest.raises(DomainError):
        service.probe(path)


def test_preview_selectors_are_bounded_and_finite(video) -> None:
    service, path = video
    for timestamp, frame in [(None, None), (0, 0), (float("nan"), None), (None, 10000), (999, None)]:
        with pytest.raises(DomainError):
            service.frame(path, timestamp, frame)


def test_real_local_scrfd_inference_no_faces_is_not_a_stub(video) -> None:
    service, path = video
    require_governed_model(service.detector.catalog, service.detector.catalog.detector()["model_id"])
    result = service.inspect(InspectRequest(path=path, max_samples=2))
    assert result.tracks == ()
    assert result.sampled_frames == 2 and result.truncated
    assert len(result.detector_sha256) == 64
    assert service.detector._session.get_providers() == ["CPUExecutionProvider"]
    assert (service.root / f"roop_3d/.runtime/inspections/{result.inspection_id}.json").is_file()


def test_actual_vfr_timestamps_and_audio_stream_metadata(tmp_path: Path) -> None:
    path = tmp_path / "roop_3d/.runtime/media/vfr.mkv"
    path.parent.mkdir(parents=True)
    with av.open(str(path), "w") as output:
        video = output.add_stream("ffv1", rate=25)
        video.width, video.height, video.pix_fmt = 64, 64, "yuv420p"
        video.codec_context.time_base = Fraction(1, 1000)
        audio = output.add_stream("pcm_s16le", rate=48000)
        audio.layout = "mono"
        audio.metadata["language"] = "eng"
        for index, pts in enumerate((0, 40, 120, 160, 280)):
            frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), index * 20, np.uint8), format="bgr24")
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for packet in video.encode(frame):
                output.mux(packet)
        for packet in video.encode():
            output.mux(packet)
        samples = av.AudioFrame.from_ndarray(np.zeros((1, 14400), dtype=np.int16), format="s16", layout="mono")
        samples.sample_rate, samples.pts, samples.time_base = 48000, 0, Fraction(1, 48000)
        for packet in audio.encode(samples):
            output.mux(packet)
        for packet in audio.encode():
            output.mux(packet)
    result = MediaService(tmp_path, ModelCatalog()).probe(path.relative_to(tmp_path).as_posix())
    assert result.vfr is True and result.vfr_state == "VERIFIED_LOCAL"
    assert result.audio_tracks[0].sample_rate == 48000
    assert result.audio_tracks[0].channels == 1
    assert result.audio_tracks[0].language == "eng"


def test_renamed_network_playlist_is_denied_without_connection(tmp_path: Path) -> None:
    import select
    import socket
    path = tmp_path / "roop_3d/.runtime/media/playlist.mp4"
    path.parent.mkdir(parents=True)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        path.write_text(f"#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\nhttp://127.0.0.1:{port}/secret.ts\n#EXT-X-ENDLIST\n")
        with pytest.raises(DomainError) as error:
            MediaService(tmp_path, ModelCatalog()).probe(path.relative_to(tmp_path).as_posix())
        assert error.value.code == "MEDIA_DECODE_FAILED"
        assert not select.select([listener], [], [], 0.1)[0]
