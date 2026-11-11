from pathlib import Path
from types import SimpleNamespace

import pytest

from roop_3d.execution.worker_pools import GopChunk, PostEncodeWorkerPool, encode_nvenc_chunk


def test_post_encode_pool_stitches_completed_chunks_in_frame_order(tmp_path):
    first = tmp_path / "chunk-0.mp4"
    second = tmp_path / "chunk-1.mp4"
    audio = tmp_path / "source.mp4"
    for path in (first, second, audio):
        path.write_bytes(b"fixture")
    stitched = {}

    def stitcher(paths, destination, *, audio_source, ffmpeg):
        stitched.update(paths=paths, destination=destination, audio_source=audio_source, ffmpeg=ffmpeg)
        return Path(destination)

    pool = PostEncodeWorkerPool(workers=2)
    try:
        pool.submit_chunk(GopChunk(4, 7, second, gop_size=4, final=True))
        pool.submit_chunk(GopChunk(0, 4, first, gop_size=4))
        result = pool.finalize(tmp_path / "final.mp4", audio_source=audio, ffmpeg="ffmpeg", stitcher=stitcher)
        assert result.result() == tmp_path / "final.mp4"
    finally:
        pool.close()

    assert stitched["paths"] == [first.resolve(), second.resolve()]
    assert stitched["audio_source"] == audio
    assert stitched["ffmpeg"] == "ffmpeg"


def test_gop_chunk_rejects_unaligned_start_and_nonfinal_end():
    with pytest.raises(ValueError, match="start must be GOP aligned"):
        GopChunk(2, 8, "chunk.mp4", gop_size=4)
    with pytest.raises(ValueError, match="end must be GOP aligned"):
        GopChunk(0, 6, "chunk.mp4", gop_size=4)


def test_nvenc_chunk_runs_on_the_worker_with_lossless_and_gop_options(tmp_path, monkeypatch):
    source, destination = tmp_path / "render.mp4", tmp_path / "chunk.mp4"
    source.write_bytes(b"prepared")
    calls = {}
    monkeypatch.setattr("roop_3d.execution.worker_pools.shutil.which", lambda binary: "ffmpeg.exe")

    def run(command, **kwargs):
        calls["command"] = command
        calls["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr("roop_3d.execution.worker_pools.subprocess.run", run)

    assert encode_nvenc_chunk(source, destination, gop_size=48) == destination.resolve()
    command = calls["command"]
    assert "h264_nvenc" in command
    assert command[command.index("-tune") + 1] == "lossless"
    assert command[command.index("-qp") + 1] == "0"
    assert command[command.index("-g") + 1] == "48"
