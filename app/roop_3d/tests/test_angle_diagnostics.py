import hashlib
import json

import cv2
import numpy as np
import pytest

from roop_3d.engine.volumetric import angle_diagnostics as ad


def affine(scale=1.0, degrees=0.0, tx=0.0, ty=0.0):
    a, b = scale * np.cos(np.radians(degrees)), scale * np.sin(np.radians(degrees))
    return np.array(((a, -b, tx), (b, a, ty)))


LANDMARKS = np.array(((100, 100), (160, 100), (130, 130), (110, 160), (150, 160)), np.float64)


def test_decomposition_recovers_similarity_parameters():
    s, theta, tx, ty = ad.decompose_similarity(affine(1.7, -23.0, 4.0, -9.0))
    assert (s, theta, tx, ty) == pytest.approx((1.7, -23.0, 4.0, -9.0))


def test_second_difference_ignores_constant_velocity_but_sees_noise():
    steady = [LANDMARKS + (3.0 * i, -1.0 * i) for i in range(3)]
    assert ad.landmark_jitter(*steady) == pytest.approx(0.0, abs=1e-12)
    noisy = [steady[0], steady[1] + (1.5, 0.0), steady[2]]
    # second difference of one 1.5 px outlier in the middle = 3.0 px on every point; IOD = 60 px
    assert ad.landmark_jitter(*noisy) == pytest.approx(3.0 / 60.0)


def test_affine_jitter_is_scale_and_unwrapped_angle_aware():
    still = [ad.decompose_similarity(affine(2.0, 179.0 + i, 10 + 4 * i, 5)) for i in range(3)]
    jitter = ad.affine_jitter(*still, iod=60.0)
    assert jitter["dtheta_deg"] == pytest.approx(0.0, abs=1e-9) and jitter["dt_rel"] == pytest.approx(0.0, abs=1e-9)
    bumped = [ad.decompose_similarity(affine(s, 0.0, 0.0, 0.0)) for s in (2.0, 2.2, 2.0)]
    assert ad.affine_jitter(*bumped, iod=60.0)["ds_rel"] == pytest.approx(0.4 / 2.2)


def test_ring_metric_counts_dark_fringe_only_where_plate_was_bright():
    plate = np.full((40, 40, 3), 120, np.uint8)
    plate[:, :6] = 10                                  # a genuinely dark background strip stays unflagged
    final = plate.copy()
    alpha = np.zeros((40, 40), np.float32)
    alpha[10:30, 10:30] = 1.0
    alpha[8:10, 10:30] = 0.3                           # outer feather ring above the face
    final[8:10, 10:30] = 5                             # ... which came out black while the plate was bright
    alpha[30:32, 10:30] = 0.7                          # inner ring
    alpha[10:30, 2:4] = 0.3                            # ring over the dark strip
    final[10:30, 2:4] = 5
    stats = ad.ring_luma_stats(alpha, final, plate)
    assert stats["dark_px"] == 2 * 20
    assert stats["out_min"] < ad.DARK_LUMA
    clean = ad.ring_luma_stats(alpha, plate, plate)
    assert clean["dark_px"] == 0 and clean["in_luma_step"] == pytest.approx(0.0)


def test_frame_metrics_maps_crop_alpha_into_frame_and_measures_sharpness_ratio():
    rng = np.random.default_rng(3)
    plate = rng.integers(60, 200, (120, 120, 3), dtype=np.uint8)
    output = cv2.GaussianBlur(plate, (0, 0), 3)       # a soft replacement is much less sharp than the plate
    alpha = cv2.GaussianBlur(np.pad(np.ones((40, 40), np.float32), 12), (0, 0), 3)
    inverse = affine(1.0, 0.0, 30.0, 30.0)
    metrics = ad.frame_metrics(plate, output, alpha, inverse)
    assert 0.0 < metrics["sharpness_ratio"] < 0.2
    assert "ring" in metrics


def test_mask_stats_are_resolution_independent():
    small = np.zeros((128, 128), np.float32)
    small[32:96, 40:100] = 1
    large = cv2.resize(small, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    a, b = ad.mask_stats(small), ad.mask_stats(large)
    assert a["area"] == pytest.approx(b["area"], rel=0.02)
    assert a["cx"] == pytest.approx(b["cx"] / 8, abs=0.6) and a["cy"] == pytest.approx(b["cy"] / 8, abs=0.6)


def swap_diag(frame, views, shift=0.0):
    coverage = np.zeros((64, 64), np.float32)
    coverage[16:48, 16 + int(shift):48 + int(shift)] = 1
    return {"crop_matrix": affine(1.0, 0.0, 0.0, 0.0), "landmarks5": LANDMARKS + (frame * 2.0, 0.0),
            "coverage": coverage, "coverage_scale": 1.0, "views": views}


def view(index, weight, yaw=0.0):
    return {"index": index, "yaw": yaw, "pitch": 0.0, "raw": 0.1, "weight": weight}


def test_recorder_logs_view_switches_dwell_and_weight_steps(tmp_path):
    diagnostics = ad.AngleDiagnostics()
    diagnostics.configure("summary")
    for frame, views in enumerate([[view(1, 1.0)], [view(1, 1.0)], [view(1, .6), view(2, .4)], [view(1, .6), view(2, .4)]]):
        diagnostics.record_swap(frame, 0, swap_diag(frame, views), pose=(10.0, 0.0, 0.0))
    assert [r for r in diagnostics.records()] == []       # not finished: no composite half yet (summary level)
    summary = diagnostics.summary()
    assert summary["frames"] == 4
    assert summary["source_views"]["switches"] == 1
    assert summary["source_views"]["max_weight_delta"] == pytest.approx(0.4)
    assert summary["landmarks5_second_difference_over_iod"]["n"] == 2
    assert summary["landmarks5_second_difference_over_iod"]["max"] == pytest.approx(0.0, abs=1e-6)
    assert "mask_edge" not in summary                      # image metrics are level "full" only
    json.dumps(summary)


def test_full_level_streams_one_merged_record_per_frame_and_publishes_atomically(tmp_path):
    path = tmp_path / "job.angle.jsonl"
    diagnostics = ad.AngleDiagnostics()
    diagnostics.configure("full", path)
    for frame in range(3):
        diagnostics.record_swap(frame, 7, swap_diag(frame, [view(3, 1.0)], shift=frame), pose=(34.0, 2.0, 0.0))
        diagnostics.record_composite(frame, 7, {"ring": {"ring_px": 10, "dark_px": 4 if frame == 1 else 0, "in_luma_step": 2.0},
                                                "sharpness_ratio": 0.5})
    assert not path.exists() and (tmp_path / "job.angle.jsonl.partial").exists()      # only the partial exists while running
    artifact = diagnostics.finalize()
    assert not (tmp_path / "job.angle.jsonl.partial").exists() and path.exists()
    assert artifact == {"path": "job.angle.jsonl", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": 3}
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["frame"] for r in lines] == [0, 1, 2]
    assert set(lines[2]) >= {"frame", "track", "pose", "crop", "coverage", "jitter", "views", "ring", "sharpness"}
    assert lines[2]["jitter"]["lm5"] == pytest.approx(0.0, abs=1e-6)
    assert lines[1]["coverage"]["d_centroid_px"] == pytest.approx(1.0, abs=0.1)
    assert lines[1]["ring"]["dark_px"] == 4 and lines[0]["sharpness"]["yaw_bucket"] == 30
    summary = diagnostics.summary()
    assert summary["mask_edge"]["frames_measured"] == 3 and summary["mask_edge"]["frames_with_dark_px"] == 1
    assert summary["mask_edge"]["dark_px_total"] == 4
    assert summary["sharpness_ratio_by_abs_yaw"]["30-40"]["n"] == 3
    assert summary["jsonl"] == artifact and summary["level"] == "full" and summary["level_source"] == "job"


def test_summary_level_writes_nothing_and_reports_no_artifact(tmp_path):
    path = tmp_path / "job.angle.jsonl"
    diagnostics = ad.AngleDiagnostics()
    diagnostics.configure("summary", path)
    diagnostics.record_swap(0, 0, swap_diag(0, [view(1, 1.0)]), pose=(0.0, 0.0, 0.0))
    assert diagnostics.finalize() is None
    assert list(tmp_path.iterdir()) == []
    summary = diagnostics.summary()
    assert summary["level"] == "summary" and "jsonl" not in summary and "mask_edge" not in summary


def test_discard_removes_the_partial_and_the_published_file(tmp_path):
    path = tmp_path / "job.angle.jsonl"
    diagnostics = ad.AngleDiagnostics()
    diagnostics.configure("full", path)
    diagnostics.record_swap(0, 0, swap_diag(0, [view(1, 1.0)]), pose=(0.0, 0.0, 0.0))
    diagnostics.record_composite(0, 0, {})
    diagnostics.discard()
    assert list(tmp_path.iterdir()) == [] and diagnostics.artifact is None
    diagnostics.configure("full", path)                       # a published artifact is removed too
    diagnostics.record_swap(0, 0, swap_diag(0, [view(1, 1.0)]), pose=(0.0, 0.0, 0.0))
    diagnostics.record_composite(0, 0, {})
    assert diagnostics.finalize()["rows"] == 1 and path.exists()
    diagnostics.discard()
    assert list(tmp_path.iterdir()) == []


def test_a_full_job_without_any_dense_face_publishes_no_empty_file(tmp_path):
    diagnostics = ad.AngleDiagnostics()
    diagnostics.configure("full", tmp_path / "job.angle.jsonl")
    assert diagnostics.finalize() is None and list(tmp_path.iterdir()) == []
    assert diagnostics.summary()["jsonl"] is None


def test_noncontiguous_frames_restart_second_differences():
    diagnostics = ad.AngleDiagnostics()
    diagnostics.configure("summary")
    for frame in (0, 1, 5):          # a gap: the third sample has no valid predecessor pair
        diagnostics.record_swap(frame, 0, swap_diag(frame, [view(1, 1.0)]), pose=(0.0, 0.0, 0.0))
    assert diagnostics.summary()["landmarks5_second_difference_over_iod"]["n"] == 0
