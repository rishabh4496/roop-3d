"""Stage 14 orchestrator acceptance: deterministic selection, no unsupported configuration, no silent
substitution, no uncontrolled VRAM allocation.

The acceptance criteria are proven on *fixture registries*: the shipped manifests with the Stage 12
lifecycle gate stubbed (the real registry has zero READY models, by design) and synthetic evidence records.
One test runs against the real registry and checks it fails closed with a byte-identical report.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Settings
from roop_3d.models.registry import ModelRegistry
from roop_3d.orchestrator import (CapabilityRegistry, EvidenceStore, ManualPick, MediaFacts, OrchestratorRegistryError,
                                  SelectionRequest, Selector, build_report, compile_plan, render_text,
                                  segments_from_tracks)
from roop_3d.orchestrator.registry import ADAPTERS, PACKAGE_DIR, adapter_present
from roop_3d.orchestrator.schema import (CATEGORY_ORDER, EvidenceEngineRef, EvidenceFile, EvidenceRecord, Fallback,
                                         SceneScore)

MODELS = ModelRegistry()
FILES = ("engines_manifest.json", "render_profiles.json", "scene_rules.json")

BASE = ["pyav_video_reader", "scrfd_detector", "arcface_w600k_recognizer", "landmarks_106", "inswapper_reference",
        "liveportrait_retarget", "bisenet_face_parser", "pyav_video_writer"]
XSEG = [e if e != "bisenet_face_parser" else "xseg_visible_face" for e in BASE]
STREAM = ["ffmpeg_stream_source", "scrfd_detector", "arcface_w600k_recognizer", "landmarks_106",
          "inswapper_batch_stream", "ffmpeg_stream_writer"]


def ready(model_id, hardware):
    return {"lifecycle_state": "READY"}


def fixture_registry(tmp_path: Path, mutate=None, lifecycle=ready) -> CapabilityRegistry:
    data = {name: json.loads((PACKAGE_DIR / name).read_text("utf-8")) for name in FILES}
    if mutate:
        mutate(data)
    for name, value in data.items():
        (tmp_path / name).write_text(json.dumps(value), "utf-8")
    return CapabilityRegistry.load(MODELS, directory=tmp_path, lifecycle=lifecycle, adapter_check=lambda a: True)


def engine_of(data, engine_id):
    return next(e for e in data["engines_manifest.json"]["engines"] if e["engine_id"] == engine_id)


def with_peaks(data):
    """Give the engines that lack a declared figure one, so declared-peak admission can be exercised."""
    for engine in data["engines_manifest.json"]["engines"]:
        if engine["residency"] != "none" and engine["declared_peak_vram_mib"] is None:
            engine["declared_peak_vram_mib"] = 100
    engine_of(data, "liveportrait_retarget")["declared_peak_vram_mib"] = 2210


@pytest.fixture()
def registry(tmp_path):
    return fixture_registry(tmp_path, with_peaks)


def build_plan(registry, tier, ids, profile="adaptive"):
    request = SelectionRequest("manual", tier)
    picks = [ManualPick(registry.engine(i).category, i) for i in ids]
    plan, problems = Selector(registry)._build_plan(request, profile, picks)
    assert not problems
    return plan


def record(plan, *, score, latency, peak=1000, scenes=(), real=True, veto=True, ident=None):
    refs = tuple(EvidenceEngineRef(engine_id=e.engine_id, entry_digest=e.entry_digest, model_sha256s=e.model_sha256s,
                                   precision=e.precision, provider=e.provider) for e in plan.engines)
    return EvidenceRecord(evidence_id=ident or "ev_" + plan.digest[:12], tier=plan.tier, plan_digest=plan.digest,
                          engines=refs, measured_with="real_models" if real else "simulated",
                          composite_overall=score, scene_scores=tuple(SceneScore(category=c, score=s) for c, s in scenes),
                          latency_ms_p50=latency, peak_vram_mib=peak, safety_vetoes_passed=veto,
                          dataset_digest="d" * 64, recorded_at="2026-11-04T00:00:00Z")


def decision_of(registry, evidence, mode, tier="12gb", **kwargs):
    selection = Selector(registry, EvidenceStore(evidence)).select(SelectionRequest(mode, tier, **kwargs))
    return selection, build_report(registry, selection)


def chosen_ids(selection, index=0):
    plan = selection.decisions[index].plan
    return None if plan is None else sorted(plan.engine_ids())


def codes(rows):
    return {r["code"] for r in rows}


def rejected(report, engine_id, index=0):
    return next(row for row in report["decisions"][index]["rejected"] if row["engine_id"] == engine_id)


# ------------------------------------------------------------------- the real registry
def test_real_registry_loads_with_every_adapter_present():
    registry = CapabilityRegistry.load()
    assert len(registry.engines) == 21
    assert all(adapter_present(a) for a in ADAPTERS), [a for a in ADAPTERS if not adapter_present(a)]
    assert {e.category for e in registry.engines.values()} == set(CATEGORY_ORDER)
    assert len(set(registry.digests.values())) == 3


def test_real_registry_fails_closed_and_report_is_byte_identical():
    registry = CapabilityRegistry.load()
    reports = []
    for _ in range(2):
        selection = Selector(registry, EvidenceStore()).select(SelectionRequest("quality", "12gb"))
        reports.append(json.dumps(build_report(registry, selection), sort_keys=True))
    assert reports[0] == reports[1]
    report = json.loads(reports[0])
    neural = [e for e in registry.engines.values() if e.models]
    states = {m.model_id: registry.model_states(e, "12gb", 12288)[m.model_id] for e in neural for m in e.models}
    if not any(state == "READY" for state in states.values()):
        assert report["status"] == "INFEASIBLE"
        for engine in neural:
            assert "NOT_GOVERNANCE_READY" in codes(report["engine_eligibility"][engine.engine_id])
    assert all(r["code"] in report["reason_vocabulary"] for row in report["decisions"][0]["rejected"] for r in row["reasons"])


# ------------------------------------------------------------------------- validation
def test_validation_collects_every_issue_and_fails_the_whole_registry(tmp_path):
    def mutate(data):
        engines = data["engines_manifest.json"]["engines"]
        engines.append(copy.deepcopy(engines[0]))                                     # duplicate id
        engine_of(data, "scrfd_detector")["models"][0]["model_id"] = "no_such_model"  # unknown model
        engine_of(data, "pyav_video_reader")["exclusive_with"] = ["scrfd_detector"]   # asymmetric
        engine_of(data, "tps_expression")["engine_id"] = "reference_onnx"             # reserved job-runner id
    with pytest.raises(OrchestratorRegistryError) as caught:
        fixture_registry(tmp_path, mutate)
    found = {i.code for i in caught.value.issues}
    assert {"DUPLICATE_ENGINE_ID", "MODEL_UNKNOWN", "EXCLUSIVE_NOT_SYMMETRIC", "ENGINE_ID_RESERVED"} <= found
    assert caught.value.status == 503 and caught.value.code == "ORCHESTRATOR_REGISTRY_INVALID"


@pytest.mark.parametrize("mutate,expected", [
    (lambda d: engine_of(d, "inswapper_reference").update(precisions=["FP32", "FP16"]), "PRECISION_CONTRADICTS_MODEL"),
    (lambda d: engine_of(d, "inswapper_reference").update(residency="light"), "RESIDENCY_CONTRADICTS_LEGACY"),
    (lambda d: engine_of(d, "scrfd_detector").update(providers=["CUDAExecutionProvider"]), "PROVIDER_CONTRADICTS_MODEL"),
    (lambda d: engine_of(d, "pyav_video_reader").update(
        fallback={"mode": "degrade_to", "target_engine_id": "pyav_video_writer", "reason_codes": ["X"]}), "FALLBACK_TARGET_INVALID"),
    (lambda d: d["render_profiles.json"]["profiles"][0]["required"][0].update(
        allowed_engines=["pyav_video_reader", "ffmpeg_stream_source"]), "BASELINE_NOT_PINNED"),
    (lambda d: d["scene_rules.json"]["trait_map"][0].update(scene_categories=["not_a_category"]), "SCENE_CATEGORY_UNKNOWN"),
    (lambda d: d["render_profiles.json"]["profiles"][1]["optional"][0]["allowed_engines"].append("scrfd_detector"),
     "PROFILE_ENGINE_WRONG_CATEGORY"),
    (lambda d: engine_of(d, "scrfd_detector").update(hardware_tiers=["cpu"], providers=["CPUExecutionProvider"]) or
     engine_of(d, "pyav_video_reader").update(providers=["HardwareCodec"]), "TIER_PROVIDER_MISMATCH"),
])
def test_single_contradictions_invalidate_the_registry(tmp_path, mutate, expected):
    with pytest.raises(OrchestratorRegistryError) as caught:
        fixture_registry(tmp_path, mutate)
    assert expected in {i.code for i in caught.value.issues}


def test_schema_rejects_incoherent_entries_and_invalid_evidence():
    with pytest.raises(ValueError):
        Fallback(mode="degrade_to", target_engine_id=None, reason_codes=["X"])
    with pytest.raises(ValueError):
        Fallback(mode="none", target_engine_id="x", reason_codes=[])
    data = json.loads((PACKAGE_DIR / "engines_manifest.json").read_text("utf-8"))
    bad = copy.deepcopy(data["engines"][2])
    bad["resolution"] = {"min": None, "max": None}
    from roop_3d.orchestrator.schema import EngineEntry
    with pytest.raises(ValueError):
        EngineEntry.model_validate(bad)
    with pytest.raises(ValueError):  # 49-char hash as found in the Stage 11 database
        EvidenceEngineRef(engine_id="scrfd_detector", entry_digest="a" * 64, model_sha256s=("a" * 49,),
                          precision="FP32", provider="CPUExecutionProvider")
    with pytest.raises(ValueError):
        EvidenceFile(records=(record_stub("e"), record_stub("e")))


def record_stub(ident):
    ref = EvidenceEngineRef(engine_id="scrfd_detector", entry_digest="a" * 64, precision="FP32", provider="CPUExecutionProvider")
    return EvidenceRecord(evidence_id=ident, tier="12gb", plan_digest="b" * 64, engines=(ref,), measured_with="real_models",
                          composite_overall=90, latency_ms_p50=5, safety_vetoes_passed=True, dataset_digest="c" * 64,
                          recorded_at="2026-11-04T00:00:00Z")


def test_evidence_file_loading(tmp_path):
    assert EvidenceStore.load(tmp_path / "missing.json").records == ()
    path = tmp_path / "evidence.json"
    path.write_text("{not json", "utf-8")
    with pytest.raises(DomainError) as caught:
        EvidenceStore.load(path)
    assert caught.value.code == "ORCHESTRATOR_EVIDENCE_INVALID"
    path.write_text(EvidenceFile(records=(record_stub("one"),)).model_dump_json(), "utf-8")
    assert [r.evidence_id for r in EvidenceStore.load(path).records] == ["one"]


# ------------------------------------------------------------------------ auto modes
def test_quality_and_balanced_choose_by_evidence_with_explained_rejections(registry):
    base, xseg, stream = (build_plan(registry, "12gb", ids) for ids in (BASE, XSEG, STREAM))
    low = build_plan(registry, "12gb", STREAM + ["stream_restorer_gpen"])
    vetoed = build_plan(registry, "12gb", STREAM + ["stream_restorer_codeformer"])
    fake = build_plan(registry, "12gb", STREAM + ["stream_restorer_ultramax"])
    stale_source = build_plan(registry, "12gb", BASE + ["gpen_reference"])
    stale = record(stale_source, score=99, latency=10, ident="stale")
    stale = EvidenceRecord(**{**stale.model_dump(), "plan_digest": "0" * 64})
    evidence = [record(base, score=90, latency=50), record(xseg, score=95, latency=80), record(stream, score=92, latency=20),
                record(low, score=80, latency=10), record(vetoed, score=99, latency=5, veto=False),
                record(fake, score=99, latency=5, real=False), stale]

    quality, q_report = decision_of(registry, evidence, "quality")
    assert chosen_ids(quality) == sorted(xseg.engine_ids())
    assert quality.decisions[0].basis == "evidence_quality"
    assert rejected(q_report, "bisenet_face_parser")["reasons"][0]["code"] == "DOMINATED_BY_SELECTED"
    assert rejected(q_report, "stream_restorer_codeformer")["reasons"][0]["code"] == "SAFETY_VETO"
    assert rejected(q_report, "stream_restorer_ultramax")["reasons"][0]["code"] == "EVIDENCE_NOT_REAL"
    assert rejected(q_report, "gpen_reference")["reasons"][0]["code"] == "EVIDENCE_STALE"
    assert rejected(q_report, "codeformer_reference")["reasons"][0]["code"] == "NO_EVIDENCE_FOR_PROFILE"

    balanced, b_report = decision_of(registry, evidence, "balanced")
    assert chosen_ids(balanced) == sorted(stream.engine_ids())          # fastest plan at or above the baseline's 90
    assert balanced.decisions[0].basis == "evidence_balanced"
    assert rejected(b_report, "stream_restorer_gpen")["reasons"][0]["code"] == "BELOW_BASELINE"


def test_balanced_needs_baseline_evidence_and_never_invents_a_floor(registry):
    stream = build_plan(registry, "12gb", STREAM)
    selection, report = decision_of(registry, [record(stream, score=99, latency=5)], "balanced")
    decision = selection.decisions[0]
    assert decision.basis == "baseline_unscored" and decision.reason_code == "NO_EVIDENCE_FOR_PROFILE"
    assert sorted(decision.plan.engine_ids()) == sorted(BASE)           # the declared baseline, reported as such
    assert any("floor" in note for note in report["decisions"][0]["notes"])


def test_no_evidence_falls_back_to_the_declared_baseline_only_when_it_is_feasible(registry, tmp_path):
    selection, _ = decision_of(registry, [], "quality")
    assert selection.decisions[0].basis == "baseline_unscored"
    assert selection.decisions[0].admission.peak_source == "declared"
    (tmp_path / "bare").mkdir()
    unmeasured = fixture_registry(tmp_path / "bare")  # the shipped manifest: several peaks are undeclared
    selection, report = decision_of(unmeasured, [], "quality")
    assert selection.status == "INFEASIBLE" and selection.decisions[0].plan is None
    assert "PEAK_VRAM_UNKNOWN" in codes(report["decisions"][0]["problems"])


def test_scene_aware_selection_changes_plans_only_between_scenes(registry):
    base, xseg = build_plan(registry, "12gb", BASE), build_plan(registry, "12gb", XSEG)
    evidence = [record(base, score=88, latency=40, scenes=[("frontal_faces", 90), ("extreme_yaw_and_profile", 85)]),
                record(xseg, score=95, latency=60, scenes=[("frontal_faces", 96), ("extreme_yaw_and_profile", 70)])]
    tracks = [SimpleNamespace(track_id="track_0000", scene_id=0, first_frame=0, last_frame=99,
                              pose_range=SimpleNamespace(yaw_degrees=(-6.0, 5.0), method="five_point_geometric_estimate")),
              SimpleNamespace(track_id="track_0001", scene_id=1, first_frame=100, last_frame=199,
                              pose_range=SimpleNamespace(yaw_degrees=(-42.0, 10.0), method="five_point_geometric_estimate"))]
    segments = segments_from_tracks(tracks, registry.scene_rules)
    assert [s.traits for s in segments] == [("frontal",), ("large_yaw",)]
    selection, report = decision_of(registry, evidence, "quality", segments=segments)
    assert chosen_ids(selection, 0) == sorted(xseg.engine_ids())        # frontal scene: best frontal score
    assert chosen_ids(selection, 1) == sorted(base.engine_ids())        # profile scene: best extreme-yaw score
    assert "candidate estimate" in report["decisions"][1]["scene"]["evidence"][0]
    assert report["decisions"][0]["scene"]["occlusion"] == "not_assessed"


def test_tracks_sharing_a_shot_share_one_plan_and_union_their_traits(registry):
    tracks = [SimpleNamespace(track_id=f"track_000{i}", scene_id=0, first_frame=0, last_frame=50,
                              pose_range=SimpleNamespace(yaw_degrees=yaw, method="five_point_geometric_estimate"))
              for i, yaw in enumerate([(-3.0, 4.0), (-35.0, 2.0)])]
    (segment,) = segments_from_tracks(tracks, registry.scene_rules)
    assert segment.traits == ("frontal", "large_yaw") and len(segment.tracks) == 2
    assert segment.scoring_categories == ("extreme_yaw_and_profile", "frontal_faces")
    unassessed = segments_from_tracks([SimpleNamespace(track_id="track_0009", scene_id=2, first_frame=0, last_frame=1,
                                                       pose_range=SimpleNamespace(yaw_degrees=None, method="eye_line_roll_only"))],
                                      registry.scene_rules)[0]
    assert unassessed.traits == ("pose_unassessed",) and unassessed.scoring_categories == ()


def test_selection_is_deterministic_and_independent_of_evidence_order(registry):
    base, xseg = build_plan(registry, "12gb", BASE), build_plan(registry, "12gb", XSEG)
    records = [record(base, score=90, latency=50), record(xseg, score=95, latency=80)]
    digests = set()
    for evidence in (records, list(reversed(records)), records):
        _, report = decision_of(registry, evidence, "quality")
        digests.add((report["report_digest"], json.dumps(report, sort_keys=True)))
    assert len(digests) == 1
    _, other = decision_of(registry, records[:1], "quality")
    assert other["snapshot"]["snapshot_digest"] != report["snapshot"]["snapshot_digest"]
    assert "recorded_at" not in json.dumps(report) and "timestamp" not in json.dumps(report)  # nothing time-dependent


def test_ties_break_by_a_fixed_order(registry):
    a, b = build_plan(registry, "12gb", BASE), build_plan(registry, "12gb", XSEG)
    first, _ = decision_of(registry, [record(a, score=90, latency=50), record(b, score=90, latency=50)], "quality")
    second, _ = decision_of(registry, [record(b, score=90, latency=50), record(a, score=90, latency=50)], "quality")
    assert chosen_ids(first) == chosen_ids(second)
    assert first.decisions[0].plan.key == min(a.key, b.key)


# ------------------------------------------------------------------------ admission
def test_unknown_or_oversized_measured_peaks_are_refused(registry):
    base = build_plan(registry, "12gb", BASE)
    _, unknown = decision_of(registry, [record(base, score=90, latency=50, peak=None)], "quality")
    assert unknown["decisions"][0]["basis"] == "baseline_unscored"       # the evidence is unusable, never half-trusted
    assert rejected(unknown, "xseg_visible_face")["reasons"]
    _, big = decision_of(registry, [record(base, score=90, latency=50, peak=30000)], "quality")
    assert big["decisions"][0]["basis"] == "baseline_unscored"


def test_vram_admission_follows_the_tier_residency_schedule(registry):
    # heavy: inswapper 1150, liveportrait 2210 (fixture), bisenet 580; light: scrfd 350, arcface 720, landmarks 190
    def run(tier, budget):
        selection, report = decision_of(registry, [], "manual", tier=tier, budget_mib=budget,
                                        manual=tuple(ManualPick(registry.engine(i).category, i) for i in BASE))
        return selection.decisions[0].admission, report
    six, _ = run("6gb", 4600)
    assert six.schedule == "sequential" and six.peak_vram_mib == 2210 + 350 + 720 + 190 and six.admitted
    twelve, _ = run("12gb", 12288)
    assert twelve.schedule == "co_resident" and twelve.peak_vram_mib == 1150 + 2210 + 580 + 350 + 720 + 190
    tight, report = run("12gb", 3000)                                     # same plan, too small a configured budget
    assert not tight.admitted and "EXCEEDS_VRAM_BUDGET" in codes(report["decisions"][0]["problems"])
    cpu, _ = run("cpu", 0)
    assert cpu.schedule == "cpu" and cpu.peak_vram_mib == 0 and cpu.admitted


def test_a_plan_with_an_undeclared_peak_cannot_be_admitted_manually(tmp_path):
    registry = fixture_registry(tmp_path)  # liveportrait_retarget and others declare no peak
    selection, report = decision_of(registry, [], "manual", manual=tuple(ManualPick(registry.engine(i).category, i) for i in BASE))
    assert selection.status == "INFEASIBLE"
    assert "PEAK_VRAM_UNKNOWN" in codes(report["decisions"][0]["problems"])


def test_live_vram_never_changes_the_plan(registry):
    base = build_plan(registry, "12gb", BASE)
    evidence = [record(base, score=90, latency=50)]
    a, _ = decision_of(registry, evidence, "quality")
    b, _ = decision_of(registry, evidence, "quality")  # nothing in the request or snapshot is live
    assert a.snapshot == b.snapshot and "free_vram" not in json.dumps(a.snapshot)


# --------------------------------------------------------------------------- manual
def manual(registry, ids, tier="12gb", **kwargs):
    picks = tuple(ManualPick(registry.engine(i).category, i) for i in ids)
    selection = Selector(registry).select(SelectionRequest("manual", tier, manual=picks, **kwargs))
    return selection, build_report(registry, selection)


def test_manual_accepts_exactly_what_was_picked(registry):
    selection, report = manual(registry, XSEG + ["gpen_reference"])
    assert selection.status == "FEASIBLE" and selection.decisions[0].basis == "manual"
    assert chosen_ids(selection) == sorted(XSEG + ["gpen_reference"])
    assert report["decisions"][0]["selected"]["restoration"]["engine_id"] == "gpen_reference"
    assert report["decisions"][0]["evidence"]["status"] == "none"        # unmeasured is reported, not hidden


def test_manual_reports_every_problem_and_never_substitutes(registry):
    ids = [i for i in STREAM if i != "ffmpeg_stream_writer"] + ["pyav_video_reader", "bisenet_face_parser"]
    selection, report = manual(registry, ids)
    problems = codes(report["decisions"][0]["problems"])
    assert selection.status == "INFEASIBLE" and selection.decisions[0].plan is None
    assert report["decisions"][0]["selected"] is None                    # nothing was swapped in
    assert {"REQUIRED_CATEGORY_MISSING", "NOT_IN_PROFILE"} <= problems   # both are reported, not just the first


def test_manual_rejects_conflicts_contract_breaks_and_forbidden_requests(registry):
    _, report = manual(registry, ["pyav_video_reader", "scrfd_detector", "arcface_w600k_recognizer", "landmarks_106",
                                  "inswapper_batch_stream", "liveportrait_retarget", "bisenet_face_parser",
                                  "pyav_video_writer"])
    assert "EXCLUSIVE_CONFLICT" in codes(report["decisions"][0]["problems"])
    request = SelectionRequest("manual", "12gb", manual=(
        ManualPick("identity_swapping", "inswapper_reference", "TensorrtExecutionProvider", "FP16"),))
    problems = codes(build_report(registry, Selector(registry).select(request))["decisions"][0]["problems"])
    assert {"PROVIDER_UNSUPPORTED", "PRECISION_FORBIDDEN", "REQUIRED_CATEGORY_MISSING"} <= problems


def test_manual_cannot_override_the_governance_gate(tmp_path):
    def blocked(model_id, hardware):
        return {"lifecycle_state": "LICENSE_RESTRICTED" if model_id == "4387a15d1f7fd7d8741d6d9d" else "READY"}
    registry = fixture_registry(tmp_path, with_peaks, lifecycle=blocked)
    selection, report = manual(registry, BASE)
    assert selection.decisions[0].plan is None
    assert "NOT_GOVERNANCE_READY" in codes(report["decisions"][0]["problems"])


def test_manual_without_a_gate_pass_has_no_force_option(registry):
    with pytest.raises(DomainError) as caught:
        Selector(registry).select(SelectionRequest("balanced", "12gb", manual=(ManualPick("video_decoding", "pyav_video_reader"),)))
    assert caught.value.code == "MANUAL_PICKS_NOT_ALLOWED"


def test_baseline_profile_is_a_fixed_plan(registry):
    selection, _ = manual(registry, BASE + ["gpen_reference"], profile_id="baseline")
    assert selection.status == "INFEASIBLE"


def test_resolution_limits_are_enforced(registry):
    selection, report = manual(registry, BASE, media=MediaFacts(width=12000, height=8000))
    assert "RESOLUTION_UNSUPPORTED" in codes(report["decisions"][0]["problems"])
    ok, _ = manual(registry, BASE, media=MediaFacts(width=1920, height=1080))
    assert ok.status == "FEASIBLE"


# ----------------------------------------------------------------------- compilation
def test_compilation_maps_onto_existing_runner_parameters(registry):
    reference = compile_plan(build_plan(registry, "12gb", BASE))
    assert reference.backend == "reference" and reference.expressible
    assert reference.workstation["occlusion_mode"] == "standard"
    robust = compile_plan(build_plan(registry, "12gb", XSEG + ["codeformer_reference"]))
    assert robust.workstation["occlusion_mode"] == "robust" and robust.workstation["restorer_model"] == "codeformer"
    assert robust.settings["face_restoration_policy"] == "codeformer"
    stream = compile_plan(build_plan(registry, "12gb", STREAM + ["stream_restorer_ultramax"]))
    assert stream.backend == "stream" and stream.workstation == {} and stream.settings["face_restoration_policy"] == "ultramax"
    pose = compile_plan(build_plan(registry, "12gb", [e for e in BASE if e != "liveportrait_retarget"] + ["identity_asset_builder"]))
    assert pose.workstation["render_mode"] == "pose_3d" and pose.expressible


def test_plans_the_runner_would_silently_alter_are_unsupported_and_unselectable(registry):
    no_parser = build_plan(registry, "12gb", [e for e in BASE if e != "bisenet_face_parser"])
    no_retarget = build_plan(registry, "12gb", [e for e in BASE if e != "liveportrait_retarget"])
    assert any("FACE_PARSER" in u for u in compile_plan(no_parser).unsupported)
    assert any("RETARGETS_BY_DEFAULT" in u for u in compile_plan(no_retarget).unsupported)
    evaluated = Selector(registry, EvidenceStore([record(no_retarget, score=99, latency=1)]))._evaluate(
        SelectionRequest("quality", "12gb"), no_retarget, 12288)
    assert evaluated.stage == 0 and "RUNNER_CANNOT_EXPRESS" in {c.value for c, _, _ in evaluated.structural}
    selection, _ = decision_of(registry, [record(no_retarget, score=99, latency=1)], "quality")
    assert selection.decisions[0].plan is None or selection.decisions[0].plan.key != no_retarget.key
    _, manual_report = manual(registry, [e for e in BASE if e != "liveportrait_retarget"])
    assert "RUNNER_CANNOT_EXPRESS" in codes(manual_report["decisions"][0]["problems"])   # manual cannot request it either


def test_every_manifest_engine_is_known_to_the_compiler_tables():
    from roop_3d.orchestrator import compile as table
    registry = CapabilityRegistry.load()
    for group in (table.STREAM_ENGINES, table.PROJECTION_ENGINES, table.OCCLUSION_ENGINES, set(table.RESTORER_POLICY)):
        assert set(group) <= set(registry.engines)


# ------------------------------------------------------------------------------- API
def test_dry_run_api_returns_a_report_and_never_starts_work(tmp_path):
    from roop_3d.server.app import create_app
    app = create_app(Settings(port=8030), repo_root=tmp_path, worker_enabled=False)
    headers = {"Origin": "http://127.0.0.1:8030"}
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        listing = client.get("/api/v1/orchestrator/engines?tier=12gb", headers=headers)
        assert listing.status_code == 200 and len(listing.json()["engines"]) == 21
        body = {"mode": "quality", "tier": "12gb", "tracks": [
            {"track_id": "track_0000", "scene_id": 0, "first_frame": 0, "last_frame": 9, "yaw_degrees": [-40, 3]}]}
        first = client.post("/api/v1/orchestrator/plan", json=body, headers=headers)
        second = client.post("/api/v1/orchestrator/plan", json=body, headers=headers)
        assert first.status_code == 200
        assert first.json()["report"]["report_digest"] == second.json()["report"]["report_digest"]
        assert first.json()["report"]["decisions"][0]["scene"]["traits"] == ["large_yaw"]
        assert "Orchestrator report" in first.json()["text"]
        invalid = client.post("/api/v1/orchestrator/plan", json={"mode": "manual", "tier": "99gb"}, headers=headers)
        assert invalid.status_code == 422
        forbidden = client.post("/api/v1/orchestrator/plan", json={"mode": "quality", "tier": "12gb", "manual": [
            {"category": "video_decoding", "engine_id": "pyav_video_reader"}]}, headers=headers)
        assert forbidden.status_code >= 400
