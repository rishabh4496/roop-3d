import json, glob, subprocess
V = r"G:\pinokio\api\roop-3d\app\roop_3d\outputs\test 1\phase1_verify"
eng = json.load(open(V + r"\engines_result.json"))
det = {}
for f in sorted(glob.glob(V + r"\detect_s*.json")):
    s = json.load(open(f))["summary"]
    det[s["clip"]] = s
swarm_files = ["app/roop_3d/media/service.py", "app/roop_3d/engine/detection/face_tracker.py", "app/roop_3d/media/face_analysis.py",
               "app/roop_3d/model_registry.py", "app/roop_3d/server/contracts.py", "app/roop_3d/server/app.py", "app/roop_3d/server/routes.py",
               "app/roop_3d/core/ort_session.py", "app/roop_3d/engine/runtime.py", "app/roop_3d/engine/volumetric/swapper_engine.py"]
new_files = ["app/roop_3d/engine/detection/robust_detector.py", "app/roop_3d/tests/test_robust_detection.py",
             "app/roop_3d/tests/test_swap_engine_readiness.py",
             "app/roop_3d/models/engine_readiness.py (gitignored by models/ rule: needs git add -f)"]
pre = subprocess.run(["git", "diff", "HEAD", "--name-only"], capture_output=True, text=True, cwd=r"G:\pinokio\api\roop-3d").stdout.split()
pre_existing = [p for p in pre if p not in swarm_files]
active = [{"model_id": r["model_id"], "provider": r["provider"], "init_seconds": r["init_seconds"],
           "warm_infer_ms_random_input": r["infer_ms_warm"], "weight": r["path"], "sha256": r["sha256"],
           "native_resolution": r["native_resolution"]} for r in eng["init"] if r["init"] == "OK"]
unavail = [{"model_id": r["model_id"],
            "reason": "no weight file anywhere on disk (registry holds a no-SHA/no-source candidate only)"}
           for r in eng["init"] if r["init"] != "OK"]
out = {
    "generated_by": "Sub-Agent 4 (verify-only)", "base_commit": "7fd1e2a", "date": "2026-11-14",
    "acceptance": {
        "engines_100pct_initialize": {
            "met": False, "ready": len(active), "registered": len(eng["init"]),
            "note": "5/14 registered swap engines initialize and execute on TensorRT. 9/14 cannot: weights do not exist, so 100% is impossible without acquiring files."},
        "detection_gt_99pct": {
            "met": False,
            "note": "Frames-with-detection over 3432 evaluated frames = 92.07% (analyzer path), 91.67% (tracker path). 6/12 clips are 100%; s1 98.54; s5-s10 77-89%. Remaining misses inspected are overwhelmingly back-of-head / hand-occluded (no face). That is NOT recall; no ground truth exists. False positives on back-of-head ears (s7, s9, s10) inflate the hit rate, so true recall of VISIBLE faces is unknown; plausibly >99% but not demonstrated."},
    },
    "modified_files": {
        "swarm_modified_this_phase (mtime >= 15:13 on 2026-11-14; some also contain earlier uncommitted edits)": swarm_files,
        "swarm_new_files": new_files,
        "pre_existing_uncommitted (mtime 02:45-11:33, earlier session)": pre_existing,
        "caveat": "mtime-based attribution only; routes.py/app.py/ort_session.py/runtime.py/swapper_engine.py may also carry older hunks."},
    "active_swap_engines": active, "unavailable_swap_engines": unavail,
    "engine_notes": [
        "inswapper_128_fp16.onnx is byte-identical (sha e4a3f08c...) to inswapper_128.onnx: not actually FP16; io dtypes float32; two ids, one network.",
        "hififace_unofficial_256 + crossface_hififace weights exist (roop-ultimate/app/models, roop-3d/models/cache) but have no supported_models.json spec, so they are not a registered/selectable engine.",
        "/api/v1/engines/readiness `session` is null until an engine was opened in-process; after opening it reports TensorRT for all 5.",
        "/api/v1/health returns swap_engines.state=SCANNING with empty lists on the first call after process start (background hash scan); READY once it completes."],
    "faceset": {
        "id": "797577eca6ab4cf88c7b3e075b590979",
        "loaded_via": "GET /api/v1/facesets + POST /api/v1/facesets/{id}/identity-asset (FaceSetReader)", "built": True,
        "build_seconds": 1.3, "build_key": "0c99016c...", "layout": "5pt", "reconstruction_level": "limited_multiview",
        "fit_rms_mm": 0.80, "holdout_rms_mm": 1.21, "mesh_http": 200, "glb_bytes": 482076,
        "limits": "FLAME blocked; 5-point landmarks only; profile views 0 and 4 got view gain ~0.64 (heavy lighting correction)"},
    "preview_swap": {
        "s11_f30": "OK on inswapper_128 (59 s cold) and hyperswap_1a_256 (5.5 s) via POST /api/v1/workstation/preview, CUDA provider; face visibly changed (preview_s11_compare.jpg)",
        "s3_f60": "422 PREVIEW_TRACK_MISSING: head lying sideways, estimated yaw -80 deg; face is tracked and swap_allowed but session.py auto-3D path hits `fade <= 0 -> continue` and returns zero replacements with no `withheld` entry; error text misleadingly says no mapped face is visible. First attempt also hit PREVIEW_BUDGET (120 s) on cold models."},
    "detection": {
        "method": "Per sampled frame: media.analyzer.analyze (RobustDetector-wrapped detector, UI/target-lock path, thresholds 0.5/0.3) and sequential FaceTracker(identity-lock policy) as in MediaService.inspect. All frames when clip <=400 frames, else stride 2-4 (s5 /3, s6 /4, s7 /2, s8 /2).",
        "per_video": det, "totals": {"frames": 3432, "analyzer_hits": 3160, "tracker_hits": 3146, "either": 3182},
        "miss_classification": {
            "s1": "3 misses = hand covering the face (genuine occlusion)",
            "s5/s6/s7/s9/s10": "contiguous back-of-head / ear-only turntable spans (genuine no-face). Borderline possibly-real: s6 f364,368 (profile outline; tracker had it, analyzer did not), s10 f295/296 (raw 0.43), s7 f124-130 (cheek), s5 f312-324 (glasses sliver; tracker coasted)."},
        "false_positive_evidence": "sheets/lowscore_hits_precision.png: score 0.32-0.50 boxes sit on the EAR/back of head in s9 (4/4 sampled), s7 (3/4), s10 (2/4). Low-score-only hit frames: s1 7, s2 16, s5 11, s6 6, s7 11, s9 9, s10 24. If all s7/s9/s10 low-score hits were false: s7 73%, s9 84%, s10 77%.",
        "artifacts": V + r"\sheets"},
    "tests": {
        "command": "pytest test_swap_engine_readiness test_robust_detection test_local_weight_trust test_model_registry test_swapper_catalog_and_adapter test_model_selection_regressions test_stage12_target_lock test_e2e_video_pipeline test_occlusion_compositor test_photorealistic_swap_core",
        "result": "159 passed, 0 failed, 2 warnings, 487 s (one fixture setup 448 s = cold TensorRT build). Full suite NOT run.",
        "log": r"G:\pinokio\api\roop-3d\.swarm\logs\pytest_targeted.log"},
    "known_open_issues": [
        "9 swappers have no weights (simswap_256/512, ghost_256_unet_1-3, blendswap_256, uniface_256, realswap_256/512).",
        "inswapper_128_fp16 is not FP16 (duplicate bytes).",
        "Back-of-head false positives (ear boxes at score 0.3-0.5) in s7/s9/s10 pass FaceAnalyzer's tolerant tier and would be swapped onto the ear.",
        "Preview returns PREVIEW_TRACK_MISSING (misleading) for tracked faces beyond the 3D yaw limit when pose fade is 0; no withheld reason recorded.",
        "Cold preview exceeds the 120 s PREVIEW_BUDGET (first TensorRT builds).",
        "face_occluder.onnx TensorRT/CUDA failure (audit item) not re-verified here.",
        "ONNX Runtime 'Conv running in Fallback mode' warnings during detection slow 4K scans (~0.5-1 s/frame).",
        "Tracker was run on strided frames for s5-s8 so its coasting behaviour there is not representative.",
        "Preview refuses frames > 3840x2160 pixels, so 4096x2160 clips (s4, s5, s6) get PREVIEW_SIZE_BUDGET."],
    "runtime_state_touched": [
        "roop_3d/.runtime/identity_assets/7975.../0c99016c... (rebuilt)",
        "roop_3d/.runtime/media/phase1_verify_s3.mp4 and phase1_verify_s11.mp4 (copies)",
        "roop_3d/.runtime/inspections/* (new inspection records)", "outputs/test 1/phase1_verify/*"],
    "phase2_recommendations": [
        "Acquire weights (or drop the 9 ids from the UI) rather than claim 100%; wire hififace/crossface via a supported_models.json spec since weights exist.",
        "Replace/alias inswapper_128_fp16 with a real FP16 export or hide the duplicate.",
        "Add back-of-head/ear rejection to the tolerant tier (eyes inside box, 106pt agreement >= 0.8) and re-measure.",
        "Hand-label ~200 frames (s5-s10 transitions) for a real recall/precision number; stop using frames-with-detection as the KPI.",
        "Return a withheld/skip reason when pose fade removes a swap; fix PREVIEW_TRACK_MISSING wording.",
        "Pre-warm TensorRT engines at startup so previews stay under 120 s.",
        "git add -f models/engine_readiness.py before committing."],
}
json.dump(out, open(r"G:\pinokio\api\roop-3d\.swarm\phase1_handoff.json", "w"), indent=1)
print("ok")
