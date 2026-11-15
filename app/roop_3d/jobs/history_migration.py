"""Read-only schema conversion of historical job records, never render configuration.

The caller validates the whole result and backs up the original bytes before publishing.
Only explicitly retired fields are removed; arbitrary corruption must still fail validation.
"""
from __future__ import annotations

import json

RETIRED_WORKSTATION = frozenset({"restoration_blend", "pose_strength", "pose_mode", "swap_weight",
                                "render_mode", "restorer_model"})
RETIRED_SETTINGS = frozenset({"execution_mode", "swapper_model_id", "dfm_morph_factor", "face_restoration_policy",
    "lip_sync_mode", "lip_sync_audio_path", "lip_sync_audio_stream", "restorer_threads", "restorer_pool_size",
    "enhancer_blend", "restorer_fidelity", "ultramax_native_bypass", "ultramax_min_pixels"})
RETIRED_TIMINGS = frozenset({"inswapper", "restorer", "pose_tps"})
TERMINAL = frozenset({"succeeded", "failed", "cancelled"})


def migrate_history(raw: str) -> dict | None:
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("schema_version") != 1 or not isinstance(data.get("jobs"), dict):
        return None
    changed = False
    for record in data["jobs"].values():
        if not isinstance(record, dict):
            continue
        retired = False
        for values, keys in ((record.get("request", {}).get("workstation_parameters"), RETIRED_WORKSTATION),
                             (record.get("effective_settings"), RETIRED_SETTINGS)):
            if isinstance(values, dict):
                for key in keys & values.keys():
                    values.pop(key)
                    retired = changed = True
        timings = record.get("progress", {}).get("stage_timings_ms")
        if isinstance(timings, dict):
            for key in RETIRED_TIMINGS & timings.keys():
                timings.pop(key)
                changed = True
        if retired and record.get("state") not in TERMINAL:
            record["state"] = "failed"
            record["failure"] = {"code": "LEGACY_JOB_RETIRED", "message":
                "This historical job used removed render options. Create a new true-3D job; the original record is backed up.",
                "recoverable": False}
            record["revision"] = record.get("revision", 0) + 1
    return data if changed else None
