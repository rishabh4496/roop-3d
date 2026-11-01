"""Four-strategy combinatorial Batch Matrix engine for roop-3d (#/batch).

Strategies:
  1. One-to-Many: 1 source identity applied across N target videos/images.
  2. Grouped: M source identities mapped to M target groups.
  3. Per-File Matrix: Full Cartesian product (M sources x N targets) with output naming templates.
  4. Recipe Matrix: 1 source x 1 target evaluated across multiple parameter recipes.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import re
from typing import Any
import uuid

from pydantic import Field

from roop_3d.core.settings import Contract
from roop_3d.core.workstation import WorkstationParameters


class BatchStrategy(str, Enum):
    ONE_TO_MANY = "one_to_many"
    GROUPED = "grouped"
    PER_FILE_MATRIX = "per_file_matrix"
    RECIPE_MATRIX = "recipe_matrix"


DEFAULT_OUTPUT_TEMPLATE = "{target}_{source}_{restorer}_{timestamp}.mp4"


BUILTIN_RECIPES: list[dict[str, Any]] = [
    {
        "id": "photoreal_natural",
        "name": "Natural Photoreal",
        "description": "GPEN-512 enhancement (blend 0.35), retarget pose, balanced sharpness and edge blend.",
        "restorer": "gpen",
        "parameters": {
            "pose_mode": "retarget",
            "restoration_blend": 0.35,
            "eye_open_ratio": 1.0,
            "mouth_open_ratio": 1.0,
            "edge_feather_px": 2,
            "erosion_px": 2,
            "swap_weight": 1.0,
        },
    },
    {
        "id": "high_fidelity_ultra",
        "name": "High-Fidelity Ultra",
        "description": "Strong GPEN restoration (0.80), retarget pose, high-resolution edge blending.",
        "restorer": "gpen",
        "parameters": {
            "pose_mode": "retarget",
            "restoration_blend": 0.80,
            "eye_open_ratio": 1.0,
            "mouth_open_ratio": 1.0,
            "edge_feather_px": 3,
            "erosion_px": 3,
            "swap_weight": 1.0,
        },
    },
    {
        "id": "codeformer_fidelity",
        "name": "CodeFormer Fidelity Sweep",
        "description": "CodeFormer scalar fidelity feed (w=0.85) mapped to CodeFormer training template.",
        "restorer": "codeformer",
        "parameters": {
            "pose_mode": "retarget",
            "restoration_blend": 0.70,
            "codeformer_fidelity": 0.85,
            "edge_feather_px": 3,
            "erosion_px": 2,
            "swap_weight": 1.0,
        },
    },
    {
        "id": "ultramax_restoration",
        "name": "UltraMax Deep Dynamic",
        "description": "UltraMax registered model with dynamic preprocessing contract.",
        "restorer": "ultramax",
        "parameters": {
            "pose_mode": "retarget",
            "restoration_blend": 0.65,
            "edge_feather_px": 2,
            "erosion_px": 2,
            "swap_weight": 1.0,
        },
    },
    {
        "id": "mesh_geometry_3d",
        "name": "3D Mesh Blend & Geometry",
        "description": "Full 3D mesh blend pose mode, expressive LivePortrait landmarks, geometric warping.",
        "restorer": "gpen",
        "parameters": {
            "pose_mode": "mesh_blend",
            "restoration_blend": 0.40,
            "eye_open_ratio": 1.05,
            "mouth_open_ratio": 1.05,
            "edge_feather_px": 4,
            "erosion_px": 2,
            "swap_weight": 0.90,
        },
    },
    {
        "id": "frontalization_tps_preset_a",
        "name": "TPS / Frontalization Subtle",
        "description": "Frontalization 0.25, TPS expression 0.35, gentle expression transfer.",
        "restorer": "gpen",
        "parameters": {
            "pose_mode": "retarget",
            "pose_frontalization": 0.25,
            "tps_expression_weight": 0.35,
            "restoration_blend": 0.30,
            "swap_weight": 1.0,
        },
    },
    {
        "id": "frontalization_tps_preset_b",
        "name": "TPS / Frontalization Balanced",
        "description": "Frontalization 0.50, TPS expression 0.65, full yaw/pitch correction.",
        "restorer": "gpen",
        "parameters": {
            "pose_mode": "retarget",
            "pose_frontalization": 0.50,
            "tps_expression_weight": 0.65,
            "restoration_blend": 0.45,
            "swap_weight": 1.0,
        },
    },
    {
        "id": "frontalization_tps_preset_c",
        "name": "TPS / Frontalization Extreme",
        "description": "Frontalization 0.85, TPS expression 1.0, canonical stage 0 frontalization.",
        "restorer": "gpen",
        "parameters": {
            "pose_mode": "retarget",
            "pose_frontalization": 0.85,
            "tps_expression_weight": 1.0,
            "restoration_blend": 0.50,
            "swap_weight": 1.0,
        },
    },
]


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r'[\\/*?:"<>|]', "_", name)
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned.strip("._") or "render"


def format_output_name(
    template: str,
    target: str,
    source: str,
    restorer: str = "auto",
    timestamp: str | None = None,
    recipe: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Evaluate an output naming template string such as '{target}_{source}_{restorer}_{timestamp}.mp4'."""
    target_stem = sanitize_filename(Path(target).stem)
    source_stem = sanitize_filename(Path(source).stem if ("/" in source or "\\" in source) else source)
    ts = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    recipe_str = sanitize_filename(recipe or "default")
    restorer_str = sanitize_filename(restorer or "auto")

    mapping = {
        "target": target_stem,
        "source": source_stem,
        "restorer": restorer_str,
        "timestamp": ts,
        "recipe": recipe_str,
    }
    if extra:
        for k, v in extra.items():
            mapping[k] = sanitize_filename(str(v))

    result = template
    for key, val in mapping.items():
        result = result.replace(f"{{{key}}}", val)

    if not result.endswith(".mp4"):
        result += ".mp4"
    return result


class RecipeConfig(Contract):
    id: str
    name: str
    description: str = ""
    restorer: str = "auto"
    parameters: dict[str, Any] = Field(default_factory=dict)


class TargetDescriptor(Contract):
    target_path: str
    inspection_id: str = "0" * 32
    target_name: str | None = None
    tracks: list[str] = Field(default_factory=list)
    trim_start_seconds: float | None = None
    trim_end_seconds: float | None = None


class JobManifest(Contract):
    manifest_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str
    strategy: BatchStrategy
    source: str
    target_path: str
    inspection_id: str
    face_mappings: list[dict[str, str]]
    output_filename: str
    workstation_parameters: dict[str, Any] = Field(default_factory=dict)
    trim_start_seconds: float | None = None
    trim_end_seconds: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_job_payload(self) -> dict[str, Any]:
        """Convert manifest to a standard roop_3d JobRequest payload."""
        payload: dict[str, Any] = {
            "target_path": self.target_path,
            "inspection_id": self.inspection_id,
            "face_mappings": self.face_mappings,
            "processing": "local",
            "remote_consent": False,
        }
        if self.trim_start_seconds is not None:
            payload["trim_start_seconds"] = self.trim_start_seconds
        if self.trim_end_seconds is not None:
            payload["trim_end_seconds"] = self.trim_end_seconds
        if self.workstation_parameters:
            payload["workstation_parameters"] = self.workstation_parameters
        return payload


def _resolve_target(item: str | dict[str, Any] | TargetDescriptor, default_inspection: str) -> tuple[str, str, list[str], float | None, float | None]:
    if isinstance(item, str):
        return item, default_inspection, ["track_0000"], None, None
    if isinstance(item, TargetDescriptor):
        return (
            item.target_path,
            item.inspection_id or default_inspection,
            item.tracks or ["track_0000"],
            item.trim_start_seconds,
            item.trim_end_seconds,
        )
    if isinstance(item, dict):
        return (
            item.get("target_path", item.get("path", "")),
            item.get("inspection_id", default_inspection),
            item.get("tracks", ["track_0000"]),
            item.get("trim_start_seconds"),
            item.get("trim_end_seconds"),
        )
    raise TypeError(f"Unsupported target descriptor: {type(item)}")


class BatchMatrixEngine:
    """Combinatorial batch generator for all 4 strategies."""

    @classmethod
    def generate_one_to_many(
        cls,
        source: str,
        targets: list[str | dict[str, Any] | TargetDescriptor],
        output_template: str = "{target}_{source}_{timestamp}.mp4",
        default_inspection_id: str = "0" * 32,
        timestamp: str | None = None,
        workstation_parameters: dict[str, Any] | None = None,
    ) -> list[JobManifest]:
        """Strategy 1: 1 source identity applied across N target videos/images."""
        if not source:
            raise ValueError("One-to-Many strategy requires a valid source identity")
        if not targets:
            raise ValueError("One-to-Many strategy requires at least one target")

        manifests: list[JobManifest] = []
        ts = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        for idx, t_item in enumerate(targets):
            t_path, insp_id, tracks, trim_s, trim_e = _resolve_target(t_item, default_inspection_id)
            mappings = [{"track_id": trk, "faceset_id": source} for trk in tracks]
            out_name = format_output_name(
                output_template,
                target=t_path,
                source=source,
                timestamp=ts,
                extra={"index": idx + 1},
            )
            manifest = JobManifest(
                name=f"1-to-N: {Path(t_path).name} <- {Path(source).name if '/' in source else source}",
                strategy=BatchStrategy.ONE_TO_MANY,
                source=source,
                target_path=t_path,
                inspection_id=insp_id,
                face_mappings=mappings,
                output_filename=out_name,
                workstation_parameters=workstation_parameters or {},
                trim_start_seconds=trim_s,
                trim_end_seconds=trim_e,
                metadata={"batch_index": idx, "total_in_batch": len(targets)},
            )
            manifests.append(manifest)

        return manifests

    @classmethod
    def generate_grouped(
        cls,
        groups: list[dict[str, Any]],
        output_template: str = "{target}_{source}_{timestamp}.mp4",
        default_inspection_id: str = "0" * 32,
        timestamp: str | None = None,
        workstation_parameters: dict[str, Any] | None = None,
    ) -> list[JobManifest]:
        """Strategy 2: M source identities mapped to M target groups."""
        if not groups:
            raise ValueError("Grouped strategy requires at least one group")

        manifests: list[JobManifest] = []
        ts = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        for g_idx, group in enumerate(groups):
            src = group.get("source") or group.get("facesetId")
            if not src:
                continue
            g_targets = group.get("targets") or group.get("targetIds") or []
            g_name = group.get("name") or f"Group {g_idx + 1}"

            for t_idx, t_item in enumerate(g_targets):
                t_path, insp_id, tracks, trim_s, trim_e = _resolve_target(t_item, default_inspection_id)
                mappings = [{"track_id": trk, "faceset_id": src} for trk in tracks]
                out_name = format_output_name(
                    output_template,
                    target=t_path,
                    source=src,
                    timestamp=ts,
                    extra={"group": sanitize_filename(g_name), "index": t_idx + 1},
                )
                manifest = JobManifest(
                    name=f"[{g_name}] {Path(t_path).name} <- {src}",
                    strategy=BatchStrategy.GROUPED,
                    source=src,
                    target_path=t_path,
                    inspection_id=insp_id,
                    face_mappings=mappings,
                    output_filename=out_name,
                    workstation_parameters=workstation_parameters or {},
                    trim_start_seconds=trim_s,
                    trim_end_seconds=trim_e,
                    metadata={"group_name": g_name, "group_index": g_idx, "target_index": t_idx},
                )
                manifests.append(manifest)

        return manifests

    @classmethod
    def generate_per_file_matrix(
        cls,
        sources: list[str],
        targets: list[str | dict[str, Any] | TargetDescriptor],
        restorer: str = "gpen",
        output_template: str = DEFAULT_OUTPUT_TEMPLATE,
        active_cells: list[str] | list[tuple[int, int]] | None = None,
        default_inspection_id: str = "0" * 32,
        timestamp: str | None = None,
        workstation_parameters: dict[str, Any] | None = None,
    ) -> list[JobManifest]:
        """Strategy 3: Full Cartesian product (M sources x N targets) with naming templates."""
        if not sources:
            raise ValueError("Per-File Matrix strategy requires at least one source")
        if not targets:
            raise ValueError("Per-File Matrix strategy requires at least one target")

        manifests: list[JobManifest] = []
        ts = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Parse active cells filter if present
        cell_filter: set[str] | None = None
        if active_cells is not None:
            cell_filter = set()
            for c in active_cells:
                if isinstance(c, tuple):
                    cell_filter.add(f"{c[0]}:{c[1]}")
                else:
                    cell_filter.add(str(c))

        for t_idx, t_item in enumerate(targets):
            t_path, insp_id, tracks, trim_s, trim_e = _resolve_target(t_item, default_inspection_id)
            target_key = Path(t_path).stem

            for s_idx, src in enumerate(sources):
                # Check active selection
                if cell_filter is not None:
                    # Allow matching by indices or by identifiers
                    match_idx = f"{t_idx}:{s_idx}"
                    match_id = f"{target_key}:{src}"
                    match_path = f"{t_path}:{src}"
                    if not (match_idx in cell_filter or match_id in cell_filter or match_path in cell_filter):
                        continue

                mappings = [{"track_id": trk, "faceset_id": src} for trk in tracks]
                out_name = format_output_name(
                    output_template,
                    target=t_path,
                    source=src,
                    restorer=restorer,
                    timestamp=ts,
                    extra={"t_idx": t_idx, "s_idx": s_idx},
                )
                manifest = JobManifest(
                    name=f"Matrix [{t_idx},{s_idx}]: {Path(t_path).name} x {src}",
                    strategy=BatchStrategy.PER_FILE_MATRIX,
                    source=src,
                    target_path=t_path,
                    inspection_id=insp_id,
                    face_mappings=mappings,
                    output_filename=out_name,
                    workstation_parameters=workstation_parameters or {},
                    trim_start_seconds=trim_s,
                    trim_end_seconds=trim_e,
                    metadata={"target_index": t_idx, "source_index": s_idx, "restorer": restorer},
                )
                manifests.append(manifest)

        return manifests

    @classmethod
    def generate_recipe_matrix(
        cls,
        source: str,
        target: str | dict[str, Any] | TargetDescriptor,
        recipes: list[dict[str, Any] | RecipeConfig],
        output_template: str = "{target}_{source}_{recipe}_{timestamp}.mp4",
        default_inspection_id: str = "0" * 32,
        timestamp: str | None = None,
        base_parameters: dict[str, Any] | None = None,
    ) -> list[JobManifest]:
        """Strategy 4: 1 source x 1 target rendered across multiple parameter recipes."""
        if not source:
            raise ValueError("Recipe Matrix strategy requires a source identity")
        if not recipes:
            raise ValueError("Recipe Matrix strategy requires at least one parameter recipe")

        manifests: list[JobManifest] = []
        ts = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        t_path, insp_id, tracks, trim_s, trim_e = _resolve_target(target, default_inspection_id)
        mappings = [{"track_id": trk, "faceset_id": source} for trk in tracks]

        for r_idx, r_item in enumerate(recipes):
            if isinstance(r_item, RecipeConfig):
                r_id = r_item.id
                r_name = r_item.name
                r_params = r_item.parameters
                r_restorer = r_item.restorer
            elif isinstance(r_item, dict):
                r_id = r_item.get("id", f"recipe_{r_idx + 1}")
                r_name = r_item.get("name", r_id)
                r_params = r_item.get("parameters", {})
                r_restorer = r_item.get("restorer", "auto")
            else:
                continue

            merged_params = dict(base_parameters or {})
            merged_params.update(r_params)

            out_name = format_output_name(
                output_template,
                target=t_path,
                source=source,
                restorer=r_restorer,
                timestamp=ts,
                recipe=r_id,
            )

            manifest = JobManifest(
                name=f"Recipe [{r_name}]: {Path(t_path).name} <- {source}",
                strategy=BatchStrategy.RECIPE_MATRIX,
                source=source,
                target_path=t_path,
                inspection_id=insp_id,
                face_mappings=mappings,
                output_filename=out_name,
                workstation_parameters=merged_params,
                trim_start_seconds=trim_s,
                trim_end_seconds=trim_e,
                metadata={"recipe_id": r_id, "recipe_name": r_name, "recipe_index": r_idx},
            )
            manifests.append(manifest)

        return manifests

    @classmethod
    def expand_batch(cls, spec: dict[str, Any]) -> list[JobManifest]:
        """Dispatch dynamically based on 'strategy' parameter in spec."""
        strat = spec.get("strategy")
        if not strat:
            raise ValueError("Missing 'strategy' in batch specification")

        strat_str = str(strat).lower().replace("-", "_")

        if strat_str in ("one_to_many", "one-to-many"):
            return cls.generate_one_to_many(
                source=spec["source"],
                targets=spec["targets"],
                output_template=spec.get("output_template", "{target}_{source}_{timestamp}.mp4"),
                default_inspection_id=spec.get("default_inspection_id", "0" * 32),
                timestamp=spec.get("timestamp"),
                workstation_parameters=spec.get("workstation_parameters"),
            )
        elif strat_str in ("grouped",):
            return cls.generate_grouped(
                groups=spec["groups"],
                output_template=spec.get("output_template", "{target}_{source}_{timestamp}.mp4"),
                default_inspection_id=spec.get("default_inspection_id", "0" * 32),
                timestamp=spec.get("timestamp"),
                workstation_parameters=spec.get("workstation_parameters"),
            )
        elif strat_str in ("per_file_matrix", "per-file-matrix", "matrix"):
            return cls.generate_per_file_matrix(
                sources=spec["sources"],
                targets=spec["targets"],
                restorer=spec.get("restorer", "gpen"),
                output_template=spec.get("output_template", DEFAULT_OUTPUT_TEMPLATE),
                active_cells=spec.get("active_cells"),
                default_inspection_id=spec.get("default_inspection_id", "0" * 32),
                timestamp=spec.get("timestamp"),
                workstation_parameters=spec.get("workstation_parameters"),
            )
        elif strat_str in ("recipe_matrix", "recipe-matrix", "recipe"):
            return cls.generate_recipe_matrix(
                source=spec["source"],
                target=spec["target"],
                recipes=spec.get("recipes", BUILTIN_RECIPES),
                output_template=spec.get("output_template", "{target}_{source}_{recipe}_{timestamp}.mp4"),
                default_inspection_id=spec.get("default_inspection_id", "0" * 32),
                timestamp=spec.get("timestamp"),
                base_parameters=spec.get("workstation_parameters"),
            )
        else:
            raise ValueError(f"Unknown batch strategy '{strat}'")
