"""The four per-frame stages of the reference render, shared by the job runner and the benchmark.

    Stage 1  track      detection, landmarks, 6DoF pose, track association           (tracker only)
    Stage 2  swap       full-head render per track: MakeHuman fit, UV texture, perspective Z-buffer + SH lighting (`lock`)
    Stage 3  composite  BiSeNet + XSeg + occluder masks, manual keep-target masks, skin-only LAB match, Laplacian blend
    Stage 4  write      checkpoint the finished frame (PNG sequence; the muxer reads it back)

Each stage owns disjoint mutable session state, which is what makes it legal to run them on separate threads with a
different frame in each: tracking state (``track_lock``), swap state (``lock``), compositor state (``composite_lock``).
Anything one stage needs from another travels in ``Item.meta`` -- never read back from the live session, which is
already working on a different frame.

A session without ``track_frame`` (a stub, an older engine) is run in the original single-call way: Stage 1 only
records timestamps and ``infer_frame`` tracks inline.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from roop_3d.async_pipeline.stages import Item
from roop_3d.core.errors import DomainError

STAGE_NAMES = ("track", "swap", "composite", "write")
logger = logging.getLogger(__name__)


class ReferenceStages:
    def __init__(self, session, *, request, edit_plan=None, export=None, identity_report=None,
                 working_transfer: Any = None, resume_after: int = -1,
                 write_frame: Callable[[int, Any, tuple], None] | None = None, watchdog=None,
                 scene_cuts: list | None = None, swapped_ranges: list | None = None):
        self.session, self.request, self.edit_plan, self.export = session, request, edit_plan, export
        self.identity_report, self.working_transfer = identity_report, working_transfer
        self.watchdog = watchdog
        self.resume_after, self.write_frame = resume_after, write_frame
        self.scene_cuts = scene_cuts if scene_cuts is not None else []
        self.swapped_ranges = swapped_ranges if swapped_ranges is not None else []
        self.staged = hasattr(session, "track_frame")
        self.expected_shape: tuple | None = None
        if self.staged and hasattr(session, "begin_fit_ahead"):
            session.begin_fit_ahead()

    def _transfer(self):
        """``working_transfer`` may be a callable: the reader only knows its working colour after the first frame."""
        value = self.working_transfer
        return value() if callable(value) else value

    def stages(self) -> tuple[tuple[str, Callable[[Item], Any]], ...]:
        return (("track", self.track), ("swap", self.swap), ("composite", self.composite), ("write", self.write))

    # -- Stage 1 ----------------------------------------------------------------------------------------------
    def track(self, item: Item):
        session, request = self.session, self.request
        decoded = item.payload
        self.expected_shape = decoded.image.shape
        item.meta["decoded"] = decoded
        parameters = (request.parameters_at(decoded.index)
                      if request.workstation_parameters or request.parameter_curves else None)
        item.meta.update(parameters=parameters)
        if hasattr(session, "set_frame_timestamp"):
            session.set_frame_timestamp(float(decoded.timestamp.seconds))
        if hasattr(session, "set_angle_frame"):
            session.set_angle_frame(decoded.index)
        if not self.staged:
            return None
        return session.track_frame(decoded.image)

    # -- Stage 2 ----------------------------------------------------------------------------------------------
    def swap(self, item: Item):
        session = self.session
        decoded, tracked = item.meta["decoded"], item.payload
        if self.edit_plan:
            self.edit_plan.apply(session, decoded.index)
        result = session.infer_frame(decoded.image, **({"tracked": tracked} if tracked is not None else {}))
        if self.staged:
            item.meta["frame_index"] = max(0, session.frames_processed - 1)   # only this stage advances the counter
        if self.identity_report is not None:
            self.identity_report.observe(decoded.index, result)
        if self.export:
            if getattr(result, "scene_cut", False):
                self.scene_cuts.append(decoded.index)
            if result.replacements:
                ranges = self.swapped_ranges
                if ranges and ranges[-1][1] == decoded.index:
                    ranges[-1][1] += 1
                else:
                    ranges.append([decoded.index, decoded.index + 1])
        item.meta["result"] = result

    def composite(self, item: Item):
        session, edit_plan = self.session, self.edit_plan
        decoded, result, parameters = item.meta["decoded"], item.meta["result"], item.meta["parameters"]
        if self.export and self.export.passes:
            raise DomainError("TRUE3D_RENDER_PASSES_UNAVAILABLE",
                              "Render-pass export (alpha, occluder, uv, normal, depth, relit) is not implemented "
                              "for the volumetric MakeHuman renderer", 503)
        if edit_plan is not None and getattr(edit_plan, "has_masks", False):
            # Manual keep-target masks are frame-space planes: they join the compositor as each replacement's occluder.
            result = edit_plan.with_occluders(decoded.image, result, decoded.index)
        output = session.composite_frame(
            decoded.image, result, options=parameters,
            **({"frame_index": item.meta["frame_index"]} if self.staged else {}),
            timestamp_seconds=float(decoded.timestamp.seconds))
        report = getattr(session, "last_frame_report", None)
        item.meta["timing"] = dict(report.stage_ms) if report is not None else {}
        return output

    # -- Stage 4 ----------------------------------------------------------------------------------------------
    def write(self, item: Item):
        decoded = item.meta["decoded"]
        if decoded.index > self.resume_after:
            if self.write_frame is not None:
                self.write_frame(decoded.index, item.payload, self.expected_shape)
            item.meta["saved"] = True
        return item.payload
