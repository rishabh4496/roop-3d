"""The four per-frame stages of the reference render, shared by the job runner and the benchmark.

    Stage 1  track      detection, landmarks, 6DoF pose, track association           (tracker only)
    Stage 2  swap       identity swap, LivePortrait retarget, 3D mesh / pose blend   (swapper / portrait / rasteriser)
    Stage 3  composite  restoration, semantic masks, LAB colour match, Laplacian blend (compositor)
    Stage 4  write      checkpoint the finished frame (PNG sequence; the muxer reads it back)

Each stage owns disjoint mutable session state, which is what makes it legal to run them on separate threads with a
different frame in each: tracking state (``track_lock``), swap state (``lock``), compositor state (``composite_lock``).
Anything one stage needs from another travels in ``Item.meta`` -- never read back from the live session, which is
already working on a different frame.

A session without ``track_frame`` (a stub, an older engine) is run in the original single-call way: Stage 1 only
records timestamps and ``infer_frame`` tracks inline.
"""
from __future__ import annotations

from typing import Any, Callable

from roop_3d.async_pipeline.stages import Item
from roop_3d.compositor.pipeline import CompositeOptions

STAGE_NAMES = ("track", "swap", "composite", "write")


class ReferenceStages:
    def __init__(self, session, *, request, edit_plan=None, export=None, identity_report=None,
                 lip_sync: bool = False, working_transfer: Any = None, resume_after: int = -1,
                 write_frame: Callable[[int, Any, tuple], None] | None = None,
                 save_passes: Callable[[Any, Any, int], None] | None = None,
                 scene_cuts: list | None = None, swapped_ranges: list | None = None):
        self.session, self.request, self.edit_plan, self.export = session, request, edit_plan, export
        self.identity_report, self.lip_sync, self.working_transfer = identity_report, lip_sync, working_transfer
        self.resume_after, self.write_frame, self.save_passes = resume_after, write_frame, save_passes
        self.scene_cuts = scene_cuts if scene_cuts is not None else []
        self.swapped_ranges = swapped_ranges if swapped_ranges is not None else []
        self.staged = hasattr(session, "track_frame")
        self.expected_shape: tuple | None = None

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
        options = parameters.inference_options() if parameters else {}
        item.meta.update(parameters=parameters, options=options)
        if hasattr(session, "set_frame_timestamp"):
            if self.staged:   # the composite stage owns the compositor's clock (see ReferencePipeline.composite_frame)
                session.set_frame_timestamp(float(decoded.timestamp.seconds), compositor=False)
            else:
                session.set_frame_timestamp(float(decoded.timestamp.seconds))
        if hasattr(session, "set_angle_frame"):
            session.set_angle_frame(decoded.index)
        if not self.staged:
            return None
        return session.track_frame(decoded.image, pose_blend=options.get("pose_blend"),
                                   render_mode=options.get("render_mode"))

    # -- Stage 2 ----------------------------------------------------------------------------------------------
    def swap(self, item: Item):
        session = self.session
        decoded, options, tracked = item.meta["decoded"], item.meta["options"], item.payload
        if self.edit_plan:
            self.edit_plan.apply(session, decoded.index)
        result = session.infer_frame(decoded.image, **options, **({"tracked": tracked} if tracked is not None else {}))
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
        return result

    # -- Stage 3 ----------------------------------------------------------------------------------------------
    def composite(self, item: Item):
        session, edit_plan = self.session, self.edit_plan
        decoded, result, parameters = item.meta["decoded"], item.meta["result"], item.meta["parameters"]
        collector = None
        if self.export and self.export.passes:
            from roop_3d.media.vfx import PassCollector
            collector = PassCollector(decoded.image.shape[:2], self.export.passes)
        explicit = ({"frame_index": item.meta["frame_index"], "timestamp_seconds": float(decoded.timestamp.seconds)}
                    if self.staged else {})
        output = session.composite_frame(
            decoded.image, result,
            options=(parameters.with_source_transfer(self._transfer()).composite_options() if parameters
                     else CompositeOptions(restoration_blend=(0.35 if session.compositor.restorer else 0.0))),
            occluder_masks=(edit_plan.occluders(decoded.image, result)
                            if edit_plan and edit_plan.mask is not None else None),
            lip_sync_enabled=self.lip_sync,
            **({"observer": collector.observe} if collector else {}), **explicit)
        if collector is not None and decoded.index > self.resume_after and self.save_passes is not None:
            self.save_passes(collector, result, decoded.index)
        # Read in the stage that produced it: the compositor is already working on a later frame by now.
        item.meta["timing"] = dict(getattr(getattr(session, "compositor", None), "last_timing", None) or {})
        return output

    # -- Stage 4 ----------------------------------------------------------------------------------------------
    def write(self, item: Item):
        decoded = item.meta["decoded"]
        if decoded.index > self.resume_after:
            if self.write_frame is not None:
                self.write_frame(decoded.index, item.payload, self.expected_shape)
            item.meta["saved"] = True
        return item.payload
