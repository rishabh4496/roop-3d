"""Bounded workstation parameters shared by previews and durable render requests."""
from typing import Literal
from pydantic import Field, model_serializer
from roop_3d.core.settings import Contract
from roop_3d.core.face_editor import FaceExpressionOverrides


class WorkstationParameters(Contract):
    expression_restore_factor: float = Field(default=0, ge=0, le=1)
    face_expression_overrides: FaceExpressionOverrides = Field(default_factory=dict, max_length=64)
    color_method: Literal['lab', 'reinhard_lab', 'histogram'] = 'lab'
    restorer_model: Literal['auto', 'gpen', 'codeformer', 'gfpgan', 'ultramax'] = 'auto'
    identity_blend_t: float | None = Field(default=None, ge=0, le=1)
    fidelity_weight: float = Field(default=.5, ge=0, le=1)
    restoration_blend: float = Field(default=0, ge=0, le=1)
    feather_px: float = Field(default=5, ge=0, le=64)
    color_strength: float = Field(default=0.65, ge=0, le=1)
    pose_strength: float = Field(default=0.35, ge=0, le=1)
    pose_mode: Literal['retarget', 'mesh_blend'] = 'retarget'
    swap_weight: float = Field(default=1, ge=0, le=1)
    erosion_px: float = Field(default=0, ge=-32, le=32)
    bypass: tuple[Literal['restorer', 'color'], ...] = Field(default=(), max_length=2)
    # Stage 15: blend in linear light. 'auto' follows the source's actual transfer (see composite_options).
    linear_light: bool = False
    blend_transfer: Literal['auto', 'srgb', 'bt1886', 'gamma22', 'gamma28', 'bt709_oetf', 'linear'] = 'auto'
    # Pre-swap 3D pose frontalization (reduces profile mismatch before inswapper_128) and TPS expression warping.
    pose_frontalization: float = Field(default=0, ge=0, le=1)
    tps_expression_weight: float = Field(default=0, ge=0, le=1)
    # Stage 4 render mode: 'standard' = existing identity swap; 'pose_3d' = swap projected on the SOURCE identity's own
    # geometry; 'reenactment_3d' = source identity geometry + target expression. Every mode falls back deterministically
    # to the proven path when its geometry is not reliable.
    render_mode: Literal['standard', 'pose_3d', 'reenactment_3d'] = 'standard'
    # Stage 5 compositor: 'standard' = the original face/occluder masks; 'robust' = unified face + occlusion analysis,
    # flow-aware temporal matte, depth-ordered multi-face compositing and frame-space matte/texture refinement.
    # Missing models are reported through the compositor capability, never simulated.
    occlusion_mode: Literal['standard', 'robust'] = 'standard'
    # Confidence the semantic face parser (BiSeNet) must reach before its mask is trusted over the landmark hull. 0.55 is the
    # compositor's long-standing constant, so leaving this unset changes nothing; it is opt-in on the wire.
    occlusion_threshold: float = Field(default=0.55, ge=0, le=1)
    # Stage 17: 3D Spherical Harmonics relighting, specular skin sheen, and cast shadow preservation.
    relight: bool = False
    relight_strength: float = Field(default=0.8, ge=0, le=1)
    specular_sheen: float = Field(default=0.25, ge=0, le=1)
    shadow_preservation: float = Field(default=0.85, ge=0, le=1)
    ultramax_small_face_mode: Literal['bypass', 'adaptive', 'off'] = 'off'
    ultramax_min_native_pixels: int = Field(default=320, ge=64, le=1024)
    lighting_mode: Literal['auto', 'manual'] = 'auto'
    manual_light_yaw: float = Field(default=35.0, ge=-180.0, le=180.0)
    manual_light_pitch: float = Field(default=25.0, ge=-90.0, le=90.0)
    # Ambient fill of the manual virtual key light, and a forehead / hairline matte offset. Both are opt-in: omitted from a request until set,
    # so earlier requests keep their exact serialisation and idempotency fingerprint.
    ambient_fill: float = Field(default=0.3, ge=0, le=1)
    hairline_offset_px: float = Field(default=0, ge=-32, le=32)

    @model_serializer(mode='wrap')
    def compatible_serialization(self, handler):
        value = handler(self)
        for field in ('expression_restore_factor', 'face_expression_overrides'):
            if field not in self.model_fields_set:
                value.pop(field, None)
        for field in ('color_method', 'restorer_model', 'identity_blend_t', 'fidelity_weight', 'swap_weight', 'erosion_px', 'pose_mode', 'pose_frontalization', 'tps_expression_weight', 'render_mode', 'occlusion_mode', 'linear_light', 'blend_transfer', 'ultramax_small_face_mode', 'ultramax_min_native_pixels', 'ambient_fill', 'hairline_offset_px', 'occlusion_threshold'):
            if field not in self.model_fields_set:
                value.pop(field, None)
        return value

    def inference_options(self) -> dict[str, float]:
        # The new curve changes geometric projection, never LivePortrait motion.
        options = {'pose_blend': self.pose_strength} if self.pose_mode == 'mesh_blend' else {'strength': self.pose_strength}
        # Only passed when active, so sessions that predate these controls keep working unchanged.
        if self.pose_frontalization > 0:
            options['frontalization'] = self.pose_frontalization
        if self.tps_expression_weight > 0:
            options['tps_expression'] = self.tps_expression_weight
        if self.identity_blend_t is not None:
            options['identity_blend_t'] = self.identity_blend_t
        if self.render_mode != 'standard':
            options['render_mode'] = self.render_mode
        return options

    def session_options(self) -> dict[str, str]:
        """Extra `prepare_session` arguments; empty for the standard mode so older sessions keep working unchanged."""
        return {'render_mode': self.render_mode} if self.render_mode != 'standard' else {}

    def uses_mesh_projection(self) -> bool:
        return self.pose_mode == 'mesh_blend' or self.render_mode == 'pose_3d'

    def with_source_transfer(self, transfer_code: int) -> 'WorkstationParameters':
        """Resolve ``blend_transfer='auto'`` to the display transfer matching the media's (working) transfer code."""
        from roop_3d.color.transfer import blend_transfer_for
        if self.blend_transfer != 'auto':
            return self
        return self.model_copy(update={'blend_transfer': blend_transfer_for(transfer_code)})

    def composite_options(self):
        from roop_3d.compositor.pipeline import CompositeOptions
        # Unresolved 'auto' falls back to sRGB, the pre-Stage-15 curve; callers with media context resolve it first.
        blend = 'srgb' if self.blend_transfer == 'auto' else self.blend_transfer
        return CompositeOptions(color_method=self.color_method, fidelity_weight=self.fidelity_weight, restoration_blend=0 if 'restorer' in self.bypass else self.restoration_blend,
                                expression_restore_factor=self.expression_restore_factor,
                                face_expression_overrides=self.face_expression_overrides,
                                feather_px=self.feather_px,
                                swap_weight=self.swap_weight, erosion_px=self.erosion_px,
                                occlusion_mode=self.occlusion_mode,
                                semantic_confidence_threshold=self.occlusion_threshold,
                                color_strength=0 if 'color' in self.bypass else self.color_strength,
                                relight=self.relight,
                                relight_strength=self.relight_strength,
                                specular_sheen=self.specular_sheen,
                                shadow_preservation=self.shadow_preservation,
                                lighting_mode=self.lighting_mode,
                                manual_light_yaw=self.manual_light_yaw,
                                manual_light_pitch=self.manual_light_pitch,
                                ambient_fill=self.ambient_fill,
                                hairline_offset_px=self.hairline_offset_px,
                                ultramax_small_face_mode=self.ultramax_small_face_mode,
                                ultramax_min_native_pixels=self.ultramax_min_native_pixels,
                                linear_light=self.linear_light, blend_transfer=blend)
