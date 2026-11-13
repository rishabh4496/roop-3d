"""Bounded workstation parameters shared by previews and durable render requests."""
from typing import Literal
from pydantic import Field, model_serializer, model_validator
from roop_3d.core.settings import Contract
from roop_3d.core.face_editor import FaceExpressionOverrides

#: Hair / neck / ear protection fields: absent from a serialisation (and so from fingerprints) until the caller sets them.
PROTECTION_FIELDS = ('neck_protection', 'hair_matting', 'preserve_ears_glasses', 'jaw_feather_px', 'skin_texture_retention')
#: Stage 6B swap-scope fields: same opt-in rule, absent from a serialisation (and so from fingerprints) until the caller sets them.
SCOPE_FIELDS = ('swap_scope', 'skull_blend')
#: Compositor fusion / occlusion fields: same opt-in rule. ``xseg_occlusion_strength`` only acts in ``occlusion_mode='robust'`` (the
#: visible-face networks run there); ``blend_mode`` / ``pyramid_levels`` are ``CompositeOptions.mode`` / ``.pyramid_levels``.
FUSION_FIELDS = ('xseg_occlusion_strength', 'blend_mode', 'pyramid_levels')
BlendModeName = Literal['feather', 'multi_band', 'poisson', 'frequency_separated']
#: Pyramid bands every renderer honours: the stream core clamps ``CompositeOptions.pyramid_levels`` to 4..6.
PYRAMID_LEVELS = (4, 6)
FREQUENCY_BLEND_CONFLICT = ("blend_mode 'frequency_separated' blends on encoded values and cannot be combined with linear-light "
                            "blending (occlusion_mode 'robust' and linear_light=true both blend in linear light)")

class WorkstationParameters(Contract):
    expression_restore_factor: float = Field(default=0, ge=0, le=1)
    face_expression_overrides: FaceExpressionOverrides = Field(default_factory=dict, max_length=64)
    color_method: Literal['lab', 'reinhard_lab', 'histogram', 'ycbcr', 'oklab', 'oklab_wasserstein'] = 'lab'
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
    # Directional strand matte for hair strands / bangs over the face (robust mode only; computed at frame resolution from the
    # parser's hair class, so it needs a trusted parser). Opt-in on the wire: omitted from a request until set.
    strand_matte: bool = False
    # Fit the canonical proxy head's jaw width to the target's face contour in the mesh pose-blend path (pose_mode='mesh_blend').
    # Opt-in on the wire; near-frontal faces only, otherwise it reports why it did nothing.
    jaw_adaptation: bool = False
    # Confidence the semantic face parser (BiSeNet) must reach before its mask is trusted over the landmark hull. 0.55 is the
    # compositor's long-standing constant, so leaving this unset changes nothing; it is opt-in on the wire.
    occlusion_threshold: float = Field(default=0.55, ge=0, le=1)
    # Scale of the learned visible-face (XSeg / face-occluder) "not face" vote that is subtracted from the matte in
    # occlusion_mode='robust'. 0 switches that subtraction off, 1 is the unscaled behaviour; the BiSeNet parser and manual
    # occluders always act at full strength. The standard mode runs no visible-face network, so there it has no effect.
    xseg_occlusion_strength: float = Field(default=1.0, ge=0, le=1)
    # How the swapped face is fused into the frame. ``multi_band`` (a four-band Laplacian pyramid) is the compositor default;
    # ``pyramid_levels`` is the band count of ``multi_band`` / ``frequency_separated`` and is ignored by ``feather`` / ``poisson``.
    blend_mode: BlendModeName = 'multi_band'
    pyramid_levels: int = Field(default=4, ge=PYRAMID_LEVELS[0], le=PYRAMID_LEVELS[1])
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
    # Hair / neck / ear protection and skin micro-texture (compositor/pipeline.CompositeOptions). Opt-in on the wire like the two above:
    # the defaults are the compositor's previous hard-coded behaviour, and unset fields never reach a fingerprint or cache key.
    neck_protection: bool = True
    hair_matting: bool = True
    preserve_ears_glasses: bool = True
    jaw_feather_px: float = Field(default=6.0, ge=2, le=24)
    skin_texture_retention: float = Field(default=0.25, ge=0, le=1)
    # Stage 6B swap scope. full_craniofacial = 100% source skull/jaw/forehead with the old silhouette removed (hair, ears and neck
    # warped onto the new perimeter, overhang filled with background); adaptive_skull = skull_blend is the share of beta_source in the
    # skull (the rest is beta_target); inner_face = legacy, only the inside of the target's own face outline is replaced.
    swap_scope: Literal['full_craniofacial', 'adaptive_skull', 'inner_face'] = 'full_craniofacial'
    skull_blend: float = Field(default=1.0, ge=0, le=1)
    boundary_ribbon_matting: bool = False
    submental_contact_shadow: bool = False
    contact_shadow_opacity: float = Field(default=0.12, ge=0, le=0.15)
    contact_shadow_falloff_px: float = Field(default=10.0, ge=6, le=14)
    sss_shadow_warmth: bool = False
    sss_warmth_amount: float = Field(default=0.5, ge=0, le=1)
    sensor_shutter_angle: float = Field(default=180, ge=0, le=360)
    grain_match_strength: float = Field(default=100, ge=0, le=100)

    @model_serializer(mode='wrap')
    def compatible_serialization(self, handler):
        value = handler(self)
        for field in ('expression_restore_factor', 'face_expression_overrides'):
            if field not in self.model_fields_set:
                value.pop(field, None)
        for field in ('color_method', 'restorer_model', 'identity_blend_t', 'fidelity_weight', 'swap_weight', 'erosion_px', 'pose_mode', 'pose_frontalization', 'tps_expression_weight', 'render_mode', 'occlusion_mode', 'strand_matte', 'jaw_adaptation', 'linear_light', 'blend_transfer', 'ultramax_small_face_mode', 'ultramax_min_native_pixels', 'ambient_fill', 'hairline_offset_px', 'occlusion_threshold', 'sensor_shutter_angle', 'grain_match_strength') + PROTECTION_FIELDS + SCOPE_FIELDS + FUSION_FIELDS:
            if field not in self.model_fields_set:
                value.pop(field, None)
        return value

    @model_validator(mode='after')
    def fusion_modes_agree(self):
        if self.blend_mode == 'frequency_separated' and (self.occlusion_mode == 'robust' or self.linear_light):
            raise ValueError(FREQUENCY_BLEND_CONFLICT)
        return self

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
        if self.jaw_adaptation:
            options['jaw_adaptation'] = True
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
        from roop_3d.compositor.boundary_fusion import BlendMode
        from roop_3d.compositor.pipeline import CompositeOptions, RobustOptions
        # Unresolved 'auto' falls back to sRGB, the pre-Stage-15 curve; callers with media context resolve it first.
        blend = 'srgb' if self.blend_transfer == 'auto' else self.blend_transfer
        return CompositeOptions(color_method=self.color_method, fidelity_weight=self.fidelity_weight, restoration_blend=0 if 'restorer' in self.bypass else self.restoration_blend,
                                expression_restore_factor=self.expression_restore_factor,
                                face_expression_overrides=self.face_expression_overrides,
                                feather_px=self.feather_px,
                                mode=BlendMode(self.blend_mode), pyramid_levels=self.pyramid_levels,
                                xseg_occlusion_strength=self.xseg_occlusion_strength,
                                swap_weight=self.swap_weight, erosion_px=self.erosion_px,
                                occlusion_mode=self.occlusion_mode,
                                robust=RobustOptions(strand_matte=self.strand_matte),
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
                                neck_protection=self.neck_protection,
                                hair_matting=self.hair_matting,
                                preserve_ears_glasses=self.preserve_ears_glasses,
                                jaw_feather_px=self.jaw_feather_px,
                                skin_texture_retention=self.skin_texture_retention,
                                swap_scope=self.swap_scope, skull_blend=self.skull_blend,
                                boundary_ribbon_matting=self.boundary_ribbon_matting,
                                submental_contact_shadow=self.submental_contact_shadow,
                                contact_shadow_opacity=self.contact_shadow_opacity,
                                contact_shadow_falloff_px=self.contact_shadow_falloff_px,
                                sss_shadow_warmth=self.sss_shadow_warmth,
                                sss_warmth_amount=self.sss_warmth_amount,
                                ultramax_small_face_mode=self.ultramax_small_face_mode,
                                ultramax_min_native_pixels=self.ultramax_min_native_pixels,
                                linear_light=self.linear_light, blend_transfer=blend,
                                sensor_shutter_angle=self.sensor_shutter_angle,
                                sensor_grain_strength=self.grain_match_strength)
