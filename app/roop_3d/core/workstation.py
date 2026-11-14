"""Bounded workstation parameters shared by previews and durable render requests."""
from typing import Literal
from pydantic import Field, model_serializer, model_validator
from roop_3d.core.settings import Contract

#: Hair / neck / ear protection fields: absent from a serialisation (and so from fingerprints) until the caller sets them.
PROTECTION_FIELDS = ('neck_protection', 'hair_matting', 'preserve_ears_glasses', 'jaw_feather_px', 'skin_texture_retention')
#: Fusion and occlusion fields are opt-in for request fingerprints; XSeg applies only in robust mode.
#: Blend mode and pyramid depth configure the full-frame true-3D raster blend.
FUSION_FIELDS = ('xseg_occlusion_strength', 'blend_mode', 'pyramid_levels')
BlendModeName = Literal['feather', 'multi_band', 'poisson', 'frequency_separated']
#: Pyramid bands used by the full-frame true-3D raster blend.
PYRAMID_LEVELS = (4, 6)
FREQUENCY_BLEND_CONFLICT = ("blend_mode 'frequency_separated' blends on encoded values and cannot be combined with linear-light "
                            "blending (occlusion_mode 'robust' and linear_light=true both blend in linear light)")

class WorkstationParameters(Contract):
    color_method: Literal['lab', 'reinhard_lab', 'histogram', 'ycbcr', 'oklab', 'oklab_wasserstein'] = 'lab'
    feather_px: float = Field(default=5, ge=0, le=64)
    color_strength: float = Field(default=0.65, ge=0, le=1)
    erosion_px: float = Field(default=0, ge=-32, le=32)
    bypass: tuple[Literal['color'], ...] = Field(default=(), max_length=1)
    # Stage 15: blend in linear light. 'auto' follows the source's actual transfer (see composite_options).
    linear_light: bool = False
    blend_transfer: Literal['auto', 'srgb', 'bt1886', 'gamma22', 'gamma28', 'bt709_oetf', 'linear'] = 'auto'
    # True-3D full-head render options (see ``roop_3d.server.true3d``). None = the renderer's defaults. There is no 2D / 2.5D mode.
    head_topology_engine: Literal['FLAME-FullHead-3D', 'MakeHuman-CC0-Craniofacial-3D'] | None = None
    shader_mode: Literal['3D-PBR-Subsurface-Scattering + Neural-Deferred-UV', '3D-Gaussian-Head-Avatar'] | None = None
    camera_intrinsics: dict | None = None
    # Stage 5 compositor: 'standard' = the original face/occluder masks; 'robust' = unified face + occlusion analysis,
    # flow-aware temporal matte, depth-ordered multi-face compositing and frame-space matte/texture refinement.
    # Missing models are reported through the compositor capability, never simulated.
    occlusion_mode: Literal['standard', 'robust'] = 'standard'
    # Directional strand matte for hair strands / bangs over the face (robust mode only; computed at frame resolution from the
    # parser's hair class, so it needs a trusted parser). Opt-in on the wire: omitted from a request until set.
    strand_matte: bool = False
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
        for field in ('color_method', 'erosion_px', 'head_topology_engine', 'shader_mode', 'camera_intrinsics', 'occlusion_mode', 'strand_matte', 'linear_light', 'blend_transfer', 'ambient_fill', 'hairline_offset_px', 'occlusion_threshold', 'sensor_shutter_angle', 'grain_match_strength') + PROTECTION_FIELDS + FUSION_FIELDS:
            if field not in self.model_fields_set:
                value.pop(field, None)
        return value


    @model_validator(mode='after')
    def fusion_modes_agree(self):
        if self.blend_mode == 'frequency_separated' and (self.occlusion_mode == 'robust' or self.linear_light):
            raise ValueError(FREQUENCY_BLEND_CONFLICT)
        return self

    def with_source_transfer(self, transfer_code: int) -> 'WorkstationParameters':
        """Resolve ``blend_transfer='auto'`` to the display transfer matching the media's (working) transfer code."""
        from roop_3d.color.transfer import blend_transfer_for
        if self.blend_transfer != 'auto':
            return self
        return self.model_copy(update={'blend_transfer': blend_transfer_for(transfer_code)})
