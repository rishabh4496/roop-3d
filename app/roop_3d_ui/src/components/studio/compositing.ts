import { bandedBlend, type BlendMode, type WorkstationParameters } from '../workstation/pipeline';

type Params = Partial<WorkstationParameters>;

/** Engine defaults of the compositor fusion controls (= `CompositeOptions`); the request carries a value only once it is touched. */
export const DEFAULT_BLEND_MODE: BlendMode = 'multi_band';
export const XSEG_STRENGTH_DEFAULT = 1;
export const PYRAMID_LEVELS_DEFAULT = 4;
/** Bands every renderer honours: the stream core clamps the compositor's band count to 4..6, so the wire contract offers only that. */
export const PYRAMID_LEVEL_RANGE: readonly [number, number] = [4, 6];
/** Engine id of the FFmpeg photoreal runner; it fuses with its own Laplacian pyramid and ignores the compositor's blend mode. */
export const PHOTOREAL_ENGINE = 'photoreal_3d_ffmpeg';

/** The four fusions of `compositor/boundary_fusion.py`, as the compositor really runs them. */
export const BLEND_MODES: readonly { id: BlendMode; label: string; hint: string }[] = [
  { id: 'multi_band', label: 'Multi-band pyramid',
    hint: 'Laplacian-pyramid fusion (engine default, four bands): coarse bands cross-fade over a wide area and fine bands over a narrow one, so the seam disappears without ghosting.' },
  { id: 'feather', label: 'Feather (alpha mix)',
    hint: 'A plain alpha mix through the feathered matte. The simplest fusion: any tone or detail mismatch shows at the seam.' },
  { id: 'poisson', label: 'Poisson (seamless clone)',
    hint: 'Gradient-domain seamless cloning (OpenCV): the swap takes on the surrounding tone. Every frame is solved on its own.' },
  { id: 'frequency_separated', label: 'Frequency-separated',
    hint: 'Keeps the target\'s own two finest bands (pores, stubble, hair) inside the mask and takes the swap\'s coarser structure. Colour transfer is done earlier, on skin and nose only.' },
];

export const blendModeOf = (parameters: Params): BlendMode => parameters.blend_mode ?? DEFAULT_BLEND_MODE;
export const xsegStrengthOf = (parameters: Params): number => parameters.xseg_occlusion_strength ?? XSEG_STRENGTH_DEFAULT;
export const pyramidLevelsOf = (parameters: Params): number => parameters.pyramid_levels ?? PYRAMID_LEVELS_DEFAULT;

/** Why the XSeg vote has no effect right now, or null when it acts. The standard compositor runs the BiSeNet parser only. */
export const xsegInactiveReason = (parameters: Params): string | null =>
  parameters.occlusion_mode === 'robust' ? null : 'The XSeg and face-occluder networks only run in Robust occlusion mode: switch on XSeg mask + Smart Occluder.';

/** Why the band count has no effect right now, or null when it acts. */
export function pyramidInactiveReason(parameters: Params): string | null {
  if (bandedBlend(parameters.blend_mode)) return null;
  const label = BLEND_MODES.find(item => item.id === parameters.blend_mode)?.label ?? 'This';
  return `${label} fusion has no pyramid: choose Multi-band or Frequency-separated.`;
}

/** Frequency-separated blends encoded values; Robust occlusion blends in linear light, and the engine rejects the pair. */
export const frequencyBlockedReason = (parameters: Params): string | null =>
  parameters.occlusion_mode === 'robust' ? 'Not available with Robust occlusion: that path blends in linear light, this fusion works on encoded values.' : null;

/** Changing the occlusion mode may not leave an invalid pair behind: Robust returns Frequency-separated fusion to the default. */
export const occlusionModePatch = (current: Params, mode: 'standard' | 'robust'): Params =>
  mode === 'robust' && current.blend_mode === 'frequency_separated' ? { occlusion_mode: mode, blend_mode: DEFAULT_BLEND_MODE } : { occlusion_mode: mode };
