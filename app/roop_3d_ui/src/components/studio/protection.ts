import type { WorkstationParameters } from '../workstation/pipeline';

/** Engine defaults of the hair / neck / ear controls (= `CompositeOptions`). A control set back to its default is removed from the parameters,
 *  so untouched requests, saved projects and preview cache keys stay byte-identical to earlier ones. */
export const PROTECTION_DEFAULTS = {
  neck_protection: true, hair_matting: true, preserve_ears_glasses: true, jaw_feather_px: 6, skin_texture_retention: 0.25,
} as const;
export type ProtectionKey = keyof typeof PROTECTION_DEFAULTS;
export const PROTECTION_KEYS = Object.keys(PROTECTION_DEFAULTS) as ProtectionKey[];

export const JAW_FEATHER_RANGE_PX: readonly [number, number] = [2, 24];
/** The inset slider spans ±20 px (the engine accepts ±32 for `hairline_offset_px`). */
export const HAIRLINE_INSET_RANGE_PX = 20;

export const protectionValue = <K extends ProtectionKey>(parameters: WorkstationParameters, key: K): (typeof PROTECTION_DEFAULTS)[K] =>
  (parameters[key] ?? PROTECTION_DEFAULTS[key]) as (typeof PROTECTION_DEFAULTS)[K];

export function withProtection(parameters: WorkstationParameters, key: ProtectionKey, value: boolean | number): WorkstationParameters {
  const { [key]: _old, ...rest } = parameters;
  return value === PROTECTION_DEFAULTS[key] ? rest : { ...rest, [key]: value };
}

/** True once any of the five controls (or the hairline inset) differs from the engine default. */
export const protectionTouched = (parameters: WorkstationParameters) => PROTECTION_KEYS.some(key => key in parameters) || !!parameters.hairline_offset_px;

export function resetProtection(parameters: WorkstationParameters): WorkstationParameters {
  const rest: Record<string, unknown> = { ...parameters };
  for (const key of PROTECTION_KEYS) delete rest[key];
  delete rest.hairline_offset_px;
  return rest as WorkstationParameters;
}

/**
 * Forehead hairline inset in px: + pulls the swapped region DOWN, away from the hair; − lets it grow up over the forehead.
 * It is the engine's `hairline_offset_px` with the sign flipped (that one is + = grow upward).
 */
export const hairlineInset = (parameters: WorkstationParameters): number => (parameters.hairline_offset_px ? -parameters.hairline_offset_px : 0);

export function withHairlineInset(parameters: WorkstationParameters, inset: number): WorkstationParameters {
  const { hairline_offset_px: _old, ...rest } = parameters;
  return inset === 0 ? rest : { ...rest, hairline_offset_px: -inset };           // omitted at 0 so untouched requests stay byte-identical
}

/** The jaw clamp, the hairline matting and the jaw feather only run on a 3D-projected matte (3D render modes) or the robust compositor. */
export const matteProtectionApplies = (parameters: WorkstationParameters): boolean =>
  (!!parameters.render_mode && parameters.render_mode !== 'standard') || parameters.occlusion_mode === 'robust';
