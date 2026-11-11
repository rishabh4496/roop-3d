import type { WorkstationParameters } from '../workstation/pipeline';

export type SwapScope = NonNullable<WorkstationParameters['swap_scope']>;

export const DEFAULT_SWAP_SCOPE: SwapScope = 'full_craniofacial';
/** Share of the source skull when the user first picks the adaptive blend (the engine default of 1 would equal the full lock). */
export const ADAPTIVE_START_BLEND = 0.5;

/** The three swap scopes, default first. Labels follow the engine's names (`compositor` / `rendering/silhouette_replacement.py`). */
export const SWAP_SCOPES: readonly { id: SwapScope; label: string; detail: string }[] = [
  { id: 'full_craniofacial', label: '100% Full Craniofacial + Jaw + Forehead + Background Inpaint',
    detail: '100% source bone + jaw + forehead. The target\'s old jaw is filled with background and its hair, ears and neck are re-fitted to the new skull.' },
  { id: 'adaptive_skull', label: 'Adaptive Skull Blend',
    detail: 'Blend the skull between the source (β source) and the target (β target) with the slider below.' },
  { id: 'inner_face', label: 'Inner Face',
    detail: 'Only the inside of the target\'s own face outline is replaced; its jaw, hair and ears stay exactly as they are.' },
];

export const swapScopeOf = (parameters: WorkstationParameters): SwapScope => parameters.swap_scope ?? DEFAULT_SWAP_SCOPE;
/** Share of β_source in the skull, 0..1 (1 = the source's own bone). Only meaningful for the adaptive scope. */
export const skullBlendOf = (parameters: WorkstationParameters): number => parameters.skull_blend ?? 1;

/**
 * Picking the default scope removes both keys, so untouched requests, saved projects and preview cache keys stay byte-identical
 * to earlier ones. `skull_blend` is only ever present together with `adaptive_skull`.
 */
export function withSwapScope(parameters: WorkstationParameters, scope: SwapScope): WorkstationParameters {
  const { swap_scope: _scope, skull_blend: blend, ...rest } = parameters;
  if (scope === DEFAULT_SWAP_SCOPE) return rest;
  if (scope === 'adaptive_skull') return { ...rest, swap_scope: scope, skull_blend: blend ?? ADAPTIVE_START_BLEND };
  return { ...rest, swap_scope: scope };
}

export function withSkullBlend(parameters: WorkstationParameters, blend: number): WorkstationParameters {
  if (parameters.swap_scope !== 'adaptive_skull') return parameters;
  return { ...parameters, skull_blend: Math.min(1, Math.max(0, blend)) };
}

export const swapScopeTouched = (parameters: WorkstationParameters): boolean => parameters.swap_scope !== undefined || parameters.skull_blend !== undefined;

/** The silhouette step belongs to the 3D head render modes; the plain 2D swap never produces a head silhouette to replace. */
export const swapScopeApplies = (parameters: WorkstationParameters): boolean => !!parameters.render_mode && parameters.render_mode !== 'standard';
