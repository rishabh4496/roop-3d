import type { Capabilities, Mapping, Track, FaceSet } from './contracts';
export function featureReason(caps: Capabilities | null, name: string): string | null {
  const feature = caps?.workstation.features[name];
  return feature?.enabled ? null : feature?.reason ?? 'Waiting for an available service capability';
}
export function optionReason(caps: Capabilities | null, field: string, value?: string): string | null {
  if (!caps) return 'Capabilities are not loaded';
  const fixed = caps.workstation.read_only_parameters[field] ?? caps.workstation.disabled_parameters[field];
  if (fixed) return fixed;
  const options = caps.workstation.controls[field];
  return value !== undefined && options ? (Object.hasOwn(options, value) ? options[value] : 'Not advertised by the backend') : null;
}
export function jobActions(caps: Capabilities | null, state: string): string[] {
  return Object.entries(caps?.workstation.job_controls ?? {}).filter(([, states]) => states.includes(state)).map(([action]) => action);
}
export function mappings(values: Record<string, string>, tracks: Track[], sources: FaceSet[]): Mapping[] {
  const trackIds = new Set(tracks.map(track => track.track_id));
  const sourceIds = new Set(sources.map(source => source.faceset_id));
  return Object.entries(values).filter(([track, source]) => trackIds.has(track) && sourceIds.has(source)).map(([track_id, faceset_id]) => ({ track_id, faceset_id }));
}
export class PreviewURL {
  private value: string | null = null;
  constructor(private create = URL.createObjectURL, private revoke = URL.revokeObjectURL) {}
  replace(blob: Blob): string { this.clear(); this.value = this.create(blob); return this.value; }
  clear(): void { if (this.value) this.revoke(this.value); this.value = null; }
}
