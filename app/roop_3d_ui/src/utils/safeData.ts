import type { FaceSet, Job, JobRequest, Telemetry } from '../contracts';

/**
 * The one place where data from the local service becomes UI state.
 *
 * Every list renderer in the workstation (queue drawer, track lanes, render footer, FaceSet library, hardware readouts) reads
 * `job.request.target_path`, `job.progress.frame`, `faceset.name` and friends directly. A service that is older or newer than
 * the page, a half-written job record, or a proxy error page must therefore never reach them with a missing object: these
 * normalisers drop what is unusable and fill the rest with neutral defaults, so a bad record costs one row, not the page.
 */
const isRecord = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value);
const finite = (value: unknown, fallback: number): number => (typeof value === 'number' && Number.isFinite(value) ? value : fallback);
const text = (value: unknown, fallback = ''): string => (typeof value === 'string' ? value : fallback);

/** The last path segment for either separator (`C:\clips\a.mp4`, `/home/a.mp4`), or `fallback` for an empty / non-string path. */
export function fileName(path: unknown, fallback = ''): string {
  if (typeof path !== 'string' || !path) return fallback;
  return path.split(/[\\/]/).filter(Boolean).at(-1) ?? fallback;
}

/** First `n` characters of an id, or `fallback` when the id is missing. */
export const shortId = (id: unknown, n = 8, fallback = '········'): string => (typeof id === 'string' && id ? id.slice(0, n) : fallback);

export function normalizeJob(raw: unknown): Job | null {
  if (!isRecord(raw) || typeof raw.job_id !== 'string' || !raw.job_id) return null;
  const request = isRecord(raw.request) ? raw.request : {};
  const progress = isRecord(raw.progress) ? raw.progress : {};
  const output = isRecord(raw.output) ? raw.output : null;
  const failure = isRecord(raw.failure) ? raw.failure : null;
  const total = progress.total_frames;
  return {
    ...(raw as unknown as Job),
    job_id: raw.job_id,
    state: (typeof raw.state === 'string' ? raw.state : 'queued') as Job['state'],
    request: {
      ...(request as unknown as JobRequest),
      target_path: text(request.target_path),
      face_mappings: Array.isArray(request.face_mappings) ? request.face_mappings.filter(isRecord) as unknown as JobRequest['face_mappings'] : [],
    },
    progress: {
      ...(progress as unknown as Job['progress']),
      frame: Math.max(0, finite(progress.frame, 0)),
      total_frames: typeof total === 'number' && Number.isFinite(total) && total > 0 ? total : null,
      fps: Math.max(0, finite(progress.fps, 0)),
      stage: text(progress.stage),
    },
    output: output ? { ...(output as unknown as NonNullable<Job['output']>), output_path: text(output.output_path), frames: Math.max(0, finite(output.frames, 0)) } : null,
    failure: failure ? { ...(failure as unknown as NonNullable<Job['failure']>), message: text(failure.message, 'The render failed.') } : null,
  };
}

export function normalizeJobs(raw: unknown): Job[] {
  return Array.isArray(raw) ? raw.map(normalizeJob).filter((job): job is Job => job !== null) : [];
}

export function normalizeFaceSets(raw: unknown): FaceSet[] {
  if (!Array.isArray(raw)) return [];
  return raw.flatMap((item): FaceSet[] => {
    if (!isRecord(item) || typeof item.faceset_id !== 'string' || !item.faceset_id) return [];
    return [{ ...(item as unknown as FaceSet), faceset_id: item.faceset_id, name: text(item.name, item.faceset_id), item_count: Math.max(0, finite(item.item_count, 0)) }];
  });
}

/** Hardware snapshot with a real `gpus` array, whatever the service sent. `null` when there is nothing usable. */
export function normalizeTelemetry(raw: unknown): Telemetry | null {
  if (!isRecord(raw)) return null;
  const gpus = (Array.isArray(raw.gpus) ? raw.gpus : []).filter(isRecord).map(gpu => ({
    ...(gpu as unknown as Telemetry['gpus'][number]),
    name: text(gpu.name, 'GPU'), total_vram_mib: Math.max(0, finite(gpu.total_vram_mib, 0)), free_vram_mib: Math.max(0, finite(gpu.free_vram_mib, 0)),
  }));
  return { ...(raw as unknown as Telemetry), gpus, logical_cpu_count: Math.max(0, finite(raw.logical_cpu_count, 0)) };
}
