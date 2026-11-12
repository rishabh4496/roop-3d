/**
 * The workstation session survives a page reload.
 *
 * Pinokio's Run / Dev switch is a full navigation: the Roop 3D page (an iframe of this app) is unloaded and loaded again.
 * Everything in React state - the target video, its inspection, the face-to-FaceSet mapping, sliders, trim, the selected
 * render job - used to vanish, although the render itself kept running on the server. This module keeps the small,
 * non-sensitive part of that state (ids and settings, never media bytes) in localStorage so App can rebuild it from the
 * backend after the reload. The backend stays the source of truth: the restore re-fetches the probe, inspection and job
 * and drops anything that no longer exists.
 */
export const SESSION_KEY = 'roop3d-last-session';
const VERSION = 1;
const MAX_BYTES = 256 * 1024;                       // a snapshot is ids + sliders; anything bigger is not worth keeping

export interface SessionSnapshot {
  v: number;
  savedAt: number;
  target: string;
  inspectionId: string;
  selectedJob: string;
  projectId: string;
  projectName: string;
  identityLockId: string;
  engineId: string;
  time: number;
  sourceId: string;
  identitySourceB: string;
  selectedTrack: string;
  trimStart: number;
  trimEnd: number | null;
  parameters: Record<string, unknown>;
  exportOptions: unknown;
  curves: Record<string, unknown>;
  assigned: Record<string, string>;
  faceSegments: unknown[];
}

type StorageLike = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>;
const defaultStorage = (): StorageLike | null => { try { return typeof localStorage === 'undefined' ? null : localStorage; } catch { return null; } };
const isRecord = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value);
const text = (value: unknown) => (typeof value === 'string' ? value : '');
const finite = (value: unknown, fallback = 0) => (typeof value === 'number' && Number.isFinite(value) ? value : fallback);

/** Validate untrusted storage content field by field; anything malformed falls back to an empty value. */
export function parseSession(raw: string | null | undefined): SessionSnapshot | null {
  if (!raw || raw.length > MAX_BYTES) return null;
  let value: unknown;
  try { value = JSON.parse(raw); } catch { return null; }
  if (!isRecord(value) || value.v !== VERSION || !text(value.target)) return null;
  const assigned: Record<string, string> = {};
  if (isRecord(value.assigned)) for (const [key, id] of Object.entries(value.assigned)) if (typeof id === 'string' && id) assigned[key] = id;
  return {
    v: VERSION, savedAt: finite(value.savedAt), target: text(value.target), inspectionId: text(value.inspectionId),
    selectedJob: text(value.selectedJob), projectId: text(value.projectId), projectName: text(value.projectName),
    identityLockId: text(value.identityLockId), engineId: text(value.engineId),
    time: Math.max(0, finite(value.time)), sourceId: text(value.sourceId), identitySourceB: text(value.identitySourceB),
    selectedTrack: text(value.selectedTrack), trimStart: Math.max(0, finite(value.trimStart)),
    trimEnd: typeof value.trimEnd === 'number' && Number.isFinite(value.trimEnd) ? Math.max(0, value.trimEnd) : null,
    parameters: isRecord(value.parameters) ? value.parameters : {}, exportOptions: value.exportOptions ?? null,
    curves: isRecord(value.curves) ? value.curves : {}, assigned,
    faceSegments: Array.isArray(value.faceSegments) ? value.faceSegments : [],
  };
}

export function readSession(storage: StorageLike | null = defaultStorage()): SessionSnapshot | null {
  try { return parseSession(storage?.getItem(SESSION_KEY)); } catch { return null; }
}

export function writeSession(snapshot: Omit<SessionSnapshot, 'v' | 'savedAt'>, storage: StorageLike | null = defaultStorage()): boolean {
  try {
    const raw = JSON.stringify({ ...snapshot, v: VERSION, savedAt: Date.now() });
    if (raw.length > MAX_BYTES || !snapshot.target) return false;
    storage?.setItem(SESSION_KEY, raw);
    return !!storage;
  } catch { return false; }           // private mode / quota: the workstation still works, it just cannot survive a reload
}

export function clearSession(storage: StorageLike | null = defaultStorage()): void {
  try { storage?.removeItem(SESSION_KEY); } catch { /* storage is optional */ }
}

/** The job to bring back when there is no saved session: the render that is still going (the user's "process"). */
export function activeJobToResume<T extends { job_id: string; state: string }>(jobs: readonly T[]): T | null {
  const live = new Set(['running', 'preparing', 'pausing', 'queued', 'paused']);
  return jobs.find(job => job.state === 'running') ?? jobs.find(job => live.has(job.state)) ?? null;
}
