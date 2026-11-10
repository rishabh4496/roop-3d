import type { Capabilities } from './contracts';
/**
 * An error from the local service. It is still a plain `Error` (same `message` as before), but keeps the machine
 * code from the `{error: {code, message}}` envelope so the UI can show a specific recovery instruction for it.
 */
export class ApiError extends Error {
  constructor(message: string, readonly code: string | null = null, readonly status = 0, readonly requestId: string | null = null) {
    super(message);
    this.name = 'ApiError';
  }
}
const apiError = (body: any, status: number) => new ApiError(body?.error?.message ?? `Local service returned ${status}`, body?.error?.code ?? null, status, body?.error?.request_id ?? null);
export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const normalizedPath = path.startsWith('/api/v1') ? path : `/api/v1${path}`;
  const response = await fetch(normalizedPath, { ...init, headers: { ...(typeof init.body === 'string' ? { 'Content-Type': 'application/json' } : {}), ...init.headers } });
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw apiError(error, response.status);
  }
  return response.json() as Promise<T>;
}
export const post = <T,>(path: string, value: unknown, init: RequestInit = {}) => api<T>(path, { method: 'POST', body: JSON.stringify(value), ...init });
/** HTTP 503 `{busy: true, stage, retry_after_ms}`: the GPU is owned by a job / compile / model re-bind. Not a failure. */
export class PreviewBusyError extends Error {
  constructor(readonly stage: string, readonly retryAfterMs: number) {
    super(`Engine busy (${stage})`);
    this.name = 'PreviewBusyError';
  }
}

export type LiveFace = { track_id: number; box: number[]; landmarks5: number[][] | null; yaw: number; pitch: number; roll: number };
export type LivePreviewMeta = {
  media_id: string; frame_index: number; swap_config_hash: string; solo_node: string; width: number; height: number;
  faces: LiveFace[]; applied: string[]; unsupported: string[]; notes: string[]; latency: Record<string, number>;
};
export type LivePreviewResult = { blob: Blob; meta: LivePreviewMeta | null; cached: boolean; cancelled?: boolean };

export async function postLivePreview(payload: unknown, init: RequestInit = {}): Promise<LivePreviewResult> {
  const response = await fetch('/api/preview', { method: 'POST', body: JSON.stringify(payload), cache: 'no-store', ...init, headers: { 'Content-Type': 'application/json', ...init.headers } });
  if (response.status === 204 || response.headers.get('X-Preview-Status') === 'cancelled') {
    return { blob: new Blob([]), meta: null, cached: false, cancelled: true };
  }
  if (response.status === 503) {
    const body = await response.json().catch(() => null);
    if (body?.busy === true) throw new PreviewBusyError(String(body.stage ?? 'UNKNOWN'), Number(body.retry_after_ms) > 0 ? Number(body.retry_after_ms) : 1000);
    throw apiError(body, 503);
  }
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw apiError(error, response.status);
  }
  let meta: LivePreviewMeta | null = null;
  try { meta = JSON.parse(response.headers.get('X-Preview-Meta') ?? 'null'); } catch { meta = null; }
  return { blob: await response.blob(), meta, cached: response.headers.get('X-Preview-Cache') === 'HIT' };
}
export type FacePose = { yaw: number; pitch: number; roll: number };
export type ExtractedFace = { face_id: string; frame_index: number; face_index: number; pose: FacePose; pose_cell: string; embedding_dim: number; thumbnail_base64: string };
export type ExtractFromFrameResult = {
  destination: 'source_gallery' | 'active_faceset' | 'target_cluster';
  face: ExtractedFace;
  active_source?: { faceset_id: string; name: string };
  faceset?: { faceset_id: string; name: string; item_count: number };
  inserted?: { view_index: number; pose_cell: string };
  cluster?: { rank: number; created: boolean };
};
/** Stage 5 "Use Face From Frame": face numbers are 0-based, counted left to right in that frame. */
export const extractFaceFromFrame = (body: {
  target_media_path: string;
  frame_index: number;
  face_index: number;
  destination: ExtractFromFrameResult['destination'];
  faceset_name?: string;
  faceset_id?: string;
  norm_cx?: number;
  norm_cy?: number;
}) => post<ExtractFromFrameResult>('/faces/extract_from_frame', body);
/** Stage 12: a person locked from a frame, with the multi-angle reference bank that keeps the lock through head turns. */
/** One cell of the 5 x 3 angle grid: the best face the harvester kept for that yaw / pitch bin. `bucket` is [yaw bin, pitch bin]. */
export type AngleBucket = {
  bucket: [number, number]; quality: number; quality_locked: boolean; has_crop: boolean; yaw: number; pitch: number; roll: number;
  frame_index: number; source: 'anchor' | 'auto' | 'user'; match_score: number;
};
export type TargetLock = {
  coverage?: { captured: number; quality_locked: number; total: number; buckets?: AngleBucket[] };
  /** Persistent, zero-based `face N` slot of the person (never reused, unchanged when the left-to-right order swaps). */
  face_id?: number;
  track_id?: string | null;
  lock_id: string; target_number: number; label: string; media_path: string; cluster_id: number;
  anchor_frame_index: number; bank_size: number; observed_frames: number[];
  reference_bank: { yaw: number; pitch: number; roll: number; frame_index: number; source: 'anchor' | 'auto' | 'user'; match_score: number }[];
  scan: { state: 'idle' | 'running' | 'done' | 'error' | 'skipped' | 'cancelled'; sampled?: number; requested?: number; matched?: number; message?: string; exhaustive?: boolean; skipped?: number };
  selected_face_index?: number; added?: boolean; similarity_to_bank?: number;
};
type FaceSelector = { face_index: number } | { point: [number, number] };
/** Click a face on any frame: locks that person and (in the background) samples the video for more angles of them. */
export const lockTargetFromFrame = (body: { path: string; frame_index: number; sample_count?: number; wait?: boolean; track_id?: string } & FaceSelector) =>
  post<TargetLock>('/target/lock-from-frame', body);
/** "Add This Angle to Target #n": e.g. an extreme profile that the scan could not link by itself. */
export const addTargetAngleSample = (body: { lock_id: string; frame_index: number; force?: boolean } & FaceSelector) =>
  post<TargetLock>('/target/add-angle-sample', body);
/** Unlock a target; the `face N` slot and its reference bank are kept. */
export const unlockTarget = (lockId: string) => api<{ lock_id: string; face_id: number; unlocked: boolean }>(`/target/${lockId}`, { method: 'DELETE' });
/** "Re-Scan All Angles": drop the automatically harvested references and harvest the video again (409 while a scan runs). */
export const rescanTargetAngles = (lockId: string, body: { sample_count?: number } = {}) => post<TargetLock>(`/target/${lockId}/rescan`, body);
/** The stored 256 px crop of one angle cell. `version` busts the cache when a better face replaces the cell's crop. */
export const angleCropURL = (lockId: string, bucket: readonly [number, number], version: string | number = '') =>
  `/api/v1/target/${lockId}/angle/${bucket[0]}/${bucket[1]}/crop.jpg${version === '' ? '' : `?v=${encodeURIComponent(String(version))}`}`;
export const mediaURL = (path: string) => `/api/v1/media/content?${new URLSearchParams({ path })}`;
export async function upload(file: File, capabilities: Capabilities, progress: (value: number) => void, signal: AbortSignal) {
  if (file.size > capabilities.workstation.max_asset_bytes) throw new Error('This file exceeds the advertised upload limit.');
  const session = await post<{ upload_id: string; chunk_bytes: number }>('/assets/uploads', { filename: file.name, size_bytes: file.size }, { signal });
  try {
    for (let offset = 0; offset < file.size; offset += session.chunk_bytes) {
      const chunk = file.slice(offset, offset + session.chunk_bytes);
      await api(`/assets/uploads/${session.upload_id}?offset=${offset}`, { method: 'PUT', body: chunk, signal, headers: { 'Content-Type': 'application/octet-stream' } });
      progress(Math.min(1, (offset + chunk.size) / file.size));
    }
    return await post<{ kind: 'video' | 'faceset' | 'image'; path?: string; faceset_id?: string }>(`/assets/uploads/${session.upload_id}/complete`, {}, { signal });
  } catch (error) {
    await api(`/assets/uploads/${session.upload_id}`, { method: 'DELETE' }).catch(() => undefined);
    throw error;
  }
}
