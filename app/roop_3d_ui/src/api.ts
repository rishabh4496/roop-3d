import type { Capabilities } from './contracts';
export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`/api/v1${path}`, { ...init, headers: { ...(typeof init.body === 'string' ? { 'Content-Type': 'application/json' } : {}), ...init.headers } });
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new Error(error?.error?.message ?? `Local service returned ${response.status}`);
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
export type LivePreviewResult = { blob: Blob; meta: LivePreviewMeta | null; cached: boolean };

export async function postLivePreview(payload: unknown, init: RequestInit = {}): Promise<LivePreviewResult> {
  const response = await fetch('/api/preview', { method: 'POST', body: JSON.stringify(payload), cache: 'no-store', ...init, headers: { 'Content-Type': 'application/json', ...init.headers } });
  if (response.status === 503) {
    const body = await response.json().catch(() => null);
    if (body?.busy === true) throw new PreviewBusyError(String(body.stage ?? 'UNKNOWN'), Number(body.retry_after_ms) > 0 ? Number(body.retry_after_ms) : 1000);
    throw new Error(body?.error?.message ?? 'Local service returned 503');
  }
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new Error(error?.error?.message ?? `Local service returned ${response.status}`);
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
export const extractFaceFromFrame = (body: { target_media_path: string; frame_index: number; face_index: number; destination: ExtractFromFrameResult['destination']; faceset_name?: string }) =>
  post<ExtractFromFrameResult>('/faces/extract_from_frame', body);
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
    return await post<{ kind: 'video' | 'faceset'; path?: string; faceset_id?: string }>(`/assets/uploads/${session.upload_id}/complete`, {}, { signal });
  } catch (error) {
    await api(`/assets/uploads/${session.upload_id}`, { method: 'DELETE' }).catch(() => undefined);
    throw error;
  }
}
