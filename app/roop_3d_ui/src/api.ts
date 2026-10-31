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
