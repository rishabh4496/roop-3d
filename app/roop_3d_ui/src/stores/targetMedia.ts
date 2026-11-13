import { createAtomicStore, useStoreSelector } from './atomicStore';
import { post } from '../api';

export interface TargetMediaInfo {
  path: string;
  filename: string;
  previewUrl: string;
  mediaType: 'image' | 'video' | null;
  width: number;
  height: number;
  durationSeconds: number;
  fps: number;
  objectUrl: string | null;
  sha256: string | null;
}

export const INITIAL_TARGET_MEDIA: TargetMediaInfo = {
  path: '',
  filename: '',
  previewUrl: '',
  mediaType: null,
  width: 0,
  height: 0,
  durationSeconds: 0,
  fps: 0,
  objectUrl: null,
  sha256: null,
};

export const targetMediaStore = createAtomicStore<TargetMediaInfo>(INITIAL_TARGET_MEDIA);

export function useTargetMedia<S>(selector: (state: TargetMediaInfo) => S): S {
  return useStoreSelector(targetMediaStore, selector);
}

export function setTargetMedia(patch: Partial<TargetMediaInfo>): void {
  targetMediaStore.set(patch);
}

export function clearTargetMedia(): void {
  const current = targetMediaStore.get();
  if (current.objectUrl) {
    try {
      URL.revokeObjectURL(current.objectUrl);
    } catch {
      // ignore revocation errors
    }
  }
  targetMediaStore.reset();
  // Notify backend session that target media is cleared
  void post('/target/media', null).catch(() => {});
}

export function registerTargetMedia(info: {
  path: string;
  filename: string;
  mediaType: 'image' | 'video';
  width: number;
  height: number;
  durationSeconds?: number | null;
  fps?: number | null;
  objectUrl?: string | null;
  sha256?: string | null;
}): void {
  const current = targetMediaStore.get();
  if (current.objectUrl && current.objectUrl !== info.objectUrl) {
    try {
      URL.revokeObjectURL(current.objectUrl);
    } catch {
      // ignore
    }
  }

  const resolved: TargetMediaInfo = {
    path: info.path,
    filename: info.filename,
    previewUrl: info.objectUrl || info.path,
    mediaType: info.mediaType,
    width: info.width || 0,
    height: info.height || 0,
    durationSeconds: info.durationSeconds ?? 0,
    fps: info.fps ?? 25,
    objectUrl: info.objectUrl || null,
    sha256: info.sha256 || null,
  };

  targetMediaStore.set(resolved);

  // Sync with FastAPI unified state
  void post('/target/media', {
    path: resolved.path,
    filename: resolved.filename,
    media_type: resolved.mediaType,
    width: resolved.width,
    height: resolved.height,
    duration_seconds: resolved.durationSeconds,
    fps: resolved.fps,
    sha256: resolved.sha256,
  }).catch(() => {});
}
