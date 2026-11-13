// @vitest-environment jsdom
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import TargetMediaCard from './TargetMediaCard';
import { targetMediaStore, registerTargetMedia, clearTargetMedia } from '../../stores/targetMedia';

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

vi.mock('../../api', () => ({
  mediaURL: (path: string) => `/api/v1/media/content?path=${encodeURIComponent(path)}`,
  post: vi.fn().mockResolvedValue({ ok: true }),
  api: vi.fn().mockResolvedValue({ ok: true }),
}));

describe('TargetMediaCard & Target Media Lifecycle', () => {
  beforeEach(() => {
    clearTargetMedia();
    vi.clearAllMocks();
  });

  it('renders dropzone and Add Target Media button when no media is loaded', async () => {
    const onAdd = vi.fn();
    const onReplace = vi.fn();
    const onRemove = vi.fn();
    const onDrop = vi.fn();

    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);

    try {
      await act(async () => {
        root.render(
          <TargetMediaCard
            target=""
            probe={null}
            busy={false}
            onAddTarget={onAdd}
            onReplaceTarget={onReplace}
            onRemoveTarget={onRemove}
            onDropFiles={onDrop}
          />
        );
      });

      expect(host.querySelector('[data-testid="target-media-empty"]')).not.toBeNull();
      expect(host.querySelector('[data-testid="target-dropzone"]')).not.toBeNull();
      const addBtn = host.querySelector('[data-testid="add-target-media-button"]') as HTMLButtonElement;
      expect(addBtn).not.toBeNull();
      expect(addBtn.textContent).toContain('Add Target Media');

      await act(async () => {
        addBtn.click();
      });
      expect(onAdd).toHaveBeenCalledTimes(1);
    } finally {
      await act(async () => { root.unmount(); });
      host.remove();
    }
  });

  it('renders active preview card for video target with metadata and replace/remove actions', async () => {
    const onReplace = vi.fn();
    const onRemove = vi.fn();

    act(() => {
      registerTargetMedia({
        path: 'roop_3d/.runtime/media/target.mp4',
        filename: 'target.mp4',
        mediaType: 'video',
        width: 1920,
        height: 1080,
        durationSeconds: 15.5,
        fps: 30,
      });
    });

    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);

    try {
      await act(async () => {
        root.render(
          <TargetMediaCard
            target="roop_3d/.runtime/media/target.mp4"
            probe={{
              width: 1920,
              height: 1080,
              duration_seconds: 15.5,
              fps_rational: '30/1',
              declared_frame_count: 465,
              audio_tracks: [],
            }}
            busy={false}
            onAddTarget={vi.fn()}
            onReplaceTarget={onReplace}
            onRemoveTarget={onRemove}
            onDropFiles={vi.fn()}
          />
        );
      });

      expect(host.querySelector('[data-testid="target-media-active"]')).not.toBeNull();
      expect(host.querySelector('[data-testid="target-filename"]')?.textContent).toBe('target.mp4');
      expect(host.textContent).toContain('VIDEO');
      expect(host.textContent).toContain('1920 × 1080');
      expect(host.querySelector('[data-testid="target-video-thumbnail"]')).not.toBeNull();

      const replaceBtn = host.querySelector('[data-testid="replace-target-button"]') as HTMLButtonElement;
      await act(async () => {
        replaceBtn.click();
      });
      expect(onReplace).toHaveBeenCalledTimes(1);

      const removeBtn = host.querySelector('[data-testid="remove-target-button"]') as HTMLButtonElement;
      await act(async () => {
        removeBtn.click();
      });
      expect(onRemove).toHaveBeenCalledTimes(1);
    } finally {
      await act(async () => { root.unmount(); });
      host.remove();
    }
  });

  it('renders active preview card for image target with IMAGE badge and thumbnail', async () => {
    act(() => {
      registerTargetMedia({
        path: 'roop_3d/.runtime/media/portrait.png',
        filename: 'portrait.png',
        mediaType: 'image',
        width: 1024,
        height: 1024,
        durationSeconds: 0,
        fps: 0,
      });
    });

    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);

    try {
      await act(async () => {
        root.render(
          <TargetMediaCard
            target="roop_3d/.runtime/media/portrait.png"
            probe={{
              width: 1024,
              height: 1024,
              duration_seconds: 0,
              fps_rational: null,
              declared_frame_count: 1,
              audio_tracks: [],
            }}
            busy={false}
            onAddTarget={vi.fn()}
            onReplaceTarget={vi.fn()}
            onRemoveTarget={vi.fn()}
            onDropFiles={vi.fn()}
          />
        );
      });

      expect(host.textContent).toContain('IMAGE');
      expect(host.textContent).toContain('Single Frame');
      expect(host.querySelector('[data-testid="target-image-thumbnail"]')).not.toBeNull();
    } finally {
      await act(async () => { root.unmount(); });
      host.remove();
    }
  });

  it('revokes object URLs and clears store on reset', () => {
    const revokeSpy = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});

    act(() => {
      registerTargetMedia({
        path: 'roop_3d/.runtime/media/test.mp4',
        filename: 'test.mp4',
        mediaType: 'video',
        width: 1280,
        height: 720,
        objectUrl: 'blob:http://localhost/test-blob',
      });
    });

    expect(targetMediaStore.get().path).toBe('roop_3d/.runtime/media/test.mp4');
    expect(targetMediaStore.get().objectUrl).toBe('blob:http://localhost/test-blob');

    act(() => {
      clearTargetMedia();
    });

    expect(revokeSpy).toHaveBeenCalledWith('blob:http://localhost/test-blob');
    expect(targetMediaStore.get().path).toBe('');
    expect(targetMediaStore.get().objectUrl).toBeNull();

    revokeSpy.mockRestore();
  });
});
