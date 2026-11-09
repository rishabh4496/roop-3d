// @vitest-environment jsdom
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { expect, it, vi } from 'vitest';
import LiveStudio from './components/workstation/LiveStudio';

const { api, post } = vi.hoisted(() => ({ api: vi.fn(), post: vi.fn() }));
vi.mock('./api', () => ({ api, post }));
vi.mock('./components/canvas/WebGLViewport', () => ({ default: () => null }));
vi.mock('./liveStream', () => ({ LiveFrameStream: class { start() {} stop() {} } }));
Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

it('starts and stops with the selected identity and keeps unavailable outputs disabled', async () => {
  const idle = { state: 'stopped', frames: 0, error: null, capture_to_publish_p95_ms: null,
    virtual_camera_available: false, spout_available: false, screen_available: false };
  api.mockResolvedValue(idle);
  post.mockResolvedValueOnce({ ...idle, state: 'running' }).mockResolvedValueOnce(idle);
  const host = document.createElement('div'); document.body.append(host);
  const root = createRoot(host);
  try {
    await act(async () => root.render(<LiveStudio sourceId="alice" />));
    await act(async () => host.querySelector('button')!.click());
    const buttons = () => [...host.querySelectorAll('button')];
    expect([...host.querySelectorAll<HTMLInputElement>('input[type=checkbox]')].every(input => input.disabled)).toBe(true);
    await act(async () => buttons().find(b => b.textContent === 'Start live studio')!.click());
    expect(post).toHaveBeenCalledWith('/studio/start', expect.objectContaining({ source_faceset_id: 'alice', fps: 30, virtual_camera: false, spout: false }));
    await act(async () => buttons().find(b => b.textContent === 'Stop live studio')!.click());
    expect(post).toHaveBeenCalledWith('/studio/stop', {});
  } finally {
    await act(async () => root.unmount()); host.remove();
  }
});
