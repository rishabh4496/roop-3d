// @vitest-environment jsdom
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, expect, it, vi } from 'vitest';
import PhotoSynthesisAction from './components/facemgr/PhotoSynthesisAction';
import { api, post } from './api';

vi.mock('./api', () => ({ api: vi.fn(), post: vi.fn(), postLivePreview: vi.fn(), PreviewBusyError: class extends Error {} }));
(globalThis as any).IS_REACT_ACT_ENVIRONMENT = true;
afterEach(() => { vi.useRealTimers(); vi.clearAllMocks(); });


it('synthesizes imported photos with one click, polls progress and activates the result', async () => {
  vi.useFakeTimers();
  const host = document.createElement('div'), root = createRoot(host), saved = vi.fn();
  document.body.append(host);
  vi.mocked(post).mockResolvedValue({ job_id: 'b'.repeat(32), status: 'synthesizing', progress: 0, scanned: 0 });
  vi.mocked(api).mockResolvedValue({ job_id: 'b'.repeat(32), status: 'ready', progress: 1, scanned: 36, result: { faceset_id: 'generated' } });
  await act(async () => root.render(<PhotoSynthesisAction images={['data:image/png;base64,AA==']} name="Alice" onSaved={saved}/>));
  await act(async () => host.querySelector<HTMLButtonElement>('button')!.click());
  expect(post).toHaveBeenCalledWith('/faceset/synthesize-photos', { images: ['data:image/png;base64,AA=='], name: 'Alice' });
  expect(host.querySelector<HTMLButtonElement>('button')?.disabled).toBe(true);
  await act(async () => { await vi.advanceTimersByTimeAsync(750); });
  expect(saved).toHaveBeenCalledWith('generated');
  expect(host.textContent).toContain('saved and activated');
  await act(async () => root.unmount()); host.remove();
});
