// @vitest-environment jsdom
import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import TimelineWorkstation, { zoomAnchoredScrollLeft } from './components/timeline/TimelineWorkstation';

vi.mock('./frameScrub', () => ({
  filmstripCount: () => 16,
  useFilmstrip: () => ({
    status: 'ready',
    error: '',
    data: {
      count: 2,
      thumb_width: 120,
      thumb_height: 68,
      total_frames: 300,
      duration_sec: 10,
      frames: [
        { slot: 0, frame_index: 30, requested_frame_index: 30, timestamp_sec: 1.033, data_url: 'data:image/jpeg;base64,AA==' },
        { slot: 1, frame_index: 150, requested_frame_index: 150, timestamp_sec: 5.033, data_url: 'data:image/jpeg;base64,AA==' },
      ],
    },
  }),
}));

describe('timeline navigation interactions', () => {
  beforeEach(() => {
    (globalThis as any).IS_REACT_ACT_ENVIRONMENT = true;
    (globalThis as any).ResizeObserver = class {
      observe() {}
      disconnect() {}
    };
  });

  it('zooms around the pointer without moving the point under it', () => {
    expect(zoomAnchoredScrollLeft(200, 250, 1000, 1500)).toBe(425);
    expect(zoomAnchoredScrollLeft(0, 0, 0, 1500)).toBe(0);
  });

  it('selects the actual frame shown in a filmstrip card and exposes active selection', async () => {
    const host = document.createElement('div');
    document.body.appendChild(host);
    const root = createRoot(host);
    const onSeek = vi.fn();

    await act(async () => {
      root.render(<TimelineWorkstation target="clip.mp4" currentTime={0} duration={10} fps={30} onSeek={onSeek} />);
    });

    const firstFrame = host.querySelector<HTMLButtonElement>('button[aria-label^="Seek to frame 31"]');
    expect(firstFrame).not.toBeNull();
    expect(firstFrame?.getAttribute('aria-pressed')).toBe('false');
    await act(async () => firstFrame?.click());
    expect(onSeek).toHaveBeenCalledWith(1);

    await act(async () => {
      root.render(<TimelineWorkstation target="clip.mp4" currentTime={1} duration={10} fps={30} onSeek={onSeek} />);
    });
    expect(host.querySelector('button[aria-label^="Seek to frame 31"]')?.getAttribute('aria-pressed')).toBe('true');

    await act(async () => root.unmount());
    host.remove();
  });

  it('keeps the cursor-anchored timeline point stable while zooming', async () => {
    const host = document.createElement('div');
    document.body.appendChild(host);
    const root = createRoot(host);

    await act(async () => {
      root.render(<TimelineWorkstation target="clip.mp4" currentTime={0} duration={10} fps={30} onSeek={() => {}} />);
    });

    const viewport = host.querySelector<HTMLElement>('.timeline-tracks-viewport')!;
    let scrollLeft = 200;
    Object.defineProperty(viewport, 'scrollLeft', { configurable: true, get: () => scrollLeft, set: value => { scrollLeft = value; } });
    vi.spyOn(viewport, 'getBoundingClientRect').mockReturnValue({
      x: 100, y: 0, left: 100, top: 0, right: 600, bottom: 100, width: 500, height: 100, toJSON: () => ({}),
    } as DOMRect);
    const wheel = new WheelEvent('wheel', { bubbles: true, cancelable: true, ctrlKey: true, deltaY: -100, clientX: 350 });
    await act(async () => { viewport.dispatchEvent(wheel); });

    expect(host.querySelector<HTMLInputElement>('.zoom-slider')?.value).toBe('1.5');
    expect(viewport.scrollLeft).toBe(425);

    await act(async () => root.unmount());
    host.remove();
  });

  it('pans the zoomed timeline with a middle-button drag without seeking', async () => {
    const host = document.createElement('div');
    document.body.appendChild(host);
    const root = createRoot(host);
    const onSeek = vi.fn();

    await act(async () => {
      root.render(<TimelineWorkstation target="clip.mp4" currentTime={0} duration={10} fps={30} onSeek={onSeek} />);
    });
    const viewport = host.querySelector<HTMLElement>('.timeline-tracks-viewport')!;
    let scrollLeft = 200;
    Object.defineProperty(viewport, 'scrollLeft', { configurable: true, get: () => scrollLeft, set: value => { scrollLeft = value; } });
    await act(async () => {
      viewport.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true, button: 1, clientX: 400 }));
    });
    expect(viewport.classList.contains('is-panning')).toBe(true);
    await act(async () => {
      window.dispatchEvent(new MouseEvent('pointermove', { bubbles: true, clientX: 360 }));
    });
    expect(viewport.scrollLeft).toBe(240);
    expect(onSeek).not.toHaveBeenCalled();
    expect(viewport.classList.contains('is-panning')).toBe(true);
    await act(async () => window.dispatchEvent(new MouseEvent('pointerup', { bubbles: true, button: 1 })));

    await act(async () => root.unmount());
    host.remove();
  });
});
