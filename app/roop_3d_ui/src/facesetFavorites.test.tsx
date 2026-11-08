// @vitest-environment jsdom
import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import FacesetLibrary from './components/facemgr/FacesetLibrary';

vi.mock('./components/facemgr/FacesetBuilder', () => ({ default: () => null }));

const sources = [
  { faceset_id: 'fs-zeta', name: 'Zeta', item_count: 2, frontal_count: 1 },
  { faceset_id: 'fs-alpha', name: 'Alpha', item_count: 3, frontal_count: 2 },
];

describe('FaceSet library favorites', () => {
  beforeEach(() => {
    (globalThis as any).IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    (globalThis as any).ResizeObserver = class {
      observe() {}
      disconnect() {}
    };
  });

  it('persists favorites, sorts them first, and does not select on favorite click', async () => {
    const host = document.createElement('div');
    document.body.appendChild(host);
    const root = createRoot(host);
    const onSelectSource = vi.fn();
    const render = async () => act(async () => {
      root.render(<FacesetLibrary variant="list" sources={sources} selectedSourceId="fs-zeta" onSelectSource={onSelectSource} />);
    });

    await render();
    expect(Array.from(host.querySelectorAll<HTMLElement>('.fs-row')).map(row => row.dataset.facesetId)).toEqual(['fs-zeta', 'fs-alpha']);
    const favorite = host.querySelector<HTMLButtonElement>('[aria-label="Add Alpha to favorites"]');
    expect(favorite).not.toBeNull();
    await act(async () => favorite?.click());

    expect(onSelectSource).not.toHaveBeenCalled();
    expect(Array.from(host.querySelectorAll<HTMLElement>('.fs-row')).map(row => row.dataset.facesetId)).toEqual(['fs-alpha', 'fs-zeta']);
    expect(host.querySelector('[aria-label="Remove Alpha from favorites"]')?.getAttribute('aria-pressed')).toBe('true');

    await act(async () => root.unmount());
    const restoredRoot = createRoot(host);
    await act(async () => {
      restoredRoot.render(<FacesetLibrary variant="list" sources={sources} selectedSourceId="fs-zeta" />);
    });
    expect(Array.from(host.querySelectorAll<HTMLElement>('.fs-row')).map(row => row.dataset.facesetId)).toEqual(['fs-alpha', 'fs-zeta']);
    expect(host.querySelector('[aria-label="Remove Alpha from favorites"]')?.getAttribute('aria-pressed')).toBe('true');

    await act(async () => restoredRoot.unmount());
    host.remove();
  });

  it('exposes a favorite control in the compact dropdown without selecting its row', async () => {
    const host = document.createElement('div');
    document.body.appendChild(host);
    const root = createRoot(host);
    const onSelectSource = vi.fn();
    await act(async () => {
      root.render(<FacesetLibrary sources={sources} selectedSourceId="fs-alpha" onSelectSource={onSelectSource} />);
    });

    const activeSetButton = Array.from(host.querySelectorAll<HTMLButtonElement>('button')).find(button => button.textContent?.includes('Alpha'));
    expect(activeSetButton).toBeDefined();
    await act(async () => activeSetButton?.click());
    const favorite = host.querySelector<HTMLButtonElement>('[aria-label="Add Zeta to favorites"]');
    expect(favorite).not.toBeNull();
    await act(async () => favorite?.click());

    expect(onSelectSource).not.toHaveBeenCalled();
    expect(host.querySelector('[aria-label="Remove Zeta from favorites"]')?.getAttribute('aria-pressed')).toBe('true');

    await act(async () => root.unmount());
    host.remove();
  });
});
