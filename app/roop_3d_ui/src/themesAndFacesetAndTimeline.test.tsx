// @vitest-environment jsdom
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { THEMES, themeByName, applyThemeToDom } from './themes';
import ThemeGallery from './components/ThemeGallery';
import FacesetLibrary from './components/facemgr/FacesetLibrary';
import TimelineWorkstation, { RangeField } from './components/timeline/TimelineWorkstation';
import { panAnchoredAt, clampPan, uiScale, stageSize, contentSize } from './utils/zoomPan';

describe('Themes & Styles Engine (37 Studio Themes)', () => {
  beforeEach(() => {
    (globalThis as any).IS_REACT_ACT_ENVIRONMENT = true;
  });

  it('contains all 37 studio themes across dark, AMOLED, and light families', () => {
    expect(THEMES.length).toBe(37);
    const names = THEMES.map(t => t.name);
    expect(names).toContain('Default');
    expect(names).toContain('Dracula');
    expect(names).toContain('AMOLED Black');
    expect(names).toContain('AMOLED Matrix');
    expect(names).toContain('Glass Light');
    expect(names).toContain('Sakura Light');
    expect(names).toContain('Cyberpunk Dark');
    expect(names).toContain('Tokyo Night');
  });

  it('resolves themeByName correctly with fallback', () => {
    expect(themeByName('Dracula').accent).toBe('#bd93f9');
    expect(themeByName('Nonexistent Theme').name).toBe('Default');
  });

  it('applies theme classes and dataset attributes to DOM', () => {
    const dracula = themeByName('Dracula');
    applyThemeToDom(dracula);
    expect(document.documentElement.classList.contains('theme-dracula')).toBe(true);
    expect(document.documentElement.getAttribute('data-theme-mode')).toBe('dark');
    expect(document.documentElement.dataset.theme).toBe('dark');

    const glassLight = themeByName('Glass Light');
    applyThemeToDom(glassLight);
    expect(document.documentElement.classList.contains('theme-dracula')).toBe(false);
    expect(document.documentElement.classList.contains('theme-glass-light')).toBe(true);
    expect(document.documentElement.getAttribute('data-theme-mode')).toBe('light');
    expect(document.documentElement.dataset.theme).toBe('light');
  });

  it('renders ThemeGallery grid and triggers onChange callback', async () => {
    const onChange = vi.fn();
    const host = document.createElement('div');
    document.body.appendChild(host);
    const root = createRoot(host);

    await act(async () => {
      root.render(<ThemeGallery value="Default" onChange={onChange} />);
    });

    const cards = host.querySelectorAll('.theme-card');
    expect(cards.length).toBe(37);

    // Find and click Dracula
    const draculaCard = Array.from(cards).find(c => c.textContent?.includes('Dracula')) as HTMLElement;
    expect(draculaCard).toBeDefined();

    await act(async () => {
      draculaCard.click();
    });

    expect(onChange).toHaveBeenCalledWith('Dracula');

    await act(async () => {
      root.unmount();
    });
    host.remove();
  });
});

describe('FaceSet Library Component', () => {
  beforeEach(() => {
    (globalThis as any).IS_REACT_ACT_ENVIRONMENT = true;
  });

  const dummySources = [
    { faceset_id: 'fs_01', name: 'Alice Character', item_count: 5, frontal_count: 3 },
    { faceset_id: 'fs_02', name: 'Bob Double', item_count: 8, frontal_count: 4 },
  ];

  it('renders FaceSet library with sources list and active selection', () => {
    const html = renderToStaticMarkup(
      <FacesetLibrary sources={dummySources} selectedSourceId="fs_01" />
    );
    expect(html).toContain('Alice Character');
    expect(html).toContain('FACESET LIBRARY');
    expect(html).toContain('Import .fsz');
  });

  it('opens picker and filters sources based on search query', async () => {
    const host = document.createElement('div');
    document.body.appendChild(host);
    const root = createRoot(host);

    await act(async () => {
      root.render(<FacesetLibrary sources={dummySources} selectedSourceId="fs_01" />);
    });

    // Find dropdown trigger button displaying active selection
    const dropdownBtn = Array.from(host.querySelectorAll('button')).find(b => b.textContent?.includes('Alice Character')) as HTMLButtonElement;
    expect(dropdownBtn).toBeDefined();

    await act(async () => {
      dropdownBtn.click();
    });

    const input = host.querySelector<HTMLInputElement>('.faceset-search-input');
    expect(input).toBeDefined();

    await act(async () => {
      const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')!.set!;
      nativeSetter.call(input, 'Alice');
      input!.dispatchEvent(new Event('input', { bubbles: true }));
      input!.dispatchEvent(new Event('change', { bubbles: true }));
    });

    // Check Alice is displayed
    expect(host.textContent).toContain('Alice Character');

    await act(async () => {
      root.unmount();
    });
    host.remove();
  });
});

describe('Timeline Set In and Out Enhancements', () => {
  beforeEach(() => {
    (globalThis as any).IS_REACT_ACT_ENVIRONMENT = true;
  });

  it('renders editable RangeField and commits value on click-to-edit', async () => {
    const onCommit = vi.fn();
    const host = document.createElement('div');
    document.body.appendChild(host);
    const root = createRoot(host);

    await act(async () => {
      root.render(
        <RangeField
          label="In"
          frame={30}
          time={1.0}
          fps={30}
          onCommitFrame={onCommit}
        />
      );
    });

    expect(host.textContent).toContain('In');
    expect(host.textContent).toContain('f:30');

    // Click to start editing
    const btn = host.querySelector('.range-field-btn') as HTMLButtonElement;
    await act(async () => {
      btn.click();
    });

    const input = host.querySelector('.range-field-input') as HTMLInputElement;
    expect(input).toBeDefined();
    expect(input.value).toBe('30');

    // Change input value to 60 and commit with enter
    await act(async () => {
      const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')!.set!;
      nativeSetter.call(input, '60');
      input.dispatchEvent(new Event('input', { bubbles: true }));
      input.dispatchEvent(new Event('change', { bubbles: true }));
      input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    });

    expect(onCommit).toHaveBeenCalledWith(60);

    await act(async () => {
      root.unmount();
    });
    host.remove();
  });

  it('renders TimelineWorkstation with jump in/out and range fields', () => {
    const html = renderToStaticMarkup(
      <TimelineWorkstation
        target="interview.mp4"
        currentTime={3.0}
        duration={10.0}
        fps={30}
        trimStart={1.0}
        trimEnd={6.0}
      />
    );

    expect(html).toContain('jump-in');
    expect(html).toContain('jump-out');
    expect(html).toContain('⏮ In');
    expect(html).toContain('Out ⏭');
    expect(html).toContain('trim-range-summary');
  });
});

describe('Zoom and Pan Pure Maths (zoomPan.ts)', () => {
  it('calculates anchored pan math accurately when zooming in and out', () => {
    const startPan = { x: 0, y: 0 };
    const pointer = { x: 400, y: 300 };
    const mockStage = {
      getBoundingClientRect: () => ({ left: 0, top: 0, width: 800, height: 600 }),
    } as any;

    // Zooming 1x -> 2x anchored at center (400, 300)
    const pan2x = panAnchoredAt(pointer, mockStage, 1.0, 2.0, startPan);
    expect(pan2x.x).toBeCloseTo(0, 0);
    expect(pan2x.y).toBeCloseTo(0, 0);

    // Zooming at offset (600, 450)
    const panOff = panAnchoredAt({ x: 600, y: 450 }, mockStage, 1.0, 2.0, startPan);
    expect(panOff.x).toBeCloseTo(-200, 0);
    expect(panOff.y).toBeCloseTo(-150, 0);
  });

  it('clamps pan to stage boundaries', () => {
    const mockStage = {
      getBoundingClientRect: () => ({ left: 0, top: 0, width: 800, height: 600 }),
      offsetWidth: 800,
      offsetHeight: 600,
    } as any;
    const clamped = clampPan({ x: 9999, y: 9999 }, 2.0, mockStage, mockStage);
    expect(clamped.x).toBeLessThan(9999);
    expect(clamped.y).toBeLessThan(9999);
  });
});
