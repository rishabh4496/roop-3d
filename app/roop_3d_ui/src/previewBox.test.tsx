// @vitest-environment jsdom
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { VideoViewport } from './VideoViewport';
import CompositionGridOverlay from './components/canvas/CompositionGridOverlay';

vi.mock('./components/canvas/FaceOverlayCanvas', () => ({ default: () => null }));
vi.mock('./components/canvas/WorkstationCanvas', () => ({ default: () => <div>Comparison canvas</div> }));
vi.mock('./components/canvas/MaskBrushOverlay', () => ({ default: () => null }));
vi.mock('./components/tray/FaceSelectionTray', () => ({ default: () => null }));
vi.mock('./components/timeline/TimelineWorkstation', () => ({ default: () => <div>Timeline</div>, parseFps: () => 30 }));

describe('CompositionGridOverlay Component', () => {
  it('returns null when mode is off or empty', () => {
    expect(renderToStaticMarkup(<CompositionGridOverlay mode="off" />)).toBe('');
    expect(renderToStaticMarkup(<CompositionGridOverlay />)).toBe('');
  });

  it('renders Rule of Thirds SVG guides with golden power points', () => {
    const html = renderToStaticMarkup(<CompositionGridOverlay mode="thirds" />);
    expect(html).toContain('composition-grid-overlay mode-thirds');
    expect(html).toContain('RULE OF THIRDS');
    expect(html).toContain('thirds-line');
    expect(html).toContain('guide-point');
  });

  it('renders Center Crosshair SVG guides with alignment rings and tick marks', () => {
    const html = renderToStaticMarkup(<CompositionGridOverlay mode="crosshair" />);
    expect(html).toContain('composition-grid-overlay mode-crosshair');
    expect(html).toContain('CENTER CROSSHAIR · 3D ALIGNMENT');
    expect(html).toContain('crosshair-axis');
    expect(html).toContain('crosshair-ring');
    expect(html).toContain('crosshair-center-dot');
  });

  it('renders Action Safe (90%) and Title Safe (80%) margin boxes', () => {
    const html = renderToStaticMarkup(<CompositionGridOverlay mode="safe" />);
    expect(html).toContain('composition-grid-overlay mode-safe');
    expect(html).toContain('90% ACTION SAFE');
    expect(html).toContain('80% TITLE SAFE');
    expect(html).toContain('safe-box action-safe');
    expect(html).toContain('safe-box title-safe');
  });
});

describe('VideoViewport Preview Box Controls', () => {
  beforeEach(() => {
    (globalThis as any).IS_REACT_ACT_ENVIRONMENT = true;
  });

  it('renders preview HUD toolbar with zoom, grid, overlays, snapshot and fullscreen buttons', () => {
    const html = renderToStaticMarkup(
      <VideoViewport
        target="sample.mp4"
        output=""
        time={2.5}
        setTime={() => {}}
        duration={10}
        enabled={true}
        keys={[]}
      />
    );

    // Floating HUD toolbar
    expect(html).toContain('preview-hud-toolbar');
    expect(html).toContain('Preview Zoom');
    expect(html).toContain('zoom-step-btn');
    expect(html).toContain('Fit');
    expect(html).toContain('Framing Guides');
    expect(html).toContain('Overlays');
    expect(html).toContain('Snapshot');
    expect(html).toContain('fullscreen-toggle-btn');

    // Viewport stage wrapper
    expect(html).toContain('viewport-stage');
  });

  it('renders Hold: Source button when output media is provided', () => {
    const html = renderToStaticMarkup(
      <VideoViewport
        target="sample.mp4"
        output="sample_swapped.mp4"
        time={2.5}
        setTime={() => {}}
        duration={10}
        enabled={true}
        keys={[]}
      />
    );

    expect(html).toContain('hud-peek-btn');
    expect(html).toContain('Hold: Source');
  });

  it('toggles framing grid when clicking grid button', async () => {
    const host = document.createElement('div');
    document.body.appendChild(host);
    const root = createRoot(host);

    await act(async () => {
      root.render(
        <VideoViewport
          target="sample.mp4"
          output=""
          time={1.0}
          setTime={() => {}}
          duration={5}
          enabled={true}
          keys={[]}
        />
      );
    });

    // Initially grid is off
    expect(host.querySelector('.composition-grid-overlay')).toBeNull();

    // Click grid button to activate thirds
    const gridBtn = Array.from(host.querySelectorAll('.hud-btn')).find(b => b.textContent?.includes('Grid')) as HTMLButtonElement;
    expect(gridBtn).toBeDefined();

    await act(async () => {
      gridBtn.click();
    });

    expect(host.querySelector('.composition-grid-overlay.mode-thirds')).not.toBeNull();

    // Click again to cycle to crosshair
    await act(async () => {
      gridBtn.click();
    });
    expect(host.querySelector('.composition-grid-overlay.mode-crosshair')).not.toBeNull();

    // Click again to cycle to safe
    await act(async () => {
      gridBtn.click();
    });
    expect(host.querySelector('.composition-grid-overlay.mode-safe')).not.toBeNull();

    // Click again to return to off
    await act(async () => {
      gridBtn.click();
    });
    expect(host.querySelector('.composition-grid-overlay')).toBeNull();

    await act(async () => {
      root.unmount();
    });
    host.remove();
  });

  it('cycles zoom levels and provides Fit reset', async () => {
    const host = document.createElement('div');
    document.body.appendChild(host);
    const root = createRoot(host);

    await act(async () => {
      root.render(
        <VideoViewport
          target="sample.mp4"
          output=""
          time={1.0}
          setTime={() => {}}
          duration={5}
          enabled={true}
          keys={[]}
        />
      );
    });

    const zoomLevelBtn = host.querySelector('.zoom-level-btn') as HTMLButtonElement;
    expect(zoomLevelBtn.textContent).toBe('Fit');

    // Click to zoom to 200%
    await act(async () => {
      zoomLevelBtn.click();
    });

    expect(zoomLevelBtn.textContent).toBe('200%');
    expect(host.querySelector('.hud-reset-btn')).not.toBeNull();

    // Click Reset / Fit
    const resetBtn = host.querySelector('.hud-reset-btn') as HTMLButtonElement;
    await act(async () => {
      resetBtn.click();
    });

    expect(zoomLevelBtn.textContent).toBe('Fit');
    expect(host.querySelector('.hud-reset-btn')).toBeNull();

    await act(async () => {
      root.unmount();
    });
    host.remove();
  });
});
