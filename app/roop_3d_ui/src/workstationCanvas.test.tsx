import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import WorkstationCanvas from './components/canvas/WorkstationCanvas';
import MaskBrushOverlay from './components/canvas/MaskBrushOverlay';

describe('WorkstationCanvas Comparison Modes', () => {
  it('renders 4 preview modes with active wipe controls', () => {
    const html = renderToStaticMarkup(
      <WorkstationCanvas
        target="target.mp4"
        output="swapped.mp4"
        mode="wipe"
        containerWidth={1280}
        containerHeight={720}
      />
    );

    // 4 comparison mode buttons
    expect(html).toContain('Wipe (A/B)');
    expect(html).toContain('Diff Map');
    expect(html).toContain('Alpha Blend');
    expect(html).toContain('Loupe (3.5x)');

    // Active wipe elements
    expect(html).toContain('preview-mode-btn active');
    expect(html).toContain('wipe-divider horizontal');
    expect(html).toContain('ORIGINAL');
    expect(html).toContain('SWAPPED');
    expect(html).toContain('50%');
  });

  it('renders diff map layer when mode is diff', () => {
    const html = renderToStaticMarkup(
      <WorkstationCanvas
        target="target.mp4"
        output="swapped.mp4"
        mode="diff"
        containerWidth={1280}
        containerHeight={720}
      />
    );

    expect(html).toContain('diff-canvas-layer');
    expect(html).toContain('Minor Shift');
    expect(html).toContain('Major Swap Difference');
    expect(html).toContain('diff-boost-slider');
  });

  it('renders alpha blend controls when mode is blend', () => {
    const html = renderToStaticMarkup(
      <WorkstationCanvas
        target="target.mp4"
        output="swapped.mp4"
        mode="blend"
        containerWidth={1280}
        containerHeight={720}
      />
    );

    expect(html).toContain('blend-slider');
    expect(html).toContain('alpha-blend-layer');
  });

  it('renders loupe lens badge when mode is loupe', () => {
    const html = renderToStaticMarkup(
      <WorkstationCanvas
        target="target.mp4"
        output="swapped.mp4"
        mode="loupe"
        containerWidth={1280}
        containerHeight={720}
      />
    );

    expect(html).toContain('3.5x SUB-PIXEL LENS');
  });
});

describe('MaskBrushOverlay Manual & Occlusion Editor', () => {
  it('renders brush controls, occlusion guides, and export actions', () => {
    const html = renderToStaticMarkup(
      <MaskBrushOverlay
        containerWidth={1280}
        containerHeight={720}
        mediaDimensions={{ width: 1920, height: 1080 }}
      />
    );

    // Brush modes
    expect(html).toContain('Include this area in the face swap. Paint the include mask.');
    expect(html).toContain('Exclude this area and keep the target pixels. Paint the exclusion mask.');
    expect(html).toContain('Erase the selected Include or Exclude override.');

    // Settings
    expect(html).toContain('Size:');
    expect(html).toContain('Hardness:');
    expect(html).toContain('Opacity:');

    // Occlusion guides
    expect(html).toContain('Occlusion Guides');
    expect(html).toContain('glasses, hair strands, fingers');
    expect(html).toContain('+ Fill Glasses');

    // Actions
    expect(html).toContain('Clear');
    expect(html).toContain('Done ✓');
  });
});
