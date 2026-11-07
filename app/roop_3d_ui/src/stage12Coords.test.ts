import { describe, expect, it } from 'vitest';
import { canvasToVideoCoords, videoBoxToCanvas, videoToCanvasCoords } from './utils/letterbox';

const wide = { containerW: 1000, containerH: 800, mediaW: 1920, mediaH: 1080 };   // pillarless: letterboxed top/bottom

describe('Stage 12 sub-pixel canvas <-> video coordinates', () => {
  it('maps the frame corners onto the letterboxed rectangle (object-fit: contain)', () => {
    const topLeft = videoToCanvasCoords([0, 0], wide), bottomRight = videoToCanvasCoords([1920, 1080], wide);
    expect(topLeft.x).toBeCloseTo(0, 9);
    expect(topLeft.y).toBeCloseTo((800 - 1000 * 1080 / 1920) / 2, 9);          // the bar above the picture
    expect(bottomRight.x).toBeCloseTo(1000, 9);
    expect(bottomRight.y).toBeCloseTo(800 - topLeft.y, 9);
  });

  it('round-trips arbitrary sub-pixel points exactly at every zoom and pan', () => {
    for (const zoom of [0.5, 1, 2.37, 8]) for (const pan of [{ x: 0, y: 0 }, { x: -140.5, y: 63.25 }]) {
      for (const p of [[0.25, 0.75], [960.5, 540.125], [1919.999, 1079.001], [123.456, 987.654]] as [number, number][]) {
        const screen = videoToCanvasCoords(p, { ...wide, zoom, pan });
        const back = canvasToVideoCoords([screen.x, screen.y], { ...wide, zoom, pan });
        expect(back.x).toBeCloseTo(p[0], 7);
        expect(back.y).toBeCloseTo(p[1], 7);
      }
    }
  });

  it('zooms about the container centre and pans after scaling (CSS translate(pan) scale(zoom))', () => {
    const centre = videoToCanvasCoords([960, 540], { ...wide, zoom: 4, pan: { x: 30, y: -10 } });
    expect(centre.x).toBeCloseTo(500 + 30, 9);
    expect(centre.y).toBeCloseTo(400 - 10, 9);
    const a = videoToCanvasCoords([0, 0], { ...wide, zoom: 2 }), b = videoToCanvasCoords([0, 0], wide);
    expect(a.x - 500).toBeCloseTo(2 * (b.x - 500), 9);
  });

  it('reports clicks on the letterbox bars as outside the frame', () => {
    expect(canvasToVideoCoords([500, 5], wide).inside).toBe(false);
    expect(canvasToVideoCoords([500, 400], wide).inside).toBe(true);
  });

  it('a native box keeps its aspect on screen and scales with zoom', () => {
    const box = videoBoxToCanvas([480, 270, 960, 540], { ...wide, zoom: 2 });
    expect(box.w / box.h).toBeCloseTo(480 / 270, 9);
    expect(box.w).toBeCloseTo(2 * (480 / 1920) * 1000, 9);
  });
});
