import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import WorkstationCanvas from './components/canvas/WorkstationCanvas';
import MaskBrushOverlay from './components/canvas/MaskBrushOverlay';
import { COLORMAPS, LOUPE_ZOOM, LOUPE_ZOOMS, colormapColor } from './components/canvas/webglModel';
import ActorPanel, { MIXED, actorAssignment, formatScreenTime, type ActorReport } from './components/studio/ActorPanel';
import { LAST_ROUTE_KEY, isEmptyHash, normalizeRoute, readStoredRoute, storeRoute } from './router/AppRouter';
import { buildLivePayload } from './livePreview';
import type { FaceSet, Inspection, JobRequest } from './contracts';

const canvasProps = { output: 'live', originalUrl: 'blob:original', swappedUrl: 'blob:swapped', containerWidth: 1000, containerHeight: 500, mediaDimensions: { width: 1920, height: 1080 } };
const canvas = (extra: Record<string, unknown> = {}) => renderToStaticMarkup(<WorkstationCanvas {...canvasProps} {...extra} />);

describe('Stage 23 canvas inspection modes', () => {
  it('offers all six modes: wipe, crossfade/blend, thermal diff, loupe and the 3D mesh overlay', () => {
    const html = canvas({ mode: 'wipe' });
    for (const label of ['Split Wipe', 'Crossfade', 'Alpha Blend', 'Diff Map', 'Loupe (3.5x)', '3D Mesh Overlay']) expect(html).toContain(label);
  });

  it('keeps the 3D mesh overlay usable before any swapped frame exists, unlike the A/B-only modes', () => {
    const html = renderToStaticMarkup(<WorkstationCanvas {...canvasProps} output="" swappedUrl="" outputPreviewUrl="" mode="mesh3d" />);
    const button = (label: string) => new RegExp(`<button[^>]*?(disabled)?[^>]*>(?:(?!</button>).)*${label}`, 's').exec(html);
    expect(button('3D Mesh Overlay')?.[1]).toBeUndefined();
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>(?:(?!<\/button>).)*Crossfade/s);
    expect(html).toContain('mesh3d-legend');
    expect(html).toContain('mode-mesh3d');
  });

  it('adds a thermal colormap (black to white through magenta, red and amber) and makes it the diff default', () => {
    expect(COLORMAPS[0]).toBe('thermal');
    expect(colormapColor('thermal', 0).map(Math.round)).toEqual([0, 0, 0]);
    expect(colormapColor('thermal', 1).map(Math.round)).toEqual([255, 255, 255]);
    const mid = colormapColor('thermal', 0.6);
    expect(mid[0]).toBeGreaterThan(mid[2]);                                   // warm in the middle of the ramp
    expect(canvas({ mode: 'diff' })).toContain('Diff Map');
  });

  it('runs the pore magnifier at 3.5x by default with a 4x option', () => {
    expect(LOUPE_ZOOM).toBe(3.5);
    expect(LOUPE_ZOOMS).toEqual([3.5, 4]);
    const html = canvas({ mode: 'loupe' });
    expect(html).toContain('3.5x SUB-PIXEL LENS');
    expect(html).toContain('aria-label="Loupe magnification"');
    expect(html).toMatch(/aria-pressed="true"[^>]*title="3.5x pore magnifier"/);
    expect(html).toMatch(/aria-pressed="false"[^>]*title="4x pore magnifier"/);
  });
});

describe('Stage 23 paint / erase brush scope', () => {
  const overlay = (extra: Record<string, unknown> = {}) => renderToStaticMarkup(<MaskBrushOverlay containerWidth={800} containerHeight={450} mediaDimensions={{ width: 1920, height: 1080 }} {...extra} />);

  it('keeps include, exclude and erase, and adds whole-clip vs frame-range scope', () => {
    const html = overlay();
    for (const text of ['Include', 'Exclude', 'Erase', 'Whole clip', 'Frame range']) expect(html).toContain(text);
    expect(html).toMatch(/aria-pressed="true"[^>]*>Whole clip/);
    expect(html).not.toContain('Painted frame ranges');
  });

  it('lists painted frame ranges so each can be edited or removed', () => {
    const html = overlay({ frame: 12, maskKeyframes: [{ start_frame: 10, end_frame: 20, mask_base64: 'x' }, { start_frame: 30, end_frame: null, mask_base64: 'y' }] });
    expect(html).toContain('aria-label="Painted frame ranges"');
    expect(html).toContain('frames 10–20');
    expect(html).toContain('frames 30–end');
    expect(html).toContain('aria-label="Remove mask for frames 10–20"');
  });

  it('sends the range mask to the live preview exactly where the render will use it', () => {
    const request = { face_mappings: [{ track_id: 'track_0001', faceset_id: 'a' }], target_path: 't.mp4', inspection_id: 'a'.repeat(32), mask_keyframes: [{ start_frame: 10, end_frame: 20, mask_base64: 'RANGE' }] } as unknown as JobRequest;
    const params = { bypass: [], restoration_blend: 0, pose_mode: 'retarget' } as never;
    const at = (seconds: number) => buildLivePayload({ request, time: seconds, fps: 10, parameters: params, maskBase64: 'WHOLE', restorerAvailable: false, solo: null }) as Record<string, unknown>;
    expect(at(1.5).mask_base64).toBe('RANGE');   // frame 15
    expect(at(2.5).mask_base64).toBe('WHOLE');   // frame 25, outside the range
  });
});

describe('Stage 23 people (actors) panel', () => {
  const sources: FaceSet[] = [{ faceset_id: 'alice', name: 'Alice', item_count: 5 }, { faceset_id: 'bob', name: 'Bob', item_count: 3 }];
  const inspection = { inspection_id: 'a'.repeat(32), tracks: [] } as unknown as Inspection;
  const report: ActorReport = {
    inspection_id: inspection.inspection_id, actor_count: 2, warnings: [],
    actors: [
      { actor_id: 'actor_0001', label: 'Person #1', rank: 1, track_ids: ['track_0001', 'track_0002'], persistent_ids: [7], grouping: 'persistent_id', screen_time_seconds: 72, prominence: 70, thumbnail_base64: 'AAAA', thumbnail_mime: 'image/jpeg' },
      { actor_id: 'actor_0002', label: 'Person #2', rank: 2, track_ids: ['track_0003'], persistent_ids: [], grouping: 'single_track', screen_time_seconds: 4.25, prominence: 3, thumbnail_base64: '', thumbnail_mime: 'image/jpeg' },
    ],
  };

  it('ranks people, shows screen time, and lets each have its own FaceSet', () => {
    const html = renderToStaticMarkup(<ActorPanel inspection={inspection} sources={sources} assigned={{ track_0001: 'alice', track_0002: 'alice', track_0003: 'bob' }} onAssign={() => {}} report={report} />);
    expect(html.indexOf('Person #1')).toBeLessThan(html.indexOf('Person #2'));
    expect(html).toContain('1m 12s on screen · 2 tracks');
    expect(html).toContain('4.3s on screen · 1 track');
    expect(html).toContain('2 people · 2 mapped · 2 different FaceSets');
    expect(html).toMatch(/aria-label="Person #1 source FaceSet"[^>]*>\s*<option value="">Keep original<\/option>/);
    expect(html).toContain('own 3D skull lock');
  });

  it('flags a person whose tracks disagree instead of hiding it', () => {
    expect(actorAssignment(report.actors[0], { track_0001: 'alice', track_0002: 'bob' })).toBe(MIXED);
    expect(actorAssignment(report.actors[0], { track_0001: 'alice', track_0002: 'alice' })).toBe('alice');
    expect(actorAssignment(report.actors[0], {})).toBe('');
    const html = renderToStaticMarkup(<ActorPanel inspection={inspection} sources={sources} assigned={{ track_0001: 'alice', track_0002: 'bob' }} onAssign={() => {}} report={report} />);
    expect(html).toContain('Mixed (set per track below)');
  });

  it('renders nothing before an inspection and formats screen time', () => {
    expect(renderToStaticMarkup(<ActorPanel inspection={null} sources={sources} assigned={{}} onAssign={() => {}} report={report} />)).toBe('');
    expect(formatScreenTime(0)).toBe('0s');
    expect(formatScreenTime(59.6)).toBe('60s');
    expect(formatScreenTime(125)).toBe('2m 05s');
  });
});

describe('Stage 23 route persistence', () => {
  const memory = () => { const data = new Map<string, string>(); return { getItem: (k: string) => data.get(k) ?? null, setItem: (k: string, v: string) => { data.set(k, v); } } as unknown as Storage; };

  it('remembers the last of the nine screens and ignores anything else', () => {
    const storage = memory();
    expect(readStoredRoute(storage)).toBeNull();
    for (const id of ['home', 'faceswap', 'batch', 'processing', 'facemgr', 'extras', 'gallery', 'history', 'settings']) {
      storeRoute(`#/${id}`, storage);
      expect(readStoredRoute(storage)).toBe(`#/${id}`);
    }
    storeRoute('#/nonsense', storage);
    expect(readStoredRoute(storage)).toBe('#/settings');
    storage.setItem(LAST_ROUTE_KEY, '#/evil');
    expect(readStoredRoute(storage)).toBeNull();
  });

  it('only an empty hash falls back to the remembered screen, so explicit links always win', () => {
    expect([isEmptyHash(''), isEmptyHash('#'), isEmptyHash('#/')]).toEqual([true, true, true]);
    expect(isEmptyHash('#/batch')).toBe(false);
    expect(normalizeRoute('#/batch')).toBe('#/batch');
    expect(normalizeRoute('#/queue')).toBe('#/processing');
    expect(normalizeRoute('#/studio')).toBe('#/faceswap');
  });

  it('survives unavailable storage', () => {
    const broken = { getItem: () => { throw new Error('blocked'); }, setItem: () => { throw new Error('blocked'); } } as unknown as Storage;
    expect(readStoredRoute(broken)).toBeNull();
    expect(() => storeRoute('#/batch', broken)).not.toThrow();
  });
});
