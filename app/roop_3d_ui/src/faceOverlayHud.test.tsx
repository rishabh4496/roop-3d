// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import FaceOverlayCanvas from './components/canvas/FaceOverlayCanvas';
import { LANDMARK_COLORS, drawPoseHud, facePose } from './components/canvas/poseHud';
import { calculateLetterbox, mapBoxToPixels, mapPointToPixels } from './utils/letterbox';
import { formatPose, leftToRightRank, poseCellName, toStandardPose } from './utils/poseCells';
import { useFaceSelectionStore } from './useFaceSelectionStore';

describe('pose helpers (standard frame: yaw + = nose right, pitch + = looking up)', () => {
  it('converts the server\'s legacy pitch (+ = down) and leaves standard poses alone', () => {
    expect(toStandardPose({ yaw: 10, pitch: 5, roll: 1 })).toEqual({ yaw: 10, pitch: -5, roll: 1 });
    expect(toStandardPose({ yaw: 10, pitch: 5, roll: 1 }, 'standard')).toEqual({ yaw: 10, pitch: 5, roll: 1 });
    expect(toStandardPose(null)).toBeNull();
    expect(toStandardPose({ yaw: NaN, pitch: 0, roll: 0 })).toBeNull();
  });

  it('uses the same inclusive 3×3 cell edges as the backend (±15° yaw, ±10° pitch)', () => {
    const cell = (yaw: number, pitch: number) => poseCellName({ yaw, pitch, roll: 0 });
    expect(cell(0, 0)).toBe('center_level');
    expect([cell(-15, 0), cell(15, 0), cell(0, -10), cell(0, 10)]).toEqual(['center_level', 'center_level', 'center_level', 'center_level']);
    expect([cell(-15.01, 0), cell(15.01, 0), cell(0, -10.01), cell(0, 10.01)]).toEqual(['left_level', 'right_level', 'center_down', 'center_up']);
    expect([cell(28, -16), cell(-40, 25)]).toEqual(['right_down', 'left_up']);
  });

  it('formats a fixed-width signed badge and ranks faces left to right', () => {
    expect(formatPose({ yaw: 12.46, pitch: -3, roll: 0 })).toBe('Y +12.5°  P -3.0°  R +0.0°');
    const faces = [{ bbox: [0.6, 0, 0.8, 1] }, { bbox: [0.1, 0, 0.3, 1] }, { bbox: [0.35, 0, 0.5, 1] }];
    expect(faces.map(face => leftToRightRank(faces, face))).toEqual([2, 0, 1]);
  });

  it('prefers the server\'s pose3d and otherwise converts the legacy pose', () => {
    expect(facePose({ pose: { yaw: 5, pitch: 10, roll: 2 } })).toEqual({ yaw: 5, pitch: -10, roll: 2 });
    expect(facePose({ pose: { yaw: 5, pitch: 10, roll: 2 }, pose3d: { yaw: 5, pitch: 9, roll: 2 } })).toEqual({ yaw: 5, pitch: 9, roll: 2 });
    expect(facePose({})).toBeNull();
  });
});

const fakeContext = () => {
  const calls = { arc: 0, fills: [] as string[], texts: [] as string[] };
  const context: any = {
    save() {}, restore() {}, beginPath() {}, stroke() {}, fillRect() {},
    arc() { calls.arc++; }, fill() { calls.fills.push(context.fillStyle); },
    fillText(text: string) { calls.texts.push(text); }, measureText: (text: string) => ({ width: text.length * 6 }),
    fillStyle: '', strokeStyle: '', lineWidth: 1, font: '',
  };
  return { context, calls };
};
const LB = calculateLetterbox(800, 450, 1600, 900);
const hud = (faces: any[], options: Record<string, unknown> = {}) => {
  const { context, calls } = fakeContext();
  const drawn = drawPoseHud(context, faces, {
    mapPoint: point => mapPointToPixels(point, LB), mapBox: box => mapBoxToPixels(box, LB),
    bounds: { offsetX: LB.offsetX, offsetY: LB.offsetY, width: LB.displayW, height: LB.displayH }, ...options });
  return { drawn, calls };
};
const face = (id: string, x: number, pose: any, landmarks = true) => ({
  id, bbox: [x, 0.2, x + 0.2, 0.6], pose, ...(landmarks ? { landmarks5: [[x + .04, .3], [x + .16, .3], [x + .1, .4], [x + .05, .5], [x + .15, .5]] } : {}),
});

describe('3D pose & landmark HUD', () => {
  it('draws five anatomical landmark dots and a yaw/pitch/roll badge on EVERY detected face', () => {
    const { drawn, calls } = hud([face('a', 0.1, { yaw: 12, pitch: -3, roll: 1 }), face('b', 0.6, { yaw: -30, pitch: 14, roll: -2 })].map(f => ({ ...f, pose3d: f.pose })));
    expect(calls.arc).toBe(10);
    expect(calls.fills).toEqual([...LANDMARK_COLORS, ...LANDMARK_COLORS]);
    expect(calls.texts).toEqual(['Y +12.0°  P -3.0°  R +1.0°', 'Y -30.0°  P +14.0°  R -2.0°']);
    expect(drawn.map(item => item.kind)).toEqual(['landmarks', 'pose', 'landmarks', 'pose']);
  });

  it('honours the landmark and pose toggles independently', () => {
    const faces = [{ ...face('a', 0.1, { yaw: 1, pitch: 2, roll: 3 }) }];
    expect(hud(faces, { landmarks: false }).calls).toMatchObject({ arc: 0, texts: ['Y +1.0°  P -2.0°  R +3.0°'] });  // legacy pitch 2 -> standard -2
    expect(hud(faces, { pose: false }).calls).toMatchObject({ arc: 5, texts: [] });
    expect(hud(faces, { pose: false, landmarks: false }).drawn).toEqual([]);
  });

  it('never invents landmarks or a pose that the detector did not provide', () => {
    const { calls } = hud([face('no-landmarks', 0.1, { yaw: 0, pitch: 0, roll: 0 }, false), { id: 'bare', bbox: [0.5, 0.2, 0.7, 0.6] }]);
    expect(calls.arc).toBe(0);
    expect(calls.texts).toEqual(['Y +0.0°  P +0.0°  R +0.0°']);          // only the face that has a pose gets a badge
  });

  it('keeps the badge inside the media rectangle', () => {
    const { calls } = hud([face('edge', 0.8, { yaw: 0, pitch: 0, roll: 0 })]);
    expect(calls.texts).toHaveLength(1);
  });
});

describe('FaceOverlayCanvas right-click on a face box', () => {
  let host: HTMLDivElement, root: Root;
  const item = (id: string, track: string, x: number, extra: Record<string, unknown> = {}) => ({
    id, track_id: track, frame_index: 25, timestamp: 1, bbox: [x, 0.2, x + 0.2, 0.6] as [number, number, number, number],
    landmarks5: [[x + .04, .3], [x + .16, .3], [x + .1, .4], [x + .05, .5], [x + .15, .5]] as [number, number][], confidence: 0.98,
    pose: { yaw: 5, pitch: 10, roll: 2 }, thumbnail_base64: '', ...extra });
  const rightClick = (x: number, y: number) => {
    const canvas = host.querySelector('canvas')!;
    const event = new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: x, clientY: y });
    act(() => { canvas.dispatchEvent(event); });
    return event;
  };

  beforeEach(() => {
    (globalThis as any).IS_REACT_ACT_ENVIRONMENT = true;
    vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} });
    vi.spyOn(Element.prototype, 'getBoundingClientRect').mockReturnValue({ left: 0, top: 0, right: 800, bottom: 450, width: 800, height: 450, x: 0, y: 0, toJSON() {} } as DOMRect);
    useFaceSelectionStore.getState().reset();
    // Added right-hand face first so insertion order is NOT left-to-right.
    useFaceSelectionStore.getState().addFrameDetectedFaces([
      item('B', 'track_0001', 0.6, { pose3d: { yaw: 28, pitch: -16, roll: 0 } }),
      item('A', 'track_0000', 0.1),
    ], false);
    host = document.createElement('div'); document.body.append(host); root = createRoot(host);
  });
  afterEach(async () => { await act(async () => root.unmount()); host.remove(); vi.unstubAllGlobals(); vi.restoreAllMocks(); useFaceSelectionStore.getState().reset(); });

  const mount = (onFaceContextMenu: any) => act(async () => {
    root.render(<div style={{ position: 'relative' }}><FaceOverlayCanvas mediaDimensions={{ width: 1600, height: 900 }} currentTime={1} fps={25} onFaceContextMenu={onFaceContextMenu}/></div>);
  });

  it('reports the face\'s left-to-right rank, standard-frame pose and click position', async () => {
    const menu = vi.fn();
    await mount(menu);
    expect(rightClick(560, 180).defaultPrevented).toBe(true);              // inside face B (x 0.6-0.8 of 800 = 480-640)
    expect(menu).toHaveBeenLastCalledWith(expect.objectContaining({ faceRank: 1, pose: { yaw: 28, pitch: -16, roll: 0 }, x: 560, y: 180 }));
    rightClick(160, 180);                                                   // inside face A (80-240)
    expect(menu).toHaveBeenLastCalledWith(expect.objectContaining({ faceRank: 0, pose: { yaw: 5, pitch: -10, roll: 2 } }));   // legacy pitch +10 -> -10
    expect(menu).toHaveBeenCalledTimes(2);
  });

  it('does not open the menu when the click misses every face, and works without a handler', async () => {
    const menu = vi.fn();
    await mount(menu);
    expect(rightClick(400, 420).defaultPrevented).toBe(true);              // browser menu still suppressed
    expect(menu).not.toHaveBeenCalled();
    await mount(undefined);
    expect(() => rightClick(160, 180)).not.toThrow();                      // falls back to the original assignment popover
  });

  it('can be made visual-only so a comparison canvas above it receives the pointer', async () => {
    await act(async () => { root.render(<div><FaceOverlayCanvas mediaDimensions={{ width: 1600, height: 900 }} currentTime={1} fps={25} interactive={false}/></div>); });
    expect((host.querySelector('canvas') as HTMLCanvasElement).style.pointerEvents).toBe('none');
  });
});
