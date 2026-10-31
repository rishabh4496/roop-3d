// @vitest-environment jsdom
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { beforeEach, afterEach, expect, it, vi } from 'vitest';
import useHistoryStore, { executeCommand, editCurves } from './stores/useHistoryStore';
import { evaluateCurve, evaluateParameters } from './stores/parameterCurves';
import { useFaceSelectionStore, buildFaceSegments, hydrateFaceSegments } from './useFaceSelectionStore';
import CurveEditorTrack from './components/timeline/CurveEditorTrack';
import SnapshotSlots, { SnapshotViewport } from './components/workstation/SnapshotSlots';
import FaceOverlayCanvas from './components/canvas/FaceOverlayCanvas';
import type { DetectedFace } from './contracts';

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
let host: HTMLDivElement, root: Root;
beforeEach(() => { useHistoryStore.getState().reset(); useFaceSelectionStore.getState().reset(); host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
afterEach(async () => { await act(async () => root.unmount()); host.remove(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });
const click = async (element: Element | null) => { expect(element).toBeTruthy(); await act(async () => (element as HTMLElement).click()); };
const faces = (): DetectedFace[] => [0, 10, 20].map(frameIndex => ({ id: `f${frameIndex}`, frameIndex, timestamp: frameIndex / 25, bbox: [.1, .1, .5, .5], clusterId: 'track_0000', pose: { yaw: 12, pitch: 0, roll: 0 }, embedding: [1, 0] }));

it('uses reversible commands, discards redo branches, and coalesces a parameter gesture', () => {
  let value = 0;
  executeCommand({ label: 'Parameter', mergeKey: 'one', undo: () => { value = 0; }, redo: () => { value = 1; } });
  executeCommand({ label: 'Parameter', mergeKey: 'one', undo: () => { value = 1; }, redo: () => { value = 2; } });
  expect(useHistoryStore.getState().past).toHaveLength(1); useHistoryStore.getState().undo(); expect(value).toBe(0);
  useHistoryStore.getState().redo(); expect(value).toBe(2);
  useHistoryStore.getState().undo(); executeCommand({ label: 'New branch', undo: () => { value = 0; }, redo: () => { value = 5; } });
  expect(useHistoryStore.getState().future).toHaveLength(0); expect(value).toBe(5);
});

it('evaluates linear, Bezier and hold curves at exact source frame boundaries', () => {
  const keys = [{ frame: 0, value: 0, easing: 'linear' }, { frame: 100, value: 1, easing: 'linear' }];
  expect(evaluateCurve(keys, 25, 5)).toBe(.25);
  expect(evaluateCurve([{ ...keys[0], easing: 'ease-in-out' }, keys[1]], 25, 5)).toBe(.15625);
  expect(evaluateCurve([{ ...keys[0], easing: 'hold' }, keys[1]], 99, 5)).toBe(0);
  expect(evaluateCurve([{ ...keys[0], easing: 'hold' }, keys[1]], 100, 5)).toBe(1);
  expect(evaluateParameters({ swap_weight: 1 }, { swap_weight: keys }, 50).swap_weight).toBe(.5);
});

it('edits and deletes timeline keys through undoable commands', async () => {
  await act(async () => root.render(<CurveEditorTrack frame={20} totalFrames={100} parameters={{ swap_weight: .6 }}/>));
  await click([...host.querySelectorAll('button')].find(button => button.textContent === 'Add key at playhead')!);
  expect(useHistoryStore.getState().curves.swap_weight[0]).toEqual({ frame: 20, value: .6, easing: 'linear' });
  await click([...host.querySelectorAll('button')].find(button => button.textContent === 'Delete key')!);
  expect(useHistoryStore.getState().curves.swap_weight).toHaveLength(0);
  await act(async () => useHistoryStore.getState().undo()); expect(host.querySelectorAll('circle')).toHaveLength(1);
  await act(async () => useHistoryStore.getState().undo()); expect(host.querySelectorAll('circle')).toHaveLength(0);
});

it('labels and keys the actual 2.5D / 3D blend when mesh mode is selected', async () => {
  await act(async () => root.render(<CurveEditorTrack frame={10} totalFrames={100} parameters={{ pose_mode: 'mesh_blend', pose_strength: .6 }}/>));
  const select = host.querySelector<HTMLSelectElement>('[aria-label="Curve parameter"]')!;
  await act(async () => { Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value')!.set!.call(select, 'pose_correction_strength'); select.dispatchEvent(new Event('change', { bubbles: true })); });
  expect(host.textContent).toContain('0 = TPS 2.5D warp'); expect(host.textContent).toContain('1 = perspective 3D mesh projection');
  expect(host.textContent).not.toContain('requires a configured 3D projection adapter');
  await click([...host.querySelectorAll('button')].find(button => button.textContent === 'Add key at playhead')!);
  expect(useHistoryStore.getState().curves.pose_correction_strength[0]).toMatchObject({ frame: 10, value: .6 });
});

it('blocks assignments and automatic merges for exclusions and reverses the role change', () => {
  const store = useFaceSelectionStore.getState(); store.setDetectedFaces(faces());
  store.assignSource('track_0000', 'alice'); store.setClusterMode('track_0000', 'exclude'); store.assignSource('track_0000', 'bob');
  expect(useFaceSelectionStore.getState().assignments.track_0000).toBeUndefined();
  store.addDetectedFaces([{ ...faces()[0], id: 'other', clusterId: 'track_0001' }]); store.computeClusters();
  expect(useFaceSelectionStore.getState().targetClusters).toHaveLength(2);
  expect(useFaceSelectionStore.getState().targetClusters.find(item => item.id === 'track_0000')!.mode).toBe('exclude');
  // Return to the role command's exact prior state, including its source mapping.
  useHistoryStore.getState().undo(); expect(useFaceSelectionStore.getState().assignments.track_0000).toBe('alice');
});

it('splits only from the selected frame onward and persists explicit intervals', () => {
  const store = useFaceSelectionStore.getState(); store.setDetectedFaces(faces()); store.assignSource('track_0000', 'alice');
  expect(store.splitCluster('track_0000', 10)).toBe(true);
  const state = useFaceSelectionStore.getState(), tail = state.targetClusters.find(cluster => cluster.id !== 'track_0000')!;
  expect(state.detectedFaces.find(face => face.frameIndex === 0)!.clusterId).toBe('track_0000');
  expect(state.detectedFaces.find(face => face.frameIndex === 10)!.clusterId).toBe(tail.id);
  store.setClusterMode(tail.id, 'exclude');
  const segments = buildFaceSegments(useFaceSelectionStore.getState().targetClusters, useFaceSelectionStore.getState().assignments);
  expect(segments.find(segment => segment.cluster_id === tail.id)).toMatchObject({ track_id: 'track_0000', start_frame: 10, role: 'exclude', faceset_id: null });
  store.reset(); store.setDetectedFaces(faces()); hydrateFaceSegments(segments);
  expect(useFaceSelectionStore.getState().targetClusters.find(cluster => cluster.id === tail.id)!.mode).toBe('exclude');
});

it('captures immutable slot state and compares any two populated slots with a real image wipe', async () => {
  const history = useHistoryStore.getState(), workspace = { parameters: { swap_weight: .5 }, curves: {} };
  history.captureSlot('A', { image: 'data:image/jpeg;base64,AQ==', frame: 1, workspace });
  history.captureSlot('C', { image: 'data:image/jpeg;base64,Ag==', frame: 2, workspace });
  workspace.parameters.swap_weight = 1;
  expect(useHistoryStore.getState().slots.A.workspace.parameters.swap_weight).toBe(.5);
  history.compare('A', 'C');
  await act(async () => root.render(<SnapshotViewport/>));
  expect(host.querySelectorAll('img')).toHaveLength(2);
  expect(host.querySelector('[aria-label="Snapshot A versus C"]')).toBeTruthy();
  expect(host.querySelectorAll('img')[1].style.clipPath).toContain('50%');
  await act(async () => history.live()); expect(host.querySelector('img')).toBeNull();
});

it('captures empty slots and selects populated slots without silently restoring settings', async () => {
  const capture = vi.fn(), restore = vi.fn();
  await act(async () => root.render(<SnapshotSlots onCapture={capture} onRestore={restore} captureAvailable/>));
  await click(host.querySelector('[aria-label="Capture snapshot A"]')); expect(capture).toHaveBeenCalledWith('A');
  await act(async () => useHistoryStore.getState().captureSlot('A', { image: 'data:image/jpeg;base64,AQ==', frame: 1, workspace: { marker: 1 } }));
  await click(host.querySelector('[aria-label="View snapshot A"]')); expect(restore).not.toHaveBeenCalled();
  await click([...host.querySelectorAll('button')].find(button => button.textContent === 'Restore slot settings')!); expect(restore).toHaveBeenCalledWith({ marker: 1 });
});

it('opens a right-click face menu with explicit roles, source views, camera focus and splitting', async () => {
  useFaceSelectionStore.getState().setDetectedFaces(faces());
  const rect = { x: 0, y: 0, left: 0, top: 0, right: 400, bottom: 200, width: 400, height: 200, toJSON() {} };
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue(rect);
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
  await act(async () => root.render(<FaceOverlayCanvas currentTime={.4} fps={25} mediaDimensions={{ width: 400, height: 200 }}/>));
  await act(async () => host.querySelector('canvas')!.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, clientX: 80, clientY: 40, button: 2 })));
  expect(host.querySelector('[aria-label="Face context menu"]')).toBeTruthy();
  for (const label of ['Swap Target', 'Reference Guide', 'Exclude (Never Swap)', 'Re-center pose camera', 'Split cluster from current frame']) expect(host.textContent).toContain(label);
  await click([...host.querySelectorAll('button')].find(button => button.textContent === 'Exclude (Never Swap)')!);
  expect(useFaceSelectionStore.getState().targetClusters[0].mode).toBe('exclude');
  await click([...host.querySelectorAll('button')].find(button => button.textContent === 'Re-center pose camera')!);
  expect(useFaceSelectionStore.getState().poseFocus?.faceId).toBe('f10');
});
