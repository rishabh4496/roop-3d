// @vitest-environment jsdom
import { act, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { describe, expect, it, vi } from 'vitest';
import AngularCoverageRadar from './components/three/AngularCoverageRadar';
import IdentityMorphPanel from './components/three/IdentityMorphPanel';
import { lightDirection, lightingRig, morphRig } from './components/three/PoseGeometryViewer';
import { viewerLighting, viewerMorph } from './components/three/viewerRigs';
import { DEFAULT_PARAMETERS, type WorkstationParameters } from './components/workstation/pipeline';
import {
  CELL_COUNT, cellCounts, cellIndex, cellPolygon, coverageGaps, missingProfiles, polarPoint, summarizeRadar, targetCoverage, targetPoseSamples,
} from './utils/coverageRadar';

const view = (yaw: number, pitch = 0) => ({ yaw, pitch });
const FRONT_ONLY = [view(-5), view(5), view(-20, 10), view(20, -10)];
const FULL_ARC = [-80, -65, -50, -35, -20, -5, 5, 20, 35, 50, 65, 80].flatMap(yaw => [-50, -20, 0, 20, 50].map(pitch => view(yaw, pitch)));

describe('radar grid model', () => {
  it('bins on the exact backend 12 x 8 grid, clamped at the edges', () => {
    expect(CELL_COUNT).toBe(96);
    expect(cellIndex(view(-90, -60))).toBe(0);
    expect(cellIndex(view(90, 60))).toBe(95);
    expect(cellIndex(view(0, 0))).toBe(4 * 12 + 6);          // yaw 0 starts column 6, pitch 0 starts row 4
    expect(cellIndex(view(-200, 99))).toBe(7 * 12);          // out of range clamps to the nearest edge cell
  });

  it('prefers the server bin_index over re-binning, and ignores malformed entries', () => {
    const counts = cellCounts({ observed_pose_points: [{ bin_index: 5 }, { bin_index: 5 }, { bin_index: 200 }, { bin_index: -1 }, { bin_index: 1.5 }] });
    expect(counts[5]).toBe(2);
    expect(counts.reduce((a, b) => a + b, 0)).toBe(2);
    const fromViews = cellCounts({ views: [view(-80, -55), { yaw: Number.NaN, pitch: 0 }] });
    expect(fromViews[0]).toBe(1);
    expect(fromViews.reduce((a, b) => a + b, 0)).toBe(1);
    expect(cellCounts(null).every(count => count === 0)).toBe(true);
  });

  it('maps the yaw x pitch rectangle onto a unit disc: centre is frontal, the rim is the extremes, +pitch is drawn up', () => {
    expect(polarPoint(0, 0)).toEqual({ x: 0, y: 0 });
    const right = polarPoint(90, 0), up = polarPoint(0, 60), corner = polarPoint(90, 60), down = polarPoint(0, -60);
    expect(right.x).toBeCloseTo(1); expect(right.y).toBeCloseTo(0);
    expect(up.y).toBeCloseTo(-1); expect(down.y).toBeCloseTo(1);            // SVG y grows downward
    expect(Math.hypot(corner.x, corner.y)).toBeCloseTo(1);                   // even the corner stays inside the disc
    for (const yaw of [-90, -45, 0, 45, 90]) for (const pitch of [-60, -30, 0, 30, 60]) expect(Math.hypot(polarPoint(yaw, pitch).x, polarPoint(yaw, pitch).y)).toBeLessThanOrEqual(1 + 1e-9);
    expect(cellPolygon(0, 0).length).toBe(16);
    expect(cellPolygon(5, 3, 1).length).toBe(4);
  });
});

describe('coverage gaps and the pre-render verdict', () => {
  it('flags a FaceSet that only has frontal views as missing both profiles and the three-quarter views', () => {
    const counts = cellCounts({ views: FRONT_ONLY });
    const missing = missingProfiles(counts).map(band => band.id);
    expect(missing).toEqual(['left_profile', 'right_profile']);
    const level = coverageGaps(counts).filter(gap => gap.pitch.id === 'level');
    expect(level.find(gap => gap.yaw.id === 'front')?.state).not.toBe('missing');
    expect(level.find(gap => gap.yaw.id === 'left_34')?.state).toBe('missing');
    const summary = summarizeRadar(counts);
    expect(summary.ready).toBe(false);
    expect(summary.message).toContain('left profile or right profile');
  });

  it('reports a complete arc as ready, with no profile gap', () => {
    const summary = summarizeRadar(cellCounts({ views: FULL_ARC }));
    expect(summary.profiles).toEqual([]);
    expect(summary.ready).toBe(true);
    expect(summary.percent).toBeGreaterThan(50);
  });

  it('treats a target pose as served when its own or a neighbouring cell has a view, and names the ones that are not', () => {
    const counts = cellCounts({ views: [view(0, 0), view(-30, 0)] });
    const served = targetCoverage(counts, [view(7, 0), view(-40, 5), view(70, 0), view(200, 0)]);
    expect(served.map(item => item.covered)).toEqual([true, true, false]);       // yaw 200 is out of range and dropped
    const summary = summarizeRadar(counts, [view(70, 0), view(75, 10)]);
    expect(summary.uncoveredTargets).toHaveLength(2);
    expect(summary.message).toContain('The target turns to 70°…75° yaw');
    expect(summary.ready).toBe(false);
  });

  it('says so when the FaceSet has no samples at all', () => {
    expect(summarizeRadar(new Array(CELL_COUNT).fill(0)).message).toBe('This FaceSet has no pose samples yet.');
  });

  it('converts the stored (pitch + = down) track and face poses to the standard frame, de-duplicated', () => {
    const poses = targetPoseSamples({ yaw_degrees: [-10, 50], pitch_degrees: [-4, 20] }, [{ pose: { yaw: 20, pitch: -8 } }, { pose: { yaw: 20.2, pitch: -8.1 } }, { pose: null }]);
    expect(poses).toContainEqual({ yaw: 20, pitch: 8 });                           // legacy pitch -8 (looking up) is +8 in the standard frame
    expect(poses.filter(pose => pose.yaw === 20 && pose.pitch === 8)).toHaveLength(1);
    expect(poses).toContainEqual({ yaw: 50, pitch: -8 });                          // range mid pitch (20 + -4)/2 = 8 legacy -> -8
    expect(targetPoseSamples(null)).toEqual([]);
  });
});

describe('AngularCoverageRadar', () => {
  it('draws one cell per bin, hatches the missing profile cells and lists the gaps', async () => {
    Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
    const host = document.createElement('div'); document.body.append(host); const root = createRoot(host);
    try {
      await act(async () => root.render(<AngularCoverageRadar coverage={{ views: FRONT_ONLY }} targetPoses={[view(75, 0), view(0, 0)]} livePose={view(10, 5)} name="harjot" />));
      expect(host.querySelectorAll('.radar-cell')).toHaveLength(96);
      expect(host.querySelector('svg')?.getAttribute('aria-label')).toContain('Missing left profile and right profile');
      expect(host.querySelectorAll('.radar-cell.filled').length).toBe(4);
      expect(host.querySelectorAll('.radar-cell.profile-gap').length).toBeGreaterThan(0);
      const targets = [...host.querySelectorAll('.radar-target')];
      expect(targets.map(node => node.classList.contains('uncovered'))).toEqual([true, false]);   // yaw 75 has no nearby view, yaw 0 does
      expect(host.querySelector('.radar-live')).toBeTruthy();
      expect(host.querySelector('.radar-badge')?.classList.contains('warn')).toBe(true);
      expect(host.querySelector('[role="status"]')?.textContent).toContain('No left profile or right profile view.');
      expect([...host.querySelectorAll('.coverage-gap-list li.profile')].length).toBeGreaterThan(0);
    } finally { await act(async () => root.unmount()); host.remove(); }
  });

  it('shows the empty state without inventing coverage', async () => {
    Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
    const host = document.createElement('div'); document.body.append(host); const root = createRoot(host);
    try {
      await act(async () => root.render(<AngularCoverageRadar coverage={null} />));
      expect(host.querySelector('.radar-badge')?.textContent).toBe('no views');
      expect(host.querySelectorAll('.radar-cell.filled')).toHaveLength(0);
      expect(host.textContent).toContain('Load a FaceSet to see which head angles it covers.');
      expect(host.querySelector('.coverage-gap-list')).toBeNull();
    } finally { await act(async () => root.unmount()); host.remove(); }
  });

  it('marks a complete arc as ready', async () => {
    Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
    const host = document.createElement('div'); document.body.append(host); const root = createRoot(host);
    try {
      await act(async () => root.render(<AngularCoverageRadar coverage={{ views: FULL_ARC }} targetPoses={[view(60, 10)]} compact />));
      expect(host.querySelector('.radar-badge')?.classList.contains('ok')).toBe(true);
      expect(host.querySelector('.radar-target')?.classList.contains('uncovered')).toBe(false);
      expect(host.querySelector('.coverage-gap-list')).toBeNull();                     // compact: no list
    } finally { await act(async () => root.unmount()); host.remove(); }
  });
});

describe('viewer rigs', () => {
  it('places the key light by yaw / pitch in the viewer frame', () => {
    const [x, y, z] = lightDirection(0, 0);
    expect([x, y, z]).toEqual([0, 0, 1]);
    const right = lightDirection(90, 0), up = lightDirection(0, 90);
    expect(right[0]).toBeCloseTo(1); expect(up[1]).toBeCloseTo(1);
    expect(Math.hypot(...lightDirection(35, 25))).toBeCloseTo(1);
  });

  it('turns the relight controls into key strength, ambient fill and a glossier skin, monotonically', () => {
    const low = lightingRig({ strength: 0, ambient: 0, specular: 0 }), high = lightingRig({ strength: 1, ambient: 1, specular: 1 });
    expect(high.keyIntensity).toBeGreaterThan(low.keyIntensity);
    expect(high.ambientIntensity).toBeGreaterThan(low.ambientIntensity);
    expect(high.roughness).toBeLessThan(low.roughness);
    expect(high.specularIntensity).toBeGreaterThan(low.specularIntensity);
    expect(lightingRig({ strength: Number.NaN, ambient: -3, specular: 9 }).ambientIntensity).toBeCloseTo(0.1);   // sanitised, never NaN
    expect(lightingRig(null).keyPosition).toHaveLength(3);
  });

  it('fades the target-geometry ghost out as the source mesh takes over, and clamps the hairline shift', () => {
    expect(morphRig({ blend: 0 }).targetGhostOpacity).toBeGreaterThan(morphRig({ blend: 0.6 }).targetGhostOpacity);
    expect(morphRig({ blend: 1 }).targetGhostOpacity).toBe(0);
    expect(morphRig({ hairline: 5 }).hairlineShift).toBe(1);
    expect(morphRig({ hairline: Number.NaN }).hairlineShift).toBe(0);
    expect(morphRig(null).blend).toBe(1);
  });

  it('derives the viewer lighting only while relighting is on, and uses the engine\'s own blend for the morph', () => {
    expect(viewerLighting(DEFAULT_PARAMETERS)).toBeNull();
    expect(viewerLighting({ ...DEFAULT_PARAMETERS, relight: true, lighting_mode: 'manual', manual_light_yaw: -40, manual_light_pitch: 10, ambient_fill: 0.7 }))
      .toMatchObject({ yaw: -40, pitch: 10, ambient: 0.7, manual: true });
    expect(viewerLighting({ ...DEFAULT_PARAMETERS, relight: true, lighting_mode: 'auto', manual_light_yaw: -40 })).toMatchObject({ yaw: 35, manual: false });
    expect(viewerMorph(DEFAULT_PARAMETERS, false)).toBeNull();
    expect(viewerMorph({ ...DEFAULT_PARAMETERS, pose_strength: 0.8, hairline_offset_px: -16 }, true)).toEqual({ blend: 0.8, hairline: -0.5 });
    expect(viewerMorph({ ...DEFAULT_PARAMETERS, pose_mode: 'retarget' }, true)?.blend).toBe(1);
  });
});

describe('IdentityMorphPanel', () => {
  function Harness({ onChange }: { onChange: (value: WorkstationParameters) => void }) {
    const [parameters, setParameters] = useState<WorkstationParameters>(DEFAULT_PARAMETERS);
    const [preview, setPreview] = useState(false);
    return <IdentityMorphPanel parameters={parameters} setParameters={value => { onChange(value); setParameters(value); }} preview={preview} onPreview={setPreview} />;
  }
  const change = async (input: HTMLInputElement, value: string) => act(async () => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(input, value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });

  it('writes the engine\'s mesh blend and omits the hairline offset at zero so untouched requests are unchanged', async () => {
    Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
    const seen: WorkstationParameters[] = [];
    const onChange = vi.fn((value: WorkstationParameters) => { seen.push(value); });
    const host = document.createElement('div'); document.body.append(host); const root = createRoot(host);
    try {
      await act(async () => root.render(<Harness onChange={onChange} />));
      const blend = host.querySelector<HTMLInputElement>('[aria-label="Jawline and cranial volume blend"]')!;
      const offset = host.querySelector<HTMLInputElement>('[aria-label="Forehead and hairline boundary offset"]')!;
      expect(blend.value).toBe('35');
      await change(blend, '80');
      expect(seen.at(-1)?.pose_strength).toBeCloseTo(0.8);
      await change(offset, '12');
      expect(seen.at(-1)?.hairline_offset_px).toBe(12);
      await change(offset, '0');
      expect('hairline_offset_px' in seen.at(-1)!).toBe(false);
      expect(host.querySelector('output')?.textContent).toBe('80%');
    } finally { await act(async () => root.unmount()); host.remove(); }
  });

  it('disables the blend and offers mesh blending when pose correction is set to retarget', async () => {
    Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
    const seen: WorkstationParameters[] = [];
    const host = document.createElement('div'); document.body.append(host); const root = createRoot(host);
    try {
      await act(async () => root.render(<IdentityMorphPanel parameters={{ ...DEFAULT_PARAMETERS, pose_mode: 'retarget' }} setParameters={value => seen.push(value)} preview={false} onPreview={() => {}} />));
      expect(host.querySelector<HTMLInputElement>('[aria-label="Jawline and cranial volume blend"]')!.disabled).toBe(true);
      await act(async () => host.querySelector<HTMLButtonElement>('.identity-warn button')!.click());
      expect(seen.at(-1)?.pose_mode).toBe('mesh_blend');
    } finally { await act(async () => root.unmount()); host.remove(); }
  });
});
