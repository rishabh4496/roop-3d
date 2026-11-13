// @vitest-environment jsdom
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import TrackingTelemetryPanel from './components/hud/TrackingTelemetryPanel';
import { ingestFrame, resetPipelineStream } from './stores/pipelineStream';
import { TRACKING_HISTORY, buildGraphGeometry, ingestTrackingFrame, parseTrackingFrame, resetTrackingStream, trackingStream } from './stores/trackingStream';

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

const frame = (index: number, patch: Record<string, unknown> = {}) => ({
  type: 'tracking.telemetry', job_id: 'job-1', frame_index: index, status: 'tracked', reason: 'OK', state: 'tracking', shot_cuts: 0, ...patch,
});

describe('parseTrackingFrame', () => {
  it('keeps unmeasured values null instead of inventing zeros', () => {
    const sample = parseTrackingFrame(frame(3, { yaw_deg: -72.5, jitter_px: null, stability_score: 7, quat_wxyz: [1, 0, 0] }), 10)!;
    expect(sample.yawDeg).toBe(-72.5);
    expect(sample.pitchDeg).toBeNull();
    expect(sample.jitterPx).toBeNull();
    expect(sample.stability).toBe(1);          // clamped to the 0..1 contract
    expect(sample.quat).toBeNull();            // wrong length is rejected, not truncated
    expect(sample.schemaVersion).toBe(1);
    expect(sample.flash).toBeNull();
  });
  it('recognizes the additive version 2 tracking contract', () => {
    const sample = parseTrackingFrame(frame(4, {
      schema_version: 2, quat_wxyz: [1, 0, 0, 0], translation_px: [2, 3], pose_scale_px_per_m: 4,
      translation_m: [0.12, -0.04, 0.73], reproj_rms_px: 1.25, pose_source: 'pnp_dense',
      spherical_bin: 'yaw+30/pitch-15', scene_flash: null,
    }))!;
    expect(sample.schemaVersion).toBe(2);
    expect(sample.quat).toEqual([1, 0, 0, 0]);
    expect(sample.translationPx).toEqual([2, 3]);
    expect(sample.scalePxPerM).toBe(4);
    expect(sample.translationM).toEqual([0.12, -0.04, 0.73]);
    expect(sample.reprojRmsPx).toBe(1.25);
    expect(sample.poseSource).toBe('pnp_dense');
    expect(sample.sphericalBin).toBe('yaw+30/pitch-15');
    expect(sample.flash).toBeNull();
  });

  it('rejects frames of other types and unusable indices', () => {
    expect(parseTrackingFrame({ type: 'pipeline.status' })).toBeNull();
    expect(parseTrackingFrame(frame(-1))).toBeNull();
    expect(parseTrackingFrame(null)).toBeNull();
  });
});

describe('trackingStream', () => {
  beforeEach(() => { resetTrackingStream(); resetPipelineStream(); });

  it('starts a fresh history for a new job or a rewind and replaces a resent frame', () => {
    ingestTrackingFrame(frame(0)); ingestTrackingFrame(frame(1)); ingestTrackingFrame(frame(1, { yaw_deg: 10 }));
    expect(trackingStream.get().history.map(sample => sample.frameIndex)).toEqual([0, 1]);
    expect(trackingStream.get().latest?.yawDeg).toBe(10);
    ingestTrackingFrame(frame(0));
    expect(trackingStream.get().history).toHaveLength(1);
    ingestTrackingFrame(frame(5, { job_id: 'job-2' }));
    expect(trackingStream.get().history).toHaveLength(1);
    expect(trackingStream.get().jobId).toBe('job-2');
  });

  it('bounds the history', () => {
    for (let index = 0; index < TRACKING_HISTORY + 50; index += 1) ingestTrackingFrame(frame(index));
    expect(trackingStream.get().history).toHaveLength(TRACKING_HISTORY);
    expect(trackingStream.get().history[0].frameIndex).toBe(50);
  });

  it('is reached through the shared pipeline socket handler', () => {
    ingestFrame(frame(2, { state: 'ended' }));
    expect(trackingStream.get().latest?.frameIndex).toBe(2);
    expect(trackingStream.get().ended).toBe(true);
  });
});

describe('buildGraphGeometry', () => {
  it('breaks the line at unmeasured samples and marks cuts and flashes', () => {
    const samples = [frame(0, { jitter_px: 0.5, stability_score: 1 }), frame(1, { jitter_px: null, stability_score: null }),
      frame(2, { jitter_px: 2, stability_score: 0.5, shot_cut: true }), frame(3, { jitter_px: 1, stability_score: 0.75, scene_flash: true })]
      .map(raw => parseTrackingFrame(raw)!);
    const graph = buildGraphGeometry(samples, 90, 40);
    expect(graph.jitter.split(' | ')).toHaveLength(2);
    expect(graph.cuts).toEqual([60]);
    expect(graph.flashes).toEqual([90]);
    expect(graph.jitterMax).toBe(2);
    // stability 1.0 is the top of the graph, 0.5 the middle
    expect(graph.stability.split(' | ')[0]).toBe('0.0,0.0');
    expect(graph.stability.split(' | ')[1].startsWith('60.0,20.0')).toBe(true);
  });
});

describe('TrackingTelemetryPanel', () => {
  let host: HTMLDivElement; let root: Root;
  beforeEach(() => { resetTrackingStream(); host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
  afterEach(() => { act(() => root.unmount()); host.remove(); });

  it('says so when no tracking frames have arrived', () => {
    act(() => root.render(<TrackingTelemetryPanel />));
    expect(host.querySelector('[data-testid="tracking-telemetry"]')?.getAttribute('data-state')).toBe('idle');
    expect(host.textContent).toContain('No tracking frames yet');
  });

  it('shows 6DoF badges, the pose source and the reason a head is not followed; unmeasured values read n/a', () => {
    act(() => root.render(<TrackingTelemetryPanel />));
    act(() => {
      ingestTrackingFrame(frame(1, { yaw_deg: 78.25, pitch_deg: -4, roll_deg: 1.5, quat_wxyz: [0.8, 0.1, 0.6, 0.0], translation_m: [0.01, -0.02, 0.62],
        pose_source: 'pnp_dense', spherical_bin: 'yaw+75/pitch0', reproj_rms_px: 0.84, stability_score: 0.9, jitter_px: 0.3, shot_cuts: 2 }));
      ingestTrackingFrame(frame(2, { status: 'lost', reason: 'YAW_BEYOND_LIMIT', yaw_deg: 91, shot_cuts: 2 }));
    });
    const text = (id: string) => host.querySelector(`[data-testid="${id}"]`)?.textContent ?? '';
    expect(text('trk-yaw')).toContain('+91.0°');
    expect(text('trk-status')).toContain('Head not followed');
    expect(host.querySelector('[data-testid="trk-reason"]')?.textContent).toBe('YAW_BEYOND_LIMIT');
    expect(text('trk-source')).toContain('n/a');          // the lost frame carried no pose source
    expect(text('trk-cuts')).toContain('Shot cuts 2');
    act(() => { ingestTrackingFrame(frame(3, { yaw_deg: 78.25, translation_m: [0.01, -0.02, 0.62], pose_source: 'pnp_dense', spherical_bin: 'yaw+75/pitch0', reproj_rms_px: 0.84 })); });
    expect(text('trk-position')).toContain('0.620 m');
    expect(text('trk-source')).toContain('Dense-landmark PnP');
    expect(text('trk-bin')).toContain('yaw+75/pitch0');
    expect(text('trk-reproj')).toContain('0.84 px');
  });
});
