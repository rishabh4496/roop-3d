// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import QueueDrawer from './components/queue/QueueDrawer';
import { fileName, normalizeFaceSets, normalizeJob, normalizeJobs, normalizeTelemetry, shortId } from './utils/safeData';

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe('fileName / shortId', () => {
  it('takes the last segment for either separator and survives non-strings', () => {
    expect(fileName('C:\\clips\\take 2.mp4')).toBe('take 2.mp4');
    expect(fileName('/home/me/clip.mp4')).toBe('clip.mp4');
    expect(fileName('a//b/')).toBe('b');
    for (const bad of [undefined, null, 42, '', {}]) expect(fileName(bad, 'fallback')).toBe('fallback');
    expect(shortId('abcdef0123456789')).toBe('abcdef01');
    expect(shortId(undefined)).toBe('········');
  });
});

describe('normalizeJob(s)', () => {
  it('fills every object the list renderers dereference', () => {
    const job = normalizeJob({ job_id: 'j1' })!;
    expect(job).toMatchObject({ job_id: 'j1', state: 'queued', request: { target_path: '', face_mappings: [] }, progress: { frame: 0, total_frames: null, fps: 0, stage: '' }, output: null, failure: null });
  });

  it('cleans numbers and keeps real data', () => {
    const job = normalizeJob({ job_id: 'j2', state: 'running', request: { target_path: 'a.mp4', face_mappings: [{ track_id: 't', faceset_id: 'f' }, 'junk'], extra: 1 },
      progress: { frame: -4, total_frames: 0, fps: Number.NaN, stage: 'swap', stage_timings_ms: { swap: 3 } }, output: { output_path: 'o.mp4', frames: '12' }, failure: { code: 'X' } })!;
    expect(job.request).toMatchObject({ target_path: 'a.mp4', extra: 1, face_mappings: [{ track_id: 't', faceset_id: 'f' }] });
    expect(job.progress).toMatchObject({ frame: 0, total_frames: null, fps: 0, stage: 'swap', stage_timings_ms: { swap: 3 } });
    expect(job.output).toMatchObject({ output_path: 'o.mp4', frames: 0 });
    expect(job.failure!.message).toBe('The render failed.');
  });

  it('drops records that are not jobs instead of throwing', () => {
    expect(normalizeJobs([{ job_id: 'ok' }, null, 'x', {}, { job_id: '' }, { job_id: 5 }, []]).map(job => job.job_id)).toEqual(['ok']);
    expect(normalizeJobs({ error: 'proxy page' })).toEqual([]);
    expect(normalizeJobs(null)).toEqual([]);
  });
});

describe('normalizeFaceSets / normalizeTelemetry', () => {
  it('guarantees a name and a count, drops id-less entries', () => {
    expect(normalizeFaceSets([{ faceset_id: 'a' }, { faceset_id: 'b', name: 'Bea', item_count: 3 }, { name: 'no id' }, null, { faceset_id: 'c', name: 7, item_count: '9' }]))
      .toEqual([{ faceset_id: 'a', name: 'a', item_count: 0 }, { faceset_id: 'b', name: 'Bea', item_count: 3 }, { faceset_id: 'c', name: 'c', item_count: 0 }]);
    expect(normalizeFaceSets('nope')).toEqual([]);
  });

  it('always yields a gpus array', () => {
    expect(normalizeTelemetry({})).toEqual({ gpus: [], logical_cpu_count: 0 });
    expect(normalizeTelemetry({ gpus: [{ name: 'RTX', total_vram_mib: 8192, free_vram_mib: 'x' }, null], logical_cpu_count: 16 }))
      .toEqual({ gpus: [{ name: 'RTX', total_vram_mib: 8192, free_vram_mib: 0 }], logical_cpu_count: 16 });
    expect(normalizeTelemetry(null)).toBeNull();
    expect(normalizeTelemetry('x')).toBeNull();
  });
});

describe('QueueDrawer with incomplete job records', () => {
  let host: HTMLDivElement, root: Root;
  beforeEach(() => { host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
  afterEach(() => { act(() => root.unmount()); host.remove(); });

  it('renders every row, never throws, and shows Windows paths by file name', async () => {
    const broken = [
      { job_id: 'abcdef0123', state: 'running' },                                                          // no request / progress at all
      { job_id: 'bbbbbbbb99', state: 'succeeded', request: { target_path: 'C:\\clips\\take 2.mp4' }, output: null },
      { job_id: 'cccccccc11', state: 'failed', request: { target_path: '/a/b.mp4' }, progress: { frame: 5 }, failure: { message: 'boom' } },
    ];
    await act(async () => { root.render(<QueueDrawer jobs={broken as never} isOpen onToggleOpen={() => undefined} />); });
    const cards = host.querySelectorAll('.queue-job-card');
    expect(cards).toHaveLength(3);
    expect(host.textContent).toContain('take 2.mp4');
    expect(host.textContent).not.toContain('C:\\clips');
    expect(host.textContent).toContain('Untitled render');
    expect(host.textContent).toContain('Frame 5');
    expect(host.textContent).not.toMatch(/NaN|undefined/);
  });

  it('a duplicate of a job without a request reports why instead of throwing', async () => {
    await act(async () => { root.render(<QueueDrawer jobs={[{ job_id: 'abcdef0123', state: 'queued' }] as never} isOpen onToggleOpen={() => undefined} />); });
    await act(async () => { ([...host.querySelectorAll('button')].find(b => b.textContent?.includes('Duplicate')) as HTMLButtonElement).click(); await Promise.resolve(); });
    expect(host.querySelector('.queue-notice-strip')!.textContent).toContain('Duplication failed: This job has no saved request to copy.');
  });
});
