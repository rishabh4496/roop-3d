import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { expect, test, type Page, type Route, type WebSocketRoute } from '@playwright/test';
import { boot } from './helpers/mockApi';

/**
 * Stage 17, in a real browser with real video: when a job finishes the Before / After player opens by itself, loads
 * /api/v1/jobs/{id}/download beside the original, plays both in sync from one transport, wipes, and offers "Download MP4".
 * The "original" is a test pattern and the "result" is the same pattern colour-inverted WITH audio, so a wrong layer or a missing
 * clip is visible in pixels, not only in attributes. Videos are cut with ffmpeg (skipped when it is not installed).
 */
const FFMPEG = [process.env.FFMPEG, 'G:/pinokio/bin/miniforge/Library/bin/ffmpeg.exe', 'ffmpeg'].find(candidate => {
  if (!candidate) return false;
  try { execFileSync(candidate, ['-version'], { stdio: 'ignore' }); return true; } catch { return false; }
});
const JOB_ID = 'c'.repeat(32), TARGET = 'roop_3d/.runtime/media/original.mp4';
const request = { target_path: TARGET, inspection_id: 'a'.repeat(32), face_mappings: [{ track_id: 'track_0001', faceset_id: 'alice' }], engine_id: 'photoreal_3d_ffmpeg' };
const job = (state: string) => ({ job_id: JOB_ID, state, request, effective_settings: null,
  progress: { frame: state === 'succeeded' ? 75 : 20, total_frames: 75, fps: 14, stage: 'render' },
  output: state === 'succeeded' ? { output_path: `roop_3d/.runtime/outputs/${JOB_ID}/result.mp4`, frames: 75, baked_video: true } : null, failure: null });

test.describe('Before / After output player', () => {
  test.skip(!FFMPEG, 'ffmpeg is not available');
  let dir: string, before: Buffer, after: Buffer;
  test.beforeAll(() => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), 'roop3d-compare-'));
    const make = (name: string, extra: string[], audio: boolean) => {
      const out = path.join(dir, name);
      execFileSync(FFMPEG!, ['-y', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=320x180:rate=25', ...(audio ? ['-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000'] : []),
        ...extra, '-t', '3', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-g', '5', ...(audio ? ['-c:a', 'aac'] : []), out]);
      return fs.readFileSync(out);
    };
    before = make('before.mp4', [], false);
    after = make('after.mp4', ['-vf', 'negate'], true);
  });
  test.afterAll(() => fs.rmSync(dir, { recursive: true, force: true }));

  /** Serve a file with HTTP Range support, as the real endpoint does (Chrome needs it to seek). */
  const serve = (bytes: Buffer, counter: { hits: number }) => async (route: Route) => {
    counter.hits += 1;
    const range = /bytes=(\d*)-(\d*)/.exec(route.request().headers().range ?? '');
    if (!range) return route.fulfill({ status: 200, body: bytes, headers: { 'Content-Type': 'video/mp4', 'Accept-Ranges': 'bytes', 'Content-Length': String(bytes.length) } });
    const start = Number(range[1] || 0), end = Math.min(bytes.length - 1, range[2] ? Number(range[2]) : bytes.length - 1);
    return route.fulfill({ status: 206, body: bytes.subarray(start, end + 1),
      headers: { 'Content-Type': 'video/mp4', 'Accept-Ranges': 'bytes', 'Content-Range': `bytes ${start}-${end}/${bytes.length}`, 'Content-Length': String(end - start + 1) } });
  };

  async function finishJob(page: Page) {
    const sockets: WebSocketRoute[] = [];
    let phase = 'running';
    await page.routeWebSocket('**/api/v1/ws/events', socket => { sockets.push(socket); });
    await page.route('**/api/v1/jobs', route => route.fulfill({ status: 200, json: [job(phase)] }));
    const hits = { before: { hits: 0 }, after: { hits: 0 } };
    await page.route(`**/api/v1/jobs/${JOB_ID}/download`, serve(after, hits.after));
    await page.route('**/api/v1/media/content**', serve(before, hits.before));
    await page.reload();
    await expect.poll(() => sockets.length).toBeGreaterThan(0);
    await expect(page.getByTestId('render-complete-modal')).toHaveCount(0);       // a running job never opens it
    phase = 'succeeded';
    sockets.at(-1)!.send(JSON.stringify({ type: 'job.state', job_id: JOB_ID, state: 'succeeded' }));
    return hits;
  }

  test('opens when the job finishes and loads both videos from the right URLs', async ({ page }) => {
    const { errors } = await boot(page, { route: '/#/faceswap', jobs: [job('running')] });
    const hits = await finishJob(page);
    const modal = page.getByTestId('render-complete-modal');
    await expect(modal).toBeVisible();
    await expect(modal.locator('video[aria-label="Completed render playback"]')).toHaveAttribute('src', `/api/v1/jobs/${JOB_ID}/download`);
    await expect(modal.locator('video[aria-label="Original video"]')).toHaveAttribute('src', /\/api\/v1\/media\/content\?path=roop_3d%2F\.runtime%2Fmedia%2Foriginal\.mp4$/);
    await expect.poll(() => page.evaluate(() => [...document.querySelectorAll('.ocp video')].map(video => (video as HTMLVideoElement).readyState))).toEqual([4, 4]);
    expect(hits.after.hits).toBeGreaterThan(0);
    expect(hits.before.hits).toBeGreaterThan(0);
    const link = modal.getByRole('link', { name: 'Download MP4' });
    await expect(link).toHaveAttribute('href', `/api/v1/jobs/${JOB_ID}/download`);
    await expect(link).toHaveAttribute('download', 'roop3d_cccccccc.mp4');
    expect(errors).toEqual([]);
  });

  test('both videos play, pause and seek in sync, and show different pixels (original vs result)', async ({ page }) => {
    await boot(page, { route: '/#/faceswap', jobs: [job('running')] });
    await finishJob(page);
    const modal = page.getByTestId('render-complete-modal');
    await expect(modal).toBeVisible();
    await expect.poll(() => page.evaluate(() => [...document.querySelectorAll('.ocp video')].every(video => (video as HTMLVideoElement).readyState >= 2))).toBe(true);
    const times = () => page.evaluate(() => ({ before: (document.querySelector('video[aria-label="Original video"]') as HTMLVideoElement).currentTime,
      after: (document.querySelector('video[aria-label="Completed render playback"]') as HTMLVideoElement).currentTime }));

    await modal.getByRole('button', { name: 'Play' }).click();
    await expect.poll(async () => (await times()).after, { timeout: 8000 }).toBeGreaterThan(1.0);
    const playing = await times();
    expect(Math.abs(playing.before - playing.after)).toBeLessThan(0.3);                  // one transport, both advance together
    await modal.getByRole('button', { name: 'Pause' }).click();
    // pausing re-aligns the original to the result's frame (a seek, so it settles a moment later): within one frame (1/25 s)
    await expect.poll(async () => { const t = await times(); return Math.abs(t.before - t.after); }).toBeLessThan(0.045);
    await page.waitForTimeout(500);                                                      // let the settling seek land
    const frozen = await times();
    await page.waitForTimeout(500);
    const still = await times();
    expect(still.after).toBe(frozen.after);                                              // both really stopped
    expect(still.before).toBe(frozen.before);
    expect(Math.abs(still.before - still.after)).toBeLessThan(0.045);                    // and sit on the same frame (to within one)

    // seek: the range input drives both videos to the same instant (set through the native setter so React sees the change)
    await modal.getByLabel('Seek').evaluate((input: HTMLInputElement) => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(input, '2');
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await expect.poll(async () => { const t = await times(); return Math.abs(t.before - 2) < 0.2 && Math.abs(t.after - 2) < 0.2; }).toBe(true);

    // the two layers really are different pictures: draw both current frames and compare (the result is colour-inverted)
    const difference = await page.evaluate(() => {
      const mean = (video: HTMLVideoElement) => {
        const canvas = document.createElement('canvas'); canvas.width = 64; canvas.height = 36;
        const context = canvas.getContext('2d')!; context.drawImage(video, 0, 0, 64, 36);
        const data = context.getImageData(0, 0, 64, 36).data; let sum = 0; for (let i = 0; i < data.length; i += 4) sum += data[i] + data[i + 1] + data[i + 2];
        return sum / (data.length / 4) / 3;
      };
      return Math.abs(mean(document.querySelector('video[aria-label="Original video"]')!) - mean(document.querySelector('video[aria-label="Completed render playback"]')!));
    });
    expect(difference).toBeGreaterThan(8);
  });

  test('the wipe divider reveals the result to its right; side by side shows both in full', async ({ page }) => {
    await boot(page, { route: '/#/faceswap', jobs: [job('running')] });
    await finishJob(page);
    const modal = page.getByTestId('render-complete-modal');
    await expect(modal).toBeVisible();
    await expect.poll(() => page.evaluate(() => [...document.querySelectorAll('.ocp video')].every(video => (video as HTMLVideoElement).readyState >= 2))).toBe(true);
    const clip = () => modal.locator('video[aria-label="Completed render playback"]').evaluate(video => getComputedStyle(video).clipPath);
    await expect.poll(clip).toContain('50%');
    await modal.getByLabel('Before and after divider').evaluate((input: HTMLInputElement) => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(input, '20');
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await expect.poll(clip).toContain('20%');                                            // the 50 ms clip-path transition has to settle
    await modal.locator('.ocp-stage').screenshot({ path: path.resolve('test-results/output-compare-wipe.png') });
    await modal.getByRole('button', { name: 'Side by side' }).click();
    const boxes = await modal.locator('.ocp-stage video').evaluateAll(videos => videos.map(video => { const r = video.getBoundingClientRect(); return { x: r.x, w: r.width }; }));
    expect(boxes).toHaveLength(2);
    expect(boxes[1].x).toBeGreaterThan(boxes[0].x + boxes[0].w - 2);                      // side by side, not stacked
    await modal.locator('.ocp-stage').screenshot({ path: path.resolve('test-results/output-compare-side.png') });
  });

  test('stays closed for jobs that were already finished when the page loaded', async ({ page }) => {
    await boot(page, { route: '/#/faceswap', jobs: [job('succeeded')] });
    await page.waitForTimeout(800);
    await expect(page.getByTestId('render-complete-modal')).toHaveCount(0);
  });
});
