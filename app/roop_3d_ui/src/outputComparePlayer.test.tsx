// @vitest-environment jsdom
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { beforeEach, afterEach, expect, it, vi } from 'vitest';
import OutputComparePlayer, { DRIFT_SECONDS, RenderCompleteModal, afterUrl, beforeUrl, hasComparableVideo, newlyCompleted } from './components/output/OutputComparePlayer';
import type { Job } from './contracts';

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
let host: HTMLDivElement, root: Root;
const played: HTMLMediaElement[] = [], paused: HTMLMediaElement[] = [];
beforeEach(() => {
  played.length = 0; paused.length = 0;
  // jsdom implements neither playback nor `paused` transitions: model just enough of a media element for the sync logic.
  const state = new WeakMap<HTMLMediaElement, boolean>();
  Object.defineProperty(HTMLMediaElement.prototype, 'paused', { configurable: true, get() { return state.get(this) ?? true; } });
  HTMLMediaElement.prototype.play = vi.fn(function (this: HTMLMediaElement) { state.set(this, false); played.push(this); this.dispatchEvent(new Event('play')); return Promise.resolve(); });
  HTMLMediaElement.prototype.pause = vi.fn(function (this: HTMLMediaElement) { state.set(this, true); paused.push(this); this.dispatchEvent(new Event('pause')); });
  host = document.createElement('div'); document.body.append(host); root = createRoot(host);
});
afterEach(async () => { await act(async () => root.unmount()); host.remove(); });

const job = (patch: Partial<Job> = {}): Job => ({ job_id: 'a1b2c3d4e5f6', state: 'succeeded', request: { target_path: 'roop_3d/.runtime/media/in clip.mp4' } as Job['request'],
  progress: { frame: 30, total_frames: 30, fps: 12, stage: 'render' }, output: { output_path: 'roop_3d/.runtime/outputs/a1b2c3d4e5f6/result.mp4', frames: 30, baked_video: true }, failure: null, ...patch });
const videos = () => ({ before: host.querySelector('video[aria-label="Original video"]') as HTMLVideoElement, after: host.querySelector('video[aria-label="Completed render playback"]') as HTMLVideoElement });
const button = (name: string) => [...host.querySelectorAll('button')].find(item => item.textContent === name)!;
const mount = async (element = <OutputComparePlayer beforeSrc="/b.mp4" afterSrc="/a.mp4" />) => act(async () => root.render(element));
const setTime = (video: HTMLVideoElement, value: number) => Object.defineProperty(video, 'currentTime', { configurable: true, writable: true, value });

it('points the two videos at the original target and the verified download route', async () => {
  await mount(<RenderCompleteModal job={job()} onClose={() => {}} />);
  const { before, after } = videos();
  expect(after.getAttribute('src')).toBe('/api/v1/jobs/a1b2c3d4e5f6/download');
  expect(before.getAttribute('src')).toBe('/api/v1/media/content?path=roop_3d%2F.runtime%2Fmedia%2Fin%20clip.mp4');
  expect(afterUrl('x y')).toBe('/api/v1/jobs/x%20y/download');
  expect(beforeUrl('a/b.mp4')).toBe('/api/v1/media/content?path=a%2Fb.mp4');
  const download = host.querySelector('a.ocp-download') as HTMLAnchorElement;
  expect(download.textContent).toBe('Download MP4');
  expect(download.getAttribute('href')).toBe('/api/v1/jobs/a1b2c3d4e5f6/download');
  expect(download.getAttribute('download')).toBe('roop3d_a1b2c3d4.mp4');
  expect(host.querySelector('[role="dialog"]')?.textContent).toContain('30 verified frames');
});

it('keeps the original silent and the result as the audio master', async () => {
  await mount();
  const { before, after } = videos();
  expect(before.muted).toBe(true);
  expect(after.muted).toBe(false);
  await act(async () => button('Mute').click());
  expect(after.muted).toBe(true);
});

it('plays and pauses both videos together from one transport', async () => {
  await mount();
  const { before, after } = videos();
  await act(async () => button('Play').click());
  expect(played).toContain(after);
  expect(played).toContain(before);
  expect(button('Pause').getAttribute('aria-pressed')).toBe('true');
  await act(async () => button('Pause').click());
  expect(paused).toContain(after);
  expect(paused).toContain(before);
  expect(button('Play').getAttribute('aria-pressed')).toBe('false');
});

it('seeks both videos to the same instant', async () => {
  await mount();
  const { before, after } = videos();
  setTime(before, 0); setTime(after, 0);
  Object.defineProperty(after, 'duration', { configurable: true, value: 10 });
  await act(async () => { after.dispatchEvent(new Event('loadedmetadata')); });
  const seek = host.querySelector('input[aria-label="Seek"]') as HTMLInputElement;
  expect(seek.disabled).toBe(false);
  await act(async () => { Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(seek, '4.5'); seek.dispatchEvent(new Event('input', { bubbles: true })); });
  expect(after.currentTime).toBe(4.5);
  expect(before.currentTime).toBe(4.5);
});

it('re-seeks the original only when it drifts past the threshold while playing', async () => {
  await mount();
  const { before, after } = videos();
  setTime(after, 3); setTime(before, 3 + DRIFT_SECONDS * 0.5);
  await act(async () => button('Play').click());
  setTime(before, 3 + DRIFT_SECONDS * 0.5);                       // `play` itself re-aligns; reintroduce a small drift
  await act(async () => { after.dispatchEvent(new Event('timeupdate')); });
  expect(before.currentTime).toBeCloseTo(3 + DRIFT_SECONDS * 0.5, 5);   // inside tolerance: left alone
  setTime(before, 3 + DRIFT_SECONDS * 3);
  await act(async () => { after.dispatchEvent(new Event('timeupdate')); });
  expect(before.currentTime).toBe(3);                                  // outside: pulled back to the master
});

it('mirrors playback rate onto the original', async () => {
  await mount();
  const { before, after } = videos();
  after.playbackRate = 0.5;
  await act(async () => { after.dispatchEvent(new Event('ratechange')); });
  expect(before.playbackRate).toBe(0.5);
});

it('wipes with a keyboard-accessible divider and offers side by side', async () => {
  await mount();
  const stage = host.querySelector('.ocp-stage') as HTMLElement;
  expect(stage.dataset.mode).toBe('wipe');
  expect(stage.style.getPropertyValue('--ocp-pos')).toBe('50%');
  const divider = host.querySelector('input[aria-label="Before and after divider"]') as HTMLInputElement;
  await act(async () => { Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(divider, '20'); divider.dispatchEvent(new Event('input', { bubbles: true })); });
  expect(stage.style.getPropertyValue('--ocp-pos')).toBe('20%');
  await act(async () => button('Side by side').click());
  expect(stage.dataset.mode).toBe('side');
  expect(host.querySelector('input[aria-label="Before and after divider"]')).toBeNull();
  expect(button('Side by side').getAttribute('aria-pressed')).toBe('true');
});

it('says so when a video cannot load, and still lets the user download', async () => {
  await mount();
  const { before, after } = videos();
  await act(async () => { before.dispatchEvent(new Event('error')); });
  expect(host.querySelector('[role="alert"]')?.textContent).toContain('original video could not be loaded');
  expect((host.querySelector('a.ocp-download') as HTMLAnchorElement).getAttribute('href')).toBe('/a.mp4');
  await act(async () => { after.dispatchEvent(new Event('error')); });
  expect(host.querySelector('[role="alert"]')?.textContent).toContain('rendered video could not be loaded');
  expect((button('Play') as HTMLButtonElement).disabled).toBe(true);
});

it('opens only for a job that finishes during the session, and only for a single playable video', () => {
  const running = job({ state: 'running', output: null }), done = job();
  expect(newlyCompleted(null, [done])).toBeNull();                                                 // first load: history stays quiet
  expect(newlyCompleted(new Map([[done.job_id, 'succeeded']]), [done])).toBeNull();                // already finished before
  expect(newlyCompleted(new Map([[running.job_id, 'running']]), [done])?.job_id).toBe(done.job_id); // running -> succeeded
  expect(newlyCompleted(new Map(), [done])?.job_id).toBe(done.job_id);                              // appeared already finished (fast job)
  const sequence = job({ output: { output_path: 'roop_3d/.runtime/outputs/x/frames', frames: 30, baked_video: true } });
  const passesOnly = job({ output: { output_path: 'roop_3d/.runtime/outputs/x/a.mp4', frames: 30, baked_video: false } });
  expect(hasComparableVideo(sequence)).toBe(false);
  expect(hasComparableVideo(passesOnly)).toBe(false);
  expect(hasComparableVideo(job({ state: 'failed', output: null }))).toBe(false);
  expect(newlyCompleted(new Map(), [sequence, passesOnly])).toBeNull();
});

it('closes from the Close button and labels the dialog', async () => {
  const onClose = vi.fn();
  await mount(<RenderCompleteModal job={job()} onClose={onClose} />);
  expect(host.querySelector('[role="dialog"]')?.getAttribute('aria-label')).toBe('Render complete');
  await act(async () => button('Close').click());
  expect(onClose).toHaveBeenCalledOnce();
});
