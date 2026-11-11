// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const net = vi.hoisted(() => ({ api: vi.fn(), post: vi.fn() }));
vi.mock('./api', () => ({ api: net.api, post: net.post }));

import ModelManagerModal, { progressView } from './components/modelhub/ModelManagerModal';
import RequiredModelsBanner, { BANNER_DISMISS_KEY } from './components/modelhub/RequiredModelsBanner';
import { batchStore, formatBytes, formatEta, formatSpeed, licenseText, manifestStore, parseManifest, planRequired, resetBatch, type Manifest } from './stores/modelManifest';
import { ingestFrame, pipelineStream, resetPipelineStream } from './stores/pipelineStream';

class FakeWebSocket { onopen: null | (() => void) = null; onmessage = null; onclose = null; onerror = null; constructor(public url: string) {} close() { /* nothing to release */ } }

const MB = 1024 ** 2;
type Row = Record<string, unknown>;
const row = (id: string, patch: Row = {}): Row => ({
  id, name: `Model ${id}`, kind: 'weight', category: 'swapper', state: 'MISSING', installed: false, verified: false, hash_status: 'not_installed', detail: '',
  size_bytes: 100 * MB, installed_bytes: 0, partial_bytes: 0, sha256: 'a'.repeat(64), license: 'MIT', license_badge: 'open', license_url: `https://example.org/${id}`, license_note: '',
  requires_acceptance: false, optional: false, mirror_count: 2, acquisition: 'mirror', note: '', commercial_use: 'allow', required_for_stages: ['swap'], ...patch,
});
const ready = (id: string, patch: Row = {}) => row(id, { state: 'READY', installed: true, verified: true, hash_status: 'verified', installed_bytes: 100 * MB, ...patch });
const manifestBody = (models: Row[], extra: Row = {}) => ({
  schema: 'x', models,
  summary: { total: models.length, by_state: {}, licenses: { open: 1, restricted: 1, unverified: 1 }, required_missing: models.filter(m => !m.optional && !m.installed).map(m => m.id) },
  disk: { installed_bytes: 300 * MB, download_needed_bytes: 400 * MB, free_bytes: 50 * 1024 ** 3, enough_space: true }, ...extra,
});

const FIXTURE = [
  ready('cc0_head', { name: 'MakeHuman head', license: 'CC0-1.0', category: 'other' }),
  row('mit_det', { name: 'Detector', license: 'MIT', size_bytes: 20 * MB }),
  row('apache_enh', { name: 'Enhancer', license: 'Apache-2.0', optional: true }),
  row('ccby_tex', { name: 'Texture pack', license: 'CC-BY-4.0', size_bytes: 5 * MB }),
  row('nc_swap', { name: 'Swapper', license: 'InsightFace-Non-Commercial-Research', license_badge: 'restricted', requires_acceptance: true, size_bytes: 500 * MB }),
  row('unv_rest', { name: 'Restorer', license: 'UNVERIFIED', license_badge: 'unverified', requires_acceptance: true, size_bytes: 30 * MB }),
  row('flame', { name: 'FLAME 2023', license: 'CC-BY-4.0', mirror_count: 0, acquisition: 'registration_required', note: 'Free account at the owner site.' }),
  row('bad', { name: 'Corrupt one', state: 'CORRUPT', hash_status: 'mismatch', installed: false, optional: true }),
];

let host: HTMLDivElement, root: Root;
const settle = async (ms = 0) => { await act(async () => { if (ms) await vi.advanceTimersByTimeAsync(ms); else await Promise.resolve(); await Promise.resolve(); }); };
const click = async (element: Element | null | undefined) => { await act(async () => { (element as HTMLElement).click(); await Promise.resolve(); }); };
const byText = (text: string, scope: ParentNode = host) => [...scope.querySelectorAll('button')].find(b => b.textContent?.trim() === text) as HTMLButtonElement | undefined;
const hub = () => host.querySelector('[data-testid=model-hub-modal]')!;

function route(overrides: Record<string, (init?: RequestInit) => unknown> = {}) {
  net.api.mockImplementation(async (path: string, init?: RequestInit) => {
    for (const [prefix, handler] of Object.entries(overrides)) if (path.startsWith(prefix)) return handler(init);
    if (path.startsWith('/models/manifest')) return manifestBody(FIXTURE);
    throw new Error(`unexpected GET ${path}`);
  });
  net.post.mockImplementation(async (path: string) => {
    if (overrides[`POST ${path}`]) return overrides[`POST ${path}`]();
    throw new Error(`unexpected POST ${path}`);
  });
}

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'] });
  vi.stubGlobal('WebSocket', FakeWebSocket);
  host = document.createElement('div'); document.body.append(host); root = createRoot(host);
  net.api.mockReset(); net.post.mockReset(); localStorage.clear();
  resetBatch(); resetPipelineStream(); manifestStore.set({ manifest: null, loading: false, error: '', loadedAt: 0 });
});
afterEach(() => { act(() => root.unmount()); host.remove(); resetBatch(); vi.useRealTimers(); vi.unstubAllGlobals(); });

describe('manifest parsing', () => {
  it('turns a row into a fully defaulted model and never throws on missing fields', () => {
    const manifest = parseManifest({ models: [{ id: 'only_id' }, { name: 'no id' }, null, 'junk'], summary: { required_missing: [] } });
    expect(manifest.models).toHaveLength(1);
    expect(manifest.models[0]).toMatchObject({ id: 'only_id', name: 'only_id', state: 'MISSING', installed: false, hashStatus: 'not_installed', licenseBadge: 'restricted', sizeBytes: 0, optional: false, stages: [] });
  });

  it('rejects a response that is not a manifest, with one readable message', () => {
    for (const bad of [null, 'html', {}, { models: 'x' }, 42]) expect(() => parseManifest(bad)).toThrow(/unexpected model manifest/);
  });

  it('shows the licence names people know, UNVERIFIED for unknown terms', () => {
    const named = (license: string, licenseBadge: 'open' | 'restricted' | 'unverified' = 'open') => licenseText({ license, licenseBadge });
    expect([named('CC0-1.0'), named('MIT'), named('Apache-2.0'), named('CC-BY-4.0')]).toEqual(['CC0', 'MIT', 'Apache-2.0', 'CC-BY-4.0']);
    expect(named('LOCAL-USE-ACKNOWLEDGED-UNVERIFIED', 'unverified')).toBe('UNVERIFIED');
    expect(named('UNVERIFIED', 'restricted')).toBe('UNVERIFIED');
    expect(named('InsightFace-Non-Commercial-Research', 'restricted')).toBe('InsightFace-Non-Commercial-Research');
    expect(named('')).toBe('UNKNOWN');
  });

  it('plans "download all required": open now, restricted after acceptance, the rest cannot be automated', () => {
    const plan = planRequired(parseManifest(manifestBody(FIXTURE)));
    expect(plan.open.map(m => m.id)).toEqual(['mit_det', 'ccby_tex']);
    expect(plan.needsAcceptance.map(m => m.id)).toEqual(['nc_swap', 'unv_rest']);
    expect(plan.unavailable.map(m => m.id)).toEqual(['flame']);
    expect(plan.bytesOpen).toBe(25 * MB);
    expect(plan.bytesRestricted).toBe(530 * MB);
    expect(planRequired(null)).toMatchObject({ open: [], needsAcceptance: [], unavailable: [] });
  });

  it('formats sizes, megabits as MB/s and ETAs', () => {
    expect([formatBytes(0), formatBytes(512), formatBytes(1536), formatBytes(25 * MB), formatBytes(2.5 * 1024 ** 3)]).toEqual(['0 B', '512 B', '1.5 KB', '25.0 MB', '2.5 GB']);
    expect(formatSpeed(88)).toBe('11.0 MB/s');
    expect([formatEta(null), formatEta(7.2), formatEta(65), formatEta(3725)]).toEqual(['ETA …', 'ETA 7s', 'ETA 1m 05s', 'ETA 1h 02m']);
  });
});

describe('progress cell', () => {
  const model = (patch: Row = {}) => parseManifest(manifestBody([row('m', patch)])).models[0];
  const frame = (patch: Row = {}) => ({ type: 'model_download_progress', model_id: 'm', downloaded_bytes: 42 * MB, total_bytes: 100 * MB, percent: 42, speed_mbps: 88, eta_seconds: 65, status: 'downloading', ...patch });
  const transfer = (patch: Row = {}) => { ingestFrame(frame(patch)); return pipelineStream.get().transfers.m; };

  it('says percent, MB/s and ETA while downloading', () => expect(progressView(model(), transfer())).toEqual({ percent: 42, label: '42% · 11.0 MB/s · ETA 1m 05s', tone: 'busy' }));
  it('says Verifying SHA-256 at 100 % and Queued before the first byte', () => {
    expect(progressView(model(), transfer({ status: 'verifying' }))).toMatchObject({ percent: 100, label: 'Verifying SHA-256…', tone: 'verifying' });
    expect(progressView(model(), transfer({ status: 'queued', percent: 0 }))).toMatchObject({ label: 'Queued' });
  });
  it('shows an unknown ETA as an ellipsis, not as zero', () => expect(progressView(model(), transfer({ eta_seconds: null }))!.label).toBe('42% · 11.0 MB/s · ETA …'));
  it('shows a failure with the server\'s reason', () => expect(progressView(model(), transfer({ status: 'failed', detail: 'SHA-256 mismatch on every mirror' }))).toMatchObject({ tone: 'failed', label: 'Failed: SHA-256 mismatch on every mirror' }));
  it('offers a resumable partial file and nothing for an idle model', () => {
    expect(progressView(model({ partial_bytes: 25 * MB }), undefined)).toMatchObject({ percent: 25, tone: 'idle', label: '25% on disk, resumes' });
    expect(progressView(model(), undefined)).toBeNull();
    expect(progressView(model({ state: 'READY', installed: true, partial_bytes: 25 * MB }), undefined)).toBeNull();
  });
});

describe('Model Hub modal', () => {
  const open = async () => { await act(async () => { root.render(<ModelManagerModal onClose={vi.fn()} />); }); await settle(); };

  it('lists every model with licence badge, size, hash state and per-row action', async () => {
    route(); await open();
    const rows = [...host.querySelectorAll('[data-testid^=hub-row-]')];
    expect(rows).toHaveLength(FIXTURE.length);
    const badges = Object.fromEntries(FIXTURE.map(m => [m.id as string, host.querySelector(`[data-testid=license-${m.id}]`)!.textContent]));
    expect(badges).toMatchObject({ cc0_head: 'CC0', mit_det: 'MIT', apache_enh: 'Apache-2.0', ccby_tex: 'CC-BY-4.0', unv_rest: 'UNVERIFIED', nc_swap: 'InsightFace-Non-Commercial-Research' });
    expect(host.querySelector('[data-testid=license-mit_det]')!.className).toContain('open');
    expect(host.querySelector('[data-testid=license-nc_swap]')!.className).toContain('restricted');
    expect(host.querySelector('[data-testid=license-unv_rest]')!.className).toContain('unverified');
    expect(host.querySelector('[data-testid=hash-cc0_head]')!.textContent).toContain('✓ SHA-256 verified');
    expect(host.querySelector('[data-testid=hash-bad]')!.textContent).toContain('✗ SHA-256 mismatch');
    expect(host.querySelector('[data-testid=hash-mit_det]')!.textContent).toContain('not installed');
    expect(host.querySelector('[data-testid=hub-row-mit_det]')!.textContent).toContain('20.0 MB');
    expect(byText('Download', host.querySelector('[data-testid=hub-row-mit_det]')!)).toBeDefined();
    expect(host.querySelector('[data-testid=hub-row-flame]')!.textContent).toContain('No public source');
    expect(byText('Repair', host.querySelector('[data-testid=hub-row-bad]')!)).toBeDefined();
    expect(host.querySelector('[data-testid=hub-row-cc0_head] button')).toBeNull();
    expect(host.querySelector('[data-testid=model-hub-summary]')!.textContent).toContain('1 of 8 installed');
    expect(host.querySelector('[data-testid=model-hub-summary]')!.textContent).toContain('50.0 GB');
  });

  it('draws a live progress bar from the websocket frames', async () => {
    route(); await open();
    await act(async () => { ingestFrame({ type: 'model_download_progress', model_id: 'mit_det', downloaded_bytes: 8 * MB, total_bytes: 20 * MB, percent: 40, speed_mbps: 88, eta_seconds: 65, status: 'downloading' }); });
    const bar = host.querySelector('[data-testid=hub-row-mit_det] [role=progressbar]')!;
    expect(bar.getAttribute('aria-valuenow')).toBe('40');
    expect((bar.firstElementChild as HTMLElement).style.width).toBe('40%');
    expect(host.querySelector('[data-testid=progress-mit_det]')!.textContent).toBe('40% · 11.0 MB/s · ETA 1m 05s');
    expect(byText('Download', host.querySelector('[data-testid=hub-row-mit_det]')!)).toBeUndefined();    // already running
  });

  it('"Download All Required Models" sends only the open-licence ones until the restricted licences are accepted', async () => {
    route({ 'POST /models/download': () => ({ batch_id: 'b1', state: 'running', queued: ['mit_det', 'ccby_tex'], skipped: [] }), '/models/batches/b1': () => ({ batch_id: 'b1', state: 'running', entries: {} }) });
    await open();
    const button = host.querySelector('[data-testid=download-all-required]') as HTMLButtonElement;
    expect(button.textContent).toBe('Download All Required Models');
    await click(button);
    expect(net.post).toHaveBeenCalledWith('/models/download', { models: ['mit_det', 'ccby_tex'], accepted_license: false });
    expect(batchStore.get()).toMatchObject({ phase: 'running', batchId: 'b1' });
    expect((host.querySelector('[data-testid=download-all-required]') as HTMLButtonElement).disabled).toBe(true);        // no second batch while one runs
  });

  it('after ticking the acknowledgement it includes the restricted models and says it accepted', async () => {
    route({ 'POST /models/download': () => ({ batch_id: 'b2', queued: ['mit_det'], skipped: [] }), '/models/batches/': () => ({ state: 'running', entries: {} }) });
    await open();
    await click(host.querySelector('[data-testid=accept-licenses]'));
    expect(host.textContent).toContain('2 restricted models (530 MB)');
    await click(host.querySelector('[data-testid=download-all-required]'));
    expect(net.post).toHaveBeenCalledWith('/models/download', { models: ['mit_det', 'ccby_tex', 'nc_swap', 'unv_rest'], accepted_license: true });
  });

  it('a restricted row cannot be downloaded before the acknowledgement', async () => {
    route(); await open();
    const rowButton = byText('Download', host.querySelector('[data-testid=hub-row-nc_swap]')!)!;
    expect(rowButton.disabled).toBe(true);
    expect(rowButton.title).toMatch(/licence acknowledgement/);
  });

  it('polls the batch, then verifies and repairs automatically, then re-reads the manifest', async () => {
    let polls = 0;
    route({
      'POST /models/download': () => ({ batch_id: 'b3', queued: ['mit_det'], skipped: [] }),
      '/models/batches/b3': () => ({ batch_id: 'b3', state: ++polls < 3 ? 'running' : 'completed', entries: { mit_det: { status: polls < 3 ? 'downloading' : 'ready' } } }),
      'POST /models/verify-and-repair': () => ({ ok: true, checked: 1, verified: 1, missing: [], corrupt: [], repair: null }),
    });
    await open();
    const before = net.api.mock.calls.filter(([p]) => String(p).startsWith('/models/manifest')).length;
    await click(host.querySelector('[data-testid=download-all-required]'));
    await settle(1500); expect(batchStore.get().phase).toBe('running');
    await settle(1500); await settle(1500); await settle();
    expect(net.post).toHaveBeenCalledWith('/models/verify-and-repair', { models: ['mit_det'], repair: true, accepted_license: false });
    expect(batchStore.get().phase).toBe('done');
    expect(host.querySelector('[data-testid=hub-message]')!.textContent).toBe('Checked 1 models: 1 verified.');
    expect(net.api.mock.calls.filter(([p]) => String(p).startsWith('/models/manifest')).length).toBeGreaterThan(before);
  });

  it('reports models the service refused to start, with its reasons', async () => {
    route({ 'POST /models/download': () => ({ batch_id: '', queued: [], skipped: [{ id: 'nc_swap', reason: 'LICENSE_ACCEPTANCE_REQUIRED', detail: 'non-commercial' }] }) });
    await open();
    await click(host.querySelector('[data-testid=download-all-required]'));
    expect(batchStore.get().phase).toBe('done');
    const skipped = host.querySelector('[data-testid=hub-skipped]')!;
    expect(skipped.textContent).toContain('1 model was not started');
    expect(skipped.textContent).toContain('nc_swap: license acceptance required');
  });

  it('shows a failed start as an alert and lets the user try again', async () => {
    route({ 'POST /models/download': () => { throw new Error('Not enough free space'); } });
    await open();
    await click(host.querySelector('[data-testid=download-all-required]'));
    expect(host.querySelector('[data-testid=hub-message]')!.getAttribute('role')).toBe('alert');
    expect(host.querySelector('[data-testid=hub-message]')!.textContent).toBe('Not enough free space');
    expect((host.querySelector('[data-testid=download-all-required]') as HTMLButtonElement).disabled).toBe(false);
  });

  it('Verify & repair posts the whole-manifest request', async () => {
    route({ 'POST /models/verify-and-repair': () => ({ checked: 8, verified: 1, corrupt: ['bad'], repair: null }) });
    await open();
    await click(host.querySelector('[data-testid=verify-and-repair]'));
    expect(net.post).toHaveBeenCalledWith('/models/verify-and-repair', { models: [], repair: true, accepted_license: false });
    expect(host.querySelector('[data-testid=hub-message]')!.textContent).toBe('Checked 8 models: 1 verified, 1 repaired or set aside.');
  });

  it('shows a manifest error with a working retry instead of an empty list', async () => {
    let attempts = 0;
    route({ '/models/manifest': () => { if (++attempts === 1) throw new Error('Local service returned 503'); return manifestBody(FIXTURE); } });
    await open();
    expect(host.querySelector('[role=alert]')!.textContent).toContain('Local service returned 503');
    await click(byText('Retry'));
    expect(host.querySelectorAll('[data-testid^=hub-row-]')).toHaveLength(FIXTURE.length);
  });

  it('re-reads the manifest after the live stream reconnects (a backend restart)', async () => {
    route(); await open();
    const calls = () => net.api.mock.calls.filter(([p]) => String(p).startsWith('/models/manifest')).length;
    const before = calls();
    await act(async () => { pipelineStream.set(state => ({ resyncs: state.resyncs + 1 })); });
    await settle();
    expect(calls()).toBe(before + 1);
  });

  it('filters by requirement and by text', async () => {
    route(); await open();
    const select = host.querySelector('select[aria-label="Filter models"]') as HTMLSelectElement;
    await act(async () => { select.value = 'optional'; select.dispatchEvent(new Event('change', { bubbles: true })); });
    expect([...host.querySelectorAll('[data-testid^=hub-row-]')].map(r => r.getAttribute('data-testid'))).toEqual(['hub-row-apache_enh', 'hub-row-bad']);
    await act(async () => { select.value = 'all'; select.dispatchEvent(new Event('change', { bubbles: true })); });
    const search = host.querySelector('input[aria-label="Search models"]') as HTMLInputElement;
    await act(async () => { Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(search, 'apache-2.0'); search.dispatchEvent(new Event('input', { bubbles: true })); });
    expect(host.querySelectorAll('[data-testid^=hub-row-]')).toHaveLength(1);
  });
});

describe('required-models banner', () => {
  const show = async () => { await act(async () => { root.render(<RequiredModelsBanner onOpenHub={vi.fn()} />); }); await settle(); };

  it('appears while required models are missing, with a single download button', async () => {
    route({ 'POST /models/download': () => ({ batch_id: 'b9', queued: ['mit_det', 'ccby_tex'], skipped: [] }), '/models/batches/': () => ({ state: 'running', entries: {} }) });
    await show();
    const banner = host.querySelector('[data-testid=required-models-banner]')!;
    expect(banner.textContent).toContain('5 required models are not installed.');
    expect(banner.textContent).toContain('25.0 MB to download; 1 need a manual copy');
    expect(banner.textContent).toContain('SHA-256');
    expect(byText('Download All Required Models', banner)).toBeDefined();
    await click(host.querySelector('[data-testid=banner-download-all]'));
    expect(net.post).toHaveBeenCalledWith('/models/download', { models: ['mit_det', 'ccby_tex'], accepted_license: false });
    expect(banner.textContent).toContain('Downloading 2 models…');
    expect((banner.querySelector('button[aria-label="Hide this message"]') as HTMLButtonElement).disabled).toBe(true);
  });

  it('includes the restricted models only with the acknowledgement', async () => {
    route({ 'POST /models/download': () => ({ batch_id: 'b9', queued: ['x'], skipped: [] }), '/models/batches/': () => ({ state: 'running', entries: {} }) });
    await show();
    await click(host.querySelector('[data-testid=banner-accept]'));
    await click(host.querySelector('[data-testid=banner-download-all]'));
    expect(net.post).toHaveBeenCalledWith('/models/download', { models: ['mit_det', 'ccby_tex', 'nc_swap', 'unv_rest'], accepted_license: true });
  });

  it('stays out of the way once dismissed, and comes back only if the missing set changes', async () => {
    route(); await show();
    await click(host.querySelector('button[aria-label="Hide this message"]'));
    expect(host.querySelector('[data-testid=required-models-banner]')).toBeNull();
    expect(localStorage.getItem(BANNER_DISMISS_KEY)).toBe('mit_det,ccby_tex,nc_swap,unv_rest,flame');
    act(() => root.unmount()); root = createRoot(host);
    await show();
    expect(host.querySelector('[data-testid=required-models-banner]')).toBeNull();
    route({ '/models/manifest': () => manifestBody([...FIXTURE, row('new_req', { name: 'New required' })]) });
    await act(async () => { pipelineStream.set(state => ({ resyncs: state.resyncs + 1 })); }); await settle();
    expect(host.querySelector('[data-testid=required-models-banner]')).not.toBeNull();
  });

  it('is absent when everything required is installed', async () => {
    route({ '/models/manifest': () => manifestBody([ready('a'), row('opt', { optional: true })]) });
    await show();
    expect(host.querySelector('[data-testid=required-models-banner]')).toBeNull();
  });

  it('is absent (not an error banner) while the manifest cannot be read', async () => {
    net.api.mockRejectedValue(new Error('offline'));
    await show();
    expect(host.querySelector('[data-testid=required-models-banner]')).toBeNull();
  });

  it('keeps its parsed type honest', () => { const m: Manifest = parseManifest(manifestBody([])); expect(m.models).toEqual([]); });
});
