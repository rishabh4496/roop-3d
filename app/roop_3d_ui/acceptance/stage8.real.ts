import { expect, test } from '@playwright/test';
import { readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { appRoot, ownedBrowserPath } from '../browserPaths';

const base = ownedBrowserPath(resolve(appRoot, 'roop_3d/.runtime/stage08_acceptance'));
const fixture = JSON.parse(readFileSync(resolve(base, 'fixture.json'), 'utf8'));
const artifacts = ownedBrowserPath(resolve(base, 'browser'));

test.afterEach(async ({ page }, info) => {
  if (info.status !== info.expectedStatus) await page.screenshot({ path: ownedBrowserPath(resolve(artifacts, 'failure.png')), fullPage: true });
});

test('offline production UI: local imports → identity/asset → preview → project → pause/resume → verified export', async ({ page, context }) => {
  mkdirSync(artifacts, { recursive: true });
  page.setDefaultTimeout(60_000);
  const external: string[] = [], errors: string[] = [];
  await context.route('**/*', route => {
    const url = new URL(route.request().url());
    if (url.protocol === 'http:' && url.hostname === '127.0.0.1') return route.continue();
    if (['data:', 'blob:'].includes(url.protocol)) return route.continue();
    external.push(url.origin); return route.abort();
  });
  page.on('pageerror', error => errors.push(error.message));
  await page.addInitScript(() => { localStorage.setItem('roop3d_tour_seen', 'true'); });
  await page.goto('/#/faceswap');
  expect((await (await page.request.get('/api/meta')).json()).mock).toBe(false);
  const originalConfig = await (await page.request.get('/api/v1/config')).json();
  // Test setup selects the actual local CUDA provider without changing launcher/environment files.
  const configured = await page.request.put('/api/v1/config', { data: { ...originalConfig, render_backend: 'reference', audio_policy: 'copy' } });
  expect(configured.ok()).toBeTruthy();
  await page.reload();
  try {
    await expect(page.getByRole('button', { name: /Import target video/ })).toBeEnabled({ timeout: 60_000 });
    const sourceChooser = page.waitForEvent('filechooser');
    await page.getByRole('button', { name: /Import .fsz/ }).first().click();
    const sourceDone = page.waitForResponse(r => r.url().endsWith('/complete') && r.request().method() === 'POST');
    await (await sourceChooser).setFiles(ownedBrowserPath(resolve(appRoot, fixture.source_path)));
    const imported = await (await sourceDone).json();
    expect(imported.kind).toBe('faceset');
    const sourceId = imported.faceset_id;

    const targetChooser = page.waitForEvent('filechooser');
    await page.getByRole('button', { name: /Import target video/ }).click();
    const targetDone = page.waitForResponse(r => r.url().endsWith('/complete') && r.request().method() === 'POST');
    await (await targetChooser).setFiles(ownedBrowserPath(resolve(appRoot, fixture.target_path)));
    const target = await (await targetDone).json();
    expect(target.kind).toBe('video');
    await expect(page.getByRole('button', { name: 'Inspect target faces' })).toBeEnabled({ timeout: 60_000 });
    const inspected = page.waitForResponse(r => r.url().endsWith('/media/inspect-faces'));
    await page.getByRole('button', { name: 'Inspect target faces' }).click();
    const inspection = await (await inspected).json();
    expect((await inspected).request().postDataJSON().full_video).toBe(true);
    expect(inspection.tracks.length).toBeGreaterThan(0);
    expect(inspection.truncated).toBe(false);
    const trackId = inspection.tracks[0].track_id;
    await page.getByLabel(`Identity for ${trackId}`, { exact: true }).selectOption(sourceId);
    await expect(page.getByRole('table', { name: 'Pose coverage, yaw by pitch' })).toBeVisible({ timeout: 60_000 });
    await expect(page.locator('.asset-card')).toBeVisible();
    if (await page.getByRole('button', { name: 'Build 3D identity asset', exact: true }).isVisible()) {
      await page.getByRole('button', { name: 'Build 3D identity asset', exact: true }).click();
    }
    await expect(page.getByRole('button', { name: 'View 3D asset' })).toBeVisible({ timeout: 120_000 });
    await page.getByRole('button', { name: 'View 3D asset' }).click();
    await expect(page.getByRole('button', { name: 'Hide 3D asset' })).toBeVisible();
    await page.getByLabel('Swap engine', { exact: true }).selectOption('reference_onnx');
    await page.getByLabel('Restoration model', { exact: true }).selectOption('gpen');
    await page.getByRole('textbox', { name: 'Restorer blend', exact: true }).fill('0.2');
    await page.getByRole('textbox', { name: 'Restorer blend', exact: true }).press('Enter');
    const motionPreview = page.waitForResponse(r => r.url().endsWith('/api/preview') && r.status() === 200
      && r.request().postDataJSON()?.render_mode === 'reenactment_3d', { timeout: 120_000 });
    await page.getByLabel('3D render mode', { exact: true }).selectOption('reenactment_3d');
    expect((await motionPreview).headers()['x-preview-meta']).toBeTruthy();
    const previewed = page.waitForResponse(r => r.url().endsWith('/api/preview') && r.status() === 200
      && r.request().postDataJSON()?.render_mode === 'pose_3d', { timeout: 120_000 });
    await page.getByLabel('3D render mode', { exact: true }).selectOption('pose_3d');
    const preview = await previewed;
    expect(preview.headers()['x-preview-meta']).toBeTruthy();
    await page.getByRole('textbox', { name: 'Mask feather', exact: true }).fill('6');
    await page.getByRole('textbox', { name: 'Mask feather', exact: true }).press('Enter');
    await page.locator('.compare-toggle').click();
    const projectName = `Stage 8 real offline ${Date.now()}`;
    await page.getByLabel('Project name', { exact: true }).fill(projectName);
    const savedResponse = page.waitForResponse(r => r.url().endsWith('/projects') && r.request().method() === 'POST');
    await page.getByRole('button', { name: 'Save project', exact: true }).click();
    const saved = await (await savedResponse).json();
    expect(saved.identity_assets[sourceId]).toBeTruthy();
    expect(saved.preview.viewport.view_mode).toBe('compare');
    expect(saved.request.workstation_parameters.feather_px).toBe(6);
    expect(saved.request.workstation_parameters.restoration_blend).toBe(.2);
    await page.screenshot({ path: ownedBrowserPath(resolve(artifacts, 'preview-and-project.png')), fullPage: true });

    const queuedResponse = page.waitForResponse(r => r.url().endsWith(`/projects/${saved.project_id}/jobs`) && r.request().method() === 'POST', { timeout: 120_000 });
    await page.getByRole('button', { name: /Queue render/ }).first().click();
    await expect(page.getByRole('button', { name: /Proceed & Queue Render/ })).toBeEnabled({ timeout: 120_000 });
    await page.getByRole('button', { name: /Proceed & Queue Render/ }).click();
    const queued = await (await queuedResponse).json();
    const jobId = queued.job_id;
    expect(queued.project_id).toBe(saved.project_id);
    const card = page.locator('.queue-job-card').filter({ hasText: jobId.slice(0, 8) });
    await expect.poll(async () => (await (await page.request.get(`/api/v1/jobs/${jobId}`)).json()).progress.frame,
      { timeout: 120_000 }).toBeGreaterThanOrEqual(1);
    await card.getByRole('button', { name: 'Pause', exact: true }).click();
    await expect(card.locator('.state-badge')).toHaveText('PAUSED', { timeout: 120_000 });
    await expect(card.getByRole('link', { name: /Export video/ })).toHaveCount(0);
    await page.getByLabel('Include selected paused/interrupted checkpoint when saving a project').check();
    const pausedSave = page.waitForResponse(r => r.url().endsWith(`/projects/${saved.project_id}`) && r.request().method() === 'PUT');
    await page.getByRole('button', { name: 'Save project', exact: true }).click();
    expect((await (await pausedSave).json()).checkpoint_job_id).toBe(jobId);
    await page.reload();
    await page.getByRole('button', { name: 'Projects', exact: true }).click();
    await page.locator('.saved-project').filter({ hasText: projectName }).getByRole('button', { name: 'Verify & open' }).click();
    await expect(page.getByLabel('3D render mode', { exact: true })).toHaveValue('pose_3d', { timeout: 120_000 });
    await expect(page.locator('.compare-toggle')).toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByLabel('Identity for ' + trackId, { exact: true })).toHaveValue(sourceId);
    await card.getByRole('button', { name: 'Resume', exact: true }).click();
    await expect(card.locator('.state-badge')).toHaveText('COMPLETED', { timeout: 240_000 });
    const completed = await (await page.request.get(`/api/v1/jobs/${jobId}`)).json();
    expect(completed.output.frames).toBe(fixture.frames);
    await card.getByRole('button', { name: 'Inspect output', exact: true }).click();
    const player = page.getByLabel('Completed render playback');
    await expect(player).toBeVisible();
    await expect.poll(() => player.evaluate((v: HTMLVideoElement) => v.readyState), { timeout: 60_000 }).toBeGreaterThanOrEqual(1);
    await page.screenshot({ path: ownedBrowserPath(resolve(artifacts, 'completed-output.png')), fullPage: true });
    for (const [label, filename] of [['Export video', 'export.mp4'], ['Render manifest', 'render.manifest.json']]) {
      const download = page.waitForEvent('download');
      await page.getByRole('dialog', { name: 'Rendered output' }).getByRole('link', { name: label, exact: true }).click();
      await (await download).saveAs(ownedBrowserPath(resolve(artifacts, filename)));
    }
    const history = await (await page.request.get(`/api/v1/projects/${saved.project_id}`)).json();
    const manifest = JSON.parse(readFileSync(resolve(artifacts, 'render.manifest.json'), 'utf8'));
    expect(manifest.model_hashes.restorer).toMatch(/^[a-f0-9]{64}$/);
    expect(manifest.stage12.restoration.inference).toBe('VERIFIED_LOCAL');
    expect(history.output_history.some((item: any) => item.job_id === jobId && item.state === 'succeeded')).toBe(true);
    // This host's security software injects a script independently of the application bundle.
    // It stays blocked; do not disable security software or permit its network traffic.
    const browserInjection = external.filter(origin => origin === 'http://gc.kis.v2.scr.kaspersky-labs.com');
    expect(external.filter(origin => origin !== 'http://gc.kis.v2.scr.kaspersky-labs.com')).toEqual([]); expect(errors).toEqual([]);
    writeFileSync(ownedBrowserPath(resolve(artifacts, 'evidence.json')), JSON.stringify({ job_id: jobId, project_id: saved.project_id,
      output: completed.output, model_diagnostics: completed.diagnostics, blocked_external_requests: external,
      browser_injection_requests_blocked: browserInjection, console_errors: errors,
      asset_key: saved.identity_assets[sourceId], pause_resume: true, project_reload: true }, null, 2));
  } finally {
    await page.request.put('/api/v1/config', { data: originalConfig });
  }
});
