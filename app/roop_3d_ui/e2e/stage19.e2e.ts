import { createRequire } from 'node:module';
import { expect, test, type Page } from '@playwright/test';
import { boot, ID, SCREENS, TARGET, VIEWPORTS } from './helpers/mockApi';

/**
 * Stage 19: professional UX, accessibility and documentation, in a real browser (system Chrome, mocked local service,
 * real React / WebGL / focus / keyboard handling). Screenshots of every screen at every viewport are written to
 * `test-results/stage19/` for the visual review.
 */
const require = createRequire(import.meta.url);
const AXE = require.resolve('axe-core/axe.min.js');

test.use({ viewport: { width: 1680, height: 1050 } });

const jobShape = (extra: Record<string, unknown>) => ({ job_id: 'c'.repeat(32), state: 'queued', request: { target_path: TARGET, inspection_id: ID, face_mappings: [], processing: 'local', remote_consent: false, trim_start_seconds: null, trim_end_seconds: null },
  progress: { frame: 0, total_frames: 100, fps: 0, stage: 'x' }, output: null, failure: null, ...extra });

async function axeViolations(page: Page, scope = 'document') {
  await page.addScriptTag({ path: AXE });
  return page.evaluate(async (target: string) => {
    const result = await (window as any).axe.run(target === 'document' ? document : document.querySelector(target), {
      runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa'] } });
    return result.violations.map((item: any) => ({ id: item.id, impact: item.impact, nodes: item.nodes.length, sample: item.nodes.slice(0, 3).map((node: any) => `${node.target.join(' ')} :: ${(node.failureSummary || '').split('\n').slice(1, 3).join(' ').slice(0, 160)}`) }));
  }, scope);
}

/** Themes are chosen from the swatch gallery (it replaced the header dropdown). */
async function openThemeGallery(page: Page) { await page.getByRole('button', { name: 'Open theme gallery' }).click(); await expect(page.getByTestId('theme-gallery')).toBeVisible(); }
async function pickTheme(page: Page, name: string) { await page.getByTestId('theme-gallery').locator('.theme-card', { has: page.locator('.theme-name', { hasText: name }) }).first().click(); }

/* ------------------------------------------------------------------------------------------------ onboarding */
test.describe('first-run onboarding and guided project', () => {
  test('welcome dialog is accessible, traps focus, closes with Escape and is shown only once', async ({ page }) => {
    await boot(page, { route: '/#/home', firstRun: true });
    const dialog = page.getByRole('dialog', { name: 'Welcome to Roop 3D' });
    await expect(dialog).toBeVisible();
    await expect(dialog).toHaveAttribute('aria-modal', 'true');
    await expect(page.getByRole('button', { name: 'Start a guided project' })).toBeFocused();
    await expect(dialog.getByText(/never uploaded/)).toBeVisible();
    for (let i = 0; i < 12; i++) { await page.keyboard.press('Tab'); expect(await page.evaluate(() => !!document.activeElement?.closest('[role=dialog]'))).toBe(true); }
    for (let i = 0; i < 12; i++) { await page.keyboard.press('Shift+Tab'); expect(await page.evaluate(() => !!document.activeElement?.closest('[role=dialog]'))).toBe(true); }
    await page.keyboard.press('Escape');
    await expect(dialog).toBeHidden();
    expect(await page.evaluate(() => localStorage.getItem('roop3d-onboarding-v1'))).toBe('1');
    await page.reload();
    await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible();
    await expect(page.getByRole('dialog', { name: 'Welcome to Roop 3D' })).toHaveCount(0);
  });

  test('"Start a guided project" lands on the workstation with the project name selected and a next step shown', async ({ page }) => {
    await boot(page, { route: '/#/home', firstRun: true });
    await page.getByRole('button', { name: 'Start a guided project' }).click();
    await expect(page).toHaveURL(/#\/faceswap$/);
    await expect(page.getByLabel('Project name')).toBeFocused();
    const bar = page.getByTestId('next-step-bar');
    await expect(bar).toContainText('STEP 1 OF 6');
    await expect(bar).toContainText('Import the target video');
    await expect(page).toHaveTitle(/Face Swap · Roop 3D/);
  });

  test('the Home checklist and the Guide dialog show real progress and describe every step', async ({ page }) => {
    await boot(page, { route: '/#/faceswap', openProject: true });
    await page.getByRole('button', { name: 'Guide', exact: true }).click();
    const guide = page.getByRole('dialog', { name: 'Guide' });
    await expect(guide).toBeVisible();
    const steps = guide.locator('.workflow-step');
    await expect(steps).toHaveCount(7);
    for (const id of ['target', 'source', 'inspect', 'assign']) await expect(guide.locator(`[data-step="${id}"]`)).toHaveClass(/done/);
    await expect(guide.locator('[aria-current="step"]')).toHaveCount(1);
    await expect(guide.locator('.workflow-step.current')).toContainText(/live preview|export preset/i);
    await expect(guide.locator('.workflow-step.current')).toContainText('You are done when');
    // Every topic tab has content that matches the app: spot-check the numbers the text promises.
    await guide.getByRole('tab', { name: 'FaceSets & photos' }).click();
    await expect(guide.getByRole('tabpanel')).toContainText('40 pixels');
    await expect(guide.getByRole('tabpanel')).toContainText('0.03');
    await guide.getByRole('tab', { name: 'Engines & hardware' }).click();
    await expect(guide.getByRole('tabpanel')).toContainText('this computer: 12gb');
    await guide.getByRole('tab', { name: 'Render presets' }).click();
    await expect(guide.getByRole('table', { name: 'Export presets' }).locator('tbody tr')).toHaveCount(7);
    await guide.getByRole('tab', { name: 'Queue & recovery' }).click();
    await expect(guide.getByRole('table', { name: 'Job states' })).toContainText('Interrupted');
    // Arrow keys move between tabs (WAI-ARIA tabs).
    await guide.getByRole('tab', { name: 'Queue & recovery' }).focus();
    await page.keyboard.press('ArrowRight');
    await expect(guide.getByRole('tab', { name: 'Display & keyboard' })).toHaveAttribute('aria-selected', 'true');
    await page.keyboard.press('Escape');
    await expect(guide).toBeHidden();
  });

  test('the next-step button is honest while the service is still starting', async ({ page }) => {
    await boot(page, { route: '/#/faceswap', capsDelayMs: 2500 });
    const button = page.getByTestId('next-step-bar').getByRole('button', { name: /import target video/i });
    await expect(button).toHaveAttribute('aria-disabled', 'true');
    await button.click({ force: true });   // Playwright treats aria-disabled as disabled and would wait; a user can press it
    await expect(page.locator('.notification')).toContainText(/Waiting for an available service capability|Still connecting/);
    await expect(button).not.toHaveAttribute('aria-disabled', 'true', { timeout: 10_000 });
  });
});

/* ------------------------------------------------------------------------------------------------ guidance */
test.describe('in-place guidance', () => {
  test('target faces: tutorial, FaceSet coverage reading, photo checklist, prioritised requests', async ({ page }) => {
    await boot(page, { openProject: true });
    const tutorial = page.getByTestId('target-tutorial');
    await tutorial.locator('summary').click();
    await expect(tutorial).toContainText('Swap (replace)');
    await page.locator('.studio-identity').getByText('Alice').first().click();
    const reading = page.getByTestId('coverage-reading');
    await expect(reading).toContainText('required angles are all present');
    await expect(reading).toContainText('left profile');
    await reading.getByText('Source photo checklist').click();
    await expect(reading).toContainText('Sharp and in focus');
    await expect(page.locator('.coverage-cell').first()).toHaveAttribute('aria-label', /yaw .*: (no views|\d+ views?)/);
    await expect(page.locator('.asset-requests .status-chip.request-recommended')).toBeVisible();
    await expect(page.locator('.asset-meaning')).toContainText('Some turned angles are covered');
  });

  test('engine and hardware: a recommendation with its reason, why others are unavailable, what the profile means', async ({ page }) => {
    await boot(page, { openProject: true });
    const engine = page.getByTestId('engine-explainer');
    await expect(engine).toContainText('Recommended');
    await expect(engine).toContainText('Roop 3D Reference ONNX');
    await expect(engine).toContainText('Licence restricted');
    await engine.getByText('Why can’t I select the other engines?').click();
    await expect(engine).toContainText('Experimental X');
    await expect(engine).toContainText('Not installed');
    const hardware = page.getByTestId('hardware-explainer').first();
    await expect(hardware).toContainText('12 GB (high capacity)');
    await expect(hardware).toContainText('chosen automatically');
    await hardware.getByText('What this changes').click();
    await expect(hardware).toContainText('TensorRT');
  });

  test('export preset: explained, saved with the project, and clears back to the default', async ({ page }) => {
    const { seen } = await boot(page, { openProject: true });
    const bar = page.locator('.studio-bottom').getByTestId('export-preset');   // the Editor screen has a second copy (hidden)
    const preset = bar.locator('select[aria-label="Export preset"]');
    await expect(preset.locator('option')).toHaveCount(7);
    await expect(page.locator('.project-name span')).toHaveText('SAVED');
    await preset.selectOption('high_quality_mp4');
    await expect(bar).toContainText('Use it for:');
    await expect(bar).toContainText('final copy');
    await expect(page.locator('.project-name span')).toHaveText('UNSAVED');
    await bar.getByText('More export options').click();
    await bar.getByLabel('HDR policy').selectOption('reject');
    await page.getByRole('button', { name: 'Save project' }).click();
    await expect(page.locator('.project-name span')).toHaveText('SAVED');
    expect(seen.projects.at(-1).request.export).toMatchObject({ profile: 'high_quality_mp4', hdr_policy: 'reject' });
    await preset.selectOption('standard_mp4');
    await bar.getByLabel('HDR policy').selectOption('tonemap_sdr');
    await page.getByRole('button', { name: 'Save project' }).click();
    await expect(page.locator('.project-name span')).toHaveText('SAVED');
    expect(seen.projects.at(-1).request.export.profile ?? null).toBeNull();
  });

  test('a compositing pass the render service cannot produce is disabled, with its reason', async ({ page }) => {
    await boot(page, { route: '/#/extras' });
    const pass = page.getByLabel(/Swapped-Face-Only Foreground/);
    await expect(pass).toBeDisabled();
    await expect(pass).not.toBeChecked();
    await expect(page.getByText(/not available in this build/)).toBeVisible();
    await expect(page.getByLabel(/Face-Canvas Alpha Matte/)).toBeEnabled();
  });

  test('queue and checkpoint guidance, job hints, failure recovery and progress semantics', async ({ page }) => {
    await boot(page, { openProject: true, jobs: [
      jobShape({ job_id: 'd'.repeat(32), state: 'interrupted', progress: { frame: 40, total_frames: 100, fps: 0, stage: 'x' } }),
      jobShape({ job_id: 'e'.repeat(32), state: 'failed', failure: { code: 'DISK_FULL', message: 'There is not enough free space', recoverable: true } })] });
    const help = page.getByTestId('queue-help');
    await help.locator('summary').click();
    await expect(help).toContainText('replays the analysis from the first frame');
    await expect(page.locator('.job-hint')).toContainText('Press Resume');
    const failure = page.locator('.failure-text');
    await expect(failure).toContainText('The disk is full');
    await expect(failure).toContainText('(code DISK_FULL)');
    await expect(failure).toContainText('Try this first');
    const bars = page.getByRole('progressbar');
    await expect(bars).toHaveCount(2);
    expect(await bars.evaluateAll(items => items.map(item => item.getAttribute('aria-valuenow')))).toContain('40');
  });

  test('errors say what happened and what to do; unknown ones show their message untouched', async ({ page }) => {
    await boot(page, { openProject: true, inspectError: { status: 409, code: 'LICENSE_UNVERIFIED', message: 'Model is not eligible for inference: License and redistribution terms are not fully reviewed' } });
    await page.getByRole('button', { name: 'Re-inspect faces' }).click();
    const notice = page.locator('.notification.error');
    await expect(notice).toHaveAttribute('role', 'alert');
    await expect(notice).toContainText('A model has not passed its licence and safety review');
    await expect(notice).toContainText('(code LICENSE_UNVERIFIED, request r1abc)');
    await expect(notice).toContainText('Try this first');
    await expect(notice.getByRole('button', { name: 'Reconnect' })).toHaveCount(0);
    await notice.getByText('What happened, and every step').click();
    await expect(notice).toContainText('docs/OFFLINE_MODELS.md');
  });

  test('a stopped service gets a Reconnect button and start instructions', async ({ page }) => {
    await boot(page, { offline: true, route: '/#/home' });
    const notice = page.locator('.notification.error');
    await expect(notice).toContainText('The local service is not reachable');
    await expect(notice.getByRole('button', { name: 'Reconnect' })).toBeVisible();
    await expect(notice).toContainText('press Start');
  });
});

/* ------------------------------------------------------------------------------------------------ accessibility */
test.describe('accessibility', () => {
  test('skip link is the first tab stop, becomes visible and moves focus to the workspace', async ({ page }) => {
    await boot(page, { route: '/#/home' });
    await page.keyboard.press('Tab');
    const skip = page.getByRole('button', { name: 'Skip to main content' });
    await expect(skip).toBeFocused();
    const box = (await skip.boundingBox())!;
    expect(box.y).toBeGreaterThanOrEqual(0);
    await page.keyboard.press('Enter');
    expect(await page.evaluate(() => document.activeElement?.id)).toBe('workspace');
  });

  test('every keyboard focus stop shows a visible focus indicator', async ({ page }) => {
    await boot(page, { openProject: true });
    await page.locator('body').click({ position: { x: 5, y: 5 } });
    const problems: string[] = [];
    for (let i = 0; i < 60; i++) {
      await page.keyboard.press('Tab');
      const info = await page.evaluate(() => {
        const element = document.activeElement as HTMLElement | null;
        if (!element || element === document.body) return null;
        const style = getComputedStyle(element);
        const ring = parseFloat(style.outlineWidth) >= 2 && style.outlineStyle !== 'none' && style.outlineColor !== 'rgba(0, 0, 0, 0)';
        const shadow = style.boxShadow !== 'none' && /\d+px/.test(style.boxShadow);
        return { ok: ring || shadow, label: `${element.tagName}.${String(element.className).slice(0, 40)} "${(element.getAttribute('aria-label') || element.textContent || '').trim().slice(0, 30)}"` };
      });
      if (info && !info.ok) problems.push(info.label);
    }
    expect(problems, `focus stops without a visible indicator: ${problems.join(' | ')}`).toEqual([]);
  });

  test('every control has an accessible name on every screen', async ({ page }) => {
    await boot(page, { openProject: true });
    const problems: string[] = [];
    for (const screen of SCREENS) {
      await page.evaluate(hash => { location.hash = hash; }, `#/${screen}`);
      await page.waitForTimeout(250);
      const found = await page.evaluate((name: string) => {
        const out: string[] = [];
        for (const el of document.querySelectorAll<HTMLElement>('button,input:not([type=hidden]),select,textarea,a[href],[role=button],[role=tab],summary')) {
          if (el.closest('[hidden],[inert]')) continue;
          const rect = el.getBoundingClientRect(), style = getComputedStyle(el);
          if (!rect.width || !rect.height || style.visibility === 'hidden' || style.display === 'none') continue;
          const label = el.getAttribute('aria-label') || el.getAttribute('aria-labelledby') || el.textContent?.trim() || el.getAttribute('title') || (el as HTMLInputElement).labels?.length || el.getAttribute('placeholder');
          const desc = `${name}: ${el.tagName}.${String(el.className).slice(0, 40)} "${(el.getAttribute('aria-label') || el.textContent || '').trim().slice(0, 24)}"`;
          if (!label) out.push(`NO NAME ${desc}`);
        }
        return out;
      }, screen);
      problems.push(...found);
    }
    expect(problems, problems.join('\n')).toEqual([]);
  });

  test('every dialog: labelled, modal, focus moves in and is trapped, Escape closes, focus returns', async ({ page }) => {
    await boot(page, { openProject: true });
    const cases: Array<{ opener: string; name: RegExp; open: () => Promise<void> }> = [
      { opener: 'Guide', name: /^Guide$/, open: () => page.getByRole('button', { name: 'Guide', exact: true }).click() },
      { opener: 'How to Use', name: /./, open: () => page.getByTestId('header-tour-button').click() },
      { opener: 'Projects', name: /Project library/, open: () => page.getByRole('button', { name: 'Projects', exact: true }).click() },
      { opener: 'Open theme gallery', name: /Theme Gallery/, open: () => page.getByRole('button', { name: 'Open theme gallery' }).click() },
    ];
    for (const item of cases) {
      await item.open();
      const dialog = page.getByRole('dialog').last();
      await expect(dialog, item.opener).toBeVisible();
      await expect(dialog).toHaveAttribute('aria-modal', 'true');
      const name = await dialog.evaluate(el => el.getAttribute('aria-label') || document.getElementById(el.getAttribute('aria-labelledby') || '')?.textContent || '');
      expect(name, `${item.opener} dialog name`).toMatch(item.name);
      expect(await page.evaluate(() => !!document.activeElement?.closest('[role=dialog]')), `${item.opener}: focus starts inside`).toBe(true);
      for (let i = 0; i < 40; i++) { await page.keyboard.press('Tab'); if (!(await page.evaluate(() => !!document.activeElement?.closest('[role=dialog]')))) throw new Error(`${item.opener}: focus escaped the dialog after ${i + 1} Tab presses`); }
      await page.keyboard.press('Escape');
      await expect(page.getByRole('dialog'), `${item.opener}: Escape closes`).toHaveCount(0);
      const returned = await page.evaluate(() => (document.activeElement?.textContent ?? '') + (document.activeElement?.getAttribute('aria-label') ?? ''));
      expect(returned, `${item.opener}: focus returns to the opener`).toMatch(new RegExp(item.opener === 'How to Use' ? 'How to Use' : item.opener));
    }
  });

  test('the pre-flight linter and the queue output viewer are proper dialogs too', async ({ page }) => {
    await boot(page, { openProject: true });
    await page.evaluate(() => sessionStorage.removeItem('roop3d_skip_linter'));
    await page.getByRole('button', { name: /Queue render/ }).first().click();
    const dialog = page.getByRole('dialog').last();
    await expect(dialog).toBeVisible();
    await expect(dialog).toHaveAttribute('aria-modal', 'true');
    expect(await dialog.evaluate(el => !!(el.getAttribute('aria-label') || el.getAttribute('aria-labelledby')))).toBe(true);
    expect(await page.evaluate(() => !!document.activeElement?.closest('[role=dialog]'))).toBe(true);
    await page.keyboard.press('Escape');
    await expect(page.getByRole('dialog')).toHaveCount(0);
  });

  test('reduced motion: system setting, and the user override in both directions', async ({ page }) => {
    await boot(page, { route: '/#/home' });
    const duration = () => page.evaluate(() => {
      const probe = document.createElement('span'); probe.className = 'skeleton-line'; document.body.append(probe);
      const value = parseFloat(getComputedStyle(probe).animationDuration) * (getComputedStyle(probe).animationDuration.endsWith('ms') ? 0.001 : 1); probe.remove(); return value;
    });
    await page.emulateMedia({ reducedMotion: 'no-preference' });
    await page.evaluate(() => { localStorage.removeItem('roop3d-motion-v1'); });
    await page.reload();
    expect(await page.evaluate(() => document.documentElement.dataset.motion)).toBe('full');
    expect(await duration()).toBeGreaterThan(1);
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await expect.poll(() => page.evaluate(() => document.documentElement.dataset.motion)).toBe('reduce');
    expect(await duration()).toBeLessThan(0.01);
    // "Never reduce" wins over the system setting; "Always reduce" wins when the system allows motion.
    await page.getByRole('button', { name: 'Guide', exact: true }).click();
    await page.getByRole('tab', { name: 'Display & keyboard' }).click();
    const select = page.getByLabel('Reduce motion');
    await select.selectOption('full');
    expect(await page.evaluate(() => document.documentElement.dataset.motion)).toBe('full');
    expect(await duration()).toBeGreaterThan(1);
    await page.emulateMedia({ reducedMotion: 'no-preference' });
    await select.selectOption('reduce');
    expect(await page.evaluate(() => document.documentElement.dataset.motion)).toBe('reduce');
    expect(await duration()).toBeLessThan(0.01);
    await expect(page.getByTestId('a11y-controls')).toContainText('Motion is currently reduced');
    await page.reload();
    expect(await page.evaluate(() => document.documentElement.dataset.motion)).toBe('reduce');
  });

  test('scalable UI: the layout reflows at 150% zoom and the interface-size control works', async ({ page }) => {
    await boot(page, { openProject: true, width: 1280, height: 800 });
    await page.keyboard.press('Control+Equal');
    await page.keyboard.press('Control+Equal');
    const zoom = await page.evaluate(() => parseFloat(localStorage.getItem('roop3d_zoom') || '1'));
    expect(zoom).toBeGreaterThan(1.05);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
    await page.getByRole('button', { name: /Reset zoom/ }).click();
    expect(await page.evaluate(() => localStorage.getItem('roop3d_zoom'))).toBe('1');
    // 200% (a 640 CSS px wide viewport) must still not scroll sideways: WCAG 1.4.10 reflow.
    await page.setViewportSize({ width: 640, height: 800 });
    for (const screen of ['home', 'faceswap', 'facemgr', 'settings']) {
      await page.evaluate(hash => { location.hash = hash; }, `#/${screen}`);
      await page.waitForTimeout(250);
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), `${screen} at 640px`).toBe(true);
    }
  });

  for (const screen of ['home', 'faceswap', 'facemgr', 'settings'] as const) {
    test(`axe (WCAG 2.2 AA) finds no violations on ${screen}`, async ({ page }) => {
      await boot(page, { openProject: true, route: `/#/${screen}` });
      await page.evaluate(hash => { location.hash = hash; }, `#/${screen}`);
      await page.waitForTimeout(600);
      const violations = await axeViolations(page);
      expect(violations, JSON.stringify(violations, null, 1)).toEqual([]);
    });
  }

  test('axe finds no violations in the guide, welcome and theme dialogs', async ({ page }) => {
    await boot(page, { route: '/#/home', firstRun: true });
    let violations = await axeViolations(page, '[data-testid=welcome-dialog]');
    expect(violations, `welcome ${JSON.stringify(violations, null, 1)}`).toEqual([]);
    await page.keyboard.press('Escape');
    await page.getByRole('button', { name: 'Guide', exact: true }).click();
    for (const tab of ['Workflow', 'Target faces', 'FaceSets & photos', 'Engines & hardware', 'Render presets', 'Queue & recovery', 'Display & keyboard']) {
      await page.getByRole('tab', { name: tab }).click();
      violations = await axeViolations(page, '[data-testid=guide-dialog]');
      expect(violations, `guide/${tab} ${JSON.stringify(violations, null, 1)}`).toEqual([]);
    }
  });
});

/* ------------------------------------------------------------------------------------------------ contrast across themes */
test.describe('contrast in every theme', () => {
  test('all themes keep muted, accent and danger text and button text at 4.5:1 or better', async ({ page }) => {
    await boot(page, { route: '/#/home' });
    await openThemeGallery(page);
    const names: string[] = await page.evaluate(() => [...document.querySelectorAll('[data-testid="theme-gallery"] .theme-name')].map(node => node.textContent ?? ''));
    expect(names.length).toBeGreaterThanOrEqual(37);
    const failures: string[] = [];
    for (const name of names) {
      await pickTheme(page, name);
      await page.waitForTimeout(60);
      const ratios = await page.evaluate(() => {
        const probe = document.createElement('i'); document.body.append(probe);
        const read = (token: string, property: 'color' | 'background-color' = 'color') => {
          probe.style.cssText = `position:fixed;${property}:var(${token})`;
          const m = getComputedStyle(probe).getPropertyValue(property).match(/[\d.]+/g)!.map(Number);
          return { r: m[0], g: m[1], b: m[2], a: m.length > 3 ? m[3] : 1 };
        };
        const over = (fg: any, bg: any) => ({ r: fg.r * fg.a + bg.r * (1 - fg.a), g: fg.g * fg.a + bg.g * (1 - fg.a), b: fg.b * fg.a + bg.b * (1 - fg.a) });
        const lum = (c: any) => { const f = [c.r, c.g, c.b].map(v => { const x = v / 255; return x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4; }); return 0.2126 * f[0] + 0.7152 * f[1] + 0.0722 * f[2]; };
        const ratio = (a: any, b: any) => { const x = lum(a), y = lum(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); };
        const bg = read('--bg', 'background-color');
        const flat = (token: string) => { const c = read(token, 'background-color'); return c.a < 1 ? over(c, bg) : c; };
        const grounds: Record<string, any> = { bg, surface: flat('--surface'), raised: flat('--raised'), soft: flat('--soft') };
        const out: Record<string, number> = {};
        for (const [ground, value] of Object.entries(grounds)) for (const token of ['--text', '--muted', '--accent', '--danger']) out[`${token}/${ground}`] = ratio(over(read(token), value), value);
        out['--ink/--accent'] = ratio(over(read('--ink'), flat('--accent')), flat('--accent'));
        probe.remove();
        return out;
      });
      const bad = Object.entries(ratios).filter(([, value]) => value < 4.5).map(([key, value]) => `${key}=${value.toFixed(2)}`);
      if (bad.length) failures.push(`${name}: ${bad.join(' ')}`);
    }
    expect(failures, failures.join('\n')).toEqual([]);
  });

  for (const theme of ['Glass Light', 'AMOLED Black', 'Arctic Light']) {
    test(`axe (WCAG 2.2 AA) finds no violations on the workstation in ${theme}`, async ({ page }) => {
      await boot(page, { openProject: true });
      await openThemeGallery(page);
      await pickTheme(page, theme);
      await page.keyboard.press('Escape');
      await page.waitForTimeout(400);
      const violations = await axeViolations(page);
      expect(violations, JSON.stringify(violations, null, 1)).toEqual([]);
    });
  }
});

/* ------------------------------------------------------------------------------------------------ responsive */
test.describe('responsive behaviour and visual sweep', () => {
  for (const view of VIEWPORTS) {
    test(`${view.name}: no screen scrolls sideways or clips, and the preview stays large (screenshots)`, async ({ page }) => {
      await boot(page, { openProject: true, width: view.width, height: view.height });
      const problems: string[] = [];
      for (const screen of SCREENS) {
        await page.evaluate(hash => { location.hash = hash; }, `#/${screen}`);
        await page.waitForTimeout(350);
        const overflow = await page.evaluate(() => ({ scrollW: document.documentElement.scrollWidth, innerW: innerWidth }));
        if (overflow.scrollW > overflow.innerW + 1) problems.push(`${screen}: page is ${overflow.scrollW}px wide in a ${overflow.innerW}px window`);
        // Clipped content: any visible control whose box starts left of the window or ends right of it (scroll containers excluded).
        const clipped = await page.evaluate(() => {
          const out: string[] = [];
          for (const el of document.querySelectorAll<HTMLElement>('button,a[href],select,input:not([type=hidden])')) {
            if (el.closest('[hidden],[inert],.pipeline-flow,.screen-nav,.run-steps,.studio-timeline-slot,.timeline-scroll,[data-scroll],.guide-tabs,.virtual-list,.studio-timeline')) continue;
            const r = el.getBoundingClientRect(), s = getComputedStyle(el);
            if (!r.width || s.visibility === 'hidden' || s.display === 'none') continue;
            if (r.left < -1 || r.right > innerWidth + 1) out.push(`${el.tagName}.${String(el.className).slice(0, 30)} "${(el.textContent || '').trim().slice(0, 20)}" ${Math.round(r.left)}..${Math.round(r.right)}`);
          }
          return out.slice(0, 5);
        });
        if (clipped.length) problems.push(`${screen}: clipped controls ${clipped.join(' | ')}`);
        await page.screenshot({ path: `test-results/stage19/${view.name}-${screen}.png` });
      }
      expect(problems, problems.join('\n')).toEqual([]);

      await page.evaluate(() => { location.hash = '#/faceswap'; });
      await page.waitForTimeout(500);
      const preview = page.locator('.studio-center .viewport').first();
      await expect(preview).toBeVisible();
      const box = (await preview.boundingBox())!;
      // The preview is the point of the screen: wide, tall enough to judge a face, and not pushed below the fold.
      expect(box.width, 'preview width').toBeGreaterThanOrEqual(Math.min(view.width * 0.55, 560));
      expect(box.height, 'preview height').toBeGreaterThanOrEqual(view.height >= 800 ? 340 : 300);
      if (view.width < 1280) {
        expect(box.y, 'preview starts in the upper half of the window').toBeLessThan(view.height * 0.5);
        if (view.height <= 800) expect(box.y + box.height, 'preview fully visible without scrolling').toBeLessThanOrEqual(view.height + 12);
      }
    });
  }

  test('the preview request does not depend on the window size (no resolution traded for responsiveness)', async ({ browser }) => {
    const sample = async (width: number, height: number) => {
      const context = await browser.newContext({ viewport: { width, height } });
      const page = await context.newPage();
      const { seen } = await boot(page, { openProject: true, width, height });
      await page.waitForTimeout(1200);
      // The filmstrip asks for as many thumbnails as fit the width; the playhead frame itself must never change.
      const result = { previews: JSON.stringify(seen.previews.slice(0, 2)), frames: seen.previewFrames.filter(url => /timestamp=0$/.test(url)).slice(0, 1) };
      await context.close();
      return result;
    };
    const desktop = await sample(1680, 1050), tablet = await sample(820, 1180), laptop = await sample(1024, 768);
    expect(desktop.previews.length).toBeGreaterThan(2);
    expect(desktop.frames.length).toBe(1);
    expect(tablet.previews).toBe(desktop.previews);
    expect(laptop.previews).toBe(desktop.previews);
    expect(tablet.frames).toEqual(desktop.frames);
    expect(laptop.frames).toEqual(desktop.frames);
    for (const url of desktop.frames) expect(url).not.toMatch(/width|height|scale|size/i);
  });

  test('Face Manager keeps all three panels reachable on a tablet', async ({ page }) => {
    await boot(page, { route: '/#/facemgr', width: 820, height: 1180 });
    await expect(page.locator('.facemgr-sphere-panel')).toBeVisible();
    const box = (await page.locator('.facemgr-sphere-panel').boundingBox())!;
    expect(box.x + box.width).toBeLessThanOrEqual(820 + 1);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  });
});
