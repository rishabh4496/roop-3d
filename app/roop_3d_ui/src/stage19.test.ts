// @vitest-environment jsdom
import { describe, expect, it, beforeEach } from 'vitest';
import { ERROR_GUIDES, explainError, isOfflineError } from './components/guide/errorGuide';
import { describeCoverage, parseRequest } from './components/guide/coverageGuide';
import { ENGINE_STATUS, HARDWARE_PROFILES, JOB_STATE_HELP, PRESET_GUIDANCE, QUEUE_GUIDE, READINESS, SOURCE_PHOTO_RULES, VIEW_REASONS, WORKFLOW_STEPS, recommendEngine } from './components/guide/helpContent';
import { markOnboardingSeen, onboardingSeen, workflowProgress } from './components/guide/workflowState';
import { DEFAULT_EXPORT } from './components/export/VfxExportPanel';
import { normalizeExport } from './components/export/exportOptions';
import { applyMotionPref, getMotionPref, reducedMotion, setMotionPref } from './utils/motionPrefs';
import { AA, contrast, enforce, parseColor, readable, type RGB, type ThemeColors } from './utils/themeContrast';
import { ApiError } from './api';
import { PIPELINE_NODES } from './components/workstation/pipeline';
import { DEFAULT_KEYS, KEY_GROUPS, KEY_LABELS } from './components/controls/Hotkeys';

const rgb = (hex: string): RGB => { const c = parseColor(hex)!; return { r: c.r, g: c.g, b: c.b }; };

describe('error guidance', () => {
  it('maps the real backend codes to a title, a cause and at least one step', () => {
    for (const [code, guide] of Object.entries(ERROR_GUIDES)) {
      expect(guide.title, code).not.toBe('');
      expect(guide.cause, code).not.toBe('');
      expect(guide.steps.length, code).toBeGreaterThan(0);
    }
    expect(explainError({ code: 'DISK_FULL' })?.title).toBe('The disk is full');
    expect(explainError({ code: 'LICENSE_UNVERIFIED' })?.steps.join(' ')).toContain('docs/OFFLINE_MODELS.md');
  });
  it('falls back by code family, by a code named inside the message, then by browser failure text, and never invents', () => {
    expect(explainError({ code: 'JOB_CHECKPOINT_FRAMES_MISSING' })).toBe(ERROR_GUIDES.JOB_CHECKPOINT_INVALID);
    expect(explainError({ code: 'LICENSE_SOMETHING_NEW' })).toBe(ERROR_GUIDES.LICENSE_UNVERIFIED);
    expect(explainError({ message: 'Job failed with PIPELINE_OOM at frame 12' })).toBe(ERROR_GUIDES.PIPELINE_OOM);
    expect(explainError({ message: 'Failed to fetch' })).toBe(ERROR_GUIDES.SERVICE_OFFLINE);
    expect(explainError({ message: 'Local service returned 502' })).toBe(ERROR_GUIDES.SERVICE_OFFLINE);
    expect(explainError({ code: 'TOTALLY_UNKNOWN_CODE', message: 'Something odd' })).toBeNull();
    expect(explainError({})).toBeNull();
  });
  it('offers Reconnect only for connectivity problems', () => {
    expect(isOfflineError({ message: 'Failed to fetch' })).toBe(true);
    expect(isOfflineError({ code: 'LICENSE_UNVERIFIED', message: 'x' })).toBe(false);
    expect(isOfflineError({ message: 'No faces' })).toBe(false);
  });
  it('ApiError keeps the plain Error message and carries the code', () => {
    const error = new ApiError('Disk is full', 'DISK_FULL', 507, 'r1');
    expect(error).toBeInstanceOf(Error);
    expect(error.message).toBe('Disk is full');
    expect([error.code, error.status, error.requestId, error.name]).toEqual(['DISK_FULL', 507, 'r1', 'ApiError']);
  });
});

describe('workflow progress', () => {
  const base = { projectSaved: false, target: '', faceSetCount: 0, inspected: false, mappingCount: 0, previewSeen: false, jobs: [] as any[] };
  it('starts at the target video and counts six required steps', () => {
    const progress = workflowProgress(base);
    expect(progress.next?.id).toBe('target');
    expect(progress.nextNumber).toBe(1);
    expect([progress.doneCount, progress.total]).toEqual([0, 6]);
    expect(progress.steps.map(step => step.id)).toEqual(WORKFLOW_STEPS.map(step => step.id));
  });
  it('is order independent for a FaceSet that already exists and reports the true position of the next step', () => {
    const progress = workflowProgress({ ...base, faceSetCount: 3 });
    expect(progress.next?.id).toBe('target');
    expect(progress.nextNumber).toBe(1);   // not doneCount + 1
    expect(progress.doneCount).toBe(1);
    expect(progress.steps.find(step => step.id === 'source')?.state).toBe('done');
  });
  it('walks the steps in order and finishes only when a job exists for this target', () => {
    const target = 'roop_3d/.runtime/media/t.mp4';
    const job = (state: string, path = target) => ({ state, request: { target_path: path } }) as any;
    const ready = { ...base, target, faceSetCount: 1, inspected: true, mappingCount: 1, previewSeen: true };
    expect(workflowProgress(ready).next?.id).toBe('render');
    expect(workflowProgress({ ...ready, jobs: [job('failed')] }).next?.id).toBe('render');
    expect(workflowProgress({ ...ready, jobs: [job('queued', 'another.mp4')] }).next?.id).toBe('render');
    const done = workflowProgress({ ...ready, jobs: [job('running')] });
    expect(done.next).toBeNull();
    expect(done.nextNumber).toBe(0);
    expect(done.doneCount).toBe(6);
  });
  it('treats the project step as recommended, never blocking', () => {
    const progress = workflowProgress({ ...base, target: 'a.mp4' });
    expect(progress.steps.find(step => step.id === 'project')?.state).toBe('todo');
    expect(progress.next?.id).toBe('source');
  });
  it('every step explains its goal, how, what done looks like and where to go', () => {
    for (const step of WORKFLOW_STEPS) { expect(step.goal).not.toBe(''); expect(step.how.length).toBeGreaterThan(0); expect(step.check).not.toBe(''); expect(step.route).toMatch(/^#\//); }
  });
  describe('onboarding flag', () => {
    beforeEach(() => localStorage.clear());
    it('is unseen until marked, and a blocked store never nags', () => {
      expect(onboardingSeen()).toBe(false);
      markOnboardingSeen();
      expect(onboardingSeen()).toBe(true);
    });
  });
});

describe('FaceSet coverage reading', () => {
  const empty = () => [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]];
  it('names what is missing, most important first', () => {
    expect(describeCoverage(empty()).text).toMatch(/No views/);
    const frontOnly = empty(); frontOnly[1][2] = 3;
    const front = describeCoverage(frontOnly);
    expect(front.missingRequired).toEqual(['three-quarter left (−45°)', 'three-quarter right (+45°)']);
    expect(front.text).toBe('1 of 15 angles covered. Still needed: three-quarter left (−45°) and three-quarter right (+45°).');
    const required = empty(); required[1][1] = 1; required[1][2] = 1; required[0][3] = 1;
    const partial = describeCoverage(required);
    expect(partial.missingRequired).toEqual([]);
    expect(partial.missingRecommended).toEqual(['left profile (−75°)', 'right profile (+75°)']);
    expect(partial.text).toContain('required angles are all present');
    const full = empty(); for (const column of [0, 1, 2, 3, 4]) full[1][column] = 1;
    expect(describeCoverage(full).text).toBe('5 of 15 angles covered, including both profiles.');
  });
  it('reads the backend request priorities', () => {
    expect(parseRequest('[required] Add a three-quarter portrait')).toEqual({ level: 'required', text: 'Add a three-quarter portrait' });
    expect(parseRequest('[Recommended] Add a side profile').level).toBe('recommended');
    expect(parseRequest('[blocking] Remove other people').level).toBe('blocking');
    expect(parseRequest('No prefix at all')).toEqual({ level: 'info', text: 'No prefix at all' });
  });
});

describe('help content stays tied to the application', () => {
  it('pins the photo-quality numbers the backend applies (reconstruction/faceset_quality.py)', () => {
    const text = SOURCE_PHOTO_RULES.map(rule => `${rule.rule} ${rule.detail}`).join(' ');
    for (const fact of ['0.03', '40 pixels', '35%', '0.30']) expect(text, fact).toContain(fact);
    expect(Object.keys(VIEW_REASONS)).toEqual(expect.arrayContaining(['TOO_BLURRY', 'FACE_TOO_SMALL', 'FACE_TRUNCATED_BY_IMAGE_EDGE', 'DUPLICATE_IMAGE', 'IDENTITY_OUTLIER']));
    expect(Object.keys(READINESS)).toEqual(['insufficient', 'blocked_inconsistent_identity', 'front_only', 'partial_coverage', 'full_coverage']);
  });
  it('covers every hardware profile, preset and job state the app can report', () => {
    expect(Object.keys(HARDWARE_PROFILES).sort()).toEqual(['12gb', '6gb', 'auto', 'cpu']);
    expect(Object.keys(PRESET_GUIDANCE).sort()).toEqual(['archival_hevc10', 'color_preserving', 'high_quality_mp4', 'image_sequence', 'lossless_intermediate', 'mov_workflow', 'standard_mp4']);
    expect(Object.keys(JOB_STATE_HELP).sort()).toEqual(['cancelled', 'cancelling', 'failed', 'interrupted', 'paused', 'pausing', 'preparing', 'queued', 'running', 'succeeded']);
    expect(Object.keys(ENGINE_STATUS)).toEqual(expect.arrayContaining(['READY_FOR_INFERENCE', 'LICENSE_RESTRICTED', 'NOT_INSTALLED', 'INCOMPATIBLE']));
  });
  it('is honest that a resume replays the analysis from the first frame', () => {
    expect(QUEUE_GUIDE.join(' ')).toContain('replays the analysis from the first frame');
  });
});

describe('engine recommendation', () => {
  const engine = (id: string, status: string, selectable: boolean) => ({ engine_id: id, name: id, status, selectable });
  it('recommends only a selectable engine, preferring one that is ready', () => {
    const { engine: pick } = recommendEngine([engine('x', 'NOT_INSTALLED', false), engine('a', 'LICENSE_RESTRICTED', true), engine('b', 'READY_FOR_INFERENCE', true)]);
    expect(pick?.engine_id).toBe('b');
    expect(recommendEngine([engine('a', 'LICENSE_RESTRICTED', true)]).reason).toContain('licence');
  });
  it('says so plainly when nothing can run', () => {
    const result = recommendEngine([engine('x', 'NOT_INSTALLED', false)]);
    expect(result.engine).toBeNull();
    expect(result.reason).toContain('Environment & Offline Manager');
  });
});

describe('export options normalisation', () => {
  it('serialises "nothing chosen" exactly like the original default, with or without the server defaults filled in', () => {
    const echoed = { ...DEFAULT_EXPORT, profile: null, hdr_policy: 'tonemap_sdr', subtitles: 'copy', sequence_format: 'png' } as any;
    expect(JSON.stringify(normalizeExport(echoed))).toBe(JSON.stringify(DEFAULT_EXPORT));
    expect(JSON.stringify(normalizeExport({ ...DEFAULT_EXPORT, profile: 'standard_mp4' } as any))).toBe(JSON.stringify(DEFAULT_EXPORT));
  });
  it('keeps real choices and drops the sequence format outside the image-sequence preset', () => {
    const chosen = normalizeExport({ ...DEFAULT_EXPORT, profile: 'high_quality_mp4', hdr_policy: 'reject', subtitles: 'drop', sequence_format: 'tiff' } as any);
    expect(chosen).toMatchObject({ profile: 'high_quality_mp4', hdr_policy: 'reject', subtitles: 'drop' });
    expect(chosen.sequence_format).toBeUndefined();
    expect(normalizeExport({ ...DEFAULT_EXPORT, profile: 'image_sequence', sequence_format: 'dpx' } as any).sequence_format).toBe('dpx');
  });
});

describe('motion preference', () => {
  beforeEach(() => { localStorage.clear(); document.documentElement.removeAttribute('data-motion'); });
  it('follows the system by default and the user override wins in both directions', () => {
    expect(getMotionPref()).toBe('system');
    setMotionPref('reduce');
    expect([getMotionPref(), reducedMotion(), document.documentElement.dataset.motion]).toEqual(['reduce', true, 'reduce']);
    setMotionPref('full');
    expect([getMotionPref(), reducedMotion(), document.documentElement.dataset.motion]).toEqual(['full', false, 'full']);
    setMotionPref('system');
    expect(getMotionPref()).toBe('system');
    applyMotionPref();
    expect(document.documentElement.dataset.motion).toBeDefined();
  });
});

describe('theme contrast enforcement', () => {
  const dark: ThemeColors = { bg: rgb('#111719'), surface: rgb('#171e20'), raised: rgb('#1d2628'), soft: rgb('#1e3831'), text: rgb('#dce4e4'), muted: rgb('#8d9c9f'), accent: rgb('#91e6c9'), ink: rgb('#132b25'), danger: rgb('#ffad9a') };
  it('measures with the WCAG formula', () => {
    expect(contrast(rgb('#000000'), rgb('#ffffff'))).toBeCloseTo(21, 1);
    expect(contrast(rgb('#8d9c9f'), rgb('#1e3831'))).toBeCloseTo(4.44, 1);
  });
  it('parses hex, rgb(), rgba() and color(srgb) as browsers serialise them', () => {
    expect(parseColor('#fff')).toEqual({ r: 255, g: 255, b: 255, a: 1 });
    expect(parseColor('rgb(10, 20, 30)')).toEqual({ r: 10, g: 20, b: 30, a: 1 });
    expect(parseColor('rgba(0, 0, 0, 0.35)')?.a).toBe(0.35);
    expect(parseColor('color(srgb 1 0.5 0 / 0.5)')).toEqual({ r: 255, g: 127.5, b: 0, a: 0.5 });
    expect(parseColor('not a colour')).toBeNull();
  });
  it('leaves a passing theme untouched and fixes only the failing token', () => {
    expect(enforce({ ...dark, muted: rgb('#a7b6b9') })).toEqual({});
    const fixed = enforce(dark);
    expect(Object.keys(fixed)).toEqual(['--muted']);
  });
  it('makes every adjusted token reach AA against all four surfaces, in light and dark themes', () => {
    const light: ThemeColors = { bg: rgb('#e5e9f0'), surface: rgb('#f5f7fa'), raised: rgb('#e9eef2'), soft: rgb('#dde2e6'), text: rgb('#1f2937'), muted: rgb('#8d9c9f'), accent: rgb('#E94560'), ink: rgb('#132b25'), danger: rgb('#F87171') };
    for (const colors of [dark, light]) {
      const grounds = [colors.bg, colors.surface, colors.raised, colors.soft];
      for (const [token, original] of [['--muted', colors.muted], ['--accent', colors.accent], ['--danger', colors.danger]] as const) {
        const value = enforce(colors)[token] ? rgb(enforce(colors)[token]!.replace(/rgb\((\d+), (\d+), (\d+)\)/, (_m, r, g, b) => `#${[r, g, b].map(n => (+n).toString(16).padStart(2, '0')).join('')}`)) : original;
        for (const ground of grounds) expect(contrast(value, ground), `${token}`).toBeGreaterThanOrEqual(AA - 0.05);
      }
    }
  });
  it('leaves button text readable on the (possibly adjusted) accent fill', () => {
    const toRgb = (css: string): RGB => { const m = css.match(/\d+/g)!.map(Number); return { r: m[0], g: m[1], b: m[2] }; };
    for (const accent of ['#6366f1', '#E94560', '#ff007f', '#5e81ac', '#91e6c9']) {
      for (const ink of ['#132b25', '#ffffff', '#111111']) {
        const fixed = enforce({ ...dark, accent: rgb(accent), ink: rgb(ink) });
        const fill = fixed['--accent'] ? toRgb(fixed['--accent']) : rgb(accent), text = fixed['--ink'] ? toRgb(fixed['--ink']) : rgb(ink);
        expect(contrast(text, fill), `${ink} on ${accent}`).toBeGreaterThanOrEqual(AA - 0.05);
      }
    }
  });
  it('moves a failing foreground toward the target and never past it', () => {
    expect(readable(rgb('#777777'), [rgb('#ffffff')], rgb('#000000'))).not.toEqual(rgb('#777777'));
    expect(readable(rgb('#777777'), [rgb('#777777')], rgb('#777777'))).toEqual(rgb('#777777'));   // unreachable: ends at the target, never beyond
  });
});

describe('consistent controls', () => {
  it('has one buffer shortcut per preview pipeline node, in order, labelled with that node', () => {
    PIPELINE_NODES.forEach((node, index) => {
      const action = `buffer${index + 1}`;
      expect(DEFAULT_KEYS[action], `${action} has a default key`).toBe(String(index + 1));
      expect(KEY_LABELS[action], action).toContain(node.label.replace(' (SH)', ''));
    });
    expect(Object.keys(DEFAULT_KEYS).filter(key => key.startsWith('buffer'))).toHaveLength(PIPELINE_NODES.length);
  });
  it('lists every shortcut in exactly one group', () => {
    const grouped = KEY_GROUPS.flatMap(group => group.actions);
    expect([...grouped].sort()).toEqual(Object.keys(DEFAULT_KEYS).sort());
    expect(new Set(grouped).size).toBe(grouped.length);
    expect(new Set(Object.values(DEFAULT_KEYS)).size).toBe(Object.keys(DEFAULT_KEYS).length);   // no two actions share a key
  });
});
