// @vitest-environment jsdom
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import BatchMatrixScreen from './screens/BatchMatrixScreen';
import HistoryScreen from './screens/HistoryScreen';
import type { FaceSet, Project } from './contracts';

const BatchMatrix = BatchMatrixScreen as React.ComponentType<any>;
const History = HistoryScreen as React.ComponentType<any>;

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
let host: HTMLDivElement, root: Root;

beforeEach(() => {
  localStorage.clear();
  history.replaceState(null, '', '/#/batch');
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
  vi.restoreAllMocks();
});

const mount = async (element: React.ReactNode) => {
  await act(async () => root.render(element));
};

const click = async (element: Element | null) => {
  expect(element).toBeTruthy();
  await act(async () => (element as HTMLElement).click());
};

const input = async (element: HTMLInputElement, value: string) => {
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(element, value);
    element.dispatchEvent(new Event('change', { bubbles: true }));
  });
};

const mockSources: FaceSet[] = [
  { faceset_id: 'src_alice', name: 'Alice Source', item_count: 14 },
  { faceset_id: 'src_bob', name: 'Bob Source', item_count: 8 },
];

const mockProjects: Project[] = [
  {
    project_id: '11111111111111111111111111111111',
    name: 'Project Verified Alpha',
    request: {
      target_path: 'roop_3d/.runtime/media/interview.mp4',
      inspection_id: 'ins_1',
      face_mappings: [{ track_id: 'track_0000', faceset_id: 'src_alice' }],
      processing: 'local',
      remote_consent: false,
      trim_start_seconds: 0,
      trim_end_seconds: 10,
    },
    effective_settings: {},
    last_completed_frame: -1,
    checkpoint_job_id: null,
    saved_at: '2026-11-01T00:00:00Z',
    inputs_verified: true,
    status: 'Inputs Verified',
  },
  {
    project_id: '22222222222222222222222222222222',
    name: 'Interrupted Render Beta',
    request: {
      target_path: 'roop_3d/.runtime/media/action.mp4',
      inspection_id: 'ins_2',
      face_mappings: [{ track_id: 'track_0000', faceset_id: 'src_bob' }],
      processing: 'local',
      remote_consent: false,
      trim_start_seconds: 5,
      trim_end_seconds: 35,
    },
    effective_settings: {},
    last_completed_frame: 45,
    checkpoint_job_id: 'job_interrupted_99',
    saved_at: '2026-11-01T01:00:00Z',
    inputs_verified: true,
    status: 'Inputs Verified',
    checkpoint: {
      job_id: 'job_interrupted_99',
      completed_frame: 45,
      completed_segment: 1,
      total_segments: 3,
      execution_provider: 'CUDAExecutionProvider',
      input_hashes: { target: 'sha_target_123' },
      faceset_ids: ['src_bob'],
      pose_cache: { track_0000: [0.1, 0.2, 0.3] },
      keyframe_curves: { swap_weight: [] },
      has_binary_checkpoint: true,
    },
  },
  {
    project_id: '33333333333333333333333333333333',
    name: 'Missing Media Gamma',
    request: {
      target_path: 'roop_3d/.runtime/media/deleted_clip.mp4',
      inspection_id: 'ins_3',
      face_mappings: [{ track_id: 'track_0000', faceset_id: 'src_alice' }],
      processing: 'local',
      remote_consent: false,
      trim_start_seconds: null,
      trim_end_seconds: null,
    },
    effective_settings: {},
    last_completed_frame: 10,
    checkpoint_job_id: null,
    saved_at: '2026-11-01T02:00:00Z',
    inputs_verified: false,
    status: 'Missing Source Media',
    verification_reason: 'Target video deleted from disk',
  },
];

describe('BatchMatrixScreen', () => {
  it('renders all 4 batch strategies and allows switching tabs', async () => {
    await mount(
      <BatchMatrix
        sources={mockSources}
        projects={mockProjects}
        jobs={[]}
        refreshJobs={vi.fn()}
      />
    );

    expect(host.textContent).toContain('Batch Matrix Engine');
    expect(host.textContent).toContain('One-to-Many');
    expect(host.textContent).toContain('Grouped');
    expect(host.textContent).toContain('Per-File Matrix');
    expect(host.textContent).toContain('Recipe Matrix');

    // Default strategy is One-to-Many
    expect(host.textContent).toContain('Strategy A: One-to-Many');

    // Switch to Grouped
    const groupedTab = Array.from(host.querySelectorAll('button[role="tab"]')).find(b => b.textContent?.includes('Grouped'));
    await click(groupedTab!);
    expect(host.textContent).toContain('Strategy B: Grouped Target Batch');

    // Switch to Per-File Matrix
    const matrixTab = Array.from(host.querySelectorAll('button[role="tab"]')).find(b => b.textContent?.includes('Per-File Matrix'));
    await click(matrixTab!);
    expect(host.textContent).toContain('Strategy C: Per-File Cartesian Matrix Grid');

    // Switch to Recipe Matrix
    const recipeTab = Array.from(host.querySelectorAll('button[role="tab"]')).find(b => b.textContent?.includes('Recipe Matrix'));
    await click(recipeTab!);
    expect(host.textContent).toContain('Strategy D: Recipe Matrix');
    expect(host.textContent).toContain('Natural Photoreal');
    expect(host.textContent).toContain('High-Fidelity Ultra');
  });

  it('supports automatic video segment splitting toggle and options', async () => {
    await mount(
      <BatchMatrix
        sources={mockSources}
        projects={mockProjects}
        jobs={[]}
        refreshJobs={vi.fn()}
      />
    );

    const splitToggle = host.querySelector<HTMLInputElement>('.split-toggle-label input');
    expect(splitToggle).toBeTruthy();
    expect(splitToggle?.checked).toBe(false);

    // Toggle ON segment splitting
    await click(splitToggle);
    expect(splitToggle?.checked).toBe(true);
    expect(host.textContent).toContain('Max Segment Duration (seconds)');
    expect(host.textContent).toContain('Lossless FFmpeg stream-copy concatenation on finish');
  });

  it('operates Cartesian Per-File Matrix with intersection cell toggling and bulk controls', async () => {
    await mount(
      <BatchMatrix
        sources={mockSources}
        projects={mockProjects}
        jobs={[]}
        refreshJobs={vi.fn()}
      />
    );

    // Navigate to Matrix tab
    const matrixTab = Array.from(host.querySelectorAll('button[role="tab"]')).find(b => b.textContent?.includes('Per-File Matrix'));
    await click(matrixTab!);

    // Initially 0 cells selected
    expect(host.textContent).toContain('0 / 6 Cells Active');

    // Click "Select All"
    const selectAllBtn = Array.from(host.querySelectorAll('button')).find(b => b.textContent === 'Select All');
    await click(selectAllBtn!);
    expect(host.textContent).toContain('6 / 6 Cells Active');

    // Click "Clear All"
    const clearBtn = Array.from(host.querySelectorAll('button')).find(b => b.textContent === 'Clear All');
    await click(clearBtn!);
    expect(host.textContent).toContain('0 / 6 Cells Active');

    // Click individual matrix cell
    const cell = host.querySelector('.matrix-cell');
    await click(cell!);
    expect(host.textContent).toContain('1 / 6 Cells Active');
  });

  it('supports creating custom parameter recipes in Recipe Matrix', async () => {
    await mount(
      <BatchMatrix
        sources={mockSources}
        projects={mockProjects}
        jobs={[]}
        refreshJobs={vi.fn()}
      />
    );

    const recipeTab = Array.from(host.querySelectorAll('button[role="tab"]')).find(b => b.textContent?.includes('Recipe Matrix'));
    await click(recipeTab!);

    const createRecipeBtn = Array.from(host.querySelectorAll('button')).find(b => b.textContent?.includes('Create Custom Preset'));
    await click(createRecipeBtn!);

    expect(host.textContent).toContain('Create Custom Parameter Recipe');

    // Fill in recipe name
    const nameInput = host.querySelector<HTMLInputElement>('.project-modal input[placeholder*="Extreme"]');
    expect(nameInput).toBeTruthy();
    await input(nameInput!, 'Cinematic Film Grain');

    const addBtn = Array.from(host.querySelectorAll('.project-modal button')).find(b => b.textContent === 'Add Recipe');
    await click(addBtn!);

    expect(host.textContent).toContain('Cinematic Film Grain');
  });
});

describe('HistoryScreen', () => {
  it('displays projects with validation badges (Inputs Verified vs Missing Source Media)', async () => {
    // Mock the validate endpoint response
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url: any) => {
      const urlStr = String(url);
      if (urlStr.includes('/validate')) {
        const id = urlStr.split('/projects/')[1]?.split('/')[0];
        const match = mockProjects.find(p => p.project_id === id);
        return {
          ok: true,
          status: 200,
          json: async () => ({
            project_id: id,
            name: match?.name,
            inputs_verified: match?.inputs_verified,
            status: match?.status,
            verification_reason: match?.verification_reason,
            last_completed_frame: match?.last_completed_frame,
            checkpoint: match?.checkpoint,
          }),
        } as any;
      }
      return { ok: true, status: 200, json: async () => ({}) } as any;
    });

    await mount(
      <History
        projects={mockProjects}
        jobs={[]}
        sources={mockSources}
        openProject={vi.fn()}
        refreshProjects={vi.fn()}
        refreshJobs={vi.fn()}
      />
    );

    expect(host.textContent).toContain('Project Checkpoints & History');
    expect(host.textContent).toContain('Project Verified Alpha');
    expect(host.textContent).toContain('Interrupted Render Beta');
    expect(host.textContent).toContain('Missing Media Gamma');

    // Validation Badges
    expect(host.innerHTML).toContain('Inputs Verified');
    expect(host.innerHTML).toContain('Missing Source Media');

    // Resumable badge on interrupted project
    expect(host.textContent).toContain('RESUMABLE AT FRAME 45');
  });

  it('triggers Load & Resume Project action for interrupted projects', async () => {
    const openProjectMock = vi.fn().mockResolvedValue(undefined);
    const postMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ job_id: 'job_interrupted_99', state: 'queued' }),
    } as any);

    await mount(
      <History
        projects={mockProjects}
        jobs={[]}
        sources={mockSources}
        openProject={openProjectMock}
        refreshProjects={vi.fn()}
        refreshJobs={vi.fn()}
      />
    );

    const resumeBtn = Array.from(host.querySelectorAll('button')).find(b => b.textContent?.includes('Load & Resume Project'));
    expect(resumeBtn).toBeTruthy();

    await click(resumeBtn!);

    expect(openProjectMock).toHaveBeenCalledWith('22222222222222222222222222222222');
    expect(window.location.hash).toBe('#/faceswap');
  });

  it('opens checkpoint inspection modal showing atomic JSON + binary state details', async () => {
    await mount(
      <History
        projects={mockProjects}
        jobs={[]}
        sources={mockSources}
        openProject={vi.fn()}
        refreshProjects={vi.fn()}
        refreshJobs={vi.fn()}
      />
    );

    const inspectBtn = Array.from(host.querySelectorAll('button')).find(b => b.textContent === 'Inspect Checkpoint');
    expect(inspectBtn).toBeTruthy();

    await click(inspectBtn!);

    expect(host.textContent).toContain('ATOMIC STATE CHECKPOINT INSPECTION');
    expect(host.textContent).toContain('Integrity & Hash Verification');
    expect(host.textContent).toContain('Checkpoint Runtime State');
    expect(host.textContent).toContain('Raw State Manifest (.r3dp JSON)');
  });
});
