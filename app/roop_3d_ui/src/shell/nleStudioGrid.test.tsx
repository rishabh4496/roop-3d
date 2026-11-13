// @vitest-environment jsdom
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { describe, expect, it, vi } from 'vitest';
import TopBar from './TopBar';
import RenderFooter from '../components/studio/RenderFooter';

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

describe('4-Zone NLE Studio Grid Components', () => {
  it('TopBar: renders title, offline isolation badge, GPU/VRAM telemetry, and Manage Models button', async () => {
    const onManageModels = vi.fn();
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);

    try {
      await act(async () => {
        root.render(
          <TopBar
            title="Workstation · Roop 3D"
            projectName="Test Scene"
            onProjectName={() => {}}
            dirty={false}
            connection="online"
            zoom={1}
            onZoom={() => {}}
            onZoomReset={() => {}}
            themeMode="dark"
            onToggleMode={() => {}}
            onThemes={() => {}}
            tourFresh={false}
            onTour={() => {}}
            onGuide={() => {}}
            layoutMode="studio"
            onLayout={() => {}}
            projectsDisabled={false}
            busy={false}
            canSave={true}
            onPalette={() => {}}
            onProjects={() => {}}
            onSave={() => {}}
            onSaveCopy={() => {}}
            onManageModels={onManageModels}
          />
        );
      });

      // 1. App Title
      const title = host.querySelector('.topbar-title');
      expect(title?.textContent).toBe('Workstation · Roop 3D');

      // 2. Offline Health / Isolation badge
      const isolationBadge = host.querySelector('[data-testid="topbar-isolation-badge"]');
      expect(isolationBadge).toBeTruthy();
      expect(isolationBadge?.textContent).toContain('LOCAL ISOLATED');

      // 3. GPU / VRAM telemetry widget
      const vramWidget = host.querySelector('[data-testid="topbar-vram"]') || host.querySelector('.topbar-telemetry-badge');
      expect(vramWidget).toBeTruthy();

      // 4. Manage Models button
      const manageBtn = host.querySelector('[data-testid="manage-models-button"]') as HTMLButtonElement;
      expect(manageBtn).toBeTruthy();
      expect(manageBtn.textContent).toContain('Manage Models');
      await act(async () => manageBtn.click());
      expect(onManageModels).toHaveBeenCalledTimes(1);
    } finally {
      await act(async () => root.unmount());
      host.remove();
    }
  });

  it('RenderFooter: sticky footer renders primary Render/Swap buttons and prominent Cancel button when render is active', async () => {
    const onStart = vi.fn();
    const onCancel = vi.fn();
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);

    const activeJob: any = {
      job_id: 'job-123',
      state: 'running',
      progress: { frame: 45, total_frames: 100, fps: 24.5, stage: 'SWAP' },
    };

    try {
      await act(async () => {
        root.render(
          <RenderFooter
            job={activeJob}
            startBlockedReason={null}
            frameBlockedReason={null}
            onStart={onStart}
            onCancel={onCancel}
          />
        );
      });

      // Active state displays Cancel button
      const cancelBtn = host.querySelector('[data-testid="render-cancel-btn"]') as HTMLButtonElement;
      expect(cancelBtn).toBeTruthy();
      expect(cancelBtn.textContent).toContain('Cancel Render');

      await act(async () => cancelBtn.click());
      expect(onCancel).toHaveBeenCalledTimes(1);

      // Status text indicates active progress
      const status = host.querySelector('[data-testid="render-status"]');
      expect(status?.textContent).toContain('45');
    } finally {
      await act(async () => root.unmount());
      host.remove();
    }
  });
});
