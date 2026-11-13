// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { SettingsInspector } from './SettingsInspector';
import type { Capabilities, Config } from './contracts';

const mockCaps: Capabilities = {
  enabled_modes: ['workstation'],
  rendering_worker_available: true,
  legal_parameter_bounds: {
    type: 'object',
    properties: {
      hardware_profile: { type: 'string', enum: ['auto', 'cpu', '6gb', '12gb'], description: 'VRAM ceiling' },
      execution_threads: { type: 'integer', minimum: 1, maximum: 8 },
      execution_mode: { type: 'string', enum: ['swap', 'volumetric'] },
      temporal_beta: { type: 'number', minimum: 0, maximum: 10 },
      face_restoration_policy: { type: 'string', enum: ['off', 'gpen', 'codeformer'] },
      video_timing_policy: { type: 'string', enum: ['preserve', 'cfr'] },
      diagnostics_level: { type: 'string', enum: ['info', 'debug'] },
      port: { type: 'integer', minimum: 1024, maximum: 65535 },
    },
  },
  inspection_parameter_bounds: {},
  job_parameter_bounds: {},
  models: [
    { model_id: 'm1', name: 'gpen_bfr_512.onnx', state: 'VERIFIED_LOCAL', license_state: 'academic', inference_state: 'VERIFIED_LOCAL', sha256: 'a'.repeat(64) },
    { model_id: 'm2', name: 'CodeFormerv0.1.onnx', state: 'VERIFIED_LOCAL', license_state: 'academic', inference_state: 'VERIFIED_LOCAL', sha256: 'b'.repeat(64) },
  ],
  workstation: {
    features: {},
    controls: {
      hardware_profile: { auto: null, cpu: null, '6gb': null, '12gb': null },
      face_restoration_policy: { off: null, gpen: null, codeformer: null },
    },
    active_hardware_profile: '6gb',
    resolved_hardware_profile: '6gb',
    read_only_parameters: { port: 'Port must be changed offline' },
    disabled_parameters: {},
    job_states: [],
    job_controls: {},
    upload_chunk_bytes: 1048576,
    max_asset_bytes: 104857600,
  },
};

const mockConfig: Config = {
  hardware_profile: '6gb',
  execution_threads: 4,
  execution_mode: 'swap',
  temporal_beta: 0.04,
  face_restoration_policy: 'gpen',
  video_timing_policy: 'preserve',
  diagnostics_level: 'info',
  port: 8030,
};

describe('SettingsInspector all-visible domains & validation', () => {
  let host: HTMLDivElement, root: Root;

  beforeEach(() => {
    (globalThis as any).IS_REACT_ACT_ENVIRONMENT = true;
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    host.remove();
  });

  it('renders every schema domain and control in the default view', async () => {
    await act(async () => {
      root.render(
        <SettingsInspector
          caps={mockCaps}
          config={mockConfig}
          save={vi.fn()}
          busy={false}
        />
      );
    });

    expect(host.querySelector('.settings-domain-tabs')).toBeNull();
    for (const domain of ['compute', 'spatial', 'pipeline', 'finishing', 'ux', 'advanced']) {
      const panel = host.querySelector(`#panel-${domain}`) as HTMLElement;
      expect(panel).toBeTruthy();
      expect(panel.hidden).toBe(false);
      expect(getComputedStyle(panel).display).not.toBe('none');
    }
    for (const name of ['hardware_profile', 'execution_mode', 'temporal_beta', 'face_restoration_policy', 'video_timing_policy', 'diagnostics_level', 'port']) {
      expect(host.querySelector(`#setting-${name}`)).toBeTruthy();
    }
  });

  it('keeps controls from all groups available without selecting a category', async () => {
    await act(async () => {
      root.render(
        <SettingsInspector
          caps={mockCaps}
          config={mockConfig}
          save={vi.fn()}
          busy={false}
        />
      );
    });

    expect(host.querySelector('[role="tablist"]')).toBeNull();
    expect(host.querySelector('#panel-compute select')).toBeTruthy();
    expect(host.querySelector('#panel-spatial input')).toBeTruthy();
    expect(host.querySelector('#panel-pipeline select')).toBeTruthy();
    expect(host.querySelector('#panel-finishing select')).toBeTruthy();
    expect(host.querySelector('#panel-ux select')).toBeTruthy();
    expect(host.querySelector('#panel-advanced input')).toBeTruthy();
  });

  it('displays active and resolved hardware profile badges in Compute tab', async () => {
    await act(async () => {
      root.render(
        <SettingsInspector
          caps={mockCaps}
          config={mockConfig}
          save={vi.fn()}
          busy={false}
        />
      );
    });

    const badge = host.querySelector('.profile-badge')!;
    expect(badge.textContent).toContain('Active Profile: 6gb');
    expect(badge.textContent).toContain('Resolved: 6gb');
  });

  it('displays recorded models with verified badges', async () => {
    await act(async () => {
      root.render(
        <SettingsInspector
          caps={mockCaps}
          config={mockConfig}
          save={vi.fn()}
          busy={false}
        />
      );
    });

    const modelList = host.querySelector('.model-list')!;
    expect(modelList.textContent).toContain('Recorded model artifacts 2');
    expect(modelList.textContent).toContain('gpen_bfr_512.onnx');
    expect(modelList.textContent).toContain('CodeFormerv0.1.onnx');

    const badges = [...modelList.querySelectorAll('.model-badge')];
    expect(badges.length).toBe(2);
    expect(badges[0].textContent).toBe('VERIFIED_LOCAL');
    expect(badges[0].classList.contains('verified')).toBe(true);
  });

  it('saves modified draft settings on form submission', async () => {
    const saveMock = vi.fn();
    await act(async () => {
      root.render(
        <SettingsInspector
          caps={mockCaps}
          config={mockConfig}
          save={saveMock}
          busy={false}
        />
      );
    });

    const select = host.querySelector<HTMLSelectElement>('#setting-hardware_profile')!;
    await act(async () => {
      select.value = '12gb';
      select.dispatchEvent(new Event('change', { bubbles: true }));
    });

    const form = host.querySelector('form')!;
    await act(async () => {
      form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    });

    expect(saveMock).toHaveBeenCalledTimes(1);
    expect(saveMock).toHaveBeenCalledWith(
      expect.objectContaining({
        hardware_profile: '12gb',
      })
    );
  });
});
