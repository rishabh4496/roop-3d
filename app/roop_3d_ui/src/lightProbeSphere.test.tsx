// @vitest-environment jsdom
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { describe, expect, it, vi } from 'vitest';
import LightProbeSphere from './components/three/LightProbeSphere';
import { DEFAULT_PARAMETERS } from './components/workstation/pipeline';
import { isDisabled, setSlider, sliderValue, thumb } from './sliderTestKit';

describe('LightProbeSphere', () => {
  it('renders light probe with auto-detected direction and color temperature', () => {
    Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);

    try {
      act(() => {
        root.render(
          <LightProbeSphere
            parameters={DEFAULT_PARAMETERS}
            detectedLightDirection={[0.85, 0.0, 0.52]}
            colorTemperatureK={5800}
          />
        );
      });

      expect(host.textContent).toContain('3D LIGHT PROBE & RELIGHTING');
      expect(host.textContent).toContain('5800 K');
      expect(host.textContent).toContain('Scene Auto-Detected');
      expect(host.querySelector('[aria-label="Auto-Match Target Scene 3D Lighting"]')).toBeTruthy();
      expect(host.querySelector('[aria-label="Manual 3D Virtual Key Light"]')).toBeTruthy();
    } finally {
      act(() => root.unmount());
      host.remove();
    }
  });

  it('allows switching to Manual 3D Virtual Key Light and adjusting relighting sliders', async () => {
    Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    const setParameters = vi.fn();

    try {
      act(() => {
        root.render(
          <LightProbeSphere
            parameters={DEFAULT_PARAMETERS}
            setParameters={setParameters}
          />
        );
      });

      // Switch to manual key light
      const manualBtn = host.querySelector<HTMLButtonElement>('[aria-label="Manual 3D Virtual Key Light"]')!;
      act(() => {
        manualBtn.click();
      });
      expect(setParameters).toHaveBeenCalledWith(expect.objectContaining({ lighting_mode: 'manual' }));

      // Adjust Relight Strength slider
      expect(sliderValue(thumb(host, 'Relight Strength'))).toBe(0.8); // unset parameter shows its default
      await setSlider(thumb(host, 'Relight Strength'), 0.92);
      expect(setParameters).toHaveBeenCalledWith(expect.objectContaining({ relight_strength: 0.92 }));

      // Adjust Specular Skin Sheen slider
      await setSlider(thumb(host, 'Specular Skin Sheen'), 0.45);
      expect(setParameters).toHaveBeenCalledWith(expect.objectContaining({ specular_sheen: 0.45 }));

      // Adjust Shadow Preservation slider
      await setSlider(thumb(host, 'Shadow Preservation'), 0.78);
      expect(setParameters).toHaveBeenCalledWith(expect.objectContaining({ shadow_preservation: 0.78 }));
      // the parameters prop is a constant here, so the mode switch above did not take effect: ambient fill is still gated to manual mode
      expect(isDisabled(thumb(host, 'Ambient Fill'))).toBe(true);
    } finally {
      act(() => root.unmount());
      host.remove();
    }
  });
});
