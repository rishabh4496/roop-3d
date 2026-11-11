// Test helpers for PrecisionRangeSlider (role="slider" thumbs, not <input type="range">). They drive it the way a user does, through the
// pointer on the track, so the code under test sees the same events as in a browser. Not imported by app code.
import { act } from 'react';

/** The slider thumb with this accessible name inside `root`. */
export const thumb = (root: ParentNode, name: string) => root.querySelector<HTMLElement>(`[role="slider"][aria-label="${name}"]`)!;

/** Presses the track at the position that corresponds to `value` (then releases), like clicking there. Snapping/clamping is the slider's job. */
export async function setSlider(slider: HTMLElement, value: number) {
  const row = slider.closest('.ul-slider__row')!;
  const track = row.querySelector<HTMLElement>('[data-testid="slider-track"]')!;
  const min = Number(slider.getAttribute('aria-valuemin')), max = Number(slider.getAttribute('aria-valuemax'));
  track.getBoundingClientRect = () => ({ left: 0, top: 0, width: 1000, height: 22, right: 1000, bottom: 22, x: 0, y: 0, toJSON() {} }) as DOMRect;
  const x = ((value - min) / (max - min)) * 1000;
  await act(async () => { track.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true, button: 0, clientX: x })); });
  await act(async () => { track.dispatchEvent(new MouseEvent('pointerup', { bubbles: true, button: 0, clientX: x })); });
}

export const sliderValue = (slider: HTMLElement) => Number(slider.getAttribute('aria-valuenow'));
export const isDisabled = (slider: HTMLElement) => slider.getAttribute('aria-disabled') === 'true';
