import type { Locator, Page } from '@playwright/test';

/** Current value of a PrecisionRangeSlider thumb (`role="slider"`), read from its ARIA state: they are not <input>s, so `inputValue()` does not apply. */
export const sliderValue = async (thumb: Locator) => Number(await thumb.getAttribute('aria-valuenow'));

/**
 * Sets a PrecisionRangeSlider the way a person does: click the track near the wanted value, then nudge with the arrow keys
 * (Shift = 10 steps) until the slider reports exactly `value`. `fill()` cannot be used: the thumb is a role="slider" element, not an <input>.
 */
export async function setSlider(page: Page, thumb: Locator, value: number, step = 1) {
  const min = Number(await thumb.getAttribute('aria-valuemin')), max = Number(await thumb.getAttribute('aria-valuemax'));
  const box = (await thumb.locator('xpath=..').boundingBox())!;
  await page.mouse.click(box.x + ((value - min) / (max - min)) * box.width, box.y + box.height / 2);
  await thumb.focus();
  for (let guard = 0; guard < 400; guard++) {
    const remaining = Math.round((value - (await sliderValue(thumb))) / step);
    if (remaining === 0) return;
    const big = Math.abs(remaining) >= 10;
    await page.keyboard.press(`${big ? 'Shift+' : ''}${remaining > 0 ? 'ArrowRight' : 'ArrowLeft'}`);
  }
  throw new Error(`slider did not settle on ${value}`);
}
