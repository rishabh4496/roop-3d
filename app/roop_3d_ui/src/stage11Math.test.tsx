// @vitest-environment jsdom
/** Stage 26 inline expression math: typed into any numeric parameter input, evaluated safely, with an instant preview. */
import { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import NumericInput, { constrain, evaluateExpression } from './components/controls/NumericInput';
import SettingControl from './components/controls/SettingControl';
import { SETTINGS } from './components/controls/settingsRegistry';
import { FinishingControls, DEFAULT_EXPORT } from './components/export/VfxExportPanel';

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
let host: HTMLDivElement, root: Root;
beforeEach(() => { host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
afterEach(async () => { await act(async () => root.unmount()); host.remove(); });
const type = async (element: HTMLInputElement, value: string) => act(async () => { Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(element, value); element.dispatchEvent(new Event('input', { bubbles: true })); });
const enter = async (element: Element) => act(async () => { element.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true })); });
const field = (label: string) => host.querySelector<HTMLInputElement>(`input[aria-label="${label}"]`)!;
const spec = (id: string) => SETTINGS.find(item => item.id === id)!;

describe('evaluateExpression', () => {
  it('evaluates ordinary arithmetic: 0.4 * 1.5 -> 0.6', () => {
    expect(constrain(evaluateExpression('0.4 * 1.5'))).toBe(0.6);
    expect(evaluateExpression('(1 + 2) * 3 / 9')).toBe(1);
    expect(evaluateExpression('2 × 3 ÷ 4 − 1')).toBe(0.5);                    // × ÷ − typed from a numeric keypad / phone
  });
  it('treats a leading + * / as relative to the field value, so "*1.25", "/2" and "+0.05" work in place', () => {
    expect(constrain(evaluateExpression('*1.25', 0.4))).toBe(0.5);
    expect(constrain(evaluateExpression('/2', 0.4))).toBe(0.2);
    expect(constrain(evaluateExpression('+0.05', 0.4))).toBe(0.45);
    expect(constrain(evaluateExpression('+-0.1', 0.4))).toBe(0.3);
    expect(constrain(evaluateExpression('*(1 + 0.25)', 0.4))).toBe(0.5);
    expect(evaluateExpression('*2', -3)).toBe(-6);
    expect(evaluateExpression('*10', 1e-7)).toBeCloseTo(1e-6, 12);          // exponent-form current values tokenize
    expect(evaluateExpression('*3', null as unknown as number)).toBe(0);     // an empty field counts as 0
  });
  it('keeps a leading minus a negative literal (angles / offsets)', () => { expect(evaluateExpression('-0.5', 0.4)).toBe(-0.5); });
  it('never executes text: only numbers, parentheses and + - * /', () => {
    for (const bad of ['window.alert(1)', 'alert`1`', '*process.exit()', '/0', '*', '1;2', '2**3', '(1+2', 'NaN', '0x10', '1'.repeat(129), '*1e999']) expect(() => evaluateExpression(bad, 1)).toThrow();
  });
});

describe('inline math across the scrub inputs', () => {
  it('SettingControl: TPS expression weight and mask feather radius', async () => {
    function Panel() {
      const [tps, setTps] = useState(0.4), [feather, setFeather] = useState(5);
      return <><SettingControl spec={spec('tps_expression_weight')} value={tps} onChange={setTps}/><SettingControl spec={spec('feather_px')} value={feather} onChange={setFeather}/></>;
    }
    await act(async () => root.render(<Panel/>));
    const tps = field('TPS expression weight'), feather = field('Mask feather');
    expect(tps.value).toBe('0.4');

    await type(tps, '0.4 * 1.5');
    expect(host.querySelector('[data-testid="expression-preview"]')?.textContent).toBe('= 0.6');   // instant evaluation while typing
    await enter(tps); expect(tps.value).toBe('0.6');
    await type(tps, '/2'); await enter(tps); expect(tps.value).toBe('0.3');
    await type(tps, '+0.05'); await enter(tps); expect(tps.value).toBe('0.35');
    await type(tps, '*10'); await enter(tps); expect(tps.value).toBe('1');                         // clamped to the 0..1 range

    await type(feather, '*1.5'); await enter(feather); expect(feather.value).toBe('7.5');
    await type(feather, '/2'); await enter(feather); expect(feather.value).toBe('3.75');
    await type(feather, '+0.25'); await enter(feather); expect(feather.value).toBe('4');
    await type(feather, '2 * 40'); await enter(feather); expect(feather.value).toBe('64');         // clamped to max
  });

  it('plain numbers show no preview chip; invalid expressions flag the field and Escape restores it', async () => {
    function Field() { const [value, setValue] = useState<number | null>(100); return <NumericInput label="Resolution" value={value} min={0} max={200} onChange={setValue}/>; }
    await act(async () => root.render(<Field/>));
    const input = field('Resolution');
    await type(input, '150'); expect(host.querySelector('[data-testid="expression-preview"]')).toBeNull();
    await type(input, '*0.5'); expect(host.querySelector('[data-testid="expression-preview"]')?.textContent).toBe('= 50');
    await type(input, '100 / 0'); await enter(input); expect(input.getAttribute('aria-invalid')).toBe('true');
    await act(async () => { input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })); }); expect(input.value).toBe('100');
  });

  it('film-grain finishing inputs accept the same expressions (and integer seeds round)', async () => {
    function Grain() { const [value, setValue] = useState(DEFAULT_EXPORT.finishing); return <><FinishingControls value={value} onChange={setValue}/><output data-grain>{JSON.stringify(value.grain)}</output></>; }
    await act(async () => root.render(<Grain/>));
    await type(field('Grain Intensity'), '0.04'); await enter(field('Grain Intensity'));
    await type(field('Grain Intensity'), '*1.5'); await enter(field('Grain Intensity'));
    await type(field('Grain Size'), '1 + 1'); await enter(field('Grain Size'));
    await type(field('Grain seed'), '10 / 4'); await enter(field('Grain seed'));
    expect(JSON.parse(host.querySelector('[data-grain]')!.textContent!)).toMatchObject({ intensity: 0.06, size: 2, seed: 3 });
  });
});
