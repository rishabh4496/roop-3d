// @vitest-environment jsdom
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { expect, it, vi } from 'vitest';
import CommandPalette, { type Command } from './CommandPalette';
import Rail from './Rail';

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

const mount = async (node: React.ReactNode) => {
  const host = document.createElement('div'); document.body.append(host);
  const root = createRoot(host);
  await act(async () => root.render(node));
  return { host, unmount: async () => { await act(async () => root.unmount()); host.remove(); } };
};
const type = async (input: HTMLInputElement, value: string) => {
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(input, value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
};
const key = async (target: Element, k: string) => act(async () => { target.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true })); });

const commands = (run: (id: string) => void): Command[] => [
  { id: 'a', label: 'Go to Face Swap', group: 'Navigate', run: () => run('a') },
  { id: 'b', label: 'Go to Face Manager', group: 'Navigate', run: () => run('b') },
  { id: 'c', label: 'Browse themes', group: 'Appearance', keywords: 'palette colour', run: () => run('c') },
  { id: 'd', label: 'Save project', group: 'Project', disabled: true, run: () => run('d') },
];

it('command palette: labelled modal dialog, filters by label/group/keywords, hides disabled commands', async () => {
  const { host, unmount } = await mount(<CommandPalette open onClose={() => {}} commands={commands(() => {})} />);
  try {
    const dialog = document.querySelector('[role="dialog"]')!;
    expect(dialog.getAttribute('aria-modal')).toBe('true');
    expect(dialog.getAttribute('aria-label')).toBe('Command palette');
    const options = () => [...document.querySelectorAll('[role="option"]')].map(node => node.textContent);
    expect(options()).toEqual(['Go to Face Swap', 'Go to Face Manager', 'Browse themes']); // the disabled command is not offered
    const input = document.querySelector('input')! as HTMLInputElement;
    await type(input, 'face man');
    expect(options()).toEqual(['Go to Face Manager']);
    await type(input, 'colour'); // matches the hidden keyword
    expect(options()).toEqual(['Browse themes']);
    await type(input, 'zzzz');
    expect(document.body.textContent).toContain('No matching command');
  } finally { await unmount(); void host; }
});

it('command palette: arrow keys move the selection, Enter closes then runs it, Escape closes without running', async () => {
  vi.useFakeTimers();
  const ran: string[] = []; const closed = vi.fn();
  const { unmount } = await mount(<CommandPalette open onClose={closed} commands={commands(id => ran.push(id))} />);
  try {
    const dialog = document.querySelector('[role="dialog"]')!;
    expect(document.querySelector('[aria-selected="true"]')?.textContent).toBe('Go to Face Swap');
    await key(dialog, 'ArrowDown');
    expect(document.querySelector('[aria-selected="true"]')?.textContent).toBe('Go to Face Manager');
    await key(dialog, 'ArrowUp'); await key(dialog, 'ArrowUp'); // clamps at the first row
    expect(document.querySelector('[aria-selected="true"]')?.textContent).toBe('Go to Face Swap');
    await key(dialog, 'ArrowDown'); await key(dialog, 'ArrowDown');
    await key(dialog, 'Enter');
    expect(closed).toHaveBeenCalledTimes(1);
    await act(async () => { vi.runAllTimers(); });
    expect(ran).toEqual(['c']);
    await key(document.body, 'Escape');
    expect(closed).toHaveBeenCalledTimes(2);
    expect(ran).toEqual(['c']);
  } finally { vi.useRealTimers(); await unmount(); }
});

it('command palette renders nothing while closed', async () => {
  const { unmount } = await mount(<CommandPalette open={false} onClose={() => {}} commands={commands(() => {})} />);
  try { expect(document.querySelector('[role="dialog"]')).toBeNull(); } finally { await unmount(); }
});

it('rail: nine screens, marks the current one, collapses with the button and Ctrl+B, and remembers the choice', async () => {
  localStorage.clear();
  const { host, unmount } = await mount(<Rail route="#/settings" />);
  try {
    const links = [...host.querySelectorAll('nav[aria-label="Workstation screens"] a')];
    expect(links).toHaveLength(9);
    expect(links.filter(link => link.getAttribute('aria-current') === 'page').map(link => link.textContent)).toEqual(['Settings']);
    const rail = host.querySelector('.rail')!;
    const wasCollapsed = rail.classList.contains('collapsed');
    await act(async () => (host.querySelector('.rail-collapse') as HTMLButtonElement).click());
    expect(rail.classList.contains('collapsed')).toBe(!wasCollapsed);
    await act(async () => { window.dispatchEvent(new KeyboardEvent('keydown', { key: 'b', ctrlKey: true, bubbles: true, cancelable: true })); });
    expect(rail.classList.contains('collapsed')).toBe(wasCollapsed);
    expect(localStorage.getItem('roop3d-rail-collapsed')).toBe(wasCollapsed ? '1' : '0');
  } finally { await unmount(); }
});
