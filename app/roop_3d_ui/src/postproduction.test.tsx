// @vitest-environment jsdom
import { act, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { describe, expect, it, vi } from 'vitest';
import { buildTimeline, exportFCPXML, exportOTIO, exportEDL, frameTimecode } from './utils/nleExporter';
import { passPFM } from './utils/meshPassExporter';
import VfxExportPanel, { DEFAULT_EXPORT } from './components/export/VfxExportPanel';

const plan = { name: 'A & B "scene"', sourceUrl: 'file:///C:/media/a%20b.mov', swappedUrl: 'file:///C:/swap.mov', alphaUrl: 'file:///C:/alpha.mkv',
  frameRate: '24000/1001', frameCount: 240, mediaOffset: 24, vfr: false, sceneCuts: [60, 120], ranges: [[24, 100], [150, 180]] };

describe('frame-accurate timeline interchange', () => {
  it('splits all boundaries, preserves gaps and offsets trimmed media frames', () => {
    const timeline = buildTimeline(plan);
    expect(timeline.tracks[0].clips.map((c: any) => [c.start, c.end])).toEqual([[0,24],[24,60],[60,100],[100,120],[120,150],[150,180],[180,240]]);
    expect(timeline.tracks[1].clips.map((c: any) => [c.start, c.end, c.mediaIn])).toEqual([[24,60,0],[60,100,36],[150,180,126]]);
    expect(timeline.tracks[2].clips.map((c: any) => [c.start, c.end])).toEqual([[24,60],[60,100],[150,180]]);
  });
  it('writes rational XML time values, escaped names, and connected lanes', () => {
    const content = exportFCPXML(plan), doc = new DOMParser().parseFromString(content, 'application/xml');
    expect(doc.querySelector('parsererror')).toBeNull();
    expect(doc.querySelector('project')?.getAttribute('name')).toBe(plan.name);
    expect(doc.querySelector('format')?.getAttribute('frameDuration')).toBe('1001/24000s');
    const connected = doc.querySelector('asset-clip[lane="1"]')!;
    expect(connected.getAttribute('offset')).toBe('24024/24000s');
    expect(connected.getAttribute('start')).toBe('0/24000s');
    expect(doc.querySelectorAll('asset-clip[lane="2"]')).toHaveLength(3);
  });
  it('writes OTIO gaps and single-track CMX files without losing timeline positions', () => {
    const otio = JSON.parse(exportOTIO(plan));
    expect(otio.tracks.children).toHaveLength(3);
    const clips = otio.tracks.children[1].children;
    expect(clips[0].OTIO_SCHEMA).toBe('Gap.1');
    expect(clips[0].source_range.duration.value).toBe(24);
    expect(clips[1].media_references.DEFAULT_MEDIA.target_url).toBe(plan.swappedUrl);
    expect(clips[1].source_range.start_time.value).toBe(0);
    expect(clips.at(-1).source_range.duration.value).toBe(60);
    const edls = exportEDL(plan);
    expect(Object.keys(edls)).toEqual(['V1.edl', 'V2.edl', 'V3.edl']);
    expect(edls['V2.edl']).toContain('00:00:00:00 00:00:01:12 00:00:01:00 00:00:02:12');
  });
  it('handles DF minute boundaries and rejects unsupported or ambiguous timing', () => {
    expect(frameTimecode(1800, '30000/1001', true)).toBe('00:01:00:02');
    expect(frameTimecode(17982, '30000/1001', true)).toBe('00:10:00:00');
    expect(frameTimecode(3600, '60000/1001', true)).toBe('00:01:00:04');
    expect(() => buildTimeline({ ...plan, vfr: true })).toThrow('CFR');
    expect(() => buildTimeline({ ...plan, frameRate: '0/0' })).toThrow();
    expect(() => buildTimeline({ ...plan, sceneCuts: [1.5] })).toThrow();
    expect(() => buildTimeline({ ...plan, ranges: [[0, 50]] })).toThrow();
    expect(() => frameTimecode(100, '25', true)).toThrow();
    expect(buildTimeline({ ...plan, alphaUrl: null }).tracks[2].clips).toHaveLength(0);
  });
});

it('exports floating-point PFM values without display conversion or row reversal', async () => {
  const blob = passPFM(2, 1, new Float32Array([-1, .5, 1, 1, 10, 20, 30, 1]));
  // jsdom Blob has FileReader but no arrayBuffer.
  const bytes: ArrayBuffer = await new Promise(resolve => { const reader = new FileReader(); reader.onload = () => resolve(reader.result as ArrayBuffer); reader.readAsArrayBuffer(blob); });
  const header = new TextEncoder().encode('PF\n2 1\n-1.0\n');
  const values = new DataView(bytes, header.length);
  expect(values.getFloat32(0, true)).toBe(-1);
  expect(values.getFloat32(4, true)).toBe(.5);
  expect(values.getFloat32(12, true)).toBe(10);
});

it('connects pass selection and passes-only rendering while gating unavailable geometry', async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  const host = document.createElement('div'); document.body.append(host); const root = createRoot(host), queue = vi.fn();
  function Panel() { const [value, setValue] = useState<any>(DEFAULT_EXPORT); return <VfxExportPanel value={value} onChange={setValue} onQueue={queue} canQueue meshBlend={false}/>; }
  try {
    await act(async () => root.render(<Panel/>));
    const alpha = [...host.querySelectorAll('label')].find(label => label.textContent === 'Face-Canvas Alpha Matte')!.querySelector('input')!;
    const baked = [...host.querySelectorAll('label')].find(label => label.textContent === 'Final baked video')!.querySelector('input')!;
    await act(async () => alpha.click()); await act(async () => baked.click());
    const button = [...host.querySelectorAll('button')].find(item => item.textContent === 'Queue export render')!;
    expect(button.disabled).toBe(false);
    expect(host.querySelector<HTMLInputElement>('[aria-label="Grain Intensity"]')!.disabled || host.querySelector<HTMLInputElement>('[aria-label="Grain Intensity"]')!.closest('fieldset')!.disabled).toBe(true);
    const normal = [...host.querySelectorAll('label')].find(label => label.textContent === '3D Normal Map Pass')!.querySelector('input')!;
    expect(normal.disabled).toBe(true);
    await act(async () => button.click()); expect(queue).toHaveBeenCalledOnce();
    await act(async () => alpha.click()); expect(button.disabled).toBe(true);
  } finally { await act(async () => root.unmount()); host.remove(); }
});
