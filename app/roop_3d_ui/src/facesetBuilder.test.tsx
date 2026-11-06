// @vitest-environment jsdom
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { expect, it, vi } from 'vitest';
import FacesetBuilder from './components/facemgr/FacesetBuilder';
import { post } from './api';
vi.mock('./components/facemgr/BuilderCoverageSphere', () => ({default: () => <div>3D coverage sphere</div>}));
vi.mock('./api', () => ({ api: vi.fn(), upload: vi.fn(), post: vi.fn() }));
(globalThis as any).IS_REACT_ACT_ENVIRONMENT = true;

it('edits the active set, filters 96 bins, removes a crop, and activates the saved identity', async () => {
  const samples = [0,1].map(i => ({sample_id: `sample${i}`,bin_index: 48+i,yaw: i*30,pitch:0,roll:0,score:.9,thumbnail_base64:'AA==',occlusion_tier:'full'}));
  vi.mocked(post).mockResolvedValueOnce({job_id:'a'.repeat(32), status:'ready', progress:1,scanned:2,error:null,clusters:[{rank:0,detections:2,coverage:2,samples,selected_ids:['sample0','sample1']}]}).mockResolvedValueOnce({faceset_id:'existing'});
  const saved = vi.fn(); const host = document.createElement('div'); document.body.append(host); const root = createRoot(host);
  await act(async () => root.render(<FacesetBuilder sourceId="existing" onSaved={saved}/>));
  const click = async (text: string) => { const button = [...host.querySelectorAll('button')].find(b => b.textContent?.includes(text)); expect(button).toBeTruthy(); await act(async () => button!.click()); };
  await click('Build / Edit'); await click('Edit active');
  expect(post).toHaveBeenCalledWith('/faceset/extract-from-media', {image_paths:[],faceset_id:'existing'});
  expect(host.querySelectorAll('.fsb-grid button')).toHaveLength(96);
  expect(host.textContent).toContain('Sphere Coverage: 2 / 96');
  await act(async () => (host.querySelector('.fsb-gallery button') as HTMLButtonElement).click());
  expect(host.textContent).toContain('Sphere Coverage: 1 / 96');
  await click('Save & Activate');
  expect(post).toHaveBeenLastCalledWith('/faceset/save-fsz', expect.objectContaining({faceset_id:'existing',selected_ids:['sample1']}));
  expect(saved).toHaveBeenCalledWith('existing'); expect(host.querySelector('[role=dialog]')).toBeNull();
  await act(async () => root.unmount()); host.remove();
});
