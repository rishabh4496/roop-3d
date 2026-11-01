// @vitest-environment jsdom
import React,{act} from 'react';
import {createRoot} from 'react-dom/client';
import {expect,it,vi} from 'vitest';
import FaceManager from './components/facemgr/FaceManager';
import {buildLivePayload} from './livePreview';
import {DEFAULT_PARAMETERS} from './components/workstation/pipeline';
vi.mock('@react-three/fiber',()=>({Canvas:()=>null}));
vi.mock('@react-three/drei',()=>({OrbitControls:()=>null}));
vi.mock('./components/three/PoseGeometryViewer',()=>({generateFibonacciSphere:()=>[{sampleCount:0}],projectViewsToLattice:()=>[{sampleCount:0}],FaceSetOrientationSphere:()=>null}));
Object.assign(globalThis,{IS_REACT_ACT_ENVIRONMENT:true});
const json=(value:unknown)=>new Response(JSON.stringify(value),{headers:{'Content-Type':'application/json'}});
it('loads durable bank assignments and reloads saved archive cells after fill',async()=>{
 const bank={revision:1,unresolved_tracks:[],clusters:[{id:'person_0000',rank:0,name:'Returning actor',track_ids:['track_0000','track_0002'],faceset_id:'alice',faces:[]}]};
 let filled=false;
 const calls:unknown[]=[];
 const fetch=vi.fn(async(url:string,init?:RequestInit)=>{
  if(url.endsWith('/bank/discover'))return json(bank);
  if(url.endsWith('/pose_matrix')) return json({faceset_id:'alice',cells:Array.from({length:9},(_,i)=>({cell:`cell_${i}`,support:filled && i===0 ? 1 : 0,best:filled && i===0 ? {sharpness:.78,quality:.81} : null,views:[]}))});
  if(url.endsWith('/fill_missing_angles')) {calls.push(JSON.parse(init!.body as string));filled=true;return json({added:[{}],sampled_frames:30,stop_reason:'end_of_video'});}
  throw new Error(url);
 });
 vi.stubGlobal('fetch',fetch);
 const host=document.createElement('div');document.body.append(host);const root=createRoot(host);const assign=vi.fn();
 try {
  await act(async()=>root.render(<FaceManager target="target.mp4" inspection={{inspection_id:'a'.repeat(32),tracks:[]}} sources={[{faceset_id:'alice',name:'Alice'}]} onAssignSource={assign}/>));
  expect(assign.mock.calls).toEqual([['track_0000','alice'],['track_0002','alice']]);
  expect(host.querySelectorAll('[role="gridcell"]')).toHaveLength(9);
  const fill=[...host.querySelectorAll('button')].find(button=>button.textContent?.includes('Fill Missing Angles'))!;
  expect(fill.disabled).toBe(false);
  await act(async()=>fill.click());
  expect(calls).toEqual([{target_media_path:'target.mp4',faceset_id:'alice'}]);
  expect(host.textContent).toContain('0.78 / Quality: 0.81');
  expect(fetch.mock.calls.filter(([url])=>url.endsWith('/pose_matrix'))).toHaveLength(2);
 } finally {await act(async()=>root.unmount());host.remove();vi.unstubAllGlobals();}
});
it('sends CodeFormer and the identity pair when restoration is animated from zero',()=>{
 const recipe={method:'slerp' as const,source_face_ids:['alice','bob'],weights:[1,0]};
 const payload=buildLivePayload({request:{target_path:'x',inspection_id:'a'.repeat(32),face_mappings:[{track_id:'track_0000',faceset_id:'alice'}],processing:'local',remote_consent:false,trim_start_seconds:null,trim_end_seconds:null,identity_blend:recipe,parameter_curves:{tracks:{restorer_opacity:[{frame:0,value:0,easing:'linear'},{frame:10,value:1,easing:'linear'}]}}},
 time:1,fps:25,parameters:{...DEFAULT_PARAMETERS,restorer_model:'codeformer',fidelity_weight:.8},restorerAvailable:true});
 expect(payload).toMatchObject({restorer:'codeformer',fidelity_weight:.8,identity_blend:recipe,frame_index:25});
});
