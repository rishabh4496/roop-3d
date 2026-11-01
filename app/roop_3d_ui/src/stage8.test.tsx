import {it,expect} from 'vitest';
import {rankClusters,poseGrid} from './components/facemgr/faceBank';
import {evaluateCurve} from './stores/parameterCurves';
it('preserves discovery rank, real embeddings, merges and cosine ordering',()=>{
 const faces=[{id:'a',embedding:[1,0],yaw:0,pitch:0},{id:'b',embedding:[0,1],yaw:20,pitch:0},{id:'c',embedding:[.99,.01],yaw:0,pitch:0}];
 const groups=rankClusters(faces,.2);
 expect(groups.map(g=>g.faces.map(f=>f.id))).toEqual([['a','c'],['b']]);
 expect(rankClusters([...faces,{id:'d',embedding:[0,1]}],.2)[0].id).toBe(groups[0].id);
 expect(rankClusters(faces.map(f=>({...f,manualGroup:groups[0].id})),.2)).toHaveLength(1);
 expect(faces[1].embedding).toEqual([0,1]);
 expect(poseGrid(faces)).toHaveLength(9);
});
it('uses the same cubic Bezier as the backend',()=>{
 expect(evaluateCurve([{frame:0,value:0,easing:'bezier',control1:.2,control2:.8},{frame:100,value:1}],25,0)).toBeCloseTo(.2125,12);
});
