// Keep discovery ranks and original biometric measurements when manually merging tracks.
export function rankClusters(faces, tolerance) {
  const groups=[];
  const cosine=(a,b)=> {
    if (!a?.length || a.length !== b?.length) return -1;
    const norm=Math.hypot(...a)*Math.hypot(...b);
    return norm ? a.reduce((sum,v,i)=>sum+v*b[i],0)/norm : -1;
  };
  for (const face of faces) {
    const ranked=groups.map(group=>({group,score:cosine(face.embedding,group.faces[0].embedding)}))
      .sort((a,b)=>b.score-a.score || a.group.rank-b.group.rank);
    const manual=groups.find(group=>face.manualGroup && group.id===face.manualGroup);
    const group=manual || (!face.manualGroup && ranked[0]?.score >= 1-tolerance ? ranked[0].group : null);
    if (group) group.faces.push(face);
    else groups.push({id:face.manualGroup || `bank_${face.id}`,rank:groups.length,name:`Person ${String.fromCharCode(65+groups.length)}`,faces:[face]});
  }
  return groups;
}
export const poseBin=(value,edge)=>value < -edge ? -1 : value > edge ? 1 : 0;
export function poseGrid(faces) {
  return [-1,0,1].flatMap(pitch=>[-1,0,1].map(yaw=>({yaw,pitch,faces:faces.filter(face=>poseBin(face.yaw,15)===yaw && poseBin(-face.pitch,10)===pitch)
    .sort((a,b)=>(b.sharpness ?? -1)-(a.sharpness ?? -1))})));
}
