"""Durable target-person bank. Discovery ranks survive updates and explicit merges."""
from __future__ import annotations
import base64
import json
from typing import Literal
import cv2
import numpy as np
from fastapi import APIRouter, Request, Query
from pydantic import Field
from roop_3d.core.settings import Contract
from roop_3d.core.errors import DomainError
from roop_3d.core.storage import atomic_json, exclusive, owned
from roop_3d.core.pose import estimate_pose_5pt
from roop_3d.media.schema import FaceInspection
from roop_3d.media.detector import iou
from roop_3d.engine.volumetric.preprocess import aligned_crop
from roop_3d.server.face_extract import _unit

router = APIRouter(prefix='/api/v1/faces/bank')

class DiscoverBank(Contract):
    target_media_path: str = Field(min_length=1,max_length=1024)
    inspection_id: str | None = Field(default=None,pattern=r'^[0-9a-f]{32}$')

class EditBank(Contract):
    target_media_path: str = Field(min_length=1,max_length=1024)
    revision: int = Field(ge=0,strict=True)
    action: Literal['rename','merge','assign']
    cluster_ids: tuple[str,...] = Field(min_length=1,max_length=64)
    name: str | None = Field(default=None,min_length=1,max_length=128)
    faceset_id: str | None = Field(default=None,pattern=r'^[A-Za-z0-9_-]{1,64}$')

class FaceBank:
    def __init__(self, service):
        self.service, self.root = service, service.state.root

    def location(self, media):
        path,_ = self.service._media_file(media)
        digest=self.service._target_digest(path)
        return f'roop_3d/.runtime/face_banks/{digest}.json',digest

    def load(self, relative, digest):
        path=owned(self.root,relative)
        if path.is_file():
            result=json.loads(path.read_text('utf-8'))
            if result['target_sha256']!=digest:
                raise DomainError('BANK_TARGET_CHANGED','Face bank target does not match',409)
            return result
        return {'version':1,'revision':0,'target_sha256':digest,'next_rank':0,'clusters':[], 'aliases':{}, 'unresolved_tracks':[], 'inspection_id':None}

    @staticmethod
    def public(store):
        return {**store,'clusters':[{key:value for key,value in cluster.items() if key not in ('centroid','anchors')}
                   for cluster in sorted(store['clusters'],key=lambda cluster:cluster['rank'])]}

    def read(self, media):
        relative,digest=self.location(media)
        with exclusive(self.root,relative+'.lock'):
            return self.public(self.load(relative,digest))

    def discover(self, value):
        relative,digest=self.location(value.target_media_path)
        resolved_inspection_id = value.inspection_id
        path = owned(self.root,f'roop_3d/.runtime/inspections/{resolved_inspection_id}.json') if resolved_inspection_id else None
        if path is None or not path.is_file():
            inspections_dir = owned(self.root, 'roop_3d/.runtime/inspections')
            found_path = None
            if inspections_dir.is_dir():
                for p in sorted(inspections_dir.glob('*.json'), key=lambda x: x.stat().st_mtime, reverse=True):
                    try:
                        data = json.loads(p.read_text('utf-8'))
                        if data.get('target_sha256') == digest:
                            found_path = p
                            resolved_inspection_id = p.stem
                            break
                    except (OSError, ValueError):
                        continue
            if found_path is None:
                raise DomainError('INSPECTION_NOT_FOUND','Inspect the target first',422)
            path = found_path
        inspection=FaceInspection.model_validate_json(path.read_text('utf-8'))
        if inspection.target_sha256!=digest: raise DomainError('INSPECTION_TARGET_MISMATCH','Inspection belongs to different media',409)
        with exclusive(self.root,relative+'.lock'):
            store=self.load(relative,digest)
            if store.get('inspection_id') != resolved_inspection_id:
                for cluster in store['clusters']:
                    cluster['track_ids']=[]
                    cluster['faces']=[]
                store['inspection_id']=resolved_inspection_id
            known={track for cluster in store['clusters'] for track in cluster['track_ids']}
            unresolved=[]
            for track in inspection.tracks[:64]:
                if track.track_id in known: continue
                observations=[]
                box=track.bounding_box
                reference=np.array([box.x1,box.y1,box.x2,box.y2])
                sample_frames={track.first_frame,track.last_frame}
                middle=(track.first_frame+track.last_frame)//2
                if middle<10000: sample_frames.add(middle)
                for frame in sorted(sample_frames):
                    if frame<10000:
                        image=self.service.state.media.frame(value.target_media_path,None,frame)
                    else:
                        # Seek exact observed endpoint timestamps rather than hit the bounded frame-index replay API.
                        seconds=track.first_timestamp_seconds if frame==track.first_frame else track.last_timestamp_seconds
                        image=self.service.state.media.frame(value.target_media_path,seconds,None)
                    detections,_=self.service._detector().detect(image,.5)
                    candidates=sorted(detections,key=lambda face:iou(np.asarray(face[0]),reference),reverse=True)
                    if not candidates or iou(np.asarray(candidates[0][0]),reference)<.1: continue
                    _,points=candidates[0]
                    embedding=_unit(self.service._recognizer().embed(image,np.asarray(points,np.float32)))
                    crop=aligned_crop(image,points,512).image
                    sharpness=float(min(1,cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()/500))
                    ok,thumbnail=cv2.imencode('.jpg',cv2.resize(crop,(128,128)))
                    if not ok: continue
                    yaw,pitch,roll=estimate_pose_5pt(points).legacy_tuple()
                    observations.append((embedding,{'id':f'{track.track_id}_{frame}','trackId':track.track_id,'frame':frame,
                        'yaw':yaw,'pitch':pitch,'roll':roll,'sharpness':sharpness,'quality':sharpness,
                        'thumbnailUrl':'data:image/jpeg;base64,'+base64.b64encode(thumbnail).decode()}))
                if not observations:
                    unresolved.append(track.track_id)
                    continue
                anchor=observations[0][0]
                observations=[item for item in observations if float(item[0]@anchor)>=.6]
                centroid=_unit(np.mean([item[0] for item in observations],axis=0))
                ranked=sorted([(max(float(centroid@_unit(np.asarray(anchor))) for anchor in [cluster['centroid'], *cluster.get('anchors',[])]),cluster)
                               for cluster in store['clusters']],key=lambda pair:(-pair[0],pair[1]['rank']))
                if ranked and ranked[0][0]>=.6:
                    cluster=ranked[0][1]
                    count=cluster['count']
                    cluster['centroid']=_unit(np.asarray(cluster['centroid'])*count+centroid).tolist()
                    cluster['count']=count+1
                else:
                    if len(store['clusters'])>=64: raise DomainError('BANK_LIMIT','At most 64 people per target',409)
                    rank=store['next_rank'];store['next_rank']+=1
                    cluster={'id':f'person_{rank:04d}','rank':rank,'name':f'Person {rank+1}','centroid':centroid.tolist(),
                             'count':1,'track_ids':[],'faces':[],'faceset_id':None,'anchors':[centroid.tolist()]}
                    store['clusters'].append(cluster)
                cluster['track_ids'].append(track.track_id)
                cluster['faces'].extend(item[1] for item in observations)
                known.add(track.track_id)
            store['unresolved_tracks']=unresolved
            store['revision']+=1
            atomic_json(self.root,relative,store)
            return self.public(store)

    def edit(self,value):
        relative,digest=self.location(value.target_media_path)
        with exclusive(self.root,relative+'.lock'):
            store=self.load(relative,digest)
            if value.revision!=store['revision']: raise DomainError('BANK_CONFLICT','Reload the bank before editing',409)
            ids=set(value.cluster_ids)
            selected=sorted([cluster for cluster in store['clusters'] if cluster['id'] in ids],key=lambda c:c['rank'])
            if len(ids)!=len(selected): raise DomainError('BANK_PERSON_NOT_FOUND','Person no longer exists',404)
            if value.action=='rename':
                if len(selected)!=1 or not value.name or not value.name.strip(): raise DomainError('BANK_NAME_REQUIRED','Name one selected person',422)
                selected[0]['name']=value.name.strip()
            elif value.action=='assign':
                if value.faceset_id: self.service._resolve_faceset(value.faceset_id)
                for cluster in selected: cluster['faceset_id']=value.faceset_id
            else:
                if len(selected)<2: raise DomainError('BANK_MERGE_REQUIRED','Select at least two people',422)
                survivor=selected[0]
                assignments={c['faceset_id'] for c in selected if c['faceset_id']}
                if len(assignments)>1: raise DomainError('BANK_ASSIGNMENT_CONFLICT','Clear conflicting source assignments before merging',409)
                survivor['anchors']=[anchor for cluster in selected for anchor in cluster.get('anchors',[cluster['centroid']])][:256]
                survivor['centroid']=_unit(sum(np.asarray(c['centroid'])*c['count'] for c in selected)).tolist()
                survivor['count']=sum(c['count'] for c in selected)
                survivor['track_ids']=list(dict.fromkeys(track for c in selected for track in c['track_ids']))
                survivor['faces']=[face for c in selected for face in c['faces']]
                survivor['faceset_id']=next(iter(assignments),None)
                for removed in selected[1:]: store['aliases'][removed['id']]=survivor['id']
                store['clusters']=[c for c in store['clusters'] if c['id'] not in ids or c is survivor]
            store['revision']+=1
            atomic_json(self.root,relative,store)
            return self.public(store)

@router.get('')
def read_bank(request:Request,target_media_path:str=Query(min_length=1,max_length=1024)):
    return FaceBank(request.app.state.face_extract).read(target_media_path)

@router.post('/discover')
def discover_bank(value:DiscoverBank,request:Request):
    with request.app.state.engine_gate.preview_section():
        return FaceBank(request.app.state.face_extract).discover(value)

@router.post('/edit')
def edit_bank(value:EditBank,request:Request):
    return FaceBank(request.app.state.face_extract).edit(value)
