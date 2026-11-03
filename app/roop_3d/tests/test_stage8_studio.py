"""Stage 8 studio integration. Neural sessions are replaced by explicit numerical fixtures."""
import json
from dataclasses import replace
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from roop_3d.core.storage import atomic_json, owned
from roop_3d.core.pose import Pose3D
from roop_3d.tests.test_remediation_stage5 import extraction_app, extract, detection, FakeDetector, angle_vector, bare_swapper, make_document, make_view
from roop_3d.tests.test_workstation_preview import preview_fixture, FixtureSession, arrays
from roop_3d.tests.test_remediation_stage4 import live_payload
from roop_3d.server import workstation_preview as preview_module
from roop_3d.server.face_bank import FaceBank
from roop_3d.engine.volumetric.session import Replacement
from roop_3d.faceset.identity_blend import blend_documents, IdentityBlendSpec
BASE='http://127.0.0.1:8030'


def test_bank_discovers_reentry_persists_edits_and_keeps_rank_gaps(tmp_path,monkeypatch):
    app,service,request=extraction_app(tmp_path,monkeypatch)
    box,points=detection(28,Pose3D(yaw=0,pitch=0,roll=0))
    service.detector=FakeDetector([(box,points)])
    class Recognizer:
        def embed(self,image,points): return angle_vector(0,1 if image[0,0,0]>100 else 0)
    service.recognizer=Recognizer()
    monkeypatch.setattr(service.state.media,'frame',lambda path,timestamp,frame:np.full((64,96,3),255 if frame==1 or timestamp==800 else 0,np.uint8))
    path=owned(tmp_path,f"roop_3d/.runtime/inspections/{request['inspection_id']}.json")
    inspection=json.loads(path.read_text())
    track=inspection['tracks'][0]
    inspection['tracks']=[{**track,'track_id':f'track_{i:04d}','persistent_id':i,'first_frame':i,'last_frame':i,
                          'bounding_box':dict(zip(('x1','y1','x2','y2'),map(float,box)))} for i in range(3)]
    atomic_json(tmp_path,path.relative_to(tmp_path).as_posix(),inspection)
    body={'target_media_path':request['target_path'],'inspection_id':request['inspection_id']}
    with TestClient(app,base_url=BASE) as client:
        found=client.post('/api/v1/faces/bank/discover',json=body)
        assert found.status_code==200,found.text
        bank=found.json(); assert len(bank['clusters'])==2
        assert bank['clusters'][0]['track_ids']==['track_0000','track_0002']
        assert [c['rank'] for c in bank['clusters']]==[0,1]
        assert 'centroid' not in bank['clusters'][0]
        def edit(action,ids,**extra):
            nonlocal bank
            response=client.post('/api/v1/faces/bank/edit',json={'target_media_path':request['target_path'],'revision':bank['revision'],
                                  'action':action,'cluster_ids':ids,**extra})
            assert response.status_code==200,response.text
            bank=response.json()
        a,b=[c['id'] for c in bank['clusters']]
        edit('rename',[a],name='Returning actor')
        edit('assign',[a],faceset_id='alice')
        edit('merge',[b,a])
        assert bank['clusters'][0]['rank']==0 and bank['next_rank']==2
        assert bank['aliases'][b]==a and bank['clusters'][0]['faceset_id']=='alice'
        assert bank['clusters'][0]['track_ids']==['track_0000','track_0002','track_0001']
        restored=FaceBank(service).read(request['target_path'])
        assert restored==bank
        # New inspection labels must be re-associated by identity, not reused blindly.
        new_id='b'*32
        renamed={**inspection,'inspection_id':new_id,'tracks':[dict(track) for track in inspection['tracks']]}
        renamed['tracks'][0].update(first_frame=20000,last_frame=20000,first_timestamp_seconds=800,last_timestamp_seconds=800)
        renamed['tracks'][1].update(first_frame=0,last_frame=0)
        atomic_json(tmp_path,f'roop_3d/.runtime/inspections/{new_id}.json',renamed)
        rescanned=client.post('/api/v1/faces/bank/discover',json={**body,'inspection_id':new_id}).json()
        assert len(rescanned['clusters'])==1 and rescanned['clusters'][0]['id']==a
        assert rescanned['clusters'][0]['name']=='Returning actor' and rescanned['clusters'][0]['faceset_id']=='alice'
        bank=rescanned
        again=client.post('/api/v1/faces/bank/discover',json={**body,'inspection_id':new_id}).json()
        assert again['clusters']==bank['clusters']
        # Omitting inspection_id auto-looks up the latest inspection by digest
        found_auto = client.post('/api/v1/faces/bank/discover', json={'target_media_path': request['target_path']})
        assert found_auto.status_code == 200, found_auto.text
        assert found_auto.json()['clusters'] == bank['clusters']
        conflict=client.post('/api/v1/faces/bank/edit',json={'target_media_path':request['target_path'],'revision':0,'action':'rename','cluster_ids':[a],'name':'stale'})
        assert conflict.status_code==409


def test_fill_scans_video_then_matrix_reads_saved_archive(tmp_path,monkeypatch):
    app,service,request=extraction_app(tmp_path,monkeypatch)
    class RotatingDetector:
        def detect(self,image,threshold):
            pose=Pose3D(yaw=-30,pitch=20,roll=0) if image.mean()>50 else Pose3D(yaw=30,pitch=-20,roll=0)
            return [detection(28,pose)],{}
    with TestClient(app,base_url=BASE) as client:
        initial=extract(client,face_index=1)
        assert initial.status_code==201,initial.text
        faceset_id=initial.json()['faceset']['faceset_id']
        service.detector=RotatingDetector()
        body={'target_media_path':request['target_path'],'faceset_id':faceset_id}
        filled=client.post('/api/v1/faces/fill_missing_angles',json=body)
        assert filled.status_code==200,filled.text
        result=filled.json()
        assert result['scan_complete'] and result['sampled_frames']>3 and len(result['added'])==2
        matrix=client.get(f'/api/v1/facesets/{faceset_id}/pose_matrix').json()
        assert len(matrix['cells'])==9 and sum(c['support'] for c in matrix['cells'])==3
        assert {c['cell'] for c in matrix['cells'] if c['support']}=={'center_level','right_down','left_up'}
        assert all(c['thumbnailUrl'].startswith('data:image/jpeg') and c['best']['sharpness']>=0 for c in matrix['cells'] if c['support'])
        second=client.post('/api/v1/faces/fill_missing_angles',json=body).json()
        assert second['added']==[]
        assert client.get(f'/api/v1/facesets/{faceset_id}/pose_matrix').json()['item_count']==3


def test_codeformer_uses_training_alignment_and_actual_fidelity_feed(monkeypatch):
    from roop_3d.compositor.stream_restoration import OnnxFaceRestorer, CODEFORMER_REFERENCE
    from roop_3d.compositor.restoration import CodeFormerRestorer
    from roop_3d.engine.runtime import RuntimeConfig
    calls=[]
    class Model:
        record={'graph':{'inputs':[{'name':'x','shape':[1,3,512,512],'dtype':'FLOAT'}, {'name':'w','shape':[],'dtype':'DOUBLE'}]}}
        def run(self,feeds):
            calls.append(feeds)
            return {'y':np.full((1,3,512,512),float(feeds['w']),np.float32)}
        def release(self): pass
    monkeypatch.setattr('roop_3d.engine.runtime.OrtModel',lambda *args,**kwargs:Model())
    adapter=CodeFormerRestorer(None,RuntimeConfig())
    image=np.full((512,512,3),80,np.uint8)
    adapter.fidelity_weight=.25
    a=adapter.restore_aligned(image,CODEFORMER_REFERENCE)
    adapter.fidelity_weight=.75
    b=adapter.restore_aligned(image,CODEFORMER_REFERENCE)
    assert calls[0]['x'].shape==(1,3,512,512)
    assert float(calls[0]['w'])==.25 and calls[0]['w'].dtype==np.float64
    assert float(calls[1]['w'])==.75 and not np.array_equal(a,b)
    assert adapter.capability()['inference']=='VERIFIED_THIS_SESSION'
    adapter.release(); assert adapter.runner.model is None


def test_preview_pair_and_fidelity_curves_reach_the_same_frame(tmp_path,monkeypatch):
    app,fixture=preview_fixture(tmp_path,monkeypatch)
    bob=owned(tmp_path,'roop_3d/outputs/facesets/bob.fsz');bob.write_bytes(b'fixture')
    atomic_json(tmp_path,'roop_3d/.runtime/facesets/bob.json',{'faceset_id':'bob','name':'Bob','item_count':1,'schema_version':1})
    seen=[]
    class Session(FixtureSession):
        def prepare_session(self,source,**kwargs): self.source=source
        def infer_frame(self,image,**kwargs):
            seen.append((kwargs,self.source.metadata))
            result=arrays()[1]
            points=np.array([[8,10],[24,10],[16,16],[10,24],[22,24]],np.float32)
            return SimpleNamespace(**{**vars(result),'replacements':tuple(replace(item,landmarks5=points) for item in result.replacements)})
    weights=[]
    class Restorer:
        fidelity_weight=.5
        def __init__(self,*args): pass
        def restore_aligned(self,image,points): weights.append(self.fidelity_weight);return image.copy()
        def release(self): pass
    monkeypatch.setattr(preview_module,'ReferencePipeline',Session)
    monkeypatch.setattr('roop_3d.compositor.restoration.CodeFormerRestorer',Restorer)
    payload=live_payload(fixture,restorer='codeformer',restoration_blend=.7,fidelity_weight=.2,
        identity_blend={'source_face_ids':['alice','bob'],'weights':[1,0],'method':'slerp'},
        parameter_curves={'tracks':{'fidelity_weight':[{'frame':0,'value':.1},{'frame':4,'value':.9}],
                                   'identity_blend_t':[{'frame':0,'value':0},{'frame':4,'value':1}]}})
    with TestClient(app,base_url=BASE) as client:
        response=client.post('/api/preview',json=payload)
        assert response.status_code==200,response.text
        metadata=json.loads(response.headers['x-preview-meta'])
        assert metadata['unsupported']==[] and 'fidelity_weight' in metadata['applied']
        assert weights[-1]==pytest.approx(.5)
        assert seen[-1][0]['identity_blend_t']==pytest.approx(.5)
        assert len(seen[-1][1]['identity_curve_pair'])==2
        saved=client.post('/api/v1/projects',json={'name':'Animated pair','request':{**fixture['request'],
            'identity_blend':payload['identity_blend'],'parameter_curves':payload['parameter_curves']}})
        assert saved.status_code==201,saved.text
        assert saved.json()['request']['identity_blend']['source_face_ids']==['alice','bob']


@pytest.mark.parametrize('t,expected',[(0,(1,0)),(.25,(np.cos(np.pi/8),np.sin(np.pi/8))),(.5,(2**-.5,2**-.5)),(1,(0,1))])
def test_slerp_inference_preserves_unit_rank_and_endpoints(t,expected):
    a,b=angle_vector(0),angle_vector(90)
    doc_a=make_document('a',[make_view(0,0,0,a)])
    doc_b=make_document('b',[make_view(0,0,0,b)])
    doc=blend_documents([doc_a,doc_b],IdentityBlendSpec(source_face_ids=('a','b'),weights=(1,0)))
    swapper=bare_swapper(doc)
    class Model:
        feeds=None
        def run(self,feeds): self.feeds=feeds;return {'output':np.full((1,3,128,128),.5,np.float32)}
    swapper.model=Model();swapper.projection=np.eye(512,dtype=np.float32)
    points=np.array([[35,45],[85,45],[60,65],[40,90],[80,90]],np.float32)
    swapper.infer_frame(np.zeros((128,128,3),np.uint8),points,identity_blend_t=t)
    embedding=swapper.model.feeds['source'][0]
    assert embedding[:2]==pytest.approx(expected,abs=1e-6)
    assert np.linalg.norm(embedding)==pytest.approx(1,abs=1e-6)


def test_image_headers_are_bounded_before_decode_and_build_never_invents_identity(tmp_path,monkeypatch):
    import base64
    from roop_3d.core.image_bytes import decode_face_image
    from roop_3d.core.errors import DomainError
    huge=b'\x89PNG\r\n\x1a\n'+(13).to_bytes(4,'big')+b'IHDR'+(100_000).to_bytes(4,'big')*2+b'\x08\x02\x00\x00\x00'+b'\x00'*4
    with pytest.raises(DomainError): decode_face_image(huge)
    image=np.zeros((48,48,3),np.uint8)
    for suffix in ('.png','.jpg'):
        data=cv2.imencode(suffix,image)[1].tobytes()
        assert decode_face_image(data).shape==image.shape
    app,service,_=extraction_app(tmp_path,monkeypatch)
    service.detector=FakeDetector([])
    with TestClient(app,base_url=BASE) as client:
        invalid=client.post('/api/v1/facesets/build',json={'name':'Invalid','views':[{'image_base64':base64.b64encode(b'invalid').decode()}]})
        assert invalid.status_code==422
        no_face=client.post('/api/v1/facesets/build',json={'name':'No identity','views':[{'image_base64':base64.b64encode(cv2.imencode('.png',image)[1]).decode()}]})
        assert no_face.status_code==422
        assert no_face.json()['error']['code']=='FACE_COUNT_INVALID'


def test_render_evaluates_both_curves_and_fingerprints_both_sources(tmp_path,monkeypatch):
    from roop_3d.tests.test_stage12_worker import _fixture, _Reader
    from roop_3d.engine.pipeline_runner import MasterPipelineRunner
    from roop_3d.jobs.schema import JobRequest
    from roop_3d.core.errors import DomainError
    app,worker,payload=_fixture(tmp_path,count=4)
    owned(tmp_path,'roop_3d/outputs/facesets/bob.fsz').write_bytes(b'fixture-b')
    atomic_json(tmp_path,'roop_3d/.runtime/facesets/bob.json',{'faceset_id':'bob','name':'Bob','item_count':1,'schema_version':1})
    class Reader(_Reader):
        b_hash='b'*64
        def read(self,path,*,faceset_id):
            doc=super().read(path,faceset_id=faceset_id)
            if faceset_id=='bob':
                vector=tuple(angle_vector(90))
                doc=doc.model_copy(update={'canonical_embedding':vector,'medoid_embedding':vector,'source_sha256':self.b_hash,'views':tuple(view.model_copy(update={'embedding':vector}) for view in doc.views)})
            return doc
    seen=[];weights=[]
    class Session(FixtureSession):
        def prepare_session(self,source,**kwargs): self.source=source
        def infer_frame(self,image,**kwargs):
            seen.append(kwargs['identity_blend_t'])
            result=arrays()[1]
            points=np.array([[8,10],[24,10],[16,16],[10,24],[22,24]],np.float32)
            return SimpleNamespace(**{**vars(result),'replacements':tuple(replace(item,landmarks5=points) for item in result.replacements)})
        def composite_frame(self,image,result,*,options,**kwargs):
            return self.compositor.compose(image,result.replacements,{0:result.observations[0].observation.landmarks106},options=options)
        def capability(self): return {}
    class Restorer:
        fidelity_weight=.5
        def __init__(self,*args): pass
        def restore_aligned(self,image,points): weights.append(self.fidelity_weight);return image.copy()
        def release(self): pass
    monkeypatch.setattr('roop_3d.compositor.restoration.CodeFormerRestorer',Restorer)
    reader=Reader()
    runner=MasterPipelineRunner(tmp_path,app.state.models,app.state.media,lambda:app.state.settings,
             faceset_reader=reader,pipeline_factory=lambda catalog,config:Session(catalog,config))
    request=JobRequest.model_validate({**payload,'identity_blend':{'source_face_ids':['alice','bob'],'weights':[1,0]},
         'workstation_parameters':{'restorer_model':'codeformer','restoration_blend':.5},
         'parameter_curves':{'tracks':{'identity_blend_t':[{'frame':0,'value':0},{'frame':3,'value':1}],
                                      'fidelity_weight':[{'frame':0,'value':.2},{'frame':3,'value':.8}]}}})
    record,_=app.state.jobs.create(request,'animated-render')
    prepared=runner.prepare(record)
    assert prepared.checkpoint.faceset_sha256['bob']=='b'*64
    output=runner.run_prepared(record,SimpleNamespace(profile='cpu',provider='CPUExecutionProvider'),prepared=prepared,
                              control=lambda:None,mark_running=lambda:None,report_progress=lambda progress:None)
    assert output.frames==4
    assert seen==pytest.approx([0,1/3,2/3,1])
    assert weights==pytest.approx([.2,.4,.6,.8])
    reader.b_hash='c'*64
    with pytest.raises(DomainError,match='source FaceSet changed'): runner.prepare(record)
