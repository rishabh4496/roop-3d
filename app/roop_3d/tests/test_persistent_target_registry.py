import numpy as np
from roop_3d.identity.target_lock import ClusterRegistry, FaceRef

def test_strict_profile_gate_keeps_slot_after_reentry():
    registry = ClusterRegistry()
    a = np.array([1., 0., 0.], np.float32)
    profile = np.array([.43, np.sqrt(1-.43**2), 0.], np.float32)
    first = registry.associate([FaceRef(a,0,0,0,[0,0,100,100])],0)[0]
    again = registry.associate([FaceRef(profile,75,0,0,[500,0,600,100])],200)[0]
    assert again.cluster_id == first.cluster_id


def test_fifteen_pose_buckets_retain_best_quality():
    from roop_3d.identity.target_lock import BankEntry, PersonCluster
    cluster = PersonCluster(1)
    for pitch in (-30,0,30):
        for yaw in (-60,-30,0,30,60):
            cluster.consider(BankEntry(np.array([1.,0.]),yaw,pitch,0,1,[0,0,10,10],'auto',quality=10))
    assert len(cluster.bank) == 15
    cluster.consider(BankEntry(np.array([1.,0.]),0,0,0,2,[0,0,10,10],'auto',quality=20))
    assert cluster.reference_for_pose(0,0).frame_index == 2


def test_hungarian_assignment_avoids_greedy_fragmentation():
    r = ClusterRegistry()
    def f(e,x): return FaceRef(np.array(e,np.float32),0,0,0,[x,0,x+100,100])
    r.associate([f([1,0,0],0),f([0,1,0],300)],0)
    result = r.associate([f([.8,.6,0],150), f([.7,0,np.sqrt(.51)],150)],1)
    assert [a.cluster_id for a in result] == [2,1]
    assert len(r.clusters) == 2


def test_strict_gate_rejects_spatial_overlap_without_identity():
    r = ClusterRegistry()
    r.associate([FaceRef(np.array([1.,0.]),0,0,0,[0,0,10,10])],0)
    result = r.associate([FaceRef(np.array([.41,np.sqrt(1-.41**2)]),0,0,0,[0,0,10,10])],1)
    assert result[0].cluster_id == 2


def test_bins_include_outer_endpoints_and_do_not_claim_unseen_angles():
    from roop_3d.identity.target_lock import pose_bucket
    assert pose_bucket(-75,-45) == (0,0)
    assert pose_bucket(75,45) == (4,2)
    assert pose_bucket(-45,-15) == (1,1)
    assert pose_bucket(76,0) is None
    assert pose_bucket(0,float('nan')) is None


def test_kalman_handles_backward_scrubbing_and_long_absences():
    from roop_3d.identity.target_lock import KalmanBox
    k = KalmanBox([0,0,10,10],0)
    k.update([10,0,20,10],1)
    assert k.predict(2)[0] > 10
    before = k.x.copy()
    k.update([500,0,510,10],0)
    np.testing.assert_array_equal(k.x,before)
    k.update([500,0,510,10],200)
    np.testing.assert_array_equal(k.predict(201),[500,0,510,10])


def test_render_queries_locked_pose_bank_and_uses_persistent_ids():
    from types import SimpleNamespace
    from roop_3d.engine.volumetric.session import ReferencePipeline
    from roop_3d.engine.detection.face_tracker import FaceObservation, TrackedFace
    from roop_3d.identity.target_lock import TargetLockStore
    store = TargetLockStore()
    emb = np.array([1.,0.],np.float32)
    lock = store.create('v',0,FaceRef(emb,60,0,0,[0,0,100,100],landmarks3d=np.ones((68,3),np.float32)))
    session = object.__new__(ReferencePipeline)
    session.target_registry = store.registry('v')
    session.angle_frame_index = 10
    session.tracker = SimpleNamespace(frame_index=10)
    face = FaceObservation(np.array([200,0,300,100]),np.zeros((5,2)),np.zeros((106,2)),emb,(60,0,0))
    tracked = session._persistent_target_faces([TrackedFace(99,face,1,0)])[0]
    assert tracked.track_id == 0
    reference = session._target_reference(tracked)
    assert reference.yaw == 60
    moved = session._bank_landmarks(reference,face.box)
    assert moved[0,0] == 201


def test_perspective_pose_recovers_known_rotation_and_translation():
    import cv2
    from roop_3d.core.pose import _MODEL5, rotation_from_euler, estimate_pose_6dof
    reflection = np.diag([1.,1.,-1.])
    camera = np.array([[1000.,0,500.],[0,1000.,400.],[0,0,1.]])
    rvec = cv2.Rodrigues(reflection @ rotation_from_euler(60,20,-10) @ reflection)[0]
    points = cv2.projectPoints(_MODEL5 @ reflection,rvec,np.array([.1,.05,1.]),camera,None)[0].reshape(-1,2)
    result = estimate_pose_6dof(points,1000,800)
    assert result is not None
    np.testing.assert_allclose([result['yaw'],result['pitch'],result['roll']],[60,20,-10],atol=.1)
    np.testing.assert_allclose(result['translation'],[.1,.05,1.],atol=.01)


def test_background_harvest_scans_unsampled_frames_and_retains_crops():
    from types import SimpleNamespace
    from roop_3d.server.target_lock import scan_keyframes
    from roop_3d.identity.target_lock import TargetLockStore
    store = TargetLockStore()
    e = np.array([1.,0.],np.float32)
    lock = store.create('v',0,FaceRef(e,0,0,0,[0,0,100,100]))
    visited = []
    image = np.random.default_rng(1).integers(0,255,(100,100,3),dtype=np.uint8)
    def read(path,frame_index):
        visited.append(frame_index)
        return SimpleNamespace(frame_index=frame_index,image=image)
    def analyze(path,result,threshold):
        return [SimpleNamespace(embedding=e,box=np.array([0,0,100,100]),
            pose_legacy=(60 if result.frame_index==2 else 0,0,0),score=.9,
            validity={'occlusion_ratio':.25},kps68_3d=np.ones((68,3)))]
    frames = SimpleNamespace(info=lambda p: SimpleNamespace(total_frames=5),read=read,analyzed_faces=analyze)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(target_locks=store,media=SimpleNamespace(frames=frames))))
    scan_keyframes(request,lock,2,.5)
    assert set(visited) == set(range(5))
    assert lock.scan['exhaustive'] is True
    bank = store.registry('v').clusters[lock.cluster_id]
    assert bank.coverage()['captured'] == 2
    entry = bank.reference_for_pose(60,0)
    assert entry.frame_index == 2 and entry.crop.shape == (256,256,3)
    assert entry.landmarks3d.shape == (68,3) and entry.occlusion_ratio == .25
    assert lock.scan['message'] == '2/2 available video angles locked'


def test_bad_embeddings_cannot_poison_registry():
    r = ClusterRegistry()
    refs = [FaceRef(e,0,0,0,[0,0,10,10]) for e in (None,np.zeros(2),np.array([float('nan'),1]))]
    assert all(a.cluster_id is None for a in r.associate(refs,0))
    assert not r.clusters


def test_stage_one_assigns_before_stabilizing_and_survives_cut():
    from roop_3d.engine.detection.face_tracker import FaceTracker, FaceObservation
    from types import SimpleNamespace
    r = ClusterRegistry()
    tracker = object.__new__(FaceTracker)
    tracker.target_registry = r
    tracker.released = False
    tracker.max_tracks = 32
    tracker.threshold, tracker.low_threshold = 0.5, None
    tracker._tracks = {}
    tracker.frame_index = tracker.scene_id = 0
    tracker.stabilize = False
    tracker.policy = None
    def observation(e,x):
        return FaceObservation(np.array([x,0,x+100,100]),np.zeros((5,2)),np.zeros((106,2)),np.array(e),(0,0,0))
    first = tracker.assign([observation([1.,0.],0),observation([0.,1.],300)])
    tracker._tracks.clear()  # scene cut or disappearance must not clear identity slots
    second = tracker.assign([observation([0.,1.],0),observation([1.,0.],300)])
    assert [f.track_id for f in first] == [0,1]
    assert [f.track_id for f in second] == [1,0]
    assert tracker.active_ids == frozenset({0,1})


def test_clicked_face_uses_whole_frame_assignment_instead_of_rematching_alone():
    from roop_3d.identity.target_lock import TargetLockStore
    store = TargetLockStore()
    r = store.registry('v')
    def f(e,x): return FaceRef(np.array(e,np.float32),0,0,0,[x,0,x+100,100])
    r.associate([f([1,0,0],0),f([0,1,0],300)],0)
    clicked = f([.8,.6,0],150)
    pair = r.associate([clicked,f([.7,0,np.sqrt(.51)],150)],1)
    lock = store.create('v',1,clicked,cluster_id=pair[0].cluster_id)
    assert lock.cluster_id == 2
    assert lock.public(r)['face_id'] == 1


def test_rescanned_identical_embedding_score_never_exceeds_one():
    """float32 rounding of a re-seen embedding gave cosines like 1.0000002, which AngleDetection rejects."""
    from roop_3d.angle_tracker import AngleDetection
    from roop_3d.identity.target_lock import PersonCluster, _unit
    rng = np.random.default_rng(0)
    for _ in range(2000):
        raw = rng.standard_normal(512).astype(np.float32)
        raw /= np.linalg.norm(raw)
        cluster = PersonCluster(1)
        cluster.consider(FaceRef(raw, 0, 0, 0, [0, 0, 100, 100]).entry(0))
        if float(cluster.bank[0].embedding @ _unit(raw)) > 1.0:
            break
    else:
        raise AssertionError("no seed produced an unclamped dot above 1.0")
    score, _ = cluster.score(raw)
    assert score <= 1.0
    landmarks = np.array([[30, 30], [70, 30], [50, 50], [35, 70], [65, 70]], np.float32)
    AngleDetection(np.array([0, 0, 100, 100], np.float32), landmarks, (0., 0., 0.), raw, confidence=score)
