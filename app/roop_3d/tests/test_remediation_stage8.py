import numpy as np
import pytest
from roop_3d.temporal import OneEuroFilter
from roop_3d.temporal_stabilizer import PoseHysteresis
from roop_3d.core.animation import CurveKey, ParameterCurves, evaluate_curve
from roop_3d.core.workstation import WorkstationParameters

def test_noise_and_turn():
    rng=np.random.default_rng(8)
    samples=rng.normal(0,.5,(2400,5,2))
    f=OneEuroFilter()
    out=np.array([f.update(v,i/120) for i,v in enumerate(samples)])
    assert out[400:].std() < samples[400:].std()*.2
    turn=np.array([f.update(np.full((5,2),v),20+(i+1)/120) for i,v in enumerate(np.linspace(0,80,60))])
    assert abs(turn[-1].mean()-80)<5

@pytest.mark.parametrize('sign',[-1,1])
def test_pose_hysteresis(sign):
    gate=PoseHysteresis()
    assert gate.update(sign*14,0)==(0,0)
    for v in [14.9,15.1,16,14]*20:
        assert gate.update(sign*v,0)==(0,0)
    assert gate.update(sign*18,0)==(sign,0)
    assert gate.update(sign*14,0)==(sign,0)
    assert gate.update(sign*12,0)==(0,0)

@pytest.mark.parametrize('frame',[0,1,7,23,49,50,87,99,100,150])
def test_bezier_exact(frame):
    keys=(CurveKey(frame=0,value=0,easing='bezier',control1=.2,control2=.8),CurveKey(frame=100,value=1))
    t=min(1,frame/100)
    expected=3*(1-t)**2*t*.2+3*(1-t)*t*t*.8+t**3
    assert evaluate_curve(keys,frame)==pytest.approx(expected,abs=1e-12)
    curves=ParameterCurves(tracks={'tps_expression_weight':keys})
    assert curves.evaluate(frame,WorkstationParameters()).tps_expression_weight==pytest.approx(expected)

def test_step_and_validation():
    keys=(CurveKey(frame=4,value=.2,easing='step'),CurveKey(frame=9,value=.8))
    assert evaluate_curve(keys,8)==.2
    assert evaluate_curve(keys,9)==.8
    with pytest.raises(ValueError): OneEuroFilter(beta=-1)


@pytest.mark.parametrize('pitch',[-25,25])
def test_hysteresis_keeps_standard_pitch_matrix_orientation(pitch):
    from roop_3d.tests.test_remediation_stage5 import full_coverage_document, bare_swapper
    from roop_3d.core.pose import Pose3D
    document=full_coverage_document()
    swapper=bare_swapper(document)
    pose=Pose3D(yaw=0,pitch=pitch,roll=0)
    expected=swapper._matrices[None].lookup(pose).view_index
    assert swapper.reference_embedding(None,pose)[1]==expected
