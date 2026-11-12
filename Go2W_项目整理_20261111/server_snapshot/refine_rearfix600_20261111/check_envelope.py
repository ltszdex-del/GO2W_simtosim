from pathlib import Path
import json,math,random,sys,torch
kind=sys.argv[1];root=Path(sys.argv[2]);sys.path.insert(0,str(root))
from config import COMMAND
from schedule import Commands
from yaw_envelope import yaw_limit,yaw_limit_tensor,brake_to_limit
c=COMMAND;forward,lateral=c['speed_limits'];backward=c['min_forward_speed']
expected=4. if kind=='wheel' else 3.14
assert c['yaw_speed_limit']==expected
assert c['yaw_cruise_limit']==(2. if kind=='wheel' else 1.6)
limits=[]
for ratio in (0.,.1,.25,.5,.75,.9,1.):
    value=yaw_limit(ratio*forward,0.,forward,backward,lateral,expected,c['yaw_cruise_limit'])
    assert math.isclose(value,yaw_limit(ratio*backward,0.,forward,backward,lateral,expected,c['yaw_cruise_limit']),abs_tol=1e-10)
    limits.append(value)
assert all(a>=b for a,b in zip(limits,limits[1:])) and limits[0]==expected and limits[-1]==c['yaw_cruise_limit']
torch.manual_seed(77)
vel=torch.randn(2000,3)
tensor=yaw_limit_tensor(vel,forward,backward,lateral,expected,c['yaw_cruise_limit'])
scalar=torch.tensor([yaw_limit(*v[:2].tolist(),forward,backward,lateral,expected,c['yaw_cruise_limit']) for v in vel])
assert torch.allclose(tensor,scalar,atol=5e-7)
# Integrate mirrored commands through spin-up, translation, releases and reversals.
cmd=Commands(2,'cpu');maximum=0.
for tick in range(1600):
    axes=torch.tensor([[0.,0.,1.] if tick<160 else [1.,0.,1.] if tick<600 else [0.,0.,0.] if tick<850 else [-1.,.7,-1.] if tick<1300 else [0.,0.,0.]])
    axes=torch.cat((axes,axes*torch.tensor([1.,-1.,-1.])))
    old=cmd.velocity.clone();cmd.tick(axes,.02)
    assert torch.allclose(cmd.velocity[0]*torch.tensor([1.,-1.,-1.]),cmd.velocity[1],atol=1e-6)
    assert torch.isfinite(cmd.velocity).all()
    delta=(cmd.velocity[:,2]-old[:,2]).abs()
    assert (delta<=(c['yaw_acceleration']+c['yaw_brake'])*.02+.006).all()
    maximum=max(maximum,float(cmd.velocity[:,2].abs().max()))
    if tick==155:assert math.isclose(float(cmd.velocity[0,2]),expected,abs_tol=1e-5)
    if tick==599:assert math.isclose(float(cmd.velocity[0,2]),c['yaw_cruise_limit'],abs_tol=1e-5)
assert not cmd.velocity.any()
result=dict(kind=kind,passed=True,speed_ratios=[0.,.1,.25,.5,.75,.9,1.],yaw_caps=limits,observed_reference_max=maximum,
    yaw_acceleration=c['yaw_acceleration'],yaw_brake=c['yaw_brake'])
(root/'yaw_envelope_verified.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
