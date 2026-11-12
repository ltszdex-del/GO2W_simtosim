from pathlib import Path
import sys,math,json,torch
h=Path(__file__).resolve().parent;kind=sys.argv[1]
sys.path.insert(0,str(h/'runtime'));sys.path.insert(0,str(h/kind))
from config import COMMAND,CONTROL
from schedule import Commands
sys.path.insert(0,str(h/'runtime'))
from commands import CommandIntegrator,profile_from_contract
profile=profile_from_contract(COMMAND,CONTROL);profile.update(wheel_reference_speed_m_s=0.,wheel_envelope_lever=0.)
runtime=CommandIntegrator(profile);training=Commands(1,'cpu')
torch.manual_seed(17);max_error=0.
for tick in range(2200):
    if tick%70==0:axes=torch.randint(-1,2,(1,3)).float()
    keys={k for k,a in zip(('W','A','Q'),axes[0]) if a>0}|{k for k,a in zip(('S','D','E'),axes[0]) if a<0}
    result=runtime.step(keys,.02);expected=training.tick(axes,.02)[0]
    error=float((torch.tensor(result)-expected).abs().max());max_error=max(max_error,error)
    assert error<2e-5,(tick,result,expected,error)
# High->low while moving: no acceleration reset or instantaneous cap clipping.
runtime.stop()
for _ in range(400):runtime.step({'W','Q'},.02)
old=runtime.velocity[:];runtime.speed_scale=.5
for tick in range(500):
    actual=runtime.step({'W','Q'},.02)
    for i,brake in enumerate((profile['forward_brake_m_s2'],profile['lateral_brake_m_s2'],profile['yaw_brake_rad_s2'])):
        assert abs(actual[i]-old[i])<=brake*.02+1e-6,(tick,i,old,actual)
    old=actual
assert math.isclose(actual[0],COMMAND['speed_limits'][0]/2,abs_tol=1e-8)
assert math.isclose(actual[2],COMMAND['yaw_cruise_limit']/2,abs_tol=1e-8)
print(json.dumps(dict(kind=kind,passed=True,scalar_tensor_command_max_error=max_error,smooth_k_limits=True)))
