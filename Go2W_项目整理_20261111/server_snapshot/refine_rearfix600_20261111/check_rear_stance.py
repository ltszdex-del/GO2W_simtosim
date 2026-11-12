from pathlib import Path
import sys,torch,json
root=Path(sys.argv[1]);sys.path.insert(0,str(root))
from backward_stance import backward_stance
from config import GAIT
ref=torch.tensor(GAIT['stance_feet_reference'])
feet=ref[None].repeat(5,1,1)
feet[1,2:,1]*=.8;feet[2,2:,1]*=.5;feet[3,2:,1]*=-1.;feet[4,2:,1]*=1.3
command=torch.tensor([[-1.,0.,0.]]).repeat(5,1)
value=backward_stance(feet,ref,command,command[:,:2])
assert value['width_cost'][0]==0 and value['width_cost'][4]==0
assert 0<value['width_cost'][1]<value['width_cost'][2]<=value['width_cost'][3]
mirror=feet[:,[1,0,3,2]].clone();mirror[:,:,1]*=-1
other=backward_stance(mirror,ref,command,command[:,:2])
for k in ('width_cost','side_cost','gate'):assert torch.allclose(value[k],other[k],atol=1e-5),k
# A backward reference and residual backward actual speed both enable the guard.
zero=torch.zeros_like(command)
assert (backward_stance(feet,ref,zero,command[:,:2])['gate']==1).all()
assert not backward_stance(feet,ref,zero,zero[:,:2])['gate'].any()
turn=command.clone();turn[:,2]=4.
assert (backward_stance(feet,ref,turn,turn[:,:2])['gate']>=.25).all()
x=feet[1:2].clone().requires_grad_(True)
cost=backward_stance(x,ref,command[:1],command[:1,:2])['width_cost'].sum();cost.backward()
assert x.grad[0,2,1]>0 and x.grad[0,3,1]<0,'Gradient must push both rear wheels outward'
result=dict(passed=True,widths=value['rear_width_m'].tolist(),required_min=float(value['required_rear_width_m'][0]),costs=value['width_cost'].tolist(),mirror=True,gradient_outward=True)
(root/'rear_stance_verified.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
