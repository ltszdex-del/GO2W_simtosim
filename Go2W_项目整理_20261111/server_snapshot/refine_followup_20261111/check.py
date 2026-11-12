from pathlib import Path
import sys,json,torch,hashlib
kind=sys.argv[1];r=Path('/home/ubuntu/Go2W')/f'go2w_{kind}_followup_20261111';sys.path.insert(0,str(r))
from config import COMMAND,GAIT
from schedule import Commands
ck=torch.load(r/'resume/parent.pt',map_location='cpu',weights_only=False)
for name,digest in ck['manifest']['source_sha256'].items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==digest
axes=torch.tensor([[1.,0.,1.],[-1.,0.,-1.]])
c=Commands(2,'cpu');c.speed_cap=3.
for _ in range(300):c.tick(axes,.02)
if kind=='wheel':
    assert torch.allclose(c.velocity[:,0],torch.tensor([3.,-2.5]),atol=1e-5),c.velocity
    assert torch.allclose(c.velocity[:,2],torch.tensor([2.,-2.]),atol=1e-5),c.velocity
    assert COMMAND['yaw_acceleration']==2.4 and COMMAND['yaw_brake']==9.6
else:
    from period_math import sample_period
    periods=sample_period(1000,'cpu',True)
    assert float(periods.min())>=.23 and float(periods.max())<=.33
    assert torch.allclose(c.velocity[:,2],torch.tensor([1.6,-1.6]),atol=1e-5)
print(json.dumps(dict(kind=kind,status='passed',command=COMMAND,period=GAIT['period'],period_range=GAIT.get('period_range_s'))))
