"""Inspect the staged student in real MuJoCo without activating its configuration."""
import json,sys
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;sys.path.insert(0,str(BASE))
import bootstrap
cfg=bootstrap.load_config();candidate=json.loads((HERE/'candidate_policy.json').read_text().replace('/home/ubuntu/Go2W/',bootstrap.SNAPSHOT.as_posix()+'/'))
cfg['policies']['handstand'].update(candidate)
bootstrap.load_config=lambda:json.loads(json.dumps(cfg))
from run import create_runtime,advance
env,c=create_runtime();samples=[]
def tick(keys=()):
    c.feed('student_diagnostic',list(keys));advance(env,c)
with torch.inference_mode():
    for _ in range(100):tick()
    tick(['H'])
    for k in range(1400):
        tick()
        if k%25==0 or c.mode=='error':
            d=env.robot.data
            row=dict(frame=k,mode=c.mode,gravity=d.projected_gravity_b.torch[0].tolist(),
                height=float(d.root_pos_w.torch[0,2]),message=c.message)
            samples.append(row);print(json.dumps(row),flush=True)
        if c.mode=='error':break
(HERE/'student_diagnostic.json').write_text(json.dumps(samples,indent=2),encoding='utf-8')
