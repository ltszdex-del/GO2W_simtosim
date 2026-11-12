"""Collect real MuJoCo teacher labels on teacher, mixed and student trajectories."""
import contextlib,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;sys.path.insert(0,str(BASE))
import bootstrap
from run import create_runtime,advance
cfg_path=BASE/'runtime/config.json';unchanged=cfg_path.read_bytes();cfg=bootstrap.load_config()
records=[];labels=[];buckets=[];ages=[];episodes=[];runids=[]
def run(case,alpha):
    bootstrap.load_config=lambda:json.loads(json.dumps(cfg))
    env,c=create_runtime();a=c.adapters['handstand']
    teacher=a.networks.Teacher().eval()
    teacher.load_state_dict(torch.load(a.teacher_path,map_location='cpu',weights_only=False)['model'])
    if case=='wheel':c._set_mobility('wheel',env.snapshot())
    def tick(keys=()):c.feed('targeted_collection',list(keys));advance(env,c)
    with torch.inference_mode():
        for _ in range(100):tick()
        if case!='idle':
            for _ in range(75):tick(['W' if case=='wheel' else case])
            for _ in range(100):tick()
        def observe(state):
            obs=a.ctx._get_observations();target=teacher(obs['policy']);student=a.model(obs['student'])
            age=float(a.ctx.episode_length_buf[0])*.02;p=a.ctx.physical()
            drift=float(a.ctx.anchor_error(p).norm(dim=-1).amax())
            disagreement=float((target[:,12:14]-student[:,12:14]).square().mean().sqrt())
            hard=age>2.0 and (drift>.12 or disagreement>.04)
            bucket=3 if hard and len(records)%2==0 else (0 if age<1.5 else (1 if age<3.0 else 2))
            records.append(obs['student'][0].numpy().astype(np.float16));labels.append(target[0].numpy().copy())
            buckets.append(bucket);ages.append(age);runids.append(len(episodes))
            a.collection_action=(1.-alpha)*target+alpha*student
            return obs['student']
        a.observe=observe;a.infer=lambda observation:a.collection_action
        tick(['H'])
        for k in range(800):
            tick()
            if c.mode=='error':break
            if hasattr(a,'ctx') and float(a.ctx.episode_length_buf[0])*.02>=12.:break
        row=dict(case=case,student_fraction=alpha,end_mode=c.mode,h_age=float(a.ctx.episode_length_buf[0])*.02,
            total_samples=len(records));episodes.append(row);print(json.dumps(row),flush=True)
with (HERE/'targeted_collection.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log):
    for case in ('idle','W','E','wheel'):
        for alpha in (0.,.5,1.):run(case,alpha)
assert cfg_path.read_bytes()==unchanged
path=HERE/'mujoco_targeted4179.npz'
np.savez_compressed(path,observations=np.stack(records),targets=np.stack(labels),
    bucket=np.array(buckets,dtype=np.int64),age=np.array(ages,dtype=np.float32),episode=np.array(runids,dtype=np.int64))
receipt=dict(samples=len(records),episodes=episodes,bucket_counts=np.bincount(buckets,minlength=4).tolist(),
    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),teacher_sha256=cfg['policies']['handstand']['teacher_sha256'],
    observation_dim=2657,label_dim=16,physics_motor_dt=.001,control_dt=.02,
    note='Labels are always teacher4179 on the actual visited state; no state edits or external forces; production config unchanged')
(HERE/'targeted_dataset.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt,indent=2),flush=True)
