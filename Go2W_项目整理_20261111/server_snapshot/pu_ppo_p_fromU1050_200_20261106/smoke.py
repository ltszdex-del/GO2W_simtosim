import argparse,json,math
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
import torch
from env import PUEnv,PUCfg,ROOT
from model import Policy
try:
 torch.set_num_threads(4);cfg=PUCfg();cfg.scene.num_envs=16;cfg.sim.device='cuda:0'
 e=PUEnv(cfg,'P');model=Policy().to(e.device).eval()
 o,_=e.reset();assert o['policy'].shape==(16,279)
 init=json.loads((ROOT/'standing_initialization.json').read_text());assert torch.allclose(e.robot.data.joint_pos.torch[:,e.legs],torch.tensor(init['q'],device=e.device)[None].expand(16,-1),atol=1e-6)
 assert torch.allclose(e.robot.data.root_pos_w.torch[:,2]-e.scene.env_origins[:,2],torch.full((16,),init['height']+.002,device=e.device),atol=1e-6)
 e.released[:]=True;e.release_time[:]=.5;e._pre_physics_step(torch.zeros(16,12,device=e.device));e._apply_action();assert (e.torque==0).all()
 e.pd_count=None;o,_=e.reset();last=e.target_velocity.clone();max_rate=0.;max_acc=0.;episodes=[];samples=[]
 with torch.inference_mode():
  for i in range(500):
   o,reward,term,trunc,info=e.step(model(o['policy']));assert e.pd_count==4
   assert torch.isfinite(o['policy']).all() and torch.isfinite(reward).all()
   max_rate=max(max_rate,float(e.target_velocity.abs().max()));mask=~(term|trunc)
   if mask.any():max_acc=max(max_acc,float(((e.target_velocity-last)[mask]/.02).abs().max()))
   last=e.target_velocity.clone()
   if 'episode' in info:episodes.append(info['episode'].copy())
   if i%100==99:
    m=e.metrics();row=dict(step=i+1,height_mean=float(m['z'].mean()),released=float(e.released.float().mean()),poweroff=float((e.gain==0).float().mean()),finished=float((e.completion_time>=0).float().mean()),episode=info.get('episode'));samples.append(row);print(json.dumps(row),flush=True)
 assert max_rate<=1.4001 and max_acc<=3.0001,(max_rate,max_acc)
 n=sum(x['count'] for x in episodes)
 summary={k:sum(x[k]*x['count'] for x in episodes)/n for k in episodes[0] if k!='count'} if n else {}
 result=dict(status='passed',steps=500,observation_dim=279,pd_substeps=4,all16_released_effort_zero=True,max_target_rate=max_rate,max_target_accel=max_acc,parent_policy_under_new_timing=summary,episodes=n,samples=samples)
 (ROOT/'smoke_result.json').write_text(json.dumps(result,indent=2));print('RESULT',json.dumps(result),flush=True);e.close()
except BaseException:
 import traceback;traceback.print_exc();app.close(exit_code=1)
else:app.close()
