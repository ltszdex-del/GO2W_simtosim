import argparse,json,math
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
import torch
from env import PUEnv,PUCfg,ROOT
from model import Policy
try:
 torch.set_num_threads(4);cfg=PUCfg();cfg.scene.num_envs=16;cfg.sim.device='cuda:0'
 e=PUEnv(cfg,'U');model=Policy().to(e.device).eval()
 c=torch.load('/home/ubuntu/Go2W/pu_ppo_u_fast3s_20261106/logs/U_fast3s_resume300/final.pt',map_location=e.device,weights_only=False);model.load_state_dict(c['model'])
 o,_=e.reset();assert o['policy'].shape==(16,216)
 e._pre_physics_step(model(o['policy']));e._apply_action();assert (e.torque==0).all(),'Initial poweroff step must have zero effort'
 e.pd_count=None;o,_=e.reset();last=e.target_velocity.clone();max_rate=0.;max_acc=0.;max_mirror=0.;samples=[];episodes=[]
 with torch.inference_mode():
  for i in range(1000):
   o,reward,term,trunc,info=e.step(model(o['policy']));assert e.pd_count==4
   assert torch.isfinite(o['policy']).all() and torch.isfinite(reward).all()
   max_rate=max(max_rate,float(e.target_velocity.abs().max()));mask=~(term|trunc)
   if mask.any():max_acc=max(max_acc,float(((e.target_velocity-last)[mask]/.02).abs().max()))
   max_mirror=max(max_mirror,float((e.target-e.mirror(e.target)).abs().max()));last=e.target_velocity.clone()
   if 'episode' in info:episodes.append(info['episode'].copy())
   if i in (49,99,149,199,299,499,999):
    m=e.metrics();row=dict(step=i+1,height_mean=float(m['z'].mean()),roll_abs_deg=float(m['roll'].abs().max()*180/math.pi),pitch_abs_deg=float(m['pitch'].abs().max()*180/math.pi),wheel_camber_max_deg=float(m['front_camber'].max()*180/math.pi),standing_time=e.standing_time.tolist(),episode=info.get('episode'));samples.append(row);print(json.dumps(row),flush=True)
 assert max_rate<=1.2001 and max_acc<=2.4001 and max_mirror<1e-6,(max_rate,max_acc,max_mirror)
 n=sum(x['count'] for x in episodes);summary={k:sum(x[k]*x['count'] for x in episodes)/n for k in episodes[0] if k!='count'} if n else {}
 result=dict(status='passed',steps=1000,observation_dim=216,pd_substeps=4,initial_zero_effort=True,max_target_rate=max_rate,max_target_accel=max_acc,max_target_mirror_error=max_mirror,parent_policy_under_new_control=summary,episodes=n,samples=samples)
 (ROOT/'smoke_result.json').write_text(json.dumps(result,indent=2));print('RESULT',json.dumps(result),flush=True);e.close()
except BaseException:
 import traceback;traceback.print_exc();app.close(exit_code=1)
else:app.close()
