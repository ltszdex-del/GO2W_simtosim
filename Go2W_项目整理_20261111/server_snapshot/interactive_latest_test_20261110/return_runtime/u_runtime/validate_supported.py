import argparse,json,math
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
try:
 import torch
 from env import PUEnv,PUCfg,ROOT
 report={}
 cfg=PUCfg();cfg.scene.num_envs=8;cfg.sim.device='cuda:0'
 env=PUEnv(cfg,'U',probe=True);obs,_=env.reset()
 expected=torch.tensor(env.states[0]['q'],device=env.device)
 expected_wheels=torch.tensor(env.states[0]['wheel_q'],device=env.device)
 assert (env.robot.data.joint_pos.torch[:,env.legs]-expected).abs().max()<1e-6
 assert (env.robot.data.joint_pos.torch[:,env.wheels]-expected_wheels).abs().max()<1e-6
 assert ((expected>=env.actual_limits[0,:,0])&(expected<=env.actual_limits[0,:,1])).all()
 maximum=0.;reasons={}
 for step in range(200):
  obs,reward,term,trunc,info=env.step(torch.zeros(8,12,device=env.device))
  assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
  assert env.pd_count==4
  if step==0:assert env.torque.abs().max()==0
  m=env.metrics();maximum=max(maximum,float(torch.maximum(m['roll'].abs(),m['pitch'].abs()).max()*180/math.pi))
  if step in (0,4,9,19,49,99,199):
   row=dict(step=step,height=float(m['z'].mean()),body_angle_max_deg=maximum,qe=float(m['qe'].mean()),belly_gap=float(m['belly'].mean()),camber_max_deg=float(m['front_camber'].max()*180/math.pi),failures={k:float(v.float().mean()) for k,v in env.reason_masks.items()})
   print(json.dumps(row),flush=True);report[str(step)]=row
 report.update(exact_reference_joints=True,exact_reference_wheel_angles=True,initial_torque_zero=True,pd_substeps=4,finite=True)
 (ROOT/'reference_validation.json').write_text(json.dumps(report,indent=2));env.close()
except BaseException:
 import traceback;traceback.print_exc();app.close(exit_code=1)
else:app.close()
