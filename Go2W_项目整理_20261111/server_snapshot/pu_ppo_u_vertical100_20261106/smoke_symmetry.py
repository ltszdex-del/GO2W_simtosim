import argparse,json,math
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
try:
 import torch
 from env import PUEnv,PUCfg,ROOT
 from model import Policy
 cfg=PUCfg();cfg.scene.num_envs=32;cfg.sim.device='cuda:0'
 env=PUEnv(cfg,'U')
 ck=torch.load('/home/ubuntu/Go2W/pu_ppo_proneref_matched_20261106/logs/U_adjusted_proneref400/final.pt',weights_only=False,map_location=env.device)
 model=Policy().to(env.device);model.load_state_dict(ck['model']);model.eval()
 obs,_=env.reset()
 maximum=0.
 for _ in range(150):
  with torch.no_grad():obs,reward,term,trunc,info=env.step(model(obs['policy']))
  assert torch.isfinite(reward).all() and torch.isfinite(obs['policy']).all()
  assert env.pd_count==4
  maximum=max(maximum,float((env.target-env.mirror(env.target)).abs().max()))
 assert maximum<1e-5,maximum
 m=env.metrics()
 report=dict(parent_iteration=ck['iteration']+1,finite=True,target_mirror_residual_max_rad=maximum,observation_dim=obs['policy'].shape[1],height_mean=float(m['z'].mean()),symmetry_rms_deg=float(m['symmetry_rms'].mean()*180/math.pi),wheel_camber_max_deg=float(m['front_camber'].max()*180/math.pi),note='Control/interface smoke test, not trained policy acceptance')
 (ROOT/'symmetry_smoke.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);env.close()
except BaseException:
 import traceback;traceback.print_exc();app.close(exit_code=1)
else:app.close()
