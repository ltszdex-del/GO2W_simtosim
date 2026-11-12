import argparse,json
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
try:
 import torch
 from env import PUEnv,PUCfg,ROOT
 cfg=PUCfg();cfg.scene.num_envs=8;cfg.sim.device='cuda:0'
 env=PUEnv(cfg,'P');obs,_=env.reset()
 for _ in range(200):
  obs,reward,term,trunc,info=env.step(torch.zeros(8,12,device=env.device))
  assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
 env.released[:]=True;env.release_time[:]=1.1
 env._pre_physics_step(torch.zeros(8,12,device=env.device));env._apply_action()
 assert env.gain.max()==0 and env.torque.abs().max()==0
 assert torch.equal(env.goal,env.prone)
 (ROOT/'p_validation.json').write_text(json.dumps(dict(finite=True,P_goal_is_user_reference=True,all16_torque_exactly_zero_after_release=True)))
 print('P VALIDATED',flush=True);env.close()
except BaseException:
 import traceback;traceback.print_exc();app.close(exit_code=1)
else:app.close()
