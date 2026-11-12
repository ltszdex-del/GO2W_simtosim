import argparse,json
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
import torch
from .env import PUEnv,PUCfg,ROOT
from .model import Policy
try:
 torch.set_num_threads(4);cfg=PUCfg();cfg.scene.num_envs=1;cfg.sim.device='cuda:0';e=PUEnv(cfg,'U');model=Policy().to(e.device).eval()
 ck=torch.load(ROOT/'logs/U_vertical_resume100/policy_1049.pt',map_location=e.device,weights_only=False);model.load_state_dict(ck['model']);o,_=e.reset();samples=[]
 with torch.inference_mode():
  for i in range(900):
   o,reward,term,trunc,info=e.step(model(o['policy']))
   if (term|trunc).any():raise RuntimeError('U failed before pose capture')
   if i>=800:
    d=e.robot.data;m=e.metrics();samples.append(dict(q=d.joint_pos.torch[0,e.legs].tolist(),wheel_q=d.joint_pos.torch[0,e.wheels].tolist(),height=float(m['z'][0]),roll=float(m['roll'][0]),pitch=float(m['pitch'][0]),camber_deg=(m['front_camber'][0]*180/3.141592653589793).tolist()))
 result=dict(source_checkpoint=str(ROOT/'logs/U_vertical_resume100/policy_1049.pt'),source_iteration=1050,window_s=[16.,18.],samples=samples)
 (ROOT/'standing_pose_for_P.json').write_text(json.dumps(result,indent=2));print('CAPTURED',json.dumps(samples[-1]),flush=True);e.close()
except BaseException:
 import traceback;traceback.print_exc();app.close(exit_code=1)
else:app.close()
