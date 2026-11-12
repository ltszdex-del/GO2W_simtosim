"""Replay the exact level-P/U controller and checkpoint, looping full episodes."""
import argparse,hashlib,json,time,traceback
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app;env=None
try:
 import torch,carb.settings
 torch.set_num_threads(4)
 from env import PUEnv,PUCfg,ROOT,GOALS
 from model import Policy
 checkpoint=Path(args.checkpoint)
 if checkpoint.is_dir():
  candidates=list(checkpoint.glob('policy_*.pt'))
  checkpoint=checkpoint/'final.pt' if (checkpoint/'final.pt').exists() else max(candidates,key=lambda p:int(p.stem.split('_')[-1]))
 ck=torch.load(checkpoint,map_location='cpu',weights_only=False);manifest=ck['manifest'];iteration=ck['iteration']+1
 assert manifest['task_version']=='P-Ustand-v1'
 for name,digest in manifest['code'].items():
  assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,'Checkpoint code mismatch: '+name
 for name,digest in manifest['assets'].items():
  assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,'Checkpoint asset mismatch: '+name
 assert manifest['task'] in ('P','U')
 settings=carb.settings.get_settings()
 for key,value in {'/rtx/rendermode':'RealTimePathTracing','/rtx/post/aa/op':4,'/rtx/post/dlss/execMode':2,'/rtx-transient/dlssg/enabled':False}.items():settings.set(key,value)
 cfg=PUCfg();cfg.scene.num_envs=1;cfg.sim.device='cuda:0'
 cfg.viewer.eye=(1.3,1.3,.75);cfg.viewer.lookat=(0.,0.,.20)
 env=PUEnv(cfg,manifest['task']);model=Policy().to(env.device).eval();model.load_state_dict(ck['model']);obs,_=env.reset()
 from omni.kit.viewport.utility import get_active_viewport,capture_viewport_to_file
 viewport=get_active_viewport();viewport.texture_resolution=(1280,720)
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 print(f'READY {manifest["task"]} iteration{iteration}; checkpoint={checkpoint}; loop20s or failure; 106.75.23.13:49100',flush=True)
 with torch.inference_mode(),(out/'trace.jsonl').open('w',buffering=1) as log:
  step=0
  while app.is_running():
   start=time.perf_counter();obs,reward,term,trunc,info=env.step(model(obs['policy']))
   if step%50==0 or bool((term|trunc)[0]):
    m=env.metrics();row=dict(step=step,iteration=iteration,height=float(m['z'][0]),belly_gap=float(m['belly'][0]),
     roll_deg=float(m['roll'][0]*180/3.141592653589793),pitch_deg=float(m['pitch'][0]*180/3.141592653589793),
     camber_deg=(m['front_camber'][0]*180/3.141592653589793).tolist(),
     reset=bool((term|trunc)[0]),episode=info.get('episode'))
    print(json.dumps(row),flush=True);log.write(json.dumps(row)+'\n')
   if step in (100,400):capture_viewport_to_file(viewport,str(out/f'frame_{step:04d}.png'))
   step+=1;time.sleep(max(0,.02-(time.perf_counter()-start)))
except BaseException:traceback.print_exc()
finally:
 if env is not None:env.close()
 app.close()
