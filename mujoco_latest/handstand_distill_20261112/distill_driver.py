"""DAgger distillation of the exact deployed H teacher, with paired evaluation."""
import argparse,hashlib,json,shutil,sys,traceback
from pathlib import Path
from isaaclab.app import AppLauncher

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser()
p.add_argument('--num_envs',type=int,default=2048)
p.add_argument('--iterations',type=int,default=1000)
p.add_argument('--checkpoint',required=True)
p.add_argument('--log_dir',required=True)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
import torch
torch.set_num_threads(4);torch.manual_seed(42)
from go2w_mimic.interface import manifest,verify
from go2w_mimic.settings import MOTION
checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
interface=manifest(MOTION);verify(checkpoint['interface'],interface)
assert checkpoint['kind']=='teacher' and checkpoint['iteration']==4179
folder=Path(a.log_dir);folder.mkdir(parents=True,exist_ok=True)
params=folder/'params';params.mkdir(exist_ok=True)
(params/'interface.json').write_text(json.dumps(interface,indent=2))
(params/'arguments.json').write_text(json.dumps(vars(a),indent=2))
shutil.copy2(MOTION,params/'stand_to_handstand.npz')
launcher=AppLauncher(a);app=launcher.app;env=None;exit_code=1
try:
    from go2w_mimic.env import MimicEnv
    from go2w_mimic.env_cfg import MimicEnvCfg
    from go2w_mimic.learning import distill
    from go2w_mimic.networks import Teacher,Student
    cfg=MimicEnvCfg();cfg.scene.num_envs=a.num_envs;cfg.sim.device=a.device or 'cuda:0'
    cfg.seed=42;cfg.motion_path=str(MOTION);cfg.randomize=True
    cfg.domain_level=float(checkpoint['curriculum']);cfg.episode_length_s=12.
    env=MimicEnv(cfg)
    print('[READY] DAgger teacher4179 envs='+str(a.num_envs)+' iterations='+str(a.iterations),flush=True)
    distill(env,app,folder,interface,a.iterations,checkpoint)
    rows=[json.loads(x) for x in (folder/'progress.jsonl').read_text().splitlines() if x.strip()]
    assert [x['iteration'] for x in rows]==list(range(a.iterations))
    student_path=folder/f'student_{a.iterations-1}.pt'
    student_ck=torch.load(student_path,map_location='cpu',weights_only=False)
    verify(student_ck['interface'],interface)
    models={}
    models['teacher']=Teacher().to(env.device).eval();models['teacher'].load_state_dict(checkpoint['model'])
    models['student']=Student().to(env.device).eval();models['student'].load_state_dict(student_ck['model'])
    # Same environment masses and nominal resets; no observation noise in this
    # held-out evaluation. Ten-second episodes never hit the twelve-second timeout.
    env.cfg.randomize=False
    evaluations={}
    with torch.inference_mode():
        for kind,model in models.items():
            torch.manual_seed(18217);obs,_=env.reset()
            alive=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
            hold_good=torch.zeros(env.num_envs,device=env.device)
            error=torch.zeros((),device=env.device);samples=0
            for k in range(500):
                action=model(obs['policy' if kind=='teacher' else 'student'])
                if kind=='student':
                    error+=(action-models['teacher'](obs['policy'])).square().mean();samples+=1
                obs,_,term,trunc,_=env.step(action)
                alive&=~(term|trunc)
                if k>=250:hold_good+=(env.robot.data.projected_gravity_b.torch[:,0]>.9).float()
            good=alive&(hold_good/250>.95)
            evaluations[kind]=dict(trials=env.num_envs,success_rate=float(good.float().mean()),
                survival_rate=float(alive.float().mean()),mean_hold_fraction=float((hold_good/250).mean()),
                online_action_mse=float(error/max(1,samples)))
            print('[EVALUATION] '+json.dumps({kind:evaluations[kind]}),flush=True)
    evaluations['passed']=evaluations['student']['success_rate']>=max(.9,.9*evaluations['teacher']['success_rate'])
    (folder/'paired_evaluation.json').write_text(json.dumps(evaluations,indent=2))
    exit_code=0
except BaseException:
    traceback.print_exc()
finally:
    (folder/'exit_code').write_text(str(exit_code))
    if env is not None:env.close()
    app.close(exit_code=exit_code)
