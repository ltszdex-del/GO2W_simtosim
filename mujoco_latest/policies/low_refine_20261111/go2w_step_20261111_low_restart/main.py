"""Five-stage WASD PPO training / arrow visualization. Starts Kit before env imports."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from isaaclab.app import AppLauncher

root=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--mode',choices=['train','smoke','play'],default='train')
p.add_argument('--iterations',type=int,default=3000)
p.add_argument('--num_envs',type=int,default=1024)
p.add_argument('--checkpoint')
p.add_argument('--stage',type=int,choices=range(1,6),default=1)
p.add_argument('--log_dir',required=True)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
from config import contract,URDF
manifest=contract()
manifest['curriculum']=dict(stages=5,promotion=.90,consecutive_gates=2,gate_every=100,
    test_seconds=11,rmse_xy=.25,heading_deg=5,tilt_deg=10,height_error=.08,stop_speed=.08)
manifest['curriculum']['promotion_scope']='overall; directions reported separately'
manifest['urdf_sha256']=hashlib.sha256(URDF.read_bytes()).hexdigest()
manifest['source_sha256']={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in root.glob('*.py')}
folder=Path(a.log_dir).resolve();folder.mkdir(parents=True,exist_ok=True)
params=folder/'params';(params/'source').mkdir(parents=True,exist_ok=True)
for x in root.glob('*.py'):shutil.copy2(x,params/'source'/x.name)
(params/'manifest.json').write_text(json.dumps(manifest,indent=2))
(params/'arguments.json').write_text(json.dumps(vars(a),indent=2))
launcher=AppLauncher(a);app=launcher.app;env=None;exit_code=0
try:
    import torch
    from env import LocomotionEnv
    from env_cfg import EnvCfg
    from learning import Policy,train
    torch.set_num_threads(4);torch.manual_seed(42)
    cfg=EnvCfg();cfg.scene.num_envs=a.num_envs;cfg.seed=42;cfg.sim.device=a.device or 'cuda:0'
    cfg.training=a.mode=='train';cfg.episode_length_s=12.
    cfg.scene.env_spacing=40.
    cfg.viewer.eye=(5.,8.,5.);cfg.viewer.lookat=(3.,0.,.4)
    env=LocomotionEnv(cfg);env.stage=a.stage
    env.render_enabled=bool(a.livestream)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False) if a.checkpoint else None
    if ck and ck['manifest']!=manifest:raise ValueError('Checkpoint source/interface mismatch')
    (params/'runtime_physics.json').write_text(json.dumps(dict(
        mass=float(env.robot.data.body_mass.torch[0].sum()),dt=env.physics_dt,control_dt=env.step_dt)))
    print(f'[READY] {a.mode} envs={env.num_envs} stage={env.stage} output={folder}',flush=True)
    if a.mode=='train':train(env,app,folder,manifest,a.iterations,ck)
    else:
        model=Policy().to(env.device)
        if ck:model.load_state_dict(ck['model']);env.stage=ck['curriculum']['stage']
        if a.mode=='play' and not ck:raise ValueError('play requires --checkpoint')
        if a.mode=='smoke':
            # Physics smoke stays stationary; command schedule verified separately.
            env.schedule.tick=lambda stage,age,dt:torch.zeros(env.num_envs,3,device=env.device)
        obs,_=env.reset();steps=failures=0
        arrows=None
        if a.mode in ('play','smoke'):
            from arrows import VelocityArrows
            arrows=VelocityArrows()
        with torch.no_grad():
            while app.is_running() and (a.mode=='play' or steps<150):
                start=time.monotonic()
                action=model(obs['policy']) if ck else torch.zeros(env.num_envs,12,device=env.device)
                obs,r,term,trunc,_=env.step(action);steps+=1;failures+=int(term.sum())
                assert env.pd_updates==4 and torch.isfinite(r).all() and torch.isfinite(obs['policy']).all()
                if arrows:arrows.update(env)
                if a.livestream:time.sleep(max(0.,env.step_dt-(time.monotonic()-start)))
        result=dict(steps=steps,failures=failures,actor_shape=list(obs['policy'].shape),pd_substeps=env.pd_updates)
        (folder/'smoke.json').write_text(json.dumps(result));print('[SMOKE] '+json.dumps(result),flush=True)
        if failures:raise RuntimeError('Physical standing smoke failed')
        if a.mode=='smoke':
            damping=env.robot.data.joint_damping.torch[0,env.wheel_ids]
            assert torch.allclose(damping,torch.full_like(damping,100.)),damping
            print('[LOCK] damping='+str(damping.cpu().tolist()),flush=True)
        if a.mode=='smoke':
            from gate import evaluate
            # Exercise held-out evaluation, failure masking and restoration.
            assert not evaluate(env,model,folder,-1,[1]), 'Untrained actor unexpectedly passed'
    (folder/'exit_code').write_text('0')
except BaseException:
    import traceback
    traceback.print_exc();exit_code=1
    (folder/'exit_code').write_text('1')
finally:
    if env is not None:env.close()
    app.close(exit_code=exit_code)
