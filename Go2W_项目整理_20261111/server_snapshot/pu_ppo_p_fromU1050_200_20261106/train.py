"""Standalone Gaussian-latent/tanh PPO, matching the existing handstand PPO conventions."""
import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser()
parser.add_argument('--task',choices=['P'],required=True)
parser.add_argument('--num_envs',type=int,default=1024)
parser.add_argument('--max_iterations',type=int,default=200)
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--log_dir',required=True)
parser.add_argument('--checkpoint')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if args.checkpoint:parser.error('This P run starts from random weights')
launcher=AppLauncher(args); app=launcher.app
import torch
from env import PUEnv,PUCfg,ROOT,GOALS
from model import Policy
from settings import CONTROL, REWARDS, WHEEL_ALIGNMENT

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    torch.manual_seed(args.seed); torch.set_num_threads(8)
    folder=Path(args.log_dir).resolve(); folder.mkdir(parents=True,exist_ok=True)
    params=folder/'params'; params.mkdir(exist_ok=True)
    for f in ROOT.glob('*.py'): shutil.copy2(f,params/f.name)
    shutil.copy2(ROOT/'goals.json',params/'goals.json')
    data_names=('reference_pose.json','reference_states.json','collision_geometry.json','collision_audit.json','limb_collision_geometry.json','reference_validation.json','p_validation.json','reference_adjustments.json','reference_physics.json','pronereference.usd','reward_port.json','standing_initialization.json','standing_source_samples.json')
    for name in data_names:
        shutil.copy2(ROOT/name,params/name)
    manifest=dict(task=args.task,phase='stand.1s descent1.9s; full release and stable endpoint by3s; verify2s hold afterwards',reference_motion=None,actor_dim=279,action_dim=12,
        history=dict(frames=3,order='oldest_first',reset='repeat initial frame'),
        frame_order=['gravity_b[3]','angular_velocity_b*.25[3]','leg_q_rad[12]','joint_vel*.05[16]',
            'previous_action[12]','reference_q-q[12]','previous_target_rad[12]','task_onehot_P_U[2]',
            'phase_blend[1]','deadline_reached[1]','height[1]','linear_velocity_b[3]','vertical_velocity_w[1]','body_load_bw[1]','motor_gain[1]','target_velocity*.25[12]'],
        observation_clip=10,normalization='fixed scales above; no running statistics',
        linear_velocity_actor='body linear velocity plus world vertical velocity; no running normalization',joint_order=GOALS['joint_order'],
        wheel_order=['FR_foot_joint','FL_foot_joint','RR_foot_joint','RL_foot_joint'],
        action='12 residual joint angles +/-.10rad around a direct quintic stand-to-prone reference; motor target max1.4rad/s, acceleration3rad/s2',
        task_version='P-Ustand-v1',initialization='random network; initial physical pose sampled from accepted U1050 standing replay',
        control=CONTROL,rewards=REWARDS,wheel_alignment=WHEEL_ALIGNMENT,reward_port=json.loads((ROOT/'reward_port.json').read_text()),
        wheels='zero speed damping .2; P release fades ALL16 motor efforts to exact zero',kp=70,kd=2.5,effort_limit_nm=23.5,physics_dt=.005,control_dt=.02,
        goals=GOALS,assets={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'assets').rglob('*') if p.is_file()},
        code={p.name:digest(p) for p in ROOT.glob('*.py')},archive_reward_code={p.name:digest(p) for p in (ROOT/'archive_reward_source').glob('*') if p.is_file()},arguments=vars(args),
        ppo=dict(steps=64,epochs=5,minibatches=4,gamma=.995,lam=.95,clip=.2,lr=.0003,entropy=.0005),
        curriculum='none: P starts standing, same user USD prone goal and self-collision assets; target full release and stable endpoint by3s',
        poweroff='User pronereference.usd is the posture reference, adjustments documented; self collision ON. P belly contact with mean body support >.08BW for .12s triggers latched all-motor release over.5s',
        support_feedforward='none for P',
        attitude_assist='calf motor offset clamp(2*pitch*front+2*roll*right,+/-.15rad), through leg motors only; fades to0 with P gains',
        startup='impact failure ignored first.1s for reset contact transient; raw contact peaks always logged; tilt5deg enforced immediately',
        head='actual visual-derived hull geometry; penetration >3mm fails',
        completion='belly supported and ALL16 efforts zero, low joint/body speed and pose error, body peak<2BW, touchdown speed<.2m/s by3s; verify2s stable hold afterwards; level +/-5deg throughout')
    manifest['code'].update({name:digest(ROOT/name) for name in data_names})
    (params/'manifest.json').write_text(json.dumps(manifest,indent=2))
    cfg=PUCfg(); cfg.scene.num_envs=args.num_envs; cfg.seed=args.seed; cfg.sim.device=args.device or 'cuda:0'
    env=PUEnv(cfg,args.task)
    (params/'physics.json').write_text(json.dumps(dict(body_names=env.robot.body_names,
        contact_names=env.contacts.body_names,mass_weight_n=env.weight,leg_limits=env.limits[0].tolist(),
        calibration=dict(head_clearance='visual-derived URDF convex hull floor geometry; head force not separately measurable after fixed joint merge',wheel_force_n=.01,peak_force_n=3*env.weight)),indent=2))
    model=Policy().to(env.device); opt=torch.optim.Adam(model.parameters(),lr=3e-4)
    first=0
    def save(i,name=None):
        dest=folder/(name or f'policy_{i:04d}.pt'); temp=dest.with_suffix('.tmp')
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),iteration=i,level=env.level,
            manifest=manifest,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),temp)
        temp.replace(dest)
    obs,_=env.reset(); x=obs['policy']; iteration=first-1
    try:
        with (folder/'progress.jsonl').open('a',buffering=1) as log:
            for iteration in range(first,first+args.max_iterations):
                start=time.perf_counter(); rollout=[]; episodes=[]
                for _ in range(64):
                    with torch.no_grad():
                        dist=model.distribution(x); raw=dist.sample(); val=model.value(x); lp=dist.log_prob(raw).sum(-1)
                        obs,r,term,trunc,info=env.step(raw.tanh())
                        r=r.clone(); timeouts=trunc&~term
                        r[timeouts]+=.995*model.value(env.terminal_obs[timeouts])
                    rollout.append((x.clone(),raw,lp,val,r,term|trunc)); x=obs['policy']
                    if 'episode' in info: episodes.append(info['episode'].copy())
                data=[torch.stack(a) for a in zip(*rollout)]
                ox,raw,oldlp,values,rewards,dones=data
                with torch.no_grad():
                    adv=torch.zeros_like(rewards); carry=torch.zeros_like(rewards[0]); last=model.value(x)
                    for t in reversed(range(64)):
                        following=last if t==63 else values[t+1]
                        delta=rewards[t]+.995*following*(~dones[t])-values[t]
                        carry=delta+.995*.95*(~dones[t])*carry; adv[t]=carry
                    returns=(adv+values).flatten(); adv=(adv.flatten()-adv.mean())/(adv.std()+1e-8)
                ox=ox.flatten(0,1); raw=raw.flatten(0,1); oldlp=oldlp.flatten(); values=values.flatten()
                losses=[]
                for _ in range(5):
                    for idx in torch.randperm(len(ox),device=env.device).chunk(4):
                        dist=model.distribution(ox[idx]); ratio=(dist.log_prob(raw[idx]).sum(-1)-oldlp[idx]).exp()
                        pl=-torch.minimum(ratio*adv[idx],ratio.clamp(.8,1.2)*adv[idx]).mean()
                        v=model.value(ox[idx]); vc=values[idx]+(v-values[idx]).clamp(-.2,.2)
                        vl=.5*torch.maximum((v-returns[idx]).square(),(vc-returns[idx]).square()).mean()
                        loss=pl+vl-.0005*dist.entropy().sum(-1).mean()
                        if not torch.isfinite(loss): raise RuntimeError('Nonfinite PPO loss')
                        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
                        with torch.no_grad(): model.log_std.clamp_(math.log(.1),math.log(.4))
                        losses.append(float(loss.detach()))
                n=sum(e['count'] for e in episodes)
                row=dict(iteration=iteration+1,level=env.level,episodes=n,loss=sum(losses)/len(losses),
                    fps=round(64*env.num_envs/(time.perf_counter()-start)),seconds=time.perf_counter()-start)
                m=env.metrics()
                loaded=env.torque[:,:12].reshape(-1,4,3)[:,:,1:].abs().amax(-1)>2.01
                loaded_camber=env.p['front_camber']
                row.update(height_mean=float(m['z'].mean()),goal_joint_rms=float(m['qe'].mean().sqrt()),
                    loaded_wheel_camber_max_deg=float(torch.where(loaded,loaded_camber,torch.zeros_like(loaded_camber)).max()*180/math.pi),
                    loaded_wheel_over10_fraction=float((loaded&(loaded_camber>math.radians(10))).float().mean()),
                    front_pair_mean_deg=float(m['front_pair'].mean()*180/math.pi),
                    front_pair_max_deg=float(m['front_pair'].max()*180/math.pi),
                    front_camber_max_deg=float(m['front_camber'].max()*180/math.pi),
                    upward_speed_max_m_s=float(m['vz'].max()),leg_speed_max_rad_s=float(m['leg_speed'].max()),
                    target_speed_max_rad_s=float(env.target_velocity.abs().max()),
                    roll_abs_max_deg=float(m['roll'].abs().max()*180/math.pi),pitch_abs_max_deg=float(m['pitch'].abs().max()*180/math.pi),
                    belly_gap_min_m=float(m['belly'].min()),motor_gain_mean=float(env.gain.mean()),
                    released_fraction=float(env.released.float().mean()),
                    head_clearance_min=float(m['clearance'].min()),peak_contact_n=float(m['peak'].max()),
                    drift_max_m=float(m['drift'].max()),hold_mean_s=float(env.hold.mean()),
                    torque_saturation_fraction=float((env.torque.abs()>23.49).float().mean()))
                row.update(body_peak_bw=float(m['body_peak_bw'].max()),episode_body_peak_bw=float(env.episode_body_peak.max()),touchdown_speed_max_m_s=float(env.touchdown_speed.max()),reference_joint_rms=float((m['q']-env.reference()).square().mean().sqrt()))
                if n: row.update({k:sum(e[k]*e['count'] for e in episodes)/n for k in episodes[0] if k!='count'})
                if len(env.success_window)>=256 and sum(env.success_window)/len(env.success_window)>.8:
                    env.level=min(1.,env.level+.1); env.success_window=[]
                print(json.dumps(row),flush=True); log.write(json.dumps(row)+'\n')
                if (iteration+1)%50==0: save(iteration)
            save(iteration,'final.pt')
            (folder/'completed.json').write_text(json.dumps(dict(iterations=args.max_iterations,final_iteration=iteration+1)))
    finally:
        save(iteration); env.close()

try: main()
except BaseException:
    import traceback
    traceback.print_exc()
    app.close(exit_code=1)
else: app.close()
