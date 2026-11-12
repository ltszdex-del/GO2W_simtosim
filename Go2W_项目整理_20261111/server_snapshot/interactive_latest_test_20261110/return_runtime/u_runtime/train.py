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
parser.add_argument('--task',choices=['U'],required=True)
parser.add_argument('--num_envs',type=int,default=1024)
parser.add_argument('--max_iterations',type=int,default=100)
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--log_dir',required=True)
parser.add_argument('--checkpoint')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if not args.checkpoint or args.task!='U':parser.error('U continuation requires --checkpoint')
launcher=AppLauncher(args); app=launcher.app
import torch
from .env import PUEnv,PUCfg,ROOT,GOALS
from .model import Policy
from .settings import CONTROL, REWARDS, WHEEL_ALIGNMENT

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    torch.manual_seed(args.seed); torch.set_num_threads(8)
    folder=Path(args.log_dir).resolve(); folder.mkdir(parents=True,exist_ok=True)
    params=folder/'params'; params.mkdir(exist_ok=True)
    for f in ROOT.glob('*.py'): shutil.copy2(f,params/f.name)
    shutil.copy2(ROOT/'goals.json',params/'goals.json')
    data_names=('reference_pose.json','reference_states.json','collision_geometry.json','collision_audit.json','limb_collision_geometry.json','reference_validation.json','p_validation.json','reference_adjustments.json','reference_physics.json','pronereference.usd','symmetry_config.json','speed_config.json')
    for name in data_names:
        shutil.copy2(ROOT/name,params/name)
    manifest=dict(task=args.task,phase='height schedule2.3s, standing endpoint deadline3s; verify1s hold afterwards',reference_motion=None,actor_dim=216,action_dim=12,
        history=dict(frames=3,order='oldest_first',reset='repeat initial frame'),
        frame_order=['gravity_b[3]','angular_velocity_b*.25[3]','leg_q_rad[12]','joint_vel*.05[16]',
            'previous_action[12]','goal_q-q[12]','previous_target_rad[12]','task_onehot_P_U[2]'],
        observation_clip=10,normalization='fixed scales above; no running statistics',
        linear_velocity_actor='none; use proprioceptive history',joint_order=GOALS['joint_order'],
        wheel_order=['FR_foot_joint','FL_foot_joint','RR_foot_joint','RL_foot_joint'],
        symmetry='mirror action/target/target velocity per FR_FL and RR_RL; hips antisymmetric, thigh/calf equal; extra feet/joint symmetry and world wheel lean cost; completion joint5deg feet2cm wheels5deg',
        action='12 target velocities:1.4*clamp(goal_q-target,-1,1)+.5*tanh(actor), max1.2rad/s, acceleration2.4rad/s2; hips gain1.8; preserve mirror and loaded camber cost',
        task_version='U-vertical-v1',initialization='resume U1000 model and optimizer; add100 updates at reduced learning rate',
        control=CONTROL,rewards=REWARDS,wheel_alignment=WHEEL_ALIGNMENT,speed_config=json.loads((ROOT/'speed_config.json').read_text()),
        wheels='zero speed damping .2; retain wheel orientation costs and final5deg limit',kp=70,kd=2.5,standing_hip_kp=210,standing_hip_kd=6,effort_limit_nm=23.5,physics_dt=.005,control_dt=.02,
        goals=GOALS,assets={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'assets').rglob('*') if p.is_file()},
        code={p.name:digest(p) for p in ROOT.glob('*.py')},arguments=vars(args),
        ppo=dict(steps=64,epochs=5,minibatches=4,gamma=.995,lam=.95,clip=.2,lr=.0001,entropy=.0005),
        curriculum='none: U same adjusted user prone reference, self collision ON, no in-episode root/joint state writes',
        poweroff='User pronereference.usd is the posture reference, adjustments documented; self collision ON. U first control step is zero effort, then motor gain ramps over.02s',
        support_feedforward='U: analytic foot Jacobian transpose times .9*weight/4, same global motor gain and torque cap; link lengths .213,.2264m, side offset.1435m',
        attitude_assist='calf motor offset clamp(2*pitch*front+2*roll*right,+/-.15rad), through leg motors only',
        startup='impact failure ignored first.1s for reset contact transient; raw contact peaks always logged; tilt5deg enforced immediately',
        head='actual visual-derived hull geometry; penetration >3mm fails; U after lifting >25mm recontact <3mm fails',
        standing='getup plus sustained standing to20s; level +/-5deg at all steps; loaded wheel world camber beyond10deg penalized')
    manifest['code'].update({name:digest(ROOT/name) for name in data_names})
    (params/'manifest.json').write_text(json.dumps(manifest,indent=2))
    cfg=PUCfg(); cfg.scene.num_envs=args.num_envs; cfg.seed=args.seed; cfg.sim.device=args.device or 'cuda:0'
    env=PUEnv(cfg,args.task)
    (params/'physics.json').write_text(json.dumps(dict(body_names=env.robot.body_names,
        contact_names=env.contacts.body_names,mass_weight_n=env.weight,leg_limits=env.limits[0].tolist(),
        calibration=dict(head_clearance='visual-derived URDF convex hull floor geometry; head force not separately measurable after fixed joint merge',wheel_force_n=.01,peak_force_n=3*env.weight)),indent=2))
    model=Policy().to(env.device); opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    first=0
    if args.checkpoint:
        ck=torch.load(args.checkpoint,map_location=env.device,weights_only=False)
        if ck['manifest']['assets']!=manifest['assets'] or ck['manifest']['goals']!=GOALS or ck['manifest']['task']!=args.task:
            raise ValueError('Checkpoint interface mismatch')
        if ck['manifest']['task_version']!='U-fast3s-v1' or ck['manifest']['actor_dim']!=216:raise ValueError('Unexpected parent U policy')
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); first=ck['iteration']+1; env.level=ck['level']
        for group in opt.param_groups:group['lr']=1e-4
        manifest['parent_checkpoint']=str(Path(args.checkpoint).resolve());manifest['parent_iteration']=first
        torch.set_rng_state(ck['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([r.cpu() for r in ck['cuda_rng']])
        (params/'resume_changes.json').write_text(json.dumps(dict(parent=str(Path(args.checkpoint).resolve()),
            parent_iteration=ck['iteration']+1,additional_iterations=args.max_iterations,
            old_code=ck['manifest']['code'],new_code=manifest['code'],control=CONTROL,rewards=REWARDS,
            wheel_alignment=WHEEL_ALIGNMENT,optimizer='restored',actor_interface='unchanged 216/12'),indent=2))
    def save(i,name=None):
        dest=folder/(name or f'policy_{i:04d}.pt'); temp=dest.with_suffix('.tmp')
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),iteration=i,level=env.level,
            manifest=manifest,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),temp)
        temp.replace(dest)
    save(first-1,'baseline.pt')  # Exact parent actor under the validated fast controller.
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
                row_sym=dict(symmetry_rms_deg=float(m['symmetry_rms'].mean()*180/math.pi),symmetry_max_deg=float(m['symmetry_max'].max()*180/math.pi),foot_mirror_max_m=float(m['foot_mirror_max'].max()))
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
                row.update(row_sym)
                row.update(stood_fraction_current=float((env.standing_time>=0).float().mean()),on_time_fraction_current=float(((env.standing_time>=0)&(env.standing_time<=3.000001)&~env.failed).float().mean()))
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
