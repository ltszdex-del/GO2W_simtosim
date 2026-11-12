"""PPO landing trainer. Frozen-H warm-up samples are excluded from actor/critic losses."""
import argparse,hashlib,json,math,shutil,time
from pathlib import Path
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser()
parser.add_argument('--num_envs',type=int,default=1024)
parser.add_argument('--iterations',type=int,default=1000)
parser.add_argument('--log_dir',default='logs/train1000')
parser.add_argument('--checkpoint')
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--allow_reward_control_update',action='store_true')
parser.add_argument('--adapt_fast_return',action='store_true')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
args.headless=True
launcher=AppLauncher(args);app=launcher.app
import torch
from settings import *
from env import ReturnCfg,ReturnEnv,LEG_NAMES,WHEEL_NAMES
from model import Policy
from ppo_math import masked_gae

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    torch.set_num_threads(6);torch.manual_seed(args.seed)
    cfg=ReturnCfg();cfg.scene.num_envs=args.num_envs;cfg.sim.device=args.device or 'cuda:0';cfg.seed=args.seed
    env=ReturnEnv(cfg)
    folder=ROOT/args.log_dir;folder.mkdir(parents=True,exist_ok=True)
    params=folder/'params';params.mkdir(exist_ok=True)
    for f in ROOT.glob('*.py'):shutil.copy2(f,params/f.name)
    for name in ('U_final_standing.json','early_handstand_states.json','early_handstand_capture.json','provenance.json'):
        shutil.copy2(ROOT/name,params/name)
    manifest=dict(schema='go2w-handstand-return-v1',kind='handstand_return',actor_dim=ACTOR_DIM,frame_dim=FRAME,history=HISTORY,
        action_dim=ACTION_DIM,hidden=[256,128,128],history_order='oldest first, repeat first frame after reset',
        observation_clip=10.,normalization='fixed scales in frame; no running statistics',
        frame_order=['gravity_b[3]','omega_b*.25[3]','leg_q-U_final_actual_q[12]','joint_dq*.05[16]',
            'last_applied_action[16]','target-U_final_loaded_target[12]','wheel_target*.05[4]',
            'U_final_actual_q-leg_q[12]','root_lin_vel_b[3]','height[1]','foot_clearance[4]',
            'foot_contacts[4]','heading_error_sin_cos[2]','return_active_and_phase[2]','rear_landed_fraction[1]','front_anchor_heading*.2^-1[4]','axle_offsets_body*.05^-1[6]',
            'axle_direction_difference_body*.15^-1[6]','rear_positions_heading*.5^-1[4]'],
        simulation_measurements=['root linear velocity','root height','foot clearance','contacts'],
        joint_order=LEG_NAMES+WHEEL_NAMES,control=CONTROL,task=TASK,rewards=REWARDS,ppo=PPO,
        physics=dict(dt=DT,decimation=DECIMATION,quaternion='xyzw',wheel_implicit_damping=0.,self_collision=True),
        teacher=json.loads((ROOT/'provenance.json').read_text()),
        source_sha256={f.name:digest(f) for f in ROOT.glob('*.py')},
        state_sha256={name:digest(ROOT/name) for name in ('U_final_standing.json','early_handstand_states.json')},
        urdf_sha256=digest(ROOT/'assets/go2w_description/urdf/go2w_description.urdf'),
        warmup='Frozen H4099 continues until random dwell expires AND measured motion threshold; no physical state writes at handoff',
        training_mask='Warm-up rows excluded from both PPO actor and critic updates; GAE carry zeroed on warm-up',
        timeout='One-second recovery deadline is terminal failure; no timeout bootstrap',
        adaptation=dict(parent=args.checkpoint,fast_return=args.adapt_fast_return,reset_critic_and_optimizer=args.adapt_fast_return),
        arguments=vars(args))
    (params/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (params/'physics.json').write_text(json.dumps(dict(body_names=env.robot.body_names,joint_names=env.robot.joint_names,
        nominal_first_mass_kg=float(env.robot.data.body_mass.torch[0].sum()),
        wheel_damping=env.robot.data.joint_damping.torch[0,env.wheels].tolist(),
        target_U_actual_q=env.goal_q.tolist(),target_U_loaded_target=env.goal_target.tolist()),indent=2))
    model=Policy().to(env.device);opt=torch.optim.Adam(model.parameters(),lr=PPO['lr']);first=0
    if args.checkpoint:
        ck=torch.load(args.checkpoint,map_location=env.device,weights_only=False)
        assert ck['manifest']['schema']==manifest['schema']
        changes={}
        fast_task={'guidance_easy_s','guidance_hard_s','recovery_timeout_s','success_hold_s',
            'require_soft_landing_for_success','stand_horizontal_speed_m_s','stand_vertical_speed_m_s',
            'stand_pose_angular_speed_rad_s','fail_rear_peak_bodyweights'}
        fast_control={'leg_rate','leg_acceleration','handoff_leg_rate','rear_softening_seconds','rear_prepare_s','U_motor_blend_time_s'}
        fast_rewards={'rear_soft_touch_bonus','success_bonus','speed_success_bonus','rear_approach_speed',
            'descent_speed','touchdown_speed','rear_impact','force_slew','time'}
        if args.adapt_fast_return:
            assert TASK['recovery_timeout_s']==1. and TASK['success_hold_s']==.1
            assert TASK['require_soft_landing_for_success'] is False
        for key in ('source_sha256','state_sha256','control','task','rewards','urdf_sha256'):
            if ck['manifest'][key]==manifest[key]:continue
            if not (args.allow_reward_control_update or args.adapt_fast_return):raise ValueError('Checkpoint mismatch: '+key)
            old,new=ck['manifest'][key],manifest[key]
            changed={k for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
            if key=='source_sha256':
                assert old.keys()==new.keys()
                assert changed<={'env.py','settings.py','train.py'}
            elif args.adapt_fast_return and key=='task':assert changed<=fast_task
            elif key=='control':
                allowed=fast_control if args.adapt_fast_return else {'front_wheel_pitch_kp','front_wheel_pitch_kd','front_wheel_descent_bias'}
                assert changed<=allowed
            elif key=='rewards':
                if args.adapt_fast_return:assert changed<=fast_rewards
            else:raise ValueError('Unsafe migration: '+key)
            changes[key]=dict(before=old,after=new)
        for key in ('actor_dim','frame_dim','history','action_dim','joint_order','physics','teacher'):
            assert ck['manifest'][key]==manifest[key],key
        (folder/'resume_changes.json').write_text(json.dumps(dict(parent=args.checkpoint,changes=changes),indent=2))
        fresh_critic={k:v.clone() for k,v in model.critic.state_dict().items()} if args.adapt_fast_return else None
        model.load_state_dict(ck['model'])
        if args.adapt_fast_return:model.critic.load_state_dict(fresh_critic)
        else:opt.load_state_dict(ck['optimizer'])
        first=ck['iteration']+1;env.level=ck['curriculum']
        torch.set_rng_state(ck['torch_rng'].cpu())
        if ck.get('cuda_rng'):torch.cuda.set_rng_state_all([r.cpu() for r in ck['cuda_rng']])
    iteration=first-1
    def save(name):
        path=folder/name;tmp=path.with_suffix('.tmp')
        torch.save(dict(kind='handstand_return',model=model.state_dict(),optimizer=opt.state_dict(),
            iteration=iteration,curriculum=env.level,manifest=manifest,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),tmp)
        tmp.replace(path)
    obs,_=env.reset();x=obs['policy']
    try:
        with (folder/'progress.jsonl').open('a',buffering=1) as progress,(folder/'status.log').open('a',buffering=1) as status:
            for iteration in range(first,first+args.iterations):
                begin=time.perf_counter();rollout=[];episodes=[]
                for _ in range(PPO['steps']):
                    if not app.is_running():raise RuntimeError('Isaac app stopped before training completed')
                    with torch.no_grad():
                        distribution=model.distribution(x);raw=distribution.sample()
                        val=model.value(x);lp=distribution.log_prob(raw).sum(-1)
                        ob,r,term,trunc,info=env.step(raw.tanh())
                        mask=env.last_step_mask.clone();r=r.clone()
                        timeout=trunc&~term&mask
                        if timeout.any():r[timeout]+=PPO['gamma']*model.value(env.terminal_obs[timeout])
                    rollout.append((x.clone(),raw,lp,val,r,term|trunc,mask));x=ob['policy']
                    if 'episode' in info:episodes.append(dict(info['episode']))
                ox,raw,oldlp,values,rewards,dones,mask=[torch.stack(v) for v in zip(*rollout)]
                with torch.no_grad():
                    adv,returns=masked_gae(rewards,dones,values,model.value(x),mask,PPO['gamma'],PPO['lam'])
                valid=mask.flatten();samples=int(valid.sum())
                if samples<128:raise RuntimeError(f'Only {samples} controlled samples; inspect teacher warm-up/handoff')
                ox=ox.flatten(0,1)[valid];raw=raw.flatten(0,1)[valid];oldlp=oldlp.flatten()[valid]
                values=values.flatten()[valid];returns=returns.flatten()[valid];adv=adv.flatten()[valid]
                adv=(adv-adv.mean())/(adv.std()+1e-8)
                losses=[]
                for _ in range(PPO['epochs']):
                    for ids in torch.randperm(samples,device=env.device).chunk(PPO['minibatches']):
                        dist=model.distribution(ox[ids]);ratio=(dist.log_prob(raw[ids]).sum(-1)-oldlp[ids]).exp()
                        policy=-torch.minimum(ratio*adv[ids],ratio.clamp(1-PPO['clip'],1+PPO['clip'])*adv[ids]).mean()
                        value=model.value(ox[ids]);clipped=values[ids]+(value-values[ids]).clamp(-PPO['clip'],PPO['clip'])
                        vloss=.5*torch.maximum((value-returns[ids]).square(),(clipped-returns[ids]).square()).mean()
                        loss=policy+vloss-PPO['entropy']*dist.entropy().sum(-1).mean()
                        if not torch.isfinite(loss):raise RuntimeError('Nonfinite PPO loss')
                        opt.zero_grad(set_to_none=True);loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),PPO['grad_clip']);opt.step()
                        with torch.no_grad():model.log_std.clamp_(math.log(PPO['min_std']),math.log(PPO['max_std']))
                        losses.append(float(loss.detach()))
                completed=sum(e['count'] for e in episodes)
                aggregate={key:sum(e[key]*e['count'] for e in episodes)/completed for key in episodes[0] if key!='count'} if completed else {}
                if len(env.success_window)>=512:
                    if sum(env.success_window[-512:])/512>=.8:
                        env.level=min(1.,env.level+.15);env.success_window=[]
                    else:env.success_window=env.success_window[-512:]
                row=dict(iteration=iteration+1,checkpoint_iteration=iteration,completed=completed,controlled_samples=samples,
                    sample_fraction=samples/(PPO['steps']*env.num_envs),loss=sum(losses)/len(losses),
                    fps=round(PPO['steps']*env.num_envs/(time.perf_counter()-begin)),seconds=time.perf_counter()-begin,
                    action_std=float(model.log_std.detach().exp().mean()))
                row.update(info.get('log',{}));row.update(aggregate)
                success='--' if not completed else f"{aggregate['success']:.1%}"
                message=(f"iter={iteration+1} success={success} completed={completed} active={row['sample_fraction']:.1%} "
                    f"pitch={row.get('pitch_deg',0):.1f}deg rear={row.get('rear_landed_fraction',0):.1%} "
                    f"hold={row.get('best_hold_s',0):.2f}s impact={row.get('rear_peak_bw',0):.2f}BW "
                    f"coax={row.get('front_coaxial_cm',0):.1f}/{row.get('rear_coaxial_cm',0):.1f}cm yaw={row.get('heading_abs_deg',0):.1f}deg "
                    f"warm_fail={env.teacher_resets} level={env.level:.2f} fps={row['fps']}")
                print(message,flush=True);status.write(message+'\n');progress.write(json.dumps(row)+'\n')
                if (iteration+1)%50==0:save(f'policy_{iteration:04d}.pt')
            save('final.pt')
            (folder/'completed.json').write_text(json.dumps(dict(iterations=args.iterations,checkpoint_iteration=iteration)))
    finally:
        save(f'policy_{iteration:04d}.pt');env.close()

try:main()
except BaseException:
    import traceback
    traceback.print_exc();app.close(exit_code=1)
else:app.close()
