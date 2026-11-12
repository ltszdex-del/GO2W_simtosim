"""Asymmetric PPO with exact latent likelihood ratios for tanh-bounded actions."""
import json
import math
import time
from pathlib import Path
import torch
from torch import nn
from symmetry import mirror_action, mirror_observation
from config import ACTOR_DIM, CRITIC_DIM, PPO, CONTROL

def mlp(inputs,outputs,last_gain=1.):
    layers=[]
    for a,b in zip([inputs,256,128],[256,128,128]):
        layer=nn.Linear(a,b)
        nn.init.orthogonal_(layer.weight,math.sqrt(2)); nn.init.zeros_(layer.bias)
        layers.extend([layer,nn.ELU()])
    last=nn.Linear(128,outputs)
    nn.init.orthogonal_(last.weight,last_gain); nn.init.zeros_(last.bias)
    return nn.Sequential(*layers,last)

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor=mlp(ACTOR_DIM,12,.01)
        self.register_buffer('stand',torch.tensor(CONTROL['stand']),persistent=False)
        self.critic=mlp(CRITIC_DIM,1)
        self.log_std=nn.Parameter(torch.full((12,),math.log(.3)))
    def symmetric_mean(self,obs):
        return .5*(self.actor(obs)+mirror_action(self.actor(mirror_observation(obs,self.stand,True))))
    def distribution(self,obs):
        std_log=.5*(self.log_std+self.log_std.reshape(4,3)[[1,0,3,2]].flatten())
        return torch.distributions.Normal(self.symmetric_mean(obs),std_log.exp())
    def value(self,obs):return self.critic(obs).squeeze(-1)
    def forward(self,obs):return torch.tanh(self.symmetric_mean(obs))

def advantages(rewards,dones,values,last,gamma,lam):
    carry=torch.zeros_like(last)
    adv=torch.zeros_like(rewards)
    for t in reversed(range(len(rewards))):
        following=last if t==len(rewards)-1 else values[t+1]
        alive=(~dones[t]).float()
        delta=rewards[t]+gamma*following*alive-values[t]
        carry=delta+gamma*lam*alive*carry
        adv[t]=carry
    return adv,adv+values

def checkpoint(folder,model,optimizer,iteration,manifest):
    path=Path(folder)/f'policy_{iteration}.pt'
    tmp=path.with_suffix('.tmp')
    torch.save(dict(kind='blind_velocity',model=model.state_dict(),optimizer=optimizer.state_dict(),
        iteration=iteration,manifest=manifest,ppo=PPO,torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        curriculum=getattr(model,'curriculum',{'stage':1,'streak':0}),
        backward_course=getattr(model,'backward_course',dict(cap=.5)),
        qe_curriculum=getattr(model,'qe_curriculum',dict(stage=1,streak=0))),tmp)
    tmp.replace(path)
    (Path(folder)/'latest.txt').write_text(str(path)+'\n')
    return path

def train(env,app,folder,manifest,iterations,resume=None):
    model=Policy().to(env.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=PPO['lr'])
    first=0
    model.curriculum=dict(stage=1,streak=0)
    if resume:
        if resume['manifest']!=manifest:raise ValueError('Resume contract/source mismatch; use matching snapshot')
        model.load_state_dict(resume['model']);optimizer.load_state_dict(resume['optimizer'])
        first=resume['iteration']+1
        model.curriculum=resume.get('curriculum',dict(stage=1,streak=0))
        env.stage=model.curriculum['stage']
        torch.set_rng_state(resume['torch_rng'].cpu())
        if resume.get('cuda_rng') and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([x.cpu() for x in resume['cuda_rng']])
    env.backward_limit=1.5
    model.backward_course=dict(cap=1.5,adaptive=False)
    from task_schedule import TrainingSchedule as MixedSchedule
    env.stage=3
    model.curriculum=dict(stage=3,streak=0)
    model.qe_curriculum=dict(resume.get('qe_curriculum',dict(stage=1,streak=0))) if resume else dict(stage=1,streak=0)
    training_phase=None
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        teacher=Policy().to(env.device).eval()
        teacher.load_state_dict(resume['model']);teacher.requires_grad_(False)
    env.qe_speed_cap=.8
    obs,_=env.reset()
    T,N=PPO['steps'],env.num_envs
    iteration=first-1
    with (Path(folder)/'progress.jsonl').open('a',encoding='utf-8') as log:
        try:
            for iteration in range(first,first+iterations):
                requested_phase='front_width_comprehensive500'
                if requested_phase!=training_phase:
                    env.schedule=MixedSchedule(env.num_envs,env.device,56789,model.qe_curriculum['stage'])
                    training_phase=requested_phase
                    obs,_=env.reset()
                    counts=[int(env.schedule.qe_mask.sum()),int((~env.schedule.qe_mask).sum())]
                    print('[TRAINING_PHASE] '+json.dumps(dict(phase=training_phase,iteration=iteration,counts=counts,names=['qe_curriculum','original_translation'])),flush=True)
                    (Path(folder)/'phase.json').write_text(json.dumps(dict(phase=training_phase,iteration=iteration,counts=counts)))
                env.symmetry_progress=1.
                env.stance_progress=1.  # Preserve the already-completed U reward ramp.
                env.diagonal_progress=1.
                env.gait_progress=1.  # Parent completed the 75-update reward ramp.
                env.qe_speed_cap=1.6
                env.schedule.turn_progress=1.
                start=time.perf_counter()
                data={k:[] for k in ('actor','critic','raw','logp','value','reward','done')}
                episodes=[]
                direction_sums={name:[0.,0.,0.] for name in ('forward','backward','left','right')}
                for _ in range(T):
                    if not app.is_running():return
                    x,c=obs['policy'],obs['critic']
                    with torch.no_grad():
                        dist=model.distribution(x)
                        noise=torch.randn(env.num_envs//2,12,device=env.device)
                        paired_noise=torch.stack((noise,mirror_action(noise)),dim=1).flatten(0,1)
                        raw=dist.loc+dist.scale*paired_noise
                        value=model.value(c);lp=dist.log_prob(raw).sum(-1)
                        nxt,reward,term,trunc,info=env.step(raw.tanh())
                        reward=reward.clone()
                        timeout=trunc & ~term
                        # Must use terminal state before automatic reset, not next episode's initial state.
                        reward[timeout]+=PPO['gamma']*model.value(env.terminal_critic[timeout])
                    for k,v in zip(data,(x,c,raw,lp,value,reward,term|trunc)):
                        data[k].append(v.detach().clone())
                    if info.get('episode'):episodes.append(info['episode'].copy())
                    for name,totals in direction_sums.items():
                        for j,key in enumerate(('count','error_sq_sum','failed_count')):
                            totals[j]+=float(info['log'][f'direction/{name}/{key}'])
                    obs=nxt
                data={k:torch.stack(v) for k,v in data.items()}
                with torch.no_grad():
                    adv,returns=advantages(data['reward'],data['done'],data['value'],
                        model.value(obs['critic']),PPO['gamma'],PPO['lam'])
                flat={k:v.flatten(0,1) for k,v in data.items()}
                adv=(adv.flatten()-adv.mean())/(adv.std()+1e-8);returns=returns.flatten()
                losses=[]
                for _ in range(PPO['epochs']):
                    for idx in torch.randperm(T*N,device=env.device).chunk(PPO['minibatches']):
                        dist=model.distribution(flat['actor'][idx])
                        logratio=dist.log_prob(flat['raw'][idx]).sum(-1)-flat['logp'][idx]
                        ratio=logratio.exp()
                        policy=-torch.minimum(ratio*adv[idx],ratio.clamp(1-PPO['clip'],1+PPO['clip'])*adv[idx]).mean()
                        value=model.value(flat['critic'][idx])
                        clipped=flat['value'][idx]+(value-flat['value'][idx]).clamp(-PPO['clip'],PPO['clip'])
                        vloss=.5*torch.maximum((value-returns[idx]).square(),(clipped-returns[idx]).square()).mean()
                        entropy=dist.entropy().sum(-1).mean() # Latent Gaussian entropy, not exact tanh entropy.
                        # Protect existing translation behavior only when clearly moving
                        # without yaw. No simulator-only inputs are added to the actor.
                        newest=flat['actor'][idx].reshape(-1,10,57)[:,-1]
                        keep=(newest[:,50:52].norm(dim=-1)>.2)&(newest[:,50]<.05)&(newest[:,52].abs()<1e-6)
                        with torch.no_grad():old_action=teacher(flat['actor'][idx])
                        prior=((dist.loc.tanh()-old_action).square().mean(-1)*keep).sum()/keep.sum().clamp(min=1)
                        loss=policy+vloss-PPO['entropy']*entropy
                        # Actor-only consistency; PPO ratios remain on real sampled transitions.
                        mirrored=mirror_observation(flat['actor'][idx],env.stand,True)
                        symmetry=(model.actor(mirrored).tanh()-mirror_action(model.actor(flat['actor'][idx]).tanh()).detach()).square().mean()
                        loss=loss+PPO['symmetry_weight']*symmetry
                        if not torch.isfinite(loss):raise RuntimeError('Non-finite PPO loss')
                        optimizer.zero_grad(set_to_none=True);loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),PPO['grad_norm']);optimizer.step()
                        with torch.no_grad():model.log_std.clamp_(math.log(.06),math.log(PPO['max_std']))
                        losses.append([float(policy.detach()),float(vloss.detach()),float(entropy.detach()),
                                       float(((ratio-1)-logratio).mean().detach())])
                row=dict(iteration=iteration,seconds=round(time.perf_counter()-start,2),
                    fps=round(T*N/(time.perf_counter()-start)),action_std=float(model.log_std.exp().mean().detach()),
                    symmetry_loss=float(symmetry.detach()),
                    loss=torch.tensor(losses).mean(0).tolist(),
                    **{k:float(v) for k,v in info['log'].items()})
                for name,(count,error,failed) in direction_sums.items():
                    row[f'direction/{name}/samples']=count
                    row[f'direction/{name}/rollout_rmse']=math.sqrt(error/max(count,1.))
                    row[f'direction/{name}/rollout_failure_fraction']=failed/max(count,1.)
                row['training_phase']=training_phase
                model.backward_course=dict(cap=1.5,adaptive=False)
                total=sum(e['count'] for e in episodes)
                if total:
                    row['completed_episodes']=total
                    for k in ('return_mean','duration_s','failure_fraction'):
                        row['episode/'+k]=sum(e[k]*e['count'] for e in episodes)/total
                print(f"phase={training_phase} iter={iteration+1} fps={row['fps']} std={row['action_std']:.3f} "
                    f"rmse=({row['track/rmse_x']:.3f},{row['track/rmse_y']:.3f},"
                    f"{row['track/rmse_yaw']:.3f}) back_cap={env.backward_limit:.1f} back_rmse={row['backward/rmse_x']:.3f} low={row['backward/low_height_fraction']:.3f} fold={row['backward/front_fold_fraction']:.3f} episode_s={row.get('episode/duration_s',0):.2f}",flush=True)
                print('[DIRECTIONS] '+ ' '.join(f"{name}:rmse={row[f'direction/{name}/rollout_rmse']:.3f},fail={row[f'direction/{name}/rollout_failure_fraction']:.3f},samples={int(row[f'direction/{name}/samples'])}" for name in direction_sums),flush=True)
                row['qe_curriculum']=dict(model.qe_curriculum)
                row['qe_speed_cap']=env.qe_speed_cap
                print(f"[QE] stage={model.qe_curriculum['stage']} rmse_xy={row['task/qe/rmse_xy']:.3f} rmse_yaw={row['task/qe/rmse_yaw']:.3f} translation_rmse={row['task/translation/rmse_xy']:.3f}",flush=True)
                log.write(json.dumps(row)+'\n');log.flush()
                if (iteration-first+1)%100==0:
                    from qe_gate import evaluate_qe
                    eligible_qe=evaluate_qe(env,model,Path(folder),iteration)
                    if model.qe_curriculum['stage']==1:
                        model.qe_curriculum['streak']=model.qe_curriculum['streak']+1 if eligible_qe else 0
                        if model.qe_curriculum['streak']>=2:
                            model.qe_curriculum=dict(stage=2,streak=0)
                            env.schedule.qe_stage=2
                            print('[QE_PROMOTION] translating and turning enabled',flush=True)
                    obs,_=env.reset()
                    checkpoint(folder,model,optimizer,iteration,manifest)
                if (iteration-first+1)%100==0:
                    from gate import evaluate
                    eligible=evaluate(env,model,Path(folder),iteration,range(1,env.stage+1))
                    # Keep stage 2 fixed for all 300 updates; gates remain held-out regression tests.
                    obs,_=env.reset()
                    checkpoint(folder,model,optimizer,iteration,manifest)
                if (iteration+1)%25==0:checkpoint(folder,model,optimizer,iteration,manifest)
                row_stage=dict(iteration=iteration,**model.curriculum)
                (Path(folder)/'curriculum.json').write_text(json.dumps(row_stage))
        finally:
            if iteration>=first:checkpoint(folder,model,optimizer,iteration,manifest)
    for period in ((.23,.28,.33) if iterations>=100 else ()):
        env.evaluation_period=period
        sub=Path(folder)/('period_'+str(period));sub.mkdir(exist_ok=True)
        from gate import evaluate
        from qe_gate import evaluate_qe
        evaluate(env,model,sub,iteration,[1,2,3])
        evaluate_qe(env,model,sub,iteration)
    return model
