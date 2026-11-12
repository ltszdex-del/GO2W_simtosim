"""Asymmetric PPO with exact latent likelihood ratios for tanh-bounded actions."""
import copy
import json
import math
import time
from pathlib import Path
import torch
from torch import nn
from symmetry import mirror_action, mirror_observation
from config import ACTOR_DIM, CRITIC_DIM, PPO, GAIT, CONTROL

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
        self.actor=mlp(ACTOR_DIM,16,.01)
        self.register_buffer('stand',torch.tensor(CONTROL['stand']),persistent=False)
        self.critic=mlp(CRITIC_DIM,1)
        self.log_std=nn.Parameter(torch.full((16,),math.log(.3)))
    def symmetric_mean(self,obs):
        return .5*(self.actor(obs)+mirror_action(self.actor(mirror_observation(obs,self.stand,False))))
    def distribution(self,obs):
        paired=self.log_std.reshape(-1)
        order=[3,4,5,0,1,2,9,10,11,6,7,8,13,12,15,14]
        return torch.distributions.Normal(self.symmetric_mean(obs),(.5*(paired+paired[order])).exp())
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
        curriculum=getattr(model,'curriculum',{'stage':1,'streak':0})),tmp)
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
    from training_schedule import TrainingSchedule as Schedule
    env.schedule=Schedule(env.num_envs,env.device,78221)
    env.drive_speed_cap=3.
    parent_scale=env.scale.clone()
    full_scale=env.scale.clone();env.scale.copy_(parent_scale)
    optimizer.param_groups[0]['lr']=PPO['lr']
    with torch.no_grad():model.log_std.clamp_(max=math.log(PPO['max_std']))
    straight_teacher=None
    assert resume is not None, "Refinement requires the trained parent"
    obs,_=env.reset()
    T,N=PPO['steps'],env.num_envs
    iteration=first-1
    with (Path(folder)/'progress.jsonl').open('a',encoding='utf-8') as log:
        try:
            for iteration in range(first,first+iterations):
                progress=min(1.,(iteration-first)/60.)
                env.scale.copy_(parent_scale+(full_scale-parent_scale)*progress)
                env.drive_speed_cap=3.
                env.turn_cap=4.
                env.turn_course_progress=1.
                start=time.perf_counter()
                data={k:[] for k in ('actor','critic','raw','logp','value','reward','done')}
                episodes=[]
                for _ in range(T):
                    if not app.is_running():return
                    x,c=obs['policy'],obs['critic']
                    with torch.no_grad():
                        dist=model.distribution(x)
                        raw=dist.sample();value=model.value(c);lp=dist.log_prob(raw).sum(-1)
                        nxt,reward,term,trunc,info=env.step(raw.tanh())
                        reward=reward.clone()
                        timeout=trunc & ~term
                        # Must use terminal state before automatic reset, not next episode's initial state.
                        reward[timeout]+=PPO['gamma']*model.value(env.terminal_critic[timeout])
                    for k,v in zip(data,(x,c,raw,lp,value,reward,term|trunc)):
                        data[k].append(v.detach().clone())
                    if info.get('episode'):episodes.append(info['episode'].copy())
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
                        loss=policy+vloss-PPO['entropy']*entropy
                        # Actor-only consistency; PPO ratios remain on real sampled transitions.
                        mirrored=mirror_observation(flat['actor'][idx],env.stand,False)
                        symmetry=(model.actor(mirrored).tanh()-mirror_action(model.actor(flat['actor'][idx]).tanh()).detach()).square().mean()
                        loss=loss+PPO['symmetry_weight']*symmetry
                        if not torch.isfinite(loss):raise RuntimeError('Non-finite PPO loss')
                        optimizer.zero_grad(set_to_none=True);loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),PPO['grad_norm']);optimizer.step()
                        with torch.no_grad():model.log_std.clamp_(math.log(.06),math.log(PPO['max_std']))
                        losses.append([float(policy.detach()),float(vloss.detach()),float(entropy.detach()),
                                       float(((ratio-1)-logratio).mean().detach())])
                row=dict(yaw_limit_mean=float(env.command.current_yaw_limit.mean()),yaw_limit_min=float(env.command.current_yaw_limit.min()),yaw_limit_max=float(env.command.current_yaw_limit.max()),iteration=iteration,seconds=round(time.perf_counter()-start,2),
                    drive_speed_cap=env.drive_speed_cap,yaw_course_cap=4.,leg_scale_progress=progress,turn_course_progress=env.turn_course_progress,
                    fps=round(T*N/(time.perf_counter()-start)),action_std=float(model.log_std.exp().mean().detach()),
                    symmetry_loss=float(symmetry.detach()),
                    loss=torch.tensor(losses).mean(0).tolist(),
                    **{k:float(v) for k,v in info['log'].items()})
                total=sum(e['count'] for e in episodes)
                if total:
                    row['completed_episodes']=total
                    for k in ('return_mean','duration_s','failure_fraction'):
                        row['episode/'+k]=sum(e[k]*e['count'] for e in episodes)/total
                print(f"iter={iteration+1} fps={row['fps']} std={row['action_std']:.3f} "
                    f"rmse=({row['track/rmse_x']:.3f},{row['track/rmse_y']:.3f},"
                    f"{row['track/rmse_yaw']:.3f}) episode_s={row.get('episode/duration_s',0):.2f} "
                    f"turn_course={env.turn_course_progress:.2f} contact={row['wheel/four_contact']:.3f}",flush=True)
                log.write(json.dumps(row)+'\n');log.flush()
                if (iteration+1)%100==0:
                    from gate import evaluate
                    env.scale.copy_(full_scale)
                    eligible=evaluate(env,model,Path(folder),iteration,range(1,env.stage+1))
                    env.scale.copy_(parent_scale+(full_scale-parent_scale)*progress)
                    obs,_=env.reset()
                    checkpoint(folder,model,optimizer,iteration,manifest)
                if (iteration+1)%50==0:
                    from turn_gate import evaluate_turns
                    evaluate_turns(env,model,Path(folder),iteration)
                    obs,_=env.reset()
                    checkpoint(folder,model,optimizer,iteration,manifest)
                if (iteration+1)%25==0:checkpoint(folder,model,optimizer,iteration,manifest)
                row_stage=dict(iteration=iteration,**model.curriculum)
                (Path(folder)/'curriculum.json').write_text(json.dumps(row_stage))
        finally:
            if iteration>=first:checkpoint(folder,model,optimizer,iteration,manifest)
    return model

