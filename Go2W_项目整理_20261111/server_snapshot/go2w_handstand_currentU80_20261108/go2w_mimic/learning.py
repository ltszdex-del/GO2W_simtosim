"""可独立阅读的 PPO 与 DAgger，实现不依赖某个 rsl_rl 内部版本。

PPO 的采样是 Gaussian latent -> tanh -> 有界动作。
更新时在同一个 latent 上算概率比，tanh Jacobian 在新旧概率之比中抵消。
entropy 使用 latent Gaussian 的熵作探索正则，不声称是有界动作的精确熵。
"""
import json
import math
import time
from pathlib import Path
import torch
from .networks import Teacher,Student
from .settings import *


def advantages(rewards, dones, values, last_value, gamma, lam):
    """GAE。rewards 已包含 timeout 的终点价值补偿；dones 阻断跨回合递推。"""
    adv=torch.zeros_like(rewards)
    carry=torch.zeros_like(last_value)
    for t in reversed(range(len(rewards))):
        following=last_value if t==len(rewards)-1 else values[t+1]
        alive=1.-dones[t].float()
        delta=rewards[t]+gamma*following*alive-values[t]
        carry=delta+gamma*lam*alive*carry
        adv[t]=carry
    return adv,adv+values


def save(path, model, optimizer, kind, iteration, interface, curriculum=0.):
    torch.save(dict(kind=kind,model=model.state_dict(),optimizer=optimizer.state_dict(),
        iteration=iteration,interface=interface,curriculum=curriculum,
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),path)


def train_teacher(env, app, folder, interface, iterations, checkpoint=None):
    model=Teacher().to(env.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=PPO["learning_rate"])
    first=0
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        first=checkpoint["iteration"]+1
        env.domain_level=checkpoint.get("curriculum",env.domain_level)
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if checkpoint.get("cuda_rng") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint["cuda_rng"]])
    # 旧检查点的标准差约 0.43；续训第一步采样前就收紧探索上限。
    with torch.no_grad():
        model.log_std.clamp_(max=math.log(PPO["max_action_std"]))
    optimizer.param_groups[0]['lr']=1e-5
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        anchor=Teacher().to(env.device).eval()
        anchor.load_state_dict(checkpoint['model']);anchor.requires_grad_(False)
    obs,_=env.reset()
    T,N=PPO["steps"],env.num_envs
    log=open(Path(folder)/"progress.jsonl","a",encoding="utf-8")
    status=open(Path(folder)/"status.log","a",encoding="utf-8")
    iteration=first-1
    try:
        for iteration in range(first,first+iterations):
            start=time.perf_counter()
            rollout={k:[] for k in ("obs","raw","logp","value","reward","done")}
            episode_summaries=[]
            for _ in range(T):
                if not app.is_running(): return
                x=obs["policy"]
                with torch.no_grad():
                    dist=model.distribution(x)
                    raw=dist.sample()
                    value=model.value(x)
                    logp=dist.log_prob(raw).sum(-1)
                    next_obs,reward,term,trunc,info=env.step(torch.tanh(raw))
                    if "episode" in info:
                        episode_summaries.append(dict(info["episode"]))
                    # 正常超时允许价值延续；跌倒终止不 bootstrap。
                    timeouts=trunc & ~term
                    reward=reward.clone()
                    if timeouts.any():
                        reward[timeouts]+=PPO["gamma"]*model.value(
                            env.terminal_teacher_obs[timeouts])
                for k,v in zip(rollout,(x,raw,logp,value,reward,term|trunc)):
                    rollout[k].append(v.detach().clone())
                obs=next_obs
            data={k:torch.stack(v) for k,v in rollout.items()}
            with torch.no_grad():
                adv,returns=advantages(data["reward"],data["done"],data["value"],
                    model.value(obs["policy"]),PPO["gamma"],PPO["lam"])
            flat={k:v.flatten(0,1) for k,v in data.items()}
            adv=(adv.flatten()-adv.mean())/(adv.std()+1e-8)
            returns=returns.flatten()
            losses=[]
            for _ in range(PPO["epochs"]):
                for idx in torch.randperm(T*N,device=env.device).chunk(PPO["minibatches"]):
                    dist=model.distribution(flat["obs"][idx])
                    ratio=(dist.log_prob(flat["raw"][idx]).sum(-1)-flat["logp"][idx]).exp()
                    policy=-torch.minimum(ratio*adv[idx],
                        ratio.clamp(1-PPO["clip"],1+PPO["clip"])*adv[idx]).mean()
                    predicted=model.value(flat["obs"][idx])
                    clipped=flat["value"][idx]+(predicted-flat["value"][idx]).clamp(-PPO["clip"],PPO["clip"])
                    vloss=.5*torch.maximum((predicted-returns[idx]).square(),
                                           (clipped-returns[idx]).square()).mean()
                    entropy=dist.entropy().sum(-1).mean()
                    with torch.no_grad():old_dist=anchor.distribution(flat['obs'][idx])
                    anchor_kl=torch.distributions.kl_divergence(old_dist,dist).sum(-1).mean()
                    loss=policy+PPO["value_coef"]*vloss-PPO["entropy_coef"]*entropy+anchor_kl
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),PPO["max_grad_norm"])
                    optimizer.step()
                    with torch.no_grad():
                        model.log_std.clamp_(-2.3,math.log(PPO["max_action_std"]))
                    losses.append(torch.stack((policy.detach(),vloss.detach(),entropy.detach())))
            mean=torch.stack(losses).mean(0).cpu().tolist()
            completed=sum(e["completed_count"] for e in episode_summaries)
            episode_stats=({k:sum(e[k]*e["completed_count"] for e in episode_summaries)/completed
                for k in ("success","best_hold_s","reward","domain_level")} if completed else {})
            episode_stats["completed_count"]=completed
            row=dict(iteration=iteration,steps_per_second=round(T*N/(time.perf_counter()-start)),
                policy_loss=mean[0],value_loss=mean[1],entropy=mean[2],
                action_std=float(model.log_std.detach().exp().mean()),
                **episode_stats)
            row.update({k:float(v) for k,v in info.get("log",{}).items()})
            episode=row.get("success")
            progress=(f"iter={iteration+1} success={episode:.1%}" if episode is not None
                      else f"iter={iteration+1} success=--")
            progress+=(f" hold={row['best_hold_s']:.2f}s" if 'best_hold_s' in row else " hold=--")
            progress+=(f" return={row['reward']:.1f}" if 'reward' in row else " return=--")
            progress+=(f" lift={row.get('reward/lift',0.):.2f}"
                       f" height={row.get('reward/height',0.):.2f}"
                       f" std={row['action_std']:.3f}"
                       f" fps={row['steps_per_second']}")
            print(progress,flush=True)
            status.write(progress+"\n"); status.flush()
            log.write(json.dumps(row)+"\n"); log.flush()
            if (iteration+1)%100==0:
                save(Path(folder)/f"teacher_{iteration}.pt",model,optimizer,"teacher",iteration,
                     interface,env.domain_level)
    finally:
        save(Path(folder)/f"teacher_{iteration}.pt",model,optimizer,"teacher",iteration,
             interface,env.domain_level)
        status.close()
        log.close()


def distill(env, app, folder, interface, iterations, teacher_checkpoint):
    """DAgger：让学生亲自执行，在学生到达的状态询问教师的动作。

    只离线模仿教师成功轨迹会缺少“学生偏离后如何恢复”的样本；
    所以这里保留滚动回放缓冲，并逐步把执行动作全部交给学生。
    """
    teacher=Teacher().to(env.device).eval()
    teacher.load_state_dict(teacher_checkpoint["model"])
    teacher.requires_grad_(False)
    student=Student().to(env.device)
    optimizer=torch.optim.Adam(student.parameters(),lr=3e-4)
    capacity=max(8192,env.num_envs)
    inputs=torch.empty(capacity,STUDENT_DIM,device=env.device)
    targets=torch.empty(capacity,ACTION_DIM,device=env.device)
    cursor=filled=0
    obs,_=env.reset()
    log=open(Path(folder)/"progress.jsonl","a",encoding="utf-8")
    iteration=-1
    try:
        for iteration in range(iterations):
            start=time.perf_counter()
            beta=.5*max(0.,1.-iteration/max(1,iterations*.3))
            for _ in range(32):
                if not app.is_running(): return
                with torch.no_grad():
                    desired=teacher(obs["policy"])
                    action=student(obs["student"])
                    choose=torch.rand(env.num_envs,1,device=env.device)<beta
                    action=torch.where(choose,desired,action)
                    ids=(torch.arange(env.num_envs,device=env.device)+cursor)%capacity
                    inputs[ids],targets[ids]=obs["student"],desired
                    cursor=(cursor+env.num_envs)%capacity
                    filled=min(capacity,filled+env.num_envs)
                    obs,_,_,_,info=env.step(action)
            errors=[]
            for _ in range(32):
                ids=torch.randint(filled,(min(512,filled),),device=env.device)
                loss=(student(inputs[ids])-targets[ids]).square().mean()
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(),1.)
                optimizer.step()
                errors.append(loss.detach())
            row=dict(iteration=iteration,teacher_mix=beta,mse=float(torch.stack(errors).mean()),
                     seconds=round(time.perf_counter()-start,2),**info.get("episode",{}))
            print(json.dumps(row),flush=True); log.write(json.dumps(row)+"\n"); log.flush()
            if (iteration+1)%100==0:
                save(Path(folder)/f"student_{iteration}.pt",student,optimizer,"student",iteration,
                     interface,env.domain_level)
    finally:
        save(Path(folder)/f"student_{iteration}.pt",student,optimizer,"student",iteration,
             interface,env.domain_level)
        log.close()
