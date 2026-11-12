"""Held-out finite episodes; auto resets can never erase an earlier failure."""
import json
import math
import torch
from schedule import Schedule

@torch.no_grad()
def evaluate(env,model,folder,iteration,stages):
    old_schedule,old_training,old_stage=env.schedule,env.cfg.training,env.stage
    results=[]
    # Evaluation must not consume training policy/reset RNG.
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(900000+iteration)
            env.cfg.training=False
            for stage in stages:
                env.stage=stage
                env.schedule=Schedule(env.num_envs,env.device,700000+iteration*7+stage)
                obs,_=env.reset()
                n=env.num_envs;dev=env.device
                alive=torch.ones(n,dtype=torch.bool,device=dev)
                err=torch.zeros(n,device=dev);count=0
                heading=torch.zeros(n,device=dev);tilt=heading.clone();height=heading.clone()
                stop=heading.clone();stop_count=0
                # 11 seconds, below the 12 second episode timeout.
                for step in range(550):
                    obs,_,term,trunc,_=env.step(model(obs['policy']))
                    alive &= ~(term|trunc)
                    p=env._physical;d=env.robot.data
                    if 50<=step<450:
                        err+=(p['velocity']-env.command.velocity[:,:2]).square().sum(-1);count+=1
                    if step>=25:
                        heading=torch.maximum(heading,p['heading_error'].abs())
                        tilt=torch.maximum(tilt,torch.acos((-d.projected_gravity_b.torch[:,2]).clamp(-1,1)))
                        height=torch.maximum(height,(p['height']-.395).abs())
                    if step>=525:stop+=p['velocity'].norm(dim=-1);stop_count+=1
                rmse=(err/count).sqrt();stop/=stop_count
                passed=alive&(rmse<.25)&(heading<math.radians(5))&(tilt<math.radians(10))&(height<.08)&(stop<.08)
                groups=[float(passed[torch.arange(n,device=dev)%4==g].float().mean()) for g in range(4)]
                row=dict(iteration=iteration,stage=stage,trials=n,passed=int(passed.sum()),
                    rate=float(passed.float().mean()),direction_group_rates=groups,
                    rmse=float(rmse.mean()),stop_speed=float(stop.mean()),
                    heading_peak_deg=float(heading.mean()*180/math.pi),survived=int(alive.sum()))
                row['eligible']=row['rate']>=.90  # Overall rate; direction rates remain diagnostic
                results.append(row)
                with (folder/'gates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print('[GATE] '+json.dumps(row),flush=True)
    finally:
        env.schedule=old_schedule;env.cfg.training=old_training;env.stage=old_stage
    return all(r['eligible'] for r in results)
