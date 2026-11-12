import torch,json,math
from qe_schedule import MixedSchedule

@torch.no_grad()
def evaluate_qe(env,model,folder,iteration):
    old=(env.schedule,env.cfg.training,env.stage)
    eligible=False
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(810000+iteration)
            env.cfg.training=False;env.stage=3
            for stage in (1,2):
                env.schedule=MixedSchedule(env.num_envs,env.device,820000+iteration+stage,stage)
                obs,_=env.reset();n=env.num_envs;dev=env.device
                zero=lambda:torch.zeros(n,device=dev)
                alive=torch.ones(n,dtype=torch.bool,device=dev)
                ex,ew,drift,heading,tilt,height,stop,stopw=[zero() for _ in range(8)]
                count=stop_count=0
                for k in range(550):
                    obs,_,term,trunc,_=env.step(model(obs['policy']));alive&=~(term|trunc)
                    p=env._physical;c=env.command.velocity
                    if 50<=k<450:
                        measured=p['com_velocity'] if stage==1 else p['velocity']
                        ex+=(measured-c[:,:2]).square().sum(-1)
                        ew+=(p['yaw_rate']-c[:,2]).square();count+=1
                    if k>=25:
                        drift=torch.maximum(drift,(p['com_position']-env.rotation_anchor).norm(dim=-1))
                        heading=torch.maximum(heading,p['heading_error'].abs())
                        tilt=torch.maximum(tilt,torch.acos((-env.robot.data.projected_gravity_b.torch[:,2]).clamp(-1,1)))
                        height=torch.maximum(height,(p['height']-.395).abs())
                    if k>=525:
                        stop+=p['com_velocity'].norm(dim=-1);stopw+=p['yaw_rate'].abs();stop_count+=1
                rx=(ex/count).sqrt();rw=(ew/count).sqrt();stop/=stop_count;stopw/=stop_count
                good=alive&(rx<(.12 if stage==1 else .30))&(rw<.25)&(heading<math.radians(10))&(tilt<math.radians(10))&(height<.08)&(stop<.08)&(stopw<.12)
                if stage==1:good&=drift<.12
                mask=env.schedule.qe_mask;sign=env.schedule.yaw_sign
                rates=[float(good[mask&(sign==s)].float().mean()) for s in (1,-1)]
                row=dict(iteration=iteration,qe_stage=stage,qe_trials=int(mask.sum()),q_rate=rates[0],e_rate=rates[1],rate=float(good[mask].float().mean()),
                    survived=int(alive[mask].sum()),rmse_xy=float(rx[mask].mean()),rmse_yaw=float(rw[mask].mean()),
                    com_drift_m=float(drift[mask].mean()) if stage==1 else None,stop_speed=float(stop[mask].mean()),stop_yaw=float(stopw[mask].mean()),
                    eligible=min(rates)>=.90)
                with (folder/'qe_gates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print('[QE_GATE] '+json.dumps(row),flush=True)
                if stage==1:eligible=row['eligible']
    finally:env.schedule,env.cfg.training,env.stage=old
    return eligible
