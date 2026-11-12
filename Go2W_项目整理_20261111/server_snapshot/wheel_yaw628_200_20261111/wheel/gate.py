import torch,json,math
from wheel_schedule import Schedule
@torch.no_grad()
def evaluate(env,model,folder,iteration,stages):
    old=(env.schedule,env.cfg.training,env.stage)
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(310000+iteration);env.cfg.training=False
            env.schedule=Schedule(env.num_envs,env.device,400000+iteration);obs,_=env.reset()
            n=env.num_envs;z=lambda:torch.zeros(n,device=env.device)
            alive=torch.ones(n,dtype=torch.bool,device=env.device);ex,ew,ct,stop,stopw,tilt=[z() for _ in range(6)];count=stopn=0
            roll=z();pitch=z();stoptilt=z()
            for k in range(550):
                obs,_,term,trunc,_=env.step(model(obs['policy']));alive&=~(term|trunc)
                p=env._physical;c=env.command.velocity
                if 50<=k<450:
                    ex+=(p['velocity']-c[:,:2]).square().sum(-1);ew+=(p['yaw_rate']-c[:,2]).square();ct+=p['contact'].all(-1).float();count+=1
                if k>=25:
                    g=env.robot.data.projected_gravity_b.torch
                    roll=torch.maximum(roll,torch.atan2(g[:,1],-g[:,2]).abs())
                    pitch=torch.maximum(pitch,torch.atan2(g[:,0],-g[:,2]).abs())
                if k>=25:tilt=torch.maximum(tilt,torch.acos((-env.robot.data.projected_gravity_b.torch[:,2]).clamp(-1,1)))
                if k>=525:
                    stop+=p['velocity'].norm(dim=-1);stopw+=p['yaw_rate'].abs();stopn+=1
                    stoptilt=torch.maximum(stoptilt,torch.acos((-env.robot.data.projected_gravity_b.torch[:,2]).clamp(-1,1)))
            ex=(ex/count).sqrt();ew=(ew/count).sqrt();ct/=count;stop/=stopn;stopw/=stopn
            base=alive&(ex<.30)&(ew<.25)&(ct>.90)&(stop<.10)&(stopw<.12)
            strict=base&(tilt<math.radians(10))
            good=base&(roll<math.radians(10))&(pitch<math.radians(25))&(stoptilt<math.radians(10))
            group=torch.arange(n,device=env.device)%8
            row=dict(iteration=iteration,strict_old_tilt_rate=float(strict.float().mean()),pitch_peak_deg=float(pitch.mean())*180/math.pi,roll_peak_deg=float(roll.mean())*180/math.pi,rate=float(good.float().mean()),survived=int(alive.sum()),rmse_xy=float(ex.mean()),rmse_yaw=float(ew.mean()),four_contact=float(ct.mean()),stop_speed=float(stop.mean()),stop_yaw=float(stopw.mean()),group_rates=[float(good[group==g].float().mean()) for g in range(8)])
            with (folder/'gates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print('[WHEEL_GATE] '+json.dumps(row),flush=True);return row['rate']>=.9
    finally:env.schedule,env.cfg.training,env.stage=old
