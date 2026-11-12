"""Held-out full-speed turning/braking assessment, independent of training curriculum."""
import torch,json,math

class EvaluationSchedule:
    def __init__(self,n,device):
        self.n=n;self.device=device;ids=torch.arange(n,device=device)
        self.group=ids%8;self.pulsed=(ids//8)%2==1
    def reset(self,ids):pass
    def tick(self,stage,age,dt):
        a=torch.zeros(self.n,3,device=self.device);g=self.group
        active=(age>=.5)&(age<9.)
        for k in (0,1,4,5,6,7):
            mask=(g==k)&active;a[mask,0]=1. if k in (0,4,5) else -1.
        for k in range(2,8):
            mask=(g==k)&active
            if k>=4:mask&=age>=1.8
            sign=1. if k in (2,4,6) else -1.
            # Held-out release/reversal interval after reaching the initial turn.
            value=torch.where(self.pulsed & (age>=4.) & (age<5.1),0.,sign)
            value=torch.where(self.pulsed & (age>=6.2),-sign,value)
            a[mask,2]=value[mask]
        return a

@torch.no_grad()
def evaluate_turns(env,model,folder,iteration):
    old=(env.schedule,env.cfg.training)
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(930000+iteration);env.cfg.training=False
            env.schedule=EvaluationSchedule(env.num_envs,env.device)
            obs,_=env.reset();n=env.num_envs;dev=env.device
            zero=lambda:torch.zeros(n,device=dev)
            alive=torch.ones(n,dtype=torch.bool,device=dev)
            ex,ew,ct,stop,stopw,roll,pitch,drift,refw,actualw=[zero() for _ in range(10)]
            count=stopn=0
            for k in range(550):
                obs,_,term,trunc,_=env.step(model(obs['policy']));alive&=~(term|trunc)
                p=env._physical;c=env.command.velocity
                if 50<=k<450:
                    ex+=(p['velocity']-c[:,:2]).square().sum(-1)
                    ew+=(p['yaw_rate']-c[:,2]).square();ct+=p['contact'].all(-1).float()
                    refw+=c[:,2].abs();actualw+=p['yaw_rate'].abs();count+=1
                if k>=25:
                    g=env.robot.data.projected_gravity_b.torch
                    roll=torch.maximum(roll,torch.atan2(g[:,1],-g[:,2]).abs())
                    pitch=torch.maximum(pitch,torch.atan2(g[:,0],-g[:,2]).abs())
                    drift=torch.maximum(drift,(p['com_position']-env.rotation_anchor).norm(dim=-1))
                if k>=500:
                    stop+=p['velocity'].norm(dim=-1);stopw+=p['yaw_rate'].abs();stopn+=1
            rx=(ex/count).sqrt();rw=(ew/count).sqrt();ct/=count;stop/=stopn;stopw/=stopn
            pure=(env.schedule.group==2)|(env.schedule.group==3)
            good=alive&(rx<torch.where(pure,.12,.30))&(rw<.25)&(ct>.90)&(stop<.10)&(stopw<.12)
            good&=(roll<math.radians(10))&(pitch<math.radians(25))
            good&=(~pure)|(drift<.15)
            names=['forward_brake','backward_brake','pure_Q','pure_E','forward_Q','forward_E','backward_Q','backward_E']
            groups=[]
            for i,name in enumerate(names):
                mask=env.schedule.group==i
                groups.append(dict(task=name,rate=float(good[mask].float().mean()),survived=int(alive[mask].sum()),
                    trials=int(mask.sum()),rmse_xy=float(rx[mask].mean()),rmse_yaw=float(rw[mask].mean()),
                    mean_abs_reference_yaw=float(refw[mask].mean()/count),mean_abs_actual_yaw=float(actualw[mask].mean()/count),
                    roll_peak_deg=float(roll[mask].mean())*180/math.pi,pitch_peak_deg=float(pitch[mask].mean())*180/math.pi,
                    four_contact=float(ct[mask].mean()),stop_speed=float(stop[mask].mean()),stop_yaw=float(stopw[mask].mean()),
                    com_drift_m=float(drift[mask].mean()) if i in (2,3) else None))
            row=dict(iteration=iteration,rate=float(good.float().mean()),groups=groups,
                     note='Full command envelope; any reset invalidates survival; 10deg roll/25deg pitch limits unchanged.')
            with (folder/'turn_gates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print('[TURN_GATE] '+json.dumps(row),flush=True)
            return min(x['rate'] for x in groups)>=.9
    finally:env.schedule,env.cfg.training=old
