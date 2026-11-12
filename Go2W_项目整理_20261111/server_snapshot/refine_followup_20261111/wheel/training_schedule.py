"""Straight balance, unpredictable turns, long holds, release and reversal."""
import torch
class TrainingSchedule:
    def __init__(self,n,device,seed):
        self.device=device;self.group=torch.arange(n,device=device)%16
        self.combo_mask=self.group>=8;self.low_speed_mask=(self.group>=8)&(self.group<12)
        self.timer=torch.zeros(n,device=device);self.yaw=torch.zeros(n,device=device)
        self.gen=torch.Generator(device=device).manual_seed(seed)
    def reset(self,ids):self.timer[ids]=0.;self.yaw[ids]=0.
    def tick(self,stage,age,dt):
        self.timer-=dt;due=self.timer<=0;n=int(due.sum())
        self.timer[due]=.4+2.*torch.rand(n,device=self.device,generator=self.gen)
        self.yaw[due]=torch.randint(-1,2,(n,),device=self.device,generator=self.gen).float()
        a=torch.zeros(len(age),3,device=self.device);g=self.group
        sign=torch.where(g%2==0,1.,-1.)
        phase=(age-.5).clamp(min=0).remainder(6.)
        a[g<4,0]=sign[g<4]
        a[(g>=2)&(g<4),2]=self.yaw[(g>=2)&(g<4)]*(age[(g>=2)&(g<4)]>1.5)
        pure=(g>=4)&(g<8)
        a[pure,2]=sign[pure]*torch.where(phase[pure]<3.6,1.,torch.where(phase[pure]<5.,0.,-1.))
        combo=g>=8;a[combo,0]=torch.where(g[combo]%4<2,1.,-1.)
        a[combo,2]=self.yaw[combo]*(age[combo]>1.5)
        # Dedicated abrupt releases with dwell to learn both braking and recovery.
        stop=((g%4>=2)&(age>=4.5)&(age<6.5))|(age>=9.)|(age<.5)
        a[stop]=0.
        return a
