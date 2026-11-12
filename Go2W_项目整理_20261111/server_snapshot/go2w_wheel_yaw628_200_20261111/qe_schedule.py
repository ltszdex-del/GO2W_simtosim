import torch
from schedule import Schedule

class MixedSchedule(Schedule):
    """Fixed 50/50 task assignment; independent random translation and yaw."""
    def __init__(self,n,device,seed,qe_stage=1):
        super().__init__(n,device,seed)
        ids=torch.arange(n,device=device)
        self.qe_mask=ids<n//2;self.qe_stage=qe_stage
        self.yaw_sign=torch.where(ids%2==0,1.,-1.)
        self.yaw_timer=torch.zeros(n,device=device);self.yaw_value=torch.zeros(n,device=device)
        self.pulsed=ids%4>=2
    def reset(self,ids):
        super().reset(ids);self.yaw_timer[ids]=0.;self.yaw_value[ids]=0.
    def tick(self,stage,age,dt):
        axes=super().tick(3,age,dt)
        self.yaw_timer-=dt;due=self.yaw_timer<=0;n=int(due.sum())
        self.yaw_timer[due]=1.+1.5*self.rand((n,))
        if self.qe_stage==1:
            self.yaw_value[due]=(self.rand((n,))>.25).float()*self.yaw_sign[due]
            yaw=torch.where(self.pulsed,self.yaw_value,self.yaw_sign)
            axes[self.qe_mask,:2]=0.
        else:
            self.yaw_value[due]=torch.floor(self.rand((n,))*3)-1.
            yaw=self.yaw_value
        axes[self.qe_mask,2]=yaw[self.qe_mask]
        axes[(age<.5)|(age>=9.)]=0.
        return axes
