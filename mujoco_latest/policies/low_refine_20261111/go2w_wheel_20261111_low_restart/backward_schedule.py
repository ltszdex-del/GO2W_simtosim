"""Backward-only fine tuning: 50% cruise, 25% stop, 25% random release."""
import torch
from schedule import Schedule

class BackwardSchedule(Schedule):
    def __init__(self,n,device,seed):
        super().__init__(n,device,seed)
        slot=torch.arange(n,device=device)%4
        self.group=torch.where(slot<2,0,slot-1)
        self.period=torch.full((n,),4.,device=device)
    def reset(self,ids):
        super().reset(ids)
        self.period[ids]=3.5+2.*self.rand((len(ids),))
    def tick(self,stage,age,dt):
        self.timer-=dt
        due=self.timer[:,0]<=0
        n=int(due.sum())
        self.value[due,0]=-(self.rand((n,))<.75).float()
        self.timer[due,0]=.4+2.1*self.rand((n,))
        axes=torch.zeros(self.n,3,device=self.device)
        phase=((age-.5).clamp(min=0)/self.period).remainder(1.)
        axes[:,0]=torch.where(self.group==0,-1.,torch.where(self.group==1,-(phase<.65).float(),self.value[:,0]))
        axes[(age<.5)|(age>=9.)]=0.
        return axes
