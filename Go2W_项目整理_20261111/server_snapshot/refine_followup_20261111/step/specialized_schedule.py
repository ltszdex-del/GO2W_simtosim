"""20% dedicated backward; remaining 80% balanced eight command scenarios."""
import torch
from schedule import Schedule

GROUP_NAMES=('backward_special','forward','backward','left','right',
    'longitudinal_stop','lateral_stop','longitudinal_reverse','lateral_reverse')

def specialized_phase(iteration):return 4200<=iteration<4400

class SpecializedSchedule(Schedule):
    def __init__(self,n,device,seed):
        super().__init__(n,device,seed)
        ids=torch.arange(n,device=device);cut=round(n*.2)
        self.group=torch.where(ids<cut,0,1+(ids-cut)%8)
        self.period=torch.full((n,),4.,device=device)
    def reset(self,ids):
        super().reset(ids)
        self.period[ids]=3.5+2.*self.rand((len(ids),))
    def tick(self,stage,age,dt):
        axes=torch.zeros(self.n,3,device=self.device);g=self.group
        axes[(g==0)|(g==2),0]=-1.;axes[g==1,0]=1.
        axes[g==3,1]=1.;axes[g==4,1]=-1.
        phase=((age-.5).clamp(min=0)/self.period).remainder(1.)
        stop=self.sign*(phase<.65)
        reversal=self.sign*torch.where(phase<.5,1.,-1.)
        axes[g==5,0]=stop[g==5];axes[g==6,1]=stop[g==6]
        axes[g==7,0]=reversal[g==7];axes[g==8,1]=reversal[g==8]
        axes[(age<.5)|(age>=9.)]=0.
        return axes
