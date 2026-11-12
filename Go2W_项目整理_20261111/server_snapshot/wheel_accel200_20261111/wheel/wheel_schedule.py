import torch
from schedule import Schedule as Base
class Schedule(Base):
    def __init__(self,n,device,seed):
        super().__init__(n,device,seed);self.period=torch.full((n,),4.,device=device)
    def reset(self,ids):
        super().reset(ids);self.period[ids]=3.+2*self.rand((len(ids),))
    def tick(self,stage,age,dt):
        a=super().tick(3,age,dt);g=torch.arange(self.n,device=self.device)%8
        x=a[:,0].clone();w=a[:,1].clone();a.zero_()
        phase=((age-.5).clamp(min=0)/self.period).remainder(1.)
        a[g==0,0]=(phase[g==0]<.65).float();a[g==1,0]=-(phase[g==1]<.65).float()
        a[g==2,2]=1;a[g==3,2]=-1
        a[g==4,0]=1;a[g==4,2]=w[g==4]
        a[g==5,0]=-1;a[g==5,2]=w[g==5]
        a[g==6,0]=x[g==6];a[g==6,2]=w[g==6]
        a[g==7,0]=torch.where(phase[g==7]<.5,1.,-1.)
        a[(age<.5)|(age>=9.)]=0
        return a
