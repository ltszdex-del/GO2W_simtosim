"""Mirrored adjacent pairs, comprehensive commands, rest/brake/reversal."""
import torch

class TrainingSchedule:
    def __init__(self,n,device,seed,qe_stage=1):
        assert n%2==0
        self.n=n;self.device=device;self.qe_stage=qe_stage;self.turn_progress=1.
        lookup=torch.tensor([0,1,2,3,4,5,6,7,8,9]*4,device=device)
        self.group=lookup[(torch.arange(n,device=device)//2)%40]
        self.qe_mask=(self.group==5)|(self.group==6)
    def reset(self,ids):pass
    def tick(self,stage,age,dt):
        g=self.group[0::2];t=age[0::2];a=torch.zeros(len(g),3,device=self.device)
        cycle=1.6+.8*((torch.arange(len(g),device=self.device))%4)
        phase=(t-.5).clamp(min=0).remainder(cycle)*4./cycle
        pulse=(phase<2.8).float();reverse=torch.where(phase<2.,1.,-1.)
        for k in range(10):
            m=g==k
            if k==0:a[m,0]=pulse[m]
            elif k==1:a[m,0]=-pulse[m]
            elif k==2:a[m,1]=pulse[m]*reverse[m]
            elif k==3:a[m,0]=pulse[m];a[m,1]=pulse[m]
            elif k==4:a[m,0]=-pulse[m];a[m,1]=pulse[m]
            elif k==5:a[m,2]=1.
            elif k==6:a[m,2]=pulse[m]*reverse[m]
            elif k==7:a[m,0]=pulse[m];a[m,2]=pulse[m]*reverse[m]
            elif k==8:a[m,0]=-pulse[m];a[m,2]=pulse[m]*reverse[m]
            else:a[m,0]=pulse[m]*reverse[m];a[m,1]=pulse[m];a[m,2]=pulse[m]*reverse[m]
        a[(t<.5)|(t>=9.)]=0.
        result=a.repeat_interleave(2,dim=0);result[1::2,1:]*=-1
        return result
