"""Acceleration reference and five-stage WASD curriculum. No applied forces."""
import math
import torch
from config import COMMAND
from directional_reference import planar_tensor
from yaw_envelope import yaw_limit_tensor,brake_to_limit_tensor

def wrap(x):return torch.atan2(torch.sin(x),torch.cos(x))

class Commands:
    def __init__(self,n,device):
        self.velocity=torch.zeros(n,3,device=device)
        self.acceleration=torch.zeros_like(self.velocity)
        self.heading=torch.zeros(n,device=device)
    def reset(self,ids,heading=0.):
        self.velocity[ids]=0;self.acceleration[ids]=0;self.heading[ids]=heading
    def tick(self,axes,dt):
        old=self.velocity.clone();u=axes[:,:2].clamp(-1,1)
        self.velocity[:,:2]=planar_tensor(self.velocity[:,:2],u,dt,
            COMMAND['axis_acceleration'],COMMAND['axis_brake'],COMMAND['speed_limits'])
        self.velocity[:,0].clamp_(min=COMMAND['min_forward_speed'])
        limit=yaw_limit_tensor(self.velocity,COMMAND['speed_limits'][0],COMMAND['min_forward_speed'],
            COMMAND['speed_limits'][1],COMMAND['yaw_speed_limit'],COMMAND['yaw_cruise_limit'],COMMAND['yaw_speed_decay'])
        self.current_yaw_limit=torch.minimum(limit,torch.as_tensor(getattr(self,'yaw_cap',COMMAND['yaw_speed_limit']),device=limit.device))
        candidate=planar_tensor(old[:,2:3],axes[:,2:3],dt,
            [COMMAND['yaw_acceleration']],[COMMAND['yaw_brake']],[1.e9])[:,0]
        self.velocity[:,2]=brake_to_limit_tensor(candidate,old[:,2],self.current_yaw_limit,COMMAND['yaw_brake'],dt)
        self.heading=wrap(self.heading+.5*(old[:,2]+self.velocity[:,2])*dt)
        self.acceleration=(self.velocity-old)/dt
        return self.velocity

class Schedule:
    def __init__(self,n,device,seed):
        self.device=device;self.n=n
        self.gen=torch.Generator(device=device).manual_seed(seed)
        self.timer=torch.zeros(n,2,device=device);self.value=torch.zeros(n,2,device=device)
        self.primary=torch.zeros(n,dtype=torch.long,device=device)
        self.sign=torch.ones(n,device=device);self.secondary=torch.ones(n,device=device)
        self.key_timer=torch.zeros(n,4,device=device);self.key_down=torch.zeros(n,4,device=device)
    def rand(self,shape):return torch.rand(shape,device=self.device,generator=self.gen)
    def reset(self,ids):
        self.primary[ids]=(ids.long()%4)//2
        self.sign[ids]=torch.where(ids%2==0,1.,-1.)
        self.secondary[ids]=torch.where(self.rand((len(ids),))<.5,-1.,1.)
        self.timer[ids]=0;self.value[ids]=0
        self.key_timer[ids]=0;self.key_down[ids]=0
    def tick(self,stage,age,dt):
        self.timer-=dt;due=self.timer<=0;count=int(due.sum())
        self.value[due]=torch.floor(self.rand((count,))*3)-1
        self.timer[due]=1.+1.5*self.rand((count,))
        axes=torch.zeros(self.n,3,device=self.device);ids=torch.arange(self.n,device=self.device)
        if stage==1:axes[ids,self.primary]=self.sign
        elif stage in (2,3,4):
            axes[ids,self.primary]=self.value[ids,self.primary]
            if stage==3:axes[ids,1-self.primary]=self.value[ids,1-self.primary]
            if stage==4:axes[ids,1-self.primary]=self.value[ids,1-self.primary]
        else:
            self.key_timer-=dt;change=self.key_timer<=0;nk=int(change.sum())
            self.key_down[change]=(self.rand((nk,))<.5).float()
            self.key_timer[change]=.3+2.2*self.rand((nk,))
            axes[:,0]=self.key_down[:,0]-self.key_down[:,1]
            axes[:,1]=self.key_down[:,2]-self.key_down[:,3]
        axes[(age<.5)|(age>=9.)]=0
        return axes
