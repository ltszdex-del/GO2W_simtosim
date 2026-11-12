"""Same clock-driven acceleration integrator for training and interactive adapters."""
import torch
from config import COMMAND

def wrap(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))

class Commands:
    def __init__(self, n, device):
        self.velocity = torch.zeros(n,3,device=device)
        self.heading = torch.zeros(n,device=device)
        self.acceleration = torch.zeros(n,3,device=device)
        self.speed = torch.tensor(COMMAND['speed'],device=device)
        self.accel = torch.tensor(COMMAND['accel'],device=device)
        self.brake = torch.tensor(COMMAND['brake'],device=device)

    def reset(self, ids, heading=0.):
        self.velocity[ids] = 0
        self.acceleration[ids] = 0
        self.heading[ids] = heading

    def tick(self, key_axes, dt):
        # Axes in {-1,0,1}; opposite keys map to zero/braking before this method.
        axes = key_axes.clamp(-1,1).clone()
        axes[:,:2] /= axes[:,:2].norm(dim=-1,keepdim=True).clamp(min=1.)
        old = self.velocity.clone()
        active = axes != 0
        brake_change = -old.sign()*torch.minimum(old.abs(), self.brake*dt)
        self.velocity += torch.where(active, axes*self.accel*dt, brake_change)
        scaled = self.velocity[:,:2]/self.speed[:2]
        self.velocity[:,:2] /= scaled.norm(dim=-1,keepdim=True).clamp(min=1.)
        self.velocity[:,2].clamp_(-self.speed[2], self.speed[2])
        self.acceleration = (self.velocity-old)/dt
        self.heading = wrap(self.heading+self.velocity[:,2]*dt)
        return self.velocity

def keys_to_axes(keys):
    keys = set(k.upper() for k in keys)
    return [int('W' in keys)-int('S' in keys),
            int('A' in keys)-int('D' in keys),
            int('Q' in keys)-int('E' in keys)]
