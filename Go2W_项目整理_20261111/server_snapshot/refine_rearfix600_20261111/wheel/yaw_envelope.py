"""Reference-speed yaw envelope; scalar deployment and tensor training agree."""
import math


def yaw_limit(vx, vy, forward, backward, lateral, stationary, cruising, decay=3.):
    speed=min(1., math.hypot(vx/(forward if vx>=0 else abs(backward)), vy/lateral))
    weight=(math.exp(-decay*speed)-math.exp(-decay))/(1.-math.exp(-decay))
    return cruising+(stationary-cruising)*weight


def yaw_limit_tensor(velocity, forward, backward, lateral, stationary, cruising, decay=3.):
    import torch
    longitudinal=torch.where(velocity[:,0]>=0,forward,abs(backward))
    speed=torch.hypot(velocity[:,0]/longitudinal,velocity[:,1]/lateral).clamp(max=1.)
    weight=(torch.exp(-decay*speed)-math.exp(-decay))/(1.-math.exp(-decay))
    return cruising+(stationary-cruising)*weight


def brake_to_limit(candidate, old, limit, brake, dt):
    """A falling limit requests braking; it never teleports the reference."""
    bound=max(limit,abs(old)-brake*dt)
    return max(-bound,min(bound,candidate))


def brake_to_limit_tensor(candidate, old, limit, brake, dt):
    import torch
    bound=torch.maximum(torch.as_tensor(limit,device=old.device),old.abs()-brake*dt)
    return torch.minimum(torch.maximum(candidate,-bound),bound)
