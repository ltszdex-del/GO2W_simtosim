"""Sagittal reflection for the actual 57 x 10 policy history (FR,FL,RR,RL)."""
import torch

def mirror_action(a):
    legs=a[...,:12].reshape(*a.shape[:-1],4,3)[...,[1,0,3,2],:].clone()
    legs[...,0]*=-1
    result=legs.flatten(-2)
    if a.shape[-1]==16:
        result=torch.cat((result,a[...,12:][...,[1,0,3,2]]),-1)
    return result

def mirror_observation(obs,stand,stepping):
    x=obs.reshape(-1,10,57);y=x.clone()
    y[...,1]*=-1                 # polar gravity
    y[...,[3,5]]*=-1             # axial angular velocity
    y[...,6:18]=mirror_action(x[...,6:18]+stand)-stand
    y[...,18:34]=mirror_action(x[...,18:34])
    y[...,34:50]=mirror_action(x[...,34:50])
    y[...,[51,52,53]]*=-1        # vy, yaw rate, sin(heading)
    if stepping:y[...,55:57]*=-1 # reflection exchanges diagonal gait phases
    return y.reshape_as(obs)
