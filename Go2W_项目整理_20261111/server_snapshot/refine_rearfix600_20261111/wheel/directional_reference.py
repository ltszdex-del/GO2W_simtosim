"""Shared scalar/tensor command rules; brake only until zero during a reversal."""
import math

def advance_axis(value, axis, dt, acceleration, brake):
    if dt <= 0:raise ValueError('dt must be positive')
    if abs(value)<1e-6:value=0.
    axis=max(-1.,min(1.,axis))
    if not axis:
        result=math.copysign(max(0.,abs(value)-brake*dt),value)
        return 0. if abs(result)<.005 else result
    a=axis*acceleration
    if value*axis<0:
        stopping_time=abs(value)/(abs(a)+brake)
        if dt<=stopping_time:return value+math.copysign(abs(a)+brake,a)*dt
        return a*(dt-stopping_time)
    return value+a*dt

def planar_tensor(value,axes,dt,acceleration,brake,limits):
    import torch
    value=torch.where(value.abs()<1e-6,0.,value)
    a=torch.as_tensor(acceleration,device=value.device,dtype=value.dtype)*axes.clamp(-1,1)
    b=torch.as_tensor(brake,device=value.device,dtype=value.dtype)
    scale=torch.as_tensor(limits,device=value.device,dtype=value.dtype)
    active=axes!=0
    opposing=value*axes<0
    stop_time=value.abs()/(a.abs()+b)
    combined=value+a.sign()*(a.abs()+b)*dt
    crossed=a*(dt-stop_time).clamp(min=0)
    reverse=torch.where(dt<=stop_time,combined,crossed)
    released=value.sign()*(value.abs()-b*dt).clamp(min=0)
    released=torch.where(released.abs()<.005,0.,released)
    result=torch.where(active,torch.where(opposing,reverse,value+a*dt),released)
    return result/(result/scale).norm(dim=-1,keepdim=True).clamp(min=1.)
