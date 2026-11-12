"""Body sagittal reflection. Polar y reverses; axial x/z reverse."""
import torch
from symmetry import mirror_action,mirror_observation

def mirror_polar(x):
    y=x.clone();y[...,1]*=-1;return y

def mirror_axial(x):
    y=x.clone();y[...,[0,2]]*=-1;return y

def mirror_feet(x):
    return mirror_polar(x[:,[1,0,3,2]])

def paired_costs(q,dq,gravity,omega,feet,velocity,yaw_rate):
    def cost(x,y,scale):return ((x-y)/scale).square().mean(-1).clamp(max=4).repeat_interleave(2)
    pose=cost(q[0::2],mirror_action(q[1::2]),q.new_tensor([.3,.5,.5]*4))
    speed=cost(dq[0::2],mirror_action(dq[1::2]),6.)
    body=cost(gravity[0::2],mirror_polar(gravity[1::2]),.3)
    body+=.25*cost(omega[0::2],mirror_axial(omega[1::2]),3.)
    placement=cost(feet[0::2].flatten(1),mirror_feet(feet[1::2]).flatten(1),.10)
    linear=cost(velocity[0::2],mirror_polar(velocity[1::2]),.4)
    angular=((yaw_rate[0::2]+yaw_rate[1::2])/1.2).square().clamp(max=4).repeat_interleave(2)
    return dict(pose=pose,joint_velocity=speed,body=body,feet=placement,linear=linear,yaw=angular)
