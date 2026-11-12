"""Shared reward calculations; actuator targets and observation layout stay unchanged."""
import torch

def stopping_mask(keys,reference):
    return ((keys[:,0]==0)&(keys[:,2]==0)&(reference[:,0].abs()<.2)
            &(reference[:,2].abs()<.08))

def excess_lateral_slip(feet_b,point_b,axis_b,reference,contact):
    expected=torch.stack((reference[:,0:1]-reference[:,2:3]*feet_b[:,:,1],
                          reference[:,1:2]+reference[:,2:3]*feet_b[:,:,0]),-1)
    actual=(point_b*axis_b).sum(-1)
    required=(expected*axis_b).sum(-1)
    return ((actual-required).square()*contact).sum(-1)/contact.sum(-1).clamp(min=1)

def residual_yaw_opposition(action,reference,yaw_rate,residual_scale):
    residual=.5*(action[:,[12,14]].mean(-1)-action[:,[13,15]].mean(-1))*residual_scale
    direction=reference[:,2].sign()
    deficit=((reference[:,2].abs()-direction*yaw_rate)/reference[:,2].abs().clamp(min=.2)).clamp(0,2)
    opposing=(-direction*residual/residual_scale).clamp(min=0).square()
    active=(reference[:,2].abs()>.15).float()
    return active*deficit*opposing
