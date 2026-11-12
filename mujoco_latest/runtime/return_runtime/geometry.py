"""World-space wheel axle geometry, independent of Isaac Sim."""
import torch
import torch.nn.functional as F

def wheel_pair_geometry(centers,axes):
    # Inputs [N,4,3], ordered FR FL RR RL. Opposite axis signs describe the same line.
    axes=F.normalize(axes,dim=-1)
    right=axes[:,(0,2)];left=axes[:,(1,3)]
    dot=(right*left).sum(-1)
    aligned_left=left*torch.where(dot<0,-1.,1.)[...,None]
    angle=dot.abs().clamp(0,1).acos()
    delta=centers[:,(1,3)]-centers[:,(0,2)]
    off_r=delta-(delta*right).sum(-1,keepdim=True)*right
    off_l=delta-(delta*left).sum(-1,keepdim=True)*left
    offset=((off_r.square().sum(-1)+off_l.square().sum(-1))*.5).sqrt()
    return angle,offset,(off_r+off_l)*.5,right-aligned_left
