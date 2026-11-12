"""Command-dependent foot motion guidance. All positions/velocities use yaw frame."""
import torch

def swing_velocity(command,feet):
    tangent=torch.stack((-feet[...,1],feet[...,0]),-1)
    return 2.*(command[:,None,:2]+command[:,None,2:3]*tangent)

def support_margin(feet,point,contacts):
    """Signed distance to convex hull of active contacts, including skewed supports.

    Enumerate six candidate edges; an edge belongs to the hull iff all other
    active contacts lie on its inward side. Fewer than three contacts is unsafe.
    """
    center=(feet*contacts[...,None]).sum(1)/contacts.sum(-1,keepdim=True).clamp(min=1)
    margins=[]
    for i in range(4):
        for j in range(i+1,4):
            a=feet[:,i];edge=feet[:,j]-a
            cross=lambda v:edge[:,0]*v[...,1]-edge[:,1]*v[...,0]
            side=cross(center-a).sign()
            side=torch.where(side==0,torch.ones_like(side),side)
            distances=torch.stack([cross(feet[:,k]-a)*side for k in range(4)],-1)
            hull=contacts[:,i]&contacts[:,j]&((distances>=-1e-6)|~contacts).all(-1)&(edge.norm(dim=-1)>1e-5)
            margin=cross(point-a)*side/edge.norm(dim=-1).clamp(min=1e-5)
            margins.append(torch.where(hull,margin,torch.full_like(margin,1e3)))
    result=torch.stack(margins,-1).amin(-1)
    return torch.where((contacts.sum(-1)>=3)&(result<999),result,torch.full_like(result,-.1))
