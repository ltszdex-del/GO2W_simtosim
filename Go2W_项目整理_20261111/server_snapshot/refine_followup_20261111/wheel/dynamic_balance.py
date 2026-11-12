"""Soft planar COM/ZMP guidance for acceleration and cornering, not a hard pose."""
import torch

def balance_metrics(velocity,reference,yaw_rate,acceleration_ref,acceleration_measured,
                    feet_xy,com_xy,com_height,root_height,contacts=None):
    speed=torch.maximum(velocity.norm(dim=-1),reference[:,:2].norm(dim=-1))
    turn=torch.maximum(yaw_rate.abs(),reference[:,2].abs())
    quiet=torch.exp(-(speed/.25).pow(4)-(turn/.35).pow(4))
    fraction=(speed/2.5).clamp(0,1)
    lo=feet_xy.amin(1);hi=feet_xy.amax(1)
    support_count=torch.full_like(speed,4.)
    if contacts is not None:
        support_count=contacts.sum(-1)
        contact_lo=feet_xy.masked_fill(~contacts[...,None],float('inf')).amin(1)
        contact_hi=feet_xy.masked_fill(~contacts[...,None],-float('inf')).amax(1)
        usable=support_count>=3
        lo=torch.where(usable[:,None],contact_lo,lo)
        hi=torch.where(usable[:,None],contact_hi,hi)
    middle=.5*(lo+hi);h=com_height.clamp(.15,.55)
    # Body-yaw reference acceleration includes the centripetal term omega x v.
    acceleration=acceleration_ref.clone()
    acceleration[:,0]-=reference[:,2]*reference[:,1]
    acceleration[:,1]+=reference[:,2]*reference[:,0]
    offset=h[:,None]*acceleration.clamp(-4.,4.)/9.81
    coasting=(1.-acceleration_ref[:,0].abs()).clamp(0,1)
    offset[:,0]+=-velocity[:,0].sign()*.025*fraction*coasting
    room=(.5*(hi-lo)-.045).clamp(min=.025,max=.14)
    offset=torch.maximum(-room,torch.minimum(room,offset))
    desired=middle+offset
    com_track=torch.exp(-((com_xy-desired)/torch.tensor([.07,.06],device=com_xy.device)).square().sum(-1))
    zmp=com_xy-h[:,None]*acceleration_measured.clamp(-8.,8.)/9.81
    margin=torch.minimum(zmp-lo,hi-zmp)
    outside=((.015-margin).clamp(min=0)/.05).square().sum(-1)
    outside+=(3-support_count).clamp(min=0).float()
    corner=(acceleration[:,1].abs()/2.).clamp(0,1)
    upper=.415-.045*fraction-.025*corner
    lower=torch.full_like(upper,.30)
    height_cost=((root_height-upper).clamp(min=0)/.05).square()+((lower-root_height).clamp(min=0)/.05).square()
    return dict(quiet=quiet,speed=speed,desired_com_x=desired[:,0],desired_com_y=desired[:,1],
        com_offset=offset[:,0],com_offset_y=offset[:,1],lateral_acceleration=acceleration[:,1],
        com_track=com_track,zmp_cost=outside.clamp(max=6),zmp_margin=margin.amin(-1),
        height_cost=height_cost.clamp(max=9))
