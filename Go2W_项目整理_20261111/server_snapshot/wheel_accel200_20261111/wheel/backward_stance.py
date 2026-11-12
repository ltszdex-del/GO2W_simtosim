"""Backward support clearance; no prescribed leg posture or maximum width."""
import torch

def backward_stance(feet, reference, command, actual_velocity, width_fraction=.95, side_fraction=.90):
    # FR, FL, RR, RL. Signed side distances also catch crossing rear wheels.
    rear=feet[:,2:,1];sign=rear.new_tensor([-1.,1.])
    nominal=(reference[3,1]-reference[2,1])
    width=rear[:,1]-rear[:,0]
    width_cost=((width_fraction*nominal-width).clamp(min=0)/.04).square().clamp(max=6.)
    side_min=.5*side_fraction*nominal
    side_cost=((side_min-rear*sign).clamp(min=0)/.04).square().mean(-1).clamp(max=6.)
    backwards=(-torch.minimum(command[:,0],actual_velocity[:,0])/.25).clamp(0,1)
    # Preserve room for the lateral COM shift needed during combined turns.
    straight=.25+.75*torch.exp(-(command[:,2]/.6).square())
    gate=backwards*straight
    return dict(width_cost=width_cost,side_cost=side_cost,gate=gate,rear_width_m=width,
        required_rear_width_m=torch.ones_like(width)*width_fraction*nominal,
        inward_fraction=((width<width_fraction*nominal)&(backwards>.5)).float())
