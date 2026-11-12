"""Soft safety margins for FR, FL, RR, RL wheel centers, without a fixed stance."""
import math
import torch

def geometry_costs(feet, reference, center_height, contact, gravity, acceleration_x, radius):
    # Signed distances also penalize crossing wheels. Only intrusion below a
    # minimum is penalized; wider/longer stance is freely selectable.
    longitudinal=feet[:,:2,0]-feet[:,2:,0]
    nominal_long=reference[:2,0]-reference[2:,0]
    front_rear=((.8*nominal_long-longitudinal).clamp(min=0)/.05).square().mean(-1).clamp(max=4)
    widths=feet[:,[1,3],1]-feet[:,[0,2],1]
    nominal_width=reference[[1,3],1]-reference[[0,2],1]
    axle=((.8*nominal_width-widths).clamp(min=0)/.05).square().mean(-1).clamp(max=4)
    gap=(center_height-radius-.015).clamp(min=0)
    lift=(gap/.04).square().mean(-1).clamp(max=4)
    missing=(~contact).float().mean(-1)
    pitch=torch.atan2(gravity[:,0],-gravity[:,2]).abs()
    allowance=(math.radians(5)+torch.atan(acceleration_x.abs()/9.81)).clamp(max=math.radians(12))
    excess=(pitch-allowance).clamp(min=0)
    pitch_cost=(excess/math.radians(5)).square().clamp(max=9)
    return dict(front_rear_cost=front_rear,axle_cost=axle,missing_contact=missing,
                lift_cost=lift,pitch_cost=pitch_cost,front_rear_min_m=longitudinal.min(-1).values,
                front_width_m=widths[:,0],rear_width_m=widths[:,1],
                pitch_abs_deg=pitch*180/math.pi,pitch_allowance_deg=allowance*180/math.pi)
