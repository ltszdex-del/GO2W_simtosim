CONTROL=dict(leg_rate_rad_s=1.4,leg_accel_rad_s2=3.,goal_velocity_gain=.1,residual_velocity_rad_s=.25,power_on_ramp_s=.02,release_s=.5,contact_confirm_s=.12,kp=70.,kd=2.5,body_angle_limit_deg=5.,loaded_wheel_limit_deg=10.,attitude_calf_offset_gain=2.,attitude_calf_offset_max_rad=.15,startup_impact_grace_s=.1)
import json
from pathlib import Path
REWARDS=json.loads((Path(__file__).parent/'reward_port.json').read_text())['weights']
WHEEL_ALIGNMENT=dict(frame='world',wheels='all four',limit_deg=10.,mode='P diagnostic only; wheel motors retain zero-speed damping until released')
