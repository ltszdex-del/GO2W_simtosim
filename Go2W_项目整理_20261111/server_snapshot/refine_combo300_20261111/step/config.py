"""Portable policy contract. SI units, PhysX/Isaac Lab 3 + Isaac Sim 6.2."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
URDF = ROOT / 'assets/go2w_description/urdf/go2w_description.urdf'
ORDER = ('FR', 'FL', 'RR', 'RL')
LEGS = [f'{leg}_{j}_joint' for leg in ORDER for j in ('hip', 'thigh', 'calf')]
WHEELS = [f'{leg}_foot_joint' for leg in ORDER]
FEET = [f'{leg}_foot' for leg in ORDER]
HISTORY, FRAME, ACTOR_DIM, CRITIC_DIM = 10, 57, 570, 68
CONTROL = dict(physics_dt=.005, decimation=4, kp=70., kd=2.5, wheel_kd=.2,
               effort=23.5, leg_rate=6., wheel_accel=40., wheel_residual=6.,
               leg_scale=[.25, .55, .55]*4, stand=[0., .8, -1.5]*4,
               wheel_radius=.086, half_track=.19)
COMMAND = dict(min_forward_speed=-1.5, max_speed=2.3, acceleration=2.0, damping=5.0, brake_acceleration=4.6, release_mode="linear_brake", speed_limits=[2.3,1.3], axis_acceleration=[2.,1.], axis_brake=[4.6,2.6], reversal="input_plus_brake_until_zero")
CONTROL['leg_scale']=[.5,.7,.7]*4
CONTROL['wheel_speed']=25.
CONTROL['wheel_lock_damping']=100.
GAIT=dict(period=.30, swing_height=.05, contact_weight=.9, clearance_weight=.9, hip_angle_weight=.09, hip_speed_weight=.0225, swing_amplitude_weight=.15, excess_clearance_weight=0.0, swing_deficit_weight=.5, swing_contact_weight=.3, stance_slip_weight=.15, posture_weight=.045, upright_weight=1.0, track_peak_weight=1.5, track_error_weight=.35, reward_ramp_iterations=0)
GAIT.update(diagonal_reward=.5, diagonal_mismatch=.6, swing_drag=.35, diagonal_ramp=75)
GAIT.update(height_lower=.365,height_upper=.425,low_height_weight=.8,front_fold_weight=.35,backward_initial_cap=1.5,backward_max_cap=1.5)
GAIT.update(U_stance_q=[-0.04337595775723457, 1.0254065990447998, -1.499308466911316, 0.04145922511816025, 1.0249897241592407, -1.4988728761672974, -0.00539023894816637, 1.0618525743484497, -1.5532371997833252, 0.0028602248057723045, 1.0614906549453735, -1.5534567832946777],U_stance_height=0.3914146423339844,U_stance_speed_scale=.25,U_stance_weight=1.)
GAIT.update(failure_penalty=2.5,front_fold_scope='all_translation',diagonal_tolerance_s=.04)
GAIT['diagonal_mismatch']=.3
COMMAND.update(yaw_speed_limit=1.2,yaw_acceleration=1.5,yaw_brake=2.4)
GAIT.update(rotation_drift_weight=1.,rotation_stance_slip=.35,rotation_swing_tangent=.25,rotation_stop_yaw=1.,rotation_stop_support=1.)
PPO = dict(steps=32, epochs=4, minibatches=4, gamma=.99, lam=.95,
           clip=.2, lr=3e-4, entropy=.001, grad_norm=1., max_std=.4)

def contract():
    return dict(schema='go2w-wasd-locked-v1', actor_dim=ACTOR_DIM,
        critic_dim=CRITIC_DIM, action_dim=12, history=HISTORY, history_order='oldest first',
        observation_order=['gravity_b(3)', 'omega_b*.25(3)', 'q_leg-stand(12)',
            'dq_all*.05(16)', 'previous_bounded_action(16)', 'vx_vy_wz_reference(3)',
            'sin_heading_error,cos_heading_error(2)','motion-gated sin_gait_phase,cos_gait_phase(2); gain=max(vref/.15,abs(wref)/.15) clipped[0,1]'],
        normalization='fixed scales above; clip frame to [-10,10]; no running actor normalization',
        actor_excludes=['root linear velocity', 'height', 'contacts', 'terrain map', 'camera'],
        critic_extra=['yaw-frame vx,vy,world vz,world wz(4)', 'height(1)',
                      'wheel contact flags(4)', 'anchor error in yaw frame(2)'],
        frame='body yaw-only horizontal frame, x forward y left; quaternion xyzw in runtime',
        action='tanh Gaussian: 12 leg position offsets; wheel action slots fixed zero',
        wheel_feedforward='none; implicit wheel zero-velocity drive damping=100, effort cap=23.5Nm',
        previous_action='last bounded network action, before target rate limiting',
        phase=GAIT, joint_order=LEGS+WHEELS, control=CONTROL, command=COMMAND,
        hidden=[256,128,128], physics=dict(friction=.8,self_collision=True,
            fixed_joint_merge=True,armature=0.,passive_friction=0.),
        deployment_note='Heading error uses relative integrated IMU yaw; actor has no privileged velocity.')

GAIT.update(stance_feet_reference=[[0.11462120711803436, -0.20339162647724152, -0.30542758107185364], [0.1146635115146637, 0.202809140086174, -0.305805504322052], [-0.2725781202316284, -0.19163329899311066, -0.3026198744773865], [-0.27242448925971985, 0.19086721539497375, -0.30299127101898193]],wheel_center_offsets=[[0.0, -0.048, 0.0], [0.0, 0.048, 0.0], [0.0, -0.048, 0.0], [0.0, 0.048, 0.0]])
GAIT['sagittal_weight']=.04

GAIT.update(forward_outward_hip_weight=.25,forward_outward_hip_deadband=.03,forward_outward_hip_scale=.2)

PPO.update(symmetry_weight=.15,lr=5e-5)
GAIT.update(refinement_version='command_foot_guidance_convex_support_v1')
GAIT.update(swing_deficit_weight=.9,swing_contact_weight=.6,swing_drag=.6,sagittal_weight=.15)

GAIT.update(period_range_s=[.25,.35],period_sampling='uniform_per_environment_per_episode',specialist_task='qe')

CONTROL.update(leg_rate=8.)
PPO.update(lr=1e-4,symmetry_weight=1.,max_std=.35)
GAIT.update(specialist_task='comprehensive',symmetry_mode='exact_actor_paired_physics_v1',
    gait_phase_offset='adjacent pair: 0/pi',paired_physics=True)
_legacy_contract=contract
def contract():
    m=_legacy_contract();m['schema']='go2w-step-mirror-v2'
    m['inference']='tanh(0.5*(actor(obs)+mirror_action(actor(mirror_obs(obs)))))'
    return m

PPO.update(lr=5e-5)
GAIT.update(front_turn_extra_width_m=.012,front_turn_width_weight=.06,comprehensive_fraction=.9)

COMMAND.update(yaw_speed_limit=1.6)
COMMAND.update(yaw_acceleration=2.25,yaw_brake=3.6,axis_brake=[9.2,5.2],brake_acceleration=9.2)

GAIT.update(period=.28,period_range_s=[.23,.33])

GAIT['sampling_revision']='combo_focused_300_20261111'
