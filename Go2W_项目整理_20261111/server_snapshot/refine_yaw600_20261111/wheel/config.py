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
COMMAND.update(yaw_speed_limit=.8,yaw_acceleration=1.,yaw_brake=1.6)
GAIT.update(rotation_drift_weight=1.,rotation_stance_slip=.35,rotation_swing_tangent=.25,rotation_stop_yaw=1.,rotation_stop_support=1.)
PPO = dict(steps=32, epochs=4, minibatches=4, gamma=.99, lam=.95,
           clip=.2, lr=3e-4, entropy=.005, grad_norm=1., max_std=.65)

def contract():
    return dict(schema='go2w-wasd-locked-v1', actor_dim=ACTOR_DIM,
        critic_dim=CRITIC_DIM, action_dim=12, history=HISTORY, history_order='oldest first',
        observation_order=['gravity_b(3)', 'omega_b*.25(3)', 'q_leg-stand(12)',
            'dq_all*.05(16)', 'previous_bounded_action(16)', 'vx_vy_wz_reference(3)',
            'sin_heading_error,cos_heading_error(2)','sin_gait_phase,cos_gait_phase(2)'],
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

_leg_contract=contract
CONTROL.update(wheel_lock_damping=3.,wheel_residual=4.,wheel_speed=30.1,leg_rate=2.,leg_scale=[.08,.12,.12]*4)
import json as _json
_U=_json.loads((ROOT/'U_final_standing.json').read_text())
CONTROL['stand']=_U['target']
COMMAND.update(max_speed=1.5,min_forward_speed=-1.,speed_limits=[1.5,1.3])
def contract():
    m=_leg_contract();m.update(schema='go2w-wheel-drive-v1',action_dim=16,action='12 small leg offsets + 4 residual wheel velocities',wheel_feedforward='(vx-1.5*actual_wheel_center_y*wz)/radius + bounded wheel residual; implicit damping3; accel60 brake40',previous_action='last bounded16 action')
    return m

GAIT.update(stance_feet_reference=[[0.11462120711803436, -0.20339162647724152, -0.30542758107185364], [0.1146635115146637, 0.202809140086174, -0.305805504322052], [-0.2725781202316284, -0.19163329899311066, -0.3026198744773865], [-0.27242448925971985, 0.19086721539497375, -0.30299127101898193]],wheel_center_offsets=[[0.0, -0.048, 0.0], [0.0, 0.048, 0.0], [0.0, -0.048, 0.0], [0.0, 0.048, 0.0]])
COMMAND.update(max_speed=2.5,min_forward_speed=-2.5,speed_limits=[2.5,1.3],axis_acceleration=[10/3,1.],backward_acceleration=5.,axis_brake=[4.,2.6],brake_acceleration=4.,wheel_envelope_lever=0.3050874397158623)
CONTROL.update(wheel_accel=60.,wheel_brake_accel=40.)

CONTROL.update(leg_scale=[.14,.34,.34]*4,leg_rate=4.)
PPO.update(lr=1e-4,entropy=.002,max_std=.4)
GAIT.update(dynamic_balance=True,posture_low_speed_only=True,leg_freedom_ramp_updates=60,speed_ramp_updates=100)

PPO.update(entropy=.001,max_std=.35)

PPO.update(lr=5e-5,entropy=.001,max_std=.30,straight_preservation=.10)
GAIT.update(turn_course_updates=100,turn_training_mix=[.25,.375,.375],turn_gate_every=50,turn_strategy='centripetal_COM_contact_support_required_skid_v1')

PPO.update(symmetry_weight=.15,lr=5e-5,straight_preservation=.02)
GAIT.update(refinement_version='command_foot_guidance_convex_support_v1')
GAIT.update(turn_training_mix=[.25,.25,.25,.25],turn_training_groups=['straight','pure_QE','low_speed_combined','high_speed_combined'])

# Fresh PPO run: 3 rad/s wheel yaw envelope, no inherited policy prior.
COMMAND.update(yaw_speed_limit=3.)
PPO.update(lr=3e-4,entropy=.005,max_std=.5,straight_preservation=0.)
GAIT.update(turn_course_updates=400,scratch_training=True)

# Independent leg-position and wheel-velocity actions; no fixed skid feedforward.
CONTROL.update(leg_scale=[.30,.60,.60]*4, wheel_action_scale=30.1)
GAIT.update(refinement_version='direct_wheel_velocity_v2',turn_course_updates=500)
_previous_contract=contract
def contract():
    m=_previous_contract()
    m.update(schema='go2w-wheel-direct-v2',
        action='12 leg position offsets + 4 absolute wheel velocity actions',
        wheel_feedforward='none; wheel_target=30.1*tanh_action; accel60 brake40',
        previous_action='last bounded16 action before rate limiting')
    return m

COMMAND.update(yaw_brake=4.)
CONTROL.update(wheel_brake_accel=100.,leg_rate=6.)
PPO.update(lr=1e-4,entropy=.002,max_std=.35,symmetry_weight=.5)
GAIT.update(refinement_version='symmetric_straight_smooth_brake_recovery_v3')
_direct_contract=contract
def contract():
    m=_direct_contract();m['schema']='go2w-wheel-mirror-brake-v3'
    m['inference']='sagittal projected actor; wheel gait phase unchanged'
    m['wheel_feedforward']='none; absolute wheel velocity actions; accel60 brake100'
    return m

COMMAND.update(yaw_acceleration=2.,yaw_brake=8.)
GAIT.update(refinement_version='wheel_clearance_contact_pitch_v4',minimum_pair_fraction=.8,minimum_axle_fraction=.8,pitch_idle_allowance_deg=5.,pitch_dynamic_allowance_deg=12.)

COMMAND.update(yaw_speed_limit=2.,max_speed=3.,speed_limits=[3.,1.3],axis_acceleration=[4.,1.2],backward_acceleration=6.,axis_brake=[4.8,3.12],brake_acceleration=4.8,yaw_acceleration=2.4,yaw_brake=9.6)

GAIT['sampling_revision']='combo_focused_300_20261111'

# Final capability refinement: reference translation determines yaw ceiling.
COMMAND.update(yaw_speed_limit=4.0,yaw_cruise_limit=2.0,yaw_speed_decay=3.,yaw_acceleration=4.8,yaw_brake=19.2)
