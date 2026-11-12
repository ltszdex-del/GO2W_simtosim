"""统一的训练/部署接口。单位：米、秒、弧度、牛·米。

本项目借鉴 MimicAgent 的参考模仿 + PPO 教师 + DAgger 学生流程，
是独立的教学实现，不是论文作者的逐行复现或已验收控制器。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets/go2w_description/urdf/go2w_description.urdf"
MOTION = ROOT / "motions/stand_to_handstand.npz"
TASK = "Go2W-Mimic-Handstand-v1"
ORDER = ("FR", "FL", "RR", "RL")
LEG_NAMES = [f"{leg}_{part}_joint" for leg in ORDER for part in ("hip", "thigh", "calf")]
WHEEL_NAMES = [f"{leg}_foot_joint" for leg in ORDER]
FOOT_NAMES = [f"{leg}_foot" for leg in ORDER]
DT, DECIMATION = 0.005, 4  # 200 Hz 物理与力矩环；50 Hz 策略。
CONTROL_DT = DT * DECIMATION
HISTORY = 50  # 1 秒本体感觉历史，旧帧在前，新帧在后。
PROPRIO_DIM = 52  # gravity3, angular_velocity3, q_error12, qdot16, previous_action16, yaw_sin_cos2
GOAL_DIM = 57  # previous 55 + anchored front-support displacement / .2m (body XY)
PRIV_DIM = 31  # v3,z1,contact4,COM3,mass1,gains2,friction1,delay1,lookahead_q12,g3
TEACHER_DIM = PROPRIO_DIM + GOAL_DIM + PRIV_DIM
STUDENT_DIM = HISTORY * PROPRIO_DIM + GOAL_DIM
ACTION_DIM = 16
WHEEL_GUARD = dict(collision_force_epsilon=0.01, max_front_axis_angle_rad=0.174533,
    common_front_hip_limit_rad=0.05)

# Official URDF solver limits; explicit PD retains its conservative 23.5 Nm cap.
PHYSICS = dict(joint_velocity_limits={".*_hip_joint":30.1, ".*_thigh_joint":30.1,
    ".*_calf_joint":20.07, ".*_foot_joint":30.1}, armature=0.,
    friction=0., dynamic_friction=0., viscous_friction=0.)

# 显式力矩 PD 在每个物理子步执行，使 Isaac 与 MuJoCo 更容易对齐。
# 这些是待动力学验证的初值，并非宇树提供的倒立控制参数。
# 轮轴惯量约 0.0012 kg·m²；5 ms 显式速度环使用低增益，避免接触时符号交替振荡。
CONTROL = dict(kp=70.0, kd=2.5, wheel_kd=0.2, effort=23.5,
               leg_residual=0.55, hip_residual=0.25, leg_rate=8.0,
               wheel_scale=6.0, wheel_acceleration=40.0, soft_limit=0.95)
REWARDS = dict(joints=1.0, joint_velocity=0.2, orientation=6.0, heading=1.5, height=1.0,
               lift=2.0, height_progress=1.0, sitting=-2.0,
               feet=1.0, root_velocity=0.3, angular_velocity=1.0, contact=1.0,
               hold=3.0, hold_soft=2.0, hold_stability=4.0, balance=0.8,
               front_parallel=0.5, rear_parallel=0.5,
               slip=-0.3, impact=-0.2,
               front_slip=-1.5, over_height=-2.0,
               front_travel=-2.0, front_speed=-0.5, turning=-0.75,
               hold_translation=-1.0, hold_spin=-1.5,
               action_rate=-0.02, hold_action_rate=-8.0,
               hold_action_acceleration=-1.0,
               # Hold only: +50% leg-speed penalty, plus all four wheel joints.
               front_axis_parallel=2.0, hold_leg_speed=-0.3, hold_wheel_speed=-0.1,
               effort=-0.01, acceleration=-0.03,
               limits=-1.0, bad_contact=-1.0)
PPO = dict(steps=64, epochs=5, minibatches=4, gamma=0.995, lam=0.95,
           clip=0.2, value_coef=1.0, entropy_coef=0.0005, max_action_std=0.25,
           learning_rate=3e-4, max_grad_norm=1.0)
