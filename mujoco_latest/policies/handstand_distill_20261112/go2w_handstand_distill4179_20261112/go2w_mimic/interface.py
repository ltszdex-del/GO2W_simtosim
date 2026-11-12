"""权重、参考和控制代码一起编号，避免同样维度却使用不同参考的旧错误。"""
import hashlib
import json
from pathlib import Path
from .settings import *


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(motion=MOTION):
    # 代码哈希参与契约：导出和播放必须使用与训练一致的观测、控制和网络。
    sources={n:sha(ROOT/"go2w_mimic"/n) for n in
             ("env.py","env_cfg.py","networks.py","settings.py","reference.py")}
    return dict(task=TASK,teacher_observations=TEACHER_DIM,student_observations=STUDENT_DIM,
        history=HISTORY,history_order="oldest_to_newest",proprio_dim=PROPRIO_DIM,goal_dim=GOAL_DIM,
        actions=16,action_output="bounded [-1,1]; do NOT apply tanh again",
        leg_order=LEG_NAMES,wheel_order=WHEEL_NAMES,control=CONTROL,reward_weights=REWARDS,
        control_dt=CONTROL_DT,physics_dt=DT,physics=PHYSICS,wheel_guard=WHEEL_GUARD,
        heading_convention="atan2(-R01,R11): horizontal lateral body axis",
        proprio_order=["gravity_body(3)","omega_body*0.25(3)","leg_q-reference_q(12)",
                       "joint_velocity*0.05(16:legs_then_wheels)","previous_bounded_action(16)",
                       "yaw_sin_cos(2)"],
        goal_order=["reference_gravity(3)","reference_linear_velocity_body(3)",
            "reference_angular_velocity_body*0.25(3)","reference_leg_q(12)",
            "reference_leg_qd*0.05(12)","reference_minus_actual_axles_body(12:FR_FL_RR_RL)",
            "phase_and_finished(2)","reference_contacts(4)","reference_wheel_velocity*0.05(4)",
            "front_support_displacement_in_heading_frame/0.2m(2)"],
        position_anchor="initial measured front axle XY, fixed per episode; requires tracking/odometry at deployment",
        normalization="fixed scaling above; no RunningMeanStd",
        reset_history="repeat first proprio sample 50 times; never carry across reset",
        motion_sha256=sha(motion),urdf_sha256=sha(URDF),sources=sources,initial_standing_sha256=sha(ROOT/"U_final_standing.json"))


def verify(saved,current,allow_reward_update=False,allow_physics_update=False):
    for key in ("initial_standing_sha256","task","teacher_observations","student_observations","motion_sha256",
                "urdf_sha256","sources","control","reward_weights","physics",
                "heading_convention","goal_order","position_anchor","wheel_guard"):
        if saved.get(key)==current.get(key):
            continue
        if allow_physics_update and key=="physics":
            continue
        if allow_physics_update and key=="sources":
            changed={name for name in saved[key] if saved[key][name]!=current[key].get(name)}
            if changed <= {"env.py","env_cfg.py","settings.py"} and saved[key].keys()==current[key].keys():
                continue
        if allow_reward_update and key in ("reward_weights","wheel_guard"):
            continue
        if allow_reward_update and key=="sources":
            changed={name for name in saved[key] if saved[key][name]!=current[key].get(name)}
            if changed <= {"env.py","env_cfg.py","settings.py"} and saved[key].keys()==current[key].keys():
                continue
        raise ValueError(f"Checkpoint/interface mismatch: {key}. Use its saved code and motion.")


def write_json(path, data):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
