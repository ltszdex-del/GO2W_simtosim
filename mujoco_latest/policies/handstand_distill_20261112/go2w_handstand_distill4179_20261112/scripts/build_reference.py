"""离线生成参考 NPZ。运行本脚本不会训练、不会启动 Isaac。

第一版使用人工关键帧 + URDF FK/IK；替换了 MimicAgent 的 LLM 轨迹生成环节。
这保留了论文“先描述运动、再在物理中学会实现”的核心。
前轮保持名义接地，后轮在蹬地结束前保持名义接地；之后收腿。
注意：检查通过只说明几何合理，不能证明力矩足够或过渡动态稳定。
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from go2w_mimic.kinematics import Kinematics, transform
from go2w_mimic.settings import URDF, MOTION, LEG_NAMES, FOOT_NAMES, ORDER


def interpolate(times, values, samples):
    """C1 连续的三次 Hermite 插值；首尾速度为零，中间保持连续速度。

    相邻静止关键帧之间的斜率也设为零，保证开头的站立等待段不提前动作。
    """
    values = np.asarray(values)
    slopes = np.zeros_like(values)
    for i in range(1, len(times)-1):
        a, b = values[i]-values[i-1], values[i+1]-values[i]
        slopes[i] = np.where(a*b > 0, (a+b)/(times[i+1]-times[i-1]), 0)
    result = []
    for t in samples:
        i = min(max(np.searchsorted(times, t, side="right")-1, 0), len(times)-2)
        h = times[i+1]-times[i]
        s = np.clip((t-times[i])/h, 0, 1)
        result.append((2*s**3-3*s*s+1)*values[i] + (s**3-2*s*s+s)*h*slopes[i]
                      + (-2*s**3+3*s*s)*values[i+1] + (s**3-s*s)*h*slopes[i+1])
    return np.asarray(result)


def generate():
    kin = Kinematics(URDF)
    seed = json.loads((ROOT/"motions/seed_keyframes.json").read_text(encoding="utf-8"))
    # 起点来自 getup 名义关节角；高度由轮胎接地重算，避免固定高度引入穿地。
    keys = [seed["keyframes"][0]] + seed["keyframes"]
    times = np.array([0., .4, 1.25, 1.8, 2.5, 3.6])
    qkeys = np.array([[k["joint_positions_rad"][n] for n in LEG_NAMES] for k in keys])
    pkeys = np.array([k["pitch_rad"] for k in keys])
    radius = .086  # 与 bundled URDF cylinder 相同；不是估计的接触点高度。
    initial_fk = kin.fk(dict(zip(LEG_NAMES, qkeys[0])))
    initial_feet = np.array([initial_fk[n][:3, 3] for n in FOOT_NAMES])
    anchor_x = initial_feet[:2, 0].mean()
    initial_height = radius-initial_feet[:2, 2].mean()
    rear_world = initial_feet[2:]+np.array([0, 0, initial_height])

    def root_for(q, pitch):
        poses = kin.fk(dict(zip(LEG_NAMES, q)), transform(rpy=(0, pitch, 0)))
        feet = np.array([poses[n][:3, 3] for n in FOOT_NAMES])
        # 名义前轮支点不移动；真实策略允许有限滚动纠偏，奖励不锁定世界坐标。
        root = transform((anchor_x-feet[:2, 0].mean(), 0, radius-feet[:2, 2].mean()),
                         (0, pitch, 0))
        return root

    # 校正蹬地关键帧：后轮的目标仍在地面，用 IK 求腿角。
    root = root_for(qkeys[2], pkeys[2])
    for j, leg in enumerate(ORDER[2:]):
        target = root[:3, :3].T@(rear_world[j]-root[:3, 3])
        qkeys[2, 6+3*j:9+3*j] = kin.solve_leg(leg, target, qkeys[2, 6+3*j:9+3*j])

    t = np.linspace(0, times[-1], round(times[-1]/.02)+1)
    q = interpolate(times, qkeys, t)
    pitch = interpolate(times, pkeys, t)
    roots, feet, gravity, com = [], [], [], []
    for i, ti in enumerate(t):
        root = root_for(q[i], pitch[i])
        if ti <= times[2]:
            for j, leg in enumerate(ORDER[2:]):
                target = root[:3, :3].T@(rear_world[j]-root[:3, 3])
                q[i, 6+3*j:9+3*j] = kin.solve_leg(leg, target, q[i, 6+3*j:9+3*j])
        fk = kin.fk(dict(zip(LEG_NAMES, q[i])), root)
        roots.append(root[:3, 3])
        feet.append([fk[n][:3, 3] for n in FOOT_NAMES])
        gravity.append(root[:3, :3].T@np.array([0, 0, -1.]))
        com.append(kin.com(fk))
    roots, feet, gravity, com = map(np.asarray, (roots, feet, gravity, com))
    feet_b = []
    for i in range(len(t)):
        R = transform(rpy=(0, pitch[i], 0))[:3, :3]
        feet_b.append((feet[i]-roots[i])@R)
    feet_b = np.asarray(feet_b)
    velocity = np.gradient(roots, .02, axis=0)
    qd = np.gradient(q, .02, axis=0)
    pitch_rate = np.gradient(pitch, .02)
    velocity[[0, -1]], qd[[0, -1]], pitch_rate[[0, -1]] = 0, 0, 0
    velocity_b = np.array([transform(rpy=(0, p, 0))[:3, :3].T@v
                           for p, v in zip(pitch, velocity)])
    angular_velocity = np.zeros((len(t), 3))
    angular_velocity[:, 1] = pitch_rate
    # 连续接触目标：离地 2 cm 后 rear target=0，避免二值边界突然改变奖励。
    contact = np.clip(1-(feet[:, :, 2]-radius)/.02, 0, 1)
    limits = np.array([kin.limits[n] for n in LEG_NAMES])
    lo, hi = limits.mean(1)-np.diff(limits)[:, 0]*.475, limits.mean(1)+np.diff(limits)[:, 0]*.475
    checks = dict(joint_limits=bool(np.all((q >= lo-1e-6)&(q <= hi+1e-6))),
                  no_axle_penetration=bool(np.min(feet[:, :, 2]-radius) >= -.002),
                  front_contact=bool(np.max(np.abs(feet[:, :2, 2]-radius)) < .002),
                  finite=bool(np.isfinite(q).all() and np.isfinite(roots).all()))
    if not all(checks.values()):
        raise ValueError(f"Reference failed geometric checks: {checks}")
    np.savez(MOTION, time=t.astype("f4"), joints=q.astype("f4"),
             joint_velocity=qd.astype("f4"), root_position=roots.astype("f4"),
             gravity=gravity.astype("f4"), root_velocity=velocity_b.astype("f4"),
             angular_velocity=angular_velocity.astype("f4"), feet_body=feet_b.astype("f4"),
             contact=contact.astype("f4"), wheel_velocity=np.zeros((len(t), 4), dtype="f4"))
    report = dict(method="manual keyframes + URDF FK/IK; no physics or policy training",
                  checks=checks, source_urdf_sha256=hashlib.sha256(URDF.read_bytes()).hexdigest(),
                  motion_sha256=hashlib.sha256(MOTION.read_bytes()).hexdigest(),
                  initial_height_m=float(roots[0, 2]), duration_s=float(t[-1]),
                  max_joint_speed_rad_s=float(abs(qd).max()),
                  min_rear_clearance_m=float((feet[:, 2:, 2]-radius).min()),
                  final_com_minus_front_axle_m=(com[-1]-feet[-1, :2].mean(0)).tolist(),
                  limitation="FK is not a dynamics proof. Run Isaac zero-action and policy rollouts.")
    (ROOT/"motions/reference_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                       encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    generate()
