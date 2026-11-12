"""站立→趴卧：物理动作、观察、柔和触地奖励与完整回合验收。

世界坐标 Z 为地面法向。只有 reset 写机器人实际姿态；过程中只下发关节目标。
网络输出 12 个小幅残差，不能直接传送机身或瞬间重置到趴卧。
"""
import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz
from .profile import LEG_NAMES, WHEEL_NAMES, phase_times


def state(env):
    if not hasattr(env, "_liedown"):
        robot = env.scene["robot"]
        sensor = env.scene["contact_forces"]
        legs, _ = robot.find_joints(LEG_NAMES, preserve_order=True)
        wheels, _ = robot.find_joints(WHEEL_NAMES, preserve_order=True)
        feet, _ = sensor.find_bodies(".*_foot")
        body, _ = sensor.find_bodies("base")
        head, _ = sensor.find_bodies("Head.*")
        p = env.cfg.profile
        s = dict(legs=legs, wheels=wheels, feet=feet, body=body, head=head)
        for name in ("stand", "crouch", "prone"):
            s[name] = torch.tensor([p["poses"][name][n] for n in LEG_NAMES], device=env.device)
        # 用实际 USD 的质量归一化力，避免把 N 和整机重量倍数混用。
        s["weight"] = robot.root_physx_view.get_masses().to(env.device).sum(dim=1) * 9.81
        for key in ("hold", "best_hold", "body_peak", "touchdown_speed", "previous_vz", "previous_force", "peak_load_rate"):
            s[key] = torch.zeros(env.num_envs, device=env.device)
        s["previous_contact"] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        s["last_step"] = -1
        env._liedown = s
    return env._liedown


def elapsed(env):
    return env.episode_length_buf.float() * env.step_dt


def smooth(x):
    """五次插值：段首、段尾的速度和加速度均为零。"""
    x = x.clamp(0, 1)
    return x**3 * (10 - 15*x + 6*x*x)


def blends(env, time=None):
    a, b, c, _, _ = phase_times(env.cfg.profile)
    t = elapsed(env) if time is None else time
    return smooth((t-a)/(b-a)), smooth((t-b)/(c-b))


def reference(env, time=None):
    s = state(env)
    a, b = blends(env, time)
    return s["stand"] + a[:, None]*(s["crouch"]-s["stand"]) + b[:, None]*(s["prone"]-s["crouch"])


def height_reference(env):
    h = env.cfg.profile["heights_m"]
    a, b = blends(env)
    return h["stand"] + a*(h["crouch"]-h["stand"]) + b*(h["prone"]-h["crouch"])


def gravity_reference(env, final=False):
    pitch = torch.full((env.num_envs,), env.cfg.profile["prone_pitch_rad"], device=env.device)
    if not final:
        pitch = pitch * blends(env)[1]
    return torch.stack((pitch.sin(), torch.zeros_like(pitch), -pitch.cos()), dim=-1)


class LieDownLegAction(ActionTerm):
    """参考姿态 + 残差。限制指令速度、加速度，实际触地冲击仍由奖励约束。"""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.ids, _ = self._asset.find_joints(LEG_NAMES, preserve_order=True)
        self._raw = torch.zeros((self.num_envs, 12), device=self.device)
        self._target = self._asset.data.default_joint_pos[:, self.ids].clone()
        self.velocity = torch.zeros_like(self._target)

    @property
    def action_dim(self): return 12
    @property
    def raw_actions(self): return self._raw
    @property
    def processed_actions(self): return self._target

    def process_actions(self, actions):
        self._raw[:] = actions.clamp(-1, 1)
        p, dt = self._env.cfg.profile, self._env.step_dt
        gate = (elapsed(self._env) >= p["timing_s"]["stand"]).float()[:, None]
        desired = reference(self._env, elapsed(self._env)+dt) + gate*p["control"]["residual_scale_rad"]*self._raw
        lim = self._asset.data.soft_joint_pos_limits[:, self.ids]
        desired = desired.clamp(lim[..., 0], lim[..., 1])
        delta = desired-self._target
        vmax, amax = p["control"]["target_rate_limit_rad_s"], p["control"]["target_acceleration_rad_s2"]
        v = (delta/dt).clamp(-vmax, vmax)
        self.velocity += (v-self.velocity).clamp(-amax*dt, amax*dt)
        self._target = (self._target+self.velocity*dt).clamp(lim[..., 0], lim[..., 1])

    def apply_actions(self):
        self._asset.set_joint_position_target(self._target, joint_ids=self.ids)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0
        self.velocity[ids] = 0
        self._target[ids] = self._asset.data.joint_pos[ids][:, self.ids]


@configclass
class LieDownLegActionCfg(ActionTermCfg):
    class_type: type = LieDownLegAction


class WheelHoldAction(ActionTerm):
    """四轮不参与网络输出，以有限 PD 力矩保持轮轴；不改关节硬限位。"""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.ids, _ = self._asset.find_joints(WHEEL_NAMES, preserve_order=True)
        self._raw = torch.zeros((self.num_envs, 0), device=self.device)
        self._target = torch.zeros((self.num_envs, 4), device=self.device)
    @property
    def action_dim(self): return 0
    @property
    def raw_actions(self): return self._raw
    @property
    def processed_actions(self): return self._target
    def process_actions(self, actions): pass
    def apply_actions(self):
        self._asset.set_joint_position_target(self._target, joint_ids=self.ids)
        self._asset.set_joint_velocity_target(torch.zeros_like(self._target), joint_ids=self.ids)
    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._target[ids] = self._asset.data.joint_pos[ids][:, self.ids]


@configclass
class WheelHoldActionCfg(ActionTermCfg):
    class_type: type = WheelHoldAction


def reset_standing(env, env_ids):
    s, r = state(env), env.scene["robot"]
    ids = env_ids if env_ids is not None else torch.arange(env.num_envs, device=env.device)
    p = env.cfg.profile
    root = r.data.default_root_state[ids].clone()
    root[:, :3] += env.scene.env_origins[ids]
    root[:, 7:13] = 0
    q = r.data.default_joint_pos[ids].clone()
    if env.cfg.random_reset:
        q[:, s["legs"]] += (2*torch.rand(len(ids), 12, device=env.device)-1)*p["randomization"]["joint_rad"]
        angles = (2*torch.rand(len(ids), 2, device=env.device)-1)*p["randomization"]["tilt_rad"]
        root[:, 3:7] = quat_from_euler_xyz(angles[:, 0], angles[:, 1], torch.zeros(len(ids), device=env.device))
    limits = r.data.soft_joint_pos_limits[ids]
    q = q.clamp(limits[..., 0], limits[..., 1])
    r.write_root_pose_to_sim(root[:, :7], env_ids=ids)
    r.write_root_velocity_to_sim(root[:, 7:13], env_ids=ids)
    r.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
    for key in ("hold", "best_hold", "body_peak", "touchdown_speed", "previous_vz", "previous_force", "peak_load_rate"):
        s[key][ids] = 0
    s["previous_contact"][ids] = False


def root_height(env):
    return (env.scene["robot"].data.root_pos_w[:, 2]-env.scene.env_origins[:, 2])[:, None]
def xy_offset(env):
    return env.scene["robot"].data.root_pos_w[:, :2]-env.scene.env_origins[:, :2]
def joint_relative(env):
    return env.scene["robot"].data.joint_pos[:, state(env)["legs"]]-state(env)["stand"]
def reference_error(env):
    return env.scene["robot"].data.joint_pos[:, state(env)["legs"]]-reference(env)
def controller_error(env):
    return env.action_manager.get_term("legs").processed_actions-env.scene["robot"].data.joint_pos[:, state(env)["legs"]]
def controller_velocity(env): return env.action_manager.get_term("legs").velocity
def phase(env):
    a, _, c, d, _ = phase_times(env.cfg.profile)
    return torch.stack(((elapsed(env)-a).clamp(0, c-a)/(c-a), (elapsed(env)>=d).float()), dim=-1)
def support(env):
    return (env.scene["contact_forces"].data.net_forces_w[:, state(env)["feet"]].norm(dim=-1)>2).float()
def body_contact(env):
    return (env.scene["contact_forces"].data.net_forces_w[:, state(env)["body"]].norm(dim=-1).amax(1)>2).float()[:, None]


def measures(env):
    s, r = state(env), env.scene["robot"]
    forces = env.scene["contact_forces"].data.net_forces_w_history.norm(dim=-1)
    body_peak = forces[:, :, s["body"]].amax(dim=(1, 2))/s["weight"]
    body_mean = forces[:, :, s["body"]].mean(dim=(1, 2))/s["weight"]
    return dict(height=root_height(env)[:, 0], speed=r.data.root_lin_vel_w.norm(dim=1),
        angular_speed=r.data.root_ang_vel_w.norm(dim=1), vz=r.data.root_lin_vel_w[:, 2],
        body_peak=body_peak, body_mean=body_mean,
        foot_peak=forces[:, :, s["feet"]].amax(dim=1)/s["weight"][:, None],
        head_force=forces[:, :, s["head"]].amax(dim=(1, 2)),
        pose_error=(r.data.joint_pos[:, s["legs"]]-s["prone"]).square().mean(1).sqrt(),
        tilt=torch.acos((r.data.projected_gravity_b*gravity_reference(env, final=True)).sum(1).clamp(-1, 1)))


def update_status(env):
    """每个策略步只计一次连续达标时间，记录整回合最坏冲击，防止摔下去后刷成功。"""
    s = state(env)
    if s["last_step"] == env.common_step_counter: return
    m, p = measures(env), env.cfg.profile["success"]
    contact = m["body_mean"] > 0.02
    touch = contact & ~s["previous_contact"]
    s["touchdown_speed"] = torch.maximum(s["touchdown_speed"], torch.where(touch, (-s["previous_vz"]).clamp(min=0), 0))
    s["body_peak"] = torch.maximum(s["body_peak"], m["body_peak"])
    rate = ((m["body_mean"]-s["previous_force"])/env.step_dt).clamp(min=0)
    s["peak_load_rate"] = torch.maximum(s["peak_load_rate"], rate)
    good = ((elapsed(env)>=phase_times(env.cfg.profile)[3])
        & ((m["height"]-env.cfg.profile["heights_m"]["prone"]).abs()<p["height_error_m"])
        & (m["pose_error"]<p["joint_rms_rad"]) & (m["tilt"]<p["tilt_rad"])
        & (m["speed"]<p["speed_m_s"]) & (m["angular_speed"]<p["angular_rad_s"])
        & (env.scene["robot"].data.joint_vel[:, s["legs"]].abs().amax(1)<p["joint_speed_rad_s"])
        & (m["body_mean"]>p["body_support_bw"]) & (m["head_force"]<3)
        & (s["body_peak"]<p["max_body_peak_bw"]) & (s["touchdown_speed"]<p["max_touchdown_speed_m_s"]))
    s["hold"] = torch.where(good, s["hold"]+env.step_dt, 0)
    s["best_hold"] = torch.maximum(s["best_hold"], s["hold"])
    s["previous_contact"] = contact
    s["previous_vz"] = m["vz"].clone()
    s["previous_force"] = m["body_mean"].clone()
    s["last_step"] = env.common_step_counter


def reference_tracking(env): return torch.exp(-reference_error(env).square().mean(1)/0.025)
def height_tracking(env): return torch.exp(-((root_height(env)[:, 0]-height_reference(env))/0.04).square())
def orientation_penalty(env):
    return (env.scene["robot"].data.projected_gravity_b-gravity_reference(env)).square().sum(1)
def final_pose_reward(env):
    m = measures(env)
    return (elapsed(env)>=phase_times(env.cfg.profile)[2]).float()*torch.exp(
        -(m["pose_error"]/0.15).square()-(m["speed"]/0.1).square()-(m["angular_speed"]/0.3).square())
def held_reward(env):
    update_status(env)
    return (state(env)["hold"]>=env.cfg.profile["success"]["hold_s"]).float()
def approach_speed(env):
    m, p = measures(env), env.cfg.profile["impact"]
    return (m["height"]<p["near_ground_height_m"]).float()*((-m["vz"]-p["descent_m_s"]).clamp(min=0)/0.1).square()
def impact_peak(env):
    m, p = measures(env), env.cfg.profile["impact"]
    return (m["body_peak"]-p["body_peak_bw"]).clamp(min=0).square() + (m["foot_peak"]-p["foot_peak_bw"]).clamp(min=0).square().sum(1)
def body_impulse(env):
    # 四个 5ms 样本覆盖 20ms 控制周期；约束超出允许支撑力的接触冲量。
    s, p = state(env), env.cfg.profile["impact"]
    f = env.scene["contact_forces"].data.net_forces_w_history[:, :, s["body"]].norm(dim=-1).sum(2)/s["weight"][:, None]
    return ((f-p["body_peak_bw"]).clamp(min=0).sum(1)*env.physics_dt/0.02).square()
def joint_motion(env): return env.scene["robot"].data.joint_vel[:, state(env)["legs"]].square().mean(1)
def vertical_spin(env): return env.scene["robot"].data.root_ang_vel_w[:, 2].square()
def slip(env):
    s = state(env)
    r = env.scene["robot"]
    ids, _ = r.find_bodies(".*_foot")
    return (r.data.body_lin_vel_w[:, ids, :2].square().sum(2)*support(env)).sum(1)
def late_effort(env):
    return (elapsed(env)>=phase_times(env.cfg.profile)[2]).float()*env.scene["robot"].data.applied_torque[:, state(env)["legs"]].square().mean(1)
def early_body_contact(env): return (elapsed(env)<phase_times(env.cfg.profile)[1]).float()*body_contact(env)[:, 0]
def head_contact(env): return (measures(env)["head_force"]-2).clamp(min=0).clamp(max=100)/10
def drift(env): return xy_offset(env).square().sum(1)


def failure(env):
    update_status(env)
    r, p = env.scene["robot"], env.cfg.profile["failure"]
    tilt = torch.acos((-r.data.projected_gravity_b[:, 2]).clamp(-1, 1))
    return ((tilt>p["tilt_rad"]) | (xy_offset(env).norm(dim=1)>p["drift_m"])
        | (root_height(env)[:, 0]<p["minimum_height_m"])
        | ~torch.isfinite(r.data.root_state_w).all(1) | ~torch.isfinite(r.data.joint_pos).all(1))
