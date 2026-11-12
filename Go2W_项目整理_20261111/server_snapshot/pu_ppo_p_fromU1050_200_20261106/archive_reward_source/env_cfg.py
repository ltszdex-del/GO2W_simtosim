"""robot_lab 场景 + 本项目观察、动作和奖励。环境之间过滤碰撞。"""
from collections import deque
import torch
import isaaclab.envs.mdp as lab_mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm, ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm, SceneEntityCfg
from isaaclab.utils import configclass
from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_go2w.flat_env_cfg import UnitreeGo2WFlatEnvCfg
from . import mdp
from .profile import LEG_NAMES, load_profile, phase_times
from .robot_cfg import make_robot_cfg


@configclass
class ActionsCfg:
    legs = mdp.LieDownLegActionCfg(asset_name="robot")
    wheels = mdp.WheelHoldActionCfg(asset_name="robot")


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 共 95 维；包含滤波器的目标误差和速度，避免控制历史对网络不可见。
        linear_velocity = ObsTerm(func=lab_mdp.base_lin_vel, scale=2.0)
        angular_velocity = ObsTerm(func=lab_mdp.base_ang_vel, scale=0.25)
        gravity = ObsTerm(func=lab_mdp.projected_gravity)
        joint_relative = ObsTerm(func=mdp.joint_relative)
        joint_velocity = ObsTerm(func=lab_mdp.joint_vel_rel, scale=0.05)
        previous_action = ObsTerm(func=lab_mdp.last_action)
        xy = ObsTerm(func=mdp.xy_offset)
        height = ObsTerm(func=mdp.root_height)
        reference_error = ObsTerm(func=mdp.reference_error)
        phase = ObsTerm(func=mdp.phase)
        wheel_support = ObsTerm(func=mdp.support)
        body_contact = ObsTerm(func=mdp.body_contact)
        controller_error = ObsTerm(func=mdp.controller_error)
        controller_velocity = ObsTerm(func=mdp.controller_velocity, scale=0.25)
        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    trajectory = RewTerm(func=mdp.reference_tracking, weight=2.0)
    height = RewTerm(func=mdp.height_tracking, weight=2.0)
    final_pose = RewTerm(func=mdp.final_pose_reward, weight=4.0)
    settled_hold = RewTerm(func=mdp.held_reward, weight=3.0)
    orientation = RewTerm(func=mdp.orientation_penalty, weight=-2.0)
    vertical_spin = RewTerm(func=mdp.vertical_spin, weight=-2.0)
    approach_speed = RewTerm(func=mdp.approach_speed, weight=-1.0)
    impact_peak = RewTerm(func=mdp.impact_peak, weight=-1.0)
    body_impulse = RewTerm(func=mdp.body_impulse, weight=-0.5)
    early_body_contact = RewTerm(func=mdp.early_body_contact, weight=-2.0)
    head_contact = RewTerm(func=mdp.head_contact, weight=-2.0)
    wheel_slip = RewTerm(func=mdp.slip, weight=-0.3)
    drift = RewTerm(func=mdp.drift, weight=-1.0)
    joint_motion = RewTerm(func=mdp.joint_motion, weight=-0.01)
    action_change = RewTerm(func=lab_mdp.action_rate_l2, weight=-0.01)
    joint_limits = RewTerm(func=lab_mdp.joint_pos_limits, weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_NAMES)})
    # 趴稳后鼓励降低持续输出；仍不等价于已经验证能切断电源。
    late_effort = RewTerm(func=mdp.late_effort, weight=-0.002)
    failure = RewTerm(func=lab_mdp.is_terminated, weight=-5.0)


@configclass
class EventsCfg:
    reset = EventTerm(func=mdp.reset_standing, mode="reset")


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=lab_mdp.time_out, time_out=True)
    failure = DoneTerm(func=mdp.failure)


@configclass
class LieDownEnvCfg(UnitreeGo2WFlatEnvCfg):
    profile: dict = None
    random_reset: bool = True
    record_episodes: bool = False
    def __post_init__(self):
        super().__post_init__()
        self.profile = self.profile or load_profile()
        self.scene.robot = make_robot_cfg(self.profile)
        self.scene.num_envs, self.scene.env_spacing = 512, 2.5
        self.scene.replicate_physics = True
        self.scene.filter_collisions = True
        self.scene.height_scanner = self.scene.height_scanner_base = None
        self.scene.contact_forces.update_period = 0.0
        self.scene.contact_forces.history_length = 4
        self.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3))
        self.scene.sky_light.spawn = sim_utils.DomeLightCfg(intensity=1500.0)
        self.actions, self.observations = ActionsCfg(), ObservationsCfg()
        self.rewards, self.events, self.terminations = RewardsCfg(), EventsCfg(), TerminationsCfg()
        self.curriculum = None
        self.commands.base_velocity.ranges.lin_vel_x = (0, 0)
        self.commands.base_velocity.ranges.lin_vel_y = (0, 0)
        self.commands.base_velocity.ranges.ang_vel_z = (0, 0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.debug_vis = False
        self.episode_length_s = phase_times(self.profile)[-1]
        self.decimation, self.sim.dt = 4, 0.005
        self.sim.render_interval = 4
        self.sim.physics_material = self.scene.terrain.physics_material
        self.viewer.eye, self.viewer.lookat = (2.3, 2.3, 1.1), (0, 0, 0.2)


class LieDownEnv(ManagerBasedRLEnv):
    def __init__(self, *args, **kwargs):
        self.episode_records = deque(maxlen=10000)
        super().__init__(*args, **kwargs)

    def _reset_idx(self, env_ids):
        if hasattr(self, "_liedown") and len(env_ids):
            s = mdp.state(self)
            valid = self.episode_length_buf[env_ids]>0
            ids = env_ids[valid]
            if len(ids):
                failed = self.termination_manager.terminated[ids]
                # 必须在回合终点仍连续趴稳，且整回合冲击合格，才计成功。
                ok = (s["hold"][ids]>=self.cfg.profile["success"]["hold_s"]) & ~failed
                stats = {
                    "Metrics/liedown/success_rate": ok.float().mean(),
                    "Metrics/liedown/best_hold_s": s["best_hold"][ids].mean(),
                    "Metrics/liedown/body_peak_bw": s["body_peak"][ids].mean(),
                    "Metrics/liedown/touchdown_speed_m_s": s["touchdown_speed"][ids].mean(),
                    "Metrics/liedown/peak_load_rate_bw_s": s["peak_load_rate"][ids].mean(),
                }
                if self.cfg.record_episodes:
                    rows = torch.stack((ok.float(), failed.float(), s["best_hold"][ids], s["body_peak"][ids], s["touchdown_speed"][ids]), 1).cpu().tolist()
                    for success, failure, hold, peak, speed in rows:
                        self.episode_records.append(dict(success=bool(success), failed=bool(failure), best_hold_s=hold, body_peak_bw=peak, touchdown_speed_m_s=speed))
                super()._reset_idx(env_ids)
                self.extras.setdefault("log", {}).update(stats)
                return
        super()._reset_idx(env_ids)
