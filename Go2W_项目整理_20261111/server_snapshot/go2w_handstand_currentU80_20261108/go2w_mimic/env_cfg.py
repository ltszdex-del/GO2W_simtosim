"""Isaac Lab DirectRLEnv 的场景配置，只在 AppLauncher 启动后导入。

目标版本：Isaac Lab 3.0 + Isaac Sim 6.2（默认 PhysX 后端）。
仍使用 Isaac/PhysX 训练。MuJoCo 只用于离线参考检查与后续验收。
这里直接配置同一份 URDF，因此新项目无需依赖 robot_lab 的训练管理器。
"""
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from .settings import *


@configclass
class MimicSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.8, dynamic_friction=.8,
            restitution=0.0, friction_combine_mode="average")))
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        # Isaac Lab 3.0 / Isaac Sim 6.2：URDF 导入器不再接受
        # `replace_cylinders_with_capsules`（柱面几何已按默认规则处理）和
        # `joint_drive`（零增益由下方 ImplicitActuatorCfg 的刚度/阻尼=0 保证，
        # 显式力矩由 env._apply_action 每物理步发送，无需在导入期预设驱动）。
        spawn=sim_utils.UrdfFileCfg(asset_path=str(URDF), fix_base=False, merge_fixed_joints=True,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False,
                linear_damping=0., angular_damping=0., max_depenetration_velocity=1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True,
                solver_position_iteration_count=8, solver_velocity_iteration_count=4)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0., 0., .412),
            joint_pos={".*_hip_joint": 0., ".*_thigh_joint": .8,
                       ".*_calf_joint": -1.5, ".*_foot_joint": 0.},
            joint_vel={".*": 0.}),
        soft_joint_pos_limit_factor=CONTROL["soft_limit"],
        # PhysX 的隐式驱动增益为 0。env._apply_action 每个物理步计算并发送力矩。
        # Isaac Lab 3.0：构造期关节限幅字段由 effort_limit_sim/velocity_limit_sim
        # 更名为 joint_effort_limit/joint_velocity_limit。
        actuators={"motors": ImplicitActuatorCfg(joint_names_expr=[".*"],
            stiffness=0., damping=0., friction=PHYSICS["friction"],
            dynamic_friction=PHYSICS["dynamic_friction"],
            viscous_friction=PHYSICS["viscous_friction"], armature=PHYSICS["armature"],
            joint_effort_limit=CONTROL["effort"],
            joint_velocity_limit=PHYSICS["joint_velocity_limits"])})
    contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*",
                                update_period=0., history_length=DECIMATION)
    front_right_rear = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/FR_foot",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/RR_foot","{ENV_REGEX_NS}/Robot/RL_foot"],
        update_period=0.,history_length=DECIMATION)
    front_left_rear = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/FL_foot",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/RR_foot","{ENV_REGEX_NS}/Robot/RL_foot"],
        update_period=0.,history_length=DECIMATION)
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(
        intensity=1800., color=(.9,.9,.9)))


@configclass
class MimicEnvCfg(DirectRLEnvCfg):
    decimation = DECIMATION
    episode_length_s = 8.0
    action_space = ACTION_DIM
    observation_space = TEACHER_DIM
    state_space = 0
    motion_path: str = str(MOTION)
    randomize: bool = True
    domain_level: float = 0.2
    history_length: int = HISTORY
    sim = sim_utils.SimulationCfg(dt=DT, render_interval=DECIMATION,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.8, dynamic_friction=.8,
            restitution=0.0, friction_combine_mode="average"))
    scene = MimicSceneCfg(num_envs=1024, env_spacing=4.0,
        replicate_physics=True, filter_collisions=True)

