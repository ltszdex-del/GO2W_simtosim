"""Independent single-robot scene. Imported only after the Kit app starts."""
from pathlib import Path
import math
import hashlib
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

ORDER = ('FR','FL','RR','RL')
LEGS = [f'{s}_{j}_joint' for s in ORDER for j in ('hip','thigh','calf')]
WHEELS = [f'{s}_foot_joint' for s in ORDER]

@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path='/World/ground', spawn=sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.8,restitution=0.)))
    light = AssetBaseCfg(prim_path='/World/Light', spawn=sim_utils.DomeLightCfg(intensity=1800.,color=(.9,.9,.9)))
    robot = ArticulationCfg(prim_path='{ENV_REGEX_NS}/Robot',
        spawn=sim_utils.UrdfFileCfg(asset_path=str(Path(__file__).parent/'assets/go2w_description/urdf/go2w_description.urdf'),
            fix_base=False,merge_fixed_joints=True,activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False,linear_damping=0.,angular_damping=0.,max_depenetration_velocity=1.),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True,solver_position_iteration_count=8,solver_velocity_iteration_count=4)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.,0.,.412),joint_pos={'.*_hip_joint':0.,'.*_thigh_joint':.8,'.*_calf_joint':-1.5,'.*_foot_joint':0.},joint_vel={'.*':0.}),
        soft_joint_pos_limit_factor=1.,
        actuators={'motors':ImplicitActuatorCfg(joint_names_expr=['.*'],stiffness=0.,damping=0.,friction=0.,
            dynamic_friction=0.,viscous_friction=0.,armature=0.,joint_effort_limit=23.5,
            joint_velocity_limit={'.*_hip_joint':30.1,'.*_thigh_joint':30.1,'.*_calf_joint':20.07,'.*_foot_joint':30.1})})
    contacts = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*',update_period=0.,history_length=4)

@configclass
class EnvCfg(DirectRLEnvCfg):
    decimation=4
    episode_length_s=1e9
    action_space=16
    observation_space=1
    state_space=0
    sim=sim_utils.SimulationCfg(dt=.005,render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.8,restitution=0.))
    scene=SceneCfg(num_envs=1,env_spacing=4.,replicate_physics=True,filter_collisions=True)

