from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim
from config import URDF, ACTOR_DIM, CONTROL

@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path='/World/ground',spawn=sim.GroundPlaneCfg(
        physics_material=sim.RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.8,restitution=0.)))
    light = AssetBaseCfg(prim_path='/World/light',spawn=sim.DomeLightCfg(intensity=1800.))
    robot = ArticulationCfg(prim_path='{ENV_REGEX_NS}/Robot',
        spawn=sim.UrdfFileCfg(asset_path=str(URDF),fix_base=False,merge_fixed_joints=True,
            activate_contact_sensors=True,
            rigid_props=sim.RigidBodyPropertiesCfg(disable_gravity=False,linear_damping=0.,
                angular_damping=0.,max_depenetration_velocity=1.),
            articulation_props=sim.ArticulationRootPropertiesCfg(enabled_self_collisions=True,
                solver_position_iteration_count=8,solver_velocity_iteration_count=4)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.,0.,.412),
            joint_pos={'.*_hip_joint':0.,'.*_thigh_joint':.8,'.*_calf_joint':-1.5,'.*_foot_joint':0.},
            joint_vel={'.*':0.}),soft_joint_pos_limit_factor=.95,
        actuators={'motors':ImplicitActuatorCfg(joint_names_expr=['.*'],stiffness=0.,
            damping={'.*_hip_joint':0.,'.*_thigh_joint':0.,'.*_calf_joint':0.,'.*_foot_joint':CONTROL['wheel_lock_damping']},
            friction=0.,dynamic_friction=0.,viscous_friction=0.,armature=0.,
            joint_effort_limit=CONTROL['effort'],joint_velocity_limit={'.*_hip_joint':30.1,
                '.*_thigh_joint':30.1,'.*_calf_joint':20.07,'.*_foot_joint':30.1})})
    contacts = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*',update_period=0.,history_length=4)
    front_right_rear = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/FR_foot',
        filter_prim_paths_expr=['{ENV_REGEX_NS}/Robot/RR_foot','{ENV_REGEX_NS}/Robot/RL_foot'],
        update_period=0.,history_length=4)
    front_left_rear = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/FL_foot',
        filter_prim_paths_expr=['{ENV_REGEX_NS}/Robot/RR_foot','{ENV_REGEX_NS}/Robot/RL_foot'],
        update_period=0.,history_length=4)

@configclass
class EnvCfg(DirectRLEnvCfg):
    decimation=4
    episode_length_s=10.
    action_space=12
    observation_space=ACTOR_DIM
    state_space=0
    training: bool=True
    sim=sim.SimulationCfg(dt=.005,render_interval=4,
        physics_material=sim.RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.8,restitution=0.))
    scene=SceneCfg(num_envs=1024,env_spacing=8.,replicate_physics=True,filter_collisions=True)
