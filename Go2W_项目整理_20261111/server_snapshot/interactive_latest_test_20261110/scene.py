"""Independent single-robot scene. Imported only after the Kit app starts."""
from pathlib import Path
import math
import hashlib
from commands import world_to_yaw_frame
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
        soft_joint_pos_limit_factor=.95,
        actuators={'motors':ImplicitActuatorCfg(joint_names_expr=['.*'],stiffness=0.,damping=0.,friction=0.,
            dynamic_friction=0.,viscous_friction=0.,armature=0.,joint_effort_limit=23.5,
            joint_velocity_limit={'.*_hip_joint':30.1,'.*_thigh_joint':30.1,'.*_calf_joint':20.07,'.*_foot_joint':30.1})})
    contacts = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*',update_period=0.,history_length=4)
    front_right_rear = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/FR_foot',
        filter_prim_paths_expr=['{ENV_REGEX_NS}/Robot/RR_foot','{ENV_REGEX_NS}/Robot/RL_foot'],update_period=0.,history_length=4)
    front_left_rear = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/FL_foot',
        filter_prim_paths_expr=['{ENV_REGEX_NS}/Robot/RR_foot','{ENV_REGEX_NS}/Robot/RL_foot'],update_period=0.,history_length=4)

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

class StandingEnv(DirectRLEnv):
    def __init__(self,cfg,control):
        self.control=control
        super().__init__(cfg)
        self._physics_handles_decimation=False
        self.legs,_=self.robot.find_joints(LEGS,preserve_order=True)
        self.wheels,_=self.robot.find_joints(WHEELS,preserve_order=True)
        assert len(self.legs)==12 and len(self.wheels)==4
        self.ids=self.legs+self.wheels
        self.target=torch.tensor(control.get('initial_leg_target',[0.,.8,-1.5]*4),device=self.device).unsqueeze(0)
        self.goal=self.target.clone()
        self.wheel_target=torch.zeros((1,4),device=self.device)
        self.wheel_goal=self.wheel_target.clone()
        self.torque=torch.zeros((1,16),device=self.device)
        self.pd_updates=0
        self.handstand_context=None
    def set_handstand_context(self,ctx):
        self.set_motor_context(ctx,'handstand',0.)
    def set_motor_context(self,ctx,mode,wheel_damping):
        damping=torch.full((1,4),wheel_damping,device=self.device)
        self.robot.write_joint_damping_to_sim(damping,joint_ids=self.wheels)
        assert torch.allclose(self.robot.data.joint_damping.torch[:,self.wheels],damping)
        self.handstand_context=ctx
        if ctx is None:
            self.goal=self.target.clone()
            self.wheel_target.zero_();self.wheel_goal.zero_()
        print('[MODE]',mode,'wheel damping',wheel_damping,'source controller at 200Hz',flush=True)
    def _setup_scene(self):
        self.robot=self.scene.articulations['robot']
        self.contacts=self.scene.sensors['contacts']
    def _pre_physics_step(self,actions):
        self.pd_updates=0
        if self.handstand_context is not None:
            self.target=self.handstand_context.target
            self.wheel_target=self.handstand_context.wheel_target
            return
        rate=self.control['leg_target_rate_rad_s']*self.step_dt
        self.target+=(self.goal-self.target).clamp(-rate,rate)
        rate=self.control['wheel_target_rate_rad_s2']*self.step_dt
        self.wheel_target+=(self.wheel_goal-self.wheel_target).clamp(-rate,rate)
    def _apply_action(self):
        if self.handstand_context is not None:
            self.handstand_context._apply_action()
            self.torque=self.handstand_context.torque
            self.pd_updates+=1
            return
        d,c=self.robot.data,self.control
        q=d.joint_pos.torch[:,self.legs]
        dq=d.joint_vel.torch[:,self.ids]
        self.torque=torch.cat((c['kp']*(self.target-q)-c['kd']*dq[:,:12],c['wheel_kd']*(self.wheel_target-dq[:,12:])),dim=-1)
        self.robot.set_joint_velocity_target(torch.zeros_like(self.wheel_target),joint_ids=self.wheels)
        self.torque=self.torque.clamp(-c['effort_limit_nm'],c['effort_limit_nm'])
        self.robot.set_joint_effort_target(self.torque,joint_ids=self.ids)
        self.pd_updates+=1
    def set_motor_targets(self,command):
        leg=torch.as_tensor(command['leg_pos_rad'],device=self.device,dtype=self.target.dtype).reshape(1,12)
        wheel=torch.as_tensor(command['wheel_vel_rad_s'],device=self.device,dtype=self.target.dtype).reshape(1,4)
        if not torch.isfinite(leg).all() or not torch.isfinite(wheel).all():raise ValueError('Nonfinite policy output')
        limits=self.robot.data.soft_joint_pos_limits.torch[:,self.legs]
        self.goal=leg.clamp(limits[...,0],limits[...,1])
        speed=self.control['wheel_speed_limit_rad_s']
        self.wheel_goal=wheel.clamp(-speed,speed)
    def standing_target(self):
        self.set_motor_targets(dict(leg_pos_rad=[0.,.8,-1.5]*4,wheel_vel_rad_s=[0.]*4))
    def safe_stop(self):
        self.wheel_goal.zero_()
        if self.handstand_context is not None:
            self.handstand_context.wheel_target.zero_()
            self.handstand_context.target_velocity.zero_()
    def snapshot(self):
        d=self.robot.data
        x,y,z,w=d.root_quat_w.torch[0].tolist() # installed Isaac Lab 3 convention: XYZW
        yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        velocity=d.root_lin_vel_w.torch[0].tolist()
        actual=world_to_yaw_frame(velocity[0],velocity[1],yaw)+[float(d.root_ang_vel_w.torch[0,2])]
        return dict(yaw_rad=yaw,actual_velocity_yaw_m_s_rad_s=actual,
            joint_pos_rad=d.joint_pos.torch[:,self.ids].clone(),
            joint_vel_rad_s=d.joint_vel.torch[:,self.ids].clone(),
            projected_gravity_b=d.projected_gravity_b.torch.clone(),
            angular_velocity_b=d.root_ang_vel_b.torch.clone(),
            root_position_w=d.root_pos_w.torch.clone(),
            leg_target_rad=self.target.clone(),wheel_target_rad_s=self.wheel_target.clone())
    def contract(self,mode=None):
        asset=Path(self.control.get('robot_asset_root',Path(__file__).parent/'assets/go2w_description'))
        files=sorted(p for p in asset.rglob('*') if p.is_file() and p.suffix.lower() in ('.urdf','.stl','.obj','.dae','.mtl','.png','.jpg'))
        digest=hashlib.sha256()
        for p in files:
            digest.update(p.relative_to(asset).as_posix().encode()+b'\0');digest.update(p.read_bytes())
        c=dict(self.control)
        if mode=='wheel_locomotion':c.update(wheel_kd=0.,leg_target_rate_rad_s=self.control.get('wheel_drive_leg_target_rate_rad_s',2.),wheel_speed_limit_rad_s=30.1,
            wheel_target_rate_rad_s2=self.control.get('wheel_drive_target_rate_rad_s2',40.))
        if mode=='handstand_return':c.update(wheel_kd=.2,leg_target_rate_rad_s=6.,wheel_speed_limit_rad_s=18.)
        if mode=='handstand':c.update(wheel_kd=.2,leg_target_rate_rad_s=8.,wheel_speed_limit_rad_s=30.1)
        if mode in ('lying_down','getting_up','standing_hold'):
            c.update(wheel_kd=.2,leg_target_rate_rad_s=1.4 if mode=='lying_down' else 1.2,wheel_speed_limit_rad_s=30.1)
        return dict(joint_order=LEGS+WHEELS,
            output={'leg_pos_rad':12,'wheel_vel_rad_s':4},
            pd={k:c[k] for k in ('kp','kd','wheel_kd')},
            limits={k:c[k] for k in ('effort_limit_nm','leg_target_rate_rad_s','wheel_target_rate_rad_s2','wheel_speed_limit_rad_s')},
            timing={'physics_dt':self.physics_dt,'control_dt':self.step_dt,'substeps':self.cfg.decimation},
            asset_sha256=digest.hexdigest(),asset_hash_scope='relative paths + URDF/meshes/material/textures source bytes',
            quaternion_order='xyzw',command_frame='body_yaw_horizontal',command_axes=['vx','vy','wz'])
    def _get_observations(self):
        return {'policy':torch.zeros((1,1),device=self.device)}
    def _get_rewards(self):
        return torch.zeros(1,device=self.device)
    def _get_dones(self):
        # Never automatically teleport/reset a fallen robot.
        zero=torch.zeros(1,dtype=torch.bool,device=self.device)
        return zero,zero
    def _reset_idx(self,env_ids):
        super()._reset_idx(env_ids)
        if env_ids is None: env_ids=self.robot._ALL_INDICES
        pose=self.robot.data.default_root_pose.torch[env_ids].clone()
        pose[:,:3]+=self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim_index(root_pose=pose,env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=self.robot.data.default_root_vel.torch[env_ids],env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=self.robot.data.default_joint_pos.torch[env_ids],env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=self.robot.data.default_joint_vel.torch[env_ids],env_ids=env_ids)
        self.robot.reset(env_ids)
