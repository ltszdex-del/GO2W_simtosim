"""MuJoCo state/actuator bridge for the unchanged embedded policy controllers."""
import bootstrap
bootstrap.install_tensor_compatibility()
import math
import re
from types import SimpleNamespace as NS
import mujoco
import numpy as np
import torch
from scene import StandingEnv, LEGS, WHEELS


def wrapped(value):
    return NS(torch=torch.as_tensor(np.array(value, copy=True), dtype=torch.float32))


class Robot:
    def __init__(self, env):
        self.env=env; self.data=NS(); self.joint_names=LEGS+WHEELS
        self.body_names=[mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_BODY,i) for i in range(1,env.model.nbody)]
        self.body_names=[n.replace('_wheel_link','_foot').replace('base_link','base') for n in self.body_names]
        self._ALL_INDICES=torch.tensor([0])
    def find_joints(self,names,preserve_order=True):return self._find(names,self.joint_names)
    def find_bodies(self,names,preserve_order=True):return self._find(names,self.body_names)
    @staticmethod
    def _find(names,available):
        names=[names] if isinstance(names,str) else names
        ids=[i for pattern in names for i,n in enumerate(available) if re.fullmatch(pattern,n)]
        return ids,[available[i] for i in ids]
    def set_joint_effort_target(self,values,joint_ids):
        self.env.effort[np.asarray(joint_ids)]=values.detach().cpu().numpy().reshape(-1)
    def set_joint_velocity_target(self,values,joint_ids):
        for i,value in zip(joint_ids,values.detach().cpu().numpy().reshape(-1)):
            if i<12:raise ValueError('Only wheel velocity drives supported')
            self.env.drive_target[i-12]=value
    def write_joint_damping_to_sim(self,values,joint_ids):
        for i,value in zip(joint_ids,values.detach().cpu().numpy().reshape(-1)):
            if i<12:raise ValueError('Wheel damping expected')
            self.env.drive_damping[i-12]=value
        self.data.joint_damping=wrapped([np.r_[np.zeros(12),self.env.drive_damping]])


class Sensor:
    def __init__(self,robot,filtered=False):
        self.body_names=robot.body_names;self.find_bodies=robot.find_bodies
        self.data=NS(net_forces_w_history=wrapped(np.zeros((1,4,len(self.body_names),3))),
                     normal_force_matrix_w_history=wrapped(np.zeros((1,4,1,2,3))))


class Environment(StandingEnv):
    def __init__(self,config,scene):
        self.control=config;self.device='cpu';self.num_envs=1
        self.physics_dt=.005;self.step_dt=.02;self.cfg=NS(decimation=4)
        self.model=mujoco.MjModel.from_xml_path(str(scene));self.data=mujoco.MjData(self.model)
        names=[n.replace('_foot_joint','_wheel_joint') for n in LEGS+WHEELS]
        self.joints=np.array([mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names])
        assert (self.joints>=0).all()
        self.qids=self.model.jnt_qposadr[self.joints];self.dids=self.model.jnt_dofadr[self.joints]
        self.motors=np.array([mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,n.removesuffix('_joint')) for n in names])
        self.drives=np.array([mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,s+'_drive') for s in ('FR','FL','RR','RL')])
        self.legs=list(range(12));self.wheels=list(range(12,16));self.ids=self.legs+self.wheels
        self.effort=np.zeros(16);self.drive_target=np.zeros(4);self.drive_damping=np.full(4,100.)
        self.robot=Robot(self);self.contacts=Sensor(self.robot)
        self.scene=NS(env_origins=torch.zeros(1,3),sensors={'contacts':self.contacts,
                     'front_right_rear':Sensor(self.robot,True),'front_left_rear':Sensor(self.robot,True)})
        limits=self.model.jnt_range[self.joints].copy();limits[12:]=[-1e6,1e6]
        mid=limits.mean(1);half=np.diff(limits,axis=1).ravel()*.475
        self.robot.data.joint_pos_limits=wrapped(limits[None])
        self.robot.data.soft_joint_pos_limits=wrapped(np.stack((mid-half,mid+half),-1)[None])
        self.robot.data.body_mass=wrapped(self.model.body_mass[1:][None])
        self.robot.data.body_inertia=wrapped(self.model.body_inertia[1:][None])
        self.reset()
    def set_motor_context(self,ctx,mode,wheel_damping):
        self.motor_mode=mode
        super().set_motor_context(ctx,mode,wheel_damping)
    def reset(self):
        mujoco.mj_resetData(self.model,self.data)
        initial=self.control['initial_standing'];pose=initial['root_pose']
        self.data.qpos[:3]=pose[:3];self.data.qpos[3:7]=[pose[6],*pose[3:6]]
        self.data.qpos[self.qids]=initial['q']+initial['wheel_q']
        self.target=torch.tensor(self.control['initial_leg_target'])[None]
        self.goal=self.target.clone();self.wheel_target=torch.zeros(1,4);self.wheel_goal=torch.zeros(1,4)
        self.torque=torch.zeros(1,16);self.handstand_context=None;self.pd_updates=0
        self.motor_mode='reset'
        self.effort.fill(0);self.drive_target.fill(0);self.drive_damping.fill(100)
        self.contacts.data.net_forces_w_history.torch.zero_()
        for sensor in self.scene.sensors.values():sensor.data.normal_force_matrix_w_history.torch.zero_()
        mujoco.mj_forward(self.model,self.data);self.refresh()
        self.robot.data.joint_damping=wrapped([np.r_[np.zeros(12),self.drive_damping]])
    def refresh(self):
        m,d=self.model,self.data;r=self.robot.data
        r.joint_pos=wrapped(d.qpos[self.qids][None]);r.joint_vel=wrapped(d.qvel[self.dids][None])
        positions=d.xpos[1:];quat=d.xquat[1:][:,[1,2,3,0]]
        velocities=np.zeros((m.nbody-1,6));comvel=np.zeros_like(velocities)
        for i in range(1,m.nbody):
            mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_XBODY,i,velocities[i-1],0)
            mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,i,comvel[i-1],0)
        r.body_link_pos_w=wrapped(positions[None]);r.body_link_quat_w=wrapped(quat[None])
        r.body_com_pos_w=wrapped(d.xipos[1:][None])
        r.body_link_lin_vel_w=wrapped(velocities[None,:,3:]);r.body_link_ang_vel_w=wrapped(velocities[None,:,:3])
        r.body_lin_vel_w=wrapped(comvel[None,:,3:])
        r.root_pos_w=wrapped(positions[:1]);r.root_quat_w=wrapped(quat[:1])
        r.root_lin_vel_w=wrapped(comvel[:1,3:]);r.root_ang_vel_w=wrapped(velocities[:1,:3])
        rotation=d.xmat[1].reshape(3,3)
        r.root_lin_vel_b=wrapped((rotation.T@comvel[0,3:])[None])
        r.root_ang_vel_b=wrapped((rotation.T@velocities[0,:3])[None])
        r.projected_gravity_b=wrapped((rotation.T@np.array([0.,0.,-1.]))[None])
        r.root_state_w=wrapped(np.r_[positions[0],quat[0],comvel[0,3:],velocities[0,:3]][None])
    def update_contacts(self):
        m,d=self.model,self.data;forces=np.zeros((m.nbody-1,3));pairs=np.zeros((2,2,3))
        feet=[self.robot.body_names.index(s+'_foot')+1 for s in ('FR','FL','RR','RL')]
        for i in range(d.ncon):
            contact=d.contact[i]; wrench=np.zeros(6);mujoco.mj_contactForce(m,d,i,wrench)
            world=contact.frame.reshape(3,3).T@wrench[:3]
            a,b=m.geom_bodyid[contact.geom1],m.geom_bodyid[contact.geom2]
            if a:forces[a-1]-=world
            if b:forces[b-1]+=world
            normal=contact.frame.reshape(3,3)[0]*wrench[0]
            for f in range(2):
                for rear in range(2):
                    if {a,b}=={feet[f],feet[rear+2]}:pairs[f,rear]+=normal
        history=self.contacts.data.net_forces_w_history.torch
        history[:,:-1]=history[:,1:].clone();history[:,-1]=torch.tensor(forces,dtype=torch.float32)
        for f,name in enumerate(('front_right_rear','front_left_rear')):
            history=self.scene.sensors[name].data.normal_force_matrix_w_history.torch
            history[:,:-1]=history[:,1:].clone();history[:,-1,0]=torch.tensor(pairs[f],dtype=torch.float32)
    def step(self,action=None):
        self._pre_physics_step(action)
        pu=getattr(self.handstand_context,'local_pu_isolation',False)
        subdivisions=getattr(self.handstand_context,'local_motor_subdivisions',5 if pu else 1)
        if subdivisions not in (1,5):raise ValueError('Unsupported local motor subdivision')
        original_dt=self.model.opt.timestep
        self.model.opt.timestep=self.physics_dt/subdivisions
        try:
            for _ in range(4):
                for inner in range(subdivisions):
                    self._apply_action()
                    # Conservative official envelope: 30 rad/s hip/thigh/wheels,
                    # 20.07 rad/s calf. Never command torque that accelerates an
                    # overspeed joint further; passive/contact overspeed is recorded.
                    velocity=self.data.qvel[self.dids]
                    limits=np.array([30.,30.,20.07]*4+[30.]*4)
                    self.effort=np.where((np.abs(velocity)>=limits)&(self.effort*velocity>0),0.,self.effort)
                    self.drive_target=np.clip(self.drive_target,-30.,30.)
                    self.model.actuator_gainprm[self.drives,0]=self.drive_damping
                    self.model.actuator_biasprm[self.drives,2]=-self.drive_damping
                    self.data.ctrl[self.motors]=self.effort
                    self.data.ctrl[self.drives]=self.drive_target
                    mujoco.mj_step(self.model,self.data)
                    mujoco.mj_forward(self.model,self.data)
                    self.refresh()
                self.update_contacts()
        finally:
            self.model.opt.timestep=original_dt
        if not np.isfinite(self.data.qpos).all():raise RuntimeError('Nonfinite physical state')
