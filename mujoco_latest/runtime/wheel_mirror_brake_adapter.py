import torch,json,hashlib,math
from pathlib import Path
from types import SimpleNamespace as NS
from torch import nn
from wasd_adapter import asset_hash
from commands import profile_from_contract
from idle_recovery import recovery_goal,blend_step

class WheelAdapter:
    def __init__(self,spec,device):
        self.spec,self.device=spec,device;root=Path(spec['training_root'])
        self.path=Path(spec['checkpoint'])
        if self.path.suffix=='.txt':self.path=Path(self.path.read_text().strip())
        self.path=str(self.path);ck=torch.load(self.path,map_location='cpu',weights_only=False);self.ck=ck;m=ck['manifest']
        assert m['schema']=='go2w-wheel-mirror-brake-v3' and (m['actor_dim'],m['action_dim'])==(570,16)
        for n,h in m['source_sha256'].items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
        assert hashlib.sha256((root/'assets/go2w_description/urdf/go2w_description.urdf').read_bytes()).hexdigest()==m['urdf_sha256']
        assets=Path(spec['asset_root']);source=root/'assets/go2w_description'
        if asset_hash(assets)!=asset_hash(source):
            assert spec.get('collision_geometry_override')
            from asset_compatibility import collision_only_migration
            collision_only_migration(source/'urdf/go2w_description.urdf',assets/'urdf/go2w_description.urdf')
            for p in source.rglob('*'):
                if p.is_file() and p.suffix.lower()!='.urdf':assert p.read_bytes()==(assets/p.relative_to(source)).read_bytes()
        self.c=m['control'];c=self.c
        self.command_profile=profile_from_contract(m['command'],c)
        # Training uses motor saturation; no fixed wheel-separation command cap.
        self.command_profile.update(wheel_envelope_lever=0.,wheel_reference_speed_m_s=0.)
        for key in ('min_forward_m_s', 'max_forward_m_s', 'max_lateral_m_s', 'max_yaw_rad_s', 'yaw_cruise_limit_rad_s'):self.command_profile[key]*=spec.get('command_limit_scale',1.)
        self.dynamic_feedforward='actual_wheel_center_y' in m['wheel_feedforward']
        self.center_offsets=torch.tensor(m['phase'].get('wheel_center_offsets',[[0.,-.048,0.],[0.,.048,0.]]*2),device=device)
        assert (c['wheel_lock_damping'],c['wheel_residual'])==(3.,4.)
        assert c['leg_rate'] in (6.,), 'Unsupported wheel leg-rate contract'
        self.stand=torch.tensor(c['stand'],device=device)[None];self.scale=torch.tensor(c['leg_scale'],device=device)[None]
        self.manifest=dict(observation_order=m['observation_order'],normalization=m['normalization'],scales=c['leg_scale'],history=10,phase='original .30s phase',previous_action=m['previous_action'],target='four-wheel drive',task_id='wheel_drive',joint_order=m['joint_order'],output={'leg_pos_rad':12,'wheel_vel_rad_s':4},pd={'kp':70.,'kd':2.5,'wheel_kd':0.},limits={'effort_limit_nm':23.5,'leg_target_rate_rad_s':c['leg_rate'],'wheel_target_rate_rad_s2':c['wheel_accel'],'wheel_speed_limit_rad_s':30.1},timing={'physics_dt':.005,'control_dt':.02,'substeps':4},asset_sha256=asset_hash(assets))
    def load(self):
        self.actor=nn.Sequential(nn.Linear(570,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,128),nn.ELU(),nn.Linear(128,16)).to(self.device).eval()
        self.actor.load_state_dict({k.removeprefix('actor.'):v for k,v in self.ck['model'].items() if k.startswith('actor.')},strict=True)
        self.actor=_MirrorNetwork(self.actor,self.stand)
    def bind_env(self,env):
        self.env=env
        self.feet,_=env.robot.find_bodies(['FR_foot','FL_foot','RR_foot','RL_foot'],preserve_order=True)
        assert len(self.feet)==4
        if 'wheel_command_profile' in env.control:
            assert env.control['wheel_command_profile']==self.command_profile,'Wheel command profile mismatch'
    def wheel_center_y(self):
        from isaaclab.utils.math import quat_apply,quat_apply_inverse
        d=self.env.robot.data
        q=d.body_link_quat_w.torch[:,self.feet]
        centers=d.body_link_pos_w.torch[:,self.feet]+quat_apply(q.reshape(-1,4),self.center_offsets).reshape(1,4,3)
        root=d.root_quat_w.torch[:,None,:].expand(-1,4,-1).reshape(-1,4)
        return quat_apply_inverse(root,(centers-d.root_pos_w.torch[:,None,:]).reshape(-1,3)).reshape(1,4,3)[:,:,1]
    def reset(self,state):
        e=self.env;self.target=e.target.clone();self.target_velocity=torch.zeros(1,12,device=self.device);self.wheel_target=e.wheel_target.clone()
        self.torque=torch.zeros(1,16,device=self.device);self.previous=torch.zeros_like(self.torque);self.history=None;self.age=0.
        self.idle_blend=0.;self.idle_requested=False;self.idle_goal=0.
        self.limits=e.robot.data.soft_joint_pos_limits.torch[:,e.legs];e.set_motor_context(self,'wheel_drive',3.)
    def observe(self,state):
        command=state['command']['command_m_s_rad_s'];heading=state['command']['yaw_target_rad'];error=heading-state['yaw_rad']
        assert abs(command[1])<1e-6
        phase=self.age*2*math.pi/.3
        frame=torch.cat((state['projected_gravity_b'],state['angular_velocity_b']*.25,state['joint_pos_rad'][:,:12]-self.stand,state['joint_vel_rad_s']*.05,self.previous,torch.tensor([command],device=self.device),torch.tensor([[math.sin(error),math.cos(error),math.sin(phase),math.cos(phase)]],device=self.device)),-1).clamp(-10,10)
        assert frame.shape==(1,57)
        self.history=frame[:,None].repeat(1,10,1) if self.history is None else torch.cat((self.history[:,1:],frame[:,None]),1)
        self.command=command
        keys=state['command'].get('keys',[])
        d=self.env.robot.data
        actual_speed=float(d.root_lin_vel_w.torch[0,:2].norm())
        actual_yaw=abs(float(d.root_ang_vel_w.torch[0,2]))
        self.idle_goal=recovery_goal(keys,command)
        self.idle_requested=self.idle_goal>.5
        self.idle_actual_speed=actual_speed;self.idle_actual_yaw=actual_yaw
        return self.history.flatten(1)
    @torch.inference_mode()
    def infer(self,obs):return self.actor(obs).tanh()
    def apply(self,action,dt):
        c=self.c;desired=(self.stand+self.scale*action[:,:12]).clamp(self.limits[...,0],self.limits[...,1])
        enabled=self.env.control.get('wheel_idle_assist',False)
        goal=self.idle_goal if enabled else 0.
        self.idle_blend=blend_step(self.idle_blend,goal,dt)
        neutral=torch.tensor(self.env.control['initial_leg_target'],device=self.device)[None]
        desired=torch.lerp(desired,neutral,self.idle_blend).clamp(self.limits[...,0],self.limits[...,1])
        self.target+=(desired-self.target).clamp(-c['leg_rate']*dt,c['leg_rate']*dt)
        wheel=c['wheel_action_scale']*action[:,12:]*(1.-self.idle_blend)
        decelerating=self.wheel_target*(wheel-self.wheel_target)<0
        rate=torch.where(decelerating,c.get('wheel_brake_accel',c['wheel_accel']),c['wheel_accel'])*dt
        self.wheel_target+=(wheel-self.wheel_target).clamp(-rate,rate)
        self.previous=action.clone();self.age+=dt
        return dict(leg_pos_rad=self.target,wheel_vel_rad_s=self.wheel_target)
    def _apply_action(self):
        e=self.env;d=e.robot.data
        leg=70*(self.target-d.joint_pos.torch[:,e.legs])-2.5*d.joint_vel.torch[:,e.legs]
        self.torque=torch.cat((leg,torch.zeros(1,4,device=self.device)),-1).clamp(-23.5,23.5)
        e.robot.set_joint_velocity_target(self.wheel_target,joint_ids=e.wheels)
        e.robot.set_joint_effort_target(self.torque,joint_ids=e.ids)
    def is_complete(self,state):return False

def create(spec,device):return WheelAdapter(spec,device)

"""Sagittal reflection for the actual 57 x 10 policy history (FR,FL,RR,RL)."""
import torch

def mirror_action(a):
    legs=a[...,:12].reshape(*a.shape[:-1],4,3)[...,[1,0,3,2],:].clone()
    legs[...,0]*=-1
    result=legs.flatten(-2)
    if a.shape[-1]==16:
        result=torch.cat((result,a[...,12:][...,[1,0,3,2]]),-1)
    return result

def mirror_observation(obs,stand,stepping):
    x=obs.reshape(-1,10,57);y=x.clone()
    y[...,1]*=-1                 # polar gravity
    y[...,[3,5]]*=-1             # axial angular velocity
    y[...,6:18]=mirror_action(x[...,6:18]+stand)-stand
    y[...,18:34]=mirror_action(x[...,18:34])
    y[...,34:50]=mirror_action(x[...,34:50])
    y[...,[51,52,53]]*=-1        # vy, yaw rate, sin(heading)
    if stepping:y[...,55:57]*=-1 # reflection exchanges diagonal gait phases
    return y.reshape_as(obs)

class _MirrorNetwork(nn.Module):
    def __init__(self,raw,stand):super().__init__();self.raw=raw;self.stand=stand
    def forward(self,obs):
        return .5*(self.raw(obs)+mirror_action(self.raw(mirror_observation(obs,self.stand,False))))
