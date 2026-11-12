"""Level descent and ascent sharing physically measured passive resting states.

No in-episode root or joint state writes, no intermediate pose trajectory.
The actor adjusts endpoint-biased target velocities; motor targets have rate and acceleration limits.
"""
import json, math
import torch
from isaaclab.envs import DirectRLEnv
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz
from base_env import PUEnv as Base, PUCfg as BaseCfg, ROOT, GOALS
from geometry import clearances, roll_pitch, limb_clearances
from settings import CONTROL, REWARDS, WHEEL_ALIGNMENT

@configclass
class PUCfg(BaseCfg):
    episode_length_s=20.

class PUEnv(Base):
    def __init__(self,cfg,task='P',probe=False):
        super().__init__(cfg,task,probe)
        self.feet,_=self.robot.find_bodies(['FR_foot','FL_foot','RR_foot','RL_foot'],preserve_order=True)
        self.actual_limits=self.robot.data.joint_pos_limits.torch[:,self.legs]
        self.states=json.loads((ROOT/'reference_states.json').read_text())
        if not self.states: raise RuntimeError('No verified level passive start states')
        n=self.num_envs; z=lambda *shape:torch.zeros(*shape,device=self.device)
        self.target_velocity=z(n,12);self.gain=z(n);self.contact_time=z(n);self.release_time=z(n)
        self.released=torch.zeros(n,dtype=torch.bool,device=self.device)
        self.load_blend=z(n,4)
        self.head_was_clear=torch.zeros(n,dtype=torch.bool,device=self.device)
        self.limb_indices=[[i for i,name in enumerate(self.contacts.body_names) if name.startswith(leg+'_')]
                           for leg in ('FR','FL','RR','RL')]

    def camber(self):
        axis=torch.zeros(self.num_envs,4,3,device=self.device);axis[:,:,1]=1
        axis=quat_apply(self.robot.data.body_link_quat_w.torch[:,self.feet],axis)
        return axis[:,:,2].abs().clamp(0,1).asin(),axis

    def _pre_physics_step(self,actions):
        self.extras.pop('episode',None)
        if self.pd_count is not None and self.pd_count!=4:raise RuntimeError('PD must run on each physics substep')
        self.pd_count=0;self.old_action.copy_(self.action);self.action=actions.detach().clamp(-1,1)
        # Give the actor immediate control of relative leg speeds even far from
        # the endpoint. A saturated absolute endpoint controller otherwise gives
        # every action the same motor command during the critical first seconds.
        delta=self.goal-self.target
        rate=CONTROL['leg_rate_rad_s'];acc=CONTROL['leg_accel_rad_s2']
        desired=(CONTROL['goal_velocity_gain']*delta.clamp(-1,1)+CONTROL['residual_velocity_rad_s']*self.action).clamp(-rate,rate)
        distance=torch.where(desired>=0,self.limits[...,1]-self.target,self.target-self.limits[...,0]).clamp(min=0)
        desired=desired.sign()*torch.minimum(desired.abs(),(2*acc*distance).sqrt())
        dv=acc*self.step_dt;self.target_velocity+=(desired-self.target_velocity).clamp(-dv,dv)
        step=self.target_velocity*self.step_dt
        self.target=(self.target+step).clamp(self.actual_limits[...,0],self.actual_limits[...,1])
        if self.task=='U':
            self.gain=(self.episode_length_buf*self.step_dt/CONTROL['power_on_ramp_s']).clamp(0,1)
            aligned=self.camber()[0]<math.radians(10)
            self.load_blend=torch.where(aligned,(self.load_blend+self.step_dt).clamp(max=1),torch.zeros_like(self.load_blend))
        else:
            belly,_=clearances(self)
            force=self.contacts.data.net_forces_w_history.torch.norm(dim=-1).amax(1)[:,self.base].amax(-1)
            touching=(belly.abs()<.004)&(force>.2*self.weight)
            self.contact_time=torch.where(touching,self.contact_time+self.step_dt,torch.zeros_like(self.contact_time))
            self.released|=self.contact_time>=CONTROL['contact_confirm_s']
            self.release_time+=self.released*self.step_dt
            self.gain=(1-self.release_time/CONTROL['release_s']).clamp(0,1)

    def _apply_action(self):
        d=self.robot.data
        legs=70*(self.target-d.joint_pos.torch[:,self.legs])-2.5*d.joint_vel.torch[:,self.legs]
        # Stabilize attitude through leg motors only, without writing base pose.
        roll,pitch=roll_pitch(self)
        front=torch.tensor([1.,1.,-1.,-1.],device=self.device)
        right=torch.tensor([1.,-1.,1.,-1.],device=self.device)
        gain=CONTROL['attitude_calf_offset_gain'];cap=CONTROL['attitude_calf_offset_max_rad']
        correction=(gain*pitch[:,None]*front+gain*roll[:,None]*right).clamp(-cap,cap)
        legs[:,2::3]+=70*correction
        if self.task=='U':
            # Analytic foot Jacobian transpose supports a share of body weight.
            # This supplies initial support before attitude error develops.
            q=d.joint_pos.torch[:,self.legs].reshape(-1,4,3)
            h,t,c=q.unbind(-1);sh,ch=h.sin(),h.cos()
            side=torch.tensor([-1.,1.,-1.,1.],device=self.device)
            x=-.213*t.sin()-.2264*(t+c).sin()
            z=-.213*t.cos()-.2264*(t+c).cos()
            y=.1435*side
            gy=d.projected_gravity_b.torch[:,1,None]
            gz=d.projected_gravity_b.torch[:,2,None]
            gx=d.projected_gravity_b.torch[:,0,None]
            jh=-gy*(-sh*y-ch*z)-gz*(ch*y-sh*z)
            dz=-x;dx=z
            jt=-gx*dx+gy*sh*dz-gz*ch*dz
            dz=.2264*(t+c).sin();dx=-.2264*(t+c).cos()
            jc=-gx*dx+gy*sh*dz-gz*ch*dz
            support=-torch.stack((jh,jt,jc),-1)*(.9*self.weight/4)
            legs+=support.reshape(-1,12)

        self.torque=torch.cat((legs,-.2*d.joint_vel.torch[:,self.wheels]),-1).clamp(-23.5,23.5)*self.gain[:,None]
        self.robot.set_joint_effort_target(self.torque,joint_ids=self.ids);self.pd_count+=1

    def metrics(self):
        m=super().metrics();d=self.robot.data
        belly,head=clearances(self);roll,pitch=roll_pitch(self);camber,axis=self.camber()
        force=self.contacts.data.net_forces_w_history.torch.norm(dim=-1).amax(1)
        limbs=torch.stack([force[:,indices].amax(-1)>1. for indices in self.limb_indices],-1)
        if self.task=='P' and (self.gain==0).any():
            limbs&=limb_clearances(self).abs()<.004
        ge=roll.square()+pitch.square()
        m.update(belly=belly,clearance=head,roll=roll,pitch=pitch,ge=ge,
                 front_camber=camber,front_pair=(axis[:,0]*axis[:,1]).sum(-1).abs().clamp(0,1).acos(),
                 limb_contacts=limbs,vz=d.root_lin_vel_w.torch[:,2],leg_speed=d.joint_vel.torch[:,self.legs].abs().amax(-1))
        endpoint=(m['ze'].square()/.0225 if self.task=='U' else belly.square()/.0225)
        m['phi']=-(endpoint+ge/.09+.25*m['qe']/.09).clamp(max=100)
        return m

    def _get_dones(self):
        m=self.metrics();self.p=m
        level=(m['roll'].abs()<=math.radians(5))&(m['pitch'].abs()<=math.radians(5))
        if self.task=='P':
            stable=(m['belly'].abs()<.004)&(m['base']>.2*self.weight)&m['limb_contacts'].all(-1)&(self.gain==0)
            stable&=(m['v']<.035)&(m['w']<.1)&(m['qe']<.25**2)
        else:
            stable=(m['ze']<.035)&(m['qe']<.18**2)&(m['base']<.05*self.weight)&(m['clearance']>.02)
            stable&=(m['v']<.08)&(m['w']<.18)&(m['front_camber'].amax(-1)<=math.radians(10))
        stable&=level
        self.hold=torch.where(stable,self.hold+self.step_dt,torch.zeros_like(self.hold))
        self.success=self.hold>=1.;self.new_success=self.success&~self.ever_success;self.ever_success|=self.success
        head=m['clearance']<-.003
        if self.task=='U':
            head|=self.head_was_clear&(m['clearance']<.003)
            self.head_was_clear|=m['clearance']>.025
        reasons=dict(body_tilt=~level,head=head,wheel=m['wheel']>.01,
            limits=((m['q']<self.actual_limits[...,0]-.025)|(m['q']>self.actual_limits[...,1]+.025)).any(-1),
            impact=(m['peak']>3*self.weight)&(self.episode_length_buf*self.step_dt>.1),drift=m['drift']>.4)
        self.reason_masks=reasons;self.failed=torch.stack(list(reasons.values())).any(0)
        timeout=self.episode_length_buf>=self.max_episode_length-1
        if self.probe:return torch.zeros_like(timeout),torch.zeros_like(timeout)
        return self.failed,timeout

    def _get_rewards(self):
        m=self.p;progress=.995*m['phi']-self.old_phi;self.old_phi=m['phi'].clone()
        endpoint=(torch.exp(-m['ze'].square()/.01) if self.task=='U' else torch.exp(-m['belly'].square()/.01))
        initial_height=GOALS['stand' if self.task=='P' else 'prone']['height']
        target_height=GOALS['prone' if self.task=='P' else 'stand']['height']
        fraction=((m['z']-initial_height)/(target_height-initial_height)).clamp(0,1)
        # No constant positive reward for staying horizontal at the source pose.
        source_qe=(self.source-self.goal).square().mean()
        joint_improvement=torch.exp(-m['qe']/.16)-torch.exp(-source_qe/.16)
        source_endpoint=math.exp(-(initial_height-target_height)**2/.01)
        density=(3*(endpoint-source_endpoint)+joint_improvement+fraction*torch.exp(-m['ge']/.0012)
            +4*(self.hold>0).float()-.15-.5*m['drift'].square()-.0002*self.torque.square().sum(-1)
            -.12*self.robot.data.joint_vel.torch[:,self.legs].square().mean(-1)
            -25*(m['vz'].abs()-.08).clamp(min=0).square()
            -.02*(self.action-self.old_action).square().mean(-1)
            -2*(m['peak']/self.weight-1.3).clamp(min=0).square())
        if self.task=='U':
            load=self.torque[:,:12].reshape(-1,4,3)[:,:,1:].abs().amax(-1)>2
            density-=12*(m['front_camber']-math.radians(10)).clamp(min=0).square().mul(load).sum(-1)
        else:density+=2*((self.gain==0)&(m['belly'].abs()<.004)).float()
        reward=1.5*progress+density*self.step_dt+12*self.new_success.float()-15*self.failed.float()
        self.returns+=reward;self.terminal_obs=torch.cat((self.history[:,1:],self.frame()[:,None,:]),1).flatten(1).clone()
        return reward

    def _reset_idx(self,ids):
        DirectRLEnv._reset_idx(self,ids)
        if ids is None:ids=self.robot._ALL_INDICES
        if not hasattr(self,'states'):return
        if hasattr(self,'success') and len(ids):
            success=self.success[ids]&~self.failed[ids]
            self.extras['episode']=dict(count=len(ids),success=float(success.float().mean()),reward=float(self.returns[ids].mean()),level=0.,
                timeout=float((~self.failed[ids]&~success).float().mean()),**{k:float(v[ids].float().mean()) for k,v in self.reason_masks.items()})
        n=len(ids);dev=self.device
        if self.task=='U':
            sample=[self.states[i] for i in torch.randint(len(self.states),(n,)).tolist()]
            q=torch.tensor([s['q'] for s in sample],device=dev);pose=torch.tensor([s['root_pose'] for s in sample],device=dev)
        else:
            q=self.stand[None].repeat(n,1)
            pose=self.robot.data.default_root_pose.torch[ids].clone();pose[:,:3]=0;pose[:,2]=GOALS['stand']['height']+.002
            zeros=torch.zeros(n,device=dev);pose[:,3:]=quat_from_euler_xyz(zeros,zeros,zeros)
        pose[:,:3]+=self.scene.env_origins[ids]
        joints=self.robot.data.default_joint_pos.torch[ids].clone();joints[:,self.legs]=q
        if self.task=='U':joints[:,self.wheels]=torch.tensor([s['wheel_q'] for s in sample],device=dev)
        self.robot.write_root_pose_to_sim_index(root_pose=pose,env_ids=ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n,6,device=dev),env_ids=ids)
        self.robot.write_joint_position_to_sim_index(position=joints,env_ids=ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joints),env_ids=ids);self.robot.reset(ids)
        self.target[ids]=q;self.target_velocity[ids]=0;self.gain[ids]=0 if self.task=='U' else 1
        self.contact_time[ids]=0;self.release_time[ids]=0;self.released[ids]=False;self.load_blend[ids]=0
        self.action[ids]=0;self.old_action[ids]=0;self.hold[ids]=0;self.returns[ids]=0
        self.ever_success[ids]=False;self.pending[ids]=True;self.anchor[ids]=pose[:,:2]
        self.head_was_clear[ids]=False
        gravity=torch.zeros(n,3,device=dev);gravity[:,2]=-1
        g=quat_apply_inverse(pose[:,3:],gravity)
        roll=torch.atan2(-g[:,1],-g[:,2]);pitch=g[:,0].clamp(-1,1).asin()
        height=pose[:,2]-self.scene.env_origins[ids,2]
        if self.task=='U':endpoint=(height-GOALS['stand']['height']).square()/.0225
        else:
            geometry=json.loads((ROOT/'collision_geometry.json').read_text())
            points=torch.tensor(geometry['base'],device=dev)
            rotated=quat_apply(pose[:,None,3:].expand(-1,len(points),-1),points[None].expand(n,-1,-1))
            endpoint=(rotated[:,:,2].amin(-1)+height).square()/.0225
        self.old_phi[ids]=-(endpoint+(roll.square()+pitch.square())/.09+.25*(q-self.goal).square().mean(-1)/.09).clamp(max=100)
