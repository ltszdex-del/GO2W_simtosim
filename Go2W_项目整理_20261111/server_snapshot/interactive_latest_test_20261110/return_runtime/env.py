"""H4099 warm-up -> PPO-controlled, rear-leg-cushioned return to U1099 stance.

Only _reset_idx writes physical states. Warm-up -> recovery changes motor ownership
and reference variables, with no position, velocity or orientation reset.
"""
import json,math,sys
from types import SimpleNamespace as NS
import torch
import warp as wp
from .embedded_engine import EmbeddedEnvBase as DirectRLEnv
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse,quat_mul,quat_apply
from .settings import *
from .geometry import wheel_pair_geometry
from .teacher_context import make_teacher,make_u_controller
from go2w_mimic.env_cfg import MimicEnvCfg
from go2w_mimic.settings import LEG_NAMES,WHEEL_NAMES,FOOT_NAMES
from .u_runtime.geometry import clearances

def wrap(x):return torch.atan2(x.sin(),x.cos())
def smooth(x):return x**3*(10-15*x+6*x*x)

@configclass
class ReturnCfg(MimicEnvCfg):
    action_space=ACTION_DIM
    observation_space=ACTOR_DIM
    state_space=0
    episode_length_s=12.
    randomize=False

class ReturnEnv(DirectRLEnv):
    def __init__(self,cfg):
        super().__init__(cfg)
        self._physics_handles_decimation=False
        self.legs,_=self.robot.find_joints(LEG_NAMES,preserve_order=True)
        self.wheels,_=self.robot.find_joints(WHEEL_NAMES,preserve_order=True)
        self.ids=self.legs+self.wheels
        self.feet,_=self.robot.find_bodies(FOOT_NAMES,preserve_order=True)
        self.contact_ids,_=self.contacts.find_bodies(FOOT_NAMES,preserve_order=True)
        self.other_contacts=[i for i in range(len(self.contacts.body_names)) if i not in self.contact_ids]
        self.limits=self.robot.data.soft_joint_pos_limits.torch[:,self.legs]
        self.actual_limits=self.robot.data.joint_pos_limits.torch[:,self.legs]
        self.bank_json=json.loads((ROOT/'early_handstand_states.json').read_text())
        self.bank={key:torch.tensor([r[key] for r in self.bank_json],device=self.device,dtype=torch.float32)
            for key in self.bank_json[0] if key not in ('score',)}
        self.U=json.loads((ROOT/'U_final_standing.json').read_text())
        self.goal_q=torch.tensor(self.U['q'],device=self.device)
        self.goal_target=torch.tensor(self.U['target'],device=self.device)
        self.goal_height=float(self.U['height'])
        self.goal_gravity=quat_apply_inverse(torch.tensor(self.U['root_pose'][3:],device=self.device),
            torch.tensor([0.,0.,-1.],device=self.device))
        self.goal_pitch=float(torch.atan2(self.goal_gravity[0],-self.goal_gravity[2]))
        n=self.num_envs;z=lambda *s:torch.zeros(*s,device=self.device)
        self.action=z(n,16);self.old_action=z(n,16);self.target=z(n,12);self.target_velocity=z(n,12)
        self.wheel_target=z(n,4);self.torque=z(n,16);self.history=z(n,HISTORY,FRAME)
        self.fill_history=torch.ones(n,dtype=torch.bool,device=self.device)
        self.recovering=torch.zeros(n,dtype=torch.bool,device=self.device)
        self._controlled=self.recovering.clone();self.last_step_mask=self.recovering.clone()
        self.elapsed=z(n);self.warmup_age=z(n);self.warmup_duration=z(n);self.guidance_duration=z(n)
        self.handoff_target=z(n,12);self.handoff_pitch=z(n);self.handoff_height=z(n)
        self.heading_goal=z(n);self.front_anchor=z(n,2,2);self.landing_time=torch.full((n,2),-1.,device=self.device)
        self.previous_contacts=torch.zeros(n,4,dtype=torch.bool,device=self.device)
        self.rear_touch=torch.zeros(n,2,dtype=torch.bool,device=self.device)
        self.touchdown_speed=z(n,2);self.rear_impact_peak=z(n);self.episode_touchdown_speed=z(n)
        self.previous_foot_vz=z(n,4);self.previous_normal_force=z(n,4)
        self.air_time=z(n);self.max_air=z(n);self.axis_bad_time=z(n);self.body_bad_time=z(n)
        self.hold=z(n);self.best_hold=z(n);self.returns=z(n);self.old_phi=z(n)
        self.peak_heading=z(n);self.peak_roll=z(n);self.peak_coaxial=z(n,2)
        self.pose_rate=z(n);self.previous_quat=z(n,4);self.previous_quat[:,3]=1.
        self.pitch_rate=z(n);self.previous_pitch=z(n);self.u_blend=z(n)
        self.jitter_ema=z(n,3);self.handoff_jitter=z(n,3)
        self.handoff_without_jitter=torch.zeros(n,dtype=torch.bool,device=self.device)
        self.level=0.;self.success_window=[];self.reason_masks={};self.pd_updates=0
        self.teacher_resets=0;self.handoffs=0;self.last_step_handoffs=0
        self.success=self.recovering.clone();self.failed=self.recovering.clone()
        self.terminal_obs=z(n,ACTOR_DIM)
        self.h,self.h_model=cfg.teacher_context,cfg.teacher_model
        self._randomize_fixed_physics()
        self.u=make_u_controller(self)
        self.weight=self.h.masses.sum(-1)*9.81
        self.h.gain[:]=1.;self.h.speed[:]=1.

    def _setup_scene(self):
        self.robot=self.scene.articulations['robot'];self.contacts=self.scene.sensors['contacts']

    def _randomize_fixed_physics(self):
        if not getattr(self.cfg,'physics_randomization',True):return
        n=self.num_envs
        scale=1+(torch.rand(n)-.5)*2*TASK['fixed_mass_randomization_fraction']
        ids=wp.from_torch(torch.arange(n,dtype=torch.int32))
        masses=wp.to_torch(self.robot.root_view.get_masses()).cpu().clone()
        inertia=wp.to_torch(self.robot.root_view.get_inertias()).cpu().clone()
        self.robot.root_view.set_masses(wp.from_torch(masses*scale[:,None]),indices=ids)
        self.robot.root_view.set_inertias(wp.from_torch(inertia*scale[:,None,None]),indices=ids)
        mat=wp.to_torch(self.robot.root_view.get_material_properties()).cpu().clone()
        lo,hi=TASK['fixed_friction_range'];mu=torch.empty(n).uniform_(lo,hi)
        mat[:,:,0]=mu[:,None];mat[:,:,1]=mu[:,None];mat[:,:,2]=0.
        self.robot.root_view.set_material_properties(wp.from_torch(mat),indices=ids)
        self.h.masses*=scale.to(self.device)[:,None]
        self.h.mass_scale=scale.to(self.device);self.h.mu=mu.to(self.device)

    def _pre_physics_step(self,actions):
        self.extras.pop('episode',None)
        if self.pd_updates not in (0,4):raise RuntimeError('PD did not execute four physical substeps')
        self.pd_updates=0
        self.u.pd_count=0
        self._controlled=self.recovering.clone();self.last_step_mask=self._controlled.clone()
        self.last_step_handoffs=0
        self.old_action.copy_(self.action)
        with torch.no_grad():ha=self.h_model(self.h._get_observations()['policy'])
        self.h._pre_physics_step(ha)
        self.action=torch.where(self._controlled[:,None],actions.detach().clamp(-1,1),ha)
        warm=~self._controlled
        self.target[warm]=self.h.target[warm];self.target_velocity[warm]=self.h.target_velocity[warm]
        self.wheel_target[warm]=self.h.wheel_target[warm]
        phase=(self.elapsed/self.guidance_duration.clamp(min=.1)).clamp(0,1)
        blend=smooth(phase)
        gravity=self.robot.data.projected_gravity_b.torch
        pitch=torch.atan2(gravity[:,0],-gravity[:,2])
        guide=self.joint_guide(pitch,phase)
        scale=torch.full((12,),CONTROL['leg_residual'],device=self.device);scale[::3]=CONTROL['hip_residual']
        # Residuals become smaller after landing, without removing actor control.
        residual_scale=1-.65*blend
        desired=guide+scale*residual_scale[:,None]*self.action[:,:12]
        age=(self.elapsed[:,None]-self.landing_time).clamp(min=0)
        cushion=((self.landing_time>=0)*torch.exp(-age/.22)).amax(-1)
        desired[:,8::3]-=.12*cushion[:,None]
        desired[:,7::3]+=.06*cushion[:,None]
        common=((desired[:,0]+desired[:,3])*.5).clamp(-.10,.10)
        differential=((desired[:,0]-desired[:,3])*.5).clamp(-.085,.085)
        desired[:,0]=common+differential;desired[:,3]=common-differential
        desired=desired.clamp(self.limits[...,0],self.limits[...,1])
        velocity=((desired-self.target)/self.step_dt).clamp(-CONTROL['leg_rate'],CONTROL['leg_rate'])
        dv=CONTROL['leg_acceleration']*self.step_dt
        velocity=self.target_velocity+(velocity-self.target_velocity).clamp(-dv,dv)
        velocity=velocity.clamp(-CONTROL['leg_rate'],CONTROL['leg_rate'])
        self.target[self._controlled]=(self.target+velocity*self.step_dt).clamp(self.limits[...,0],self.limits[...,1])[self._controlled]
        self.target_velocity[self._controlled]=velocity[self._controlled]
        wheel_goal=(self.action[:,12:]*CONTROL['wheel_scale']).clamp(-CONTROL['wheel_speed_limit'],CONTROL['wheel_speed_limit'])
        pitch_reference=self.handoff_pitch+blend*(self.goal_pitch-self.handoff_pitch)
        wheel_prior=CONTROL['front_wheel_pitch_kp']*(pitch-pitch_reference)+CONTROL['front_wheel_pitch_kd']*self.pitch_rate
        wheel_prior+=CONTROL['front_wheel_descent_bias']*(1-phase).square()
        wheel_goal[:,:2]+=wheel_prior[:,None]
        wheel_goal=wheel_goal.clamp(-CONTROL['wheel_speed_limit'],CONTROL['wheel_speed_limit'])
        cap=CONTROL['wheel_acceleration']*self.step_dt
        updated=self.wheel_target+(wheel_goal-self.wheel_target).clamp(-cap,cap)
        self.wheel_target[self._controlled]=updated[self._controlled]
        rear_supported=self.previous_contacts[:,2:].all(-1)&(self.landing_time>=0).all(-1)
        u_goal=torch.exp(-(pitch/CONTROL['U_blend_pitch_scale_rad']).square())*rear_supported
        self.u_blend+=(u_goal-self.u_blend)*(self.step_dt/CONTROL['U_motor_blend_time_s'])

    def joint_guide(self,pitch,phase):
        blend=smooth(phase)
        actual=((self.handoff_pitch-pitch)/(self.handoff_pitch-self.goal_pitch).clamp(min=.5)).clamp(0,1)
        front=CONTROL['front_joint_time_blend']*blend+(1-CONTROL['front_joint_time_blend'])*actual
        # Prepare a little in handstand; unfold mainly as the BODY descends.
        # Time alone must not fully extend rear wheels toward front support.
        prepared=smooth((self.elapsed/CONTROL['rear_prepare_s']).clamp(0,1))
        rear=.2*prepared+.8*smooth((actual*1.4).clamp(0,1))
        guide=self.handoff_target+blend[:,None]*(self.goal_target-self.handoff_target)
        guide[:,:6]=self.handoff_target[:,:6]+front[:,None]*(self.goal_target[:6]-self.handoff_target[:,:6])
        guide[:,6:]=self.handoff_target[:,6:]+rear[:,None]*(self.goal_target[6:]-self.handoff_target[:,6:])
        return guide

    def _apply_action(self):
        self.h._apply_action()
        d=self.robot.data;q=d.joint_pos.torch[:,self.legs];dq=d.joint_vel.torch[:,self.ids]
        leg=CONTROL['kp']*(self.target-q)+CONTROL['kd']*(self.target_velocity-dq[:,:12])
        age=(self.elapsed[:,None]-self.landing_time).clamp(min=0)
        cushion=(self.landing_time>=0)*(1-age/CONTROL['rear_softening_seconds']).clamp(0,1)
        for local,leg_index in enumerate((2,3)):
            sl=slice(leg_index*3,leg_index*3+3)
            soft=CONTROL['rear_landing_kp']*(self.target[:,sl]-q[:,sl])
            soft+=CONTROL['rear_landing_kd']*(self.target_velocity[:,sl]-dq[:,sl])
            leg[:,sl]=leg[:,sl]*(1-cushion[:,local,None])+soft*cushion[:,local,None]
        wheel=CONTROL['wheel_kd']*(self.wheel_target-dq[:,12:])
        recovery=torch.cat((leg,wheel),-1).clamp(-CONTROL['effort_nm'],CONTROL['effort_nm'])
        self.u.target=self.target;self.u._apply_action()
        ublend=self.u_blend
        recovery=recovery*(1-ublend[:,None])+self.u.torque*ublend[:,None]
        fade=((self.elapsed+self.step_dt)/CONTROL['teacher_motor_crossfade_s']).clamp(0,1)
        recovery=recovery*fade[:,None]+self.h.torque*(1-fade[:,None])
        self.torque=torch.where(self._controlled[:,None],recovery,self.h.torque)
        self.torque=self.torque.clamp(-CONTROL['effort_nm'],CONTROL['effort_nm'])
        self.robot.set_joint_effort_target(self.torque,joint_ids=self.ids)
        self.pd_updates+=1

    def metrics(self):
        d=self.robot.data;p=self.h.physical();g=d.projected_gravity_b.torch
        pitch=torch.atan2(g[:,0],-g[:,2]);roll=torch.atan2(-g[:,1],torch.sqrt(g[:,0].square()+g[:,2].square()))
        height=d.root_pos_w.torch[:,2]-self.scene.env_origins[:,2]
        force=self.contacts.data.net_forces_w_history.torch[:,:,self.contact_ids,:].norm(dim=-1)
        peak=force.amax(1);normal=p['force'].mean(1)
        belly,head=clearances(self)
        heading=wrap(self.h.stable_heading()-self.heading_goal)
        # Preserve sampled U's small real asymmetry; penalize deviation from it.
        q=d.joint_pos.torch[:,self.legs].reshape(-1,4,3);goal=self.goal_q.reshape(4,3)
        mirror=torch.tensor([-1.,1.,1.],device=self.device)
        symmetry=torch.cat((q[:,0]-q[:,1]*mirror-(goal[0]-goal[1]*mirror),
                            q[:,2]-q[:,3]*mirror-(goal[2]-goal[3]*mirror)),-1)
        rear_xy=p['feet_w'][:,2:,:2]-p['feet_w'][:,:2,:2].mean(1)[:,None,:]
        c,s=self.heading_goal.cos(),self.heading_goal.sin()
        rear_heading_x=c[:,None]*rear_xy[:,:,0]+s[:,None]*rear_xy[:,:,1]
        # Rear wheels land behind CURRENT front support; translation is scored separately.
        landing_gate=((.25-p['clearance'][:,2:])/.20).clamp(0,1)
        wrong_side=(((rear_heading_x+.12).clamp(min=0)/.15).square().clamp(max=9)*landing_gate).mean(-1)
        wheel_quat=d.body_link_quat_w.torch[:,self.feet]
        local_y=torch.zeros(self.num_envs,4,3,device=self.device);local_y[:,:,1]=1.
        axes=quat_apply(wheel_quat.reshape(-1,4),local_y.reshape(-1,3)).reshape(-1,4,3)
        axle_angle,axle_offset,offset_vector,axis_difference=wheel_pair_geometry(p['feet_w'],axes)
        local_x=torch.zeros(self.num_envs,3,device=self.device);local_x[:,0]=1.
        forward=quat_apply(d.root_quat_w.torch,local_x)
        sagittal_tilt=(-s*forward[:,0]+c*forward[:,1]).clamp(-1,1).asin()
        roll_rate=c*d.root_ang_vel_w.torch[:,0]+s*d.root_ang_vel_w.torch[:,1]
        rear_lateral=(-s[:,None]*rear_xy[:,:,0]+c[:,None]*rear_xy[:,:,1]).mean(-1)
        rear_gate=((.25-p['clearance'][:,2:].mean(-1))/.20).clamp(0,1)
        front_delta=p['feet_w'][:,:2,:2]-self.front_anchor
        anchor_heading=torch.stack((c[:,None]*front_delta[:,:,0]+s[:,None]*front_delta[:,:,1],
            -s[:,None]*front_delta[:,:,0]+c[:,None]*front_delta[:,:,1]),-1)
        rear_heading=torch.stack((rear_heading_x,-s[:,None]*rear_xy[:,:,0]+c[:,None]*rear_xy[:,:,1]),-1)
        wheel_gap=torch.cdist(p['feet_w'][:,:2],p['feet_w'][:,2:]).amin(dim=(1,2))
        front_speed=d.body_link_lin_vel_w.torch[:,self.feet[:2],:2].norm(dim=-1)
        yaw_rate=d.root_ang_vel_w.torch[:,2]
        phase=(self.elapsed/self.guidance_duration.clamp(min=.1)).clamp(0,1);b=smooth(phase)
        pitch_ref=self.handoff_pitch+b*(self.goal_pitch-self.handoff_pitch)
        height_ref=self.handoff_height+b*(self.goal_height-self.handoff_height)
        goal_error=(d.joint_pos.torch[:,self.legs]-self.goal_q).square().mean(-1)
        return dict(**p,pitch=pitch,roll=roll,height=height,peak=peak,normal=normal,head=head,belly=belly,
            wheel_gap=wheel_gap,axle_angle=axle_angle,axle_offset=axle_offset,offset_vector=offset_vector,axis_difference=axis_difference,
            sagittal_tilt=sagittal_tilt,sagittal_roll_rate=roll_rate,rear_lateral=rear_lateral,rear_gate=rear_gate,
            anchor_heading=anchor_heading,rear_heading=rear_heading,symmetry_rms=symmetry.square().mean(-1).sqrt(),wrong_landing_side=wrong_side,front_speed=front_speed,yaw_rate=yaw_rate,heading_error=heading,phase=phase,blend=b,pitch_reference=pitch_ref,height_reference=height_ref,
            q_error=goal_error,foot_vz=d.body_link_lin_vel_w.torch[:,self.feet,2],
            vxy=d.root_lin_vel_w.torch[:,:2].norm(dim=-1),vz=d.root_lin_vel_w.torch[:,2],
            omega=d.root_ang_vel_b.torch.norm(dim=-1),leg_rms=d.joint_vel.torch[:,self.legs].square().mean(-1).sqrt(),
            wheel_rms=d.joint_vel.torch[:,self.wheels].square().mean(-1).sqrt())

    def _handoff(self,mask,m):
        if not mask.any():return
        # No write_*_to_sim here. Transfer the actual running targets and velocity.
        self.recovering[mask]=True;self.elapsed[mask]=0.
        # The frozen H motor commands contain oscillatory tracking offsets.
        # Start the return reference from the measured pose/rate, never latch
        # a transient asymmetric command. Physical state is untouched; the
        # existing 120ms torque crossfade bridges motor ownership.
        actual_q=self.robot.data.joint_pos.torch[:,self.legs]
        actual_dq=self.robot.data.joint_vel.torch[:,self.legs]
        self.handoff_target[mask]=actual_q[mask]
        self.target[mask]=actual_q[mask]
        self.target_velocity[mask]=actual_dq[mask].clamp(-CONTROL['handoff_leg_rate'],CONTROL['handoff_leg_rate'])
        self.handoff_pitch[mask]=m['pitch'][mask];self.handoff_height[mask]=m['height'][mask]
        self.front_anchor[mask]=m['feet_w'][mask,:2,:2]
        self.old_phi[mask]=self.potential(m)[mask]
        self.handoff_jitter[mask]=self.jitter_ema[mask]
        self.handoff_without_jitter[mask]=~self.jittered[mask]
        count=int(mask.sum());self.handoffs+=count;self.last_step_handoffs=count

    def potential(self,m):
        return -(m['pitch']-self.goal_pitch).abs()-.4*(m['height']-self.goal_height).abs()-.15*m['q_error']

    def _get_dones(self):
        self.h.episode_length_buf+=1
        self.elapsed+=self._controlled*self.step_dt;self.warmup_age+=(~self._controlled)*self.step_dt
        m=self.metrics();d=self.robot.data
        self.pitch_rate=.8*self.pitch_rate+.2*wrap(m['pitch']-self.previous_pitch)/self.step_dt
        self.previous_pitch=m['pitch'].clone()
        self.jitter_ema=.75*self.jitter_ema+.25*torch.stack((m['leg_rms'],m['omega'],m['wheel_rms']),-1)
        self.jittered=(self.jitter_ema[:,0]>TASK['jitter_leg_rms_rad_s'])|(self.jitter_ema[:,1]>TASK['jitter_body_angular_rad_s'])|(self.jitter_ema[:,2]>TASK['jitter_wheel_rms_rad_s'])
        conjugate=self.previous_quat.clone();conjugate[:,:3]*=-1
        delta=quat_mul(d.root_quat_w.torch,conjugate)
        rate=2*torch.atan2(delta[:,:3].norm(dim=-1),delta[:,3].abs().clamp(min=1e-6))/self.step_dt
        self.pose_rate=.85*self.pose_rate+.15*rate
        self.previous_quat=d.root_quat_w.torch.clone()
        contact=m['contact']>.5
        self.air_time=torch.where(contact.any(-1),0.,self.air_time+self.step_dt)
        self.max_air=torch.maximum(self.max_air,self.air_time)
        self.rear_touch=contact[:,2:]&~self.previous_contacts[:,2:]&self._controlled[:,None]
        first_touch=self.rear_touch&(self.landing_time<0)
        self.rear_touch=first_touch
        self.landing_time=torch.where(first_touch,self.elapsed[:,None],self.landing_time)
        self.touchdown_speed=torch.where(first_touch,(-self.previous_foot_vz[:,2:]).clamp(min=0),0.)
        self.episode_touchdown_speed=torch.maximum(self.episode_touchdown_speed,self.touchdown_speed.amax(-1))
        rear_peak=(m['peak'][:,2:]/self.weight[:,None]).amax(-1)
        self.rear_impact_peak=torch.maximum(self.rear_impact_peak,rear_peak*self._controlled)
        self.axis_bad_time=torch.where(m['front_axis_angle']>TASK['front_axis_fail_rad'],self.axis_bad_time+self.step_dt,0.)
        self.body_bad_time=torch.where(m['bad']>.25*self.weight,self.body_bad_time+self.step_dt,0.)
        reasons=dict(wheel_collision=m['wheel_collision'],head=m['head']<-.003,
            rear_impact=(m['peak'][:,2:]/self.weight[:,None]).amax(-1)>TASK['fail_rear_peak_bodyweights'],
            body_ground=(self.body_bad_time>.06)|(m['belly']<-.004),
            height=(m['height']<TASK['fail_height_min_m'])|(m['height']>TASK['fail_height_max_m']),
            roll=m['roll'].abs()>.45,pitch=(m['pitch']<-.35)|(m['pitch']>2.05),
            front_axis=self.axis_bad_time>TASK['front_axis_fail_time_s'],air=self.air_time>TASK['fail_air_s'],
            heading=m['heading_error'].abs()>.6,
            limits=((d.joint_pos.torch[:,self.legs]<self.actual_limits[...,0]-.03)|(d.joint_pos.torch[:,self.legs]>self.actual_limits[...,1]+.03)).any(-1),
            drift=(m['feet_w'][:,:2,:2]-self.front_anchor).norm(dim=-1).amax(-1)>1.2,
            nonfinite=~torch.isfinite(d.root_state_w.torch).all(-1))
        self.reason_masks=reasons;self.failed=torch.stack(list(reasons.values())).any(0)
        ready=(~self._controlled)&(self.warmup_age>=self.warmup_duration)&self.jittered
        eligible=(m['pitch']>1.18)&(m['pitch']<1.95)&(m['clearance'][:,2:].amin(-1)>.07)
        self._handoff(ready&eligible&~self.failed,m)
        # A teacher that fails before motor ownership is transferred is reset and
        # logged separately. It never becomes a failure label for unused PPO actions.
        warm_timeout=(~self.recovering)&(self.warmup_age>self.warmup_duration+TASK['warmup_extra_timeout_s'])
        self.failed|=warm_timeout
        self.reason_masks['teacher_warmup_timeout']=warm_timeout
        good=self._controlled&(m['phase']>.65)&((m['pitch']-self.goal_pitch).abs()<TASK['stand_angle_rad'])
        good&=(m['roll'].abs()<TASK['stand_angle_rad'])&(m['height']-self.goal_height).abs().lt(TASK['stand_height_error_m'])
        good&=(m['q_error']<TASK['stand_joint_rms_rad']**2)&contact.all(-1)&(self.landing_time>=0).all(-1)
        good&=(m['vxy']<TASK['stand_horizontal_speed_m_s'])&(m['vz'].abs()<TASK['stand_vertical_speed_m_s'])
        good&=(self.pose_rate<TASK['stand_pose_angular_speed_rad_s'])&(m['heading_error'].abs()<TASK['stand_heading_error_rad'])
        good&=(m['bad']<.1*self.weight)&~self.failed
        if TASK['require_soft_landing_for_success']:
            good&=(self.episode_touchdown_speed<TASK['success_rear_touch_speed_m_s'])
            good&=(self.rear_impact_peak<TASK['success_rear_peak_bodyweights'])
        self.hold=torch.where(good,self.hold+self.step_dt,0.);self.best_hold=torch.maximum(self.best_hold,self.hold)
        self.success=self._controlled&(self.hold>=TASK['success_hold_s']-1e-6)&~self.failed&(self.elapsed<=TASK['recovery_timeout_s']+1e-5)
        deadline=self._controlled&(self.elapsed>=TASK['recovery_timeout_s']-1e-5)&~self.success
        self.reason_masks['deadline']=deadline
        self.failed|=deadline
        # Missing a real task deadline is terminal failure, not a bootstrapped timeout.
        timeout=torch.zeros_like(self.failed)
        self.peak_heading=torch.maximum(self.peak_heading,m['heading_error'].abs()*self._controlled)
        self.peak_roll=torch.maximum(self.peak_roll,m['roll'].abs()*self._controlled)
        self.peak_coaxial=torch.maximum(self.peak_coaxial,m['axle_offset']*self._controlled[:,None])
        self.m=m
        self.terminal_obs=torch.cat((self.history[:,1:],self.frame(m)[:,None,:]),1).flatten(1).clone()
        return self.failed|self.success,timeout

    def _get_rewards(self):
        m=self.m;d=self.robot.data;b=m['blend'];controlled=self._controlled.float()
        phi=self.potential(m);progress=phi-self.old_phi;self.old_phi=phi.clone()
        stand=torch.exp(-((m['pitch']-self.goal_pitch)/.15).square()-(m['roll']/.15).square()
            -((m['height']-self.goal_height)/.055).square()-m['q_error']/.20**2)
        density=REWARDS['pitch_tracking']*torch.exp(-((m['pitch']-m['pitch_reference'])/.40).square())
        density+=REWARDS['height_tracking']*torch.exp(-((m['height']-m['height_reference'])/.12).square())
        guide=self.joint_guide(m['pitch'],m['phase'])
        density+=REWARDS['joint_guidance']*torch.exp(-(d.joint_pos.torch[:,self.legs]-guide).square().mean(-1)/.5**2)
        density+=REWARDS['standing_pose']*stand+REWARDS['four_feet']*stand*m['contact'].mean(-1)
        density+=REWARDS['standing_hold']*(self.hold>0)
        density+=REWARDS['heading']*(m['heading_error']/.20).square().clamp(max=9)
        density+=REWARDS['body_symmetry']*(m['symmetry_rms']/.15).square().clamp(max=9)
        density+=REWARDS['body_roll']*(m['roll']/.15).square().clamp(max=9)
        density+=REWARDS['horizontal_yaw_rate']*(m['yaw_rate']/.5).square().clamp(max=9)
        density+=REWARDS['front_axle_speed']*(m['front_speed']/.15).square().mean(-1).clamp(max=9)
        density+=REWARDS['rear_landing_direction']*m['wrong_landing_side']
        density+=REWARDS['wheel_clearance']*((.22-m['wheel_gap']).clamp(min=0)/.05).square().clamp(max=9)
        density+=REWARDS['front_coaxial']*(torch.sqrt(1+(m['axle_offset'][:,0]/.04).square())-1).clamp(max=9)
        density+=REWARDS['rear_coaxial']*(torch.sqrt(1+(m['axle_offset'][:,1]/.04).square())-1).clamp(max=9)
        density+=REWARDS['rear_axis_parallel']*(m['axle_angle'][:,1]/.174533).square().clamp(max=9)
        density+=REWARDS['sagittal_tilt']*(m['sagittal_tilt']/.12).square().clamp(max=9)
        density+=REWARDS['sagittal_roll_rate']*(m['sagittal_roll_rate']/.5).square().clamp(max=9)
        density+=REWARDS['rear_lateral_offset']*(m['rear_lateral']/.05).square().clamp(max=9)*m['rear_gate']
        density+=REWARDS['rear_touch_sync']*((m['clearance'][:,2]-m['clearance'][:,3])/.08).square().clamp(max=9)*m['rear_gate']
        approach=((-m['foot_vz'][:,2:]-.45).clamp(min=0)/.6).square().clamp(max=9)
        density+=REWARDS['rear_approach_speed']*(approach*((.20-m['clearance'][:,2:])/.17).clamp(0,1)*(~self.previous_contacts[:,2:])).mean(-1)
        density+=REWARDS['angular_speed']*(m['omega']/2).square().clamp(max=9)
        density+=REWARDS['pose_angular_speed']*(self.pose_rate/2).square().clamp(max=9)
        density+=REWARDS['descent_speed']*((-m['vz']-.35).clamp(min=0)/.5).square().clamp(max=9)
        density+=REWARDS['rear_impact']*(m['peak'][:,2:]/self.weight[:,None]-TASK['rear_impact_target_bodyweights']).clamp(min=0).square().mean(-1)
        density+=REWARDS['force_slew']*((m['normal']-self.previous_normal_force)/self.weight[:,None]).square().mean(-1).clamp(max=9)
        density+=REWARDS['air_excess']*((self.air_time-TASK['allowed_air_s']).clamp(min=0)/.2).square()
        density+=(REWARDS['action_rate']+b*REWARDS['settled_action_rate'])*(self.action-self.old_action).square().mean(-1)
        density+=REWARDS['joint_speed']*(m['leg_rms']/4).square().clamp(max=9)
        density+=REWARDS['wheel_speed']*(m['wheel_rms']/6).square().clamp(max=9)
        density+=REWARDS['torque']*(self.torque/CONTROL['effort_nm']).square().mean(-1)
        density+=REWARDS['front_parallel']*(m['front_axis_angle']/TASK['front_axis_penalty_rad']).square().clamp(max=9)
        travel=(m['feet_w'][:,:2,:2]-self.front_anchor).norm(dim=-1)
        density+=REWARDS['front_travel']*((travel-.02).clamp(min=0)/.10).square().mean(-1).clamp(max=9)
        density+=REWARDS['standing_translation']*stand*(m['vxy']/.20).square().clamp(max=9)
        density+=REWARDS['slip']*m['slip'].mean(-1).clamp(max=4)+REWARDS['time']
        touch=self.rear_touch.float()
        touch_bonus=REWARDS['rear_soft_touch_bonus']*(touch*torch.exp(-(self.touchdown_speed/TASK['rear_touch_speed_target_m_s']).square())).sum(-1)
        touch_cost=REWARDS['touchdown_speed']*(touch*(self.touchdown_speed-.35).clamp(min=0).square()).sum(-1)
        reward=(density*self.step_dt+REWARDS['return_progress']*progress+touch_bonus+touch_cost
            +(REWARDS['success_bonus']+REWARDS['speed_success_bonus']*(1-self.elapsed/TASK['recovery_timeout_s']).clamp(0,1))*self.success+REWARDS['failure']*self.failed)*controlled
        self.returns+=reward
        active=controlled.sum().clamp(min=1)
        avg=lambda v:float((v*controlled).sum()/active)
        self.extras['log']=dict(wheel_gap_cm=avg(m['wheel_gap'])*100,front_coaxial_cm=avg(m['axle_offset'][:,0])*100,rear_coaxial_cm=avg(m['axle_offset'][:,1])*100,
            rear_axis_deg=avg(m['axle_angle'][:,1])*180/math.pi,sagittal_abs_deg=avg(m['sagittal_tilt'].abs())*180/math.pi,
            rear_lateral_cm=avg(m['rear_lateral'].abs())*100,
            active_fraction=float(controlled.mean()),handoffs=self.last_step_handoffs,
            total_handoffs=self.handoffs,teacher_resets=self.teacher_resets,level=self.level,
            symmetry_rms_rad=avg(m['symmetry_rms']),roll_abs_deg=avg(m['roll'].abs())*180/math.pi,heading_abs_deg=avg(m['heading_error'].abs())*180/math.pi,front_travel_m=avg(travel.mean(-1)),front_speed_m_s=avg(m['front_speed'].mean(-1)),wrong_landing_side=avg(m['wrong_landing_side']),height=avg(m['height']),pitch_deg=avg(m['pitch'])*180/math.pi,standing_pose=avg(stand),
            rear_landed_fraction=avg((self.landing_time>=0).all(-1).float()),four_contact_fraction=avg(m['contact'].amin(-1)),
            hold_s=avg(self.hold),rear_peak_bw=avg(self.rear_impact_peak),touchdown_speed=avg(self.episode_touchdown_speed),
            raw_angular_speed=avg(m['omega']),pose_angular_speed=avg(self.pose_rate),
            jitter_at_handoff_leg=avg(self.handoff_jitter[:,0]),jitter_at_handoff_body=avg(self.handoff_jitter[:,1]),
            handoff_without_detected_jitter=avg(self.handoff_without_jitter.float()),max_air_s=avg(self.max_air))
        self.previous_contacts=m['contact']>.5;self.previous_foot_vz=m['foot_vz'].clone()
        self.previous_normal_force=m['normal'].clone()
        return reward

    def frame(self,m=None):
        m=self.metrics() if m is None else m;d=self.robot.data
        heading=m['heading_error'];phase=(self.elapsed/self.guidance_duration.clamp(min=.1)).clamp(0,1)
        frame=torch.cat((d.projected_gravity_b.torch,d.root_ang_vel_b.torch*.25,
            d.joint_pos.torch[:,self.legs]-self.goal_q,d.joint_vel.torch[:,self.ids]*.05,self.action,
            self.target-self.goal_target,self.wheel_target*.05,self.goal_q-d.joint_pos.torch[:,self.legs],
            d.root_lin_vel_b.torch,m['height'][:,None],m['clearance'],m['contact'],
            torch.stack((heading.sin(),heading.cos()),-1),
            torch.stack((self.recovering.float(),phase),-1),
            (self.landing_time>=0).float().mean(-1,keepdim=True),self.geometry_frame(m)),-1).clamp(-10,10)
        assert frame.shape==(self.num_envs,FRAME),frame.shape
        return frame

    def geometry_frame(self,m):
        root_q=self.robot.data.root_quat_w.torch[:,None,:].expand(-1,2,-1).reshape(-1,4)
        body=lambda v:quat_apply_inverse(root_q,v.reshape(-1,3)).reshape(self.num_envs,6)
        return torch.cat((m['anchor_heading'].flatten(1)/.2,body(m['offset_vector'])/.05,
            body(m['axis_difference'])/.15,m['rear_heading'].flatten(1)/.5),-1)

    def _get_observations(self):
        frame=self.frame();self.history[:,:-1]=self.history[:,1:].clone();self.history[:,-1]=frame
        self.history[self.fill_history]=frame[self.fill_history,None,:]
        self.fill_history[:]=False
        return {'policy':self.history.flatten(1).clone()}

    def _reset_idx(self,ids):
        if ids is None:ids=torch.arange(self.num_envs,device=self.device)
        if hasattr(self,'h') and len(ids):
            completed=ids[self.episode_length_buf[ids]>0]
            active=completed[self._controlled[completed]]
            warm=completed[~self._controlled[completed]];self.teacher_resets+=len(warm)
            if len(active):
                s=(self.success[active]&~self.failed[active]).float()
                self.success_window.extend(s.cpu().tolist())
                self.extras['episode']=dict(completion_s=float(self.elapsed[active].mean()),successful_elapsed_contribution_s=float((self.elapsed[active]*s).mean()),peak_heading_deg=float(self.peak_heading[active].mean())*180/math.pi,
                    peak_roll_deg=float(self.peak_roll[active].mean())*180/math.pi,
                    peak_front_coaxial_cm=float(self.peak_coaxial[active,0].mean())*100,
                    peak_rear_coaxial_cm=float(self.peak_coaxial[active,1].mean())*100,count=len(active),success=float(s.mean()),reward=float(self.returns[active].mean()),
                    best_hold_s=float(self.best_hold[active].mean()),rear_touchdown_speed=float(self.episode_touchdown_speed[active].mean()),
                    rear_peak_bw=float(self.rear_impact_peak[active].mean()),
                    **{'fail_'+key:float(value[active].float().mean()) for key,value in self.reason_masks.items()})
        super()._reset_idx(ids)
        if not hasattr(self,'h'):return
        n=len(ids);pick=torch.randint(len(self.bank_json),(n,),device=self.device)
        b={key:value[pick] for key,value in self.bank.items()}
        pose=b['root_pose'].clone();pose[:,:3]+=self.scene.env_origins[ids]
        q=self.robot.data.default_joint_pos.torch[ids].clone();q[:,self.ids]=b['q']
        dq=torch.zeros_like(q);dq[:,self.ids]=b['dq']
        velocity=b['root_velocity'].clone()
        velocity[:,:3]+=(torch.rand(n,3,device=self.device)*2-1)*(.005+.025*self.level)
        velocity[:,3:]+=(torch.rand(n,3,device=self.device)*2-1)*(.01+.09*self.level)
        self.robot.write_root_pose_to_sim_index(root_pose=pose,env_ids=ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=velocity,env_ids=ids)
        self.robot.write_joint_position_to_sim_index(position=q,env_ids=ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=dq,env_ids=ids);self.robot.reset(ids)
        for name in ('action','old_action','older_action','target','target_velocity','wheel_target','delayed'):
            getattr(self.h,name)[ids]=b[name]
        self.h.front_anchor[ids]=b['front_anchor']+self.scene.env_origins[ids,None,:2]
        self.h.anchor_pending[ids]=False;self.h.fill_history[ids]=True
        self.h.history[ids]=0;self.h.episode_length_buf[ids]=b['clock_step'].long()
        self.action[ids]=b['action'];self.old_action[ids]=b['old_action'];self.target[ids]=b['target']
        self.target_velocity[ids]=b['target_velocity'];self.wheel_target[ids]=b['wheel_target']
        self.front_anchor[ids]=self.h.front_anchor[ids]
        self.heading_goal[ids]=0.
        x,y,z,w=pose[:,3:].unbind(-1)
        self.heading_goal[ids]=torch.atan2(2*(w*z-x*y),1-2*(x*x+z*z))
        self.previous_quat[ids]=pose[:,3:]
        self.recovering[ids]=False;self._controlled[ids]=False
        self.handoff_target[ids]=b['target'];self.handoff_pitch[ids]=b['pitch_rad']
        self.handoff_height[ids]=b['root_pose'][:,2]
        for tensor in (self.elapsed,self.warmup_age,self.torque,self.air_time,self.max_air,self.axis_bad_time,
                       self.body_bad_time,self.hold,self.best_hold,self.returns,self.old_phi,self.pose_rate,self.peak_heading,self.peak_roll,self.peak_coaxial,
                       self.jitter_ema,self.handoff_jitter,self.touchdown_speed,self.rear_impact_peak,
                       self.episode_touchdown_speed,self.previous_foot_vz,self.previous_normal_force):tensor[ids]=0.
        self.pitch_rate[ids]=0.;self.previous_pitch[ids]=b['pitch_rad'];self.u_blend[ids]=0.
        self.landing_time[ids]=-1.;self.previous_contacts[ids]=False;self.previous_contacts[ids,:2]=True
        self.rear_touch[ids]=False;self.handoff_without_jitter[ids]=False
        self.success[ids]=False;self.failed[ids]=False
        max_warm=.85+(TASK['warmup_max_s']-.85)*self.level
        self.warmup_duration[ids]=TASK['warmup_min_s']+torch.rand(n,device=self.device)*(max_warm-TASK['warmup_min_s'])
        self.guidance_duration[ids]=TASK['guidance_easy_s']+(TASK['guidance_hard_s']-TASK['guidance_easy_s'])*self.level
        self.fill_history[ids]=True;self.history[ids]=0.
