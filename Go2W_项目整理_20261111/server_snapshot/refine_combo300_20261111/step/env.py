"""Blind actor, privileged critic, no trajectory imitation and no state edits outside reset."""
import torch
from period_math import reference_phase, sample_period
from locomotion_guidance import swing_velocity
from paired_symmetry import paired_costs
from symmetry import mirror_action
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_from_euler_xyz, quat_apply, quat_apply_inverse
from schedule import Commands, Schedule, wrap
from config import CONTROL as C, LEGS, WHEELS, FEET, HISTORY, FRAME, GAIT

def yaw_of(q):
    x,y,z,w=q.unbind(-1)
    return torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))

def horizontal(vector, yaw):
    co,si=yaw.cos(),yaw.sin()
    return torch.stack((co*vector[:,0]+si*vector[:,1],-si*vector[:,0]+co*vector[:,1]),-1)

class LocomotionEnv(DirectRLEnv):
    def __init__(self,cfg):
        super().__init__(cfg)
        self._physics_handles_decimation=False
        self.leg_ids,_=self.robot.find_joints(LEGS,preserve_order=True)
        self.wheel_ids,_=self.robot.find_joints(WHEELS,preserve_order=True)
        self.feet,_=self.robot.find_bodies(FEET,preserve_order=True)
        self.contact_ids,_=self.contacts.find_bodies(FEET,preserve_order=True)
        self.other_ids=[i for i,n in enumerate(self.contacts.body_names) if n not in FEET]
        self.head_ids=[i for i,n in enumerate(self.contacts.body_names) if 'head' in n.lower()]
        assert len(self.leg_ids)==12 and len(self.wheel_ids)==4 and len(self.feet)==4
        self.ids=self.leg_ids+self.wheel_ids
        n,dev=self.num_envs,self.device
        assert n%2==0
        self.phase_offset=(torch.arange(n,device=dev)%2)*torch.pi
        self.gait_period=torch.full((n,),.30,device=dev)
        z=lambda *shape:torch.zeros(*shape,device=dev)
        self.stance_feet_reference=torch.tensor(GAIT['stance_feet_reference'],device=dev)
        self.wheel_center_offsets=torch.tensor(GAIT['wheel_center_offsets'],device=dev)
        self.U_stance=torch.tensor(GAIT['U_stance_q'],device=dev)
        self.stand=torch.tensor(C['stand'],device=dev)
        self.scale=torch.tensor(C['leg_scale'],device=dev)
        self.limits=self.robot.data.soft_joint_pos_limits.torch[:,self.leg_ids].clone()
        self.command=Commands(n,dev)
        self.keys=z(n,3)
        self.until_sample=torch.zeros(n,dtype=torch.long,device=dev)
        self.target=self.stand.expand(n,-1).clone()
        self.wheel_target=z(n,4)
        self.action,self.previous_action=z(n,16),z(n,16)
        self.torque=z(n,16)
        self.history=z(n,HISTORY,FRAME)
        self.fill_history=torch.ones(n,dtype=torch.bool,device=dev)
        self.anchor=z(n,2)
        self.rotation_anchor=z(n,2)
        self.rotation_pending=torch.ones(n,dtype=torch.bool,device=dev)
        self.anchor_mode=torch.zeros(n,dtype=torch.bool,device=dev)
        self.air_time=z(n,4)
        self.was_contact=torch.ones(n,4,dtype=torch.bool,device=dev)
        self.clearance_history=torch.zeros(n,3,4,device=dev)
        self.lift_history=torch.zeros(n,3,4,dtype=torch.bool,device=dev)
        self.bad_time=z(n)
        self.gain=torch.ones(n,2,device=dev)
        self.returns=z(n)
        self.rollouts=[]
        self.pd_updates=0
        self.settling_steps=25
        self.stage=1
        self.schedule=Schedule(n,dev,12345)

    def _setup_scene(self):
        self.robot=self.scene.articulations['robot']
        self.contacts=self.scene.sensors['contacts']

    def _sample_commands(self,ids):
        n=len(ids)
        if not n:return
        # 20% rest, 40% single axis, 30% two axes, 10% all axes.
        r=torch.rand(n,device=self.device)
        count=torch.where(r<.2,0,torch.where(r<.6,1,torch.where(r<.9,2,3)))
        rank=torch.rand(n,3,device=self.device).argsort(-1).argsort(-1)
        axes=(rank<count[:,None]).float()*(torch.randint(0,2,(n,3),device=self.device)*2-1)
        self.keys[ids]=axes
        self.until_sample[ids]=torch.randint(45,130,(n,),device=self.device)

    def _pre_physics_step(self,actions):
        if actions.shape[-1]!=12:raise ValueError('Locked-wheel policy needs 12 leg actions')
        actions=torch.cat((actions,torch.zeros(self.num_envs,4,device=self.device)),dim=-1)
        if self.pd_updates not in (0,4):raise RuntimeError(f'PD substeps {self.pd_updates}')
        self.pd_updates=0
        self.extras.pop('episode',None)
        self.keys.copy_(self.schedule.tick(self.stage,self.episode_length_buf*self.step_dt,self.step_dt))
        self.command.backward_limit=1.5  # Reference lower bound is enforced by Commands.tick.
        self.command.yaw_cap=getattr(self,'qe_speed_cap',1.6) if self.cfg.training else 1.6
        self.command.tick(self.keys,self.step_dt)
        d=self.robot.data
        # Anchor only pure rotation / stationary modes; never anchor normal translation.
        fixed=self.command.velocity[:,:2].norm(dim=-1)<.025
        entering=fixed & ~self.anchor_mode
        self.anchor[entering]=d.root_pos_w.torch[entering,:2]
        self.anchor_mode=fixed
        pending=entering|self.rotation_pending
        self.rotation_anchor[pending]=self.physical()['com_position'][pending]
        self.rotation_pending[:]=False
        self.previous_action.copy_(self.action)
        self.action=actions.detach().clamp(-1,1)
        goal=(self.stand+self.scale*self.action[:,:12]).clamp(self.limits[...,0],self.limits[...,1])
        self.target_rate_fraction=((goal-self.target).abs()>C['leg_rate']*self.step_dt).float().mean(-1)
        self.target+=(goal-self.target).clamp(-C['leg_rate']*self.step_dt,C['leg_rate']*self.step_dt)
        y=torch.tensor([-C['half_track'],C['half_track']]*2,device=self.device)
        wheel_goal=torch.zeros_like(self.wheel_target)
        self.wheel_target+=(wheel_goal-self.wheel_target).clamp(-C['wheel_accel']*self.step_dt,
                                                              C['wheel_accel']*self.step_dt)

    def _apply_action(self):
        self.pd_updates+=1
        d=self.robot.data
        dq=d.joint_vel.torch[:,self.ids]
        leg=C['kp']*self.gain[:,:1]*(self.target-d.joint_pos.torch[:,self.leg_ids])-C['kd']*self.gain[:,1:] *dq[:,:12]
        # PhysX implicit zero-velocity drive provides the high damping. Do not
        # apply -100*dq explicitly at 5ms; wheel inertia makes that unstable.
        wheel=torch.zeros_like(dq[:,12:])
        self.robot.set_joint_velocity_target(torch.zeros_like(wheel),joint_ids=self.wheel_ids)
        self.raw_torque=torch.cat((leg,wheel),-1)
        self.torque=self.raw_torque.clamp(-C['effort'],C['effort'])
        self.robot.set_joint_effort_target(self.torque,joint_ids=self.ids)

    def physical(self):
        d=self.robot.data
        yaw=yaw_of(d.root_quat_w.torch)
        velocity=horizontal(d.root_lin_vel_w.torch,yaw)
        height=d.root_pos_w.torch[:,2]-self.scene.env_origins[:,2]
        history=self.contacts.data.net_forces_w_history.torch
        contact=history[:,:,self.contact_ids,2].amax(1)>3.
        bad=history[:,:,self.other_ids].norm(dim=-1).amax(dim=(1,2))
        head=history[:,:,self.head_ids].norm(dim=-1).amax(dim=(1,2)) if self.head_ids else torch.zeros_like(bad)
        collision=[]
        for name in ('front_right_rear','front_left_rear'):
            matrix=self.scene.sensors[name].data.normal_force_matrix_w_history
            if matrix is None:raise RuntimeError('Wheel-pair collision sensor unavailable')
            collision.append(matrix.torch.norm(dim=-1).flatten(1).amax(-1))
        pair_force=torch.stack(collision,-1).amax(-1)
        wheel_quat=d.body_link_quat_w.torch[:,self.feet]
        axis=torch.zeros(self.num_envs,4,3,device=self.device);axis[:,:,1]=1.
        axis=quat_apply(wheel_quat,axis)
        offset=torch.zeros_like(axis)
        offset[:,:,1]=torch.tensor([-.048,.048,-.048,.048],device=self.device)
        offset=quat_apply(wheel_quat,offset)
        offset[:,:,2]-=C['wheel_radius']
        point_velocity=d.body_link_lin_vel_w.torch[:,self.feet]+torch.cross(
            d.body_link_ang_vel_w.torch[:,self.feet],offset,dim=-1)
        axis[:,:,2]=0
        axis=axis/axis.norm(dim=-1,keepdim=True).clamp(min=.1)
        lateral_slip=((point_velocity*axis).sum(-1).square()*contact).mean(-1)
        masses=d.body_mass.torch;weights=masses/masses.sum(-1,keepdim=True)
        com_position=(d.body_com_pos_w.torch*weights[...,None]).sum(1)[:,:2]
        com_velocity=horizontal((d.body_com_lin_vel_w.torch*weights[...,None]).sum(1),yaw)
        fq=d.body_link_quat_w.torch[:,self.feet]
        centers=d.body_link_pos_w.torch[:,self.feet]+quat_apply(fq.reshape(-1,4),self.wheel_center_offsets[None].expand(self.num_envs,-1,-1).reshape(-1,3)).reshape(self.num_envs,4,3)
        rq=d.root_quat_w.torch[:,None,:].expand(-1,4,-1).reshape(-1,4)
        feet_b=quat_apply_inverse(rq,(centers-d.root_pos_w.torch[:,None,:]).reshape(-1,3)).reshape(self.num_envs,4,3)
        return dict(feet_b=feet_b,com_position=com_position,com_velocity=com_velocity,yaw=yaw,velocity=velocity,height=height,contact=contact,bad=bad,head=head,
            lateral_slip=lateral_slip, foot_point_velocity=point_velocity,
            pair_force=pair_force,heading_error=wrap(self.command.heading-yaw),
            anchor_error=horizontal(d.root_pos_w.torch[:,:2]-self.anchor,yaw),
            yaw_rate=d.root_ang_vel_w.torch[:,2])

    def frame(self,p):
        d=self.robot.data
        result=torch.cat((d.projected_gravity_b.torch,d.root_ang_vel_b.torch*.25,
            d.joint_pos.torch[:,self.leg_ids]-self.stand,d.joint_vel.torch[:,self.ids]*.05,
            self.action,self.command.velocity,p['heading_error'].sin()[:,None],
            p['heading_error'].cos()[:,None],
            torch.sin((reference_phase(self.episode_length_buf*self.step_dt,self.gait_period)+self.phase_offset))[:,None],
            torch.cos((reference_phase(self.episode_length_buf*self.step_dt,self.gait_period)+self.phase_offset))[:,None]),-1).clamp(-10,10)

        motion_gain=torch.maximum(self.command.velocity[:,:2].norm(dim=-1),self.command.velocity[:,2].abs()).div(.15).clamp(0,1)
        result[:,-2:]*=motion_gain[:,None]
        return result

    def critic(self,frame,p):
        return torch.cat((frame,p['velocity'],self.robot.data.root_lin_vel_w.torch[:,2:3],
            p['yaw_rate'][:,None],p['height'][:,None],p['contact'].float(),p['anchor_error']),-1)

    def _get_observations(self):
        p=self.physical()
        f=self.frame(p)
        self.history[:,:-1]=self.history[:,1:].clone()
        self.history[:,-1]=f
        self.history[self.fill_history]=f[self.fill_history,None,:].expand(-1,HISTORY,-1)
        self.fill_history[:]=False
        assert f.shape[-1]==FRAME
        return {'policy':self.history.flatten(1).clone(),'critic':self.critic(f,p)}

    def _get_dones(self):
        p=self.physical()
        self._physical=p
        self.bad_time=torch.where(p['bad']>30.,self.bad_time+self.step_dt,0.)
        d=self.robot.data
        self.failed=(p['height']<.19)|(p['height']>.75)|(d.projected_gravity_b.torch[:,2]>-.65)
        self.failed|=(self.bad_time>.06)|(p['head']>10.)|(p['pair_force']>.1)
        self.failed|=~torch.isfinite(d.root_state_w.torch).all(-1)
        # Reset both trajectories together, so physical comparisons remain phase-aligned.
        if self.cfg.training:
            self.failed=self.failed.reshape(-1,2).any(-1).repeat_interleave(2)
        self.terminal_critic=self.critic(self.frame(p),p).clone()
        timeout=self.episode_length_buf>=self.max_episode_length
        return self.failed,timeout

    def _get_rewards(self):
        p,d=self._physical,self.robot.data
        cmd=self.command.velocity
        err=torch.cat((p['velocity']-cmd[:,:2],(p['yaw_rate']-cmd[:,2])[:,None]),-1)
        no_yaw=cmd[:,2].abs()<.03
        no_translation=cmd[:,:2].norm(dim=-1)<.025
        idle=no_yaw & no_translation
        qe_task=self.schedule.qe_mask if hasattr(self.schedule,'qe_mask') else torch.zeros_like(idle)
        rotation_task=no_translation & (qe_task | ~no_yaw)
        rotation_stop=idle & qe_task
        rotation_moving=no_translation & ~no_yaw
        q=d.joint_pos.torch[:,self.leg_ids]
        dq=d.joint_vel.torch[:,self.ids]
        # A mild touchdown reward encourages stepping for lateral commands without a prescribed gait.
        touchdown=p['contact'] & ~self.was_contact
        self.air_time+=self.step_dt
        air_reward=((self.air_time-.08).clamp(0,.25)*touchdown).sum(-1)
        self.air_time[p['contact']]=0
        self.was_contact=p['contact'].clone()
        terms=dict(
            track_x=GAIT['track_peak_weight']*torch.exp(-(err[:,0]/.4).square())-GAIT['track_error_weight']*err[:,0].square().clamp(max=9),
            track_y=GAIT['track_peak_weight']*torch.exp(-(err[:,1]/.4).square())-GAIT['track_error_weight']*err[:,1].square().clamp(max=9),
            track_yaw=1.8*torch.exp(-(err[:,2]/.4).square())-.4*err[:,2].square().clamp(max=4),
            heading=-.8*(p['heading_error']/.2).square().clamp(max=9),
            anchor=-.2*no_translation*torch.where(~rotation_task,p['anchor_error'].norm(dim=-1),
                (p['com_position']-self.rotation_anchor).norm(dim=-1)).div(.15).square().clamp(max=9),
            rotation_com_drift=-GAIT['rotation_drift_weight']*rotation_task*((p['com_position']-self.rotation_anchor).norm(dim=-1)/.15).square().clamp(max=9),
            rotation_com_speed=-1.*rotation_task*(p['com_velocity'].norm(dim=-1)/.2).square().clamp(max=9),
            rotation_stop_yaw=-GAIT['rotation_stop_yaw']*rotation_stop*(p['yaw_rate']/.25).square().clamp(max=9),
            rotation_stop_support=GAIT['rotation_stop_support']*rotation_stop*p['contact'].float().mean(-1),
            rotation_stop_joint=-.004*rotation_stop*dq[:,:12].square().mean(-1),
            idle=.5*idle*torch.exp(-p['velocity'].square().sum(-1)/.01-p['yaw_rate'].square()/.04),
            upright=-GAIT['upright_weight']*d.projected_gravity_b.torch[:,:2].square().sum(-1),
            height=-20.*(p['height']-.395).square(),
            vertical=-d.root_lin_vel_w.torch[:,2].square(),
            roll_pitch_rate=-.03*d.root_ang_vel_b.torch[:,:2].square().sum(-1),
            posture=-GAIT['posture_weight']*(q-self.stand).square().mean(-1),
            action_rate=-.02*(self.action[:,:12]-self.previous_action[:,:12]).square().mean(-1)
                        -.005*(self.action[:,12:]-self.previous_action[:,12:]).square().mean(-1),
            effort=-.015*(self.torque/C['effort']).square().mean(-1),
            power=-.0001*(self.torque*dq).abs().sum(-1),
            lateral_slip=-.05*p['lateral_slip'],
            saturation=-.05*(self.raw_torque.abs()>C['effort']).float().mean(-1),
            stepping=.5*(cmd[:,1].abs()>.04)*air_reward,
            idle_joint_speed=-.002*idle*dq[:,:12].square().sum(-1))
        # Broad soft ranges permit stepping, penalizing only excessive front folding.
        backward=cmd[:,0]<-.05
        front=q[:,:6].reshape(-1,2,3)
        fold=((front[:,:,1]-1.45).clamp(min=0)/.35).square()
        fold+=((-2.0-front[:,:,2]).clamp(min=0)/.35).square()
        ramp=getattr(self,'diagonal_progress',1.)
        terms['low_height']=-ramp*GAIT['low_height_weight']*((GAIT['height_lower']-p['height']).clamp(min=0)/.04).square().clamp(max=4)
        terms['high_height']=-ramp*.2*((p['height']-GAIT['height_upper']).clamp(min=0)/.04).square().clamp(max=4)
        terms['front_fold']=-ramp*GAIT['front_fold_weight']*(cmd[:,:2].norm(dim=-1)>.05)*fold.mean(-1).clamp(max=4)
        # Reference-speed gate stays strong near rest and fades continuously
        # at walking speed. No stand/action normalization changes.
        low_speed=torch.exp(-(cmd[:,:2].norm(dim=-1)/GAIT['U_stance_speed_scale']).square()-(cmd[:,2]/.2).square())
        widths=torch.tensor([.2,.35,.35]*4,device=self.device)
        U_error=((q-self.U_stance)/widths).square().mean(-1)
        stance_ramp=getattr(self,'stance_progress',1.)
        terms['U_stance']=stance_ramp*low_speed*(GAIT['U_stance_weight']*torch.exp(-U_error)-.4*U_error.clamp(max=4))
        terms['U_stance_height']=stance_ramp*.4*low_speed*torch.exp(-((p['height']-GAIT['U_stance_height'])/.03).square())
        terms['U_stance_calm']=-stance_ramp*.004*low_speed*dq[:,:12].square().mean(-1)
        # Weak soft guidance, scaled by the longitudinal fraction of the command.
        share=cmd[:,0].abs()/(cmd[:,:2].abs().sum(-1)+1e-6)
        sagittal=((p['feet_b'][:,:,1]-self.stance_feet_reference[:,1])/.08).square().mean(-1).clamp(max=4)
        terms['sagittal_plane']=-GAIT['sagittal_weight']*torch.exp(-(cmd[:,2]/.25).square())*share*(cmd[:,:2].norm(dim=-1)>.1)*sagittal
        hips=q[:,[0,3,6,9]];neutral=self.U_stance[[0,3,6,9]]
        signs=torch.tensor([-1.,1.,-1.,1.],device=self.device)
        outward=(signs*(hips-neutral)-GAIT['forward_outward_hip_deadband']).clamp(min=0)
        forward_share=cmd[:,0].clamp(min=0)/(cmd[:,:2].abs().sum(-1)+.25*cmd[:,2].abs()+1e-6)
        forward_gate=forward_share*(cmd[:,0]/.3).clamp(0,1)
        terms['forward_outward_hip']=-GAIT['forward_outward_hip_weight']*forward_gate*(outward/GAIT['forward_outward_hip_scale']).square().mean(-1).clamp(max=9)
        reward=sum(terms.values())*self.step_dt-GAIT['failure_penalty']*self.failed.float()
        self.returns+=reward
        self.extras['log']={
            'forward/outward_hip_rad':outward.mean().detach(),'forward/hip_gate':forward_gate.mean().detach(),
            'stance/low_speed_weight':low_speed.mean().detach(),
            'stance/q_rmse_rad':((q-self.U_stance).square().mean(-1)*low_speed).sum().div(low_speed.sum().clamp(min=1)).sqrt().detach(),
            'stance/height_error_m':((p['height']-GAIT['U_stance_height']).abs()*low_speed).sum()/low_speed.sum().clamp(min=1),
            'stance/ramp':torch.tensor(stance_ramp,device=self.device),
            'backward/cap_m_s':torch.tensor(self.command.backward_limit,device=self.device),
            'backward/samples':backward.sum().detach(),
            'backward/rmse_x':((err[:,0].square()*backward).sum()/backward.sum().clamp(min=1)).sqrt().detach(),
            'backward/low_height_fraction':((p['height']<.365)*backward).sum()/backward.sum().clamp(min=1),
            'backward/front_fold_fraction':((fold.max(-1).values>0)*backward).sum()/backward.sum().clamp(min=1),
            **{'reward/'+k:v.mean().detach() for k,v in terms.items()},
            'track/rmse_x':err[:,0].square().mean().sqrt().detach(),
            'track/rmse_y':err[:,1].square().mean().sqrt().detach(),
            'track/rmse_yaw':err[:,2].square().mean().sqrt().detach(),
            'gait/period_min_s':self.gait_period.min().detach(),
            'gait/period_max_s':self.gait_period.max().detach(),
            'physical/height':p['height'].mean().detach(),
            'physical/failure_fraction':self.failed.float().mean().detach()}
        # Direction-separated rollout sums; never hide an absent direction with NaN.
        masks={'forward':cmd[:,0]>.05,'backward':cmd[:,0]<-.05,
               'left':cmd[:,1]>.05,'right':cmd[:,1]<-.05}
        for name,mask in masks.items():
            self.extras['log'][f'direction/{name}/count']=mask.sum().detach()
            self.extras['log'][f'direction/{name}/error_sq_sum']=(err[:,:2].square().sum(-1)*mask).sum().detach()
            self.extras['log'][f'direction/{name}/failed_count']=(self.failed*mask).sum().detach()
        # Diagonal pairs: FR+RL, FL+RR. Reward support/swing timing and actual
        # wheel-bottom clearance; no kinematic overrides or root forces.
        phase=(reference_phase(self.episode_length_buf*self.step_dt,self.gait_period)+self.phase_offset)
        offsets=torch.tensor([0.,torch.pi,torch.pi,0.],device=self.device)
        wave=torch.sin(phase[:,None]+offsets)
        swing=wave>0
        moving=(cmd[:,:2].norm(dim=-1)>.05)|(cmd[:,2].abs()>.1)
        clearance=(d.body_link_pos_w.torch[:,self.feet,2]-self.scene.env_origins[:,2:3]-C['wheel_radius']).clamp(min=0)
        desired=GAIT['swing_height']*wave.clamp(min=0)
        phase_boundary=wave.abs()<torch.sin(2*torch.pi*.04/self.gait_period)[:,None]
        contact_reward=((p['contact']==~swing)|phase_boundary).float().mean(-1)
        clearance_reward=torch.exp(-((clearance-desired)/.03).square()).mean(-1)
        guidance=moving*(GAIT['contact_weight']*contact_reward+GAIT['clearance_weight']*clearance_reward)
        # Soft amplitude costs retain the ability to step sideways and balance.
        hips=q[:,[0,3,6,9]]
        hip_angle_cost=GAIT['hip_angle_weight']*((hips/.35).square().mean(-1))
        hip_speed_cost=GAIT['hip_speed_weight']*((dq[:,[0,3,6,9]]/4.).square().mean(-1))
        amplitude_cost=GAIT['swing_amplitude_weight']*((q-self.stand).abs()-.45).clamp(min=0).square().mean(-1)
        excess_clearance_cost=GAIT['excess_clearance_weight']*moving*(
            ((clearance-GAIT['swing_height']).clamp(min=0)/.03).square().mean(-1).clamp(max=9))
        self.extras['log']['gait/excess_clearance_cost']=excess_clearance_cost.mean().detach()
        # Locked wheels require stance contacts to remain near stationary in the world.
        # Average only the intended mid-swing legs, so planted legs cannot mask a missed swing.
        mid_swing=wave>.5
        swing_count=mid_swing.sum(-1).clamp(min=1)
        deficit=(((desired-clearance).clamp(min=0)/.04).square()*mid_swing).sum(-1)/swing_count
        missed=(p['contact'] & mid_swing).float().sum(-1)/swing_count
        slip_speed=p['foot_point_velocity'][...,:2].norm(dim=-1)
        contact_count=p['contact'].sum(-1).clamp(min=1)
        slip_cost=(((slip_speed-.08).clamp(min=0)/.75).square().clamp(max=4)*p['contact']).sum(-1)/contact_count
        ramp=getattr(self,'gait_progress',1.)
        new_cost=moving*ramp*(GAIT['swing_deficit_weight']*deficit.clamp(max=4)
            +GAIT['swing_contact_weight']*missed+GAIT['stance_slip_weight']*slip_cost)
        guidance-=new_cost
        # FR+RL and FL+RR in the declared FR,FL,RR,RL order. Reward
        # simultaneous real clearance only while the opposite pair supports.
        pair_ids=torch.tensor([[0,3],[1,2]],device=self.device)
        pair_h=clearance[:,pair_ids]
        pair_contact=p['contact'][:,pair_ids]
        pair_swing=mid_swing[:,pair_ids].all(-1)
        support=pair_contact.flip(1).float().mean(-1)
        lifted=(pair_h>.015) & ~pair_contact
        self.clearance_history=torch.cat((self.clearance_history[:,1:],clearance[:,None]),1)
        self.lift_history=torch.cat((self.lift_history[:,1:],((clearance>.015)&~p['contact'])[:,None]),1)
        recent_h=self.clearance_history.amax(1)[:,pair_ids]
        recently_lifted=self.lift_history.any(1)[:,pair_ids]
        pair_quality=(recent_h.amin(-1)/.035).clamp(0,1)*support*pair_swing
        mismatch=(recently_lifted.any(-1) & ~recently_lifted.all(-1)).float()
        pair_mask=pair_swing.float()
        pair_count=pair_mask.sum(-1).clamp(min=1)
        paired=(pair_quality.sum(-1)/pair_count)
        unmatched=(mismatch*pair_mask).sum(-1)/pair_count
        drag=(p['contact'].float()*mid_swing*slip_speed.clamp(max=2.)).sum(-1)/swing_count
        diagonal_ramp=getattr(self,'diagonal_progress',1.)
        diagonal=moving*diagonal_ramp*(GAIT['diagonal_reward']*paired
            -GAIT['diagonal_mismatch']*unmatched-GAIT['swing_drag']*drag)
        guidance+=diagonal
        # Swing feet advance tangentially around the current measured COM.
        # Stance feet should remain planted in world coordinates. No target/state overrides.
        yaw4=p['yaw'][:,None].expand(-1,4).reshape(-1)
        point_b=horizontal(p['foot_point_velocity'].reshape(-1,3),yaw4).reshape(self.num_envs,4,2)
        desired_swing=swing_velocity(cmd,p['feet_b'][...,:2])
        error=((point_b-desired_swing)/.45).square().sum(-1)
        tangent_score=(torch.exp(-error)*mid_swing).sum(-1)/swing_count
        rotation_slip=((slip_speed/.2).square().clamp(max=4)*p['contact']).sum(-1)/contact_count
        rotation_guidance=moving*.9*tangent_score-rotation_task*GAIT['rotation_stance_slip']*rotation_slip
        # Signed lateral coordinates avoid rewarding crossed or tightly gathered feet.
        side=torch.tensor([-1.,1.,-1.,1.],device=self.device)
        narrow=((.14-p['feet_b'][...,1]*side).clamp(min=0)/.05).square().mean(-1).clamp(max=4)
        # Swing contact/slip is scored per leg, so a healthy pair cannot hide dragging FL/RR.
        per_leg_drag=(p['contact']*mid_swing*(slip_speed/.3).square().clamp(max=4)).sum(-1)/swing_count
        guidance+=rotation_guidance-moving*(.35*narrow+.45*per_leg_drag)
        for label,mask in (('q',cmd[:,2]>.1),('e',cmd[:,2]<-.1),('forward',(cmd[:,0]>.1)&no_yaw)):
            den=mask.sum().clamp(min=1)
            self.extras['log'][f'quality/{label}/count']=mask.sum()
            self.extras['log'][f'quality/{label}/swing_score']=(tangent_score*mask).sum()/den
            self.extras['log'][f'quality/{label}/yaw_rmse']=((err[:,2].square()*mask).sum()/den).sqrt()
            for k,leg in enumerate(('FR','FL','RR','RL')):
                active=mask&mid_swing[:,k]
                self.extras['log'][f'quality/{label}/{leg}_missed_lift']=(p['contact'][:,k]*active).sum()/active.sum().clamp(min=1)
        self.extras['log']['reward/rotation_guidance']=rotation_guidance.mean().detach()
        self.extras['log']['rotation/tangent_score']=(tangent_score*rotation_moving).sum()/rotation_moving.sum().clamp(min=1)
        self.extras['log']['rotation/stop_yaw_abs']=(p['yaw_rate'].abs()*rotation_stop).sum()/rotation_stop.sum().clamp(min=1)
        self.extras['log']['rotation/stop_four_contact']=(p['contact'].all(-1)*rotation_stop).sum()/rotation_stop.sum().clamp(min=1)
        active_pairs=pair_swing & moving[:,None]
        self.extras['log'].update({
            'gait/diagonal_ramp':torch.tensor(diagonal_ramp,device=self.device),
            'gait/diagonal_both_lifted':(lifted.all(-1)*active_pairs).sum()/active_pairs.sum().clamp(min=1),
            'gait/diagonal_unmatched':(mismatch*active_pairs).sum()/active_pairs.sum().clamp(min=1),
            'gait/swing_drag_m_s':(drag*moving).sum()/moving.sum().clamp(min=1),
            'reward/diagonal':diagonal.mean().detach()})
        active_mid=mid_swing & moving[:,None]
        high_speed=moving & (cmd[:,:2].norm(dim=-1)>1.)
        self.extras['log'].update({
            'gait/reward_ramp':torch.tensor(ramp,device=self.device),
            'gait/swing_deficit_m':((desired-clearance).clamp(min=0)*active_mid).sum()/active_mid.sum().clamp(min=1),
            'gait/missed_swing_contact':((p['contact'] & active_mid).float().sum()/active_mid.sum().clamp(min=1)),
            'gait/stance_slip_m_s':(slip_speed*p['contact']*moving[:,None]).sum()/(p['contact'] & moving[:,None]).sum().clamp(min=1),
            'gait/high_speed_slip_m_s':(slip_speed*p['contact']*high_speed[:,None]).sum()/(p['contact'] & high_speed[:,None]).sum().clamp(min=1),
            'gait/new_cost':new_cost.mean().detach(),
            'gait/target_rate_limit_fraction':self.target_rate_fraction.mean().detach()})
        guidance-=hip_angle_cost+hip_speed_cost+amplitude_cost+excess_clearance_cost
        self.extras['log'].update({
            'smallstep/hip_angle_cost':hip_angle_cost.mean().detach(),
            'smallstep/hip_speed_cost':hip_speed_cost.mean().detach(),
            'smallstep/amplitude_cost':amplitude_cost.mean().detach()})
        if hasattr(self.schedule,'group'):
            for group in range(3):
                mask=self.schedule.group==group
                self.extras['log'][f'group/{group}/count']=mask.sum().detach()
                self.extras['log'][f'group/{group}/rmse_x']=err[mask,0].square().mean().sqrt().detach()
                self.extras['log'][f'group/{group}/rmse_y']=err[mask,1].square().mean().sqrt().detach()
                self.extras['log'][f'group/{group}/actual_vx']=p['velocity'][mask,0].mean().detach()
                self.extras['log'][f'group/{group}/reference_vx']=cmd[mask,0].mean().detach()
        if hasattr(self.schedule,'qe_mask'):
            for name,mask in (('qe',self.schedule.qe_mask),('translation',~self.schedule.qe_mask)):
                self.extras['log'][f'task/{name}/rmse_xy']=((err[:,:2].square().sum(-1)*mask).sum()/mask.sum().clamp(min=1)).sqrt().detach()
                self.extras['log'][f'task/{name}/rmse_yaw']=((err[:,2].square()*mask).sum()/mask.sum().clamp(min=1)).sqrt().detach()
                self.extras['log'][f'task/{name}/failure']=(self.failed.float()*mask).sum().detach()/mask.sum().clamp(min=1)
        if self.cfg.training:
            # Adjacent episodes share command timing, gait period and initial perturbations.
            costs=paired_costs(q,dq[:,:12],d.projected_gravity_b.torch,
                d.root_ang_vel_b.torch,p['feet_b'],p['com_velocity'],p['yaw_rate'])
            paired=sum(weight*costs[key] for key,weight in
                [('pose',.25),('joint_velocity',.15),('body',.15),('feet',.15),('linear',.15),('yaw',.15)])
            guidance-=getattr(self,'symmetry_progress',1.)*paired
            self.extras['log'].update({'mirror/'+key:value.mean().detach() for key,value in costs.items()})
            self.extras['log']['mirror/paired_cost']=paired.mean().detach()
        # Only a 12 mm increase per front foot, as a weak yaw-dependent preference.
        front_signed=p['feet_b'][:,:2,1]*q.new_tensor([-1.,1.])
        desired_width=self.stance_feet_reference[:2,1].abs()+GAIT['front_turn_extra_width_m']
        turn_gate=(cmd[:,2].abs()/.4).clamp(0,1)
        width_cost=((front_signed-desired_width)/.025).square().mean(-1).clamp(max=4)
        guidance-=GAIT['front_turn_width_weight']*turn_gate*width_cost
        self.extras['log']['front_turn/width_m']=(front_signed.sum(-1)*turn_gate).sum()/turn_gate.sum().clamp(min=1)
        reward+=guidance*self.step_dt
        self.returns+=guidance*self.step_dt
        self.extras['log'].update(
            **{'reward/gait_guidance':guidance.mean().detach(),
               'gait/contact_match':contact_reward.mean().detach(),
               'gait/clearance_m':clearance.mean().detach(),
               'lock/wheel_speed_rms':dq[:,12:].square().mean().sqrt().detach()})
        return reward

    def _reset_idx(self,ids):
        if ids is None:ids=torch.arange(self.num_envs,device=self.device)
        if hasattr(self,'returns'):
            active=ids[self.episode_length_buf[ids]>0]
            if len(active):
                self.extras['episode']=dict(count=len(active),
                    return_mean=float(self.returns[active].mean()),
                    duration_s=float(self.episode_length_buf[active].float().mean()*self.step_dt),
                    failure_fraction=float(self.failed[active].float().mean()))
        self.robot.reset(ids)
        super()._reset_idx(ids)
        if not hasattr(self,'command'):return
        if self.cfg.training:
            assert torch.equal(ids[0::2]//2,ids[1::2]//2), 'Incomplete mirrored reset pair'
            self.gait_period[ids]=sample_period(len(ids)//2,self.device,True).repeat_interleave(2)
        else:
            self.gait_period[ids]=sample_period(len(ids),self.device,False,getattr(self,'evaluation_period',.28))
        n=len(ids)
        q=self.robot.data.default_joint_pos.torch[ids].clone()
        root=self.robot.data.default_root_state.torch[ids].clone()
        root[:,:3]=self.scene.env_origins[ids]+torch.tensor([0.,0.,.412],device=self.device)
        root[:,7:]=0
        yaw=torch.zeros(n,device=self.device)
        if self.cfg.training:
            q[:,self.leg_ids]+=(torch.rand(n,12,device=self.device)*2-1)*.015
            yaw=(torch.rand(n,device=self.device)*2-1)*3.14159265
            self.gain[ids]=(.95+torch.rand(n//2,2,device=self.device)*.1).repeat_interleave(2,dim=0)
            q[1::2,self.leg_ids]=mirror_action(q[0::2,self.leg_ids])
            yaw[1::2]=-yaw[0::2]
        else:
            # Distinct, reproducible small initial perturbations, not identical replicas.
            q[:,self.leg_ids]+=(torch.rand(n,12,device=self.device)*2-1)*.01
            yaw=(torch.rand(n,device=self.device)*2-1)*.03
        root[:,3:7]=quat_from_euler_xyz(torch.zeros_like(yaw),torch.zeros_like(yaw),yaw)
        self.robot.write_root_pose_to_sim(root[:,:7],ids)
        self.robot.write_root_velocity_to_sim(root[:,7:],ids)
        self.robot.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=ids)
        self.target[ids]=q[:,self.leg_ids]
        for b in (self.action,self.previous_action,self.wheel_target,self.torque,self.returns,
                  self.bad_time,self.air_time,self.keys):b[ids]=0
        self.clearance_history[ids]=0;self.lift_history[ids]=False
        self.rotation_pending[ids]=True
        self.command.reset(ids,yaw)
        self.until_sample[ids]=self.settling_steps+1
        self.anchor[ids]=root[:,:2]
        self.anchor_mode[ids]=True
        self.was_contact[ids]=True
        self.fill_history[ids]=True
        if hasattr(self,'schedule'):self.schedule.reset(ids)
