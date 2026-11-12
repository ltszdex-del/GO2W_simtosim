"""Blind actor, privileged critic, no trajectory imitation and no state edits outside reset."""
import torch
from locomotion_guidance import support_margin
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_from_euler_xyz, quat_apply, quat_apply_inverse
from schedule import Commands, wrap
from wheel_schedule import Schedule
from dynamic_balance import balance_metrics
from turning import stopping_mask, excess_lateral_slip, residual_yaw_opposition
from symmetry import mirror_action
from geometry_rewards import geometry_costs
from config import COMMAND
C_YAW_LIMIT=COMMAND['yaw_speed_limit']
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
        self.balance_velocity=z(n,2);self.balance_accel=z(n,2)
        self.balance_warm=torch.zeros(n,dtype=torch.bool,device=dev)
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
        if actions.shape[-1]!=16:raise ValueError('Wheel-drive policy needs 16 actions')
        if self.pd_updates not in (0,4):raise RuntimeError(f'PD substeps {self.pd_updates}')
        self.pd_updates=0
        self.extras.pop('episode',None)
        self.keys.copy_(self.schedule.tick(self.stage,self.episode_length_buf*self.step_dt,self.step_dt))
        self.command.backward_limit=1.5  # Reference lower bound is enforced by Commands.tick.
        self.command.speed_cap=getattr(self,'drive_speed_cap',COMMAND['speed_limits'][0]) if self.cfg.training else COMMAND['speed_limits'][0]
        self.command.yaw_cap=getattr(self,'turn_cap',C_YAW_LIMIT) if self.cfg.training else C_YAW_LIMIT
        # Fresh-policy course ramps every turning group to the full yaw envelope.
        if self.cfg.training and hasattr(self.schedule,'combo_mask'):
            progress=getattr(self,'turn_course_progress',0.)
            combo=self.schedule.combo_mask
            self.command.speed_cap=torch.full_like(self.command.velocity[:,0],getattr(self,'drive_speed_cap',2.5))
            self.command.speed_cap=torch.where(self.schedule.low_speed_mask,.35,self.command.speed_cap)
            self.command.yaw_cap=torch.full_like(self.command.velocity[:,2],.6+(C_YAW_LIMIT-.6)*progress)
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
        y=self.physical()['feet_b'][:,:,1]
        wheel_goal=C['wheel_action_scale']*self.action[:,12:]
        decelerating=(self.wheel_target*(wheel_goal-self.wheel_target)<0)
        rate=torch.where(decelerating,C['wheel_brake_accel'],C['wheel_accel'])*self.step_dt
        self.wheel_target+=(wheel_goal-self.wheel_target).clamp(-rate,rate)

    def _apply_action(self):
        self.pd_updates+=1
        d=self.robot.data
        dq=d.joint_vel.torch[:,self.ids]
        leg=C['kp']*self.gain[:,:1]*(self.target-d.joint_pos.torch[:,self.leg_ids])-C['kd']*self.gain[:,1:] *dq[:,:12]
        # PhysX implicit zero-velocity drive provides the high damping. Do not
        # apply -100*dq explicitly at 5ms; wheel inertia makes that unstable.
        wheel=torch.zeros_like(dq[:,12:])
        self.robot.set_joint_velocity_target(self.wheel_target,joint_ids=self.wheel_ids)
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
        return dict(wheel_centers_w=centers,com_position_w=(d.body_com_pos_w.torch*weights[...,None]).sum(1),feet_b=feet_b,com_position=com_position,com_velocity=com_velocity,yaw=yaw,velocity=velocity,height=height,contact=contact,bad=bad,head=head,
            lateral_slip=lateral_slip, foot_point_velocity=point_velocity,wheel_axis_w=axis,
            pair_force=pair_force,heading_error=wrap(self.command.heading-yaw),
            anchor_error=horizontal(d.root_pos_w.torch[:,:2]-self.anchor,yaw),
            yaw_rate=d.root_ang_vel_w.torch[:,2])

    def frame(self,p):
        d=self.robot.data
        return torch.cat((d.projected_gravity_b.torch,d.root_ang_vel_b.torch*.25,
            d.joint_pos.torch[:,self.leg_ids]-self.stand,d.joint_vel.torch[:,self.ids]*.05,
            self.action,self.command.velocity,p['heading_error'].sin()[:,None],
            p['heading_error'].cos()[:,None],
            torch.sin(self.episode_length_buf*self.step_dt*(2*torch.pi/GAIT['period']))[:,None],
            torch.cos(self.episode_length_buf*self.step_dt*(2*torch.pi/GAIT['period']))[:,None]),-1).clamp(-10,10)

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
        self.terminal_critic=self.critic(self.frame(p),p).clone()
        timeout=self.episode_length_buf>=self.max_episode_length
        return self.failed,timeout

    def _get_rewards(self):
        p=self._physical;d=self.robot.data;c=self.command.velocity
        ex=p['velocity'][:,0]-c[:,0];ey=p['velocity'][:,1];ew=p['yaw_rate']-c[:,2]
        q=d.joint_pos.torch[:,self.leg_ids];dq=d.joint_vel.torch[:,self.ids]
        # Compute acceleration only once per control step, never in physical().
        measured=torch.where(self.balance_warm[:,None],(p['com_velocity']-self.balance_velocity)/self.step_dt,0.).clamp(-12.,12.)
        measured+=torch.stack((-p['yaw_rate']*p['com_velocity'][:,1],p['yaw_rate']*p['com_velocity'][:,0]),-1)
        self.balance_accel.lerp_(measured,.2);self.balance_velocity.copy_(p['com_velocity']);self.balance_warm[:]=True
        delta=p['wheel_centers_w']-d.root_pos_w.torch[:,None,:]
        feet_xy=horizontal(delta.reshape(-1,3),p['yaw'][:,None].expand(-1,4).reshape(-1)).reshape(self.num_envs,4,2)
        com_delta=p['com_position_w']-d.root_pos_w.torch
        com_xy=horizontal(com_delta,p['yaw'])
        floor=(p['wheel_centers_w'][:,:,2]-C['wheel_radius']).mean(-1)
        dyn=balance_metrics(p['velocity'],c,p['yaw_rate'],self.command.acceleration[:,:2],self.balance_accel,
            feet_xy,com_xy,p['com_position_w'][:,2]-floor,p['height'],p['contact'])
        low=dyn['quiet']
        errq=((q-self.U_stance)/torch.tensor([.15,.25,.25]*4,device=self.device)).square().mean(-1)
        clearance=(d.body_link_pos_w.torch[:,self.feet,2]-self.scene.env_origins[:,2:3]-C['wheel_radius']).clamp(min=0)
        # Tracking dominates; no prescribed wheel residual, stance or turn geometry.
        stopping=stopping_mask(self.keys,c)
        turning=(c[:,2].abs()>.12).float()
        f=p['feet_b'];ref=self.stance_feet_reference
        inward=((ref[:,1].abs()-f[:,:,1].abs()-.01).clamp(min=0)/.035).square().mean(-1).clamp(max=4)
        span=f[:,:2,0].mean(-1)-f[:,2:,0].mean(-1)
        opposition=torch.zeros_like(low)
        slip=p['lateral_slip']
        zmp=com_xy-(p['com_position_w'][:,2]-floor).clamp(.15,.55)[:,None]*self.balance_accel/9.81
        margin=support_margin(feet_xy,zmp,p['contact'])
        terms=dict(
            track_v=4.*torch.exp(-(ex.square()+ey.square())/.40),
            track_w=3.*torch.exp(-ew.square()/.50),
            upright=-1.*d.projected_gravity_b.torch[:,:2].square().sum(-1),
            vertical_velocity=-.20*d.root_lin_vel_w.torch[:,2].square(),
            roll_pitch_rate=-.05*d.root_ang_vel_b.torch[:,:2].square().sum(-1),
            height=-.25*dyn['height_cost'],
            contact=.30*p['contact'].float().mean(-1),
            action_rate=-.02*(self.action-self.previous_action).square().sum(-1),
            leg_speed=-.002*dq[:,:12].square().sum(-1),
            effort=-.01*(self.torque/C['effort']).square().mean(-1),
            support_polygon=-.25*((-.01-margin).clamp(min=0)/.05).square().clamp(max=4),
            lateral_slip=-.02*slip.clamp(max=4),
            idle_speed=-1.*stopping*p['velocity'].square().sum(-1),
            idle_yaw=-.50*stopping*p['yaw_rate'].square(),
            idle_wheel=-.005*stopping*dq[:,12:].square().mean(-1))
        # Soft geometric rewards; never overwrite q, root state, actions or targets.
        straight=torch.exp(-(c[:,2]/.35).square())
        mirrored_q=mirror_action(q)
        mirrored_dq=mirror_action(dq[:,:12])
        terms['straight_pose_symmetry']=-.35*straight*((q-mirrored_q)/.3).square().mean(-1).clamp(max=4)
        terms['straight_velocity_symmetry']=-.08*straight*((dq[:,:12]-mirrored_dq)/6.).square().mean(-1).clamp(max=4)
        # As acceleration and turning demand subside, continuously encourage neutral stance.
        demand=(self.command.acceleration[:,0]/3.).square()+(c[:,0]*c[:,2]/1.5).square()+(c[:,2]/.7).square()
        recover=torch.exp(-demand)
        rest=torch.exp(-(c[:,0]/.35).square()-(c[:,2]/.35).square())
        pose_error=((q-self.stand)/q.new_tensor([.2,.35,.35]*4)).square().mean(-1).clamp(max=4)
        terms['natural_recovery']=-(.15*recover+.65*rest)*pose_error
        terms['recovery_joint_calm']=-.008*rest*dq[:,:12].square().mean(-1)
        braking=(-p['yaw_rate']*self.command.acceleration[:,2]/3.).clamp(0,1)
        terms['yaw_braking']=-.55*(braking+rest)*(p['yaw_rate']/1.2).square().clamp(max=6)
        terms['brake_drift']=-.35*rest*(p['com_velocity'].norm(dim=-1)/.25).square().clamp(max=6)
        terms['dynamic_com']=.30*(1.-rest)*dyn['com_track']
        terms['dynamic_zmp']=-.35*dyn['zmp_cost']
        terms['brake_roll_pitch']=-.08*(rest+braking)*d.root_ang_vel_b.torch[:,:2].square().sum(-1)
        geometry=geometry_costs(f,ref,p['wheel_centers_w'][:,:,2]-self.scene.env_origins[:,2:3],
            p['contact'],d.projected_gravity_b.torch,self.command.acceleration[:,0],C['wheel_radius'])
        terms['front_rear_clearance']=-.55*geometry['front_rear_cost']
        terms['axle_clearance']=-.45*geometry['axle_cost']
        terms['wheel_airborne']=-.80*geometry['missing_contact']
        terms['wheel_lift_height']=-.45*geometry['lift_cost']
        terms['excess_pitch']=-.65*geometry['pitch_cost']
        reward=sum(terms.values())*self.step_dt-6*self.failed.float();self.returns+=reward
        self.extras['log']={**{'geometry/'+k:v.mean().detach() for k,v in geometry.items()},'recovery/pose_rms_rad':(q-self.stand).square().mean().sqrt().detach(),'recovery/continuous_gate':recover.mean().detach(),'mirror/straight_pose_error':((q-mirrored_q).square().mean(-1)*straight).mean().detach(),**{'reward/'+k:v.mean().detach() for k,v in terms.items()},
            'track/rmse_x':ex.square().mean().sqrt().detach(),'track/rmse_y':ey.square().mean().sqrt().detach(),
            'track/rmse_yaw':ew.square().mean().sqrt().detach(),'physical/failure_fraction':self.failed.float().mean().detach(),
            'wheel/support_polygon_margin_m':margin.mean().detach(),
            'wheel/inward_cost':inward.mean().detach(),'wheel/span_m':span.mean().detach(),'wheel/braking_speed':(p['velocity'][:,0].abs()*stopping).sum()/stopping.sum().clamp(min=1),
            'wheel/quiet_pose_gate':low.mean().detach(),'wheel/com_target_offset':dyn['com_offset'].mean().detach(),'wheel/zmp_margin_m':dyn['zmp_margin'].mean().detach(),
            'wheel/com_target_y':dyn['com_offset_y'].mean().detach(),
            'wheel/turn_opposition':opposition.mean().detach(),'wheel/excess_lateral_slip':slip.mean().detach(),
            'wheel/stopping_fraction':stopping.float().mean().detach(),
            'wheel/leg_target_rate_fraction':self.target_rate_fraction.mean().detach(),
            'wheel/four_contact':p['contact'].all(-1).float().mean().detach(),
            'wheel/stand_q_error':(q-self.U_stance).square().mean().sqrt().detach()}
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
        n=len(ids)
        q=self.robot.data.default_joint_pos.torch[ids].clone()
        q[:,self.leg_ids]=self.U_stance
        root=self.robot.data.default_root_state.torch[ids].clone()
        root[:,:3]=self.scene.env_origins[ids]+torch.tensor([0.,0.,.398],device=self.device)
        root[:,7:]=0
        yaw=torch.zeros(n,device=self.device)
        if self.cfg.training:
            q[:,self.leg_ids]+=(torch.rand(n,12,device=self.device)*2-1)*.015
            yaw=(torch.rand(n,device=self.device)*2-1)*3.14159265
            self.gain[ids]=.95+torch.rand(n,2,device=self.device)*.1
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
        self.balance_velocity[ids]=0;self.balance_accel[ids]=0;self.balance_warm[ids]=False
        self.clearance_history[ids]=0;self.lift_history[ids]=False
        self.rotation_pending[ids]=True
        self.command.reset(ids,yaw)
        self.until_sample[ids]=self.settling_steps+1
        self.anchor[ids]=root[:,:2]
        self.anchor_mode[ids]=True
        self.was_contact[ids]=True
        self.fill_history[ids]=True
        if hasattr(self,'schedule'):self.schedule.reset(ids)
