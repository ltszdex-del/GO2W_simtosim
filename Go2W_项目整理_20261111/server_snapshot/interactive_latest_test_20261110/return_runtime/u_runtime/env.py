"""U continuation: mirror motor targets, penalize asymmetric posture and wheel lean."""
import math
import json
import torch
from isaaclab.utils.math import quat_apply_inverse
from .parent_env import PUEnv as Base, PUCfg, ROOT, GOALS
from .settings import CONTROL
SPEED=json.loads((ROOT/'speed_config.json').read_text())

class PUEnv(Base):
    def __init__(self,cfg,task='U',probe=False):
        if task!='U':raise ValueError('U continuation only')
        super().__init__(cfg,task,probe)
        self.lift_time=torch.full((self.num_envs,),-1.,device=self.device)
        self.standing_time=self.lift_time.clone();self.verified_time=self.lift_time.clone()
        self.just_stood=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)

    def mirror(self,x):
        shaped=x.reshape(-1,4,3).clone()
        sign=torch.tensor([-1.,1.,1.],device=self.device)
        for right,left in ((0,1),(2,3)):
            mean=(shaped[:,right]+shaped[:,left]*sign)*.5
            shaped[:,right]=mean;shaped[:,left]=mean*sign
        return shaped.flatten(1)

    def stage(self):
        z=self.robot.data.root_pos_w.torch[:,2]-self.scene.env_origins[:,2]
        return ((z-.12)/.12).clamp(0,1)

    def _pre_physics_step(self,actions):
        old_target=self.target.clone();old_velocity=self.target_velocity.clone()
        actions=self.mirror(actions.detach())
        stage=self.stage()
        actions[:,::3]*=(1-stage[:,None])
        super()._pre_physics_step(actions)
        # As the body rises, hip targets approach zero. Limit acceleration and
        # speed rather than clipping positions into a new instantaneous pose.
        rate=CONTROL['leg_rate_rad_s'];dv=CONTROL['leg_accel_rad_s2']*self.step_dt
        wanted=(-CONTROL['hip_goal_gain']*old_target[:,::3]+CONTROL['residual_velocity_rad_s']*actions[:,::3]).clamp(-rate,rate)
        z=self.robot.data.root_pos_w.torch[:,2]-self.scene.env_origins[:,2]
        upright=((z-.25)/.10).clamp(0,1)
        axis=self.camber()[1]
        wanted=(wanted-3.6*upright[:,None]*axis[:,:,2]).clamp(-rate,rate)
        velocity=old_velocity[:,::3]+(wanted-old_velocity[:,::3]).clamp(-dv,dv)
        self.target[:,::3]=old_target[:,::3]+velocity*self.step_dt
        self.target_velocity[:,::3]=velocity
        self.target=self.mirror(self.target).clamp(self.actual_limits[...,0],self.actual_limits[...,1])
        self.target_velocity=self.mirror(self.target_velocity)

    def _apply_action(self):
        super()._apply_action()
        d=self.robot.data
        z=d.root_pos_w.torch[:,2]-self.scene.env_origins[:,2]
        blend=((z-.25)/.10).clamp(0,1)
        # Increase standing hip stiffness/damping through real joint torques.
        extra=140*(self.target[:,::3]-d.joint_pos.torch[:,self.legs[::3]])-3.5*d.joint_vel.torch[:,self.legs[::3]]
        self.torque[:,:12:3]=(self.torque[:,:12:3]+blend[:,None]*extra*self.gain[:,None]).clamp(-23.5,23.5)
        self.robot.set_joint_effort_target(self.torque,joint_ids=self.ids)

    def metrics(self):
        m=super().metrics();d=self.robot.data
        q=m['q'].reshape(-1,4,3);sign=torch.tensor([-1.,1.,1.],device=self.device)
        err=torch.cat((q[:,0]-q[:,1]*sign,q[:,2]-q[:,3]*sign),-1)
        relative=d.body_link_pos_w.torch[:,self.feet]-d.root_pos_w.torch[:,None,:]
        feet=quat_apply_inverse(d.root_quat_w.torch[:,None,:].expand(-1,4,-1),relative)
        foot_sign=torch.tensor([1.,-1.,1.],device=self.device)
        foot_err=torch.stack((feet[:,0]-feet[:,1]*foot_sign,feet[:,2]-feet[:,3]*foot_sign),1)
        m.update(symmetry_rms=err.square().mean(-1).sqrt(),symmetry_max=err.abs().amax(-1),
                 foot_mirror_rms=foot_err.square().mean((1,2)).sqrt(),
                 foot_mirror_max=foot_err.norm(dim=-1).amax(-1))
        return m

    def _get_dones(self):
        previously=self.ever_success.clone()
        done,timeout=super()._get_dones()
        m=self.p
        extra=(m['symmetry_max']<=math.radians(5))&(m['foot_mirror_max']<=.02)
        extra&=m['front_camber'].amax(-1)<=math.radians(5)
        self.hold=torch.where(extra,self.hold,torch.zeros_like(self.hold))
        elapsed=self.episode_length_buf*self.step_dt
        lifted=(m['belly']>.025)&(m['base']<.05*self.weight)&~self.failed
        self.lift_time=torch.where((self.lift_time<0)&lifted,elapsed,self.lift_time)
        # Parent hold already enforces height/pose, low velocity, no belly support,
        # head clearance, +/-5deg torso plus this class's symmetry/wheel checks.
        good=(self.hold>0)&~self.failed
        self.just_stood=(self.standing_time<0)&good
        self.standing_time=torch.where(self.just_stood,elapsed,self.standing_time)
        verified=self.hold>=SPEED['standing_hold_s']
        self.verified_time=torch.where((self.verified_time<0)&verified,elapsed,self.verified_time)
        on_time=(self.standing_time>=0)&(self.standing_time<=SPEED['deadline_s']+1e-6)
        self.success=verified&on_time;self.new_success=self.success&~previously
        self.ever_success=previously|self.success
        return done,timeout

    def _get_rewards(self):
        reward=super()._get_rewards();m=self.p
        # Initial prone wheels lean by design; upright weight increases smoothly
        # during ascent. No motor interlock that can trap the robot on the floor.
        stage=((m['z']-.12)/.12).clamp(0,1)
        cost=(2*(m['symmetry_rms']/.1).square()
              +.5*(m['foot_mirror_rms']/.02).square()
              +2*stage*(m['front_camber']/math.radians(10)).square().mean(-1)
              +6*stage*(m['front_camber']-math.radians(10)).clamp(min=0).square().sum(-1))
        upright=((m['z']-.25)/.10).clamp(0,1)
        camber=m['front_camber']
        cost+=upright*(2*(camber/math.radians(5)).square().mean(-1)
            +8*((camber-math.radians(3)).clamp(min=0)/math.radians(5)).square().mean(-1))
        delta=-cost*self.step_dt
        elapsed=self.episode_length_buf*self.step_dt
        progress=((m['z']-GOALS['prone']['height'])/(GOALS['stand']['height']-GOALS['prone']['height'])).clamp(0,1)
        # Replace the inherited 0.08m/s ascent cost with a height-dependent
        # allowance: more speed low down, returning to 0.08m/s near standing.
        allowance=.08+.22*(1-progress).sqrt()
        allowance=torch.where(m['vz']>=0,allowance,torch.full_like(allowance,.08))
        inherited=25*(m['vz'].abs()-.08).clamp(min=0).square()
        replacement=25*(m['vz'].abs()-allowance).clamp(min=0).square()
        t=(elapsed/SPEED['target_rise_s']).clamp(0,1)
        blend=t**3*(10-15*t+6*t*t)
        height_ref=GOALS['prone']['height']+blend*(GOALS['stand']['height']-GOALS['prone']['height'])
        tracking=2*torch.exp(-((m['z']-height_ref)/.045).square())
        unfinished=self.standing_time<0
        time_cost=unfinished.float()*(.5+1.5*(elapsed-SPEED['deadline_s']).clamp(0,1))
        stable=(self.hold>0)&~self.failed
        delta+=(inherited-replacement+tracking+4*stable.float()-time_cost)*self.step_dt
        on_time=self.just_stood&(elapsed<=SPEED['deadline_s']+1e-6)
        delta+=8*on_time.float()*(1-.5*(elapsed/SPEED['deadline_s']).clamp(0,1))
        self.returns+=delta
        return reward+delta

    def _reset_idx(self,ids):
        timing=None
        if hasattr(self,'standing_time') and hasattr(self,'failed'):
            sample=self.robot._ALL_INDICES if ids is None else ids
            if len(sample):
                lift=self.lift_time[sample];stand=self.standing_time[sample];verified=self.verified_time[sample]
                timing=dict(lifted_fraction=float((lift>=0).float().mean()),stood_fraction=float((stand>=0).float().mean()),
                    verified_fraction=float((verified>=0).float().mean()),
                    on_time_fraction=float(((stand>=0)&(stand<=SPEED['deadline_s']+1e-6)&~self.failed[sample]).float().mean()),
                    lift_time_sum_s_per_episode=float(torch.where(lift>=0,lift,0.).mean()),
                    standing_time_sum_s_per_episode=float(torch.where(stand>=0,stand,0.).mean()),
                    verified_time_sum_s_per_episode=float(torch.where(verified>=0,verified,0.).mean()))
        super()._reset_idx(ids)
        if timing is not None and 'episode' in self.extras:self.extras['episode'].update(timing)
        if hasattr(self,'states'):
            if ids is None:ids=self.robot._ALL_INDICES
            self.target[ids]=self.mirror(self.target[ids])
        if hasattr(self,'standing_time'):
            self.lift_time[ids]=-1;self.standing_time[ids]=-1;self.verified_time[ids]=-1;self.just_stood[ids]=False
