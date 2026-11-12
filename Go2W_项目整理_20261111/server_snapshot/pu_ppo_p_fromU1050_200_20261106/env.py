"""Old lie_down faster_v2 reward design ported to the current explicit-PD scene."""
import math
import json
import torch
from isaaclab.utils import configclass
from parent_env import PUEnv as Base, PUCfg as BaseCfg, ROOT, GOALS
from settings import CONTROL
TIMING=json.loads((ROOT/'reward_port.json').read_text())
START=TIMING['phases']['stand_s']
DESCENT=TIMING['phases']['descent_s']
END=START+DESCENT
DEADLINE=TIMING['deadline_s']
GAP=TIMING['belly_gap_tolerance_m']

@configclass
class PUCfg(BaseCfg):
    observation_space=279
    episode_length_s=20.

class PUEnv(Base):
    def __init__(self,cfg,task='P',probe=False):
        if task!='P':raise ValueError('P speed refinement only')
        super().__init__(cfg,task,probe)
        n=self.num_envs;z=lambda *shape:torch.zeros(*shape,device=self.device)
        self.history=z(n,3,93);self.terminal_obs=z(n,279)
        self.episode_body_peak=z(n);self.touchdown_speed=z(n);self.previous_vz=z(n)
        self.previous_contact=torch.zeros(n,dtype=torch.bool,device=self.device)
        self.touch_time=torch.full((n,),-1.,device=self.device)
        self.poweroff_time=self.touch_time.clone();self.completion_time=self.touch_time.clone()
        self.just_completed=torch.zeros(n,dtype=torch.bool,device=self.device)

    def phase(self):
        elapsed=self.episode_length_buf*self.step_dt
        x=((elapsed-START)/DESCENT).clamp(0,1)
        blend=x**3*(10-15*x+6*x*x)
        return elapsed,blend

    def reference(self):
        _,blend=self.phase()
        return self.stand+blend[:,None]*(self.prone-self.stand)

    def frame(self):
        d=self.robot.data;elapsed,blend=self.phase()
        task=torch.zeros(self.num_envs,2,device=self.device);task[:,0]=1
        load=self.contacts.data.net_forces_w_history.torch.norm(dim=-1).amax(1)[:,self.base].amax(-1)/self.weight
        height=d.root_pos_w.torch[:,2]-self.scene.env_origins[:,2]
        return torch.cat((d.projected_gravity_b.torch,d.root_ang_vel_b.torch*.25,
            d.joint_pos.torch[:,self.legs],d.joint_vel.torch[:,self.ids]*.05,self.action,
            self.reference()-d.joint_pos.torch[:,self.legs],self.target,task,
            blend[:,None],(elapsed>=DEADLINE).float()[:,None],height[:,None],d.root_lin_vel_b.torch,
            d.root_lin_vel_w.torch[:,2,None],load[:,None],self.gain[:,None],self.target_velocity*.25),-1).clamp(-10,10)

    def _pre_physics_step(self,actions):
        old=self.target.clone();old_velocity=self.target_velocity.clone()
        old_contact=self.contact_time.clone();old_released=self.released.clone();old_release_time=self.release_time.clone()
        super()._pre_physics_step(actions)
        elapsed,_=self.phase()
        desired=self.reference()+.10*self.action*(elapsed>=START)[:,None]
        desired=desired.clamp(self.limits[...,0],self.limits[...,1])
        rate=CONTROL['leg_rate_rad_s'];dv=CONTROL['leg_accel_rad_s2']*self.step_dt
        velocity=((desired-old)/self.step_dt).clamp(-rate,rate)
        self.target_velocity=old_velocity+(velocity-old_velocity).clamp(-dv,dv)
        self.target=(old+self.target_velocity*self.step_dt).clamp(self.actual_limits[...,0],self.actual_limits[...,1])
        m=self.metrics()
        touching=(m['belly'].abs()<GAP)&(m['body_mean_bw']>.08)
        # Parent uses a higher threshold. Replace the counter with the old task's
        # measured support threshold while keeping a contact confirmation window.
        self.contact_time=torch.where(touching,old_contact+self.step_dt,torch.zeros_like(old_contact))
        self.released=old_released|(self.contact_time>=.12)
        self.release_time=old_release_time+self.released*self.step_dt
        self.gain=(1-self.release_time/CONTROL['release_s']).clamp(0,1)

    def metrics(self):
        m=super().metrics()
        force=self.contacts.data.net_forces_w_history.torch.norm(dim=-1)
        body=force[:,:,self.base].sum(-1)/self.weight
        m.update(body_peak_bw=body.amax(-1),body_mean_bw=body.mean(-1),
                 foot_peak_bw=force[:,:,self.feet].amax(1)/self.weight,
                 body_history_bw=body)
        return m

    def _get_dones(self):
        done,timeout=super()._get_dones();m=self.p;elapsed,_=self.phase()
        self.episode_body_peak=torch.maximum(self.episode_body_peak,m['body_peak_bw'])
        contact=(m['belly'].abs()<GAP)&(m['body_mean_bw']>.02)
        touchdown=contact&~self.previous_contact
        self.touchdown_speed=torch.maximum(self.touchdown_speed,torch.where(touchdown,(-self.previous_vz).clamp(min=0),0.))
        self.previous_contact=contact;self.previous_vz=m['vz'].clone()
        self.touch_time=torch.where((self.touch_time<0)&contact,elapsed,self.touch_time)
        self.poweroff_time=torch.where((self.poweroff_time<0)&(self.gain==0),elapsed,self.poweroff_time)
        good=(elapsed>=END)&(m['qe']<.25**2)&(m['body_mean_bw']>.08)&(m['belly'].abs()<GAP)
        good&=(self.gain==0)&(m['v']<.08)&(m['w']<.25)&(m['leg_speed']<.35)
        good&=(m['roll'].abs()<math.radians(5))&(m['pitch'].abs()<math.radians(5))
        good&=(self.episode_body_peak<2.)&(self.touchdown_speed<.2)&~self.failed
        self.just_completed=(self.completion_time<0)&good
        self.completion_time=torch.where(self.just_completed,elapsed,self.completion_time)
        # Undo parent's counter increment; count exactly once per control step.
        self.hold=torch.where(good,self._previous_hold+self.step_dt,torch.zeros_like(self.hold))
        on_time=(self.completion_time>=0)&(self.completion_time<=DEADLINE+1e-6)
        self.success=(self.hold>=2.)&on_time;self.new_success=self.success&~self._previous_ever
        self.ever_success=self._previous_ever|self.success
        return done,timeout

    def _get_rewards(self):
        m=self.p;d=self.robot.data;elapsed,blend=self.phase()
        reference_error=(m['q']-self.reference()).square().mean(-1)
        height_ref=GOALS['stand']['height']+blend*(GOALS['prone']['height']-GOALS['stand']['height'])
        tracking=torch.exp(-reference_error/.025)
        height=torch.exp(-((m['z']-height_ref)/.04).square())
        supported=(m['body_mean_bw']>.08)&(m['belly'].abs()<GAP)
        # Endpoint reward must favor genuine unloaded belly support over hovering.
        released=(self.gain==0)&supported
        final=(elapsed>=END)*(.2+.8*released.float())*torch.exp(-m['qe']/.15**2-(m['v']/.1).square()-(m['w']/.3).square())
        approach=(m['z']<.18)*((-m['vz']-.1).clamp(min=0)/.1).square()
        peak=(m['body_peak_bw']-1.3).clamp(min=0).square()+(m['foot_peak_bw']-1.).clamp(min=0).square().sum(-1)
        impulse=((m['body_history_bw']-1.3).clamp(min=0).sum(-1)*.005/.02).square()
        early=(elapsed<START+.5*DESCENT)&contact_mask(m)
        head=(-m['clearance']/.003).clamp(min=0).square()
        feet_velocity=d.body_lin_vel_w.torch[:,self.feet,:2].square().sum(-1)
        slip=(feet_velocity*(m['foot_peak_bw']*self.weight>2)).sum(-1)
        q=m['q'];limits=((self.limits[...,0]-q).clamp(min=0)+(q-self.limits[...,1]).clamp(min=0)).sum(-1)
        density=(2*tracking+2*height+4*final+3*self.success.float()-2*m['ge']
          -2*d.root_ang_vel_w.torch[:,2].square()-approach-2*peak-impulse-2*early.float()-2*head
          -.3*slip-m['drift'].square()-.01*d.joint_vel.torch[:,self.legs].square().mean(-1)
          -.01*(self.action-self.old_action).square().sum(-1)-2*limits
          -.002*(elapsed>=END)*self.torque[:,:12].square().mean(-1))
        unfinished=self.completion_time<0
        density+=6*released.float()*torch.exp(-(m['v']/.08).square()-(m['w']/.25).square())
        density+=2*blend*torch.exp(-(m['belly'].clamp(min=0)/.01).square())
        # Bound the delay cost so an intentional fall cannot cheaply avoid an
        # arbitrarily large future penalty for an otherwise recoverable miss.
        density-=unfinished.float()*(1+3*(elapsed-DEADLINE).clamp(0,1))
        on_time=self.just_completed&(elapsed<=DEADLINE+1e-6)
        bonus=8*on_time.float()*(1-.5*(elapsed/DEADLINE).clamp(0,1))
        reward=density*self.step_dt+bonus-10*self.failed.float()
        self.returns+=reward
        self.terminal_obs=torch.cat((self.history[:,1:],self.frame()[:,None,:]),1).flatten(1).clone()
        return reward

    def _reset_idx(self,ids):
        timing=None
        if hasattr(self,'completion_time') and hasattr(self,'failed'):
            sample=self.robot._ALL_INDICES if ids is None else ids
            if len(sample):
                touch=self.touch_time[sample];off=self.poweroff_time[sample];complete=self.completion_time[sample]
                timing=dict(touched_fraction=float((touch>=0).float().mean()),poweroff_fraction=float((off>=0).float().mean()),
                    completed_fraction=float((complete>=0).float().mean()),
                    on_time_fraction=float(((complete>=0)&(complete<=DEADLINE+1e-6)&~self.failed[sample]).float().mean()),
                    touch_time_sum_s_per_episode=float(torch.where(touch>=0,touch,0.).mean()),
                    poweroff_time_sum_s_per_episode=float(torch.where(off>=0,off,0.).mean()),
                    completion_time_sum_s_per_episode=float(torch.where(complete>=0,complete,0.).mean()))
        super()._reset_idx(ids)
        if timing is not None and 'episode' in self.extras:self.extras['episode'].update(timing)
        if not hasattr(self,'episode_body_peak'):return
        if ids is None:ids=self.robot._ALL_INDICES
        self.episode_body_peak[ids]=0;self.touchdown_speed[ids]=0;self.previous_vz[ids]=0;self.previous_contact[ids]=False
        self.touch_time[ids]=-1;self.poweroff_time[ids]=-1;self.completion_time[ids]=-1;self.just_completed[ids]=False

    def _apply_action(self):
        super()._apply_action()

    def _get_observations(self):
        return super()._get_observations()

    def step(self,actions):
        self._previous_hold=self.hold.clone();self._previous_ever=self.ever_success.clone()
        return super().step(actions)

def contact_mask(m):
    return (m['belly'].abs()<GAP)&(m['body_mean_bw']>.02)
