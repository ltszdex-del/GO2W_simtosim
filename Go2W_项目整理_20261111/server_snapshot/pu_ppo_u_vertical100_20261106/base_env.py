"""Endpoint-only PPO. Root writes occur exclusively in episode reset."""
import json
from pathlib import Path
import torch
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_apply
from scene_base import EnvCfg, SceneCfg, LEGS, WHEELS

ROOT = Path(__file__).resolve().parent
GOALS = json.loads((ROOT/'goals.json').read_text())
FRAME, HISTORY = 72, 3

@configclass
class PUSceneCfg(SceneCfg):
    front_right_rear = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/FR_foot',
        filter_prim_paths_expr=['{ENV_REGEX_NS}/Robot/RR_foot','{ENV_REGEX_NS}/Robot/RL_foot'],update_period=0.,history_length=4)
    front_left_rear = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/FL_foot',
        filter_prim_paths_expr=['{ENV_REGEX_NS}/Robot/RR_foot','{ENV_REGEX_NS}/Robot/RL_foot'],update_period=0.,history_length=4)

@configclass
class PUCfg(EnvCfg):
    episode_length_s = 12.
    action_space = 12
    observation_space = FRAME*HISTORY
    scene = PUSceneCfg(num_envs=1024, env_spacing=4., replicate_physics=True, filter_collisions=True)

class PUEnv(DirectRLEnv):
    def __init__(self,cfg,task='P',probe=False):
        self.task,self.probe=task,probe
        super().__init__(cfg)
        self._physics_handles_decimation=False
        self.legs,_=self.robot.find_joints(LEGS,preserve_order=True)
        self.wheels,_=self.robot.find_joints(WHEELS,preserve_order=True)
        self.ids=self.legs+self.wheels
        # Importer merges the head spheres into base. Compute exact sphere-floor
        # clearance from URDF offsets; base force near zero clearance is a proxy,
        # not an independently measured head contact force.
        self.head_offsets=torch.tensor([[.285,0,.01],[.293,0,-.06]],device=self.device)
        self.base,_=self.contacts.find_bodies('base')
        self.forbidden=[i for i,n in enumerate(self.contacts.body_names) if n not in GOALS['allowed_contact_bodies']]
        self.limits=self.robot.data.soft_joint_pos_limits.torch[:,self.legs]
        self.stand=torch.tensor(GOALS['stand']['q'],device=self.device)
        self.prone=torch.tensor(GOALS['prone']['q'],device=self.device)
        self.goal=self.prone if task=='P' else self.stand
        self.source=self.stand if task=='P' else self.prone
        self.center=(self.stand+self.prone)/2
        self.scale=torch.tensor([.5,.9,.85]*4,device=self.device)
        n=self.num_envs; z=lambda *s:torch.zeros(*s,device=self.device)
        self.target=z(n,12); self.action=z(n,12); self.old_action=z(n,12)
        self.torque=z(n,16); self.history=z(n,HISTORY,FRAME)
        self.hold=z(n); self.returns=z(n); self.anchor=z(n,2); self.old_phi=z(n)
        self.ever_success=torch.zeros(n,dtype=torch.bool,device=self.device)
        self.pending=torch.ones(n,dtype=torch.bool,device=self.device)
        self.level=0.; self.success_window=[]; self.pd_count=None
        self.weight=float(self.robot.data.body_mass.torch[0].sum())*9.81
        self.failures={}; self.terminal_obs=z(n,FRAME*HISTORY)

    def _setup_scene(self):
        self.robot=self.scene.articulations['robot']; self.contacts=self.scene.sensors['contacts']

    def _pre_physics_step(self,actions):
        self.extras.pop('episode',None)
        if self.pd_count is not None and self.pd_count!=4: raise RuntimeError('PD substep mismatch')
        self.pd_count=0
        self.old_action.copy_(self.action); self.action=actions.detach().clamp(-1,1)
        wanted=(self.center+self.scale*self.action).clamp(self.limits[...,0],self.limits[...,1])
        self.target+=(wanted-self.target).clamp(-.08,.08)  # 4 rad/s

    def _apply_action(self):
        d=self.robot.data
        self.torque=torch.cat((70*(self.target-d.joint_pos.torch[:,self.legs])-2.5*d.joint_vel.torch[:,self.legs],
            -.2*d.joint_vel.torch[:,self.wheels]),-1).clamp(-23.5,23.5)
        self.robot.set_joint_effort_target(self.torque,joint_ids=self.ids)
        self.pd_count+=1

    def frame(self):
        d=self.robot.data
        task=torch.zeros(self.num_envs,2,device=self.device); task[:,0 if self.task=='P' else 1]=1
        return torch.cat((d.projected_gravity_b.torch,d.root_ang_vel_b.torch*.25,
            d.joint_pos.torch[:,self.legs],d.joint_vel.torch[:,self.ids]*.05,self.action,
            self.goal-d.joint_pos.torch[:,self.legs],self.target,task),-1).clamp(-10,10)

    def _get_observations(self):
        frame=self.frame()
        self.history[:, :-1]=self.history[:,1:].clone(); self.history[:,-1]=frame
        self.history[self.pending]=frame[self.pending,None,:]
        self.pending[:]=False
        return {'policy':self.history.flatten(1).clone()}

    def metrics(self):
        d=self.robot.data; q=d.joint_pos.torch[:,self.legs]
        z=d.root_pos_w.torch[:,2]-self.scene.env_origins[:,2]
        goal=GOALS['prone' if self.task=='P' else 'stand']
        gravity=torch.tensor([__import__('math').sin(goal['pitch']),0,-__import__('math').cos(goal['pitch'])],device=self.device)
        qe=(q-self.goal).square().mean(-1)
        ge=(d.projected_gravity_b.torch-gravity).square().sum(-1)
        ze=(z-goal['height']).abs()
        force=self.contacts.data.net_forces_w_history.torch.norm(dim=-1).amax(1)
        centers=quat_apply(d.root_quat_w.torch[:,None,:].expand(-1,2,-1),self.head_offsets[None].expand(self.num_envs,-1,-1))+d.root_pos_w.torch[:,None,:]
        clearance=(centers[:,:,2]-.06-self.scene.env_origins[:,None,2]).amin(-1)
        head=torch.where(clearance<.003,force[:,self.base].amax(-1),torch.zeros_like(clearance))
        wheels=[]
        for name in ('front_right_rear','front_left_rear'):
            mat=self.scene.sensors[name].data.normal_force_matrix_w_history
            if mat is None: raise RuntimeError('Wheel collision filter missing')
            wheels.append(mat.torch.norm(dim=-1).flatten(1).amax(-1))
        wheel=torch.stack(wheels,-1).amax(-1)
        v=d.root_lin_vel_b.torch.norm(dim=-1); w=d.root_ang_vel_b.torch.norm(dim=-1)
        drift=(d.root_pos_w.torch[:,:2]-self.anchor).norm(dim=-1)
        phi=-(qe/.3**2+ge/.3**2+ze.square()/.15**2).clamp(max=100)
        forbidden=force[:,self.forbidden].amax(-1) if self.forbidden else torch.zeros_like(z)
        base=force[:,self.base].amax(-1)
        return dict(qe=qe,ge=ge,ze=ze,z=z,head=head,clearance=clearance,wheel=wheel,v=v,w=w,drift=drift,
            phi=phi,forbidden=forbidden,base=base,peak=force.amax(-1),q=q)

    def _get_dones(self):
        p=self.metrics(); self.p=p
        t=GOALS['tolerances']; d=self.robot.data
        stable=(p['qe']<t['joint_rms_rad']**2)&(p['ge']<t['gravity_error']**2)&(p['ze']<t['height_m'])&(p['v']<t['linear_speed_m_s'])&(p['w']<t['angular_speed_rad_s'])&(p['clearance']>t['head_clearance_m'])&(p['head']<1)&(p['wheel']<.01)&(p['forbidden']<.15*self.weight)
        if self.task=='U': stable &= p['base']<.05*self.weight
        self.hold=torch.where(stable,self.hold+self.step_dt,torch.zeros_like(self.hold))
        self.success=self.hold>=t['hold_s']
        self.new_success=self.success&~self.ever_success
        self.ever_success |= self.success
        reasons=dict(head=(p['head']>1)|(p['clearance']<-.005),wheel=p['wheel']>.01,
            flip=d.projected_gravity_b.torch[:,2]>-.5,
            limits=((p['q']<self.limits[...,0]-.03)|(p['q']>self.limits[...,1]+.03)).any(-1),
            impact=p['peak']>4*self.weight,forbidden=p['forbidden']>.5*self.weight,
            drift=p['drift']>.5)
        self.reason_masks=reasons
        self.failed=torch.stack(list(reasons.values())).any(0)
        timeout=self.episode_length_buf>=self.max_episode_length-1
        if self.probe: return torch.zeros_like(timeout),torch.zeros_like(timeout)
        return (self.success|self.failed if self.task=='P' else self.failed),timeout

    def _get_rewards(self):
        p=self.p
        progress=.995*p['phi']-self.old_phi; self.old_phi=p['phi'].clone()
        reward=(.3*progress+(.8*torch.exp(-p['qe']/.09)+.5*torch.exp(-p['ze'].square()/.01)+.4*torch.exp(-p['ge']/.09)
            +2*(self.hold>0).float()-.1-.3*p['drift'].square()-.0002*self.torque.square().sum(-1)
            -.015*(self.action-self.old_action).square().mean(-1)-.005*self.robot.data.joint_vel.torch[:,self.legs].square().mean(-1)
            -.2*(p['peak']/self.weight-1).clamp(min=0))*self.step_dt+5*self.new_success.float()-3*self.failed.float())
        self.returns+=reward
        # Terminal history before automatic episode reset, for timeout bootstrapping.
        self.terminal_obs=torch.cat((self.history[:,1:],self.frame()[:,None,:]),1).flatten(1).clone()
        return reward

    def _reset_idx(self,env_ids):
        super()._reset_idx(env_ids)
        if env_ids is None: env_ids=self.robot._ALL_INDICES
        if not hasattr(self,'target'): return
        if hasattr(self,'success') and len(env_ids):
            mask=self.failed[env_ids]; success=self.success[env_ids]&~mask
            self.extras['episode']=dict(count=len(env_ids),success=float(success.float().mean()),
                reward=float(self.returns[env_ids].mean()),level=self.level,
                timeout=float((~(mask|success)).float().mean()),
                **{k:float(v[env_ids].float().mean()) for k,v in self.reason_masks.items()})
            self.success_window.extend(success.float().tolist()); self.success_window=self.success_window[-1024:]
        n=len(env_ids); dev=self.device
        # Feasible reset distribution: endpoint mixture, never an in-episode reference.
        full=torch.rand(n,device=dev)<(.2+(.6 if self.task=='U' else .8)*self.level)
        q=torch.where(full[:,None],self.source[None],self.goal[None]).clone()
        noise=.005+.025*self.level
        if not self.probe: q+=torch.randn_like(q)*noise
        q=q.clamp(self.limits[env_ids,:,0],self.limits[env_ids,:,1])
        src=GOALS['stand' if self.task=='P' else 'prone']; dst=GOALS['prone' if self.task=='P' else 'stand']
        height=torch.where(full,torch.full((n,),src['height'],device=dev),torch.full((n,),dst['height'],device=dev))
        pitch=torch.where(full,torch.full((n,),src['pitch'],device=dev),torch.full((n,),dst['pitch'],device=dev))
        if self.probe: q[:]=self.goal; height[:]=dst['height']+.025; pitch[:]=dst['pitch']
        pose=self.robot.data.default_root_pose.torch[env_ids].clone()
        pose[:,:3]=self.scene.env_origins[env_ids]; pose[:,2]+=height+.005
        zeros=torch.zeros(n,device=dev); pose[:,3:]=quat_from_euler_xyz(zeros,pitch,zeros)
        self.robot.write_root_pose_to_sim_index(root_pose=pose,env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n,6,device=dev),env_ids=env_ids)
        joints=self.robot.data.default_joint_pos.torch[env_ids].clone(); joints[:,self.legs]=q
        self.robot.write_joint_position_to_sim_index(position=joints,env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joints),env_ids=env_ids)
        self.robot.reset(env_ids)
        self.target[env_ids]=q; self.action[env_ids]=((q-self.center)/self.scale).clamp(-1,1)
        self.old_action[env_ids]=self.action[env_ids]; self.hold[env_ids]=0; self.returns[env_ids]=0
        self.ever_success[env_ids]=False
        self.pending[env_ids]=True; self.anchor[env_ids]=pose[:,:2]
        # Potential uses the same state and normalization as the next-step metric.
        grav=torch.stack((pitch.sin(),zeros,-pitch.cos()),-1)
        g=torch.tensor([__import__('math').sin(dst['pitch']),0,-__import__('math').cos(dst['pitch'])],device=dev)
        self.old_phi[env_ids]=-((q-self.goal).square().mean(-1)/.09+(grav-g).square().sum(-1)/.09+(height+.005-dst['height']).square()/.0225).clamp(max=100)
