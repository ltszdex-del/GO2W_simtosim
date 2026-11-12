"""3799 handstand adapter. Reuses its immutable observation and 200 Hz controller code."""
import hashlib
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace as NS, MethodType
import torch
from wasd_adapter import asset_hash


class HandstandAdapter:
    def __init__(self,spec,device):
        self.spec,self.device,self.path=spec,device,spec['checkpoint']
        self.ck=torch.load(self.path,map_location='cpu',weights_only=False)
        assert self.ck['kind']=='teacher'
        self.saved=self.ck['interface']
        code=Path(spec['code_root'])
        for name,digest in self.saved['sources'].items():
            assert hashlib.sha256((code/'go2w_mimic'/name).read_bytes()).hexdigest()==digest,name
        assert hashlib.sha256(Path(spec['motion']).read_bytes()).hexdigest()==self.saved['motion_sha256']
        assets=Path(spec['asset_root'])
        if hashlib.sha256((assets/'urdf/go2w_description.urdf').read_bytes()).hexdigest()!=self.saved['urdf_sha256']:
            assert spec.get('collision_geometry_override')
            from asset_compatibility import collision_only_migration
            self.asset_migration=collision_only_migration(Path(spec['original_asset_root'])/'urdf/go2w_description.urdf',assets/'urdf/go2w_description.urdf')
            print('[ASSET_MIGRATION] H',self.asset_migration,flush=True)
        self.manifest=dict(observation_order=self.saved['proprio_order']+self.saved['goal_order'],
            normalization=self.saved['normalization'],scales=self.saved['control'],history=self.saved['history'],
            phase='original monotonic reference; final pose held',previous_action='last bounded 16D action',
            target='stand to handstand and hold',task_id=self.saved['task'],
            joint_order=self.saved['leg_order']+self.saved['wheel_order'],output={'leg_pos_rad':12,'wheel_vel_rad_s':4},
            pd={'kp':70.,'kd':2.5,'wheel_kd':.2},
            limits={'effort_limit_nm':23.5,'leg_target_rate_rad_s':8.,'wheel_target_rate_rad_s2':40.,'wheel_speed_limit_rad_s':30.1},
            timing={'physics_dt':.005,'control_dt':.02,'substeps':4},asset_sha256=asset_hash(assets),wheel_lock_damping=0.)
        assert self.saved['control']['leg_rate']==8. and self.saved['control']['wheel_kd']==.2
        sys.path.insert(0,str(code))
        # No other go2w_mimic package may have been loaded before the verified snapshot.
        existing=sys.modules.get('go2w_mimic')
        if existing:assert Path(existing.__file__).resolve().is_relative_to(code.resolve())
        self.module=importlib.import_module('go2w_mimic.env')
        self.settings=importlib.import_module('go2w_mimic.settings')
        self.networks=importlib.import_module('go2w_mimic.networks')
        self.failure=None

    def load(self):
        self.model=self.networks.Teacher().to(self.device)
        self.model.load_state_dict(self.ck['model'],strict=True);self.model.eval()
        self.motion=self.module.Motion(self.spec['motion'],self.device)

    def preparation_targets(self):
        return dict(leg_pos_rad=self.motion.sample(torch.zeros(1,device=self.device))['joints'],
                    wheel_vel_rad_s=torch.zeros(1,4,device=self.device))

    def ready_to_start(self,state):
        pose=self.preparation_targets()['leg_pos_rad']
        target_error=(state['leg_target_rad']-pose).abs().max()
        error=(state['joint_pos_rad'][:,:12]-pose).abs().max()
        speed=state['joint_vel_rad_s'][:,:12].abs().max()
        # Loaded PD equilibrium has a nonzero q error: wait for the target to
        # arrive and joints to settle, not for impossible zero gravity deflection.
        if getattr(self.env.handstand_context,'pu_task',None)=='U':
            return bool(error<.3 and self.env.handstand_context.target_velocity.abs().max()<.3)
        return bool(target_error<.01 and error<.25 and speed<.3)

    def bind_env(self,env):self.env=env

    def reset(self,state):
        env=self.env;dev=self.device;z=lambda *shape:torch.zeros(*shape,device=dev)
        ctx=NS(robot=env.robot,contacts=env.contacts,device=dev,num_envs=1,step_dt=.02,extras={},
            cfg=NS(randomize=False),leg_ids=env.legs,wheel_ids=env.wheels,ids=env.ids)
        ctx.feet,_=env.robot.find_bodies(self.settings.FOOT_NAMES,preserve_order=True)
        ctx.contact_ids,_=env.contacts.find_bodies(self.settings.FOOT_NAMES,preserve_order=True)
        ctx.other_contacts=[i for i in range(len(env.contacts.body_names)) if i not in ctx.contact_ids]
        origins=env.scene.env_origins.clone();origins[:,:2]=env.robot.data.root_pos_w.torch[:,:2]
        ctx.scene=NS(env_origins=origins,sensors=env.scene.sensors)
        ctx.motion=self.motion
        ctx.limits=env.robot.data.soft_joint_pos_limits.torch[:,env.legs]
        for name in ('action','old_action','older_action','torque'):setattr(ctx,name,z(1,16))
        ctx.target=env.target.clone()
        if 'initial_standing_sha256' in self.saved:ctx.target=self.preparation_targets()['leg_pos_rad'].clone()
        ctx.target_velocity=z(1,12);ctx.wheel_target=z(1,4);ctx.delayed=z(1,3,16)
        ctx.delay=torch.zeros(1,dtype=torch.long,device=dev)
        ctx.gain=torch.ones(1,2,device=dev);ctx.speed=torch.ones(1,device=dev)
        ctx.mass_scale=torch.ones(1,device=dev);ctx.mu=torch.full((1,),.8,device=dev)
        ctx.masses=env.robot.data.body_mass.torch.clone()
        ctx.front_anchor=z(1,2,2);ctx.anchor_pending=torch.ones(1,dtype=torch.bool,device=dev)
        ctx.history=z(1,self.settings.HISTORY,self.settings.PROPRIO_DIM)
        ctx.fill_history=torch.ones(1,dtype=torch.bool,device=dev)
        ctx.episode_length_buf=torch.zeros(1,dtype=torch.long,device=dev)
        ctx.bad_time=z(1);ctx.low_time=z(1);ctx.max_episode_length=10**9;ctx._pd_substeps=None
        for name in ('ref','physical','wheel_guard','heading','stable_heading','anchor_error','blocks',
                     '_get_observations','_pre_physics_step','_apply_action','_get_dones'):
            setattr(ctx,name,MethodType(getattr(self.module.MimicEnv,name),ctx))
        self.ctx=ctx;self.failure=None
        env.set_handstand_context(ctx)

    def observe(self,state):return self.ctx._get_observations()['policy']

    @torch.inference_mode()
    def infer(self,observation):return self.model(observation)

    def apply(self,action,dt):
        assert abs(dt-.02)<1e-9
        self.ctx._pre_physics_step(action)
        return dict(leg_pos_rad=self.ctx.target,wheel_vel_rad_s=self.ctx.wheel_target)

    def after_physics(self,state):
        self.ctx.episode_length_buf+=1
        failed,_=self.ctx._get_dones()
        if bool(failed[0]):
            p=self.ctx.physical();d=self.ctx.robot.data;g=d.projected_gravity_b.torch[0]
            report=dict(seconds=float(self.ctx.episode_length_buf[0])*.02,height=float(d.root_pos_w.torch[0,2]),
                gravity=g.tolist(),pitch=float(torch.atan2(g[0],-g[2])),
                collision_force=float(p['wheel_collision_force'][0]),front_axis_angle=float(p['front_axis_angle'][0]),
                bad_contact=float(p['bad'][0]),bad_time=float(self.ctx.bad_time[0]),low_time=float(self.ctx.low_time[0]))
            self.failure='Original handstand failure guard: '+str(report)
            print('[HANDSTAND_FAILURE]',report,flush=True)
            return True
        return False

    def is_complete(self,state):return False  # Continue controlling at the final reference frame.


def create(spec,device):return HandstandAdapter(spec,device)
