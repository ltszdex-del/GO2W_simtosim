"""Strict adapter for the 570-observation / 12-action locked-wheel policy."""
import hashlib
import math
from pathlib import Path
import torch
from torch import nn


def asset_hash(root):
    digest = hashlib.sha256()
    for p in sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in
                    ('.urdf', '.stl', '.obj', '.dae', '.mtl', '.png', '.jpg')):
        digest.update(p.relative_to(root).as_posix().encode() + b'\0')
        digest.update(p.read_bytes())
    return digest.hexdigest()


class WasdAdapter:
    def __init__(self, spec, device):
        self.spec, self.device, self.path = spec, device, spec['checkpoint']
        self.ck = torch.load(self.path, map_location='cpu', weights_only=False)
        m = self.ck['manifest']
        if self.ck['kind'] != 'blind_velocity' or m['schema'] != 'go2w-step-mirror-v2':
            raise ValueError('Unsupported checkpoint family')
        assert (m['actor_dim'], m['action_dim'], m['history']) == (570, 12, 10)
        assert m['hidden'] == [256, 128, 128]
        c = m['control']
        assert c['leg_scale'] == [.5, .7, .7] * 4
        assert c['stand'] == [0., .8, -1.5] * 4
        assert (c['physics_dt'], c['decimation'], c['kp'], c['kd'], c['leg_rate'],
                c['wheel_lock_damping'], c['effort']) == (.005, 4, 70., 2.5, 8., 100., 23.5)
        assert m['phase']['period'] == .3
        limits=tuple(m['command']['speed_limits']);brakes=tuple(m['command']['axis_brake'])
        assert (limits,brakes) in (((2.,1.),(4.,2.)),((2.3,1.3),(4.6,2.6)),((2.3,1.3),(9.2,5.2))), 'Unsupported speed/brake contract'
        assert m['command']['max_speed']==limits[0] and m['command']['brake_acceleration']==brakes[0]
        assert m['command'].get('min_forward_speed',-limits[0]) in (-limits[0],-1.5), 'Unsupported backward limit'
        self.command_contract=m['command']
        self.yaw_enabled='yaw_speed_limit' in m['command']
        self.motion_phase=any('motion-gated' in x for x in m['observation_order'])
        source = Path(spec['training_root'])
        for name, digest in m['source_sha256'].items():
            if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
                raise ValueError('Checkpoint source mismatch: ' + name)
        assets = Path(spec['asset_root'])
        source_assets=source/'assets/go2w_description'
        assert hashlib.sha256((source_assets/'urdf/go2w_description.urdf').read_bytes()).hexdigest() == m['urdf_sha256']
        self.asset_migration=None
        if asset_hash(assets)!=asset_hash(source_assets):
            assert spec.get('collision_geometry_override'), 'Explicit geometry migration required'
            from asset_compatibility import collision_only_migration
            self.asset_migration=collision_only_migration(source_assets/'urdf/go2w_description.urdf',assets/'urdf/go2w_description.urdf')
            for p in source_assets.rglob('*'):
                if p.is_file() and p.suffix.lower()!='.urdf':
                    assert p.read_bytes()==(assets/p.relative_to(source_assets)).read_bytes(),str(p)
            print('[ASSET_MIGRATION] WASD',self.asset_migration,flush=True)
        self.manifest = dict(observation_order=m['observation_order'], normalization=m['normalization'],
            scales=c['leg_scale'], history=10, phase=m['phase'], previous_action=m['previous_action'],
            target='locked-wheel WASD velocity tracking', task_id='wasd', joint_order=m['joint_order'],
            output={'leg_pos_rad':12, 'wheel_vel_rad_s':4}, pd={'kp':70.,'kd':2.5,'wheel_kd':0.},
            limits={'effort_limit_nm':23.5,'leg_target_rate_rad_s':8.,
                    'wheel_target_rate_rad_s2':40.,'wheel_speed_limit_rad_s':25.},
            timing={'physics_dt':.005,'control_dt':.02,'substeps':4},
            asset_sha256=asset_hash(assets), wheel_lock_damping=100.)
        self.stand = torch.tensor(c['stand'], device=device).reshape(1,12)
        self.scale = torch.tensor(c['leg_scale'], device=device).reshape(1,12)
        self.iteration = self.ck['iteration']

    def load(self):
        self.actor = nn.Sequential(nn.Linear(570,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),
                                   nn.Linear(128,128),nn.ELU(),nn.Linear(128,12)).to(self.device)
        weights = {k.removeprefix('actor.'):v for k,v in self.ck['model'].items() if k.startswith('actor.')}
        self.actor.load_state_dict(weights, strict=True)
        self.actor.eval()
        self.actor=_MirrorNetwork(self.actor,self.stand)

    def bind_env(self,env):
        c=env.control;m=self.command_contract
        assert [c['max_forward_m_s'],c['max_lateral_m_s']]==m['speed_limits'], 'Runtime speed mismatch'
        assert [c['forward_accel_m_s2'],c['lateral_accel_m_s2']]==m['axis_acceleration'], 'Runtime acceleration mismatch'
        assert [c['forward_brake_m_s2'],c['lateral_brake_m_s2']]==m['axis_brake'], 'Runtime brake mismatch'
        assert c.get('min_forward_m_s',-c['max_forward_m_s'])==m.get('min_forward_speed',-m['speed_limits'][0]), 'Runtime backward limit mismatch'
        if self.yaw_enabled:
            assert (c['max_yaw_rad_s'],c['yaw_accel_rad_s2'],c['yaw_brake_rad_s2'])==(m['yaw_speed_limit'],m['yaw_acceleration'],m['yaw_brake'])
        self.env=env

    def reset(self, state):
        if hasattr(self,'env'):self.env.set_motor_context(None,'locomotion',100.)
        self.previous = torch.zeros(1,16,device=self.device)
        self.history = None
        self.age = 0.
        self.heading = state['yaw_rad']

    def observe(self, state):
        command = state['command']['command_m_s_rad_s']
        if abs(command[2]) > 1e-7 and not self.yaw_enabled:
            raise ValueError('This checkpoint was trained for WASD only; yaw command must be zero')
        heading=state['command']['yaw_target_rad'] if self.yaw_enabled else self.heading
        error = heading-state['yaw_rad']
        phase = self.age * (2*math.pi/.3)
        frame = torch.cat((state['projected_gravity_b'],state['angular_velocity_b']*.25,
            state['joint_pos_rad'][:,:12]-self.stand,state['joint_vel_rad_s']*.05,
            self.previous,torch.tensor([command],device=self.device),
            torch.tensor([[math.sin(error),math.cos(error),math.sin(phase),math.cos(phase)]],device=self.device)),dim=-1).clamp(-10,10)
        if self.motion_phase:frame[:,-2:]*=min(1.,max(math.hypot(*command[:2]),abs(command[2]))/.15)
        assert frame.shape == (1,57)
        if self.history is None:self.history=frame[:,None,:].repeat(1,10,1)
        else:self.history=torch.cat((self.history[:,1:],frame[:,None,:]),dim=1)
        return self.history.flatten(1)

    @torch.inference_mode()
    def infer(self, observation):
        return self.actor(observation).tanh()

    def apply(self, action, dt):
        if action.shape != (1,12) or not torch.isfinite(action).all():
            raise ValueError('Invalid WASD action')
        self.previous = torch.cat((action,torch.zeros(1,4,device=self.device)),dim=-1)
        self.age += dt
        return dict(leg_pos_rad=self.stand+self.scale*action, wheel_vel_rad_s=torch.zeros(1,4,device=self.device))

    def is_complete(self, state):return False


def create(spec, device):return WasdAdapter(spec, device)

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
    def __init__(self,raw,stand):
        super().__init__();self.raw=raw;self.stand=stand
    def forward(self,obs):
        return .5*(self.raw(obs)+mirror_action(self.raw(mirror_observation(obs,self.stand,True))))
