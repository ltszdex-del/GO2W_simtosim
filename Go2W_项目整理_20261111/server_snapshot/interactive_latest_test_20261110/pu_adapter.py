"""Private U/P runtime modules share the live robot; complete original control laws retained."""
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from wasd_adapter import asset_hash

class PUAdapter:
    def __init__(self,spec,device):
        self.spec,self.device,self.path=spec,device,spec['checkpoint']
        self.mode=spec['mode'];self.task='P' if self.mode=='lying_down' else 'U'
        self.ck=torch.load(self.path,map_location='cpu',weights_only=False)
        m=self.ck['manifest'];assert m['task']==self.task
        root=Path(spec['source_root'])
        for name,digest in m['code'].items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
        for name,digest in m['assets'].items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
        assert (root/'assets/go2w_description/urdf/go2w_description.urdf').read_bytes()==(Path(spec['asset_root'])/'urdf/go2w_description.urdf').read_bytes()
        assert asset_hash(root/'assets/go2w_description')==asset_hash(Path(spec['asset_root']))
        runtime=Path(__file__).parent/spec['runtime_package']
        transforms=json.loads((runtime/'runtime_transform_manifest.json').read_text())
        for name,hashes in transforms.items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==hashes['original'],name
            assert hashlib.sha256((runtime/name).read_bytes()).hexdigest()==hashes['runtime'],name
        for file in root.glob('*.json'):
            assert file.read_bytes()==(runtime/file.name).read_bytes(),file.name
        self.module=importlib.import_module(spec['runtime_package']+'.env')
        self.network=importlib.import_module(spec['runtime_package']+'.model')
        self.manifest=dict(observation_order='Original PU frame; U216/P279; 3-frame oldest first',
            normalization='original fixed scales',scales='original source-defined target velocity controller',history=3,
            phase='original U/P episode clock',previous_action='previous bounded 12D action',
            target=self.mode,task_id=self.task,joint_order=m['joint_order']+m['wheel_order'],
            output={'leg_pos_rad':12,'wheel_vel_rad_s':4},pd={'kp':70.,'kd':2.5,'wheel_kd':.2},
            limits={'effort_limit_nm':23.5,'leg_target_rate_rad_s':m['control']['leg_rate_rad_s'],
                'wheel_target_rate_rad_s2':40.,'wheel_speed_limit_rad_s':30.1},
            timing={'physics_dt':.005,'control_dt':.02,'substeps':4},asset_sha256=asset_hash(Path(spec['asset_root'])))
        self.manifest['original_training_manifest']=m
        self.manifest['checkpoint_sha256']=hashlib.sha256(Path(self.path).read_bytes()).hexdigest()
        self.failure=None
        self.ready_time=0.;self.angular_history=[];self.pose_history=[]

    def load(self):
        self.model=self.network.Policy().to(self.device).eval()
        self.model.load_state_dict(self.ck['model'],strict=True)

    def bind_env(self,env):self.env=env

    def reset(self,state):
        env=self.env
        if self.mode=='standing_hold' and getattr(env.handstand_context,'pu_task',None)=='U':
            self.ctx=env.handstand_context
            self.ctx.anchor=env.robot.data.root_pos_w.torch[:,:2].clone()
            self.ctx.failed.zero_()
            self.failure=None;self.handoff_age=0.;self.ready_time=0.;self.angular_history=[];self.pose_history=[]
            return
        cfg=self.module.PUCfg();cfg.embedded_env=env
        ctx=self.module.PUEnv(cfg,self.task,False)
        ctx.limits=env.robot.data.joint_pos_limits.torch[:,env.legs]
        ctx.actual_limits=ctx.limits
        ctx.target=(env.robot.data.joint_pos.torch[:,env.legs].clone() if self.mode=='getting_up' else env.target.clone())
        if self.mode=='getting_up':ctx.target=ctx.mirror(ctx.target)
        ctx.target_velocity.zero_()
        ctx.action.zero_();ctx.old_action.zero_();ctx.history.zero_();ctx.pending[:]=True
        ctx.anchor=env.robot.data.root_pos_w.torch[:,:2].clone()
        ctx.failed=torch.zeros(1,dtype=torch.bool,device=self.device)
        ctx.success=ctx.failed.clone();ctx.new_success=ctx.failed.clone();ctx.ever_success.zero_()
        ctx.wheel_target=torch.zeros(1,4,device=self.device)
        ctx.gain[:]=1. if self.task=='P' or self.mode=='standing_hold' else 0.
        if self.mode=='standing_hold':ctx.episode_length_buf[:]=800
        ctx.pu_task=self.task
        self.ctx=ctx;self.failure=None;self.handoff_age=0.;self.ready_time=0.;self.angular_history=[];self.pose_history=[]
        ctx.old_phi=ctx.metrics()['phi'].clone()
        env.set_motor_context(ctx,self.mode,0.)

    def observe(self,state):return self.ctx._get_observations()['policy']

    @torch.inference_mode()
    def infer(self,observation):return self.model(observation)

    def apply(self,action,dt):
        ctx=self.ctx
        ctx._previous_hold=ctx.hold.clone();ctx._previous_ever=ctx.ever_success.clone()
        ctx._pre_physics_step(action)
        return dict(leg_pos_rad=ctx.target,wheel_vel_rad_s=ctx.wheel_target)

    def after_physics(self,state):
        self.ctx.episode_length_buf+=1;self.handoff_age+=.02
        self.ctx._get_dones()
        if self.task=='U':
            # Interactive rollout uses U policy without training termination masks.
            self.ctx.failed.zero_()
        if bool(self.ctx.failed[0]):
            self.failure='PU '+self.mode+' failure: '+str({k:bool(v[0]) for k,v in self.ctx.reason_masks.items()})
            print('[PU_FAILURE]',self.failure,flush=True)
            return True
        if self.task=='U' and self.mode=='getting_up':
            # Runtime mode handoff has no training completion deadline. Check
            # physical pose variation over 0.5s, retaining instantaneous guards.
            # This does not change the actor, motor controller or training score.
            import math
            yaw=state['yaw_rad']
            pose=torch.cat((state['projected_gravity_b'][0],
                torch.tensor([math.sin(yaw),math.cos(yaw)],device=self.device)))
            self.pose_history.append(pose);self.pose_history=self.pose_history[-25:]
            m=self.ctx.p
            poses=torch.stack(self.pose_history)
            pose_span=(poses.amax(0)-poses.amin(0)).norm()
            ready=(m['ze'][0]<.035 and m['qe'][0]<.18**2 and m['base'][0]<.05*self.ctx.weight
                and m['clearance'][0]>.02 and m['v'][0]<.10 and pose_span<.035
                and m['front_camber'][0].max()<.0872665 and m['symmetry_max'][0]<.0872665
                and m['foot_mirror_max'][0]<.02 and len(self.pose_history)==25)
            self.ready_time=self.ready_time+.02 if ready else 0.
        return False

    def is_complete(self,state):
        if self.mode=='standing_hold':return False
        if self.task=='U':return bool(self.ctx.success[0]) or self.ready_time>=.5
        return bool(self.ctx.success[0])

    def can_start_getup(self,state):
        if self.task!='P':return False
        m=self.ctx.metrics()
        return bool(not self.ctx.failed[0] and m['z'][0]<.18 and m['qe'][0]<.3**2
            and m['v'][0]<.10 and m['roll'][0].abs()<.0872665 and m['pitch'][0].abs()<.0872665)

def create(spec,device):return PUAdapter(spec,device)
