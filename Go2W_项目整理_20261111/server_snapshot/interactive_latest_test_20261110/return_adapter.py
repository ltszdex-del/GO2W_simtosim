"""Verified return policy; original observation and motor laws, no training resets."""
import json,hashlib,importlib
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from wasd_adapter import asset_hash

class ReturnAdapter:
    def __init__(self,spec,device):
        self.spec,self.device,self.path=spec,device,spec['checkpoint']
        self.ck=torch.load(self.path,map_location='cpu',weights_only=False);m=self.ck['manifest'];root=Path(spec['code_root'])
        assert self.ck['kind']=='handstand_return' and m['schema']=='go2w-handstand-return-v1'
        for group in ('source_sha256','state_sha256'):
            for name,digest in m[group].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
        assert hashlib.sha256((root/'teacher_4099.pt').read_bytes()).hexdigest()==m['teacher']['H_sha256']
        assert hashlib.sha256((root/'assets/go2w_description/urdf/go2w_description.urdf').read_bytes()).hexdigest()==m['urdf_sha256']
        assert asset_hash(root/'assets/go2w_description')==asset_hash(Path(spec['asset_root']))
        runtime=Path(__file__).parent/'return_runtime'
        for name,row in json.loads((runtime/'transforms.json').read_text()).items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==row['source']
            assert hashlib.sha256((runtime/name).read_bytes()).hexdigest()==row['runtime']
        self.module=importlib.import_module('return_runtime.env');self.network=importlib.import_module('return_runtime.model')
        self.manifest=dict(observation_order='Original return frame115 x history5',normalization='original fixed scales',scales='original controller',history=5,phase='manual H handoff; original .65s reference',previous_action='original bounded16',target='handstand return to U',task_id='handstand_return',joint_order=m.get('joint_order',[]),output={'leg_pos_rad':12,'wheel_vel_rad_s':4},pd={'kp':70.,'kd':2.5,'wheel_kd':.2},limits={'effort_limit_nm':23.5,'leg_target_rate_rad_s':6.,'wheel_target_rate_rad_s2':40.,'wheel_speed_limit_rad_s':18.},timing={'physics_dt':.005,'control_dt':.02,'substeps':4},asset_sha256=asset_hash(Path(spec['asset_root'])))
        from scene import LEGS,WHEELS
        self.manifest['joint_order']=LEGS+WHEELS
    def load(self):
        self.model=self.network.Policy().to(self.device).eval();self.model.load_state_dict(self.ck['model'],strict=True)
    def bind_env(self,env):self.env=env
    def reset(self,state):
        env=self.env;h=env.handstand_context
        if h is None or not hasattr(h,'motion'):raise ValueError('Return requires active H policy context')
        # Preserve the running H state for the original torque handoff.
        from go2w_mimic.networks import Teacher
        teacher=Teacher().to(self.device).eval();teacher.load_state_dict(torch.load(Path(self.spec['code_root'])/'teacher_4099.pt',map_location='cpu',weights_only=False)['model'])
        cfg=NS(embedded_env=env,teacher_context=h,teacher_model=teacher,physics_randomization=False)
        self.ctx=self.module.ReturnEnv(cfg);c=self.ctx
        c.level=float(self.ck.get('curriculum',1.));c.guidance_duration.fill_(.65)
        c.target=env.target.clone();c.wheel_target=env.wheel_target.clone();c.heading_goal=c.h.stable_heading().clone()
        c.previous_quat=env.robot.data.root_quat_w.torch.clone()
        m=c.metrics();c.previous_pitch=m['pitch'].clone();c.previous_contacts=m['contact']>.5
        c.previous_foot_vz=m['foot_vz'].clone();c.previous_normal_force=m['normal'].clone()
        c.jitter_ema=torch.stack((m['leg_rms'],m['omega'],m['wheel_rms']),-1)
        c.jittered=torch.ones(1,dtype=torch.bool,device=self.device)
        c._handoff(torch.ones(1,dtype=torch.bool,device=self.device),m);c._controlled.fill_(True)
        self.ready_time=0.;env.set_motor_context(c,'handstand_return',0.)
    def observe(self,state):return self.ctx._get_observations()['policy']
    @torch.inference_mode()
    def infer(self,observation):return self.model(observation)
    def apply(self,action,dt):
        assert abs(dt-.02)<1e-9
        self.ctx._pre_physics_step(action)
        return dict(leg_pos_rad=self.ctx.target,wheel_vel_rad_s=self.ctx.wheel_target)
    def after_physics(self,state):
        c=self.ctx;c._get_dones();c.failed.zero_()
        m=c.m;t=self.module.TASK
        good=bool((m['pitch'][0]-c.goal_pitch).abs()<t['stand_angle_rad'] and m['roll'][0].abs()<t['stand_angle_rad'] and (m['height'][0]-c.goal_height).abs()<t['stand_height_error_m'] and m['q_error'][0]<t['stand_joint_rms_rad']**2 and (m['contact'][0]>.5).all() and m['vxy'][0]<t['stand_horizontal_speed_m_s'] and m['vz'][0].abs()<t['stand_vertical_speed_m_s'] and c.pose_rate[0]<t['stand_pose_angular_speed_rad_s'])
        self.ready_time=self.ready_time+.02 if good else 0.
        return False
    def is_complete(self,state):return self.ready_time>=.10

def create(spec,device):return ReturnAdapter(spec,device)

