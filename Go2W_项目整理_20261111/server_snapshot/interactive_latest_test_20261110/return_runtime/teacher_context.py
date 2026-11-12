"""Frozen H4099 methods and U1099 motor law, sharing one physical articulation."""
import sys,json,hashlib
from pathlib import Path
from types import SimpleNamespace as NS,MethodType
import torch
from .settings import ROOT
sys.path.insert(0,str(ROOT/'teacher_code'))
from go2w_mimic.env import MimicEnv
from go2w_mimic.networks import Teacher
from go2w_mimic.reference import Motion
from go2w_mimic import settings as hs

def make_teacher(env):
    ck=torch.load(ROOT/'teacher_4099.pt',map_location=env.device,weights_only=False)
    for name,digest in ck['interface']['sources'].items():
        assert hashlib.sha256((ROOT/'teacher_code/go2w_mimic'/name).read_bytes()).hexdigest()==digest,name
    motion=ROOT/'teacher_code/motions/stand_to_handstand.npz'
    assert hashlib.sha256(motion.read_bytes()).hexdigest()==ck['interface']['motion_sha256']
    assert hashlib.sha256((ROOT/'assets/go2w_description/urdf/go2w_description.urdf').read_bytes()).hexdigest()==ck['interface']['urdf_sha256']
    n,dev=env.num_envs,env.device;z=lambda *shape:torch.zeros(*shape,device=dev)
    ctx=NS(robot=env.robot,contacts=env.contacts,device=dev,num_envs=n,step_dt=.02,extras={},cfg=NS(randomize=False),
        scene=env.scene,leg_ids=env.legs,wheel_ids=env.wheels,ids=env.ids,feet=env.feet,
        contact_ids=env.contact_ids,other_contacts=env.other_contacts,motion=Motion(str(motion),dev),
        limits=env.limits,_pd_substeps=None)
    for name in ('action','old_action','older_action','torque'):setattr(ctx,name,z(n,16))
    ctx.target=z(n,12);ctx.target_velocity=z(n,12);ctx.wheel_target=z(n,4);ctx.delayed=z(n,3,16)
    ctx.delay=torch.zeros(n,dtype=torch.long,device=dev);ctx.gain=torch.ones(n,2,device=dev)
    ctx.speed=torch.ones(n,device=dev);ctx.mass_scale=torch.ones(n,device=dev);ctx.mu=torch.full((n,),.8,device=dev)
    ctx.masses=env.robot.data.body_mass.torch.clone();ctx.front_anchor=z(n,2,2)
    ctx.anchor_pending=torch.ones(n,dtype=torch.bool,device=dev)
    ctx.history=z(n,hs.HISTORY,hs.PROPRIO_DIM);ctx.fill_history=ctx.anchor_pending.clone()
    ctx.episode_length_buf=torch.zeros(n,dtype=torch.long,device=dev)
    for name in ('ref','physical','wheel_guard','heading','stable_heading','anchor_error','blocks',
                 '_get_observations','_pre_physics_step','_apply_action'):
        setattr(ctx,name,MethodType(getattr(MimicEnv,name),ctx))
    model=Teacher().to(dev).eval();model.load_state_dict(ck['model']);model.requires_grad_(False)
    return ctx,model

def make_u_controller(env):
    from .u_runtime.env import PUEnv,PUCfg
    transforms=json.loads((ROOT/'u_runtime/runtime_transform_manifest.json').read_text())
    for name,row in transforms.items():
        assert hashlib.sha256((ROOT/'u_runtime'/name).read_bytes()).hexdigest()==row['runtime'],name
    provenance=json.loads((ROOT/'provenance.json').read_text())
    original=Path(provenance['U_project'])/'logs/U_vertical_resume100/final.pt'
    assert hashlib.sha256(original.read_bytes()).hexdigest()==provenance['U_checkpoint_sha256']
    checkpoint=torch.load(original,map_location='cpu',weights_only=False)
    for name,digest in checkpoint['manifest']['code'].items():
        if name in transforms:assert transforms[name]['original']==digest,name
        else:assert hashlib.sha256((ROOT/'u_runtime'/name).read_bytes()).hexdigest()==digest,name
    cfg=PUCfg();cfg.embedded_env=env
    ctx=PUEnv(cfg,'U',False);ctx.gain.fill_(1.);ctx.pd_count=0
    return ctx
