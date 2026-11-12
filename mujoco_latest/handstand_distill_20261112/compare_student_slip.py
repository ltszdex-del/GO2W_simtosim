"""Matched teacher/student diagnostics; active runtime config stays untouched."""
import argparse,contextlib,json,sys
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;sys.path.insert(0,str(BASE))
import bootstrap
from run import create_runtime,advance
from isaaclab.utils.math import quat_apply
cfg_path=BASE/'runtime/config.json';original=cfg_path.read_bytes();cfg=bootstrap.load_config()

def run(kind,friction=None,coarse=False):
    proposed=json.loads(json.dumps(cfg));spec=proposed['policies']['handstand']
    if kind=='teacher':spec.update(factory='handstand_adapter:create',checkpoint=spec['teacher_checkpoint'])
    bootstrap.load_config=lambda:json.loads(json.dumps(proposed))
    env,c=create_runtime();adapter=c.adapters['handstand']
    if coarse:
        # Restore the original 4 x 5 ms update in this diagnostic only.
        reset=adapter.reset
        def reset5(state):
            reset(state);adapter.ctx.local_motor_subdivisions=1
        def apply5(action,dt):
            adapter.ctx._pre_physics_step(action)
            return dict(leg_pos_rad=adapter.ctx.target,wheel_vel_rad_s=adapter.ctx.wheel_target)
        adapter.reset=reset5;adapter.apply=apply5
    if friction is not None:
        # Counterfactual on this in-memory instance only; no scene/config files changed.
        env.model.geom_friction[:,0]=friction
    teacher=adapter.networks.Teacher().eval()
    teacher.load_state_dict(torch.load(cfg['policies']['handstand']['teacher_checkpoint'],map_location='cpu',weights_only=False)['model'])
    student=adapter.networks.Student().eval()
    student.load_state_dict(torch.load(cfg['policies']['handstand']['checkpoint'],map_location='cpu',weights_only=False)['model'])
    rows=[];pending={}
    def observe(state):
        both=adapter.ctx._get_observations()
        ta=teacher(both['policy']);sa=student(both['student'])
        pending.update(teacher_action=ta[0].tolist(),student_action=sa[0].tolist())
        return both['policy' if kind=='teacher' else 'student']
    adapter.observe=observe
    with torch.inference_mode():
        for _ in range(100):c.feed('slip_diagnostic',[]);advance(env,c)
        c.feed('slip_diagnostic',['H']);advance(env,c)
        for k in range(850):
            c.feed('slip_diagnostic',[]);advance(env,c)
            if c.mode in ('handstand','error') and hasattr(adapter,'ctx'):
                ctx=adapter.ctx;p=ctx.physical();d=env.robot.data
                offsets=d.body_link_pos_w.torch.new_zeros(1,4,3)
                offsets[:,:,1]=torch.tensor([-.048,.048,-.048,.048])
                offsets=quat_apply(d.body_link_quat_w.torch[:,ctx.feet],offsets);offsets[:,:,2]-=.086
                point=d.body_link_lin_vel_w.torch[:,ctx.feet]+torch.cross(d.body_link_ang_vel_w.torch[:,ctx.feet],offsets,dim=-1)
                axes=offsets.new_zeros(1,4,3);axes[:,:,1]=1.
                axes=quat_apply(d.body_link_quat_w.torch[:,ctx.feet],axes)
                axle_axis=axes[:,:,:2];axle_axis/=axle_axis.norm(dim=-1,keepdim=True).clamp(min=.1)
                lateral=(point[:,:,:2]*axle_axis).sum(-1)
                rows.append(dict(t=float(ctx.episode_length_buf[0])*.02,mode=c.mode,gravity_x=float(d.projected_gravity_b.torch[0,0]),
                    front_anchor_error_m=ctx.anchor_error(p)[0].tolist(),front_slip_sq=p['slip'][0,:2].tolist(),
                    front_point_speed=point[0,:2,:2].norm(dim=-1).tolist(),front_lateral_speed=lateral[0,:2].tolist(),
                    front_axle_speed=p['axle_velocity_xy'][0,:2].norm(dim=-1).tolist(),
                    front_contact=p['contact'][0,:2].tolist(),front_wheel_dq=d.joint_vel.torch[0,12:14].tolist(),
                    front_wheel_target=ctx.wheel_target[0,:2].tolist(),
                    root_velocity=d.root_lin_vel_b.torch[0].tolist(),**pending))
            if c.mode=='error':break
    label=kind+('_mu'+str(friction) if friction is not None else '')+('_5ms' if coarse else '')+('_targeted' if args.targeted else '')
    (HERE/('slip_trace_'+label+'.json')).write_text(json.dumps(rows,indent=2),encoding='utf-8')
    summary=dict(kind=kind,friction=friction or .8,end_mode=c.mode,last_h_age=rows[-1]['t'],
        physics_dt=env.physics_dt,motor_substeps=adapter.ctx.local_motor_subdivisions,
        implicit_wheel_damping=env.drive_damping.tolist())
    for name,start,end in [('early',0.,2.5),('common_hold',2.5,4.),('late',4.,12.)]:
        r=[s for s in rows if start<=s['t']<end]
        if not r:continue
        def a(key):return np.array([s[key] for s in r])
        error=a('student_action')-a('teacher_action')
        summary[name]=dict(samples=len(r),mean_gravity_x=float(a('gravity_x').mean()),
            front_slip_rms_m_s=float(np.sqrt(a('front_slip_sq').mean())),
            front_point_speed_rms_m_s=float(np.sqrt(np.mean(a('front_point_speed')**2))),
            front_lateral_speed_rms_m_s=float(np.sqrt(np.mean(a('front_lateral_speed')**2))),
            front_axle_speed_rms_m_s=float(np.sqrt(np.mean(a('front_axle_speed')**2))),
            front_anchor_drift_peak_m=float(np.linalg.norm(a('front_anchor_error_m'),axis=-1).max()),
            front_wheel_speed_rms_rad_s=float(np.sqrt(np.mean(a('front_wheel_dq')**2))),
            front_wheel_target_rms_rad_s=float(np.sqrt(np.mean(a('front_wheel_target')**2))),
            per_action_mse=np.mean(error**2,axis=0).tolist(),
            front_wheel_action_delta_rms=float(np.sqrt(np.mean(error[:,12:14]**2))),
            mean_front_wheel_action_delta=error[:,12:14].mean(0).tolist())
    return summary

parser=argparse.ArgumentParser();parser.add_argument('--coarse',action='store_true');parser.add_argument('--targeted',action='store_true');args=parser.parse_args()
suffix=('_5ms' if args.coarse else '')+('_targeted' if args.targeted else '')
if args.targeted:
    cfg['policies']['handstand']['checkpoint']=(BASE/'policies/handstand_targeted_20261112/logs/targeted1000/student_999.pt').as_posix()
    cfg['policies']['handstand']['student_training_root']=(BASE/'policies/handstand_targeted_20261112').as_posix()
with (HERE/('slip_compare_console'+suffix+'.log')).open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log):
    results=[]
    for kind,mu in ([('student',None)] if args.targeted else ([('teacher',None),('student',None)] if args.coarse else [('teacher',None),('student',None),('student',1.2)])):
        results.append(run(kind,mu,args.coarse))
        (HERE/('slip_comparison'+suffix+'.json')).write_text(json.dumps(results,indent=2),encoding='utf-8')
assert cfg_path.read_bytes()==original
print(json.dumps(results,indent=2),flush=True)
