"""Read-only H diagnostics; optional finer integration is an in-memory experiment."""
import bootstrap,argparse,json
import numpy as np
import torch,mujoco
from run import create_runtime,advance

p=argparse.ArgumentParser();p.add_argument('--fine',action='store_true')
p.add_argument('--approach',choices=['idle','W','Q'],default='idle')
p.add_argument('--steps',type=int,default=1800);args=p.parse_args()
env,c=create_runtime();h=c.adapters['handstand'];physics=[];frames=[];events=[];old=None;peaks=[]
production_fine=h.manifest.get('local_mujoco_integration',{}).get('motor_solver_dt')==.001
label='_production_1ms' if production_fine else ('_fine' if args.fine else '_original')
folder=bootstrap.ROOT/'logs'/('diagnose_H_'+args.approach+label)
folder.mkdir(parents=True,exist_ok=True)
original_refresh=env.refresh
def refresh():
    original_refresh()
    if env.motor_mode!='handstand':return
    dq=env.data.qvel[env.dids].copy();q=env.data.qpos[env.qids].copy()
    minimum=min((x.dist for x in env.data.contact),default=0.)
    physics.append(np.r_[env.data.time,q,dq,env.target[0].numpy(),env.effort.copy(),minimum])
    if (np.max(np.abs(dq[6:12]))>15 or minimum<-.005) and len(peaks)<80:
        pairs=[]
        for contact in env.data.contact:
            pairs.append(dict(a=mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_BODY,int(env.model.geom_bodyid[contact.geom1])),
                b=mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_BODY,int(env.model.geom_bodyid[contact.geom2])),dist=float(contact.dist)))
        peaks.append(dict(t=env.data.time,rear_dq=dq[6:12].tolist(),contacts=pairs))
env.refresh=refresh
if args.fine and not production_fine:
    original_apply=h.apply
    def apply(action,dt):
        # Test only: preserve the original 50 Hz actor and 4 outer contact ticks.
        # Reuse the existing local 1 ms numerical path without editing H files.
        if h.ctx._pd_substeps==20:h.ctx._pd_substeps=4
        h.ctx.local_pu_isolation=True
        return original_apply(action,dt)
    h.apply=apply
with torch.inference_mode():
    for k in range(args.steps):
        keys=[args.approach] if args.approach!='idle' and 50<=k<100 else (['H'] if k==175 else [])
        c.feed('H_diagnostic',keys);s=advance(env,c)
        if c.mode!=old:
            events.append(dict(frame=k,t=float(env.data.time),mode=c.mode));old=c.mode
        if env.motor_mode=='handstand':
            ctx=h.ctx
            frames.append(dict(frame=k,t=float(env.data.time),phase=float(ctx.episode_length_buf[0])*.02,
                q=env.data.qpos[env.qids].tolist(),dq=env.data.qvel[env.dids].tolist(),target=ctx.target[0].tolist(),
                action=ctx.action[0].tolist(),target_velocity=ctx.target_velocity[0].tolist(),
                torque=ctx.torque[0].tolist(),gravity=s['physics']['gravity_b'],xyz=s['physics']['xyz']))
        if c.mode=='error':break
x=np.array(physics);summary=dict(fine=args.fine or production_fine,production_fine=production_fine,approach=args.approach,events=events,
    frames=k+1,final_mode=c.mode,message=c.message,phase=frames[-1]['phase'] if frames else None,
    physics_samples=len(physics),model_timestep_after_step=env.model.opt.timestep)
if len(x):
    rear_limit=np.array([30.,30.,20.07]*2)
    overspeed=(np.abs(x[:,23:29])>rear_limit).any(1)
    summary.update(peak_rear_joint_speed=float(np.abs(x[:,23:29]).max()),
        peak_front_joint_speed=float(np.abs(x[:,17:23]).max()),deepest_contact_m=float(x[:,-1].min()),
        first_overspeed_s=float(x[overspeed,0][0]) if overspeed.any() else None)
    np.savez_compressed(folder/'physics.npz',samples=x)
    if len(frames)>1:
        target=np.array([f['target'] for f in frames]);action=np.array([f['action'] for f in frames])
        summary.update(peak_target_step_rad=float(np.abs(np.diff(target,axis=0)).max()),
            peak_action_step=float(np.abs(np.diff(action,axis=0)).max()))
(folder/'frames.json').write_text(json.dumps(frames),encoding='utf-8')
(folder/'contacts_at_spikes.json').write_text(json.dumps(peaks,indent=2),encoding='utf-8')
(folder/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print('DIAGNOSIS',json.dumps(summary),flush=True)
